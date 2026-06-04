import json
import os

import numpy as np
import torch

from attn_heatmap import build_run_dir, build_token_entries, extract_result_summary, sanitize_slug
from query_window_similarity import extract_attention_arg, get_decoder_layers, run_observation_forward


ATTN_OUTPUT_RATIO_METRIC = "l2_norm_ratio"
ATTN_OUTPUT_RATIO_REDUCTION = "per_token"
DEFAULT_LAYER_SPEC = "all"


def extract_token_mixer_output(output):
    if isinstance(output, (tuple, list)):
        return output[0] if output else None
    if isinstance(output, dict):
        if output.get("last_hidden_state") is not None:
            return output.get("last_hidden_state")
        return output.get("hidden_states")
    return output


def find_layer_token_mixer(layer):
    for name in ("self_attn", "linear_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return name, module
    return None, None


def parse_layer_spec(layer_spec, available_layers):
    available_layers = [int(layer_idx) for layer_idx in available_layers]
    if layer_spec is None:
        layer_spec = DEFAULT_LAYER_SPEC
    layer_spec = str(layer_spec).strip().lower()
    if layer_spec in ("", "all"):
        return available_layers
    if layer_spec == "auto":
        if len(available_layers) <= 4:
            return available_layers
        positions = np.linspace(0, len(available_layers) - 1, num=4).round().astype(int)
        return [available_layers[pos] for pos in sorted(set(positions.tolist()))]

    selected = []
    available = set(available_layers)
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if start <= end else -1
            selected.extend(range(start, end + step, step))
        else:
            selected.append(int(part))
    selected = [layer_idx for layer_idx in selected if layer_idx in available]
    if not selected:
        raise ValueError(
            f"No requested layers from '{layer_spec}' are available. "
            f"Available layers: {available_layers}"
        )
    return selected


def compute_parallel_orthogonal_components(input_hidden_states, attn_output, eps=1e-12):
    input_vectors = input_hidden_states.detach().to(dtype=torch.float32)
    output_vectors = attn_output.detach().to(dtype=torch.float32)
    actual_len = min(int(input_vectors.shape[0]), int(output_vectors.shape[0]))
    if actual_len < 1:
        return None

    input_vectors = input_vectors[:actual_len]
    output_vectors = output_vectors[:actual_len]
    input_sq_norms = torch.sum(input_vectors * input_vectors, dim=-1)
    input_l2_norms = torch.sqrt(torch.clamp(input_sq_norms, min=0.0))
    output_sq_norms = torch.sum(output_vectors * output_vectors, dim=-1)
    attn_output_l2_norms = torch.sqrt(torch.clamp(output_sq_norms, min=0.0))

    dot_products = torch.sum(input_vectors * output_vectors, dim=-1)
    safe_input_sq_norms = torch.clamp(input_sq_norms, min=float(eps))
    safe_input_l2_norms = torch.clamp(input_l2_norms, min=float(eps))
    parallel_l2_norms = torch.abs(dot_products) / safe_input_l2_norms
    perpendicular_sq_norms = torch.clamp(output_sq_norms - parallel_l2_norms * parallel_l2_norms, min=0.0)
    perpendicular_l2_norms = torch.sqrt(perpendicular_sq_norms)

    ratios = attn_output_l2_norms / safe_input_l2_norms
    parallel_ratios = parallel_l2_norms / safe_input_l2_norms
    perpendicular_ratios = perpendicular_l2_norms / safe_input_l2_norms
    signed_parallel_ratios = dot_products / safe_input_sq_norms
    return {
        "ratios": ratios,
        "input_l2_norms": input_l2_norms,
        "attn_output_l2_norms": attn_output_l2_norms,
        "parallel_l2_norms": parallel_l2_norms,
        "perpendicular_l2_norms": perpendicular_l2_norms,
        "parallel_ratios": parallel_ratios,
        "perpendicular_ratios": perpendicular_ratios,
        "signed_parallel_ratios": signed_parallel_ratios,
    }


def compute_attn_output_hidden_state_ratios(model, inputs, eps=1e-12):
    layers = get_decoder_layers(model)
    if not layers:
        raise ValueError("No decoder layers were found; cannot capture attention-output ratios.")

    captured = {}
    pending_inputs = {}
    token_mixer_names = {}
    handles = []

    def make_pre_hook(layer_idx):
        def hook(_module, args, kwargs):
            hidden_states = extract_attention_arg(args, kwargs, "hidden_states", 0)
            if hidden_states is None or not torch.is_tensor(hidden_states) or hidden_states.ndim < 3:
                return
            pending_inputs[layer_idx] = hidden_states[0].detach()

        return hook

    def make_post_hook(layer_idx, token_mixer_name):
        def hook(_module, _args, _kwargs, output):
            attn_output = extract_token_mixer_output(output)
            if attn_output is None or not torch.is_tensor(attn_output) or attn_output.ndim < 3:
                return
            input_vectors = pending_inputs.pop(layer_idx, None)
            if input_vectors is None:
                return
            component_values = compute_parallel_orthogonal_components(
                input_hidden_states=input_vectors,
                attn_output=attn_output[0],
                eps=eps,
            )
            if component_values is None:
                return
            captured[layer_idx] = {
                key: value.cpu().numpy().astype(np.float32, copy=False)
                for key, value in component_values.items()
            }
            token_mixer_names[layer_idx] = token_mixer_name

        return hook

    try:
        for layer_idx, layer in enumerate(layers):
            token_mixer_name, token_mixer = find_layer_token_mixer(layer)
            if token_mixer is None:
                continue
            handles.append(token_mixer.register_forward_pre_hook(make_pre_hook(layer_idx), with_kwargs=True))
            handles.append(token_mixer.register_forward_hook(make_post_hook(layer_idx, token_mixer_name), with_kwargs=True))
        run_observation_forward(model, inputs, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()

    layer_indices = [layer_idx for layer_idx in range(len(layers)) if layer_idx in captured]
    if not layer_indices:
        raise ValueError("No attention outputs were captured during prefill.")

    token_count = min(captured[layer_idx]["ratios"].shape[0] for layer_idx in layer_indices)

    def stack_metric(metric_name):
        return np.stack(
            [captured[layer_idx][metric_name][:token_count] for layer_idx in layer_indices],
            axis=0,
        )

    ratios = stack_metric("ratios")
    input_l2_norms = stack_metric("input_l2_norms")
    attn_output_l2_norms = stack_metric("attn_output_l2_norms")
    parallel_l2_norms = stack_metric("parallel_l2_norms")
    perpendicular_l2_norms = stack_metric("perpendicular_l2_norms")
    parallel_ratios = stack_metric("parallel_ratios")
    perpendicular_ratios = stack_metric("perpendicular_ratios")
    signed_parallel_ratios = stack_metric("signed_parallel_ratios")
    mixer_names = [token_mixer_names.get(layer_idx, "unknown") for layer_idx in layer_indices]
    return (
        ratios,
        input_l2_norms,
        attn_output_l2_norms,
        parallel_l2_norms,
        perpendicular_l2_norms,
        parallel_ratios,
        perpendicular_ratios,
        signed_parallel_ratios,
        layer_indices,
        mixer_names,
    )


def _summarize_values(values, prefix=None):
    summary = {
        "sum": float(np.sum(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }
    if prefix is None:
        return summary
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def build_layer_stats(
    ratios,
    layer_indices,
    parallel_ratios=None,
    perpendicular_ratios=None,
    signed_parallel_ratios=None,
):
    stats = []
    for layer_pos, layer_idx in enumerate(layer_indices):
        values = ratios[layer_pos].astype(np.float32, copy=False)
        layer_stats = {
            "layer": int(layer_idx),
            "token_count": int(values.shape[0]),
        }
        layer_stats.update(_summarize_values(values))
        if parallel_ratios is not None:
            layer_stats.update(
                _summarize_values(
                    parallel_ratios[layer_pos].astype(np.float32, copy=False),
                    prefix="parallel_ratio",
                )
            )
        if perpendicular_ratios is not None:
            layer_stats.update(
                _summarize_values(
                    perpendicular_ratios[layer_pos].astype(np.float32, copy=False),
                    prefix="perpendicular_ratio",
                )
            )
        if signed_parallel_ratios is not None:
            layer_stats.update(
                _summarize_values(
                    signed_parallel_ratios[layer_pos].astype(np.float32, copy=False),
                    prefix="signed_parallel_ratio",
                )
            )
        stats.append(layer_stats)
    return stats


def setup_matplotlib_cache():
    cache_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "qwen35_compression_matplotlib_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)


def plot_attn_output_ratio_density(
    ratios,
    layer_indices,
    output_path,
    title,
    layer_spec=DEFAULT_LAYER_SPEC,
    xlabel=r"$\Vert\Delta_{\mathrm{attn}}\Vert_2 / \Vert x_{\mathrm{in}}\Vert_2$",
):
    setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    selected_layers = parse_layer_spec(layer_spec, layer_indices)
    layer_to_pos = {int(layer_idx): pos for pos, layer_idx in enumerate(layer_indices)}
    if len(selected_layers) <= 8:
        colors = ["#7faa3d", "#4dae6b", "#355f83", "#3c145b", "#d07c2c", "#b64b6a", "#5d8792", "#6b5b95"]
    else:
        cmap = plt.get_cmap("viridis")
        colors = [cmap(idx / max(1, len(selected_layers) - 1)) for idx in range(len(selected_layers))]

    fig, (density_ax, sum_ax) = plt.subplots(
        2,
        1,
        figsize=(7.4, 5.9),
        dpi=180,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    selected_sums = []
    for color, layer_idx in zip(colors, selected_layers):
        values = ratios[layer_to_pos[int(layer_idx)]]
        selected_sums.append(float(np.sum(values)))
        label = f"Layer {layer_idx}"
        if values.shape[0] >= 3 and np.std(values) > 0:
            sns.kdeplot(
                x=values,
                ax=density_ax,
                label=label,
                color=color,
                linewidth=2.0,
                fill=True,
                alpha=0.22,
                warn_singular=False,
            )
        else:
            density_ax.hist(
                values,
                bins=min(10, max(1, values.shape[0])),
                density=True,
                label=label,
                color=color,
                alpha=0.22,
            )
            density_ax.axvline(float(np.mean(values)), color=color, linewidth=2.0)
        density_ax.axvline(float(np.mean(values)), color=color, linewidth=1.6, linestyle="--", alpha=0.95)

    density_ax.set_title(title, fontsize=15, fontweight="bold")
    density_ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    density_ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    density_ax.legend(loc="upper right", frameon=True)
    density_ax.grid(False)

    x_positions = np.arange(len(selected_layers))
    sum_ax.bar(x_positions, selected_sums, color=colors, alpha=0.82, width=0.68)
    sum_ax.plot(x_positions, selected_sums, color="#2f2f2f", linewidth=1.1, marker="o", markersize=3.2)
    sum_ax.set_xticks(x_positions)
    sum_ax.set_xticklabels([str(layer_idx) for layer_idx in selected_layers])
    sum_ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    sum_ax.set_ylabel("Token ratio sum", fontsize=11, fontweight="bold")
    sum_ax.grid(axis="y", alpha=0.25, linewidth=0.7)

    if len(selected_layers) <= 8:
        max_sum = max(selected_sums) if selected_sums else 0.0
        offset = max_sum * 0.015 if max_sum > 0 else 0.01
        for x_pos, ratio_sum in zip(x_positions, selected_sums):
            sum_ax.text(
                x_pos,
                ratio_sum + offset,
                f"{ratio_sum:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_attn_output_l2_norm_layer_curves(
    attn_output_l2_norms,
    layer_indices,
    output_dir,
    file_prefix,
    title_prefix,
):
    setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(attn_output_l2_norms, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"attn_output_l2_norms must be 2D, got shape {values.shape}.")
    if values.shape[0] != len(layer_indices):
        raise ValueError(
            "Layer count mismatch for attn_output_l2_norms layer plots: "
            f"{values.shape[0]} rows vs {len(layer_indices)} layer indices."
        )

    token_count = values.shape[1]
    x_values = np.arange(1, token_count + 1)
    output_files = []
    for layer_pos, layer_idx in enumerate(layer_indices):
        fig_width = min(18.0, max(8.0, token_count / 45.0))
        fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=180)
        ax.plot(x_values, values[layer_pos], color="#355f83", linewidth=1.4)
        ax.fill_between(x_values, values[layer_pos], color="#4dae6b", alpha=0.16)
        ax.set_xlabel(
            "Sequence length",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel(r"$\Vert\Delta_{\mathrm{attn}}\Vert_2$", fontsize=12, fontweight="bold")
        ax.set_title(f"{title_prefix} Layer {layer_idx}", fontsize=15, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        if token_count > 0:
            tick_count = min(12, token_count)
            tick_positions = np.linspace(1, token_count, num=tick_count).round().astype(int)
            ax.set_xticks(tick_positions)
        file_name = f"{file_prefix}_layer_{int(layer_idx):03d}_attn_output_l2_norm.png"
        output_path = os.path.join(output_dir, file_name)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        output_files.append(file_name)

    return output_files


def plot_attn_output_decomposition_layer_curves(
    parallel_l2_norms,
    perpendicular_l2_norms,
    parallel_ratios,
    perpendicular_ratios,
    signed_parallel_ratios,
    layer_indices,
    output_dir,
    file_prefix,
    title_prefix,
):
    setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_arrays = {
        "parallel_l2_norms": np.asarray(parallel_l2_norms, dtype=np.float32),
        "perpendicular_l2_norms": np.asarray(perpendicular_l2_norms, dtype=np.float32),
        "parallel_ratios": np.asarray(parallel_ratios, dtype=np.float32),
        "perpendicular_ratios": np.asarray(perpendicular_ratios, dtype=np.float32),
        "signed_parallel_ratios": np.asarray(signed_parallel_ratios, dtype=np.float32),
    }
    expected_shape = metric_arrays["parallel_ratios"].shape
    if len(expected_shape) != 2:
        raise ValueError(f"Decomposition metrics must be 2D, got shape {expected_shape}.")
    if expected_shape[0] != len(layer_indices):
        raise ValueError(
            "Layer count mismatch for decomposition layer plots: "
            f"{expected_shape[0]} rows vs {len(layer_indices)} layer indices."
        )
    for metric_name, values in metric_arrays.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {metric_name}: got {values.shape}, expected {expected_shape}."
            )

    token_count = expected_shape[1]
    x_values = np.arange(1, token_count + 1)
    output_files = []
    for layer_pos, layer_idx in enumerate(layer_indices):
        fig_width = min(18.0, max(8.4, token_count / 45.0))
        fig, (ratio_ax, norm_ax) = plt.subplots(
            2,
            1,
            figsize=(fig_width, 7.0),
            dpi=180,
            sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1.0]},
        )
        ratio_ax.plot(
            x_values,
            metric_arrays["parallel_ratios"][layer_pos],
            color="#355f83",
            linewidth=1.45,
            label=r"$R_{\parallel}$",
        )
        ratio_ax.plot(
            x_values,
            metric_arrays["perpendicular_ratios"][layer_pos],
            color="#b64b6a",
            linewidth=1.45,
            label=r"$R_{\perp}$",
        )
        ratio_ax.plot(
            x_values,
            metric_arrays["signed_parallel_ratios"][layer_pos],
            color="#6b5b95",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            label=r"signed $R_{\parallel}$",
        )
        ratio_ax.axhline(0.0, color="#2f2f2f", linewidth=0.8, alpha=0.45)
        ratio_ax.set_ylabel("Ratio to input L2", fontsize=11, fontweight="bold")
        ratio_ax.set_title(f"{title_prefix} Layer {layer_idx}", fontsize=15, fontweight="bold")
        ratio_ax.legend(loc="upper right", frameon=True)
        ratio_ax.grid(axis="y", alpha=0.25, linewidth=0.7)

        norm_ax.plot(
            x_values,
            metric_arrays["parallel_l2_norms"][layer_pos],
            color="#355f83",
            linewidth=1.35,
            label=r"$\Vert\Delta x_{\parallel}\Vert_2$",
        )
        norm_ax.plot(
            x_values,
            metric_arrays["perpendicular_l2_norms"][layer_pos],
            color="#b64b6a",
            linewidth=1.35,
            label=r"$\Vert\Delta x_{\perp}\Vert_2$",
        )
        norm_ax.set_xlabel("Sequence length", fontsize=12, fontweight="bold")
        norm_ax.set_ylabel("Component L2 norm", fontsize=11, fontweight="bold")
        norm_ax.legend(loc="upper right", frameon=True)
        norm_ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        if token_count > 0:
            tick_count = min(12, token_count)
            tick_positions = np.linspace(1, token_count, num=tick_count).round().astype(int)
            norm_ax.set_xticks(tick_positions)

        file_name = f"{file_prefix}_layer_{int(layer_idx):03d}_attn_output_decomposition.png"
        output_path = os.path.join(output_dir, file_name)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        output_files.append(file_name)

    return output_files



class AttnOutputRatioRunWriter:
    def __init__(
        self,
        root_dir,
        model_name,
        out_file,
        max_prefill_tokens,
        layer_spec=DEFAULT_LAYER_SPEC,
    ):
        self.root_dir = root_dir
        self.model_name = model_name
        self.out_file = os.path.abspath(out_file)
        self.max_prefill_tokens = int(max_prefill_tokens) if max_prefill_tokens is not None else None
        self.layer_spec = layer_spec or DEFAULT_LAYER_SPEC
        self.run_dir = build_run_dir(root_dir, out_file)
        self.samples_dir = os.path.join(self.run_dir, "samples")
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.samples = []
        self._next_sample_index = 0
        os.makedirs(self.samples_dir, exist_ok=True)
        self._write_manifest()

    def new_sample(self, item):
        sample_index = self._next_sample_index
        self._next_sample_index += 1
        return AttnOutputRatioSampleWriter(self, item, sample_index)

    def register_sample(self, sample_writer):
        sample_summary = {
            "sample_id": sample_writer.sample_id,
            "sample_index": sample_writer.sample_index,
            "sample_path": sample_writer.sample_rel_path,
            "_id": sample_writer.item.get("_id"),
            "domain": sample_writer.item.get("domain"),
            "question": sample_writer.item.get("question"),
            "pred": sample_writer.result_summary.get("pred"),
            "judge": sample_writer.result_summary.get("judge"),
            "prefill_count": len(sample_writer.prefills),
            "prefill_statuses": [prefill["status"] for prefill in sample_writer.prefills],
        }
        self.samples.append(sample_summary)
        self._write_manifest()

    def _write_manifest(self):
        manifest = {
            "run_dir": self.run_dir,
            "model_name": self.model_name,
            "result_path": self.out_file,
            "attn_output_ratio_metric": ATTN_OUTPUT_RATIO_METRIC,
            "attn_output_ratio_reduction": ATTN_OUTPUT_RATIO_REDUCTION,
            "attn_output_ratio_layers": self.layer_spec,
            "attn_output_ratio_max_prefill_tokens": self.max_prefill_tokens,
            "attn_output_ratio_prefill_cap_mode": "fixed" if self.max_prefill_tokens is not None else "none",
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as fout:
            json.dump(manifest, fout, ensure_ascii=False, indent=2)


class AttnOutputRatioSampleWriter:
    def __init__(self, run_writer, item, sample_index):
        self.run_writer = run_writer
        self.item = dict(item)
        self.sample_index = int(sample_index)
        base_name = sanitize_slug(
            self.item.get("_id", f"sample_{sample_index:04d}"),
            fallback=f"sample_{sample_index:04d}",
        )
        self.sample_id = f"sample_{sample_index:04d}_{base_name}"
        self.sample_dir = os.path.join(self.run_writer.samples_dir, self.sample_id)
        self.sample_rel_path = os.path.relpath(self.sample_dir, self.run_writer.run_dir)
        self.sample_json_path = os.path.join(self.sample_dir, "sample.json")
        self.prefills = []
        self.result_summary = {}
        os.makedirs(self.sample_dir, exist_ok=True)
        self._write_sample_json()

    def capture_prefill(self, model, tokenizer, prompt_text, inputs, label, max_prefill_tokens=None):
        prefill_index = len(self.prefills)
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        prefill_cap = (
            int(max_prefill_tokens)
            if max_prefill_tokens is not None
            else self.run_writer.max_prefill_tokens
        )
        record = {
            "prefill_index": prefill_index,
            "label": label,
            "status": "pending",
            "prompt_text": prompt_text,
            "max_prefill_tokens": prefill_cap,
            "token_count": len(input_ids),
            "tokens": build_token_entries(tokenizer, input_ids),
            "ratio_file": None,
            "density_plot_file": None,
            "attn_output_l2_layer_plot_files": [],
            "attn_output_decomposition_layer_plot_files": [],
            "ratio_shape": None,
            "layer_indices": [],
            "token_mixer_names": [],
            "layer_stats": [],
        }
        self.prefills.append(record)
        self._write_sample_json()

        if prefill_cap is not None and len(input_ids) > prefill_cap:
            record["status"] = "skipped_over_cap"
            record["reason"] = f"Prompt token count {len(input_ids)} exceeds cap {prefill_cap}."
            self._write_sample_json()
            return record

        try:
            (
                ratios,
                input_l2_norms,
                attn_output_l2_norms,
                parallel_l2_norms,
                perpendicular_l2_norms,
                parallel_ratios,
                perpendicular_ratios,
                signed_parallel_ratios,
                layer_indices,
                mixer_names,
            ) = compute_attn_output_hidden_state_ratios(model=model, inputs=inputs)
            ratio_file_name = f"prefill_{prefill_index:03d}_attn_output_hidden_l2_ratio.npz"
            density_plot_file_name = f"prefill_{prefill_index:03d}_attn_output_hidden_l2_ratio_density.png"
            ratio_path = os.path.join(self.sample_dir, ratio_file_name)
            density_plot_path = os.path.join(self.sample_dir, density_plot_file_name)
            token_ids = input_ids[: ratios.shape[1]]
            np.savez_compressed(
                ratio_path,
                ratios=ratios.astype(np.float32, copy=False),
                input_l2_norms=input_l2_norms.astype(np.float32, copy=False),
                attn_output_l2_norms=attn_output_l2_norms.astype(np.float32, copy=False),
                parallel_l2_norms=parallel_l2_norms.astype(np.float32, copy=False),
                perpendicular_l2_norms=perpendicular_l2_norms.astype(np.float32, copy=False),
                parallel_ratios=parallel_ratios.astype(np.float32, copy=False),
                perpendicular_ratios=perpendicular_ratios.astype(np.float32, copy=False),
                signed_parallel_ratios=signed_parallel_ratios.astype(np.float32, copy=False),
                layer_indices=np.asarray(layer_indices, dtype=np.int16),
                token_ids=np.asarray(token_ids, dtype=np.int64),
                token_mixer_names=np.asarray(mixer_names),
                ratio_metric=np.asarray(ATTN_OUTPUT_RATIO_METRIC),
                ratio_reduction=np.asarray(ATTN_OUTPUT_RATIO_REDUCTION),
            )
            record["status"] = "saved"
            record["ratio_file"] = ratio_file_name
            record["ratio_shape"] = list(ratios.shape)
            record["layer_indices"] = [int(layer_idx) for layer_idx in layer_indices]
            record["token_mixer_names"] = mixer_names
            record["layer_stats"] = build_layer_stats(
                ratios,
                layer_indices,
                parallel_ratios=parallel_ratios,
                perpendicular_ratios=perpendicular_ratios,
                signed_parallel_ratios=signed_parallel_ratios,
            )
            try:
                plot_attn_output_ratio_density(
                    ratios=ratios,
                    layer_indices=layer_indices,
                    output_path=density_plot_path,
                    title=f"Random Sample {self.sample_index + 1}",
                    layer_spec=self.run_writer.layer_spec,
                )
                record["density_plot_file"] = density_plot_file_name
                record["attn_output_l2_layer_plot_files"] = plot_attn_output_l2_norm_layer_curves(
                    attn_output_l2_norms=attn_output_l2_norms,
                    layer_indices=layer_indices,
                    output_dir=self.sample_dir,
                    file_prefix=f"prefill_{prefill_index:03d}",
                    title_prefix=f"Random Sample {self.sample_index + 1} Attention Output L2 Norms",
                )
                record["attn_output_decomposition_layer_plot_files"] = (
                    plot_attn_output_decomposition_layer_curves(
                        parallel_l2_norms=parallel_l2_norms,
                        perpendicular_l2_norms=perpendicular_l2_norms,
                        parallel_ratios=parallel_ratios,
                        perpendicular_ratios=perpendicular_ratios,
                        signed_parallel_ratios=signed_parallel_ratios,
                        layer_indices=layer_indices,
                        output_dir=self.sample_dir,
                        file_prefix=f"prefill_{prefill_index:03d}",
                        title_prefix=f"Random Sample {self.sample_index + 1} Attention Output Decomposition",
                    )
                )
            except Exception as plot_exc:
                record["status"] = "saved_npz_plot_error"
                record["plot_error"] = str(plot_exc)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)

        self._write_sample_json()
        return record

    def finalize(self, item):
        self.result_summary = extract_result_summary(item)
        self._write_sample_json()
        self.run_writer.register_sample(self)

    def build_capture_status(self):
        keys = [
            "prefill_index",
            "label",
            "status",
            "token_count",
            "ratio_shape",
            "layer_indices",
            "reason",
            "error",
            "plot_error",
        ]
        return [{key: record[key] for key in keys if key in record} for record in self.prefills]

    def _write_sample_json(self):
        sample_payload = {
            "sample_id": self.sample_id,
            "sample_index": self.sample_index,
            "item": extract_result_summary(self.item),
            "result": self.result_summary,
            "prefills": self.prefills,
        }
        with open(self.sample_json_path, "w", encoding="utf-8") as fout:
            json.dump(sample_payload, fout, ensure_ascii=False, indent=2)
