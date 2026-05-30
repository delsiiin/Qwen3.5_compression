import json
import os

import numpy as np
import torch

from attn_heatmap import (
    build_run_dir,
    build_token_entries,
    extract_result_summary,
    get_text_config,
    sanitize_slug,
)


SIMILARITY_REDUCTION = "flatten_window"
SIMILARITY_METRIC_COSINE = "cosine_similarity"
SIMILARITY_METRIC_L2_DIFF = "l2_norm_difference"
SIMILARITY_STATE_HIDDEN = "hidden_states"
SIMILARITY_STATE_QUERY = "query_states"
SIMILARITY_STATE_HIDDEN_L2_DIFF = "hidden_states_l2_diff"
SUPPORTED_SIMILARITY_STATES = {
    SIMILARITY_STATE_HIDDEN,
    SIMILARITY_STATE_QUERY,
    SIMILARITY_STATE_HIDDEN_L2_DIFF,
}


def get_decoder_layers(model):
    candidate_paths = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidate_paths:
        module = model
        for attr in path:
            module = getattr(module, attr, None)
            if module is None:
                break
        try:
            module_len = len(module) if module is not None else 0
        except TypeError:
            module_len = 0
        if module_len > 0:
            return list(module)
    return []


def extract_hidden_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0] if output else None
    if isinstance(output, dict):
        if output.get("last_hidden_state") is not None:
            return output.get("last_hidden_state")
        return output.get("hidden_states")
    return output


def build_similarity_from_layer_windows(layer_windows):
    if not layer_windows:
        raise ValueError("No layer hidden states were captured.")
    vectors = []
    layer_indices = []
    for layer_idx, window_state in layer_windows:
        vectors.append(window_state.reshape(-1).cpu())
        layer_indices.append(int(layer_idx))
    vectors = torch.stack(vectors, dim=0)
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1, eps=1e-12)
    similarity = torch.matmul(vectors, vectors.transpose(0, 1)).cpu().numpy()
    return similarity.astype(np.float32, copy=False), layer_indices


def build_l2_difference_from_layer_windows(layer_windows):
    if not layer_windows:
        raise ValueError("No layer hidden states were captured.")
    vectors = []
    layer_indices = []
    for layer_idx, window_state in layer_windows:
        vectors.append(window_state.reshape(-1).cpu())
        layer_indices.append(int(layer_idx))
    vectors = torch.stack(vectors, dim=0).to(dtype=torch.float32)
    squared_norms = (vectors * vectors).sum(dim=1, keepdim=True)
    squared_distances = squared_norms + squared_norms.transpose(0, 1)
    squared_distances = squared_distances - (2.0 * torch.matmul(vectors, vectors.transpose(0, 1)))
    distances = torch.sqrt(torch.clamp(squared_distances, min=0.0))
    distances.fill_diagonal_(0.0)
    return distances.cpu().numpy().astype(np.float32, copy=False), layer_indices


def get_query_window_similarity_metric(similarity_state):
    if similarity_state == SIMILARITY_STATE_HIDDEN_L2_DIFF:
        return SIMILARITY_METRIC_L2_DIFF
    return SIMILARITY_METRIC_COSINE


def get_query_window_similarity_colorbar_label(similarity_state):
    metric = get_query_window_similarity_metric(similarity_state)
    if metric == SIMILARITY_METRIC_L2_DIFF:
        return "L2 norm of hidden-state difference"
    return "Cosine similarity"


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_to_query(query_states, position_embeddings):
    if position_embeddings is None:
        return query_states
    cos, sin = position_embeddings
    cos = cos.to(device=query_states.device, dtype=query_states.dtype).unsqueeze(1)
    sin = sin.to(device=query_states.device, dtype=query_states.dtype).unsqueeze(1)
    rotary_dim = min(cos.shape[-1], query_states.shape[-1])
    cos = cos[..., :rotary_dim]
    sin = sin[..., :rotary_dim]
    query_rot = query_states[..., :rotary_dim]
    query_pass = query_states[..., rotary_dim:]
    query_embed = (query_rot * cos) + (rotate_half(query_rot) * sin)
    if query_pass.shape[-1] == 0:
        return query_embed
    return torch.cat([query_embed, query_pass], dim=-1)


def extract_attention_arg(args, kwargs, name, index, default=None):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def build_query_states_from_attention(attn_module, hidden_states, position_embeddings):
    if hidden_states is None or not torch.is_tensor(hidden_states) or hidden_states.ndim < 3:
        return None
    q_proj = getattr(attn_module, "q_proj", None)
    head_dim = getattr(attn_module, "head_dim", None)
    if q_proj is None or head_dim is None:
        return None

    input_shape = hidden_states.shape[:-1]
    projected = q_proj(hidden_states)
    config = getattr(attn_module, "config", None)
    num_attention_heads = getattr(attn_module, "num_heads", None)
    if num_attention_heads is None and config is not None:
        num_attention_heads = getattr(config, "num_attention_heads", None)
    expected_query_features = (
        int(num_attention_heads) * int(head_dim)
        if num_attention_heads is not None
        else projected.shape[-1]
    )
    if projected.shape[-1] == expected_query_features * 2:
        query_states, _gate = torch.chunk(
            projected.view(*input_shape, -1, head_dim * 2),
            2,
            dim=-1,
        )
    else:
        query_states = projected.view(*input_shape, -1, head_dim)

    q_norm = getattr(attn_module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)
    query_states = query_states.transpose(1, 2)
    return apply_rotary_pos_emb_to_query(query_states, position_embeddings)


def run_observation_forward(model, inputs, output_hidden_states=False):
    forward_kwargs = {
        **inputs,
        "use_cache": False,
        "return_dict": True,
        "logits_to_keep": 1,
    }
    if output_hidden_states:
        forward_kwargs["output_hidden_states"] = True
    with torch.inference_mode():
        try:
            return model(**forward_kwargs)
        except TypeError:
            forward_kwargs.pop("logits_to_keep", None)
            return model(**forward_kwargs)


def collect_hidden_state_layer_windows(model, inputs, actual_window_size):
    layers = get_decoder_layers(model)
    if layers:
        captured_windows = {}
        handles = []

        def make_hook(layer_idx):
            def hook(_module, _inputs, output):
                hidden = extract_hidden_tensor(output)
                if hidden is None or not torch.is_tensor(hidden) or hidden.ndim < 3:
                    return
                captured_windows[layer_idx] = (
                    hidden[0, -actual_window_size:, :]
                    .detach()
                    .to(dtype=torch.float32)
                    .cpu()
                )

            return hook

        try:
            for layer_idx, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(make_hook(layer_idx)))
            run_observation_forward(model, inputs, output_hidden_states=False)
        finally:
            for handle in handles:
                handle.remove()

        layer_windows = [
            (layer_idx, captured_windows[layer_idx])
            for layer_idx in range(len(layers))
            if layer_idx in captured_windows
        ]
        if layer_windows:
            return layer_windows

    outputs = run_observation_forward(model, inputs, output_hidden_states=True)

    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise ValueError("Model did not return hidden states; cannot compute query window similarity.")

    text_config = get_text_config(model)
    num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    hidden_states = [state for state in hidden_states if state is not None]
    if num_hidden_layers is not None and len(hidden_states) >= num_hidden_layers + 1:
        layer_states = hidden_states[-num_hidden_layers:]
        layer_indices = list(range(num_hidden_layers))
    else:
        layer_states = hidden_states
        layer_indices = list(range(len(layer_states)))

    if not layer_states:
        raise ValueError("No layer hidden states were returned.")

    layer_windows = [
        (
            layer_idx,
            state[0, -actual_window_size:, :].detach().to(dtype=torch.float32).cpu(),
        )
        for layer_idx, state in zip(layer_indices, layer_states)
    ]
    return layer_windows


def compute_hidden_state_query_window_similarity(model, inputs, actual_window_size):
    layer_windows = collect_hidden_state_layer_windows(model, inputs, actual_window_size)
    return build_similarity_from_layer_windows(layer_windows)


def compute_hidden_state_query_window_l2_difference(model, inputs, actual_window_size):
    layer_windows = collect_hidden_state_layer_windows(model, inputs, actual_window_size)
    return build_l2_difference_from_layer_windows(layer_windows)


def compute_query_state_query_window_similarity(model, inputs, actual_window_size):
    layers = get_decoder_layers(model)
    if not layers:
        raise ValueError("No decoder layers were found; cannot capture query states.")

    captured_windows = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            hidden_states = extract_attention_arg(args, kwargs, "hidden_states", 0)
            position_embeddings = extract_attention_arg(args, kwargs, "position_embeddings", 1)
            query_states = build_query_states_from_attention(
                attn_module=module,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
            )
            if query_states is None or not torch.is_tensor(query_states) or query_states.ndim < 4:
                return
            captured_windows[layer_idx] = (
                query_states[0, :, -actual_window_size:, :]
                .transpose(0, 1)
                .detach()
                .to(dtype=torch.float32)
                .cpu()
            )

        return hook

    try:
        for layer_idx, layer in enumerate(layers):
            attn_module = getattr(layer, "self_attn", None)
            if attn_module is None:
                continue
            handles.append(attn_module.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))
        run_observation_forward(model, inputs, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()

    layer_windows = [
        (layer_idx, captured_windows[layer_idx])
        for layer_idx in range(len(layers))
        if layer_idx in captured_windows
    ]
    if not layer_windows:
        raise ValueError("No layer query states were captured.")
    return build_similarity_from_layer_windows(layer_windows)


def compute_query_window_similarity(model, inputs, window_size, similarity_state=SIMILARITY_STATE_HIDDEN):
    if similarity_state not in SUPPORTED_SIMILARITY_STATES:
        raise ValueError(
            f"Unsupported query window similarity state: {similarity_state}. "
            f"Expected one of {sorted(SUPPORTED_SIMILARITY_STATES)}."
        )
    input_len = int(inputs["input_ids"].shape[-1])
    actual_window_size = min(int(window_size), input_len)
    if actual_window_size < 1:
        raise ValueError("Cannot compute similarity for an empty prompt.")

    if similarity_state == SIMILARITY_STATE_QUERY:
        similarity, layer_indices = compute_query_state_query_window_similarity(
            model=model,
            inputs=inputs,
            actual_window_size=actual_window_size,
        )
    elif similarity_state == SIMILARITY_STATE_HIDDEN_L2_DIFF:
        similarity, layer_indices = compute_hidden_state_query_window_l2_difference(
            model=model,
            inputs=inputs,
            actual_window_size=actual_window_size,
        )
    else:
        similarity, layer_indices = compute_hidden_state_query_window_similarity(
            model=model,
            inputs=inputs,
            actual_window_size=actual_window_size,
        )
    return similarity, layer_indices, actual_window_size


def plot_query_window_similarity_heatmap(similarity, layer_indices, output_path, title, similarity_state):
    cache_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "qwen35_compression_matplotlib_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_layers = len(layer_indices)
    fig_size = max(6.0, min(14.0, 0.28 * num_layers + 3.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=160)
    if get_query_window_similarity_metric(similarity_state) == SIMILARITY_METRIC_L2_DIFF:
        image = ax.imshow(similarity, cmap="viridis", vmin=0.0, interpolation="nearest")
    else:
        image = ax.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Layer id")
    ax.set_ylabel("Layer id")
    tick_step = max(1, num_layers // 16)
    tick_positions = np.arange(0, num_layers, tick_step)
    tick_labels = [str(layer_indices[idx]) for idx in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(get_query_window_similarity_colorbar_label(similarity_state))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


class QueryWindowSimilarityRunWriter:
    def __init__(self, root_dir, model_name, out_file, window_size, max_prefill_tokens, similarity_state):
        if similarity_state not in SUPPORTED_SIMILARITY_STATES:
            raise ValueError(
                f"Unsupported query window similarity state: {similarity_state}. "
                f"Expected one of {sorted(SUPPORTED_SIMILARITY_STATES)}."
            )
        self.root_dir = root_dir
        self.model_name = model_name
        self.out_file = os.path.abspath(out_file)
        self.window_size = int(window_size)
        self.max_prefill_tokens = int(max_prefill_tokens) if max_prefill_tokens is not None else None
        self.similarity_state = similarity_state
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
        return QueryWindowSimilaritySampleWriter(self, item, sample_index)

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
            "query_window_size": self.window_size,
            "query_window_similarity_state": self.similarity_state,
            "query_window_similarity_metric": get_query_window_similarity_metric(self.similarity_state),
            "similarity_reduction": SIMILARITY_REDUCTION,
            "query_window_max_prefill_tokens": self.max_prefill_tokens,
            "query_window_prefill_cap_mode": "fixed" if self.max_prefill_tokens is not None else "none",
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as fout:
            json.dump(manifest, fout, ensure_ascii=False, indent=2)


class QueryWindowSimilaritySampleWriter:
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
        actual_window_size = min(self.run_writer.window_size, len(input_ids))
        window_token_start = max(0, len(input_ids) - actual_window_size)
        record = {
            "prefill_index": prefill_index,
            "label": label,
            "status": "pending",
            "prompt_text": prompt_text,
            "max_prefill_tokens": prefill_cap,
            "token_count": len(input_ids),
            "query_window_size": self.run_writer.window_size,
            "query_window_similarity_state": self.run_writer.similarity_state,
            "query_window_similarity_metric": get_query_window_similarity_metric(self.run_writer.similarity_state),
            "similarity_reduction": SIMILARITY_REDUCTION,
            "actual_window_size": actual_window_size,
            "window_token_start": window_token_start,
            "window_token_ids": input_ids[window_token_start:],
            "tokens": build_token_entries(tokenizer, input_ids),
            "similarity_file": None,
            "heatmap_file": None,
            "similarity_shape": None,
            "layer_indices": [],
        }
        self.prefills.append(record)
        self._write_sample_json()

        if prefill_cap is not None and len(input_ids) > prefill_cap:
            record["status"] = "skipped_over_cap"
            record["reason"] = f"Prompt token count {len(input_ids)} exceeds cap {prefill_cap}."
            self._write_sample_json()
            return record

        try:
            similarity, layer_indices, actual_window_size = compute_query_window_similarity(
                model=model,
                inputs=inputs,
                window_size=self.run_writer.window_size,
                similarity_state=self.run_writer.similarity_state,
            )
            similarity_file_name = (
                f"prefill_{prefill_index:03d}_{self.run_writer.similarity_state}_query_window_similarity.npz"
            )
            heatmap_file_name = (
                f"prefill_{prefill_index:03d}_{self.run_writer.similarity_state}_query_window_similarity.png"
            )
            similarity_path = os.path.join(self.sample_dir, similarity_file_name)
            heatmap_path = os.path.join(self.sample_dir, heatmap_file_name)
            np.savez_compressed(
                similarity_path,
                similarity=similarity,
                layer_indices=np.asarray(layer_indices, dtype=np.int16),
                window_token_ids=np.asarray(record["window_token_ids"], dtype=np.int64),
                window_token_start=np.asarray(window_token_start, dtype=np.int64),
                actual_window_size=np.asarray(actual_window_size, dtype=np.int64),
                query_window_similarity_state=np.asarray(self.run_writer.similarity_state),
                query_window_similarity_metric=np.asarray(
                    get_query_window_similarity_metric(self.run_writer.similarity_state)
                ),
                similarity_reduction=np.asarray(SIMILARITY_REDUCTION),
            )
            record["status"] = "saved"
            record["similarity_file"] = similarity_file_name
            record["similarity_shape"] = list(similarity.shape)
            record["layer_indices"] = layer_indices
            record["actual_window_size"] = actual_window_size
            try:
                plot_query_window_similarity_heatmap(
                    similarity=similarity,
                    layer_indices=layer_indices,
                    output_path=heatmap_path,
                    title=(
                        f"{label}: last {actual_window_size} token "
                        f"{self.run_writer.similarity_state} "
                        f"{get_query_window_similarity_metric(self.run_writer.similarity_state)}"
                    ),
                    similarity_state=self.run_writer.similarity_state,
                )
                record["heatmap_file"] = heatmap_file_name
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
            "actual_window_size",
            "query_window_similarity_state",
            "query_window_similarity_metric",
            "similarity_shape",
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
