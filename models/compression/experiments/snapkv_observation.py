import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import compute_attention_scores


@dataclass(frozen=True)
class SnapKVObservationConfig:
    budget: int
    window_size: int = 8
    kernel_size: int = 7
    max_prefill_tokens: int | None = None

    def __post_init__(self):
        if self.budget < 1:
            raise ValueError("budget must be at least 1.")
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be at least 1.")
        if self.budget <= self.window_size:
            raise ValueError("budget must be greater than window_size.")
        if self.max_prefill_tokens is not None and self.max_prefill_tokens < 1:
            raise ValueError("max_prefill_tokens must be at least 1 when provided.")


@dataclass
class SnapKVSelection:
    attn_cache: torch.Tensor
    indices: torch.Tensor


@dataclass
class CapturedLayer:
    layer_idx: int
    attention: torch.nn.Module
    hidden_window: torch.Tensor
    query_states: torch.Tensor
    key_states: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None


@dataclass
class SnapKVObservationResult:
    summary: dict[str, Any]
    orig_attn_cache: np.ndarray
    neighbor_avg_attn_cache: np.ndarray
    orig_indices: np.ndarray
    neighbor_avg_indices: np.ndarray
    layer_indices: np.ndarray


def compute_snapkv_selection(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    budget: int,
    window_size: int = 8,
    kernel_size: int = 7,
) -> SnapKVSelection:
    kv_cache_len = key_states.shape[-2]
    if kv_cache_len < budget:
        raise ValueError("kv_cache_len must be at least budget.")
    if kv_cache_len <= window_size:
        raise ValueError("kv_cache_len must be greater than window_size.")

    topk = budget - window_size
    candidate_len = kv_cache_len - window_size
    if topk > candidate_len:
        raise ValueError("budget - window_size cannot exceed candidate key count.")

    attn_weights = compute_attention_scores(query_states, key_states)
    attn_weights_sum = (
        F.softmax(
            attn_weights[:, :, -window_size:, : -window_size],
            dim=-1,
            dtype=torch.float32,
        )
        .mean(dim=-2)
        .to(query_states.dtype)
    )
    attn_cache = F.max_pool1d(
        attn_weights_sum,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    )
    indices = attn_cache.topk(topk, dim=-1).indices
    return SnapKVSelection(attn_cache=attn_cache, indices=indices)


def compute_snapkv_neighbor_observation(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    config: SnapKVObservationConfig,
) -> SnapKVObservationResult:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("inputs must include input_ids.")
    if input_ids.shape[0] != 1:
        raise ValueError("SnapKV neighbor observation currently supports batch size 1 only.")

    token_count = int(input_ids.shape[-1])
    summary = {
        "status": "pending",
        "config": asdict(config),
        "token_count": token_count,
        "valid_layer_indices": [],
        "layers": [],
    }
    if config.max_prefill_tokens is not None and token_count > config.max_prefill_tokens:
        summary["status"] = "skipped_over_cap"
        summary["reason"] = (
            f"Prompt token count {token_count} exceeds cap {config.max_prefill_tokens}."
        )
        return _empty_result(summary)

    layers = _get_decoder_layers(model)
    if len(layers) < 3:
        summary["status"] = "error"
        summary["reason"] = "At least three decoder layers are required."
        return _empty_result(summary)

    captured = _capture_layers(model, inputs, layers, config.window_size)
    orig_attn_cache = []
    neighbor_avg_attn_cache = []
    orig_indices = []
    neighbor_avg_indices = []
    valid_layer_indices = []

    for layer_idx in range(1, len(layers) - 1):
        layer_summary = {"layer_idx": layer_idx, "status": "pending"}
        prev_capture = captured.get(layer_idx - 1)
        cur_capture = captured.get(layer_idx)
        next_capture = captured.get(layer_idx + 1)
        if prev_capture is None or cur_capture is None or next_capture is None:
            layer_summary["status"] = "skipped_missing_capture"
            summary["layers"].append(layer_summary)
            continue

        kv_cache_len = int(cur_capture.key_states.shape[-2])
        candidate_len = kv_cache_len - config.window_size
        topk = config.budget - config.window_size
        layer_summary.update(
            {
                "kv_cache_len": kv_cache_len,
                "candidate_len": candidate_len,
                "topk": topk,
            }
        )
        if kv_cache_len < config.budget:
            layer_summary["status"] = "skipped_short_kv_cache"
            summary["layers"].append(layer_summary)
            continue
        if candidate_len < topk or candidate_len < 1:
            layer_summary["status"] = "skipped_insufficient_candidates"
            summary["layers"].append(layer_summary)
            continue

        try:
            orig_selection = compute_snapkv_selection(
                cur_capture.query_states,
                cur_capture.key_states,
                budget=config.budget,
                window_size=config.window_size,
                kernel_size=config.kernel_size,
            )
            avg_hidden_window = (
                prev_capture.hidden_window
                + cur_capture.hidden_window
                + next_capture.hidden_window
            ) / 3.0
            with torch.inference_mode():
                neighbor_query = _project_query_window(
                    cur_capture.attention,
                    avg_hidden_window,
                    cur_capture.position_embeddings,
                )
            neighbor_query = neighbor_query.detach().to(dtype=torch.float32, device="cpu")
            neighbor_selection = compute_snapkv_selection(
                neighbor_query,
                cur_capture.key_states,
                budget=config.budget,
                window_size=config.window_size,
                kernel_size=config.kernel_size,
            )
        except Exception as exc:
            layer_summary["status"] = "error"
            layer_summary["error"] = str(exc)
            summary["layers"].append(layer_summary)
            continue

        layer_summary.update(
            _build_layer_metrics(
                layer_idx=layer_idx,
                orig_selection=orig_selection,
                neighbor_selection=neighbor_selection,
            )
        )
        summary["layers"].append(layer_summary)

        valid_layer_indices.append(layer_idx)
        orig_attn_cache.append(_to_numpy(orig_selection.attn_cache.squeeze(0), dtype=np.float32))
        neighbor_avg_attn_cache.append(
            _to_numpy(neighbor_selection.attn_cache.squeeze(0), dtype=np.float32)
        )
        orig_indices.append(_to_numpy(orig_selection.indices.squeeze(0), dtype=np.int64))
        neighbor_avg_indices.append(_to_numpy(neighbor_selection.indices.squeeze(0), dtype=np.int64))

    summary["valid_layer_indices"] = valid_layer_indices
    summary["status"] = "saved" if valid_layer_indices else "no_valid_layers"
    return SnapKVObservationResult(
        summary=summary,
        orig_attn_cache=_stack_or_empty(orig_attn_cache, np.float32),
        neighbor_avg_attn_cache=_stack_or_empty(neighbor_avg_attn_cache, np.float32),
        orig_indices=_stack_or_empty(orig_indices, np.int64),
        neighbor_avg_indices=_stack_or_empty(neighbor_avg_indices, np.int64),
        layer_indices=np.asarray(valid_layer_indices, dtype=np.int16),
    )


def save_snapkv_neighbor_observation(
    result: SnapKVObservationResult,
    output_dir: str,
    prefix: str = "snapkv_observation",
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    npz_path = os.path.join(output_dir, f"{prefix}.npz")
    image_paths = plot_snapkv_neighbor_observation(result, output_dir, prefix=prefix)
    result.summary["image_files"] = {
        name: os.path.basename(path) for name, path in image_paths.items()
    }

    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(result.summary, fout, ensure_ascii=False, indent=2)

    np.savez_compressed(
        npz_path,
        orig_attn_cache=result.orig_attn_cache,
        neighbor_avg_attn_cache=result.neighbor_avg_attn_cache,
        orig_indices=result.orig_indices,
        neighbor_avg_indices=result.neighbor_avg_indices,
        layer_indices=result.layer_indices,
    )
    return {"summary": summary_path, "npz": npz_path, "images": image_paths}


def plot_snapkv_neighbor_observation(
    result: SnapKVObservationResult,
    output_dir: str,
    prefix: str = "snapkv_observation",
) -> dict[str, str]:
    if result.layer_indices.size == 0:
        return {}

    _setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    image_paths = {}

    image_paths["indices"] = os.path.join(output_dir, f"{prefix}_retained_indices.png")
    _plot_retained_indices(result, image_paths["indices"], plt)

    image_paths["overlap"] = os.path.join(output_dir, f"{prefix}_indices_overlap.png")
    _plot_indices_overlap(result, image_paths["overlap"], plt)

    image_paths["attn_cache"] = os.path.join(output_dir, f"{prefix}_attn_cache.png")
    _plot_attn_cache(result, image_paths["attn_cache"], plt)

    return image_paths


def _capture_layers(model, inputs, layers, window_size):
    captured = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs, _output):
            hidden_states = _get_arg_or_kwarg(args, kwargs, 0, "hidden_states")
            if hidden_states is None or not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
                return
            position_embeddings = _get_arg_or_kwarg(args, kwargs, 1, "position_embeddings")
            actual_window_size = min(int(window_size), int(hidden_states.shape[-2]))
            if actual_window_size < 1:
                return
            with torch.inference_mode():
                hidden_window = hidden_states[:, -actual_window_size:, :].detach()
                query_states = _project_query_window(module, hidden_window, position_embeddings)
                key_states = _project_key_states(module, hidden_states, position_embeddings)
            captured[layer_idx] = CapturedLayer(
                layer_idx=layer_idx,
                attention=module,
                hidden_window=hidden_window.detach().to(dtype=torch.float32, device="cpu").clone(),
                query_states=query_states.detach().to(dtype=torch.float32, device="cpu").clone(),
                key_states=key_states.detach().to(dtype=torch.float32, device="cpu").clone(),
                position_embeddings=_detach_position_embeddings(position_embeddings, actual_window_size),
            )

        return hook

    try:
        for layer_idx, layer in enumerate(layers):
            attention = _get_attention_module(layer)
            if attention is None:
                continue
            try:
                handles.append(attention.register_forward_hook(make_hook(layer_idx), with_kwargs=True))
            except TypeError:
                handles.append(attention.register_forward_hook(_wrap_hook_without_kwargs(make_hook(layer_idx))))
        _run_observation_forward(model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    return captured


def _setup_matplotlib_cache():
    cache_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "snapkv_observation_matplotlib_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)


def _plot_retained_indices(result, output_path, plt):
    orig = np.asarray(result.orig_indices)
    neighbor = np.asarray(result.neighbor_avg_indices)
    row_labels = _layer_head_labels(result.layer_indices, orig.shape[1])
    fig_width = _wide_fig_size(orig.shape[-1])
    fig_height = max(4.0, min(18.0, 0.22 * max(1, len(row_labels)) + 2.0))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        dpi=160,
        sharex=True,
        constrained_layout=True,
    )
    arrays = [
        ("Original SnapKV retained indices", orig.reshape(-1, orig.shape[-1])),
        ("Neighbor-average query retained indices", neighbor.reshape(-1, neighbor.shape[-1])),
    ]
    vmin = float(min(orig.min(), neighbor.min()))
    vmax = float(max(orig.max(), neighbor.max()))
    for ax, (title, values) in zip(axes, arrays):
        image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("Layer:head")
        _set_sparse_row_labels(ax, row_labels)
    axes[-1].set_xlabel("Top-k rank")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Retained token index")
    fig.savefig(output_path)
    plt.close(fig)


def _plot_indices_overlap(result, output_path, plt):
    overlap = _compute_overlap_matrix(result.orig_indices, result.neighbor_avg_indices)
    layer_labels = [str(int(layer_idx)) for layer_idx in result.layer_indices.tolist()]
    head_labels = [str(head_idx) for head_idx in range(overlap.shape[1])]
    fig_width = max(5.0, min(12.0, 0.38 * overlap.shape[1] + 3.0))
    fig_height = max(4.0, min(12.0, 0.32 * overlap.shape[0] + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    image = ax.imshow(overlap, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Retained-index overlap: original vs neighbor-average query")
    ax.set_xlabel("KV head")
    ax.set_ylabel("Layer")
    _set_sparse_ticks(ax, axis="x", labels=head_labels)
    _set_sparse_ticks(ax, axis="y", labels=layer_labels)
    if overlap.shape[0] * overlap.shape[1] <= 160:
        for layer_pos in range(overlap.shape[0]):
            for head_pos in range(overlap.shape[1]):
                value = overlap[layer_pos, head_pos]
                ax.text(
                    head_pos,
                    layer_pos,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.65 else "black",
                    fontsize=7,
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Intersection / top-k")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_attn_cache(result, output_path, plt):
    orig = np.asarray(result.orig_attn_cache, dtype=np.float32)
    neighbor = np.asarray(result.neighbor_avg_attn_cache, dtype=np.float32)
    orig_curves = orig.mean(axis=1)
    neighbor_curves = neighbor.mean(axis=1)
    token_positions = np.arange(orig_curves.shape[-1])
    layer_count = orig_curves.shape[0]

    ncols = 2 if layer_count <= 12 else 3
    nrows = int(np.ceil(layer_count / ncols))
    fig_width = max(9.0, min(24.0, ncols * 6.2))
    fig_height = max(4.0, min(28.0, nrows * 3.2))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        dpi=160,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    for layer_pos, ax in enumerate(axes_flat):
        if layer_pos >= layer_count:
            ax.axis("off")
            continue
        layer_idx = int(result.layer_indices[layer_pos])
        ax.plot(token_positions, orig_curves[layer_pos], label="Original SnapKV", linewidth=1.4)
        ax.plot(token_positions, neighbor_curves[layer_pos], label="Neighbor-average query", linewidth=1.4)
        ax.set_title(f"Layer {layer_idx}")
        ax.grid(alpha=0.25, linewidth=0.8)
        if layer_pos % ncols == 0:
            ax.set_ylabel("Score")
        if layer_pos >= (nrows - 1) * ncols:
            ax.set_xlabel("Token index")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("attn_cache score by token position", y=0.995)
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.16)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _compute_overlap_matrix(orig_indices, neighbor_indices):
    orig = np.asarray(orig_indices)
    neighbor = np.asarray(neighbor_indices)
    overlap = np.zeros(orig.shape[:2], dtype=np.float32)
    topk = orig.shape[-1]
    if topk < 1:
        return overlap
    for layer_pos in range(orig.shape[0]):
        for head_pos in range(orig.shape[1]):
            orig_set = set(orig[layer_pos, head_pos].tolist())
            neighbor_set = set(neighbor[layer_pos, head_pos].tolist())
            overlap[layer_pos, head_pos] = len(orig_set & neighbor_set) / topk
    return overlap


def _layer_head_labels(layer_indices, head_count):
    labels = []
    for layer_idx in layer_indices.tolist():
        for head_idx in range(head_count):
            labels.append(f"{int(layer_idx)}:{head_idx}")
    return labels


def _wide_fig_size(width_count):
    return max(8.0, min(24.0, 0.018 * max(1, width_count) + 6.0))


def _set_sparse_row_labels(ax, labels):
    _set_sparse_ticks(ax, axis="y", labels=labels)


def _set_sparse_ticks(ax, axis, labels):
    count = len(labels)
    if count == 0:
        return
    tick_step = max(1, count // 16)
    positions = np.arange(0, count, tick_step)
    tick_labels = [labels[pos] for pos in positions]
    if axis == "x":
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    elif axis == "y":
        ax.set_yticks(positions)
        ax.set_yticklabels(tick_labels)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def _wrap_hook_without_kwargs(hook):
    def wrapped(module, args, output):
        return hook(module, args, {}, output)

    return wrapped


def _run_observation_forward(model, inputs):
    forward_kwargs = {
        **inputs,
        "use_cache": False,
        "return_dict": True,
        "logits_to_keep": 1,
    }
    with torch.inference_mode():
        try:
            return model(**forward_kwargs)
        except TypeError:
            forward_kwargs.pop("logits_to_keep", None)
            return model(**forward_kwargs)


def _project_query_window(attention, hidden_window, position_embeddings):
    device = _module_device(attention)
    dtype = _module_dtype(attention)
    hidden_window = hidden_window.to(device=device, dtype=dtype)
    head_dim = _get_head_dim(attention)
    input_shape = hidden_window.shape[:-1]
    q_proj = attention.q_proj(hidden_window)
    num_attention_heads = _get_num_attention_heads(attention, q_proj.shape[-1], head_dim)

    if _uses_gated_query_projection(attention, q_proj.shape[-1], num_attention_heads, head_dim):
        query_states, _gate_states = torch.chunk(
            q_proj.view(*input_shape, num_attention_heads, head_dim * 2),
            2,
            dim=-1,
        )
    else:
        query_states = q_proj.view(*input_shape, num_attention_heads, head_dim)

    q_norm = getattr(attention, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)
    query_states = query_states.transpose(1, 2)

    if position_embeddings is None:
        return query_states
    cos, sin = _position_embeddings_to_device(position_embeddings, device)
    cos = _slice_position_embedding(cos, query_states.shape[-2])
    sin = _slice_position_embedding(sin, query_states.shape[-2])
    return _apply_rotary(query_states, cos, sin)


def _project_key_states(attention, hidden_states, position_embeddings):
    device = _module_device(attention)
    dtype = _module_dtype(attention)
    hidden_states = hidden_states.to(device=device, dtype=dtype)
    head_dim = _get_head_dim(attention)
    input_shape = hidden_states.shape[:-1]
    key_proj = attention.k_proj(hidden_states)
    num_key_value_heads = _get_num_key_value_heads(attention, key_proj.shape[-1], head_dim)
    key_states = key_proj.view(*input_shape, num_key_value_heads, head_dim)
    k_norm = getattr(attention, "k_norm", None)
    if k_norm is not None:
        key_states = k_norm(key_states)
    key_states = key_states.transpose(1, 2)

    if position_embeddings is None:
        return key_states
    cos, sin = _position_embeddings_to_device(position_embeddings, device)
    cos = _slice_position_embedding(cos, key_states.shape[-2])
    sin = _slice_position_embedding(sin, key_states.shape[-2])
    return _apply_rotary(key_states, cos, sin)


def _apply_rotary(states, cos, sin):
    cos = _unsqueeze_position_embedding(cos, states)
    sin = _unsqueeze_position_embedding(sin, states)
    return (states * cos) + (_rotate_half(states) * sin)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _unsqueeze_position_embedding(position_embedding, states):
    if position_embedding.ndim == 2:
        position_embedding = position_embedding.unsqueeze(0)
    while position_embedding.ndim < states.ndim:
        position_embedding = position_embedding.unsqueeze(1)
    return position_embedding


def _slice_position_embedding(position_embedding, seq_len):
    if position_embedding.ndim == 2:
        return position_embedding[-seq_len:, :]
    if position_embedding.ndim >= 3:
        return position_embedding[:, -seq_len:, :]
    return position_embedding


def _detach_position_embeddings(position_embeddings, window_size):
    if position_embeddings is None:
        return None
    cos, sin = position_embeddings
    cos = _slice_position_embedding(cos.detach(), window_size).to(dtype=torch.float32, device="cpu").clone()
    sin = _slice_position_embedding(sin.detach(), window_size).to(dtype=torch.float32, device="cpu").clone()
    return cos, sin


def _position_embeddings_to_device(position_embeddings, device):
    cos, sin = position_embeddings
    return cos.to(device=device), sin.to(device=device)


def _get_decoder_layers(model):
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


def _get_attention_module(layer):
    for attr in ("self_attn", "attention", "attn"):
        attention = getattr(layer, attr, None)
        if attention is not None and hasattr(attention, "q_proj") and hasattr(attention, "k_proj"):
            return attention
    return None


def _get_arg_or_kwarg(args, kwargs, index, name):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return None


def _get_head_dim(attention):
    head_dim = getattr(attention, "head_dim", None)
    if head_dim is None:
        config = getattr(attention, "config", None)
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        raise ValueError("Cannot infer attention head_dim.")
    return int(head_dim)


def _get_num_attention_heads(attention, q_projection_dim, head_dim):
    config = getattr(attention, "config", None)
    num_heads = getattr(attention, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(attention, "num_attention_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is not None:
        return int(num_heads)
    return int(q_projection_dim // head_dim)


def _get_num_key_value_heads(attention, key_projection_dim, head_dim):
    config = getattr(attention, "config", None)
    num_heads = getattr(attention, "num_key_value_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_key_value_heads", None)
    if num_heads is not None:
        return int(num_heads)
    return int(key_projection_dim // head_dim)


def _uses_gated_query_projection(attention, q_projection_dim, num_attention_heads, head_dim):
    return q_projection_dim == num_attention_heads * head_dim * 2


def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _module_dtype(module):
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _build_layer_metrics(layer_idx, orig_selection, neighbor_selection):
    orig_cache = orig_selection.attn_cache.detach().to(dtype=torch.float32, device="cpu")
    neighbor_cache = neighbor_selection.attn_cache.detach().to(dtype=torch.float32, device="cpu")
    orig_idx = orig_selection.indices.detach().to(device="cpu")
    neighbor_idx = neighbor_selection.indices.detach().to(device="cpu")

    per_head_overlap = []
    per_head_jaccard = []
    per_head_cosine = []
    per_head_l1 = []
    per_head_l2 = []
    topk = int(orig_idx.shape[-1])
    for head_idx in range(orig_idx.shape[1]):
        orig_set = set(orig_idx[0, head_idx].tolist())
        neighbor_set = set(neighbor_idx[0, head_idx].tolist())
        intersection = len(orig_set & neighbor_set)
        union = len(orig_set | neighbor_set)
        per_head_overlap.append(float(intersection / topk if topk else 0.0))
        per_head_jaccard.append(float(intersection / union if union else 0.0))

        orig_vec = orig_cache[0, head_idx].reshape(-1)
        neighbor_vec = neighbor_cache[0, head_idx].reshape(-1)
        per_head_cosine.append(float(F.cosine_similarity(orig_vec, neighbor_vec, dim=0).item()))
        diff = orig_vec - neighbor_vec
        per_head_l1.append(float(diff.abs().mean().item()))
        per_head_l2.append(float(torch.sqrt((diff * diff).mean()).item()))

    histogram_l1 = _histogram_l1(orig_cache.numpy(), neighbor_cache.numpy())
    return {
        "status": "saved",
        "attn_cache_shape": list(orig_cache.shape),
        "indices_shape": list(orig_idx.shape),
        "overlap": _summarize_values(per_head_overlap, include_per_head=True),
        "jaccard": _summarize_values(per_head_jaccard, include_per_head=True),
        "attn_cache_cosine": _summarize_values(per_head_cosine, include_per_head=True),
        "attn_cache_l1_diff": _summarize_values(per_head_l1, include_per_head=True),
        "attn_cache_l2_diff": _summarize_values(per_head_l2, include_per_head=True),
        "orig_attn_cache": _distribution_stats(orig_cache.numpy()),
        "neighbor_avg_attn_cache": _distribution_stats(neighbor_cache.numpy()),
        "histogram_l1": histogram_l1,
    }


def _summarize_values(values, include_per_head=False):
    array = np.asarray(values, dtype=np.float64)
    summary = {
        "mean": float(array.mean()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
    }
    if include_per_head:
        summary["per_head"] = [float(value) for value in array.tolist()]
    return summary


def _distribution_stats(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "quantiles": {},
        }
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "quantiles": {
            "0.00": float(np.quantile(array, 0.0)),
            "0.25": float(np.quantile(array, 0.25)),
            "0.50": float(np.quantile(array, 0.50)),
            "0.75": float(np.quantile(array, 0.75)),
            "1.00": float(np.quantile(array, 1.0)),
        },
    }


def _histogram_l1(orig_values, neighbor_values, bins=50):
    orig = np.asarray(orig_values, dtype=np.float64).reshape(-1)
    neighbor = np.asarray(neighbor_values, dtype=np.float64).reshape(-1)
    if orig.size == 0 or neighbor.size == 0:
        return 0.0
    lower = float(min(orig.min(), neighbor.min()))
    upper = float(max(orig.max(), neighbor.max()))
    if lower == upper:
        return 0.0
    orig_hist, bin_edges = np.histogram(orig, bins=bins, range=(lower, upper), density=True)
    neighbor_hist, _ = np.histogram(neighbor, bins=bin_edges, density=True)
    bin_widths = np.diff(bin_edges)
    return float(np.sum(np.abs(orig_hist - neighbor_hist) * bin_widths))


def _to_numpy(tensor, dtype):
    return tensor.detach().to(device="cpu").numpy().astype(dtype, copy=False)


def _stack_or_empty(items, dtype):
    if not items:
        return np.asarray([], dtype=dtype)
    return np.stack(items, axis=0).astype(dtype, copy=False)


def _empty_result(summary):
    return SnapKVObservationResult(
        summary=summary,
        orig_attn_cache=np.asarray([], dtype=np.float32),
        neighbor_avg_attn_cache=np.asarray([], dtype=np.float32),
        orig_indices=np.asarray([], dtype=np.int64),
        neighbor_avg_indices=np.asarray([], dtype=np.int64),
        layer_indices=np.asarray([], dtype=np.int16),
    )
