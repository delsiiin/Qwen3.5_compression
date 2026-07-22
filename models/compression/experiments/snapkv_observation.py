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
class CapturedLayer:
    layer_idx: int
    query_states: torch.Tensor
    key_states: torch.Tensor


@dataclass
class SnapKVTopKOverlapResult:
    summary: dict[str, Any]
    snapkv_topk_indices: np.ndarray
    snapkv_topk_overlap: np.ndarray
    snapkv_topk_head_overlap: np.ndarray
    snapkv_topk_layer_indices: np.ndarray


def compute_snapkv_observation(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    config: SnapKVObservationConfig,
) -> SnapKVTopKOverlapResult:
    return compute_snapkv_topk_overlap_observation(
        model=model,
        inputs=inputs,
        config=config,
    )


def compute_snapkv_topk_overlap_observation(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    config: SnapKVObservationConfig,
) -> SnapKVTopKOverlapResult:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("inputs must include input_ids.")
    if input_ids.shape[0] != 1:
        raise ValueError("SnapKV topk overlap observation currently supports batch size 1 only.")

    token_count = int(input_ids.shape[-1])
    summary = {
        "status": "pending",
        "config": asdict(config),
        "token_count": token_count,
        "valid_layer_indices": [],
        "layers": [],
        "overlap_metric": "mean_head_intersection_over_topk",
    }
    if config.max_prefill_tokens is not None and token_count > config.max_prefill_tokens:
        summary["status"] = "skipped_over_cap"
        summary["reason"] = (
            f"Prompt token count {token_count} exceeds cap {config.max_prefill_tokens}."
        )
        return _empty_topk_overlap_result(summary)

    layers = _get_decoder_layers(model)
    if not layers:
        summary["status"] = "error"
        summary["reason"] = "No decoder layers were found."
        return _empty_topk_overlap_result(summary)

    captured = _capture_layers(model, inputs, layers, config.window_size)
    topk_result = _compute_snapkv_topk_overlap(captured, config)
    summary.update(topk_result["summary"])
    topk_indices = _stack_or_empty(topk_result["indices"], np.int64)
    overlap = compute_snapkv_topk_overlap_matrix(topk_indices)
    head_overlap = compute_snapkv_topk_head_overlap_matrices(topk_indices)
    summary["topk_indices_shape"] = list(topk_indices.shape)
    summary["overlap_shape"] = list(overlap.shape)
    summary["head_overlap_shape"] = list(head_overlap.shape)
    return SnapKVTopKOverlapResult(
        summary=summary,
        snapkv_topk_indices=topk_indices,
        snapkv_topk_overlap=overlap,
        snapkv_topk_head_overlap=head_overlap,
        snapkv_topk_layer_indices=np.asarray(topk_result["layer_indices"], dtype=np.int16),
    )


def save_snapkv_observation(
    result: SnapKVTopKOverlapResult,
    output_dir: str,
    prefix: str = "snapkv_observation",
) -> dict[str, Any]:
    return save_snapkv_topk_overlap_observation(
        result=result,
        output_dir=output_dir,
        prefix=prefix,
    )


def save_snapkv_topk_overlap_observation(
    result: SnapKVTopKOverlapResult,
    output_dir: str,
    prefix: str = "snapkv_topk_overlap",
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    npz_path = os.path.join(output_dir, f"{prefix}.npz")
    image_paths = plot_snapkv_topk_overlap_observation(result, output_dir, prefix=prefix)
    result.summary["image_files"] = {
        name: os.path.basename(path) for name, path in image_paths.items()
    }

    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(result.summary, fout, ensure_ascii=False, indent=2)

    np.savez_compressed(
        npz_path,
        snapkv_topk_indices=result.snapkv_topk_indices,
        snapkv_topk_overlap=result.snapkv_topk_overlap,
        snapkv_topk_head_overlap=result.snapkv_topk_head_overlap,
        snapkv_topk_layer_indices=result.snapkv_topk_layer_indices,
    )
    return {"summary": summary_path, "npz": npz_path, "images": image_paths}


def plot_snapkv_observation(
    result: SnapKVTopKOverlapResult,
    output_dir: str,
    prefix: str = "snapkv_observation",
) -> dict[str, str]:
    return plot_snapkv_topk_overlap_observation(
        result=result,
        output_dir=output_dir,
        prefix=prefix,
    )


def plot_snapkv_topk_overlap_observation(
    result: SnapKVTopKOverlapResult,
    output_dir: str,
    prefix: str = "snapkv_topk_overlap",
) -> dict[str, str]:
    if result.snapkv_topk_layer_indices.size == 0:
        return {}

    _setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    image_paths = {}
    if result.snapkv_topk_overlap.size != 0:
        image_paths["snapkv_topk_overlap_heatmap"] = os.path.join(
            output_dir,
            f"{prefix}_heatmap.png",
        )
        _plot_snapkv_topk_overlap_heatmap(
            result.snapkv_topk_overlap,
            result.snapkv_topk_layer_indices,
            image_paths["snapkv_topk_overlap_heatmap"],
            plt=plt,
        )
    if result.snapkv_topk_head_overlap.size != 0:
        for layer_pos, layer_idx in enumerate(result.snapkv_topk_layer_indices):
            image_key = f"snapkv_topk_head_overlap_layer_{int(layer_idx):03d}"
            image_paths[image_key] = os.path.join(
                output_dir,
                f"{prefix}_head_overlap_layer_{int(layer_idx):03d}.png",
            )
            _plot_snapkv_topk_head_overlap_heatmap(
                result.snapkv_topk_head_overlap[layer_pos],
                int(layer_idx),
                image_paths[image_key],
                plt=plt,
            )
    return image_paths


def _capture_layers(model, inputs, layers, window_size):
    captured = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            hidden_states = _get_arg_or_kwarg(args, kwargs, 0, "hidden_states")
            if hidden_states is None or not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
                return
            position_embeddings = _get_arg_or_kwarg(args, kwargs, 1, "position_embeddings")
            actual_window_size = min(int(window_size), int(hidden_states.shape[-2]))
            if actual_window_size < 1:
                return
            with torch.inference_mode():
                query_hidden_states = hidden_states[:, -actual_window_size:, :].detach()
                query_states = _project_query_window(module, query_hidden_states, position_embeddings)
                key_states = _project_key_states(module, hidden_states, position_embeddings)
            captured[layer_idx] = CapturedLayer(
                layer_idx=layer_idx,
                query_states=query_states.detach().to(dtype=torch.float32, device="cpu").clone(),
                key_states=key_states.detach().to(dtype=torch.float32, device="cpu").clone(),
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


def _compute_snapkv_topk_overlap(captured, config):
    result = {
        "summary": {
            "status": "pending",
            "valid_layer_indices": [],
            "layers": [],
            "overlap_metric": "mean_head_intersection_over_topk",
        },
        "indices": [],
        "layer_indices": [],
    }
    topk = config.budget - config.window_size
    for layer_idx in sorted(captured):
        capture = captured[layer_idx]
        layer_summary = {"layer_idx": int(layer_idx), "status": "pending"}
        kv_cache_len = int(capture.key_states.shape[-2])
        candidate_len = kv_cache_len - config.window_size
        layer_summary.update(
            {
                "kv_cache_len": kv_cache_len,
                "candidate_len": candidate_len,
                "topk": topk,
            }
        )
        if kv_cache_len < config.budget:
            layer_summary["status"] = "skipped_short_kv_cache"
            result["summary"]["layers"].append(layer_summary)
            continue
        if candidate_len < topk or candidate_len < 1:
            layer_summary["status"] = "skipped_insufficient_candidates"
            result["summary"]["layers"].append(layer_summary)
            continue

        try:
            attn_cache = compute_snapkv_attn_cache(
                capture.key_states,
                capture.query_states,
                window_size=config.window_size,
                kernel_size=config.kernel_size,
            )
            keep_count = min(topk, attn_cache.shape[-1])
            indices = attn_cache.topk(keep_count, dim=-1).indices
            result["indices"].append(_to_numpy(indices.squeeze(0), dtype=np.int64))
            result["layer_indices"].append(int(layer_idx))
            layer_summary.update(
                {
                    "status": "saved",
                    "attn_cache_shape": list(attn_cache.shape),
                    "indices_shape": list(indices.shape),
                }
            )
        except Exception as exc:
            layer_summary["status"] = "error"
            layer_summary["error"] = str(exc)
        result["summary"]["layers"].append(layer_summary)

    result["summary"]["valid_layer_indices"] = [int(layer_idx) for layer_idx in result["layer_indices"]]
    result["summary"]["status"] = "saved" if result["layer_indices"] else "no_valid_layers"
    return result


def compute_snapkv_topk_overlap_matrix(topk_indices):
    topk_indices = np.asarray(topk_indices)
    if topk_indices.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if topk_indices.ndim != 3:
        raise ValueError("topk_indices must have shape [layer, kv_head, topk].")

    layer_count, head_count, topk = topk_indices.shape
    overlap = np.zeros((layer_count, layer_count), dtype=np.float32)
    if layer_count == 0 or head_count == 0 or topk == 0:
        return overlap

    for src_layer in range(layer_count):
        for dst_layer in range(layer_count):
            head_scores = []
            for head_idx in range(head_count):
                src_indices = topk_indices[src_layer, head_idx]
                dst_indices = topk_indices[dst_layer, head_idx]
                intersection = np.intersect1d(src_indices, dst_indices, assume_unique=False).size
                head_scores.append(float(intersection) / float(topk))
            overlap[src_layer, dst_layer] = float(np.mean(head_scores))
    return overlap


def compute_snapkv_topk_head_overlap_matrices(topk_indices):
    topk_indices = np.asarray(topk_indices)
    if topk_indices.size == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)
    if topk_indices.ndim != 3:
        raise ValueError("topk_indices must have shape [layer, kv_head, topk].")

    layer_count, head_count, topk = topk_indices.shape
    overlap = np.zeros((layer_count, head_count, head_count), dtype=np.float32)
    if layer_count == 0 or head_count == 0 or topk == 0:
        return overlap

    for layer_idx in range(layer_count):
        for src_head in range(head_count):
            src_indices = topk_indices[layer_idx, src_head]
            for dst_head in range(head_count):
                dst_indices = topk_indices[layer_idx, dst_head]
                intersection = np.intersect1d(src_indices, dst_indices, assume_unique=False).size
                overlap[layer_idx, src_head, dst_head] = float(intersection) / float(topk)
    return overlap


def compute_snapkv_attn_cache(
    key_states: torch.Tensor,
    query_states: torch.Tensor,
    window_size: int = 8,
    kernel_size: int = 7,
) -> torch.Tensor:
    bsz, num_key_value_heads, kv_cache_len, _ = key_states.shape
    num_key_value_groups = query_states.shape[1] // num_key_value_heads
    query_window = min(int(window_size), int(query_states.shape[-2]))
    query_states = query_states[:, :, -query_window:, :]
    if kv_cache_len <= window_size:
        raise ValueError("kv_cache_len must be greater than window_size.")

    attn_weights = compute_attention_scores(query_states, key_states)
    attention_mask = torch.ones_like(attn_weights) * float("-inf")
    attention_mask = torch.triu(attention_mask, diagonal=kv_cache_len - query_window + 1)
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = attn_weights[..., :-window_size]

    hist_len = kv_cache_len - window_size
    scores = attn_weights.view(
        bsz,
        num_key_value_heads,
        num_key_value_groups,
        query_window,
        hist_len,
    )
    attn_weights_sum = scores.mean(dim=2).mean(dim=-2)
    return F.max_pool1d(
        attn_weights_sum,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    )


def _setup_matplotlib_cache():
    cache_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "snapkv_observation_matplotlib_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)


def _plot_snapkv_topk_overlap_heatmap(overlap, layer_indices, output_path, plt):
    overlap = np.asarray(overlap, dtype=np.float32)
    layer_indices = np.asarray(layer_indices)
    layer_count = overlap.shape[0]
    if layer_count == 0:
        return

    fig_size = max(5.0, 0.28 * layer_count + 3.0)
    tick_fontsize = max(4.0, min(10.0, 180.0 / max(layer_count, 1)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=180)
    image = ax.imshow(overlap, cmap="viridis", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_title("SnapKV topk index overlap by layer")
    ax.set_xlabel("Layer id")
    ax.set_ylabel("Layer id")

    tick_positions = np.arange(layer_count, dtype=np.int64)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(
        [str(int(layer_indices[idx])) for idx in tick_positions],
        rotation=90,
        ha="center",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(
        [str(int(layer_indices[idx])) for idx in tick_positions],
        fontsize=tick_fontsize,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean per-head overlap")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_snapkv_topk_head_overlap_heatmap(overlap, layer_idx, output_path, plt):
    overlap = np.asarray(overlap, dtype=np.float32)
    head_count = overlap.shape[0]
    if head_count == 0:
        return

    fig_size = max(4.0, 0.32 * head_count + 2.5)
    tick_fontsize = max(4.0, min(10.0, 160.0 / max(head_count, 1)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=180)
    image = ax.imshow(overlap, cmap="viridis", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_title(f"SnapKV topk head overlap, layer {int(layer_idx)}")
    ax.set_xlabel("Head id")
    ax.set_ylabel("Head id")

    tick_positions = np.arange(head_count, dtype=np.int64)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(
        [str(int(idx)) for idx in tick_positions],
        rotation=90,
        ha="center",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(
        [str(int(idx)) for idx in tick_positions],
        fontsize=tick_fontsize,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Topk overlap")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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


def _project_query_window(attention, hidden_states_window, position_embeddings):
    device = _module_device(attention)
    dtype = _module_dtype(attention)
    hidden_states_window = hidden_states_window.to(device=device, dtype=dtype)
    head_dim = _get_head_dim(attention)
    input_shape = hidden_states_window.shape[:-1]
    q_proj = attention.q_proj(hidden_states_window)
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


def _to_numpy(tensor, dtype):
    return tensor.detach().to(device="cpu").numpy().astype(dtype, copy=False)


def _stack_or_empty(items, dtype):
    if not items:
        return np.asarray([], dtype=dtype)
    return np.stack(items, axis=0).astype(dtype, copy=False)


def _empty_topk_overlap_result(summary):
    return SnapKVTopKOverlapResult(
        summary=summary,
        snapkv_topk_indices=np.asarray([], dtype=np.int64),
        snapkv_topk_overlap=np.zeros((0, 0), dtype=np.float32),
        snapkv_topk_head_overlap=np.zeros((0, 0, 0), dtype=np.float32),
        snapkv_topk_layer_indices=np.asarray([], dtype=np.int16),
    )
