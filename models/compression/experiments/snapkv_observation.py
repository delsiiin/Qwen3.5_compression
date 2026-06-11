import json
import math
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
    hidden_mix_profile_path: str | None = None

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
    attention: torch.nn.Module
    hidden_states: torch.Tensor
    attn_output: torch.Tensor | None
    query_states: torch.Tensor
    key_states: torch.Tensor


@dataclass
class SnapKVObservationResult:
    summary: dict[str, Any]
    hidden_mix_attn_cache: np.ndarray
    hidden_mix_cos_importance_pre_exp: np.ndarray
    hidden_mix_cos_importance: np.ndarray
    hidden_mix_mixed_cache: np.ndarray
    hidden_mix_indices: np.ndarray
    hidden_mix_layer_indices: np.ndarray


@dataclass
class SnapKVTopKOverlapResult:
    summary: dict[str, Any]
    snapkv_topk_indices: np.ndarray
    snapkv_topk_overlap: np.ndarray
    snapkv_topk_layer_indices: np.ndarray


def compute_snapkv_observation(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    config: SnapKVObservationConfig,
) -> SnapKVObservationResult:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("inputs must include input_ids.")
    if input_ids.shape[0] != 1:
        raise ValueError("SnapKV observation currently supports batch size 1 only.")

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
    if not layers:
        summary["status"] = "error"
        summary["reason"] = "No decoder layers were found."
        return _empty_result(summary)

    captured = _capture_layers(model, inputs, layers, config.window_size)
    hidden_mix_result = _compute_hidden_mix_observation(captured, config)
    summary.update(hidden_mix_result["summary"])
    return SnapKVObservationResult(
        summary=summary,
        hidden_mix_attn_cache=_stack_or_empty(hidden_mix_result["attn_cache"], np.float32),
        hidden_mix_cos_importance_pre_exp=_stack_or_empty(hidden_mix_result["cos_importance_pre_exp"], np.float32),
        hidden_mix_cos_importance=_stack_or_empty(hidden_mix_result["cos_importance"], np.float32),
        hidden_mix_mixed_cache=_stack_or_empty(hidden_mix_result["mixed_cache"], np.float32),
        hidden_mix_indices=_stack_or_empty(hidden_mix_result["indices"], np.int64),
        hidden_mix_layer_indices=np.asarray(hidden_mix_result["layer_indices"], dtype=np.int16),
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
    summary["topk_indices_shape"] = list(topk_indices.shape)
    summary["overlap_shape"] = list(overlap.shape)
    return SnapKVTopKOverlapResult(
        summary=summary,
        snapkv_topk_indices=topk_indices,
        snapkv_topk_overlap=overlap,
        snapkv_topk_layer_indices=np.asarray(topk_result["layer_indices"], dtype=np.int16),
    )


def save_snapkv_observation(
    result: SnapKVObservationResult,
    output_dir: str,
    prefix: str = "snapkv_observation",
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    npz_path = os.path.join(output_dir, f"{prefix}.npz")
    image_paths = plot_snapkv_observation(result, output_dir, prefix=prefix)
    result.summary["image_files"] = {
        name: os.path.basename(path) for name, path in image_paths.items()
    }

    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(result.summary, fout, ensure_ascii=False, indent=2)

    np.savez_compressed(
        npz_path,
        hidden_mix_attn_cache=result.hidden_mix_attn_cache,
        hidden_mix_cos_importance_pre_exp=result.hidden_mix_cos_importance_pre_exp,
        hidden_mix_cos_importance=result.hidden_mix_cos_importance,
        hidden_mix_mixed_cache=result.hidden_mix_mixed_cache,
        hidden_mix_indices=result.hidden_mix_indices,
        hidden_mix_layer_indices=result.hidden_mix_layer_indices,
    )
    return {"summary": summary_path, "npz": npz_path, "images": image_paths}


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
        snapkv_topk_layer_indices=result.snapkv_topk_layer_indices,
    )
    return {"summary": summary_path, "npz": npz_path, "images": image_paths}


def plot_snapkv_observation(
    result: SnapKVObservationResult,
    output_dir: str,
    prefix: str = "snapkv_observation",
) -> dict[str, str]:
    if result.hidden_mix_layer_indices.size == 0:
        return {}

    _setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    image_paths = {}

    image_paths["hidden_mix_attn_cache_by_token"] = os.path.join(
        output_dir,
        f"{prefix}_hidden_mix_attn_cache_by_token.png",
    )
    _plot_hidden_mix_by_token(
        result.hidden_mix_attn_cache,
        result.hidden_mix_layer_indices,
        image_paths["hidden_mix_attn_cache_by_token"],
        title="snapkv_hidden_mix attn_cache by token position",
        ylabel="attn_cache score",
        plt=plt,
    )

    image_paths["hidden_mix_cos_importance_pre_exp_distribution"] = os.path.join(
        output_dir,
        f"{prefix}_hidden_mix_cos_importance_pre_exp_distribution.png",
    )
    _plot_hidden_mix_distribution(
        result.hidden_mix_cos_importance_pre_exp,
        result.hidden_mix_layer_indices,
        image_paths["hidden_mix_cos_importance_pre_exp_distribution"],
        title="snapkv_hidden_mix cos_importance before exp distribution",
        xlabel="cosine before exp",
        plt=plt,
    )

    image_paths["hidden_mix_cos_importance_pre_exp_by_token"] = os.path.join(
        output_dir,
        f"{prefix}_hidden_mix_cos_importance_pre_exp_by_token.png",
    )
    _plot_hidden_mix_by_token(
        result.hidden_mix_cos_importance_pre_exp,
        result.hidden_mix_layer_indices,
        image_paths["hidden_mix_cos_importance_pre_exp_by_token"],
        title="snapkv_hidden_mix cos_importance before exp by token position",
        ylabel="cosine before exp",
        plt=plt,
    )

    image_paths["hidden_mix_cos_importance_distribution"] = os.path.join(
        output_dir,
        f"{prefix}_hidden_mix_cos_importance_distribution.png",
    )
    _plot_hidden_mix_distribution(
        result.hidden_mix_cos_importance,
        result.hidden_mix_layer_indices,
        image_paths["hidden_mix_cos_importance_distribution"],
        title="snapkv_hidden_mix cos_importance distribution",
        xlabel="cos_importance",
        plt=plt,
    )

    image_paths["hidden_mix_cos_importance_by_token"] = os.path.join(
        output_dir,
        f"{prefix}_hidden_mix_cos_importance_by_token.png",
    )
    _plot_hidden_mix_by_token(
        result.hidden_mix_cos_importance,
        result.hidden_mix_layer_indices,
        image_paths["hidden_mix_cos_importance_by_token"],
        title="snapkv_hidden_mix cos_importance by token position",
        ylabel="cos_importance",
        plt=plt,
    )

    image_paths["hidden_mix_mixed_cache_by_token"] = os.path.join(
        output_dir,
        f"{prefix}_hidden_mix_mixed_cache_by_token.png",
    )
    _plot_hidden_mix_by_token(
        result.hidden_mix_mixed_cache,
        result.hidden_mix_layer_indices,
        image_paths["hidden_mix_mixed_cache_by_token"],
        title="snapkv_hidden_mix mixed_cache by token position",
        ylabel="mixed_cache score",
        plt=plt,
    )

    return image_paths


def plot_snapkv_topk_overlap_observation(
    result: SnapKVTopKOverlapResult,
    output_dir: str,
    prefix: str = "snapkv_topk_overlap",
) -> dict[str, str]:
    if result.snapkv_topk_layer_indices.size == 0 or result.snapkv_topk_overlap.size == 0:
        return {}

    _setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    image_paths = {
        "snapkv_topk_overlap_heatmap": os.path.join(
            output_dir,
            f"{prefix}_heatmap.png",
        )
    }
    _plot_snapkv_topk_overlap_heatmap(
        result.snapkv_topk_overlap,
        result.snapkv_topk_layer_indices,
        image_paths["snapkv_topk_overlap_heatmap"],
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
                attn_output = _extract_attention_output(output)
            captured[layer_idx] = CapturedLayer(
                layer_idx=layer_idx,
                attention=module,
                hidden_states=hidden_states.detach().to(dtype=torch.float32, device="cpu").clone(),
                attn_output=(
                    attn_output.detach().to(dtype=torch.float32, device="cpu").clone()
                    if torch.is_tensor(attn_output)
                    else None
                ),
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


def _compute_hidden_mix_observation(captured, config):
    result = {
        "summary": {
            "status": "pending",
            "profile_path": config.hidden_mix_profile_path,
            "valid_layer_indices": [],
            "layers": [],
        },
        "attn_cache": [],
        "cos_importance_pre_exp": [],
        "cos_importance": [],
        "mixed_cache": [],
        "indices": [],
        "layer_indices": [],
    }
    try:
        profile = _load_hidden_mix_profile(config.hidden_mix_profile_path)
    except Exception as exc:
        result["summary"]["status"] = "error"
        result["summary"]["error"] = str(exc)
        return result

    if profile.get("path") is not None:
        result["summary"]["profile_path"] = profile["path"]
        result["summary"]["profile_metric"] = profile.get("metric")

    entries = {}
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
            attn_cache = compute_snapkv_hidden_mix_attn_cache(
                capture.key_states,
                capture.query_states,
                window_size=config.window_size,
                kernel_size=config.kernel_size,
            )
            cos_importance_pre_exp, cos_importance = compute_snapkv_hidden_mix_cos_importance(
                attention=capture.attention,
                hidden_states=capture.hidden_states,
                attn_output=capture.attn_output,
                num_key_value_heads=attn_cache.shape[1],
                hist_len=attn_cache.shape[-1],
                return_pre_exp=True,
            )
            cos_importance_pre_exp = cos_importance_pre_exp.to(device=attn_cache.device, dtype=attn_cache.dtype)
            cos_importance = cos_importance.to(device=attn_cache.device, dtype=attn_cache.dtype)
            if cos_importance.shape != attn_cache.shape:
                raise ValueError("snapkv_hidden_mix cosine importance must match attn_cache shape.")
            if cos_importance_pre_exp.shape != attn_cache.shape:
                raise ValueError("snapkv_hidden_mix pre-exp cosine importance must match attn_cache shape.")
            entries[int(layer_idx)] = {
                "attn_cache": attn_cache,
                "cos_importance_pre_exp": cos_importance_pre_exp,
                "cos_importance": cos_importance,
            }
            layer_summary["status"] = "cached"
        except Exception as exc:
            layer_summary["status"] = "error"
            layer_summary["error"] = str(exc)
        result["summary"]["layers"].append(layer_summary)

    for layer_summary in result["summary"]["layers"]:
        layer_idx = int(layer_summary["layer_idx"])
        if layer_summary.get("status") != "cached":
            continue
        try:
            mix = _hidden_mix_for_layer(profile, layer_idx)
            mixed_cache = compute_snapkv_hidden_mix_mixed_cache(entries, layer_idx, mix)
            keep_count = min(topk, mixed_cache.shape[-1])
            indices = mixed_cache.topk(keep_count, dim=-1).indices
            entry = entries[layer_idx]

            result["attn_cache"].append(_to_numpy(entry["attn_cache"].squeeze(0), dtype=np.float32))
            result["cos_importance_pre_exp"].append(
                _to_numpy(entry["cos_importance_pre_exp"].squeeze(0), dtype=np.float32)
            )
            result["cos_importance"].append(_to_numpy(entry["cos_importance"].squeeze(0), dtype=np.float32))
            result["mixed_cache"].append(_to_numpy(mixed_cache.squeeze(0), dtype=np.float32))
            result["indices"].append(_to_numpy(indices.squeeze(0), dtype=np.int64))
            result["layer_indices"].append(layer_idx)

            layer_summary.update(
                {
                    "status": "saved",
                    "mix": _mix_summary(mix),
                    "attn_cache_shape": list(entry["attn_cache"].shape),
                    "cos_importance_pre_exp_shape": list(entry["cos_importance_pre_exp"].shape),
                    "cos_importance_shape": list(entry["cos_importance"].shape),
                    "mixed_cache_shape": list(mixed_cache.shape),
                    "indices_shape": list(indices.shape),
                    "attn_cache": _distribution_stats_by_head(entry["attn_cache"]),
                    "cos_importance_pre_exp": _distribution_stats_by_head(entry["cos_importance_pre_exp"]),
                    "cos_importance": _distribution_stats_by_head(entry["cos_importance"]),
                    "mixed_cache": _distribution_stats_by_head(mixed_cache),
                }
            )
        except Exception as exc:
            layer_summary["status"] = "error"
            layer_summary["error"] = str(exc)

    result["summary"]["valid_layer_indices"] = [int(layer_idx) for layer_idx in result["layer_indices"]]
    result["summary"]["status"] = "saved" if result["layer_indices"] else "no_valid_layers"
    return result


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
            attn_cache = compute_snapkv_hidden_mix_attn_cache(
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


def compute_snapkv_hidden_mix_attn_cache(
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


def compute_snapkv_hidden_mix_cos_importance(
    attention,
    hidden_states,
    attn_output,
    num_key_value_heads,
    hist_len,
    importance_epsilon=1e-12,
    return_pre_exp=False,
):
    def maybe_return(pre_exp, importance):
        if return_pre_exp:
            return pre_exp, importance
        return importance

    if (
        hidden_states is None
        or attn_output is None
        or not torch.is_tensor(hidden_states)
        or not torch.is_tensor(attn_output)
        or hidden_states.ndim != 3
        or attn_output.ndim != 3
        or hist_len < 1
    ):
        if torch.is_tensor(hidden_states):
            device = hidden_states.device
        elif torch.is_tensor(attn_output):
            device = attn_output.device
        else:
            device = torch.device("cpu")
        importance = torch.ones(1, int(num_key_value_heads), int(hist_len), device=device)
        pre_exp = torch.zeros_like(importance)
        return maybe_return(pre_exp, importance)

    batch_size = hidden_states.shape[0]
    actual_len = min(int(hidden_states.shape[1]), int(attn_output.shape[1]), int(hist_len))
    device = hidden_states.device
    pre_exp = torch.zeros(
        batch_size,
        int(num_key_value_heads),
        int(hist_len),
        dtype=torch.float32,
        device=device,
    )
    importance = torch.ones(
        batch_size,
        int(num_key_value_heads),
        int(hist_len),
        dtype=torch.float32,
        device=device,
    )
    if actual_len < 1:
        return maybe_return(pre_exp, importance)

    input_vectors = hidden_states[:, :actual_len, :].detach().to(dtype=torch.float32)
    output_vectors = attn_output[:, :actual_len, :].detach().to(device=device, dtype=torch.float32)
    cosine = _token_cosine(input_vectors, output_vectors, importance_epsilon)

    num_attention_heads = getattr(getattr(attention, "config", None), "num_attention_heads", None)
    head_dim = getattr(attention, "head_dim", None)
    hidden_size = input_vectors.shape[-1]
    if (
        num_attention_heads is not None
        and head_dim is not None
        and int(num_attention_heads) > 0
        and int(num_key_value_heads) > 0
        and int(num_attention_heads) % int(num_key_value_heads) == 0
        and hidden_size == int(num_attention_heads) * int(head_dim)
        and output_vectors.shape[-1] == hidden_size
    ):
        num_attention_heads = int(num_attention_heads)
        head_dim = int(head_dim)
        num_key_value_groups = num_attention_heads // int(num_key_value_heads)
        head_input = input_vectors.view(batch_size, actual_len, num_attention_heads, head_dim)
        head_output = output_vectors.view(batch_size, actual_len, num_attention_heads, head_dim)
        head_cosine = _token_cosine(head_input, head_output, importance_epsilon)
        cosine = head_cosine.transpose(1, 2).reshape(
            batch_size,
            int(num_key_value_heads),
            num_key_value_groups,
            actual_len,
        ).mean(dim=2)
    else:
        cosine = cosine[:, None, :].expand(batch_size, int(num_key_value_heads), actual_len)

    cosine = cosine.clamp(min=-1.0, max=1.0)
    pre_exp[:, :, :actual_len] = cosine
    importance[:, :, :actual_len] = torch.exp(cosine)
    return maybe_return(pre_exp, importance)


def compute_snapkv_hidden_mix_mixed_cache(entries, target_layer, mix):
    entry = entries[int(target_layer)]
    mixed_cache = None
    for source_layer, weight in zip(mix["sources"], mix["weights"]):
        source_entry = entries.get(int(source_layer))
        if source_entry is None:
            raise ValueError(f"snapkv_hidden_mix profile source layer {source_layer} was not captured.")
        source_cache = source_entry["attn_cache"].to(
            device=entry["attn_cache"].device,
            dtype=entry["attn_cache"].dtype,
        )
        if source_cache.shape != entry["attn_cache"].shape:
            raise ValueError("snapkv_hidden_mix requires matching attn_cache shapes within a layer group.")
        normalized_cache = _normalize_hidden_mix_attn_cache(source_cache)
        weighted = normalized_cache * float(weight)
        mixed_cache = weighted if mixed_cache is None else mixed_cache + weighted

    importance = entry["cos_importance"].to(device=mixed_cache.device, dtype=mixed_cache.dtype)
    if importance.shape != mixed_cache.shape:
        raise ValueError("snapkv_hidden_mix cosine importance must match mixed attn_cache shape.")
    return mixed_cache * importance


def _token_cosine(input_vectors, output_vectors, importance_epsilon):
    dot = (input_vectors * output_vectors).sum(dim=-1)
    input_norm = input_vectors.square().sum(dim=-1).sqrt()
    output_norm = output_vectors.square().sum(dim=-1).sqrt()
    denom = (input_norm * output_norm).clamp_min(float(importance_epsilon))
    return (dot / denom).clamp(min=-1.0, max=1.0)


def _normalize_hidden_mix_attn_cache(attn_cache):
    rank_cache = attn_cache.to(dtype=torch.float32)
    mean = rank_cache.mean(dim=-1, keepdim=True)
    centered = rank_cache - mean
    std = centered.square().mean(dim=-1, keepdim=True).sqrt()
    std = std.clamp_min(torch.finfo(rank_cache.dtype).eps)
    return (centered / std).to(dtype=attn_cache.dtype)


def _load_hidden_mix_profile(profile_path):
    if profile_path is None:
        return {"metric": None, "groups": {}, "layer_to_group": {}, "path": None}

    profile_path = os.path.abspath(os.path.expanduser(str(profile_path)))
    with open(profile_path, "r", encoding="utf-8") as handle:
        raw_profile = json.load(handle)
    if not isinstance(raw_profile, dict):
        raise ValueError(f"Hidden mix profile must be a JSON object: {profile_path}")
    groups = raw_profile.get("groups")
    if not isinstance(groups, list):
        raise ValueError("Hidden mix profile requires a list field named 'groups'.")

    layer_to_group = {}
    group_profiles = {}
    for group_idx, group_spec in enumerate(groups):
        if not isinstance(group_spec, dict):
            raise ValueError(f"Profile group {group_idx} must be an object.")
        layers = group_spec.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError(f"Profile group {group_idx} requires non-empty 'layers'.")
        layers = tuple(int(layer) for layer in layers)
        if len(set(layers)) != len(layers):
            raise ValueError(f"Profile group {group_idx} contains duplicate layers.")
        for layer in layers:
            if layer in layer_to_group:
                raise ValueError(f"Layer {layer} appears in multiple hidden mix groups.")
            layer_to_group[layer] = layers

        raw_mix = group_spec.get("mix", {})
        if raw_mix is None:
            raw_mix = {}
        if not isinstance(raw_mix, dict):
            raise ValueError(f"Profile group {group_idx} mix must be an object.")
        mix = {}
        for target_text, mix_spec in raw_mix.items():
            target_layer = int(target_text)
            if target_layer not in layers:
                raise ValueError(f"Profile group {group_idx} has mix target outside layers: {target_layer}.")
            if not isinstance(mix_spec, dict):
                raise ValueError(f"Profile group {group_idx} mix for layer {target_layer} must be an object.")
            sources = mix_spec.get("sources")
            weights = mix_spec.get("weights")
            if not isinstance(sources, list) or not isinstance(weights, list) or len(sources) != len(weights):
                raise ValueError(f"Profile group {group_idx} mix for layer {target_layer} needs equal sources/weights lists.")
            if not sources:
                raise ValueError(f"Profile group {group_idx} mix for layer {target_layer} must not be empty.")
            parsed_sources = tuple(int(source) for source in sources)
            parsed_weights = tuple(float(weight) for weight in weights)
            if any(source not in layers for source in parsed_sources):
                raise ValueError(f"Profile group {group_idx} mix source must stay within its group.")
            weight_sum = float(sum(parsed_weights))
            if (not math.isfinite(weight_sum)) or weight_sum <= 0.0:
                raise ValueError(f"Profile group {group_idx} mix weights must have a positive finite sum.")
            if any((not math.isfinite(weight)) or weight < 0.0 for weight in parsed_weights):
                raise ValueError(f"Profile group {group_idx} mix weights must be finite and non-negative.")
            mix[target_layer] = {
                "sources": parsed_sources,
                "weights": tuple(weight / weight_sum for weight in parsed_weights),
            }
        group_profiles[layers] = {"mix": mix}

    return {
        "metric": raw_profile.get("metric"),
        "groups": group_profiles,
        "layer_to_group": layer_to_group,
        "path": profile_path,
    }


def _hidden_mix_for_layer(profile, target_layer):
    target_layer = int(target_layer)
    group_layers = profile["layer_to_group"].get(target_layer, (target_layer,))
    group = profile["groups"].get(tuple(group_layers), {})
    mix = group.get("mix", {}).get(target_layer)
    if mix is None:
        return {
            "sources": (target_layer,),
            "weights": (1.0,),
            "group_layers": tuple(group_layers),
            "fallback_to_self": True,
        }
    return {
        "sources": tuple(int(source) for source in mix["sources"]),
        "weights": tuple(float(weight) for weight in mix["weights"]),
        "group_layers": tuple(group_layers),
        "fallback_to_self": False,
    }


def _mix_summary(mix):
    return {
        "sources": [int(source) for source in mix["sources"]],
        "weights": [float(weight) for weight in mix["weights"]],
        "group_layers": [int(layer_idx) for layer_idx in mix["group_layers"]],
        "fallback_to_self": bool(mix["fallback_to_self"]),
    }


def _extract_attention_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    return None


def _setup_matplotlib_cache():
    cache_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "snapkv_observation_matplotlib_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)


def _plot_hidden_mix_distribution(values, layer_indices, output_path, title, xlabel, plt):
    values = np.asarray(values, dtype=np.float32)
    layer_indices = np.asarray(layer_indices)
    layer_count = values.shape[0]
    if layer_count == 0:
        return

    ncols = 2 if layer_count <= 12 else 3
    nrows = int(np.ceil(layer_count / ncols))
    fig_width = max(9.0, min(24.0, ncols * 6.2))
    fig_height = max(4.0, min(28.0, nrows * 3.2))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        dpi=160,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    for layer_pos, ax in enumerate(axes_flat):
        if layer_pos >= layer_count:
            ax.axis("off")
            continue
        layer_values = values[layer_pos].reshape(-1)
        layer_idx = int(layer_indices[layer_pos])
        if layer_values.size == 0:
            ax.axis("off")
            continue
        lower = float(np.min(layer_values))
        upper = float(np.max(layer_values))
        if lower == upper:
            ax.axvline(lower, color="#3b82f6", linewidth=1.8)
            ax.set_ylim(0.0, 1.0)
        else:
            ax.hist(layer_values, bins=60, density=True, color="#3b82f6", alpha=0.78)
        ax.set_title(f"Layer {layer_idx}")
        ax.grid(alpha=0.22, linewidth=0.8)
        if layer_pos % ncols == 0:
            ax.set_ylabel("Density")
        if layer_pos >= (nrows - 1) * ncols:
            ax.set_xlabel(xlabel)

    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.18)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_hidden_mix_by_token(values, layer_indices, output_path, title, ylabel, plt, max_points=4096):
    values = np.asarray(values, dtype=np.float32)
    layer_indices = np.asarray(layer_indices)
    layer_count = values.shape[0]
    if layer_count == 0:
        return

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
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    for layer_pos, ax in enumerate(axes_flat):
        if layer_pos >= layer_count:
            ax.axis("off")
            continue
        layer_values = values[layer_pos]
        layer_idx = int(layer_indices[layer_pos])
        if layer_values.size == 0 or layer_values.ndim != 2:
            ax.axis("off")
            continue

        token_positions = np.arange(layer_values.shape[-1])
        mean = layer_values.mean(axis=0)
        q25 = np.quantile(layer_values, 0.25, axis=0)
        q75 = np.quantile(layer_values, 0.75, axis=0)
        lower = layer_values.min(axis=0)
        upper = layer_values.max(axis=0)
        keep = _plot_sample_indices(layer_values.shape[-1], max_points=max_points)

        x = token_positions[keep]
        ax.fill_between(x, lower[keep], upper[keep], color="#93c5fd", alpha=0.16, linewidth=0.0, label="min-max")
        ax.fill_between(x, q25[keep], q75[keep], color="#60a5fa", alpha=0.28, linewidth=0.0, label="25-75%")
        ax.plot(x, mean[keep], color="#1d4ed8", linewidth=1.2, label="mean")
        ax.set_title(f"Layer {layer_idx}")
        ax.grid(alpha=0.22, linewidth=0.8)
        if layer_pos % ncols == 0:
            ax.set_ylabel(ylabel)
        if layer_pos >= (nrows - 1) * ncols:
            ax.set_xlabel("Historical token index")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.18)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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

def _plot_sample_indices(width_count, max_points=4096):
    width_count = int(width_count)
    max_points = int(max_points)
    if width_count <= 0:
        return np.asarray([], dtype=np.int64)
    if max_points < 1 or width_count <= max_points:
        return np.arange(width_count, dtype=np.int64)
    return np.unique(np.linspace(0, width_count - 1, num=max_points, dtype=np.int64))


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


def _distribution_stats_by_head(values):
    array = _to_numpy(values, dtype=np.float32) if torch.is_tensor(values) else np.asarray(values, dtype=np.float32)
    summary = _distribution_stats(array)
    if array.ndim >= 3:
        per_head = []
        for head_idx in range(array.shape[1]):
            per_head.append(_distribution_stats(array[:, head_idx, ...]))
        summary["per_head"] = per_head
    return summary


def _to_numpy(tensor, dtype):
    return tensor.detach().to(device="cpu").numpy().astype(dtype, copy=False)


def _stack_or_empty(items, dtype):
    if not items:
        return np.asarray([], dtype=dtype)
    return np.stack(items, axis=0).astype(dtype, copy=False)


def _empty_result(summary):
    return SnapKVObservationResult(
        summary=summary,
        hidden_mix_attn_cache=np.asarray([], dtype=np.float32),
        hidden_mix_cos_importance_pre_exp=np.asarray([], dtype=np.float32),
        hidden_mix_cos_importance=np.asarray([], dtype=np.float32),
        hidden_mix_mixed_cache=np.asarray([], dtype=np.float32),
        hidden_mix_indices=np.asarray([], dtype=np.int64),
        hidden_mix_layer_indices=np.asarray([], dtype=np.int16),
    )


def _empty_topk_overlap_result(summary):
    return SnapKVTopKOverlapResult(
        summary=summary,
        snapkv_topk_indices=np.asarray([], dtype=np.int64),
        snapkv_topk_overlap=np.zeros((0, 0), dtype=np.float32),
        snapkv_topk_layer_indices=np.asarray([], dtype=np.int16),
    )
