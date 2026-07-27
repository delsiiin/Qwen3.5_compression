import json
import os
import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


SUPPORTED_HEAD_CLUSTER_OBSERVATION_METHODS = {
    "snapkv_ada",
    "tridentkv_spatial_temporal_ada",
    "tridentkv_head_cluster",
    "tridentkv",
}
HEAD_BUDGET_ATTENTION_SUBMODE = "head_budget_attention"
HEAD_CLUSTER_PCA_SUBMODE = "head_cluster_pca"
HEAD_CLUSTER_TOKEN_DISTRIBUTION_SUBMODE = "head_cluster_token_distribution"
TOKEN_SPATIAL_TEMPORAL_HEATMAP_SUBMODE = "token_spatial_temporal_heatmap"
HEAD_CLUSTER_PCA_METHODS = {
    "tridentkv_head_cluster",
    "tridentkv",
}
TOKEN_SPATIAL_TEMPORAL_HEATMAP_METHODS = {"tridentkv"}
TOKEN_HEATMAP_SELECTION_REASONS = (
    "global_sustained_response",
    "temporal_local_response",
    "cluster_specific_response",
    "spatial_temporal_local_response",
    "isolated_peak_response",
)
SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES = {
    HEAD_BUDGET_ATTENTION_SUBMODE,
    HEAD_CLUSTER_PCA_SUBMODE,
    HEAD_CLUSTER_TOKEN_DISTRIBUTION_SUBMODE,
    TOKEN_SPATIAL_TEMPORAL_HEATMAP_SUBMODE,
}


@dataclass(frozen=True)
class HeadClusterObservationConfig:
    submode: str = HEAD_BUDGET_ATTENTION_SUBMODE
    max_prefill_tokens: int | None = None
    token_count: int = 8

    def __post_init__(self):
        if self.submode not in SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES:
            raise ValueError(
                f"submode must be one of {sorted(SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES)}."
            )
        if self.max_prefill_tokens is not None and self.max_prefill_tokens < 1:
            raise ValueError("max_prefill_tokens must be at least 1 when provided.")
        if self.token_count < 1:
            raise ValueError("token_count must be at least 1.")


@dataclass
class HeadClusterObservationResult:
    summary: dict[str, Any]
    layer_indices: np.ndarray
    raw_attention: np.ndarray
    selected_hist_mask: np.ndarray
    history_selected_token_counts: np.ndarray
    recent_retained_token_counts: np.ndarray
    post_compression_token_counts: np.ndarray
    pre_compression_kv_cache_lengths: np.ndarray
    history_lengths: np.ndarray
    group_similarity: np.ndarray | None = None
    kv_head_cluster_ids: np.ndarray | None = None
    kv_head_pca_points: np.ndarray | None = None
    pca_explained_variance_ratio: np.ndarray | None = None
    query_window_lengths: np.ndarray | None = None
    cluster_counts: np.ndarray | None = None
    cluster_selected_token_counts: np.ndarray | None = None
    cluster_selected_token_ratios: np.ndarray | None = None
    selected_token_indices: np.ndarray | None = None
    selected_token_ids: np.ndarray | None = None
    selected_token_reason_ids: np.ndarray | None = None
    selected_token_metric_scores: np.ndarray | None = None
    token_heatmaps: np.ndarray | None = None
    heatmap_row_cluster_ids: np.ndarray | None = None
    heatmap_row_kv_head_indices: np.ndarray | None = None
    heatmap_row_gqa_group_indices: np.ndarray | None = None


def compute_head_cluster_observation(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    config: HeadClusterObservationConfig,
) -> HeadClusterObservationResult:
    """Run one isolated cached prefill through the real compression path."""
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("inputs must include input_ids.")
    if input_ids.shape[0] != 1:
        raise ValueError("Head-cluster observation currently supports batch size 1 only.")

    token_count = int(input_ids.shape[-1])
    summary = {
        "status": "pending",
        "config": asdict(config),
        "capture_scope": "isolated_cached_prefill",
        "token_count": token_count,
        "compression_method": None,
        "compression_config": {},
        "valid_layer_indices": [],
        "layers": [],
    }
    if config.max_prefill_tokens is not None and token_count > config.max_prefill_tokens:
        summary["status"] = "skipped_over_cap"
        summary["reason"] = (
            f"Prompt token count {token_count} exceeds cap {config.max_prefill_tokens}."
        )
        return _empty_result(summary)

    attentions = _get_compression_attentions(model)
    if not attentions:
        summary["status"] = "error"
        summary["reason"] = "No decoder attention modules with a compression cluster were found."
        return _empty_result(summary)

    methods = {str(getattr(attention.config, "method", "")) for _, attention in attentions}
    if len(methods) != 1:
        summary["status"] = "error"
        summary["reason"] = f"Expected one compression method across layers, got {sorted(methods)}."
        return _empty_result(summary)
    method = methods.pop()
    summary["compression_method"] = method
    if method not in SUPPORTED_HEAD_CLUSTER_OBSERVATION_METHODS:
        summary["status"] = "error"
        summary["reason"] = (
            f"Unsupported compression method {method!r}; expected one of "
            f"{sorted(SUPPORTED_HEAD_CLUSTER_OBSERVATION_METHODS)}."
        )
        return _empty_result(summary)
    if config.submode == HEAD_CLUSTER_PCA_SUBMODE and method not in HEAD_CLUSTER_PCA_METHODS:
        summary["status"] = "error"
        summary["reason"] = (
            f"Submode {HEAD_CLUSTER_PCA_SUBMODE!r} requires one of "
            f"{sorted(HEAD_CLUSTER_PCA_METHODS)}, got {method!r}."
        )
        return _empty_result(summary)
    if (
        config.submode == TOKEN_SPATIAL_TEMPORAL_HEATMAP_SUBMODE
        and method not in TOKEN_SPATIAL_TEMPORAL_HEATMAP_METHODS
    ):
        summary["status"] = "error"
        summary["reason"] = (
            f"Submode {TOKEN_SPATIAL_TEMPORAL_HEATMAP_SUBMODE!r} requires one of "
            f"{sorted(TOKEN_SPATIAL_TEMPORAL_HEATMAP_METHODS)}, got {method!r}."
        )
        return _empty_result(summary)

    first_cluster = attentions[0][1].kv_cluster
    summary["compression_config"] = {
        "budget": int(first_cluster.budget),
        "window_size": int(first_cluster.window_size),
        "kernel_size": int(first_cluster.kernel_size),
    }

    # This is the only place that enables method-level observation. The flag is
    # always disabled again before returning, including on failed forwards.
    for _, attention in attentions:
        cluster = attention.kv_cluster
        if not hasattr(cluster, "enable_head_cluster_observation"):
            summary["status"] = "error"
            summary["reason"] = "Compression cluster does not support observation records."
            return _empty_result(summary)
    model_state = _snapshot_observation_state(model, attentions)
    if method == "tridentkv":
        coordinated_layer_indices = tuple(int(layer_idx) for layer_idx, _ in attentions)
        for _, attention in attentions:
            attention.kv_cluster.prefill_layer_budget_layers = coordinated_layer_indices
        summary["compression_config"]["coordinated_layer_indices"] = list(
            coordinated_layer_indices
        )
    for _, attention in attentions:
        cluster = attention.kv_cluster
        cluster.enable_head_cluster_observation(
            config.submode,
            token_count=config.token_count,
        )
    try:
        _run_cached_observation_forward(model, inputs)
    except Exception as exc:
        for _, attention in attentions:
            attention.kv_cluster.clear_head_cluster_observation_records()
        summary["status"] = "error"
        summary["error"] = str(exc)
        return _empty_result(summary)
    finally:
        for _, attention in attentions:
            attention.kv_cluster.disable_head_cluster_observation()
        _restore_observation_state(model_state)

    records_by_layer = {}
    for layer_idx, attention in attentions:
        cluster = attention.kv_cluster
        records = cluster.get_head_cluster_observation_records()
        cluster.clear_head_cluster_observation_records()
        if len(records) > 1:
            summary["status"] = "error"
            summary["reason"] = (
                f"Layer {layer_idx} produced {len(records)} records in one observation prefill."
            )
            return _empty_result(summary)
        if records:
            record = records[0]
            if int(record["layer_idx"]) != int(layer_idx):
                summary["status"] = "error"
                summary["reason"] = (
                    f"Record layer {record['layer_idx']} was stored on layer {layer_idx}."
                )
                return _empty_result(summary)
            records_by_layer[int(layer_idx)] = record

    collected = []
    for layer_idx, _ in attentions:
        record = records_by_layer.get(int(layer_idx))
        layer_summary = {"layer_idx": int(layer_idx)}
        if record is None:
            layer_summary["status"] = "skipped_no_compression"
        else:
            if config.submode == TOKEN_SPATIAL_TEMPORAL_HEATMAP_SUBMODE:
                cluster_ids = record["kv_head_cluster_ids"]
                selected_indices = record["selected_token_indices"]
                selected_ids = input_ids[0].detach().to(device="cpu").index_select(
                    0,
                    selected_indices.to(dtype=torch.long),
                )
                selected_tokens = []
                for selected_pos, token_idx in enumerate(selected_indices.tolist()):
                    reason_id = int(
                        record["selected_token_reason_ids"][selected_pos].item()
                    )
                    metric_values = record["selected_token_metric_scores"][
                        selected_pos
                    ]
                    selected_tokens.append(
                        {
                            "index": int(token_idx),
                            "id": int(selected_ids[selected_pos].item()),
                            "selection_reason": TOKEN_HEATMAP_SELECTION_REASONS[
                                reason_id
                            ],
                            "selection_scores": {
                                metric_name: float(metric_values[metric_idx].item())
                                for metric_idx, metric_name in enumerate(
                                    TOKEN_HEATMAP_SELECTION_REASONS
                                )
                            },
                        }
                    )
                layer_summary.update(
                    {
                        "status": "saved",
                        "pre_compression_kv_cache_len": record[
                            "pre_compression_kv_cache_len"
                        ],
                        "history_len": record["history_len"],
                        "query_window_len": record["query_window_len"],
                        "kv_head_count": int(cluster_ids.numel()),
                        "gqa_group_count": int(
                            record["token_heatmaps"].shape[1]
                            // cluster_ids.numel()
                        ),
                        "cluster_count": int(record["cluster_count"]),
                        "selected_token_count": int(selected_indices.numel()),
                        "selected_tokens": selected_tokens,
                    }
                )
            elif config.submode == HEAD_CLUSTER_TOKEN_DISTRIBUTION_SUBMODE:
                cluster_ids = record["kv_head_cluster_ids"]
                history_counts = record["history_selected_token_counts"]
                cluster_count = int(record["cluster_count"])
                clusters = []
                for cluster_id in range(cluster_count):
                    heads = torch.where(cluster_ids == cluster_id)[0]
                    clusters.append(
                        {
                            "cluster_id": cluster_id,
                            "heads": heads.to(dtype=torch.int64).tolist(),
                            "selected_history_token_slots": int(
                                record["cluster_selected_token_counts"][
                                    cluster_id
                                ].item()
                            ),
                            "ratio": float(
                                record["cluster_selected_token_ratios"][
                                    cluster_id
                                ].item()
                            ),
                        }
                    )
                layer_summary.update(
                    {
                        "status": "saved",
                        "pre_compression_kv_cache_len": record[
                            "pre_compression_kv_cache_len"
                        ],
                        "history_len": record["history_len"],
                        "query_window_len": record["query_window_len"],
                        "kv_head_count": int(history_counts.numel()),
                        "cluster_count": cluster_count,
                        "selected_history_token_slot_total": int(
                            record[
                                "cluster_selected_token_count_total"
                            ]
                        ),
                        "has_selected_history_token_slots": bool(
                            record[
                                "cluster_selected_token_count_total"
                            ]
                            > 0
                        ),
                        "clusters": clusters,
                    }
                )
            elif config.submode == HEAD_CLUSTER_PCA_SUBMODE:
                kv_head_count = int(record["group_similarity"].shape[0])
                layer_summary.update(
                    {
                        "status": "saved",
                        "pre_compression_kv_cache_len": record[
                            "pre_compression_kv_cache_len"
                        ],
                        "history_len": record["history_len"],
                        "query_window_len": record["query_window_len"],
                        "kv_head_count": kv_head_count,
                        "cluster_count": int(
                            torch.unique(record["kv_head_cluster_ids"]).numel()
                        ),
                    }
                )
            else:
                layer_summary.update(
                    {
                        "status": "saved",
                        "pre_compression_kv_cache_len": record[
                            "pre_compression_kv_cache_len"
                        ],
                        "history_len": record["history_len"],
                        "kv_head_count": int(
                            record["post_compression_token_counts"].numel()
                        ),
                    }
                )
            collected.append(record)
        summary["layers"].append(layer_summary)

    if not collected:
        summary["status"] = "no_valid_layers"
        return _empty_result(summary)

    if config.submode == HEAD_CLUSTER_PCA_SUBMODE:
        return _build_head_cluster_pca_result(summary, collected)
    if config.submode == HEAD_CLUSTER_TOKEN_DISTRIBUTION_SUBMODE:
        return _build_head_cluster_token_distribution_result(summary, collected)
    if config.submode == TOKEN_SPATIAL_TEMPORAL_HEATMAP_SUBMODE:
        return _build_token_spatial_temporal_heatmap_result(
            summary,
            collected,
            input_ids[0],
        )

    try:
        layer_indices = np.asarray(
            [int(record["layer_idx"]) for record in collected],
            dtype=np.int16,
        )
        raw_attention = _stack_records(collected, "raw_attention", np.float32)
        selected_hist_mask = _stack_records(collected, "selected_hist_mask", np.bool_)
        history_counts = _stack_records(
            collected,
            "history_selected_token_counts",
            np.int32,
        )
        recent_counts = _stack_records(
            collected,
            "recent_retained_token_counts",
            np.int32,
        )
        post_counts = _stack_records(
            collected,
            "post_compression_token_counts",
            np.int32,
        )
    except ValueError as exc:
        summary["status"] = "error"
        summary["reason"] = f"Incompatible observation shapes across layers: {exc}"
        return _empty_result(summary)

    pre_lengths = np.asarray(
        [int(record["pre_compression_kv_cache_len"]) for record in collected],
        dtype=np.int32,
    )
    history_lengths = np.asarray(
        [int(record["history_len"]) for record in collected],
        dtype=np.int32,
    )
    summary.update(
        {
            "status": "saved",
            "valid_layer_indices": layer_indices.astype(int).tolist(),
            "raw_attention_shape": list(raw_attention.shape),
            "selected_hist_mask_shape": list(selected_hist_mask.shape),
            "post_compression_token_counts_shape": list(post_counts.shape),
        }
    )
    return HeadClusterObservationResult(
        summary=summary,
        layer_indices=layer_indices,
        raw_attention=raw_attention,
        selected_hist_mask=selected_hist_mask,
        history_selected_token_counts=history_counts,
        recent_retained_token_counts=recent_counts,
        post_compression_token_counts=post_counts,
        pre_compression_kv_cache_lengths=pre_lengths,
        history_lengths=history_lengths,
    )


def _build_token_spatial_temporal_heatmap_result(
    summary,
    collected,
    input_ids,
):
    try:
        layer_indices = np.asarray(
            [int(record["layer_idx"]) for record in collected],
            dtype=np.int16,
        )
        group_similarity = _stack_records(
            collected,
            "group_similarity",
            np.float32,
        )
        cluster_ids = _stack_records(
            collected,
            "kv_head_cluster_ids",
            np.int16,
        )
        selected_indices = _stack_records(
            collected,
            "selected_token_indices",
            np.int32,
        )
        selected_reason_ids = _stack_records(
            collected,
            "selected_token_reason_ids",
            np.int16,
        )
        selected_metric_scores = _stack_records(
            collected,
            "selected_token_metric_scores",
            np.float32,
        )
        token_heatmaps = _stack_records(
            collected,
            "token_heatmaps",
            np.float32,
        )
        row_cluster_ids = _stack_records(
            collected,
            "heatmap_row_cluster_ids",
            np.int16,
        )
        row_kv_head_indices = _stack_records(
            collected,
            "heatmap_row_kv_head_indices",
            np.int16,
        )
        row_gqa_group_indices = _stack_records(
            collected,
            "heatmap_row_gqa_group_indices",
            np.int16,
        )
    except ValueError as exc:
        summary["status"] = "error"
        summary["reason"] = f"Incompatible observation shapes across layers: {exc}"
        return _empty_result(summary)

    input_token_ids = input_ids.detach().to(device="cpu", dtype=torch.long).numpy()
    selected_ids = np.take(input_token_ids, selected_indices).astype(
        np.int64,
        copy=False,
    )
    cluster_counts = np.asarray(
        [int(record["cluster_count"]) for record in collected],
        dtype=np.int16,
    )
    pre_lengths = np.asarray(
        [int(record["pre_compression_kv_cache_len"]) for record in collected],
        dtype=np.int32,
    )
    history_lengths = np.asarray(
        [int(record["history_len"]) for record in collected],
        dtype=np.int32,
    )
    query_window_lengths = np.asarray(
        [int(record["query_window_len"]) for record in collected],
        dtype=np.int16,
    )
    summary.update(
        {
            "status": "saved",
            "valid_layer_indices": layer_indices.astype(int).tolist(),
            "selection_metric_names": list(TOKEN_HEATMAP_SELECTION_REASONS),
            "selected_token_indices_shape": list(selected_indices.shape),
            "selected_token_metric_scores_shape": list(
                selected_metric_scores.shape
            ),
            "token_heatmaps_shape": list(token_heatmaps.shape),
            "heatmap_layout": (
                "Rows preserve individual KV-head/GQA-group responses and are "
                "ordered by layer-local head cluster, KV head, then GQA group."
            ),
            "heatmap_color_scale": (
                "Each selected token is saved as a separate image whose color "
                "range is set independently to that token's finite minimum "
                "and maximum raw attention values."
            ),
        }
    )
    return HeadClusterObservationResult(
        summary=summary,
        layer_indices=layer_indices,
        raw_attention=np.asarray([], dtype=np.float32),
        selected_hist_mask=np.asarray([], dtype=np.bool_),
        history_selected_token_counts=np.asarray([], dtype=np.int32),
        recent_retained_token_counts=np.asarray([], dtype=np.int32),
        post_compression_token_counts=np.asarray([], dtype=np.int32),
        pre_compression_kv_cache_lengths=pre_lengths,
        history_lengths=history_lengths,
        group_similarity=group_similarity,
        kv_head_cluster_ids=cluster_ids,
        query_window_lengths=query_window_lengths,
        cluster_counts=cluster_counts,
        selected_token_indices=selected_indices,
        selected_token_ids=selected_ids,
        selected_token_reason_ids=selected_reason_ids,
        selected_token_metric_scores=selected_metric_scores,
        token_heatmaps=token_heatmaps,
        heatmap_row_cluster_ids=row_cluster_ids,
        heatmap_row_kv_head_indices=row_kv_head_indices,
        heatmap_row_gqa_group_indices=row_gqa_group_indices,
    )


def _build_head_cluster_token_distribution_result(summary, collected):
    try:
        layer_indices = np.asarray(
            [int(record["layer_idx"]) for record in collected],
            dtype=np.int16,
        )
        group_similarity = _stack_records(collected, "group_similarity", np.float32)
        cluster_ids = _stack_records(
            collected,
            "kv_head_cluster_ids",
            np.int16,
        )
        history_counts = _stack_records(
            collected,
            "history_selected_token_counts",
            np.int32,
        )
        cluster_selected_counts = _stack_records(
            collected,
            "cluster_selected_token_counts",
            np.int32,
        )
        cluster_selected_ratios = _stack_records(
            collected,
            "cluster_selected_token_ratios",
            np.float32,
        )
    except ValueError as exc:
        summary["status"] = "error"
        summary["reason"] = f"Incompatible observation shapes across layers: {exc}"
        return _empty_result(summary)

    cluster_counts = np.asarray(
        [int(record["cluster_count"]) for record in collected],
        dtype=np.int16,
    )
    pre_lengths = np.asarray(
        [int(record["pre_compression_kv_cache_len"]) for record in collected],
        dtype=np.int32,
    )
    history_lengths = np.asarray(
        [int(record["history_len"]) for record in collected],
        dtype=np.int32,
    )
    query_window_lengths = np.asarray(
        [int(record["query_window_len"]) for record in collected],
        dtype=np.int16,
    )
    summary.update(
        {
            "status": "saved",
            "valid_layer_indices": layer_indices.astype(int).tolist(),
            "group_similarity_shape": list(group_similarity.shape),
            "kv_head_cluster_ids_shape": list(cluster_ids.shape),
            "history_selected_token_counts_shape": list(history_counts.shape),
            "cluster_counts_shape": list(cluster_counts.shape),
            "cluster_selected_token_counts_shape": list(
                cluster_selected_counts.shape
            ),
            "cluster_selected_token_ratios_shape": list(
                cluster_selected_ratios.shape
            ),
            "cluster_padding": (
                "Cluster arrays use KV-head-count width; positions at or above "
                "cluster_counts[layer] are zero padding."
            ),
        }
    )
    return HeadClusterObservationResult(
        summary=summary,
        layer_indices=layer_indices,
        raw_attention=np.asarray([], dtype=np.float32),
        selected_hist_mask=np.asarray([], dtype=np.bool_),
        history_selected_token_counts=history_counts,
        recent_retained_token_counts=np.asarray([], dtype=np.int32),
        post_compression_token_counts=np.asarray([], dtype=np.int32),
        pre_compression_kv_cache_lengths=pre_lengths,
        history_lengths=history_lengths,
        group_similarity=group_similarity,
        kv_head_cluster_ids=cluster_ids,
        query_window_lengths=query_window_lengths,
        cluster_counts=cluster_counts,
        cluster_selected_token_counts=cluster_selected_counts,
        cluster_selected_token_ratios=cluster_selected_ratios,
    )


def _build_head_cluster_pca_result(summary, collected):
    try:
        layer_indices = np.asarray(
            [int(record["layer_idx"]) for record in collected],
            dtype=np.int16,
        )
        group_similarity = _stack_records(collected, "group_similarity", np.float32)
        cluster_ids = _stack_records(collected, "kv_head_cluster_ids", np.int16)
        pca_points = _stack_records(collected, "kv_head_pca_points", np.float32)
        explained = _stack_records(
            collected,
            "pca_explained_variance_ratio",
            np.float32,
        )
    except ValueError as exc:
        summary["status"] = "error"
        summary["reason"] = f"Incompatible observation shapes across layers: {exc}"
        return _empty_result(summary)

    pre_lengths = np.asarray(
        [int(record["pre_compression_kv_cache_len"]) for record in collected],
        dtype=np.int32,
    )
    history_lengths = np.asarray(
        [int(record["history_len"]) for record in collected],
        dtype=np.int32,
    )
    query_window_lengths = np.asarray(
        [int(record["query_window_len"]) for record in collected],
        dtype=np.int16,
    )
    summary.update(
        {
            "status": "saved",
            "valid_layer_indices": layer_indices.astype(int).tolist(),
            "group_similarity_shape": list(group_similarity.shape),
            "kv_head_cluster_ids_shape": list(cluster_ids.shape),
            "kv_head_pca_points_shape": list(pca_points.shape),
            "pca_explained_variance_ratio_shape": list(explained.shape),
            "query_window_lengths_shape": list(query_window_lengths.shape),
            "history_lengths_shape": list(history_lengths.shape),
        }
    )
    return HeadClusterObservationResult(
        summary=summary,
        layer_indices=layer_indices,
        raw_attention=np.asarray([], dtype=np.float32),
        selected_hist_mask=np.asarray([], dtype=np.bool_),
        history_selected_token_counts=np.asarray([], dtype=np.int32),
        recent_retained_token_counts=np.asarray([], dtype=np.int32),
        post_compression_token_counts=np.asarray([], dtype=np.int32),
        pre_compression_kv_cache_lengths=pre_lengths,
        history_lengths=history_lengths,
        group_similarity=group_similarity,
        kv_head_cluster_ids=cluster_ids,
        kv_head_pca_points=pca_points,
        pca_explained_variance_ratio=explained,
        query_window_lengths=query_window_lengths,
    )


def save_head_cluster_observation(
    result: HeadClusterObservationResult,
    output_dir: str,
    prefix: str = "head_cluster_observation",
    token_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    npz_path = os.path.join(output_dir, f"{prefix}.npz")
    _annotate_selected_tokens(result, token_entries)
    image_paths = plot_head_cluster_observation(result, output_dir, prefix)
    result.summary["image_files"] = {
        name: os.path.basename(path) for name, path in image_paths.items()
    }

    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(result.summary, fout, ensure_ascii=False, indent=2)
    submode = _result_submode(result)
    if submode == TOKEN_SPATIAL_TEMPORAL_HEATMAP_SUBMODE:
        np.savez_compressed(
            npz_path,
            layer_indices=result.layer_indices,
            group_similarity=result.group_similarity,
            kv_head_cluster_ids=result.kv_head_cluster_ids,
            cluster_counts=result.cluster_counts,
            selected_token_indices=result.selected_token_indices,
            selected_token_ids=result.selected_token_ids,
            selected_token_reason_ids=result.selected_token_reason_ids,
            selected_token_metric_scores=result.selected_token_metric_scores,
            token_heatmaps=result.token_heatmaps,
            heatmap_row_cluster_ids=result.heatmap_row_cluster_ids,
            heatmap_row_kv_head_indices=result.heatmap_row_kv_head_indices,
            heatmap_row_gqa_group_indices=result.heatmap_row_gqa_group_indices,
            pre_compression_kv_cache_lengths=result.pre_compression_kv_cache_lengths,
            history_lengths=result.history_lengths,
            query_window_lengths=result.query_window_lengths,
        )
    elif submode == HEAD_CLUSTER_PCA_SUBMODE:
        np.savez_compressed(
            npz_path,
            layer_indices=result.layer_indices,
            group_similarity=result.group_similarity,
            kv_head_cluster_ids=result.kv_head_cluster_ids,
            kv_head_pca_points=result.kv_head_pca_points,
            pca_explained_variance_ratio=result.pca_explained_variance_ratio,
            query_window_lengths=result.query_window_lengths,
            history_lengths=result.history_lengths,
        )
    elif submode == HEAD_CLUSTER_TOKEN_DISTRIBUTION_SUBMODE:
        np.savez_compressed(
            npz_path,
            layer_indices=result.layer_indices,
            group_similarity=result.group_similarity,
            kv_head_cluster_ids=result.kv_head_cluster_ids,
            history_selected_token_counts=result.history_selected_token_counts,
            cluster_counts=result.cluster_counts,
            cluster_selected_token_counts=result.cluster_selected_token_counts,
            cluster_selected_token_ratios=result.cluster_selected_token_ratios,
            pre_compression_kv_cache_lengths=result.pre_compression_kv_cache_lengths,
            history_lengths=result.history_lengths,
            query_window_lengths=result.query_window_lengths,
        )
    else:
        np.savez_compressed(
            npz_path,
            layer_indices=result.layer_indices,
            raw_attention=result.raw_attention,
            selected_hist_mask=result.selected_hist_mask,
            history_selected_token_counts=result.history_selected_token_counts,
            recent_retained_token_counts=result.recent_retained_token_counts,
            post_compression_token_counts=result.post_compression_token_counts,
            pre_compression_kv_cache_lengths=result.pre_compression_kv_cache_lengths,
            history_lengths=result.history_lengths,
        )
    return {"summary": summary_path, "npz": npz_path, "images": image_paths}


def _annotate_selected_tokens(result, token_entries):
    if (
        _result_submode(result) != TOKEN_SPATIAL_TEMPORAL_HEATMAP_SUBMODE
        or not token_entries
    ):
        return
    entries_by_index = {
        int(entry["index"]): entry
        for entry in token_entries
        if entry.get("index") is not None
    }
    for layer in result.summary.get("layers", []):
        for selected in layer.get("selected_tokens", []):
            entry = entries_by_index.get(int(selected["index"]))
            if entry is None:
                continue
            selected["piece"] = entry.get("piece")
            selected["text"] = entry.get("text")


def _result_submode(result):
    return result.summary.get("config", {}).get(
        "submode",
        HEAD_BUDGET_ATTENTION_SUBMODE,
    )


def plot_head_cluster_observation(
    result: HeadClusterObservationResult,
    output_dir: str,
    prefix: str = "head_cluster_observation",
) -> dict[str, str]:
    if result.layer_indices.size == 0:
        return {}

    _setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    submode = _result_submode(result)
    if submode == TOKEN_SPATIAL_TEMPORAL_HEATMAP_SUBMODE:
        return _plot_token_spatial_temporal_heatmaps(
            result,
            output_dir,
            prefix,
            plt,
        )
    if submode == HEAD_CLUSTER_PCA_SUBMODE:
        return _plot_head_cluster_pca_observation(result, output_dir, prefix, plt)
    if submode == HEAD_CLUSTER_TOKEN_DISTRIBUTION_SUBMODE:
        return _plot_head_cluster_token_distribution(
            result,
            output_dir,
            prefix,
            plt,
        )

    image_paths = {}
    for layer_pos, layer_idx_value in enumerate(result.layer_indices):
        layer_idx = int(layer_idx_value)
        budget_key = f"head_budget_layer_{layer_idx:03d}"
        budget_path = os.path.join(output_dir, f"{prefix}_{budget_key}.png")
        _plot_head_budget(
            result.history_selected_token_counts[layer_pos],
            result.recent_retained_token_counts[layer_pos],
            result.post_compression_token_counts[layer_pos],
            layer_idx,
            budget_path,
            plt,
        )
        image_paths[budget_key] = budget_path

        attention_key = f"raw_attention_layer_{layer_idx:03d}"
        attention_path = os.path.join(output_dir, f"{prefix}_{attention_key}.png")
        _plot_raw_attention(
            result.raw_attention[layer_pos],
            int(result.history_lengths[layer_pos]),
            layer_idx,
            attention_path,
            plt,
        )
        image_paths[attention_key] = attention_path
    return image_paths


def _plot_token_spatial_temporal_heatmaps(
    result,
    output_dir,
    prefix,
    plt,
):
    required = {
        "selected_token_indices": result.selected_token_indices,
        "selected_token_ids": result.selected_token_ids,
        "selected_token_reason_ids": result.selected_token_reason_ids,
        "token_heatmaps": result.token_heatmaps,
        "heatmap_row_cluster_ids": result.heatmap_row_cluster_ids,
        "heatmap_row_kv_head_indices": result.heatmap_row_kv_head_indices,
        "heatmap_row_gqa_group_indices": result.heatmap_row_gqa_group_indices,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "token_spatial_temporal_heatmap result is missing required arrays: "
            f"{', '.join(sorted(missing))}."
        )

    heatmaps = np.asarray(result.token_heatmaps, dtype=np.float32)
    layer_count = int(result.layer_indices.size)
    if heatmaps.ndim != 4 or heatmaps.shape[0] != layer_count:
        raise ValueError(
            "token_heatmaps must have shape [layer, token, head row, query]."
        )
    image_paths = {}
    layer_summaries = {
        int(layer["layer_idx"]): layer
        for layer in result.summary.get("layers", [])
        if layer.get("status") == "saved"
    }
    for layer_pos, layer_idx_value in enumerate(result.layer_indices):
        layer_idx = int(layer_idx_value)
        layer_maps = heatmaps[layer_pos]
        token_count, row_count, query_count = layer_maps.shape
        if token_count < 1:
            continue
        row_clusters = np.asarray(
            result.heatmap_row_cluster_ids[layer_pos],
            dtype=np.int16,
        )
        row_kv_heads = np.asarray(
            result.heatmap_row_kv_head_indices[layer_pos],
            dtype=np.int16,
        )
        row_groups = np.asarray(
            result.heatmap_row_gqa_group_indices[layer_pos],
            dtype=np.int16,
        )
        if not (
            row_clusters.shape
            == row_kv_heads.shape
            == row_groups.shape
            == (row_count,)
        ):
            raise ValueError("Token heatmap row metadata must match its head rows.")

        selected_summary = layer_summaries.get(layer_idx, {}).get(
            "selected_tokens",
            [],
        )
        boundaries = np.flatnonzero(np.diff(row_clusters)) + 0.5
        starts = np.r_[0, np.flatnonzero(np.diff(row_clusters)) + 1]
        ends = np.r_[starts[1:], row_count]
        for token_pos in range(token_count):
            token_map = layer_maps[token_pos]
            vmin = float(np.nanmin(token_map))
            vmax = float(np.nanmax(token_map))
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax):
                vmax = vmin + 1.0
            if vmax <= vmin:
                vmax = vmin + max(abs(vmin) * 1e-6, 1e-12)

            fig_height = max(5.2, row_count * 0.16 + 2.8)
            fig, ax = plt.subplots(
                figsize=(8.2, fig_height),
                dpi=180,
            )
            image = ax.imshow(
                token_map,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )
            for boundary in boundaries:
                ax.axhline(boundary, color="white", linewidth=1.0)
            for start, end in zip(starts, ends):
                midpoint = (float(start) + float(end) - 1.0) / 2.0
                ax.text(
                    -0.03,
                    midpoint,
                    f"C{int(row_clusters[start])}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    clip_on=False,
                )

            query_positions = np.arange(query_count)
            ax.set_xticks(query_positions)
            ax.set_xticklabels(
                [str(value) for value in range(-query_count + 1, 1)],
                fontsize=7,
            )
            ax.set_xlabel("Query-window relative position", fontsize=8)
            ax.set_yticks(np.arange(row_count))
            ax.set_yticklabels(
                [
                    f"KV{int(kv_head)}/G{int(group)}"
                    for kv_head, group in zip(row_kv_heads, row_groups)
                ],
                fontsize=5.5,
            )
            ax.set_ylabel("Heads grouped by cluster", fontsize=8)

            token_idx = int(result.selected_token_indices[layer_pos, token_pos])
            token_id = int(result.selected_token_ids[layer_pos, token_pos])
            reason_id = int(
                result.selected_token_reason_ids[layer_pos, token_pos]
            )
            reason = TOKEN_HEATMAP_SELECTION_REASONS[reason_id]
            token_text = None
            if token_pos < len(selected_summary):
                token_text = selected_summary[token_pos].get("text")
                if token_text is None:
                    token_text = selected_summary[token_pos].get("piece")
            token_text = "" if token_text is None else str(token_text)
            token_text = token_text.replace("\n", "\\n")
            if len(token_text) > 32:
                token_text = f"{token_text[:29]}..."
            text_suffix = f" {token_text!r}" if token_text else ""
            ax.set_title(
                f"TridentKV token response — layer {layer_idx}\n"
                f"token {token_idx} · id {token_id}{text_suffix} · {reason}",
                fontsize=10,
            )
            colorbar = fig.colorbar(
                image,
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )
            colorbar.set_label("Raw attention probability")
            ax.text(
                0.5,
                -0.16,
                (
                    "Rows preserve individual KV/GQA heads; white lines "
                    "delimit clusters"
                ),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )
            fig.subplots_adjust(
                left=0.19,
                right=0.88,
                top=0.88,
                bottom=0.19,
            )
            image_key = (
                f"token_spatial_temporal_heatmap_layer_{layer_idx:03d}"
                f"_token_{token_idx:06d}"
            )
            image_path = os.path.join(
                output_dir,
                f"{prefix}_{image_key}.png",
            )
            fig.savefig(image_path)
            plt.close(fig)
            image_paths[image_key] = image_path
    return image_paths


def _plot_head_cluster_token_distribution(result, output_dir, prefix, plt):
    required = {
        "cluster_counts": result.cluster_counts,
        "cluster_selected_token_ratios": result.cluster_selected_token_ratios,
        "kv_head_cluster_ids": result.kv_head_cluster_ids,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "head_cluster_token_distribution result is missing required arrays: "
            f"{', '.join(sorted(missing))}."
        )

    ratios = np.asarray(result.cluster_selected_token_ratios, dtype=np.float32)
    cluster_counts = np.asarray(result.cluster_counts, dtype=np.int16)
    cluster_ids = np.asarray(result.kv_head_cluster_ids, dtype=np.int16)
    layer_count = int(result.layer_indices.size)
    if ratios.ndim != 2 or ratios.shape[0] != layer_count:
        raise ValueError(
            "cluster_selected_token_ratios must have shape [layer, max_cluster]."
        )
    if cluster_counts.shape != (layer_count,):
        raise ValueError("cluster_counts must have shape [layer].")
    if cluster_ids.ndim != 2 or cluster_ids.shape[0] != layer_count:
        raise ValueError("kv_head_cluster_ids must have shape [layer, KV head].")

    cluster_head_counts = np.zeros_like(ratios, dtype=np.int16)
    for layer_pos, cluster_count in enumerate(cluster_counts):
        for cluster_id in range(int(cluster_count)):
            cluster_head_counts[layer_pos, cluster_id] = np.count_nonzero(
                cluster_ids[layer_pos] == cluster_id
            )

    image_key = "head_cluster_token_distribution_overview"
    image_path = os.path.join(output_dir, f"{prefix}_{image_key}.png")
    fig_height = max(4.8, min(24.0, 0.42 * layer_count + 2.8))
    fig, ax = plt.subplots(figsize=(11.0, fig_height), dpi=180)
    positions = np.arange(layer_count)
    left = np.zeros(layer_count, dtype=np.float32)
    max_cluster_count = int(cluster_counts.max(initial=0))
    cmap = plt.get_cmap("tab20")
    for cluster_id in range(max_cluster_count):
        valid = cluster_id < cluster_counts
        widths = np.where(valid, ratios[:, cluster_id], 0.0)
        bars = ax.barh(
            positions,
            widths * 100.0,
            left=left * 100.0,
            color=cmap(cluster_id % 20),
            edgecolor="white",
            linewidth=0.5,
            label=f"C{cluster_id}",
        )
        for layer_pos, (bar, width) in enumerate(zip(bars, widths)):
            if not valid[layer_pos] or float(width) < 0.045:
                continue
            ax.text(
                (left[layer_pos] + width * 0.5) * 100.0,
                bar.get_y() + bar.get_height() * 0.5,
                (
                    f"C{cluster_id} ({int(cluster_head_counts[layer_pos, cluster_id])}H)\n"
                    f"{100.0 * float(width):.1f}%"
                ),
                ha="center",
                va="center",
                fontsize=7,
            )
        left += widths

    zero_rows = np.flatnonzero(np.isclose(left, 0.0))
    for layer_pos in zero_rows:
        ax.text(
            50.0,
            layer_pos,
            "no selected history slots",
            ha="center",
            va="center",
            fontsize=8,
            color="#555555",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [
            (
                f"Layer {int(layer_idx)}\n"
                + ", ".join(
                    f"C{cluster_id}={int(cluster_head_counts[layer_pos, cluster_id])}H"
                    for cluster_id in range(int(cluster_counts[layer_pos]))
                )
            )
            for layer_pos, layer_idx in enumerate(result.layer_indices)
        ]
    )
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Share of selected historical (KV head, token) slots (%)")
    ax.set_ylabel("Decoder layer")
    ax.set_title(
        "Head-cluster token allocation by layer\n"
        "Ck (nH) means cluster k contains n KV heads; "
        "cluster ids are layer-local; fixed recent window is excluded"
    )
    ax.grid(axis="x", alpha=0.2, linewidth=0.6)
    if max_cluster_count > 0:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(max_cluster_count, 10),
            frameon=False,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(image_path)
    plt.close(fig)
    return {image_key: image_path}


def _plot_head_cluster_pca_observation(result, output_dir, prefix, plt):
    required = {
        "group_similarity": result.group_similarity,
        "kv_head_cluster_ids": result.kv_head_cluster_ids,
        "kv_head_pca_points": result.kv_head_pca_points,
        "pca_explained_variance_ratio": result.pca_explained_variance_ratio,
        "query_window_lengths": result.query_window_lengths,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "head_cluster_pca result is missing required arrays: "
            f"{', '.join(sorted(missing))}."
        )

    image_paths = {}
    for layer_pos, layer_idx_value in enumerate(result.layer_indices):
        layer_idx = int(layer_idx_value)
        partition_key = f"head_cluster_partition_layer_{layer_idx:03d}"
        partition_path = os.path.join(output_dir, f"{prefix}_{partition_key}.png")
        _plot_head_cluster_partition(
            result.group_similarity[layer_pos],
            result.kv_head_cluster_ids[layer_pos],
            layer_idx,
            partition_path,
            plt,
        )
        image_paths[partition_key] = partition_path

        pca_key = f"raw_attention_pca_layer_{layer_idx:03d}"
        pca_path = os.path.join(output_dir, f"{prefix}_{pca_key}.png")
        _plot_kv_head_attention_pca(
            result.kv_head_pca_points[layer_pos],
            result.kv_head_cluster_ids[layer_pos],
            result.pca_explained_variance_ratio[layer_pos],
            int(result.query_window_lengths[layer_pos]),
            int(result.history_lengths[layer_pos]),
            layer_idx,
            pca_path,
            plt,
        )
        image_paths[pca_key] = pca_path
    return image_paths


def _plot_head_cluster_partition(
    group_similarity,
    cluster_ids,
    layer_idx,
    output_path,
    plt,
):
    similarity = np.asarray(group_similarity, dtype=np.float32)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int16)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("group_similarity must be a square matrix.")
    if cluster_ids.shape != (similarity.shape[0],):
        raise ValueError("kv_head_cluster_ids must match group_similarity.")

    order = np.lexsort((np.arange(cluster_ids.size), cluster_ids))
    reordered = similarity[np.ix_(order, order)]
    reordered_clusters = cluster_ids[order]
    lower = float(np.nanmin(reordered))
    if 1.0 - lower < 1e-6:
        lower = 0.0

    size = max(5.8, min(10.0, 0.55 * similarity.shape[0] + 3.5))
    fig, ax = plt.subplots(figsize=(size, size), dpi=180)
    image = ax.imshow(
        reordered,
        cmap="viridis",
        vmin=max(-1.0, lower),
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    tick_positions = np.arange(order.size)
    tick_labels = [str(int(head_idx)) for head_idx in order]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("KV head id (cluster-reordered)")
    ax.set_ylabel("KV head id (cluster-reordered)")
    ax.set_title(f"KV-head cluster partition — layer {layer_idx}")

    boundaries = np.flatnonzero(np.diff(reordered_clusters)) + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=1.8)
        ax.axvline(boundary, color="white", linewidth=1.8)

    starts = np.r_[0, np.flatnonzero(np.diff(reordered_clusters)) + 1]
    ends = np.r_[starts[1:], reordered_clusters.size]
    for start, end in zip(starts, ends):
        cluster_id = int(reordered_clusters[start])
        midpoint = (float(start) + float(end) - 1.0) / 2.0
        ax.text(
            midpoint,
            -0.82,
            f"C{cluster_id}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )

    if similarity.shape[0] <= 16:
        for row in range(reordered.shape[0]):
            for column in range(reordered.shape[1]):
                value = float(reordered[row, column])
                color = "white" if value < (lower + 1.0) * 0.5 else "black"
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_kv_head_attention_pca(
    pca_points,
    cluster_ids,
    explained,
    query_window_len,
    history_len,
    layer_idx,
    output_path,
    plt,
):
    points = np.asarray(pca_points, dtype=np.float32)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int16)
    explained = np.asarray(explained, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("kv_head_pca_points must have shape [kv_head, 2].")
    if cluster_ids.shape != (points.shape[0],):
        raise ValueError("kv_head_cluster_ids must match kv_head_pca_points.")
    if explained.shape != (2,):
        raise ValueError("pca_explained_variance_ratio must have shape [2].")

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    cmap = plt.get_cmap("tab10")
    unique_clusters = np.unique(cluster_ids)
    for color_pos, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=64,
            alpha=0.86,
            color=cmap(color_pos % 10),
            edgecolors="#202020",
            linewidths=0.7,
            label=f"Cluster {int(cluster_id)}",
            zorder=2,
        )
    for head_idx, point in enumerate(points):
        ax.annotate(
            f"H{head_idx}",
            xy=(point[0], point[1]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            zorder=3,
        )

    ax.set_title(
        f"KV-head raw-attention PCA — layer {layer_idx}\n"
        f"query window={query_window_len}, history={history_len}"
    )
    ax.set_xlabel(f"PC 1 ({100.0 * float(explained[0]):.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({100.0 * float(explained[1]):.1f}% variance)")
    ax.grid(alpha=0.2, linewidth=0.7)
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_head_budget(history_counts, recent_counts, post_counts, layer_idx, output_path, plt):
    head_count = int(post_counts.shape[0])
    positions = np.arange(head_count)
    fig_width = max(6.0, 0.45 * head_count + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=180)
    ax.bar(positions, history_counts, label="Selected history", color="#4c78a8")
    ax.bar(
        positions,
        recent_counts,
        bottom=history_counts,
        label="Retained recent window",
        color="#f58518",
    )
    for position, count in zip(positions, post_counts):
        ax.text(position, int(count), str(int(count)), ha="center", va="bottom", fontsize=7)
    ax.set_title(f"Post-compression head budget — layer {layer_idx}")
    ax.set_xlabel("KV head id")
    ax.set_ylabel("Token count")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(position)) for position in positions])
    ax.legend(loc="best")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_raw_attention(raw_attention, history_len, layer_idx, output_path, plt):
    raw_attention = np.asarray(raw_attention, dtype=np.float32)
    head_count, token_count = raw_attention.shape
    x = np.arange(token_count)
    fig_width = max(8.0, min(18.0, token_count / 6000.0 + 8.0))
    fig_height = max(2.4, 1.75 * head_count)
    fig, axes = plt.subplots(
        head_count,
        1,
        sharex=True,
        figsize=(fig_width, fig_height),
        dpi=180,
    )
    if head_count == 1:
        axes = [axes]
    for head_idx, ax in enumerate(axes):
        ax.plot(x, raw_attention[head_idx], linewidth=0.7, color="#4c78a8")
        if history_len < token_count:
            ax.axvspan(
                history_len - 0.5,
                token_count - 0.5,
                color="#f58518",
                alpha=0.16,
            )
            ax.axvline(
                history_len - 0.5,
                color="#f58518",
                linewidth=0.7,
                linestyle="--",
            )
        ax.set_ylabel(f"H{head_idx}", rotation=0, labelpad=18, va="center")
        ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    axes[0].set_title(
        f"Raw pre-compression attention by KV head — layer {layer_idx}\n"
        "shaded region: fixed recent window"
    )
    axes[-1].set_xlabel("Pre-compression sequence position")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _run_cached_observation_forward(model, inputs):
    from transformers.cache_utils import DynamicCache

    forward_kwargs = {
        **inputs,
        "past_key_values": DynamicCache(config=model.config),
        "use_cache": True,
        "return_dict": True,
        "logits_to_keep": 1,
    }
    with torch.inference_mode():
        try:
            return model(**forward_kwargs)
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            forward_kwargs.pop("logits_to_keep", None)
            return model(**forward_kwargs)


def _snapshot_observation_state(model, attentions):
    targets = [(model, "length"), (model, "after_think")]
    seen_configs = set()
    for _, attention in attentions:
        config = getattr(attention, "config", None)
        if config is None or id(config) in seen_configs:
            continue
        seen_configs.add(id(config))
        targets.append((config, "compression"))
        targets.append((config, "_tridentkv_layer_budget_state"))
    for _, attention in attentions:
        targets.append((attention.kv_cluster, "prefill_layer_budget_layers"))
        targets.append((attention.kv_cluster, "_tridentkv_head_cluster_result"))

    snapshots = []
    for target, attribute in targets:
        existed = hasattr(target, attribute)
        value = getattr(target, attribute) if existed else None
        if existed and attribute == "_tridentkv_layer_budget_state":
            value = copy.deepcopy(value)
        snapshots.append((target, attribute, existed, value))
    return snapshots


def _restore_observation_state(snapshots):
    for target, attribute, existed, value in reversed(snapshots):
        if existed:
            setattr(target, attribute, value)
        elif hasattr(target, attribute):
            delattr(target, attribute)


def _get_compression_attentions(model):
    attentions = []
    for layer_idx, layer in enumerate(_get_decoder_layers(model)):
        attention = _get_attention_module(layer)
        if attention is not None and hasattr(attention, "kv_cluster"):
            attentions.append((layer_idx, attention))
    return attentions


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
            if module is not None and len(module) > 0:
                return list(module)
        except TypeError:
            pass
    return []


def _get_attention_module(layer):
    for attr in ("self_attn", "attention", "attn"):
        attention = getattr(layer, attr, None)
        if attention is not None:
            return attention
    return None


def _stack_records(records, key, dtype):
    arrays = [record[key].detach().cpu().numpy() for record in records]
    return np.stack(arrays, axis=0).astype(dtype, copy=False)


def _setup_matplotlib_cache():
    cache_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"),
        "head_cluster_observation_matplotlib_cache",
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)


def _empty_result(summary):
    return HeadClusterObservationResult(
        summary=summary,
        layer_indices=np.asarray([], dtype=np.int16),
        raw_attention=np.asarray([], dtype=np.float32),
        selected_hist_mask=np.asarray([], dtype=np.bool_),
        history_selected_token_counts=np.asarray([], dtype=np.int32),
        recent_retained_token_counts=np.asarray([], dtype=np.int32),
        post_compression_token_counts=np.asarray([], dtype=np.int32),
        pre_compression_kv_cache_lengths=np.asarray([], dtype=np.int32),
        history_lengths=np.asarray([], dtype=np.int32),
        group_similarity=np.asarray([], dtype=np.float32),
        kv_head_cluster_ids=np.asarray([], dtype=np.int16),
        kv_head_pca_points=np.asarray([], dtype=np.float32),
        pca_explained_variance_ratio=np.asarray([], dtype=np.float32),
        query_window_lengths=np.asarray([], dtype=np.int16),
        cluster_counts=np.asarray([], dtype=np.int16),
        cluster_selected_token_counts=np.asarray([], dtype=np.int32),
        cluster_selected_token_ratios=np.asarray([], dtype=np.float32),
        selected_token_indices=np.asarray([], dtype=np.int32),
        selected_token_ids=np.asarray([], dtype=np.int64),
        selected_token_reason_ids=np.asarray([], dtype=np.int16),
        selected_token_metric_scores=np.asarray([], dtype=np.float32),
        token_heatmaps=np.asarray([], dtype=np.float32),
        heatmap_row_cluster_ids=np.asarray([], dtype=np.int16),
        heatmap_row_kv_head_indices=np.asarray([], dtype=np.int16),
        heatmap_row_gqa_group_indices=np.asarray([], dtype=np.int16),
    )
