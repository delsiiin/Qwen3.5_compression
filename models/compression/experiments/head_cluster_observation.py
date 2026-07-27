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
HEAD_CLUSTER_PCA_METHODS = {
    "tridentkv_head_cluster",
    "tridentkv",
}
SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES = {
    HEAD_BUDGET_ATTENTION_SUBMODE,
    HEAD_CLUSTER_PCA_SUBMODE,
}


@dataclass(frozen=True)
class HeadClusterObservationConfig:
    submode: str = HEAD_BUDGET_ATTENTION_SUBMODE
    max_prefill_tokens: int | None = None

    def __post_init__(self):
        if self.submode not in SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES:
            raise ValueError(
                f"submode must be one of {sorted(SUPPORTED_HEAD_CLUSTER_OBSERVATION_SUBMODES)}."
            )
        if self.max_prefill_tokens is not None and self.max_prefill_tokens < 1:
            raise ValueError("max_prefill_tokens must be at least 1 when provided.")


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
        cluster.enable_head_cluster_observation(config.submode)
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
            if config.submode == HEAD_CLUSTER_PCA_SUBMODE:
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
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    npz_path = os.path.join(output_dir, f"{prefix}.npz")
    image_paths = plot_head_cluster_observation(result, output_dir, prefix)
    result.summary["image_files"] = {
        name: os.path.basename(path) for name, path in image_paths.items()
    }

    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(result.summary, fout, ensure_ascii=False, indent=2)
    if _result_submode(result) == HEAD_CLUSTER_PCA_SUBMODE:
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

    if _result_submode(result) == HEAD_CLUSTER_PCA_SUBMODE:
        return _plot_head_cluster_pca_observation(result, output_dir, prefix, plt)

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
    )
