import json
import os

import numpy as np
import torch

from attn_heatmap import (
    AttentionCaptureRecorder,
    build_run_dir,
    build_token_entries,
    extract_result_summary,
    get_text_config,
    sanitize_slug,
)
from query_window_similarity import get_decoder_layers, run_observation_forward, setup_matplotlib_cache


ATTN_LAYER_SIMILARITY_METRIC = "cosine_similarity"
ATTN_LAYER_SIMILARITY_REDUCTION = "mean_heads_flatten_attention"
ATTN_HEAD_SIMILARITY_METRIC = "cosine_similarity"
ATTN_HEAD_SIMILARITY_REDUCTION = "mean_gqa_group_heads_flatten_attention_per_layer"
ATTN_RAW_HEAD_SIMILARITY_METRIC = "cosine_similarity"
ATTN_RAW_HEAD_SIMILARITY_REDUCTION = "flatten_attention_per_layer_before_gqa_grouping"
ATTN_HEAD_MEAN_SIMILARITY_METRIC = "cosine_similarity"
ATTN_HEAD_MEAN_SIMILARITY_REDUCTION = "raw_head_to_mean_gqa_group_attention_distribution_per_layer"
ATTN_HEAD_CLUSTER_DISTANCE_METRIC = "1 - cosine_similarity"
ATTN_HEAD_CLUSTER_THRESHOLD_MODE = "mean"
ATTN_HEAD_CLUSTER_OUTLIER_METHOD = None
ATTN_LAYER_SIMILARITY_QUERY_TOKEN_SCOPE = "input_tail_window"


def select_attention_query_window(attn, window_size):
    """Keep attention distributions for the final query-token window.

    The key dimension is intentionally left intact: each selected prompt-tail
    token is still represented by its attention distribution over the full
    input context.
    """
    attn = np.asarray(attn)
    if attn.ndim != 4:
        raise ValueError("attn must have shape [layers, heads, query_tokens, key_tokens].")
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    query_token_count = int(attn.shape[2])
    if query_token_count < 1:
        raise ValueError("attn must contain at least one query token.")
    selected_count = min(window_size, query_token_count)
    query_start = query_token_count - selected_count
    return attn[:, :, query_start:, :], query_start, selected_count


def build_attention_layer_similarity(attn, layer_indices, eps=1e-12):
    attn = np.asarray(attn)
    if attn.ndim != 4:
        raise ValueError("attn must have shape [layers, heads, query_tokens, key_tokens].")
    if attn.shape[0] < 1:
        raise ValueError("No attention layers were provided.")

    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    if len(layer_indices) != attn.shape[0]:
        raise ValueError("layer_indices must match the attention layer dimension.")

    mean_head_attn = torch.as_tensor(attn, dtype=torch.float32).mean(dim=1)
    vectors = mean_head_attn.reshape(mean_head_attn.shape[0], -1)
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1, eps=float(eps))
    similarity = torch.matmul(vectors, vectors.transpose(0, 1)).cpu().numpy()
    return similarity.astype(np.float32, copy=False), layer_indices


def build_attention_head_similarity(attn, layer_indices, gqa_group_count=None, eps=1e-12):
    attn = np.asarray(attn)
    if attn.ndim != 4:
        raise ValueError("attn must have shape [layers, heads, query_tokens, key_tokens].")
    if attn.shape[0] < 1:
        raise ValueError("No attention layers were provided.")
    if attn.shape[1] < 1:
        raise ValueError("No attention heads were provided.")

    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    if len(layer_indices) != attn.shape[0]:
        raise ValueError("layer_indices must match the attention layer dimension.")

    head_count = int(attn.shape[1])
    if gqa_group_count is None:
        gqa_group_count = head_count
    gqa_group_count = int(gqa_group_count)
    if gqa_group_count < 1:
        raise ValueError("gqa_group_count must be at least 1.")
    if gqa_group_count > head_count:
        raise ValueError("gqa_group_count cannot exceed the captured attention head count.")
    if head_count % gqa_group_count != 0:
        raise ValueError(
            f"Captured attention head count {head_count} is not divisible by "
            f"gqa_group_count {gqa_group_count}."
        )

    gqa_group_size = head_count // gqa_group_count
    grouped_attn = (
        torch.as_tensor(attn, dtype=torch.float32)
        .reshape(attn.shape[0], gqa_group_count, gqa_group_size, attn.shape[2], attn.shape[3])
        .mean(dim=2)
    )
    vectors = grouped_attn.reshape(attn.shape[0], gqa_group_count, -1)
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=2, eps=float(eps))
    similarity = torch.matmul(vectors, vectors.transpose(1, 2)).cpu().numpy()
    return (
        similarity.astype(np.float32, copy=False),
        layer_indices,
        {
            "attention_head_count": head_count,
            "gqa_group_count": gqa_group_count,
            "gqa_group_size": gqa_group_size,
        },
    )


def build_raw_attention_head_similarity(attn, layer_indices, eps=1e-12):
    attn = np.asarray(attn)
    if attn.ndim != 4:
        raise ValueError("attn must have shape [layers, heads, query_tokens, key_tokens].")
    if attn.shape[0] < 1:
        raise ValueError("No attention layers were provided.")
    if attn.shape[1] < 1:
        raise ValueError("No attention heads were provided.")

    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    if len(layer_indices) != attn.shape[0]:
        raise ValueError("layer_indices must match the attention layer dimension.")

    head_count = int(attn.shape[1])
    vectors = torch.as_tensor(attn, dtype=torch.float32).reshape(attn.shape[0], head_count, -1)
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=2, eps=float(eps))
    similarity = torch.matmul(vectors, vectors.transpose(1, 2)).cpu().numpy()
    return (
        similarity.astype(np.float32, copy=False),
        layer_indices,
        {
            "raw_head_count": head_count,
        },
    )


def _compute_finite_mean(values, metric_name):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError(f"{metric_name} requires at least one finite value.")
    return float(np.mean(finite_values))


def compute_attention_head_cluster_distance_threshold(layer_head_similarity):
    layer_head_similarity = np.asarray(layer_head_similarity)
    if layer_head_similarity.ndim != 2:
        raise ValueError("layer_head_similarity must have shape [heads, heads].")
    if layer_head_similarity.shape[0] != layer_head_similarity.shape[1]:
        raise ValueError("layer_head_similarity must be square.")
    if not np.all(np.isfinite(layer_head_similarity)):
        raise ValueError("layer_head_similarity must contain only finite values.")

    head_count = int(layer_head_similarity.shape[0])
    if head_count <= 1:
        return 0.0

    upper_triangle = np.triu_indices(head_count, k=1)
    pairwise_distance = 1.0 - layer_head_similarity[upper_triangle[0], upper_triangle[1]]
    return _compute_finite_mean(
        pairwise_distance,
        "attn_head_cluster_threshold_mode",
    )


def compute_attention_head_cluster_distance_thresholds(head_similarity):
    head_similarity = np.asarray(head_similarity)
    if head_similarity.ndim != 3:
        raise ValueError("head_similarity must have shape [layers, heads, heads].")
    if head_similarity.shape[1] != head_similarity.shape[2]:
        raise ValueError("head_similarity must be square on the head dimensions.")
    if not np.all(np.isfinite(head_similarity)):
        raise ValueError("head_similarity must contain only finite values.")

    return np.asarray(
        [
            compute_attention_head_cluster_distance_threshold(layer_similarity)
            for layer_similarity in head_similarity
        ],
        dtype=np.float32,
    )


def _cluster_heads_by_complete_link_distance(layer_distance, distance_threshold):
    layer_distance = np.asarray(layer_distance, dtype=np.float64)
    head_count = int(layer_distance.shape[0])
    clusters = [[head_idx] for head_idx in range(head_count)]

    while True:
        best_pair = None
        best_distance = None
        for left_idx in range(len(clusters)):
            for right_idx in range(left_idx + 1, len(clusters)):
                pair_distances = layer_distance[np.ix_(clusters[left_idx], clusters[right_idx])]
                complete_distance = float(np.max(pair_distances))
                if complete_distance > float(distance_threshold):
                    continue
                if best_distance is None or complete_distance < best_distance:
                    best_distance = complete_distance
                    best_pair = (left_idx, right_idx)

        if best_pair is None:
            break

        left_idx, right_idx = best_pair
        merged_cluster = sorted(clusters[left_idx] + clusters[right_idx])
        clusters[left_idx] = merged_cluster
        del clusters[right_idx]

    return sorted(clusters, key=lambda cluster: cluster[0])


def _build_raw_heads_by_gqa_group(component, gqa_group_size):
    return {
        str(int(gqa_group_id)): [
            int(offset)
            for offset in range(gqa_group_size)
        ]
        for gqa_group_id in component
    }


def _build_head_mean_similarities(layer_attn, component, gqa_group_size, eps=1e-12):
    layer_attn = torch.as_tensor(layer_attn, dtype=torch.float32)
    head_mean_similarities = []
    for gqa_group_id in component:
        start_head = int(gqa_group_id * gqa_group_size)
        group_vectors = layer_attn[start_head : start_head + gqa_group_size].reshape(gqa_group_size, -1)
        mean_vector = group_vectors.mean(dim=0, keepdim=True)
        normalized_heads = torch.nn.functional.normalize(group_vectors, p=2, dim=1, eps=float(eps))
        normalized_mean = torch.nn.functional.normalize(mean_vector, p=2, dim=1, eps=float(eps))
        similarities = torch.matmul(normalized_heads, normalized_mean.transpose(0, 1)).squeeze(1).cpu().numpy()
        for head_id, similarity in enumerate(similarities):
            similarity = float(similarity)
            head_mean_similarities.append(
                {
                    "gqa_group_id": int(gqa_group_id),
                    "head_id": int(head_id),
                    "similarity": similarity,
                    "distance": float(1.0 - similarity),
                }
            )
    return head_mean_similarities


def build_attention_head_clusters(
    head_similarity,
    layer_indices,
    attn=None,
    gqa_group_size=None,
):
    head_similarity = np.asarray(head_similarity)
    if head_similarity.ndim != 3:
        raise ValueError("head_similarity must have shape [layers, heads, heads].")
    if head_similarity.shape[1] != head_similarity.shape[2]:
        raise ValueError("head_similarity must be square on the head dimensions.")
    if not np.all(np.isfinite(head_similarity)):
        raise ValueError("head_similarity must contain only finite values.")

    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    if len(layer_indices) != head_similarity.shape[0]:
        raise ValueError("layer_indices must match the head similarity layer dimension.")

    attn = None if attn is None else np.asarray(attn)
    if attn is not None:
        if attn.ndim != 4:
            raise ValueError("attn must have shape [layers, heads, query_tokens, key_tokens].")
        if attn.shape[0] != head_similarity.shape[0]:
            raise ValueError("attn must match the head similarity layer dimension.")
        if not np.all(np.isfinite(attn)):
            raise ValueError("attn must contain only finite values.")
        if gqa_group_size is None:
            raise ValueError("gqa_group_size is required when attn is provided.")
        gqa_group_size = int(gqa_group_size)
        if gqa_group_size < 1:
            raise ValueError("gqa_group_size must be at least 1.")
        if attn.shape[1] != head_similarity.shape[1] * gqa_group_size:
            raise ValueError("attn head count must equal gqa_group_count * gqa_group_size.")

    distance_thresholds = compute_attention_head_cluster_distance_thresholds(head_similarity)
    head_clusters = []
    for layer_pos, layer_idx in enumerate(layer_indices):
        layer_similarity = head_similarity[layer_pos]
        layer_distance = 1.0 - layer_similarity
        distance_threshold = float(distance_thresholds[layer_pos])
        head_count = int(layer_similarity.shape[0])
        head_components = _cluster_heads_by_complete_link_distance(layer_distance, distance_threshold)
        head_to_cluster = [-1] * head_count

        clusters = []
        for cluster_id, component in enumerate(head_components):
            for head_idx in component:
                head_to_cluster[head_idx] = cluster_id
            cluster = {
                "cluster_id": cluster_id,
                "heads": component,
            }
            if attn is None:
                cluster["pair_similarities"] = [
                    {
                        "head_i": int(component[left_pos]),
                        "head_j": int(component[right_pos]),
                        "similarity": float(layer_similarity[component[left_pos], component[right_pos]]),
                        "distance": float(layer_distance[component[left_pos], component[right_pos]]),
                    }
                    for left_pos in range(len(component))
                    for right_pos in range(left_pos + 1, len(component))
                ]
            else:
                cluster["raw_heads_by_gqa_group"] = _build_raw_heads_by_gqa_group(component, gqa_group_size)
                cluster["head_mean_similarities"] = _build_head_mean_similarities(
                    attn[layer_pos],
                    component,
                    gqa_group_size,
                )
            clusters.append(cluster)

        head_clusters.append(
            {
                "layer_idx": int(layer_idx),
                "clusters": clusters,
                "head_to_cluster": head_to_cluster,
                "distance_threshold": distance_threshold,
                "threshold_mode": ATTN_HEAD_CLUSTER_THRESHOLD_MODE,
                "outlier_method": ATTN_HEAD_CLUSTER_OUTLIER_METHOD,
                "distance_metric": ATTN_HEAD_CLUSTER_DISTANCE_METRIC,
            }
        )

    return head_clusters


def build_attention_head_cluster_payload(
    head_clusters,
    layer_indices,
    head_similarity_info,
):
    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    cluster_layers = list(head_clusters)
    return {
        "schema_version": 1,
        "cluster_scope": "per_layer",
        "unit": "gqa_group",
        "head_mean_similarity_unit": "raw_head_to_gqa_group_mean_before_grouping",
        "head_mean_similarity_metric": ATTN_HEAD_MEAN_SIMILARITY_METRIC,
        "head_mean_similarity_reduction": ATTN_HEAD_MEAN_SIMILARITY_REDUCTION,
        "attention_head_count": int(head_similarity_info["attention_head_count"]),
        "gqa_group_count": int(head_similarity_info["gqa_group_count"]),
        "gqa_group_size": int(head_similarity_info["gqa_group_size"]),
        "layer_indices": layer_indices,
        "distance_metric": ATTN_HEAD_CLUSTER_DISTANCE_METRIC,
        "threshold_mode": ATTN_HEAD_CLUSTER_THRESHOLD_MODE,
        "outlier_method": ATTN_HEAD_CLUSTER_OUTLIER_METHOD,
        "distance_threshold_by_layer": {
            str(layer_cluster["layer_idx"]): float(layer_cluster["distance_threshold"])
            for layer_cluster in cluster_layers
        },
        "cluster_count_by_layer": {
            str(layer_cluster["layer_idx"]): len(layer_cluster["clusters"])
            for layer_cluster in cluster_layers
        },
        "head_to_cluster_by_layer": {
            str(layer_cluster["layer_idx"]): list(layer_cluster["head_to_cluster"])
            for layer_cluster in cluster_layers
        },
        "clusters_by_layer": {
            str(layer_cluster["layer_idx"]): layer_cluster["clusters"]
            for layer_cluster in cluster_layers
        },
    }


def get_attention_gqa_group_count(model):
    text_config = get_text_config(model)
    num_key_value_heads = getattr(text_config, "num_key_value_heads", None)
    if num_key_value_heads is None:
        return None
    try:
        num_key_value_heads = int(num_key_value_heads)
    except (TypeError, ValueError):
        return None
    return num_key_value_heads if num_key_value_heads > 0 else None


def get_attention_layer_indices(model):
    layers = get_decoder_layers(model)
    return [
        int(layer_idx)
        for layer_idx, layer in enumerate(layers)
        if getattr(layer, "self_attn", None) is not None
    ]


def extract_attention_weights(output):
    if isinstance(output, (tuple, list)) and len(output) >= 2 and torch.is_tensor(output[1]):
        return output[1]
    if isinstance(output, dict):
        value = output.get("attn_weights")
        if value is None:
            value = output.get("attention_weights")
        if torch.is_tensor(value):
            return value
    return None


def capture_attention_layers(model, inputs, attention_layers):
    layers = get_decoder_layers(model)
    if not layers:
        raise ValueError("No decoder layers were found; cannot capture attention layer similarity.")

    expected_layers = [int(layer_idx) for layer_idx in attention_layers]
    recorder = AttentionCaptureRecorder(expected_layers)
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _args, output):
            attn_weights = extract_attention_weights(output)
            recorder.record_attention(layer_idx, attn_weights)

        return hook

    try:
        for layer_idx in expected_layers:
            if layer_idx < 0 or layer_idx >= len(layers):
                continue
            attn_module = getattr(layers[layer_idx], "self_attn", None)
            if attn_module is None:
                continue
            handles.append(attn_module.register_forward_hook(make_hook(layer_idx)))
        run_observation_forward(model, inputs, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()

    return recorder.export()


def plot_attention_layer_similarity_heatmap(
    similarity,
    layer_indices,
    output_path,
    title,
    vmin=-1.0,
    vmax=1.0,
):
    setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = os.path.abspath(os.path.expanduser(str(output_path)))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_layers = len(layer_indices)
    fig_size = max(6.0, min(14.0, 0.28 * num_layers + 3.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=160)
    image = ax.imshow(similarity, cmap="coolwarm", vmin=float(vmin), vmax=float(vmax), interpolation="nearest")
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
    colorbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_attention_head_similarity_heatmaps(
    head_similarity,
    layer_indices,
    output_dir,
    file_prefix,
    title_prefix,
    vmin=-1.0,
    vmax=1.0,
):
    head_similarity = np.asarray(head_similarity)
    if head_similarity.ndim != 3:
        raise ValueError("head_similarity must have shape [layers, heads, heads].")
    if head_similarity.shape[1] != head_similarity.shape[2]:
        raise ValueError("head_similarity must be square on the head dimensions.")

    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    if len(layer_indices) != head_similarity.shape[0]:
        raise ValueError("layer_indices must match the head similarity layer dimension.")

    setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    num_groups = int(head_similarity.shape[1])
    if num_groups < 1:
        raise ValueError("head_similarity must contain at least one GQA group.")
    fig_size = max(5.5, min(12.0, 0.3 * num_groups + 3.0))
    tick_step = max(1, num_groups // 16)
    tick_positions = np.arange(0, num_groups, tick_step)
    tick_labels = [str(group_idx) for group_idx in tick_positions]

    for layer_pos, layer_idx in enumerate(layer_indices):
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=160)
        image = ax.imshow(
            head_similarity[layer_pos],
            cmap="coolwarm",
            vmin=float(vmin),
            vmax=float(vmax),
            interpolation="nearest",
        )
        ax.set_title(f"{title_prefix}: layer {layer_idx} GQA group similarity")
        ax.set_xlabel("GQA group id")
        ax.set_ylabel("GQA group id")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Cosine similarity")
        fig.tight_layout()
        file_name = f"{file_prefix}_layer_{layer_idx:04d}_gqa_head_similarity.png"
        fig.savefig(os.path.join(output_dir, file_name))
        plt.close(fig)
        saved_files.append(file_name)

    return saved_files


def plot_raw_attention_head_similarity_heatmaps(
    raw_head_similarity,
    layer_indices,
    output_dir,
    file_prefix,
    title_prefix,
    vmin=-1.0,
    vmax=1.0,
):
    raw_head_similarity = np.asarray(raw_head_similarity)
    if raw_head_similarity.ndim != 3:
        raise ValueError("raw_head_similarity must have shape [layers, heads, heads].")
    if raw_head_similarity.shape[1] != raw_head_similarity.shape[2]:
        raise ValueError("raw_head_similarity must be square on the head dimensions.")

    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    if len(layer_indices) != raw_head_similarity.shape[0]:
        raise ValueError("layer_indices must match the raw head similarity layer dimension.")

    setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    head_count = int(raw_head_similarity.shape[1])
    if head_count < 1:
        raise ValueError("raw_head_similarity must contain at least one attention head.")
    fig_size = max(5.5, min(12.0, 0.3 * head_count + 3.0))
    tick_step = max(1, head_count // 16)
    tick_positions = np.arange(0, head_count, tick_step)
    tick_labels = [str(head_idx) for head_idx in tick_positions]

    for layer_pos, layer_idx in enumerate(layer_indices):
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=160)
        image = ax.imshow(
            raw_head_similarity[layer_pos],
            cmap="coolwarm",
            vmin=float(vmin),
            vmax=float(vmax),
            interpolation="nearest",
        )
        ax.set_title(f"{title_prefix}: layer {layer_idx} raw head similarity before GQA grouping")
        ax.set_xlabel("Attention head id")
        ax.set_ylabel("Attention head id")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Cosine similarity")
        fig.tight_layout()
        file_name = f"{file_prefix}_layer_{layer_idx:04d}_raw_head_similarity.png"
        fig.savefig(os.path.join(output_dir, file_name))
        plt.close(fig)
        saved_files.append(file_name)

    return saved_files


class AttentionLayerSimilarityRunWriter:
    def __init__(
        self,
        root_dir,
        model_name,
        out_file,
        attention_layers,
        max_prefill_tokens,
        window_size=8,
        heatmap_vmin=-1.0,
        heatmap_vmax=1.0,
        gqa_group_count=None,
        head_cluster_mode=False,
    ):
        self.root_dir = root_dir
        self.model_name = model_name
        self.out_file = os.path.abspath(out_file)
        self.attention_layers = [int(layer_idx) for layer_idx in attention_layers]
        self.max_prefill_tokens = int(max_prefill_tokens) if max_prefill_tokens is not None else None
        self.window_size = int(window_size)
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1.")
        self.heatmap_vmin = float(heatmap_vmin)
        self.heatmap_vmax = float(heatmap_vmax)
        self.gqa_group_count = int(gqa_group_count) if gqa_group_count is not None else None
        self.head_cluster_mode = bool(head_cluster_mode)
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
        return AttentionLayerSimilaritySampleWriter(self, item, sample_index)

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
            "attention_layers": self.attention_layers,
            "tail_window_attention_scope": ATTN_LAYER_SIMILARITY_QUERY_TOKEN_SCOPE,
            "tail_window_attention_size": self.window_size,
            "attn_layer_similarity_metric": ATTN_LAYER_SIMILARITY_METRIC,
            "attn_layer_similarity_reduction": ATTN_LAYER_SIMILARITY_REDUCTION,
            "attn_layer_similarity_heatmap_vmin": self.heatmap_vmin,
            "attn_layer_similarity_heatmap_vmax": self.heatmap_vmax,
            "attn_head_similarity_metric": ATTN_HEAD_SIMILARITY_METRIC,
            "attn_head_similarity_reduction": ATTN_HEAD_SIMILARITY_REDUCTION,
            "attn_head_similarity_heatmap_vmin": self.heatmap_vmin,
            "attn_head_similarity_heatmap_vmax": self.heatmap_vmax,
            "attn_head_similarity_gqa_group_count": self.gqa_group_count,
            "attn_raw_head_similarity_metric": ATTN_RAW_HEAD_SIMILARITY_METRIC,
            "attn_raw_head_similarity_reduction": ATTN_RAW_HEAD_SIMILARITY_REDUCTION,
            "attn_raw_head_similarity_heatmap_vmin": self.heatmap_vmin,
            "attn_raw_head_similarity_heatmap_vmax": self.heatmap_vmax,
            "attn_head_cluster_mode": self.head_cluster_mode,
            "attn_head_cluster_threshold_mode": ATTN_HEAD_CLUSTER_THRESHOLD_MODE,
            "attn_head_cluster_outlier_method": ATTN_HEAD_CLUSTER_OUTLIER_METHOD,
            "attn_head_cluster_distance_metric": ATTN_HEAD_CLUSTER_DISTANCE_METRIC,
            "attn_layer_similarity_max_prefill_tokens": self.max_prefill_tokens,
            "attn_layer_similarity_prefill_cap_mode": "fixed" if self.max_prefill_tokens is not None else "none",
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as fout:
            json.dump(manifest, fout, ensure_ascii=False, indent=2)


class AttentionLayerSimilaritySampleWriter:
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
            "token_ids": input_ids,
            "tokens": build_token_entries(tokenizer, input_ids),
            "attn_layer_similarity_metric": ATTN_LAYER_SIMILARITY_METRIC,
            "attn_layer_similarity_reduction": ATTN_LAYER_SIMILARITY_REDUCTION,
            "tail_window_attention_scope": ATTN_LAYER_SIMILARITY_QUERY_TOKEN_SCOPE,
            "tail_window_attention_size": self.run_writer.window_size,
            "tail_window_attention_query_token_start": None,
            "tail_window_attention_query_token_count": None,
            "attn_layer_similarity_heatmap_vmin": self.run_writer.heatmap_vmin,
            "attn_layer_similarity_heatmap_vmax": self.run_writer.heatmap_vmax,
            "attn_head_similarity_metric": ATTN_HEAD_SIMILARITY_METRIC,
            "attn_head_similarity_reduction": ATTN_HEAD_SIMILARITY_REDUCTION,
            "attn_head_similarity_heatmap_vmin": self.run_writer.heatmap_vmin,
            "attn_head_similarity_heatmap_vmax": self.run_writer.heatmap_vmax,
            "attn_head_similarity_gqa_group_count": self.run_writer.gqa_group_count,
            "attn_raw_head_similarity_metric": ATTN_RAW_HEAD_SIMILARITY_METRIC,
            "attn_raw_head_similarity_reduction": ATTN_RAW_HEAD_SIMILARITY_REDUCTION,
            "attn_raw_head_similarity_heatmap_vmin": self.run_writer.heatmap_vmin,
            "attn_raw_head_similarity_heatmap_vmax": self.run_writer.heatmap_vmax,
            "similarity_file": None,
            "heatmap_file": None,
            "head_similarity_heatmap_dir": None,
            "head_similarity_heatmap_files": [],
            "head_similarity_heatmap_count": 0,
            "raw_head_similarity_heatmap_dir": None,
            "raw_head_similarity_heatmap_files": [],
            "raw_head_similarity_heatmap_count": 0,
            "tail_window_heatmap_file": None,
            "tail_window_head_similarity_heatmap_dir": None,
            "tail_window_head_similarity_heatmap_files": [],
            "tail_window_head_similarity_heatmap_count": 0,
            "tail_window_raw_head_similarity_heatmap_dir": None,
            "tail_window_raw_head_similarity_heatmap_files": [],
            "tail_window_raw_head_similarity_heatmap_count": 0,
            "attn_head_cluster_mode": self.run_writer.head_cluster_mode,
            "head_cluster_distance_threshold_by_layer": {},
            "head_cluster_threshold_mode": ATTN_HEAD_CLUSTER_THRESHOLD_MODE,
            "head_cluster_outlier_method": ATTN_HEAD_CLUSTER_OUTLIER_METHOD,
            "head_cluster_distance_metric": ATTN_HEAD_CLUSTER_DISTANCE_METRIC,
            "head_cluster_file": None,
            "head_cluster_count_by_layer": {},
            "similarity_shape": None,
            "head_similarity_shape": None,
            "raw_head_similarity_shape": None,
            "tail_window_similarity_shape": None,
            "tail_window_head_similarity_shape": None,
            "tail_window_raw_head_similarity_shape": None,
            "attention_head_count": None,
            "raw_head_count": None,
            "gqa_group_count": None,
            "gqa_group_size": None,
            "attn_shape": None,
            "tail_window_attn_shape": None,
            "layer_indices": [],
            "missing_layers": [],
        }
        self.prefills.append(record)
        self._write_sample_json()

        if prefill_cap is not None and len(input_ids) > prefill_cap:
            record["status"] = "skipped_over_cap"
            record["reason"] = f"Prompt token count {len(input_ids)} exceeds cap {prefill_cap}."
            self._write_sample_json()
            return record

        try:
            attn, layer_indices, missing_layers = capture_attention_layers(
                model=model,
                inputs=inputs,
                attention_layers=self.run_writer.attention_layers,
            )
            window_attn, query_token_start, query_token_count = select_attention_query_window(
                attn,
                self.run_writer.window_size,
            )
            similarity, layer_indices = build_attention_layer_similarity(attn, layer_indices)
            head_similarity, layer_indices, head_similarity_info = build_attention_head_similarity(
                attn,
                layer_indices,
                gqa_group_count=self.run_writer.gqa_group_count,
            )
            raw_head_similarity, layer_indices, raw_head_similarity_info = build_raw_attention_head_similarity(
                attn,
                layer_indices,
            )
            tail_window_similarity, _ = build_attention_layer_similarity(window_attn, layer_indices)
            tail_window_head_similarity, _, _ = build_attention_head_similarity(
                window_attn,
                layer_indices,
                gqa_group_count=self.run_writer.gqa_group_count,
            )
            tail_window_raw_head_similarity, _, _ = build_raw_attention_head_similarity(
                window_attn,
                layer_indices,
            )
            head_clusters = []
            head_cluster_payload = None
            head_cluster_count_by_layer = {}
            head_cluster_distance_threshold_by_layer = {}
            if self.run_writer.head_cluster_mode:
                head_clusters = build_attention_head_clusters(
                    head_similarity,
                    layer_indices,
                    attn=attn,
                    gqa_group_size=head_similarity_info["gqa_group_size"],
                )
                head_cluster_payload = build_attention_head_cluster_payload(
                    head_clusters,
                    layer_indices,
                    head_similarity_info,
                )
                head_cluster_distance_threshold_by_layer = head_cluster_payload["distance_threshold_by_layer"]
                head_cluster_count_by_layer = head_cluster_payload["cluster_count_by_layer"]
            head_cluster_distance_thresholds = np.asarray(
                [
                    head_cluster_distance_threshold_by_layer.get(str(layer_idx), np.nan)
                    for layer_idx in layer_indices
                ],
                dtype=np.float32,
            )

            similarity_file_name = f"prefill_{prefill_index:03d}_attn_layer_similarity.npz"
            head_cluster_file_name = f"prefill_{prefill_index:03d}_attn_head_clusters.json"
            heatmap_file_name = f"prefill_{prefill_index:03d}_attn_layer_similarity.png"
            head_similarity_heatmap_dir_name = f"prefill_{prefill_index:03d}_gqa_head_similarity"
            raw_head_similarity_heatmap_dir_name = f"prefill_{prefill_index:03d}_raw_head_similarity"
            tail_window_heatmap_file_name = f"prefill_{prefill_index:03d}_tail_window_attn_layer_similarity.png"
            tail_window_head_similarity_heatmap_dir_name = f"prefill_{prefill_index:03d}_tail_window_gqa_head_similarity"
            tail_window_raw_head_similarity_heatmap_dir_name = f"prefill_{prefill_index:03d}_tail_window_raw_head_similarity"
            similarity_path = os.path.join(self.sample_dir, similarity_file_name)
            head_cluster_path = os.path.join(self.sample_dir, head_cluster_file_name)
            heatmap_path = os.path.join(self.sample_dir, heatmap_file_name)
            head_similarity_heatmap_dir = os.path.join(self.sample_dir, head_similarity_heatmap_dir_name)
            raw_head_similarity_heatmap_dir = os.path.join(self.sample_dir, raw_head_similarity_heatmap_dir_name)
            tail_window_heatmap_path = os.path.join(self.sample_dir, tail_window_heatmap_file_name)
            tail_window_head_similarity_heatmap_dir = os.path.join(
                self.sample_dir, tail_window_head_similarity_heatmap_dir_name
            )
            tail_window_raw_head_similarity_heatmap_dir = os.path.join(
                self.sample_dir, tail_window_raw_head_similarity_heatmap_dir_name
            )
            np.savez_compressed(
                similarity_path,
                similarity=similarity,
                head_similarity=head_similarity,
                raw_head_similarity=raw_head_similarity,
                tail_window_similarity=tail_window_similarity,
                tail_window_head_similarity=tail_window_head_similarity,
                tail_window_raw_head_similarity=tail_window_raw_head_similarity,
                layer_indices=np.asarray(layer_indices, dtype=np.int16),
                attn_shape=np.asarray(attn.shape, dtype=np.int64),
                tail_window_attn_shape=np.asarray(window_attn.shape, dtype=np.int64),
                tail_window_attention_scope=np.asarray(ATTN_LAYER_SIMILARITY_QUERY_TOKEN_SCOPE),
                tail_window_attention_size=np.asarray(self.run_writer.window_size, dtype=np.int64),
                tail_window_attention_query_token_start=np.asarray(query_token_start, dtype=np.int64),
                tail_window_attention_query_token_count=np.asarray(query_token_count, dtype=np.int64),
                attn_layer_similarity_metric=np.asarray(ATTN_LAYER_SIMILARITY_METRIC),
                attn_layer_similarity_reduction=np.asarray(ATTN_LAYER_SIMILARITY_REDUCTION),
                attn_layer_similarity_heatmap_vmin=np.asarray(self.run_writer.heatmap_vmin, dtype=np.float32),
                attn_layer_similarity_heatmap_vmax=np.asarray(self.run_writer.heatmap_vmax, dtype=np.float32),
                attn_head_similarity_metric=np.asarray(ATTN_HEAD_SIMILARITY_METRIC),
                attn_head_similarity_reduction=np.asarray(ATTN_HEAD_SIMILARITY_REDUCTION),
                attn_head_similarity_heatmap_vmin=np.asarray(self.run_writer.heatmap_vmin, dtype=np.float32),
                attn_head_similarity_heatmap_vmax=np.asarray(self.run_writer.heatmap_vmax, dtype=np.float32),
                attn_raw_head_similarity_metric=np.asarray(ATTN_RAW_HEAD_SIMILARITY_METRIC),
                attn_raw_head_similarity_reduction=np.asarray(ATTN_RAW_HEAD_SIMILARITY_REDUCTION),
                attn_raw_head_similarity_heatmap_vmin=np.asarray(self.run_writer.heatmap_vmin, dtype=np.float32),
                attn_raw_head_similarity_heatmap_vmax=np.asarray(self.run_writer.heatmap_vmax, dtype=np.float32),
                attn_head_cluster_mode=np.asarray(self.run_writer.head_cluster_mode, dtype=np.bool_),
                attn_head_cluster_distance_thresholds=head_cluster_distance_thresholds,
                attn_head_cluster_threshold_mode=np.asarray(ATTN_HEAD_CLUSTER_THRESHOLD_MODE),
                attn_head_cluster_outlier_method=np.asarray(ATTN_HEAD_CLUSTER_OUTLIER_METHOD or ""),
                attn_head_cluster_distance_metric=np.asarray(ATTN_HEAD_CLUSTER_DISTANCE_METRIC),
                attention_head_count=np.asarray(head_similarity_info["attention_head_count"], dtype=np.int16),
                raw_head_count=np.asarray(raw_head_similarity_info["raw_head_count"], dtype=np.int16),
                gqa_group_count=np.asarray(head_similarity_info["gqa_group_count"], dtype=np.int16),
                gqa_group_size=np.asarray(head_similarity_info["gqa_group_size"], dtype=np.int16),
                token_ids=np.asarray(input_ids, dtype=np.int64),
            )
            if head_cluster_payload is not None:
                with open(head_cluster_path, "w", encoding="utf-8") as fout:
                    json.dump(head_cluster_payload, fout, ensure_ascii=False, indent=2)

            record["status"] = "saved"
            record["similarity_file"] = similarity_file_name
            if head_cluster_payload is not None:
                record["head_cluster_file"] = head_cluster_file_name
            record["similarity_shape"] = list(similarity.shape)
            record["head_similarity_shape"] = list(head_similarity.shape)
            record["raw_head_similarity_shape"] = list(raw_head_similarity.shape)
            record["tail_window_similarity_shape"] = list(tail_window_similarity.shape)
            record["tail_window_head_similarity_shape"] = list(tail_window_head_similarity.shape)
            record["tail_window_raw_head_similarity_shape"] = list(tail_window_raw_head_similarity.shape)
            record["attention_head_count"] = head_similarity_info["attention_head_count"]
            record["raw_head_count"] = raw_head_similarity_info["raw_head_count"]
            record["gqa_group_count"] = head_similarity_info["gqa_group_count"]
            record["gqa_group_size"] = head_similarity_info["gqa_group_size"]
            record["head_cluster_distance_threshold_by_layer"] = head_cluster_distance_threshold_by_layer
            record["head_cluster_count_by_layer"] = head_cluster_count_by_layer
            record["attn_shape"] = list(attn.shape)
            record["tail_window_attn_shape"] = list(window_attn.shape)
            record["tail_window_attention_query_token_start"] = query_token_start
            record["tail_window_attention_query_token_count"] = query_token_count
            record["layer_indices"] = layer_indices
            record["missing_layers"] = missing_layers
            try:
                plot_attention_layer_similarity_heatmap(
                    similarity=similarity,
                    layer_indices=layer_indices,
                    output_path=heatmap_path,
                    title=(
                        f"{label}: attention layer similarity "
                        f"({ATTN_LAYER_SIMILARITY_REDUCTION}, {ATTN_LAYER_SIMILARITY_METRIC}, "
                        f"{len(input_ids)} tokens, range "
                        f"{self.run_writer.heatmap_vmin:g}..{self.run_writer.heatmap_vmax:g})"
                    ),
                    vmin=self.run_writer.heatmap_vmin,
                    vmax=self.run_writer.heatmap_vmax,
                )
                record["heatmap_file"] = heatmap_file_name
            except Exception as plot_exc:
                record["status"] = "saved_npz_plot_error"
                record["plot_error"] = str(plot_exc)
            try:
                head_similarity_heatmap_files = plot_attention_head_similarity_heatmaps(
                    head_similarity=head_similarity,
                    layer_indices=layer_indices,
                    output_dir=head_similarity_heatmap_dir,
                    file_prefix=f"prefill_{prefill_index:03d}",
                    title_prefix=f"{label} ({len(input_ids)} tokens)",
                    vmin=self.run_writer.heatmap_vmin,
                    vmax=self.run_writer.heatmap_vmax,
                )
                record["head_similarity_heatmap_dir"] = head_similarity_heatmap_dir_name
                record["head_similarity_heatmap_files"] = [
                    os.path.join(head_similarity_heatmap_dir_name, file_name)
                    for file_name in head_similarity_heatmap_files
                ]
                record["head_similarity_heatmap_count"] = len(head_similarity_heatmap_files)
            except Exception as head_plot_exc:
                if record["status"] == "saved":
                    record["status"] = "saved_npz_head_plot_error"
                elif record["status"] == "saved_npz_plot_error":
                    record["status"] = "saved_npz_plot_errors"
                record["head_similarity_heatmap_error"] = str(head_plot_exc)
            try:
                raw_head_similarity_heatmap_files = plot_raw_attention_head_similarity_heatmaps(
                    raw_head_similarity=raw_head_similarity,
                    layer_indices=layer_indices,
                    output_dir=raw_head_similarity_heatmap_dir,
                    file_prefix=f"prefill_{prefill_index:03d}",
                    title_prefix=f"{label} ({len(input_ids)} tokens)",
                    vmin=self.run_writer.heatmap_vmin,
                    vmax=self.run_writer.heatmap_vmax,
                )
                record["raw_head_similarity_heatmap_dir"] = raw_head_similarity_heatmap_dir_name
                record["raw_head_similarity_heatmap_files"] = [
                    os.path.join(raw_head_similarity_heatmap_dir_name, file_name)
                    for file_name in raw_head_similarity_heatmap_files
                ]
                record["raw_head_similarity_heatmap_count"] = len(raw_head_similarity_heatmap_files)
            except Exception as raw_head_plot_exc:
                if record["status"] == "saved":
                    record["status"] = "saved_npz_raw_head_plot_error"
                elif record["status"] in {"saved_npz_plot_error", "saved_npz_head_plot_error"}:
                    record["status"] = "saved_npz_plot_errors"
                record["raw_head_similarity_heatmap_error"] = str(raw_head_plot_exc)
            try:
                plot_attention_layer_similarity_heatmap(
                    similarity=tail_window_similarity,
                    layer_indices=layer_indices,
                    output_path=tail_window_heatmap_path,
                    title=(
                        f"{label}: tail-window attention layer similarity "
                        f"({ATTN_LAYER_SIMILARITY_REDUCTION}, {ATTN_LAYER_SIMILARITY_METRIC}, "
                        f"last {query_token_count} query tokens of {len(input_ids)}, range "
                        f"{self.run_writer.heatmap_vmin:g}..{self.run_writer.heatmap_vmax:g})"
                    ),
                    vmin=self.run_writer.heatmap_vmin,
                    vmax=self.run_writer.heatmap_vmax,
                )
                record["tail_window_heatmap_file"] = tail_window_heatmap_file_name
            except Exception as tail_window_plot_exc:
                record["tail_window_heatmap_error"] = str(tail_window_plot_exc)
            try:
                tail_window_head_similarity_heatmap_files = plot_attention_head_similarity_heatmaps(
                    head_similarity=tail_window_head_similarity,
                    layer_indices=layer_indices,
                    output_dir=tail_window_head_similarity_heatmap_dir,
                    file_prefix=f"prefill_{prefill_index:03d}_tail_window",
                    title_prefix=f"{label} (last {query_token_count} query tokens of {len(input_ids)})",
                    vmin=self.run_writer.heatmap_vmin,
                    vmax=self.run_writer.heatmap_vmax,
                )
                record["tail_window_head_similarity_heatmap_dir"] = tail_window_head_similarity_heatmap_dir_name
                record["tail_window_head_similarity_heatmap_files"] = [
                    os.path.join(tail_window_head_similarity_heatmap_dir_name, file_name)
                    for file_name in tail_window_head_similarity_heatmap_files
                ]
                record["tail_window_head_similarity_heatmap_count"] = len(
                    tail_window_head_similarity_heatmap_files
                )
            except Exception as tail_window_head_plot_exc:
                record["tail_window_head_similarity_heatmap_error"] = str(tail_window_head_plot_exc)
            try:
                tail_window_raw_head_similarity_heatmap_files = plot_raw_attention_head_similarity_heatmaps(
                    raw_head_similarity=tail_window_raw_head_similarity,
                    layer_indices=layer_indices,
                    output_dir=tail_window_raw_head_similarity_heatmap_dir,
                    file_prefix=f"prefill_{prefill_index:03d}_tail_window",
                    title_prefix=f"{label} (last {query_token_count} query tokens of {len(input_ids)})",
                    vmin=self.run_writer.heatmap_vmin,
                    vmax=self.run_writer.heatmap_vmax,
                )
                record["tail_window_raw_head_similarity_heatmap_dir"] = tail_window_raw_head_similarity_heatmap_dir_name
                record["tail_window_raw_head_similarity_heatmap_files"] = [
                    os.path.join(tail_window_raw_head_similarity_heatmap_dir_name, file_name)
                    for file_name in tail_window_raw_head_similarity_heatmap_files
                ]
                record["tail_window_raw_head_similarity_heatmap_count"] = len(
                    tail_window_raw_head_similarity_heatmap_files
                )
            except Exception as tail_window_raw_head_plot_exc:
                record["tail_window_raw_head_similarity_heatmap_error"] = str(tail_window_raw_head_plot_exc)
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
            "attn_layer_similarity_metric",
            "attn_layer_similarity_reduction",
            "tail_window_attention_scope",
            "tail_window_attention_size",
            "tail_window_attention_query_token_start",
            "tail_window_attention_query_token_count",
            "attn_layer_similarity_heatmap_vmin",
            "attn_layer_similarity_heatmap_vmax",
            "attn_head_similarity_metric",
            "attn_head_similarity_reduction",
            "attn_head_similarity_heatmap_vmin",
            "attn_head_similarity_heatmap_vmax",
            "attn_head_similarity_gqa_group_count",
            "attn_raw_head_similarity_metric",
            "attn_raw_head_similarity_reduction",
            "attn_raw_head_similarity_heatmap_vmin",
            "attn_raw_head_similarity_heatmap_vmax",
            "similarity_shape",
            "head_similarity_shape",
            "raw_head_similarity_shape",
            "tail_window_attn_shape",
            "tail_window_similarity_shape",
            "tail_window_head_similarity_shape",
            "tail_window_raw_head_similarity_shape",
            "attention_head_count",
            "raw_head_count",
            "gqa_group_count",
            "gqa_group_size",
            "heatmap_file",
            "head_similarity_heatmap_dir",
            "head_similarity_heatmap_files",
            "head_similarity_heatmap_count",
            "raw_head_similarity_heatmap_dir",
            "raw_head_similarity_heatmap_files",
            "raw_head_similarity_heatmap_count",
            "tail_window_heatmap_file",
            "tail_window_head_similarity_heatmap_dir",
            "tail_window_head_similarity_heatmap_files",
            "tail_window_head_similarity_heatmap_count",
            "tail_window_raw_head_similarity_heatmap_dir",
            "tail_window_raw_head_similarity_heatmap_files",
            "tail_window_raw_head_similarity_heatmap_count",
            "attn_head_cluster_mode",
            "head_cluster_distance_threshold_by_layer",
            "head_cluster_threshold_mode",
            "head_cluster_outlier_method",
            "head_cluster_distance_metric",
            "head_cluster_file",
            "head_cluster_count_by_layer",
            "reason",
            "error",
            "plot_error",
            "head_similarity_heatmap_error",
            "raw_head_similarity_heatmap_error",
            "tail_window_heatmap_error",
            "tail_window_head_similarity_heatmap_error",
            "tail_window_raw_head_similarity_heatmap_error",
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
