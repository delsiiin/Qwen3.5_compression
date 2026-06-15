import argparse
import json
import os

import numpy as np


GROUP_SCHEME_SIMILARITY = "similarity"
GROUP_SCHEME_ATTN_LAYER_SIMILARITY = "attn_layer_similarity"
GROUP_SCHEME_KEY_PCA_ANGLE = "key_pca_angle"
GROUP_SCHEME_KEY_PCA_DISTANCE = "key_pca_distance"
SUPPORTED_GROUP_SCHEMES = (
    GROUP_SCHEME_SIMILARITY,
    GROUP_SCHEME_ATTN_LAYER_SIMILARITY,
    GROUP_SCHEME_KEY_PCA_ANGLE,
    GROUP_SCHEME_KEY_PCA_DISTANCE,
)
SIMILARITY_SOURCE_QUERY_WINDOW = "query_window_similarity"
SIMILARITY_SOURCE_ATTN_LAYER = "attn_layer_similarity"
GROUP_THRESHOLD_MODE_FIXED = "fixed"
GROUP_THRESHOLD_MODE_MEAN_WITHOUT_OUTLIERS = "mean_without_outliers"
SUPPORTED_GROUP_THRESHOLD_MODES = (
    GROUP_THRESHOLD_MODE_FIXED,
    GROUP_THRESHOLD_MODE_MEAN_WITHOUT_OUTLIERS,
)
KEY_PCA_ANGLE_THRESHOLD_MODE_FIXED = "fixed"
KEY_PCA_ANGLE_THRESHOLD_MODE_MEAN_WITHOUT_OUTLIERS = "mean_without_outliers"
SUPPORTED_KEY_PCA_ANGLE_THRESHOLD_MODES = (
    KEY_PCA_ANGLE_THRESHOLD_MODE_FIXED,
    KEY_PCA_ANGLE_THRESHOLD_MODE_MEAN_WITHOUT_OUTLIERS,
)
KEY_PCA_DISTANCE_THRESHOLD_MODE_FIXED = "fixed"
KEY_PCA_DISTANCE_THRESHOLD_MODE_MEAN_WITHOUT_OUTLIERS = "mean_without_outliers"
SUPPORTED_KEY_PCA_DISTANCE_THRESHOLD_MODES = (
    KEY_PCA_DISTANCE_THRESHOLD_MODE_FIXED,
    KEY_PCA_DISTANCE_THRESHOLD_MODE_MEAN_WITHOUT_OUTLIERS,
)
GROUP_THRESHOLD_OUTLIER_METHOD_IQR = "iqr_1.5"
KEY_PCA_ANGLE_THRESHOLD_OUTLIER_METHOD_IQR = "iqr_1.5"
KEY_PCA_DISTANCE_THRESHOLD_OUTLIER_METHOD_IQR = "iqr_1.5"


def _scalar_from_npz(value, default=None):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return default


def _string_from_npz(value, default=None):
    scalar = _scalar_from_npz(value, default)
    if scalar is None:
        return default
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def load_similarity_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "similarity" not in data or "layer_indices" not in data:
            raise ValueError("similarity npz must contain 'similarity' and 'layer_indices'.")
        similarity = np.asarray(data["similarity"], dtype=np.float64)
        layer_indices = np.asarray(data["layer_indices"], dtype=np.int64)
        actual_window_size = _scalar_from_npz(data.get("actual_window_size"))
        if "attn_layer_similarity_metric" in data or "attn_layer_similarity_reduction" in data:
            similarity_source = SIMILARITY_SOURCE_ATTN_LAYER
            similarity_state = SIMILARITY_SOURCE_ATTN_LAYER
            similarity_metric = _string_from_npz(data.get("attn_layer_similarity_metric"), "cosine_similarity")
            similarity_reduction = _string_from_npz(
                data.get("attn_layer_similarity_reduction"),
                "mean_heads_flatten_attention",
            )
        else:
            similarity_source = SIMILARITY_SOURCE_QUERY_WINDOW
            similarity_state = _string_from_npz(data.get("query_window_similarity_state"), "hidden_states")
            similarity_metric = _string_from_npz(data.get("query_window_similarity_metric"), "cosine_similarity")
            similarity_reduction = _string_from_npz(data.get("similarity_reduction"), "flatten_window")

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix.")
    if layer_indices.ndim != 1 or layer_indices.shape[0] != similarity.shape[0]:
        raise ValueError("layer_indices must be 1D and match similarity size.")
    if similarity.shape[0] < 1:
        raise ValueError("similarity must contain at least one layer.")
    if not np.all(np.isfinite(similarity)):
        raise ValueError("similarity contains non-finite values.")
    if len(set(int(layer) for layer in layer_indices.tolist())) != layer_indices.shape[0]:
        raise ValueError("layer_indices contains duplicate layers.")

    return (
        similarity,
        layer_indices.astype(np.int64, copy=False),
        actual_window_size,
        str(similarity_state),
        str(similarity_source),
        str(similarity_metric),
        str(similarity_reduction),
    )


def load_attn_output_ratio_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "ratios" not in data or "layer_indices" not in data:
            raise ValueError("attn output ratio npz must contain 'ratios' and 'layer_indices'.")
        ratios = np.asarray(data["ratios"], dtype=np.float64)
        layer_indices = np.asarray(data["layer_indices"], dtype=np.int64)

    if ratios.ndim != 2:
        raise ValueError("ratios must be a 2D array.")
    if layer_indices.ndim != 1 or layer_indices.shape[0] != ratios.shape[0]:
        raise ValueError("layer_indices must be 1D and match ratios first dimension.")
    if ratios.shape[0] < 1:
        raise ValueError("ratios must contain at least one layer.")
    if not np.all(np.isfinite(ratios)):
        raise ValueError("ratios contains non-finite values.")
    if len(set(int(layer) for layer in layer_indices.tolist())) != layer_indices.shape[0]:
        raise ValueError("layer_indices contains duplicate layers.")

    return ratios, layer_indices.astype(np.int64, copy=False)


def compute_adjacent_angles_from_centers(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError("key_layer_centers must have shape [layer, 2].")
    if centers.shape[0] < 2:
        return np.empty((0,), dtype=np.float64)

    angles = np.full((centers.shape[0] - 1,), np.nan, dtype=np.float64)
    for pos in range(centers.shape[0] - 1):
        first = centers[pos]
        second = centers[pos + 1]
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            continue
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        if first_norm <= 1e-8 or second_norm <= 1e-8:
            continue
        cosine = float(np.dot(first, second) / (first_norm * second_norm))
        cosine = max(-1.0, min(1.0, cosine))
        angles[pos] = np.degrees(np.arccos(cosine))
    return angles


def compute_adjacent_distances_from_centers(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError("key_layer_centers must have shape [layer, 2].")
    if centers.shape[0] < 2:
        return np.empty((0,), dtype=np.float64)

    distances = np.full((centers.shape[0] - 1,), np.nan, dtype=np.float64)
    for pos in range(centers.shape[0] - 1):
        first = centers[pos]
        second = centers[pos + 1]
        if np.all(np.isfinite(first)) and np.all(np.isfinite(second)):
            distances[pos] = float(np.linalg.norm(second - first))
    return distances


def compute_adjacent_distances_from_distance_matrix(distance_matrix):
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("key_layer_center_distances must be a square matrix.")
    if distance_matrix.shape[0] < 2:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(
        [distance_matrix[pos, pos + 1] for pos in range(distance_matrix.shape[0] - 1)],
        dtype=np.float64,
    )


def load_hidden_state_pca_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "layer_indices" not in data:
            raise ValueError("hidden-state PCA npz must contain 'layer_indices'.")
        layer_indices = np.asarray(data["layer_indices"], dtype=np.int64)
        if "key_adjacent_layer_angles_deg" in data:
            key_adjacent_angles = np.asarray(data["key_adjacent_layer_angles_deg"], dtype=np.float64)
        elif "key_layer_centers" in data:
            key_adjacent_angles = compute_adjacent_angles_from_centers(data["key_layer_centers"])
        else:
            key_adjacent_angles = None
        if "key_adjacent_layer_distances" in data:
            key_adjacent_distances = np.asarray(data["key_adjacent_layer_distances"], dtype=np.float64)
        elif "key_layer_center_distances" in data:
            key_adjacent_distances = compute_adjacent_distances_from_distance_matrix(data["key_layer_center_distances"])
        elif "key_layer_centers" in data:
            key_adjacent_distances = compute_adjacent_distances_from_centers(data["key_layer_centers"])
        else:
            key_adjacent_distances = None

    if layer_indices.ndim != 1:
        raise ValueError("PCA layer_indices must be 1D.")
    if len(set(int(layer) for layer in layer_indices.tolist())) != layer_indices.shape[0]:
        raise ValueError("PCA layer_indices contains duplicate layers.")
    expected_angle_count = max(0, layer_indices.shape[0] - 1)
    if key_adjacent_angles is not None and key_adjacent_angles.shape != (expected_angle_count,):
        raise ValueError("key_adjacent_layer_angles_deg must have shape [layer_count - 1].")
    if key_adjacent_distances is not None and key_adjacent_distances.shape != (expected_angle_count,):
        raise ValueError("key_adjacent_layer_distances must have shape [layer_count - 1].")
    return (
        layer_indices.astype(np.int64, copy=False),
        None if key_adjacent_angles is None else key_adjacent_angles.astype(np.float64, copy=False),
        None if key_adjacent_distances is None else key_adjacent_distances.astype(np.float64, copy=False),
    )


def split_layer_groups(similarity, layer_indices, group_threshold=0.85, max_group_size=6):
    if max_group_size < 1:
        raise ValueError("max_group_size must be at least 1.")
    groups = []
    current = [int(layer_indices[0])]
    for pos in range(1, len(layer_indices)):
        should_split = len(current) >= max_group_size
        if float(similarity[pos - 1, pos]) < float(group_threshold):
            should_split = True
        if should_split:
            groups.append(current)
            current = []
        current.append(int(layer_indices[pos]))
    groups.append(current)
    return groups


def split_layer_groups_by_key_pca_angle(
    layer_indices,
    key_adjacent_angles_deg,
    angle_threshold=90.0,
    max_group_size=6,
):
    if max_group_size < 1:
        raise ValueError("max_group_size must be at least 1.")
    if angle_threshold < 0.0 or angle_threshold > 180.0:
        raise ValueError("angle_threshold must be in [0, 180].")

    layer_indices = np.asarray(layer_indices, dtype=np.int64)
    key_adjacent_angles_deg = np.asarray(key_adjacent_angles_deg, dtype=np.float64)
    expected_angle_count = max(0, layer_indices.shape[0] - 1)
    if key_adjacent_angles_deg.shape != (expected_angle_count,):
        raise ValueError("key_adjacent_angles_deg must have shape [layer_count - 1].")

    groups = []
    current = [int(layer_indices[0])]
    for pos in range(1, len(layer_indices)):
        angle = float(key_adjacent_angles_deg[pos - 1])
        should_split = len(current) >= max_group_size
        if np.isfinite(angle) and angle > float(angle_threshold):
            should_split = True
        if should_split:
            groups.append(current)
            current = []
        current.append(int(layer_indices[pos]))
    groups.append(current)
    return groups


def split_layer_groups_by_key_pca_distance(
    layer_indices,
    key_adjacent_distances,
    distance_threshold=1.0,
    max_group_size=6,
):
    if max_group_size < 1:
        raise ValueError("max_group_size must be at least 1.")
    if distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative.")

    layer_indices = np.asarray(layer_indices, dtype=np.int64)
    key_adjacent_distances = np.asarray(key_adjacent_distances, dtype=np.float64)
    expected_distance_count = max(0, layer_indices.shape[0] - 1)
    if key_adjacent_distances.shape != (expected_distance_count,):
        raise ValueError("key_adjacent_distances must have shape [layer_count - 1].")

    groups = []
    current = [int(layer_indices[0])]
    for pos in range(1, len(layer_indices)):
        distance = float(key_adjacent_distances[pos - 1])
        should_split = len(current) >= max_group_size
        if np.isfinite(distance) and distance > float(distance_threshold):
            should_split = True
        if should_split:
            groups.append(current)
            current = []
        current.append(int(layer_indices[pos]))
    groups.append(current)
    return groups


def _compute_mean_without_iqr_outliers(values, metric_name):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError(f"{metric_name} mean_without_outliers requires at least one finite value.")

    q1 = float(np.percentile(finite_values, 25.0))
    q3 = float(np.percentile(finite_values, 75.0))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered_values = finite_values[(finite_values >= lower) & (finite_values <= upper)]
    if filtered_values.size == 0:
        filtered_values = finite_values
    return float(np.mean(filtered_values))


def get_adjacent_similarity_scores(similarity):
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix.")
    if similarity.shape[0] < 2:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(
        [similarity[pos, pos + 1] for pos in range(similarity.shape[0] - 1)],
        dtype=np.float64,
    )


def compute_group_threshold(
    adjacent_similarity,
    group_threshold=0.85,
    threshold_mode=GROUP_THRESHOLD_MODE_FIXED,
):
    if threshold_mode not in SUPPORTED_GROUP_THRESHOLD_MODES:
        raise ValueError(f"group_threshold_mode must be one of {SUPPORTED_GROUP_THRESHOLD_MODES}.")
    if threshold_mode == GROUP_THRESHOLD_MODE_FIXED:
        return float(group_threshold), None

    return (
        _compute_mean_without_iqr_outliers(
            adjacent_similarity,
            "group_threshold_mode",
        ),
        GROUP_THRESHOLD_OUTLIER_METHOD_IQR,
    )


def compute_key_pca_angle_threshold(
    key_adjacent_angles_deg,
    angle_threshold=90.0,
    threshold_mode=KEY_PCA_ANGLE_THRESHOLD_MODE_FIXED,
):
    if threshold_mode not in SUPPORTED_KEY_PCA_ANGLE_THRESHOLD_MODES:
        raise ValueError(f"key_pca_angle_threshold_mode must be one of {SUPPORTED_KEY_PCA_ANGLE_THRESHOLD_MODES}.")
    if threshold_mode == KEY_PCA_ANGLE_THRESHOLD_MODE_FIXED:
        if angle_threshold < 0.0 or angle_threshold > 180.0:
            raise ValueError("angle_threshold must be in [0, 180].")
        return float(angle_threshold), None

    return (
        _compute_mean_without_iqr_outliers(
            key_adjacent_angles_deg,
            "key_pca_angle_threshold_mode",
        ),
        KEY_PCA_ANGLE_THRESHOLD_OUTLIER_METHOD_IQR,
    )


def compute_key_pca_distance_threshold(
    key_adjacent_distances,
    distance_threshold=1.0,
    threshold_mode=KEY_PCA_DISTANCE_THRESHOLD_MODE_FIXED,
):
    if threshold_mode not in SUPPORTED_KEY_PCA_DISTANCE_THRESHOLD_MODES:
        raise ValueError(
            f"key_pca_distance_threshold_mode must be one of {SUPPORTED_KEY_PCA_DISTANCE_THRESHOLD_MODES}."
        )
    if threshold_mode == KEY_PCA_DISTANCE_THRESHOLD_MODE_FIXED:
        if distance_threshold < 0.0:
            raise ValueError("distance_threshold must be non-negative.")
        return float(distance_threshold), None

    return (
        _compute_mean_without_iqr_outliers(
            key_adjacent_distances,
            "key_pca_distance_threshold_mode",
        ),
        KEY_PCA_DISTANCE_THRESHOLD_OUTLIER_METHOD_IQR,
    )


def build_group_budget_stats(groups, ratio_layer_scores):
    group_scores = []
    for layers in groups:
        missing = [int(layer) for layer in layers if int(layer) not in ratio_layer_scores]
        if missing:
            raise ValueError(f"attn output ratio npz is missing layers required by similarity profile: {missing}")
        group_scores.append(float(sum(ratio_layer_scores[int(layer)] for layer in layers)))

    total_score = float(sum(group_scores))
    budget_weights = [
        (group_score / total_score) if total_score > 0.0 else 0.0
        for group_score in group_scores
    ]
    return group_scores, budget_weights


def build_group_mix(similarity, layer_indices, groups, temperature=0.1, min_weight=0.0):
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than 0.")
    if min_weight < 0.0:
        raise ValueError("min_weight must be non-negative.")

    layer_to_pos = {int(layer): pos for pos, layer in enumerate(layer_indices.tolist())}
    group_mixes = []
    for layers in groups:
        mix = {}
        for target_layer in layers:
            target_pos = layer_to_pos[int(target_layer)]
            source_layers = [int(layer) for layer in layers]
            source_positions = [layer_to_pos[layer] for layer in source_layers]
            logits = similarity[target_pos, source_positions] / float(temperature)
            logits = logits - np.max(logits)
            weights = np.exp(logits)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                mix[str(int(target_layer))] = {"sources": [int(target_layer)], "weights": [1.0]}
                continue

            weights = weights / weight_sum
            kept = [
                (source, float(weight))
                for source, weight in zip(source_layers, weights.tolist())
                if float(weight) >= float(min_weight)
            ]
            kept_weight_sum = float(sum(weight for _, weight in kept))
            if kept_weight_sum <= 0.0 or not np.isfinite(kept_weight_sum):
                mix[str(int(target_layer))] = {"sources": [int(target_layer)], "weights": [1.0]}
                continue

            mix[str(int(target_layer))] = {
                "sources": [source for source, _ in kept],
                "weights": [weight / kept_weight_sum for _, weight in kept],
            }
        group_mixes.append(mix)
    return group_mixes



def build_profile_from_similarity(
    similarity,
    layer_indices,
    actual_window_size=None,
    similarity_state="hidden_states",
    similarity_source=SIMILARITY_SOURCE_QUERY_WINDOW,
    similarity_metric="cosine_similarity",
    similarity_reduction="flatten_window",
    group_threshold=0.85,
    group_threshold_mode=GROUP_THRESHOLD_MODE_FIXED,
    max_group_size=6,
    attn_output_ratio=None,
    attn_output_ratio_layer_indices=None,
    group_scheme=GROUP_SCHEME_SIMILARITY,
    key_pca_adjacent_angles_deg=None,
    key_pca_adjacent_distances=None,
    key_pca_angle_threshold=90.0,
    key_pca_angle_threshold_mode=KEY_PCA_ANGLE_THRESHOLD_MODE_FIXED,
    key_pca_distance_threshold=1.0,
    key_pca_distance_threshold_mode=KEY_PCA_DISTANCE_THRESHOLD_MODE_FIXED,
    temperature=0.1,
    min_weight=0.0,
):
    similarity = np.asarray(similarity, dtype=np.float64)
    layer_indices = np.asarray(layer_indices, dtype=np.int64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix.")
    if layer_indices.ndim != 1 or layer_indices.shape[0] != similarity.shape[0]:
        raise ValueError("layer_indices must be 1D and match similarity size.")
    if group_scheme not in SUPPORTED_GROUP_SCHEMES:
        raise ValueError(f"group_scheme must be one of {SUPPORTED_GROUP_SCHEMES}.")
    if group_threshold_mode not in SUPPORTED_GROUP_THRESHOLD_MODES:
        raise ValueError(f"group_threshold_mode must be one of {SUPPORTED_GROUP_THRESHOLD_MODES}.")
    if key_pca_angle_threshold_mode not in SUPPORTED_KEY_PCA_ANGLE_THRESHOLD_MODES:
        raise ValueError(f"key_pca_angle_threshold_mode must be one of {SUPPORTED_KEY_PCA_ANGLE_THRESHOLD_MODES}.")
    if key_pca_distance_threshold_mode not in SUPPORTED_KEY_PCA_DISTANCE_THRESHOLD_MODES:
        raise ValueError(
            f"key_pca_distance_threshold_mode must be one of {SUPPORTED_KEY_PCA_DISTANCE_THRESHOLD_MODES}."
        )

    adjacent_similarity = get_adjacent_similarity_scores(similarity)
    effective_group_threshold = float(group_threshold)
    group_threshold_outlier_method = None
    effective_key_pca_angle_threshold = float(key_pca_angle_threshold)
    key_pca_angle_threshold_outlier_method = None
    effective_key_pca_distance_threshold = float(key_pca_distance_threshold)
    key_pca_distance_threshold_outlier_method = None
    if group_scheme == GROUP_SCHEME_ATTN_LAYER_SIMILARITY:
        if similarity_source != SIMILARITY_SOURCE_ATTN_LAYER:
            raise ValueError(
                "attn_layer_similarity grouping requires an attention-layer similarity npz "
                "produced by --attn_layer_similarity_mode."
            )
        effective_group_threshold, group_threshold_outlier_method = compute_group_threshold(
            adjacent_similarity,
            group_threshold=group_threshold,
            threshold_mode=group_threshold_mode,
        )
        groups = split_layer_groups(similarity, layer_indices, effective_group_threshold, max_group_size)
    elif group_scheme == GROUP_SCHEME_KEY_PCA_ANGLE:
        if key_pca_adjacent_angles_deg is None:
            raise ValueError("key_pca_adjacent_angles_deg is required for key_pca_angle grouping.")
        effective_key_pca_angle_threshold, key_pca_angle_threshold_outlier_method = compute_key_pca_angle_threshold(
            key_pca_adjacent_angles_deg,
            angle_threshold=key_pca_angle_threshold,
            threshold_mode=key_pca_angle_threshold_mode,
        )
        groups = split_layer_groups_by_key_pca_angle(
            layer_indices,
            key_pca_adjacent_angles_deg,
            angle_threshold=effective_key_pca_angle_threshold,
            max_group_size=max_group_size,
        )
    elif group_scheme == GROUP_SCHEME_KEY_PCA_DISTANCE:
        if key_pca_adjacent_distances is None:
            raise ValueError("key_pca_adjacent_distances is required for key_pca_distance grouping.")
        (
            effective_key_pca_distance_threshold,
            key_pca_distance_threshold_outlier_method,
        ) = compute_key_pca_distance_threshold(
            key_pca_adjacent_distances,
            distance_threshold=key_pca_distance_threshold,
            threshold_mode=key_pca_distance_threshold_mode,
        )
        groups = split_layer_groups_by_key_pca_distance(
            layer_indices,
            key_pca_adjacent_distances,
            distance_threshold=effective_key_pca_distance_threshold,
            max_group_size=max_group_size,
        )
    else:
        effective_group_threshold, group_threshold_outlier_method = compute_group_threshold(
            adjacent_similarity,
            group_threshold=group_threshold,
            threshold_mode=group_threshold_mode,
        )
        groups = split_layer_groups(similarity, layer_indices, effective_group_threshold, max_group_size)
    group_ratio_sums = None
    group_budget_weights = None
    ratio_layer_sums = None
    if attn_output_ratio is not None:
        attn_output_ratio = np.asarray(attn_output_ratio, dtype=np.float64)
        attn_output_ratio_layer_indices = np.asarray(attn_output_ratio_layer_indices, dtype=np.int64)
        if attn_output_ratio.ndim != 2:
            raise ValueError("attn_output_ratio must be a 2D array.")
        if (
            attn_output_ratio_layer_indices.ndim != 1
            or attn_output_ratio_layer_indices.shape[0] != attn_output_ratio.shape[0]
        ):
            raise ValueError("attn_output_ratio_layer_indices must match attn_output_ratio first dimension.")
        ratio_layer_scores = {
            int(layer): float(np.var(attn_output_ratio[pos]))
            for pos, layer in enumerate(attn_output_ratio_layer_indices.tolist())
        }
        ratio_layer_sums = {
            int(layer): float(np.sum(attn_output_ratio[pos]))
            for pos, layer in enumerate(attn_output_ratio_layer_indices.tolist())
        }
        group_ratio_sums, group_budget_weights = build_group_budget_stats(groups, ratio_layer_scores)

    group_mixes = build_group_mix(
        similarity=similarity,
        layer_indices=layer_indices,
        groups=groups,
        temperature=temperature,
        min_weight=min_weight,
    )

    profile_groups = []
    for group_idx, layers in enumerate(groups):
        group_profile = {"layers": [int(layer) for layer in layers], "mix": group_mixes[group_idx]}
        if ratio_layer_sums is not None:
            group_profile["attn_output_ratio_sums"] = {
                str(int(layer)): float(ratio_layer_sums[int(layer)])
                for layer in layers
            }
        if group_ratio_sums is not None:
            group_profile["attn_output_ratio_variance_sum"] = float(group_ratio_sums[group_idx])
            group_profile["budget_weight"] = float(group_budget_weights[group_idx])
        profile_groups.append(group_profile)

    key_pca_adjacent_angles = None
    finite_key_pca_angles = None
    if key_pca_adjacent_angles_deg is not None:
        key_pca_adjacent_angles = np.asarray(key_pca_adjacent_angles_deg, dtype=np.float64)
        finite_key_pca_angles = key_pca_adjacent_angles[np.isfinite(key_pca_adjacent_angles)]
    finite_key_pca_distances = None
    if key_pca_adjacent_distances is not None:
        key_pca_adjacent_distances = np.asarray(key_pca_adjacent_distances, dtype=np.float64)
        finite_key_pca_distances = key_pca_adjacent_distances[np.isfinite(key_pca_adjacent_distances)]
    profile = {
        "metric": "cosine",
        "similarity_metric": str(similarity_metric),
        "similarity_reduction": str(similarity_reduction),
        "similarity_state": str(similarity_state),
        "similarity_source": str(similarity_source),
        "source": "single_sample",
        "group_scheme": str(group_scheme),
        "group_threshold": float(effective_group_threshold),
        "effective_group_threshold": float(effective_group_threshold),
        "group_threshold_mode": str(group_threshold_mode),
        "group_threshold_outlier_method": group_threshold_outlier_method,
        "key_pca_angle_threshold": float(effective_key_pca_angle_threshold),
        "effective_key_pca_angle_threshold": float(effective_key_pca_angle_threshold),
        "key_pca_angle_threshold_mode": str(key_pca_angle_threshold_mode),
        "key_pca_angle_threshold_outlier_method": key_pca_angle_threshold_outlier_method,
        "key_pca_distance_threshold": float(effective_key_pca_distance_threshold),
        "effective_key_pca_distance_threshold": float(effective_key_pca_distance_threshold),
        "key_pca_distance_threshold_mode": str(key_pca_distance_threshold_mode),
        "key_pca_distance_threshold_outlier_method": key_pca_distance_threshold_outlier_method,
        "mix_temperature": float(temperature),
        "mix_min_weight": float(min_weight),
        "max_group_size": int(max_group_size),
        "actual_window_size": None if actual_window_size is None else int(actual_window_size),
        "layer_count": int(len(layer_indices)),
        "group_sizes": [len(group) for group in groups],
        "mean_adjacent_similarity": (
            None if adjacent_similarity.size == 0 else float(np.mean(adjacent_similarity))
        ),
        "mean_adjacent_key_pca_angle_deg": (
            None
            if finite_key_pca_angles is None or finite_key_pca_angles.size == 0
            else float(np.mean(finite_key_pca_angles))
        ),
        "mean_adjacent_key_pca_distance": (
            None
            if finite_key_pca_distances is None or finite_key_pca_distances.size == 0
            else float(np.mean(finite_key_pca_distances))
        ),
        "groups": profile_groups,
    }
    if group_ratio_sums is not None:
        profile["budget_weight_metric"] = "attn_output_hidden_l2_ratio_variance_sum"
        profile["layer_budget_metric"] = "attn_output_hidden_l2_ratio_sum"
    return profile


def build_profile_from_npz(
    similarity_npz,
    attn_output_ratio_npz=None,
    hidden_state_pca_npz=None,
    group_scheme=GROUP_SCHEME_SIMILARITY,
    group_threshold=0.85,
    group_threshold_mode=GROUP_THRESHOLD_MODE_FIXED,
    key_pca_angle_threshold=90.0,
    key_pca_angle_threshold_mode=KEY_PCA_ANGLE_THRESHOLD_MODE_FIXED,
    key_pca_distance_threshold=1.0,
    key_pca_distance_threshold_mode=KEY_PCA_DISTANCE_THRESHOLD_MODE_FIXED,
    max_group_size=6,
    temperature=0.1,
    min_weight=0.0,
):
    (
        similarity,
        layer_indices,
        actual_window_size,
        similarity_state,
        similarity_source,
        similarity_metric,
        similarity_reduction,
    ) = load_similarity_npz(similarity_npz)
    attn_output_ratio = None
    attn_output_ratio_layer_indices = None
    if attn_output_ratio_npz is not None:
        attn_output_ratio, attn_output_ratio_layer_indices = load_attn_output_ratio_npz(attn_output_ratio_npz)
    key_pca_adjacent_angles_deg = None
    key_pca_adjacent_distances = None
    if group_scheme in (GROUP_SCHEME_KEY_PCA_ANGLE, GROUP_SCHEME_KEY_PCA_DISTANCE):
        if hidden_state_pca_npz is None:
            raise ValueError(f"--hidden_state_pca_npz is required when --group_scheme {group_scheme}.")
        pca_layer_indices, key_pca_adjacent_angles_deg, key_pca_adjacent_distances = load_hidden_state_pca_npz(
            hidden_state_pca_npz
        )
        if not np.array_equal(layer_indices, pca_layer_indices):
            raise ValueError("hidden-state PCA layer_indices must exactly match similarity layer_indices.")
        if group_scheme == GROUP_SCHEME_KEY_PCA_ANGLE and key_pca_adjacent_angles_deg is None:
            raise ValueError(
                "hidden-state PCA npz must contain 'key_adjacent_layer_angles_deg' "
                "or 'key_layer_centers' when --group_scheme key_pca_angle."
            )
        if group_scheme == GROUP_SCHEME_KEY_PCA_DISTANCE and key_pca_adjacent_distances is None:
            raise ValueError(
                "hidden-state PCA npz must contain 'key_adjacent_layer_distances', "
                "'key_layer_center_distances', or 'key_layer_centers' when --group_scheme key_pca_distance."
            )
    profile = build_profile_from_similarity(
        similarity=similarity,
        layer_indices=layer_indices,
        actual_window_size=actual_window_size,
        similarity_state=similarity_state,
        similarity_source=similarity_source,
        similarity_metric=similarity_metric,
        similarity_reduction=similarity_reduction,
        group_threshold=group_threshold,
        group_threshold_mode=group_threshold_mode,
        key_pca_angle_threshold=key_pca_angle_threshold,
        key_pca_angle_threshold_mode=key_pca_angle_threshold_mode,
        key_pca_distance_threshold=key_pca_distance_threshold,
        key_pca_distance_threshold_mode=key_pca_distance_threshold_mode,
        max_group_size=max_group_size,
        attn_output_ratio=attn_output_ratio,
        attn_output_ratio_layer_indices=attn_output_ratio_layer_indices,
        group_scheme=group_scheme,
        key_pca_adjacent_angles_deg=key_pca_adjacent_angles_deg,
        key_pca_adjacent_distances=key_pca_adjacent_distances,
        temperature=temperature,
        min_weight=min_weight,
    )
    profile["similarity_npz"] = os.path.abspath(os.path.expanduser(str(similarity_npz)))
    if attn_output_ratio_npz is not None:
        profile["attn_output_ratio_npz"] = os.path.abspath(os.path.expanduser(str(attn_output_ratio_npz)))
    if hidden_state_pca_npz is not None:
        profile["hidden_state_pca_npz"] = os.path.abspath(os.path.expanduser(str(hidden_state_pca_npz)))
    return profile


def write_profile(profile, output_path):
    output_path = os.path.abspath(os.path.expanduser(str(output_path)))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a single-sample hidden-mix profile for snapkv_neighbor_shared.")
    parser.add_argument(
        "--similarity_npz",
        required=True,
        help="Path to one query-window similarity or attention-layer similarity .npz file.",
    )
    parser.add_argument("--attn_output_ratio_npz", default=None, help="Optional attn-output ratio .npz for group budget weights.")
    parser.add_argument(
        "--hidden_state_pca_npz",
        default=None,
        help="Hidden-state PCA .npz used by PCA-based group schemes.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("hidden_mix_profile", "hidden_mix_profile.json"),
        help="Output JSON profile path.",
    )
    parser.add_argument("--group_scheme", choices=SUPPORTED_GROUP_SCHEMES, default=GROUP_SCHEME_SIMILARITY)
    parser.add_argument("--group_threshold", type=float, default=0.85)
    parser.add_argument(
        "--group_threshold_mode",
        choices=SUPPORTED_GROUP_THRESHOLD_MODES,
        default=GROUP_THRESHOLD_MODE_FIXED,
    )
    parser.add_argument("--key_pca_angle_threshold", type=float, default=90.0)
    parser.add_argument(
        "--key_pca_angle_threshold_mode",
        choices=SUPPORTED_KEY_PCA_ANGLE_THRESHOLD_MODES,
        default=KEY_PCA_ANGLE_THRESHOLD_MODE_FIXED,
    )
    parser.add_argument("--key_pca_distance_threshold", type=float, default=1.0)
    parser.add_argument(
        "--key_pca_distance_threshold_mode",
        choices=SUPPORTED_KEY_PCA_DISTANCE_THRESHOLD_MODES,
        default=KEY_PCA_DISTANCE_THRESHOLD_MODE_FIXED,
    )
    parser.add_argument("--max_group_size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--min_weight", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    profile = build_profile_from_npz(
        similarity_npz=args.similarity_npz,
        attn_output_ratio_npz=args.attn_output_ratio_npz,
        hidden_state_pca_npz=args.hidden_state_pca_npz,
        group_scheme=args.group_scheme,
        group_threshold=args.group_threshold,
        group_threshold_mode=args.group_threshold_mode,
        key_pca_angle_threshold=args.key_pca_angle_threshold,
        key_pca_angle_threshold_mode=args.key_pca_angle_threshold_mode,
        key_pca_distance_threshold=args.key_pca_distance_threshold,
        key_pca_distance_threshold_mode=args.key_pca_distance_threshold_mode,
        max_group_size=args.max_group_size,
        temperature=args.temperature,
        min_weight=args.min_weight,
    )
    output_path = write_profile(profile, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
