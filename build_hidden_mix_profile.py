import argparse
import json
import os

import numpy as np


def _scalar_from_npz(value, default=None):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return default


def load_similarity_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "similarity" not in data or "layer_indices" not in data:
            raise ValueError("similarity npz must contain 'similarity' and 'layer_indices'.")
        similarity = np.asarray(data["similarity"], dtype=np.float64)
        layer_indices = np.asarray(data["layer_indices"], dtype=np.int64)
        actual_window_size = _scalar_from_npz(data.get("actual_window_size"))
        similarity_state = _scalar_from_npz(data.get("query_window_similarity_state"), "hidden_states")

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

    return similarity, layer_indices.astype(np.int64, copy=False), actual_window_size, str(similarity_state)


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


def build_group_budget_stats(groups, ratio_layer_sums):
    group_sums = []
    for layers in groups:
        missing = [int(layer) for layer in layers if int(layer) not in ratio_layer_sums]
        if missing:
            raise ValueError(f"attn output ratio npz is missing layers required by similarity profile: {missing}")
        group_sums.append(float(sum(ratio_layer_sums[int(layer)] for layer in layers)))

    total_sum = float(sum(group_sums))
    budget_weights = [
        (group_sum / total_sum) if total_sum > 0.0 else 0.0
        for group_sum in group_sums
    ]
    return group_sums, budget_weights


def _softmax(values, temperature):
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    scaled = np.asarray(values, dtype=np.float64) / float(temperature)
    scaled = scaled - np.max(scaled)
    exp_values = np.exp(scaled)
    return exp_values / np.sum(exp_values)


def build_profile_from_similarity(
    similarity,
    layer_indices,
    actual_window_size=None,
    similarity_state="hidden_states",
    group_threshold=0.85,
    max_group_size=6,
    temperature=0.05,
    min_weight=0.03,
    attn_output_ratio=None,
    attn_output_ratio_layer_indices=None,
):
    if min_weight < 0.0:
        raise ValueError("min_weight must be non-negative.")
    similarity = np.asarray(similarity, dtype=np.float64)
    layer_indices = np.asarray(layer_indices, dtype=np.int64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix.")
    if layer_indices.ndim != 1 or layer_indices.shape[0] != similarity.shape[0]:
        raise ValueError("layer_indices must be 1D and match similarity size.")

    layer_to_pos = {int(layer): pos for pos, layer in enumerate(layer_indices.tolist())}
    groups = split_layer_groups(similarity, layer_indices, group_threshold, max_group_size)
    group_ratio_sums = None
    group_budget_weights = None
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
        ratio_layer_sums = {
            int(layer): float(np.sum(attn_output_ratio[pos]))
            for pos, layer in enumerate(attn_output_ratio_layer_indices.tolist())
        }
        group_ratio_sums, group_budget_weights = build_group_budget_stats(groups, ratio_layer_sums)

    profile_groups = []
    for group_idx, layers in enumerate(groups):
        mix = {}
        source_positions = [layer_to_pos[int(layer)] for layer in layers]
        for target_layer in layers:
            target_pos = layer_to_pos[int(target_layer)]
            scores = similarity[target_pos, source_positions]
            weights = _softmax(scores, temperature)
            kept = [
                (int(source_layer), float(weight))
                for source_layer, weight in zip(layers, weights)
                if float(weight) >= float(min_weight)
            ]
            if not kept:
                kept = [(int(target_layer), 1.0)]
            weight_sum = sum(weight for _source, weight in kept)
            mix[str(int(target_layer))] = {
                "sources": [source for source, _weight in kept],
                "weights": [weight / weight_sum for _source, weight in kept],
            }
        group_profile = {"layers": [int(layer) for layer in layers], "mix": mix}
        if group_ratio_sums is not None:
            group_profile["attn_output_ratio_sum"] = float(group_ratio_sums[group_idx])
            group_profile["budget_weight"] = float(group_budget_weights[group_idx])
        profile_groups.append(group_profile)

    adjacent = [float(similarity[pos, pos + 1]) for pos in range(max(0, len(layer_indices) - 1))]
    profile = {
        "metric": "cosine",
        "similarity_state": str(similarity_state),
        "source": "single_sample",
        "group_threshold": float(group_threshold),
        "max_group_size": int(max_group_size),
        "temperature": float(temperature),
        "min_weight": float(min_weight),
        "actual_window_size": None if actual_window_size is None else int(actual_window_size),
        "layer_count": int(len(layer_indices)),
        "group_sizes": [len(group) for group in groups],
        "mean_adjacent_similarity": None if not adjacent else float(np.mean(adjacent)),
        "groups": profile_groups,
    }
    if group_ratio_sums is not None:
        profile["budget_weight_metric"] = "attn_output_hidden_l2_ratio_sum"
    return profile


def build_profile_from_npz(
    similarity_npz,
    attn_output_ratio_npz=None,
    group_threshold=0.85,
    max_group_size=6,
    temperature=0.05,
    min_weight=0.03,
):
    similarity, layer_indices, actual_window_size, similarity_state = load_similarity_npz(similarity_npz)
    attn_output_ratio = None
    attn_output_ratio_layer_indices = None
    if attn_output_ratio_npz is not None:
        attn_output_ratio, attn_output_ratio_layer_indices = load_attn_output_ratio_npz(attn_output_ratio_npz)
    profile = build_profile_from_similarity(
        similarity=similarity,
        layer_indices=layer_indices,
        actual_window_size=actual_window_size,
        similarity_state=similarity_state,
        group_threshold=group_threshold,
        max_group_size=max_group_size,
        temperature=temperature,
        min_weight=min_weight,
        attn_output_ratio=attn_output_ratio,
        attn_output_ratio_layer_indices=attn_output_ratio_layer_indices,
    )
    profile["similarity_npz"] = os.path.abspath(os.path.expanduser(str(similarity_npz)))
    if attn_output_ratio_npz is not None:
        profile["attn_output_ratio_npz"] = os.path.abspath(os.path.expanduser(str(attn_output_ratio_npz)))
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
    parser.add_argument("--similarity_npz", required=True, help="Path to one query-window similarity .npz file.")
    parser.add_argument("--attn_output_ratio_npz", default=None, help="Optional attn-output ratio .npz for group budget weights.")
    parser.add_argument("--output", required=True, help="Output JSON profile path.")
    parser.add_argument("--group_threshold", type=float, default=0.85)
    parser.add_argument("--max_group_size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--min_weight", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    profile = build_profile_from_npz(
        similarity_npz=args.similarity_npz,
        attn_output_ratio_npz=args.attn_output_ratio_npz,
        group_threshold=args.group_threshold,
        max_group_size=args.max_group_size,
        temperature=args.temperature,
        min_weight=args.min_weight,
    )
    output_path = write_profile(profile, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
