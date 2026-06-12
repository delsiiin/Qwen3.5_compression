import json
import os

import numpy as np
import torch

from attn_heatmap import (
    build_run_dir,
    build_token_entries,
    extract_result_summary,
    sanitize_slug,
)
from attn_output_ratio import parse_layer_spec
from query_window_similarity import extract_hidden_tensor, get_decoder_layers, run_observation_forward


PCA_FIT_SCOPE = "shared_layers_tokens"
PCA_HEAD_FIT_SCOPE = "shared_heads_tokens_per_layer"
PCA_STATE_NAMES = ("key_states", "value_states")
PCA_HIDDEN_STATE_NAMES = ("hidden_states",)
PCA_COMPONENT_COUNT = 2
PCA_TOKEN_SCOPE = "all_tokens"
PCA_SUBMODE_KEY_VALUE_STATES = "key_value_states"
PCA_SUBMODE_KEY_VALUE_HEADS = "key_value_heads"
PCA_SUBMODE_HIDDEN_STATES = "hidden_states"
SUPPORTED_HIDDEN_STATE_PCA_SUBMODES = {
    PCA_SUBMODE_KEY_VALUE_HEADS,
    PCA_SUBMODE_KEY_VALUE_STATES,
    PCA_SUBMODE_HIDDEN_STATES,
}


def _wrap_hook_without_kwargs(hook):
    def wrapped(module, args, output):
        return hook(module, args, {}, output)

    return wrapped


def _get_attention_module(layer):
    for attr in ("self_attn", "attention", "attn"):
        attention = getattr(layer, attr, None)
        if attention is not None and hasattr(attention, "k_proj") and hasattr(attention, "v_proj"):
            return attention
    return None


def _get_arg_or_kwarg(args, kwargs, index, name):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return None


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


def _get_head_dim(attention):
    head_dim = getattr(attention, "head_dim", None)
    if head_dim is None:
        config = getattr(attention, "config", None)
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        raise ValueError("Cannot infer attention head_dim.")
    return int(head_dim)


def _get_num_key_value_heads(attention, projection_dim, head_dim):
    config = getattr(attention, "config", None)
    num_heads = getattr(attention, "num_key_value_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_key_value_heads", None)
    if num_heads is not None:
        return int(num_heads)
    return int(projection_dim // head_dim)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _slice_position_embedding(position_embedding, seq_len):
    if position_embedding.ndim == 2:
        return position_embedding[-seq_len:, :]
    if position_embedding.ndim >= 3:
        return position_embedding[:, -seq_len:, :]
    return position_embedding


def _unsqueeze_position_embedding(position_embedding, states):
    if position_embedding.ndim == 2:
        position_embedding = position_embedding.unsqueeze(0)
    while position_embedding.ndim < states.ndim:
        position_embedding = position_embedding.unsqueeze(1)
    return position_embedding


def _apply_rotary(states, position_embeddings):
    if position_embeddings is None:
        return states
    cos, sin = position_embeddings
    cos = _slice_position_embedding(cos.to(device=states.device, dtype=states.dtype), states.shape[-2])
    sin = _slice_position_embedding(sin.to(device=states.device, dtype=states.dtype), states.shape[-2])
    cos = _unsqueeze_position_embedding(cos, states)
    sin = _unsqueeze_position_embedding(sin, states)
    rotary_dim = min(cos.shape[-1], states.shape[-1])
    states_rot = states[..., :rotary_dim]
    states_pass = states[..., rotary_dim:]
    states_embed = (states_rot * cos[..., :rotary_dim]) + (_rotate_half(states_rot) * sin[..., :rotary_dim])
    if states_pass.shape[-1] == 0:
        return states_embed
    return torch.cat([states_embed, states_pass], dim=-1)


def project_key_states(attention, hidden_states, position_embeddings):
    k_proj = getattr(attention, "k_proj", None)
    if k_proj is None:
        raise ValueError("Attention module does not expose k_proj; cannot capture key states.")
    device = _module_device(attention)
    dtype = _module_dtype(attention)
    hidden_states = hidden_states.to(device=device, dtype=dtype)
    head_dim = _get_head_dim(attention)
    input_shape = hidden_states.shape[:-1]
    key_proj = k_proj(hidden_states)
    num_key_value_heads = _get_num_key_value_heads(attention, key_proj.shape[-1], head_dim)
    key_states = key_proj.view(*input_shape, num_key_value_heads, head_dim)
    k_norm = getattr(attention, "k_norm", None)
    if k_norm is not None:
        key_states = k_norm(key_states)
    key_states = key_states.transpose(1, 2)
    return _apply_rotary(key_states, position_embeddings)


def normalize_token_span(token_count, token_start=0, token_end=None):
    token_count = int(token_count)
    if token_count < 1:
        raise ValueError("Cannot select a token span from an empty prompt.")

    start = 0 if token_start is None else int(token_start)
    end = token_count if token_end is None else int(token_end)
    if start < 0:
        start = token_count + start
    if end < 0:
        end = token_count + end
    start = max(0, min(token_count, start))
    end = max(0, min(token_count, end))
    if end <= start:
        raise ValueError(
            f"Selected token span is empty after normalization: "
            f"start={start}, end={end}, token_count={token_count}."
        )
    return start, end


def setup_matplotlib_cache():
    cache_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "qwen35_compression_matplotlib_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)


def collect_layer_hidden_states(model, inputs, layer_spec="all", token_start=0, token_end=None):
    layers = get_decoder_layers(model)
    if not layers:
        raise ValueError("No decoder layers were found; cannot capture hidden states.")

    available_layers = list(range(len(layers)))
    selected_layers = parse_layer_spec(layer_spec, available_layers)
    selected_layer_set = set(selected_layers)
    captured = {}
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            hidden = extract_hidden_tensor(output)
            if hidden is None or not torch.is_tensor(hidden) or hidden.ndim < 3:
                return
            captured[layer_idx] = (
                hidden[0]
                .detach()
                .to(dtype=torch.float32)
                .cpu()
                .contiguous()
            )

        return hook

    try:
        for layer_idx, layer in enumerate(layers):
            if layer_idx in selected_layer_set:
                handles.append(layer.register_forward_hook(make_hook(layer_idx)))
        run_observation_forward(model, inputs, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()

    layer_indices = [layer_idx for layer_idx in selected_layers if layer_idx in captured]
    if not layer_indices:
        raise ValueError("No layer hidden states were captured during prefill.")

    full_token_count = min(int(captured[layer_idx].shape[0]) for layer_idx in layer_indices)
    hidden_size = min(int(captured[layer_idx].shape[-1]) for layer_idx in layer_indices)
    if full_token_count < 1 or hidden_size < 1:
        raise ValueError("Captured hidden states are empty.")
    actual_token_start, actual_token_end = normalize_token_span(
        full_token_count,
        token_start=token_start,
        token_end=token_end,
    )

    hidden_states = torch.stack(
        [
            captured[layer_idx][actual_token_start:actual_token_end, :hidden_size]
            for layer_idx in layer_indices
        ],
        dim=0,
    ).numpy().astype(np.float32, copy=False)
    return hidden_states, layer_indices, actual_token_start, actual_token_end


def compute_shared_pca(hidden_states, component_count=PCA_COMPONENT_COUNT):
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [layer, token, hidden].")

    layer_count, token_count, hidden_size = hidden_states.shape
    flat = hidden_states.reshape(layer_count * token_count, hidden_size).astype(np.float32, copy=False)
    if flat.shape[0] < 2:
        raise ValueError("At least two hidden-state vectors are required for PCA.")

    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = flat - mean
    centered64 = centered.astype(np.float64, copy=False)
    _u, singular_values, vh = np.linalg.svd(centered64, full_matrices=False)

    actual_components = min(int(component_count), vh.shape[0])
    components = np.zeros((component_count, hidden_size), dtype=np.float32)
    if actual_components > 0:
        components[:actual_components] = vh[:actual_components].astype(np.float32, copy=False)

    points = np.matmul(centered, components.T).astype(np.float32, copy=False)
    pca_points = points.reshape(layer_count, token_count, component_count)

    variances = (singular_values ** 2) / max(1, flat.shape[0] - 1)
    total_variance = float(np.sum(variances))
    explained = np.zeros((component_count,), dtype=np.float32)
    if total_variance > 0 and actual_components > 0:
        explained[:actual_components] = (
            variances[:actual_components] / total_variance
        ).astype(np.float32, copy=False)

    return pca_points, components, mean, explained


def compute_hidden_state_pca(model, inputs, layer_spec="all", token_start=0, token_end=None):
    hidden_states, layer_indices, actual_token_start, actual_token_end = collect_layer_hidden_states(
        model=model,
        inputs=inputs,
        layer_spec=layer_spec,
        token_start=token_start,
        token_end=token_end,
    )
    pca_points, components, mean, explained = compute_shared_pca(hidden_states)
    return pca_points, layer_indices, components, mean, explained, actual_token_start, actual_token_end


def project_value_states(attention, hidden_states):
    v_proj = getattr(attention, "v_proj", None)
    if v_proj is None:
        raise ValueError("Attention module does not expose v_proj; cannot capture value states.")
    device = _module_device(attention)
    dtype = _module_dtype(attention)
    hidden_states = hidden_states.to(device=device, dtype=dtype)
    head_dim = _get_head_dim(attention)
    input_shape = hidden_states.shape[:-1]
    value_proj = v_proj(hidden_states)
    num_key_value_heads = _get_num_key_value_heads(attention, value_proj.shape[-1], head_dim)
    value_states = value_proj.view(*input_shape, num_key_value_heads, head_dim)
    v_norm = getattr(attention, "v_norm", None)
    if v_norm is not None:
        value_states = v_norm(value_states)
    return value_states.transpose(1, 2)


def flatten_key_value_token_states(key_states, value_states, token_start, token_end):
    key_states = key_states[0, :, token_start:token_end, :].transpose(0, 1).contiguous()
    value_states = value_states[0, :, token_start:token_end, :].transpose(0, 1).contiguous()
    key_flat = key_states.reshape(key_states.shape[0], -1)
    value_flat = value_states.reshape(value_states.shape[0], -1)
    return torch.cat([key_flat, value_flat], dim=-1)


def flatten_token_states(states, token_start, token_end):
    states = states[0, :, token_start:token_end, :].transpose(0, 1).contiguous()
    return states.reshape(states.shape[0], -1)


def collect_layer_key_value_states(
    model,
    inputs,
    layer_spec="all",
    token_start=0,
    token_end=None,
):
    layers = get_decoder_layers(model)
    if not layers:
        raise ValueError("No decoder layers were found; cannot capture key/value states.")

    available_layers = list(range(len(layers)))
    selected_layers = parse_layer_spec(layer_spec, available_layers)
    selected_layer_set = set(selected_layers)
    captured = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs, _output):
            hidden_states = _get_arg_or_kwarg(args, kwargs, 0, "hidden_states")
            if hidden_states is None or not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
                return
            position_embeddings = _get_arg_or_kwarg(args, kwargs, 1, "position_embeddings")
            with torch.inference_mode():
                key_states = project_key_states(module, hidden_states, position_embeddings)
                value_states = project_value_states(module, hidden_states)
            captured[layer_idx] = {
                "key_states": key_states.detach().to(dtype=torch.float32, device="cpu").clone(),
                "value_states": value_states.detach().to(dtype=torch.float32, device="cpu").clone(),
            }

        return hook

    try:
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in selected_layer_set:
                continue
            attention = _get_attention_module(layer)
            if attention is None:
                continue
            try:
                handles.append(attention.register_forward_hook(make_hook(layer_idx), with_kwargs=True))
            except TypeError:
                handles.append(attention.register_forward_hook(_wrap_hook_without_kwargs(make_hook(layer_idx))))
        run_observation_forward(model, inputs, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()

    layer_indices = [layer_idx for layer_idx in selected_layers if layer_idx in captured]
    if not layer_indices:
        raise ValueError("No layer key/value states were captured during prefill.")

    full_token_count = min(int(captured[layer_idx]["key_states"].shape[-2]) for layer_idx in layer_indices)
    actual_token_start, actual_token_end = normalize_token_span(
        full_token_count,
        token_start=token_start,
        token_end=token_end,
    )
    return captured, layer_indices, actual_token_start, actual_token_end


def compute_key_value_state_pca(
    model,
    inputs,
    layer_spec="all",
    token_start=0,
    token_end=None,
):
    captured, layer_indices, actual_token_start, actual_token_end = collect_layer_key_value_states(
        model=model,
        inputs=inputs,
        layer_spec=layer_spec,
        token_start=token_start,
        token_end=token_end,
    )
    token_states = []
    for layer_idx in layer_indices:
        layer_capture = captured[int(layer_idx)]
        token_states.append(
            (
                flatten_token_states(layer_capture["key_states"], actual_token_start, actual_token_end),
                flatten_token_states(layer_capture["value_states"], actual_token_start, actual_token_end),
            )
        )
    min_key_feature_size = min(int(key_state.shape[-1]) for key_state, _value_state in token_states)
    min_value_feature_size = min(int(value_state.shape[-1]) for _key_state, value_state in token_states)
    key_states = torch.stack(
        [key_state[:, :min_key_feature_size] for key_state, _value_state in token_states],
        dim=0,
    ).numpy().astype(np.float32, copy=False)
    value_states = torch.stack(
        [value_state[:, :min_value_feature_size] for _key_state, value_state in token_states],
        dim=0,
    ).numpy().astype(np.float32, copy=False)
    key_pca_points, key_components, key_mean, key_explained = compute_shared_pca(key_states)
    value_pca_points, value_components, value_mean, value_explained = compute_shared_pca(value_states)
    return (
        key_pca_points,
        value_pca_points,
        layer_indices,
        key_components,
        value_components,
        key_mean,
        value_mean,
        key_explained,
        value_explained,
        actual_token_start,
        actual_token_end,
    )


def compute_key_value_head_state_pca(
    model,
    inputs,
    layer_spec="all",
    token_start=0,
    token_end=None,
):
    captured, layer_indices, actual_token_start, actual_token_end = collect_layer_key_value_states(
        model=model,
        inputs=inputs,
        layer_spec=layer_spec,
        token_start=token_start,
        token_end=token_end,
    )
    min_key_heads = min(int(captured[layer_idx]["key_states"].shape[1]) for layer_idx in layer_indices)
    min_value_heads = min(int(captured[layer_idx]["value_states"].shape[1]) for layer_idx in layer_indices)
    min_key_dim = min(int(captured[layer_idx]["key_states"].shape[-1]) for layer_idx in layer_indices)
    min_value_dim = min(int(captured[layer_idx]["value_states"].shape[-1]) for layer_idx in layer_indices)

    key_pca_points = []
    value_pca_points = []
    key_components = []
    value_components = []
    key_means = []
    value_means = []
    key_explained = []
    value_explained = []
    for layer_idx in layer_indices:
        layer_capture = captured[int(layer_idx)]
        key_states = (
            layer_capture["key_states"][0, :min_key_heads, actual_token_start:actual_token_end, :min_key_dim]
            .numpy()
            .astype(np.float32, copy=False)
        )
        value_states = (
            layer_capture["value_states"][0, :min_value_heads, actual_token_start:actual_token_end, :min_value_dim]
            .numpy()
            .astype(np.float32, copy=False)
        )
        cur_key_points, cur_key_components, cur_key_mean, cur_key_explained = compute_shared_pca(key_states)
        cur_value_points, cur_value_components, cur_value_mean, cur_value_explained = compute_shared_pca(value_states)
        key_pca_points.append(cur_key_points)
        value_pca_points.append(cur_value_points)
        key_components.append(cur_key_components)
        value_components.append(cur_value_components)
        key_means.append(cur_key_mean)
        value_means.append(cur_value_mean)
        key_explained.append(cur_key_explained)
        value_explained.append(cur_value_explained)

    return (
        np.stack(key_pca_points, axis=0).astype(np.float32, copy=False),
        np.stack(value_pca_points, axis=0).astype(np.float32, copy=False),
        layer_indices,
        list(range(min_key_heads)),
        list(range(min_value_heads)),
        np.stack(key_components, axis=0).astype(np.float32, copy=False),
        np.stack(value_components, axis=0).astype(np.float32, copy=False),
        np.stack(key_means, axis=0).astype(np.float32, copy=False),
        np.stack(value_means, axis=0).astype(np.float32, copy=False),
        np.stack(key_explained, axis=0).astype(np.float32, copy=False),
        np.stack(value_explained, axis=0).astype(np.float32, copy=False),
        actual_token_start,
        actual_token_end,
    )


def compute_pca_layer_centers(pca_points, token_mask=None):
    pca_points = np.asarray(pca_points, dtype=np.float32)
    if pca_points.ndim != 3 or pca_points.shape[-1] != 2:
        raise ValueError("pca_points must have shape [layer, token, 2].")
    if token_mask is None:
        token_mask = np.ones(pca_points.shape[:2], dtype=bool)
    token_mask = np.asarray(token_mask, dtype=bool)
    if token_mask.shape != pca_points.shape[:2]:
        raise ValueError("token_mask must have shape [layer, token].")

    centers = np.full((pca_points.shape[0], 2), np.nan, dtype=np.float32)
    for layer_pos in range(pca_points.shape[0]):
        points = pca_points[layer_pos][token_mask[layer_pos]]
        if points.shape[0] > 0:
            centers[layer_pos] = points.mean(axis=0)
    return centers


def compute_adjacent_layer_angles(centers):
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError("centers must have shape [layer, 2].")
    if centers.shape[0] < 2:
        return np.empty((0,), dtype=np.float32)

    angles = np.full((centers.shape[0] - 1,), np.nan, dtype=np.float32)
    for layer_pos in range(centers.shape[0] - 1):
        first = centers[layer_pos]
        second = centers[layer_pos + 1]
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            continue
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        if first_norm <= 1e-8 or second_norm <= 1e-8:
            continue
        cosine = float(np.dot(first, second) / (first_norm * second_norm))
        cosine = max(-1.0, min(1.0, cosine))
        angles[layer_pos] = np.degrees(np.arccos(cosine))
    return angles


def build_adjacent_layer_pairs(layer_indices):
    if len(layer_indices) > 1:
        return np.asarray(
            [
                [int(layer_indices[idx]), int(layer_indices[idx + 1])]
                for idx in range(len(layer_indices) - 1)
            ],
            dtype=np.int16,
        )
    return np.empty((0, 2), dtype=np.int16)


def plot_hidden_state_pca(pca_points, layer_indices, output_path, title, token_mask=None, empty_label="No tokens"):
    setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca_points = np.asarray(pca_points, dtype=np.float32)
    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    num_layers = len(layer_indices)
    if pca_points.ndim != 3 or pca_points.shape[-1] != 2:
        raise ValueError("pca_points must have shape [layer, token, 2].")
    if token_mask is None:
        token_mask = np.ones(pca_points.shape[:2], dtype=bool)
    token_mask = np.asarray(token_mask, dtype=bool)
    if token_mask.shape != pca_points.shape[:2]:
        raise ValueError("token_mask must have shape [layer, token].")
    centers = compute_pca_layer_centers(pca_points, token_mask=token_mask)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    cmap = plt.get_cmap("turbo")
    denom = max(1, num_layers - 1)
    plotted_any = False
    for layer_pos, layer_idx in enumerate(layer_indices):
        points = pca_points[layer_pos][token_mask[layer_pos]]
        if points.shape[0] < 1:
            continue
        plotted_any = True
        color = cmap(layer_pos / denom)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=9,
            alpha=0.52,
            color=color,
            linewidths=0,
            label=f"Layer {layer_idx}" if num_layers <= 12 else None,
            rasterized=True,
        )
        if points.shape[0] > 0:
            center = centers[layer_pos]
            if np.linalg.norm(center) > 1e-8:
                ax.annotate(
                    "",
                    xy=(center[0], center[1]),
                    xytext=(0.0, 0.0),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "alpha": 0.78,
                        "linewidth": 1.05,
                        "shrinkA": 0.0,
                        "shrinkB": 3.0,
                    },
                    zorder=2,
                )
            ax.scatter(
                center[0],
                center[1],
                s=46,
                color=color,
                edgecolors="#202020",
                linewidths=0.7,
                marker="o",
                zorder=3,
            )

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            empty_label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
        )

    norm = matplotlib.colors.Normalize(
        vmin=min(layer_indices) if layer_indices else 0,
        vmax=max(layer_indices) if layer_indices else 1,
    )
    scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Layer")
    if num_layers <= 12:
        ax.legend(loc="best", frameon=True, fontsize=8, markerscale=1.4)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(alpha=0.2, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_head_pca_layer(pca_points, head_indices, output_path, title, empty_label="No tokens"):
    setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca_points = np.asarray(pca_points, dtype=np.float32)
    head_indices = [int(head_idx) for head_idx in head_indices]
    if pca_points.ndim != 3 or pca_points.shape[-1] != 2:
        raise ValueError("pca_points must have shape [head, token, 2].")
    if len(head_indices) != pca_points.shape[0]:
        raise ValueError("head_indices must match the first dimension of pca_points.")

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    cmap = plt.get_cmap("turbo")
    denom = max(1, len(head_indices) - 1)
    plotted_any = False
    for head_pos, head_idx in enumerate(head_indices):
        points = pca_points[head_pos]
        if points.shape[0] < 1:
            continue
        plotted_any = True
        color = cmap(head_pos / denom)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=9,
            alpha=0.52,
            color=color,
            linewidths=0,
            label=f"Head {head_idx}" if len(head_indices) <= 16 else None,
            rasterized=True,
        )
        center = points.mean(axis=0)
        ax.scatter(
            center[0],
            center[1],
            s=42,
            color=color,
            edgecolors="#202020",
            linewidths=0.7,
            marker="o",
            zorder=3,
        )

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            empty_label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
        )

    norm = matplotlib.colors.Normalize(
        vmin=min(head_indices) if head_indices else 0,
        vmax=max(head_indices) if head_indices else 1,
    )
    scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Head")
    if len(head_indices) <= 16:
        ax.legend(loc="best", frameon=True, fontsize=8, markerscale=1.4)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(alpha=0.2, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_adjacent_layer_angle_curve(layer_indices, angles_deg, output_path, title):
    setup_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_indices = [int(layer_idx) for layer_idx in layer_indices]
    angles_deg = np.asarray(angles_deg, dtype=np.float32)
    expected_count = max(0, len(layer_indices) - 1)
    if angles_deg.shape != (expected_count,):
        raise ValueError("angles_deg must have shape [layer_count - 1].")

    if expected_count > 0:
        x_values = np.arange(expected_count, dtype=np.int64)
        pair_labels = [f"{layer_indices[idx]}-{layer_indices[idx + 1]}" for idx in range(expected_count)]
    else:
        x_values = np.empty((0,), dtype=np.int64)
        pair_labels = []

    fig_width = max(7.2, min(18.0, 0.38 * max(1, expected_count) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=180)
    finite = np.isfinite(angles_deg)
    if expected_count > 0 and finite.any():
        ax.plot(
            x_values,
            angles_deg,
            color="#2563eb",
            linewidth=1.6,
            marker="o",
            markersize=4.2,
        )
        ax.scatter(x_values[finite], angles_deg[finite], color="#1d4ed8", s=22, zorder=3)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid adjacent-layer angles",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Layer pair")
    ax.set_ylabel("Angle (degrees)")
    ax.set_ylim(0.0, 180.0)
    ax.set_xticks(x_values)
    if expected_count <= 30:
        ax.set_xticklabels(pair_labels, rotation=45, ha="right")
    else:
        sparse_step = max(1, expected_count // 18)
        sparse_labels = [
            pair_labels[idx] if idx % sparse_step == 0 or idx == expected_count - 1 else ""
            for idx in range(expected_count)
        ]
        ax.set_xticklabels(sparse_labels, rotation=45, ha="right")
    ax.grid(alpha=0.24, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


class HiddenStatePCARunWriter:
    def __init__(
        self,
        root_dir,
        model_name,
        out_file,
        max_prefill_tokens,
        layer_spec="all",
        token_start=0,
        token_end=None,
        submode=PCA_SUBMODE_KEY_VALUE_STATES,
    ):
        self.root_dir = root_dir
        self.model_name = model_name
        self.out_file = os.path.abspath(out_file)
        self.max_prefill_tokens = int(max_prefill_tokens) if max_prefill_tokens is not None else None
        self.layer_spec = layer_spec or "all"
        self.token_start = 0 if token_start is None else int(token_start)
        self.token_end = int(token_end) if token_end is not None else None
        self.submode = submode or PCA_SUBMODE_KEY_VALUE_STATES
        if self.submode not in SUPPORTED_HIDDEN_STATE_PCA_SUBMODES:
            raise ValueError(
                "hidden_state_pca submode must be one of "
                f"{sorted(SUPPORTED_HIDDEN_STATE_PCA_SUBMODES)}, got: {self.submode}"
            )
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
        return HiddenStatePCASampleWriter(self, item, sample_index)

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
        fit_scope = PCA_HEAD_FIT_SCOPE if self.submode == PCA_SUBMODE_KEY_VALUE_HEADS else PCA_FIT_SCOPE
        manifest = {
            "run_dir": self.run_dir,
            "model_name": self.model_name,
            "result_path": self.out_file,
            "hidden_state_pca_fit_scope": fit_scope,
            "hidden_state_pca_submode": self.submode,
            "hidden_state_pca_states": list(
                PCA_HIDDEN_STATE_NAMES
                if self.submode == PCA_SUBMODE_HIDDEN_STATES
                else PCA_STATE_NAMES
            ),
            "hidden_state_pca_components": PCA_COMPONENT_COUNT,
            "hidden_state_pca_layers": self.layer_spec,
            "hidden_state_pca_token_start": self.token_start,
            "hidden_state_pca_token_end": self.token_end,
            "hidden_state_pca_token_scope": PCA_TOKEN_SCOPE,
            "hidden_state_pca_max_prefill_tokens": self.max_prefill_tokens,
            "hidden_state_pca_prefill_cap_mode": "fixed" if self.max_prefill_tokens is not None else "none",
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as fout:
            json.dump(manifest, fout, ensure_ascii=False, indent=2)


class HiddenStatePCASampleWriter:
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
            "submode": self.run_writer.submode,
            "prompt_text": prompt_text,
            "max_prefill_tokens": prefill_cap,
            "token_count": len(input_ids),
            "token_start": self.run_writer.token_start,
            "token_end": self.run_writer.token_end,
            "actual_token_start": None,
            "actual_token_end": None,
            "selected_token_count": None,
            "selected_tokens": [],
            "tokens": build_token_entries(tokenizer, input_ids),
            "pca_file": None,
            "key_plot_file": None,
            "value_plot_file": None,
            "hidden_plot_file": None,
            "key_angle_plot_file": None,
            "value_angle_plot_file": None,
            "hidden_angle_plot_file": None,
            "key_head_plot_files": [],
            "value_head_plot_files": [],
            "key_pca_shape": None,
            "value_pca_shape": None,
            "hidden_pca_shape": None,
            "key_head_pca_shape": None,
            "value_head_pca_shape": None,
            "layer_indices": [],
            "key_head_indices": [],
            "value_head_indices": [],
            "adjacent_layer_pairs": [],
            "key_adjacent_layer_angles_deg": [],
            "value_adjacent_layer_angles_deg": [],
            "hidden_adjacent_layer_angles_deg": [],
            "key_explained_variance_ratio": [],
            "value_explained_variance_ratio": [],
            "hidden_explained_variance_ratio": [],
            "key_head_explained_variance_ratio": [],
            "value_head_explained_variance_ratio": [],
            "token_scope": PCA_TOKEN_SCOPE,
        }
        self.prefills.append(record)
        self._write_sample_json()

        if prefill_cap is not None and len(input_ids) > prefill_cap:
            record["status"] = "skipped_over_cap"
            record["reason"] = f"Prompt token count {len(input_ids)} exceeds cap {prefill_cap}."
            self._write_sample_json()
            return record

        try:
            if self.run_writer.submode == PCA_SUBMODE_HIDDEN_STATES:
                self._capture_hidden_states_prefill(
                    record=record,
                    prefill_index=prefill_index,
                    label=label,
                    input_ids=input_ids,
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                )
                self._write_sample_json()
                return record

            if self.run_writer.submode == PCA_SUBMODE_KEY_VALUE_HEADS:
                self._capture_key_value_heads_prefill(
                    record=record,
                    prefill_index=prefill_index,
                    label=label,
                    input_ids=input_ids,
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                )
                self._write_sample_json()
                return record

            (
                key_pca_points,
                value_pca_points,
                layer_indices,
                key_components,
                value_components,
                key_mean,
                value_mean,
                key_explained,
                value_explained,
                actual_token_start,
                actual_token_end,
            ) = compute_key_value_state_pca(
                model=model,
                inputs=inputs,
                layer_spec=self.run_writer.layer_spec,
                token_start=self.run_writer.token_start,
                token_end=self.run_writer.token_end,
            )
            pca_file_name = f"prefill_{prefill_index:03d}_key_value_state_pca.npz"
            key_plot_file_name = f"prefill_{prefill_index:03d}_key_state_pca.png"
            value_plot_file_name = f"prefill_{prefill_index:03d}_value_state_pca.png"
            key_angle_plot_file_name = f"prefill_{prefill_index:03d}_key_state_layer_angle.png"
            value_angle_plot_file_name = f"prefill_{prefill_index:03d}_value_state_layer_angle.png"
            pca_path = os.path.join(self.sample_dir, pca_file_name)
            key_plot_path = os.path.join(self.sample_dir, key_plot_file_name)
            value_plot_path = os.path.join(self.sample_dir, value_plot_file_name)
            key_angle_plot_path = os.path.join(self.sample_dir, key_angle_plot_file_name)
            value_angle_plot_path = os.path.join(self.sample_dir, value_angle_plot_file_name)
            token_count = int(key_pca_points.shape[1])
            selected_token_ids = input_ids[actual_token_start:actual_token_end]
            key_layer_centers = compute_pca_layer_centers(key_pca_points)
            value_layer_centers = compute_pca_layer_centers(value_pca_points)
            key_adjacent_angles = compute_adjacent_layer_angles(key_layer_centers)
            value_adjacent_angles = compute_adjacent_layer_angles(value_layer_centers)
            adjacent_layer_pairs = build_adjacent_layer_pairs(layer_indices)
            np.savez_compressed(
                pca_path,
                key_pca_points=key_pca_points.astype(np.float32, copy=False),
                value_pca_points=value_pca_points.astype(np.float32, copy=False),
                layer_indices=np.asarray(layer_indices, dtype=np.int16),
                adjacent_layer_pairs=adjacent_layer_pairs,
                token_ids=np.asarray(selected_token_ids, dtype=np.int64),
                token_start=np.asarray(actual_token_start, dtype=np.int64),
                token_end=np.asarray(actual_token_end, dtype=np.int64),
                key_layer_centers=key_layer_centers.astype(np.float32, copy=False),
                value_layer_centers=value_layer_centers.astype(np.float32, copy=False),
                key_adjacent_layer_angles_deg=key_adjacent_angles.astype(np.float32, copy=False),
                value_adjacent_layer_angles_deg=value_adjacent_angles.astype(np.float32, copy=False),
                key_explained_variance_ratio=key_explained.astype(np.float32, copy=False),
                value_explained_variance_ratio=value_explained.astype(np.float32, copy=False),
                key_pca_mean=key_mean.astype(np.float32, copy=False),
                value_pca_mean=value_mean.astype(np.float32, copy=False),
                key_pca_components=key_components.astype(np.float32, copy=False),
                value_pca_components=value_components.astype(np.float32, copy=False),
                hidden_state_pca_fit_scope=np.asarray(PCA_FIT_SCOPE),
                hidden_state_pca_states=np.asarray(PCA_STATE_NAMES),
                hidden_state_pca_submode=np.asarray(self.run_writer.submode),
                hidden_state_pca_token_scope=np.asarray(PCA_TOKEN_SCOPE),
            )
            record["status"] = "saved"
            record["pca_file"] = pca_file_name
            record["key_pca_shape"] = list(key_pca_points.shape)
            record["value_pca_shape"] = list(value_pca_points.shape)
            record["layer_indices"] = [int(layer_idx) for layer_idx in layer_indices]
            record["adjacent_layer_pairs"] = adjacent_layer_pairs.astype(int).tolist()
            record["actual_token_start"] = int(actual_token_start)
            record["actual_token_end"] = int(actual_token_end)
            record["selected_token_count"] = token_count
            record["selected_tokens"] = build_token_entries(tokenizer, selected_token_ids)
            record["key_adjacent_layer_angles_deg"] = [
                None if not np.isfinite(value) else float(value)
                for value in key_adjacent_angles.tolist()
            ]
            record["value_adjacent_layer_angles_deg"] = [
                None if not np.isfinite(value) else float(value)
                for value in value_adjacent_angles.tolist()
            ]
            record["key_explained_variance_ratio"] = [float(value) for value in key_explained.tolist()]
            record["value_explained_variance_ratio"] = [float(value) for value in value_explained.tolist()]
            try:
                plot_hidden_state_pca(
                    pca_points=key_pca_points,
                    layer_indices=layer_indices,
                    output_path=key_plot_path,
                    title=f"{label}: key-state PCA",
                    empty_label="No tokens in selected span",
                )
                record["key_plot_file"] = key_plot_file_name
                plot_hidden_state_pca(
                    pca_points=value_pca_points,
                    layer_indices=layer_indices,
                    output_path=value_plot_path,
                    title=f"{label}: value-state PCA",
                    empty_label="No tokens in selected span",
                )
                record["value_plot_file"] = value_plot_file_name
                plot_adjacent_layer_angle_curve(
                    layer_indices=layer_indices,
                    angles_deg=key_adjacent_angles,
                    output_path=key_angle_plot_path,
                    title=f"{label}: key-state adjacent-layer center-vector angle",
                )
                record["key_angle_plot_file"] = key_angle_plot_file_name
                plot_adjacent_layer_angle_curve(
                    layer_indices=layer_indices,
                    angles_deg=value_adjacent_angles,
                    output_path=value_angle_plot_path,
                    title=f"{label}: value-state adjacent-layer center-vector angle",
                )
                record["value_angle_plot_file"] = value_angle_plot_file_name
            except Exception as plot_exc:
                record["status"] = "saved_npz_plot_error"
                record["plot_error"] = str(plot_exc)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)

        self._write_sample_json()
        return record

    def _capture_key_value_heads_prefill(self, record, prefill_index, label, input_ids, model, tokenizer, inputs):
        (
            key_pca_points,
            value_pca_points,
            layer_indices,
            key_head_indices,
            value_head_indices,
            key_components,
            value_components,
            key_mean,
            value_mean,
            key_explained,
            value_explained,
            actual_token_start,
            actual_token_end,
        ) = compute_key_value_head_state_pca(
            model=model,
            inputs=inputs,
            layer_spec=self.run_writer.layer_spec,
            token_start=self.run_writer.token_start,
            token_end=self.run_writer.token_end,
        )
        pca_file_name = f"prefill_{prefill_index:03d}_key_value_head_state_pca.npz"
        pca_path = os.path.join(self.sample_dir, pca_file_name)
        selected_token_ids = input_ids[actual_token_start:actual_token_end]
        np.savez_compressed(
            pca_path,
            key_head_pca_points=key_pca_points.astype(np.float32, copy=False),
            value_head_pca_points=value_pca_points.astype(np.float32, copy=False),
            layer_indices=np.asarray(layer_indices, dtype=np.int16),
            key_head_indices=np.asarray(key_head_indices, dtype=np.int16),
            value_head_indices=np.asarray(value_head_indices, dtype=np.int16),
            token_ids=np.asarray(selected_token_ids, dtype=np.int64),
            token_start=np.asarray(actual_token_start, dtype=np.int64),
            token_end=np.asarray(actual_token_end, dtype=np.int64),
            key_head_explained_variance_ratio=key_explained.astype(np.float32, copy=False),
            value_head_explained_variance_ratio=value_explained.astype(np.float32, copy=False),
            key_head_pca_mean=key_mean.astype(np.float32, copy=False),
            value_head_pca_mean=value_mean.astype(np.float32, copy=False),
            key_head_pca_components=key_components.astype(np.float32, copy=False),
            value_head_pca_components=value_components.astype(np.float32, copy=False),
            hidden_state_pca_fit_scope=np.asarray(PCA_HEAD_FIT_SCOPE),
            hidden_state_pca_states=np.asarray(PCA_STATE_NAMES),
            hidden_state_pca_submode=np.asarray(self.run_writer.submode),
            hidden_state_pca_token_scope=np.asarray(PCA_TOKEN_SCOPE),
        )
        record["status"] = "saved"
        record["pca_file"] = pca_file_name
        record["key_head_pca_shape"] = list(key_pca_points.shape)
        record["value_head_pca_shape"] = list(value_pca_points.shape)
        record["layer_indices"] = [int(layer_idx) for layer_idx in layer_indices]
        record["key_head_indices"] = [int(head_idx) for head_idx in key_head_indices]
        record["value_head_indices"] = [int(head_idx) for head_idx in value_head_indices]
        record["actual_token_start"] = int(actual_token_start)
        record["actual_token_end"] = int(actual_token_end)
        record["selected_token_count"] = int(key_pca_points.shape[2])
        record["selected_tokens"] = build_token_entries(tokenizer, selected_token_ids)
        record["key_head_explained_variance_ratio"] = key_explained.astype(float).tolist()
        record["value_head_explained_variance_ratio"] = value_explained.astype(float).tolist()
        try:
            for layer_pos, layer_idx in enumerate(layer_indices):
                key_plot_file_name = f"prefill_{prefill_index:03d}_layer_{int(layer_idx):03d}_key_head_pca.png"
                value_plot_file_name = f"prefill_{prefill_index:03d}_layer_{int(layer_idx):03d}_value_head_pca.png"
                key_plot_path = os.path.join(self.sample_dir, key_plot_file_name)
                value_plot_path = os.path.join(self.sample_dir, value_plot_file_name)
                plot_head_pca_layer(
                    pca_points=key_pca_points[layer_pos],
                    head_indices=key_head_indices,
                    output_path=key_plot_path,
                    title=f"{label}: layer {int(layer_idx)} key heads PCA",
                    empty_label="No tokens in selected span",
                )
                plot_head_pca_layer(
                    pca_points=value_pca_points[layer_pos],
                    head_indices=value_head_indices,
                    output_path=value_plot_path,
                    title=f"{label}: layer {int(layer_idx)} value heads PCA",
                    empty_label="No tokens in selected span",
                )
                record["key_head_plot_files"].append(key_plot_file_name)
                record["value_head_plot_files"].append(value_plot_file_name)
        except Exception as plot_exc:
            record["status"] = "saved_npz_plot_error"
            record["plot_error"] = str(plot_exc)

    def _capture_hidden_states_prefill(self, record, prefill_index, label, input_ids, model, tokenizer, inputs):
        (
            pca_points,
            layer_indices,
            components,
            mean,
            explained,
            actual_token_start,
            actual_token_end,
        ) = compute_hidden_state_pca(
            model=model,
            inputs=inputs,
            layer_spec=self.run_writer.layer_spec,
            token_start=self.run_writer.token_start,
            token_end=self.run_writer.token_end,
        )
        pca_file_name = f"prefill_{prefill_index:03d}_hidden_state_pca.npz"
        plot_file_name = f"prefill_{prefill_index:03d}_hidden_state_pca.png"
        angle_plot_file_name = f"prefill_{prefill_index:03d}_hidden_state_layer_angle.png"
        pca_path = os.path.join(self.sample_dir, pca_file_name)
        plot_path = os.path.join(self.sample_dir, plot_file_name)
        angle_plot_path = os.path.join(self.sample_dir, angle_plot_file_name)
        token_count = int(pca_points.shape[1])
        selected_token_ids = input_ids[actual_token_start:actual_token_end]
        layer_centers = compute_pca_layer_centers(pca_points)
        adjacent_angles = compute_adjacent_layer_angles(layer_centers)
        adjacent_layer_pairs = build_adjacent_layer_pairs(layer_indices)
        np.savez_compressed(
            pca_path,
            hidden_pca_points=pca_points.astype(np.float32, copy=False),
            layer_indices=np.asarray(layer_indices, dtype=np.int16),
            adjacent_layer_pairs=adjacent_layer_pairs,
            token_ids=np.asarray(selected_token_ids, dtype=np.int64),
            token_start=np.asarray(actual_token_start, dtype=np.int64),
            token_end=np.asarray(actual_token_end, dtype=np.int64),
            hidden_layer_centers=layer_centers.astype(np.float32, copy=False),
            hidden_adjacent_layer_angles_deg=adjacent_angles.astype(np.float32, copy=False),
            hidden_explained_variance_ratio=explained.astype(np.float32, copy=False),
            hidden_pca_mean=mean.astype(np.float32, copy=False),
            hidden_pca_components=components.astype(np.float32, copy=False),
            hidden_state_pca_fit_scope=np.asarray(PCA_FIT_SCOPE),
            hidden_state_pca_states=np.asarray(PCA_HIDDEN_STATE_NAMES),
            hidden_state_pca_submode=np.asarray(self.run_writer.submode),
            hidden_state_pca_token_scope=np.asarray(PCA_TOKEN_SCOPE),
        )
        record["status"] = "saved"
        record["pca_file"] = pca_file_name
        record["hidden_pca_shape"] = list(pca_points.shape)
        record["layer_indices"] = [int(layer_idx) for layer_idx in layer_indices]
        record["adjacent_layer_pairs"] = adjacent_layer_pairs.astype(int).tolist()
        record["actual_token_start"] = int(actual_token_start)
        record["actual_token_end"] = int(actual_token_end)
        record["selected_token_count"] = token_count
        record["selected_tokens"] = build_token_entries(tokenizer, selected_token_ids)
        record["hidden_adjacent_layer_angles_deg"] = [
            None if not np.isfinite(value) else float(value)
            for value in adjacent_angles.tolist()
        ]
        record["hidden_explained_variance_ratio"] = [float(value) for value in explained.tolist()]
        try:
            plot_hidden_state_pca(
                pca_points=pca_points,
                layer_indices=layer_indices,
                output_path=plot_path,
                title=f"{label}: hidden-state PCA",
                empty_label="No tokens in selected span",
            )
            record["hidden_plot_file"] = plot_file_name
            plot_adjacent_layer_angle_curve(
                layer_indices=layer_indices,
                angles_deg=adjacent_angles,
                output_path=angle_plot_path,
                title=f"{label}: hidden-state adjacent-layer center-vector angle",
            )
            record["hidden_angle_plot_file"] = angle_plot_file_name
        except Exception as plot_exc:
            record["status"] = "saved_npz_plot_error"
            record["plot_error"] = str(plot_exc)


    def finalize(self, item):
        self.result_summary = extract_result_summary(item)
        self._write_sample_json()
        self.run_writer.register_sample(self)

    def build_capture_status(self):
        keys = [
            "prefill_index",
            "label",
            "status",
            "submode",
            "token_count",
            "actual_token_start",
            "actual_token_end",
            "selected_token_count",
            "key_pca_shape",
            "value_pca_shape",
            "hidden_pca_shape",
            "key_head_pca_shape",
            "value_head_pca_shape",
            "layer_indices",
            "key_head_indices",
            "value_head_indices",
            "adjacent_layer_pairs",
            "key_adjacent_layer_angles_deg",
            "value_adjacent_layer_angles_deg",
            "hidden_adjacent_layer_angles_deg",
            "key_explained_variance_ratio",
            "value_explained_variance_ratio",
            "hidden_explained_variance_ratio",
            "key_head_explained_variance_ratio",
            "value_head_explained_variance_ratio",
            "key_head_plot_files",
            "value_head_plot_files",
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
