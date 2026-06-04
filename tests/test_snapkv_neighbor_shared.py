import json
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from build_hidden_mix_profile import build_profile_from_npz, write_profile
from models.compression.methods.snapkv_neighbor_shared import (
    SnapKVNeighborShared,
    masked_eager_attention_forward,
)


class FakePast:
    def __init__(self, num_layers):
        self.layers = [types.SimpleNamespace(query_cache=None) for _ in range(num_layers)]

    def update(self, key_states, value_states, layer_idx):
        layer = self.layers[layer_idx]
        if getattr(layer, "keys", None) is None:
            layer.keys = key_states
            layer.values = value_states
        else:
            layer.keys = torch.cat([layer.keys, key_states], dim=-2)
            layer.values = torch.cat([layer.values, value_states], dim=-2)
        return layer.keys, layer.values


class FakeAttention(torch.nn.Module):
    def __init__(self, hidden_size=4, num_heads=2, head_dim=2, attn_implementation=None):
        super().__init__()
        self.head_dim = head_dim
        self.config = types.SimpleNamespace(num_attention_heads=num_heads)
        if attn_implementation is not None:
            self.config._attn_implementation = attn_implementation
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.is_causal = True
        self.training = False
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(hidden_size))


def _make_config(num_layers=3, compression=None):
    return types.SimpleNamespace(
        update_kv=True,
        compression=compression,
        num_hidden_layers=num_layers,
    )


def _position_embeddings(seq_len, head_dim=2, device=None, dtype=None):
    cos = torch.ones(1, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    return cos, sin


def _update_and_finalize(
    compressor,
    attention,
    hidden,
    position_embeddings,
    key,
    query,
    value,
    past,
    layer_cache,
    attn_output=None,
):
    result = compressor.update_kv_cache(
        attention,
        hidden,
        position_embeddings,
        key,
        query,
        value,
        past,
        layer_cache,
    )
    if attn_output is None:
        attn_output = hidden
    compressor.finalize_after_attention(attention, hidden, attn_output, layer_cache)
    return result


def _write_profile(tmp_path, profile):
    profile_path = tmp_path / "hidden_mix_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return str(profile_path)


def _write_similarity_npz(tmp_path, similarity, layer_indices=None, actual_window_size=2):
    similarity = np.asarray(similarity, dtype=np.float32)
    if layer_indices is None:
        layer_indices = np.arange(similarity.shape[0], dtype=np.int16)
    similarity_path = tmp_path / "query_window_similarity.npz"
    np.savez_compressed(
        similarity_path,
        similarity=similarity,
        layer_indices=np.asarray(layer_indices, dtype=np.int16),
        actual_window_size=np.asarray(actual_window_size, dtype=np.int64),
        query_window_similarity_state=np.asarray("hidden_states"),
    )
    return str(similarity_path)


def _write_attn_output_ratio_npz(tmp_path, ratios, layer_indices=None):
    ratios = np.asarray(ratios, dtype=np.float32)
    if layer_indices is None:
        layer_indices = np.arange(ratios.shape[0], dtype=np.int16)
    ratio_path = tmp_path / "attn_output_ratio.npz"
    np.savez_compressed(
        ratio_path,
        ratios=ratios,
        layer_indices=np.asarray(layer_indices, dtype=np.int16),
    )
    return str(ratio_path)


def test_single_sample_profile_builder_splits_groups_and_normalizes_weights(tmp_path):
    similarity = np.array(
        [
            [1.0, 0.90, 0.20, 0.10, 0.10],
            [0.90, 1.0, 0.84, 0.20, 0.10],
            [0.20, 0.84, 1.0, 0.95, 0.92],
            [0.10, 0.20, 0.95, 1.0, 0.96],
            [0.10, 0.10, 0.92, 0.96, 1.0],
        ],
        dtype=np.float32,
    )
    similarity_path = _write_similarity_npz(tmp_path, similarity)
    profile = build_profile_from_npz(
        similarity_path,
        group_threshold=0.85,
        max_group_size=2,
        temperature=0.1,
        min_weight=0.0,
    )

    assert [group["layers"] for group in profile["groups"]] == [[0, 1], [2, 3], [4]]
    assert profile["source"] == "single_sample"
    assert profile["actual_window_size"] == 2
    assert profile["group_sizes"] == [2, 2, 1]
    for group in profile["groups"]:
        for mix in group["mix"].values():
            assert abs(sum(mix["weights"]) - 1.0) < 1e-6
            assert set(mix["sources"]).issubset(set(group["layers"]))


def test_single_sample_profile_builder_adds_ratio_budget_weights(tmp_path):
    similarity = np.array(
        [
            [1.0, 0.90, 0.20, 0.10],
            [0.90, 1.0, 0.20, 0.10],
            [0.20, 0.20, 1.0, 0.90],
            [0.10, 0.10, 0.90, 1.0],
        ],
        dtype=np.float32,
    )
    ratios = np.array(
        [
            [0.0, 2.0],
            [1.0, 3.0],
            [0.0, 4.0],
            [2.0, 6.0],
        ],
        dtype=np.float32,
    )
    profile = build_profile_from_npz(
        _write_similarity_npz(tmp_path, similarity),
        attn_output_ratio_npz=_write_attn_output_ratio_npz(tmp_path, ratios),
        group_threshold=0.85,
        max_group_size=2,
        temperature=0.1,
        min_weight=0.0,
    )

    assert [group["layers"] for group in profile["groups"]] == [[0, 1], [2, 3]]
    assert profile["budget_weight_metric"] == "attn_output_hidden_l2_ratio_variance_sum"
    assert profile["groups"][0]["attn_output_ratio_variance_sum"] == 2.0
    assert profile["groups"][1]["attn_output_ratio_variance_sum"] == 8.0
    assert np.isclose(profile["groups"][0]["budget_weight"], 0.2)
    assert np.isclose(profile["groups"][1]["budget_weight"], 0.8)
    assert "attn_output_ratio_npz" in profile


def test_single_sample_profile_builder_rejects_missing_ratio_layers(tmp_path):
    similarity = np.array([[1.0, 0.9], [0.9, 1.0]], dtype=np.float32)
    try:
        build_profile_from_npz(
            _write_similarity_npz(tmp_path, similarity, layer_indices=[0, 1]),
            attn_output_ratio_npz=_write_attn_output_ratio_npz(tmp_path, [[1.0, 1.0]], layer_indices=[0]),
        )
    except ValueError:
        return
    raise AssertionError("Missing attn-output ratio layer was accepted.")


def test_single_sample_profile_builder_min_weight_fallbacks_to_self(tmp_path):
    similarity = np.array([[1.0, 0.99], [0.99, 1.0]], dtype=np.float32)
    profile = build_profile_from_npz(
        _write_similarity_npz(tmp_path, similarity),
        group_threshold=0.85,
        max_group_size=6,
        temperature=0.05,
        min_weight=1.1,
    )

    assert profile["groups"][0]["mix"]["0"] == {"sources": [0], "weights": [1.0]}
    assert profile["groups"][0]["mix"]["1"] == {"sources": [1], "weights": [1.0]}


def test_single_sample_profile_builder_rejects_invalid_npz(tmp_path):
    bad_path = tmp_path / "bad_similarity.npz"
    np.savez_compressed(
        bad_path,
        similarity=np.ones((2, 3), dtype=np.float32),
        layer_indices=np.asarray([0, 1], dtype=np.int16),
    )
    try:
        build_profile_from_npz(str(bad_path))
    except ValueError:
        return
    raise AssertionError("Invalid non-square similarity npz was accepted.")


def test_builder_generated_profile_runs_neighbor_shared_runtime(tmp_path):
    torch.manual_seed(23)
    batch, kv_heads, seq_len, head_dim = 1, 2, 8, 2
    hidden_size = kv_heads * head_dim
    similarity = np.full((4, 4), 0.92, dtype=np.float32)
    np.fill_diagonal(similarity, 1.0)
    profile = build_profile_from_npz(
        _write_similarity_npz(tmp_path, similarity),
        group_threshold=0.85,
        max_group_size=4,
        temperature=0.1,
        min_weight=0.0,
    )
    profile_path = write_profile(profile, tmp_path / "profile.json")

    config = _make_config(num_layers=4, compression=None)
    past = FakePast(num_layers=4)
    attentions = [FakeAttention(hidden_size, kv_heads, head_dim) for _ in range(4)]
    compressors = [
        SnapKVNeighborShared(
            budget=4,
            window_size=2,
            kernel_size=1,
            layer_idx=i,
            model_config=config,
            hidden_mix_profile_path=profile_path,
        )
        for i in range(4)
    ]

    for layer_idx in range(4):
        hidden = torch.randn(batch, seq_len, hidden_size)
        key = torch.randn(batch, kv_heads, seq_len, head_dim)
        value = torch.randn(batch, kv_heads, seq_len, head_dim)
        query = hidden.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
        past.layers[layer_idx].query_cache = query[:, :, -2:, :]
        _update_and_finalize(
            compressors[layer_idx],
            attentions[layer_idx],
            hidden,
            _position_embeddings(seq_len, head_dim),
            key,
            query,
            value,
            past,
            past.layers[layer_idx],
        )

    assert config._snapkv_neighbor_shared_state["groups"] == {}
    assert all(hasattr(layer, "kv_valid_mask") for layer in past.layers)


def test_triplet_shared_topk_packs_variable_head_lengths_and_recent_window():
    torch.manual_seed(7)
    batch, kv_heads, seq_len, head_dim = 1, 2, 8, 2
    hidden_size = kv_heads * head_dim
    config = _make_config(num_layers=3, compression=None)
    past = FakePast(num_layers=3)
    attentions = [FakeAttention(hidden_size, kv_heads, head_dim) for _ in range(3)]
    compressors = [
        SnapKVNeighborShared(budget=4, window_size=2, kernel_size=1, layer_idx=i, model_config=config)
        for i in range(3)
    ]

    for layer_idx in range(3):
        hidden = torch.randn(batch, seq_len, hidden_size)
        key = torch.randn(batch, kv_heads, seq_len, head_dim)
        value = torch.randn(batch, kv_heads, seq_len, head_dim)
        query = hidden.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
        past.layers[layer_idx].query_cache = query[:, :, -2:, :]
        _update_and_finalize(
            compressors[layer_idx],
            attentions[layer_idx],
            hidden,
            _position_embeddings(seq_len, head_dim),
            key,
            query,
            value,
            past,
            past.layers[layer_idx],
        )
        if layer_idx < 2:
            assert getattr(past.layers[layer_idx], "keys", None) is None

    assert config._snapkv_neighbor_shared_state["groups"] == {}
    total_historical = 0
    all_lengths = []
    for layer in past.layers:
        assert hasattr(layer, "kv_valid_mask")
        assert layer.kv_valid_mask.shape[:2] == (batch, kv_heads)
        assert torch.equal(layer.kv_valid_mask.sum(dim=-1), layer.kv_lengths)
        assert torch.all(layer.kv_lengths >= 2)
        total_historical += int((layer.kv_lengths - 2).sum().item())
        all_lengths.extend(layer.kv_lengths.reshape(-1).tolist())

    assert total_historical == 3 * kv_heads * (4 - 2)
    assert len(set(all_lengths)) > 1


def test_shared_global_rank_scores_are_normalized_per_layer_head():
    compressor = SnapKVNeighborShared(
        budget=4,
        window_size=2,
        kernel_size=1,
        layer_idx=0,
        model_config=_make_config(num_layers=1),
    )
    scores = torch.tensor(
        [
            [
                [10.0, 12.0, 14.0, 999.0],
                [100.0, 110.0, 120.0, 130.0],
            ]
        ]
    )
    valid_mask = torch.tensor(
        [
            [
                [True, True, True, False],
                [True, True, True, True],
            ]
        ]
    )

    normalized = compressor._normalize_scores_for_global_rank(scores, valid_mask)

    assert normalized[0, 0, 3] == torch.finfo(normalized.dtype).min
    for head_idx in range(scores.shape[1]):
        valid = valid_mask[0, head_idx]
        normalized_values = normalized[0, head_idx, valid]
        score_values = scores[0, head_idx, valid]
        assert torch.allclose(normalized_values.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(normalized_values.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
        assert torch.equal(normalized_values.argsort(), score_values.argsort())


def test_attn_output_ratio_weights_historical_token_selection(tmp_path):
    batch, kv_heads, seq_len, head_dim = 1, 1, 4, 2
    hidden_size = kv_heads * head_dim
    profile_path = _write_profile(
        tmp_path,
        {
            "metric": "cosine",
            "groups": [
                {
                    "layers": [0],
                    "mix": {"0": {"sources": [0], "weights": [1.0]}},
                }
            ],
        },
    )
    config = _make_config(num_layers=1, compression=None)
    past = FakePast(num_layers=1)
    attention = FakeAttention(hidden_size, kv_heads, head_dim)
    compressor = SnapKVNeighborShared(
        budget=2,
        window_size=1,
        kernel_size=1,
        layer_idx=0,
        model_config=config,
        hidden_mix_profile_path=profile_path,
    )

    hidden = torch.ones(batch, seq_len, hidden_size)
    key = torch.zeros(batch, kv_heads, seq_len, head_dim)
    value = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1).expand(-1, -1, -1, head_dim).clone()
    query = hidden.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    past.layers[0].query_cache = query[:, :, -1:, :]
    attn_output = hidden.clone()
    attn_output[:, 1, :] *= 10.0

    _update_and_finalize(
        compressor,
        attention,
        hidden,
        _position_embeddings(seq_len, head_dim),
        key,
        query,
        value,
        past,
        past.layers[0],
        attn_output=attn_output,
    )

    kept_values = past.layers[0].values[0, 0, :, 0]
    assert torch.equal(kept_values, torch.tensor([1.0, 3.0]))


def test_profile_four_layer_group_uses_variable_shared_budget(tmp_path):
    torch.manual_seed(17)
    batch, kv_heads, seq_len, head_dim = 1, 2, 8, 2
    hidden_size = kv_heads * head_dim
    profile_path = _write_profile(
        tmp_path,
        {
            "metric": "cosine",
            "groups": [
                {
                    "layers": [0, 1, 2, 3],
                    "mix": {
                        str(layer_idx): {"sources": [layer_idx], "weights": [1.0]}
                        for layer_idx in range(4)
                    },
                }
            ],
        },
    )
    config = _make_config(num_layers=4, compression=None)
    past = FakePast(num_layers=4)
    attentions = [FakeAttention(hidden_size, kv_heads, head_dim) for _ in range(4)]
    compressors = [
        SnapKVNeighborShared(
            budget=4,
            window_size=2,
            kernel_size=1,
            layer_idx=i,
            model_config=config,
            hidden_mix_profile_path=profile_path,
        )
        for i in range(4)
    ]

    for layer_idx in range(4):
        hidden = torch.randn(batch, seq_len, hidden_size)
        key = torch.randn(batch, kv_heads, seq_len, head_dim)
        value = torch.randn(batch, kv_heads, seq_len, head_dim)
        query = hidden.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
        past.layers[layer_idx].query_cache = query[:, :, -2:, :]
        _update_and_finalize(
            compressors[layer_idx],
            attentions[layer_idx],
            hidden,
            _position_embeddings(seq_len, head_dim),
            key,
            query,
            value,
            past,
            past.layers[layer_idx],
        )
        if layer_idx < 3:
            assert getattr(past.layers[layer_idx], "keys", None) is None

    assert config._snapkv_neighbor_shared_state["groups"] == {}
    total_historical = 0
    for layer in past.layers:
        assert hasattr(layer, "kv_valid_mask")
        total_historical += int((layer.kv_lengths - 2).sum().item())
    assert total_historical == 4 * kv_heads * (4 - 2)


def test_profile_budget_weights_shift_historical_budget_between_groups(tmp_path):
    torch.manual_seed(29)
    batch, kv_heads, seq_len, head_dim = 1, 2, 8, 2
    hidden_size = kv_heads * head_dim
    profile_path = _write_profile(
        tmp_path,
        {
            "metric": "cosine",
            "groups": [
                {
                    "layers": [0, 1],
                    "budget_weight": 0.75,
                    "mix": {str(layer_idx): {"sources": [layer_idx], "weights": [1.0]} for layer_idx in [0, 1]},
                },
                {
                    "layers": [2, 3],
                    "budget_weight": 0.25,
                    "mix": {str(layer_idx): {"sources": [layer_idx], "weights": [1.0]} for layer_idx in [2, 3]},
                },
            ],
        },
    )
    config = _make_config(num_layers=4, compression=None)
    past = FakePast(num_layers=4)
    attentions = [FakeAttention(hidden_size, kv_heads, head_dim) for _ in range(4)]
    compressors = [
        SnapKVNeighborShared(
            budget=4,
            window_size=2,
            kernel_size=1,
            layer_idx=i,
            model_config=config,
            hidden_mix_profile_path=profile_path,
        )
        for i in range(4)
    ]

    for layer_idx in range(4):
        hidden = torch.randn(batch, seq_len, hidden_size)
        key = torch.randn(batch, kv_heads, seq_len, head_dim)
        value = torch.randn(batch, kv_heads, seq_len, head_dim)
        query = hidden.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
        past.layers[layer_idx].query_cache = query[:, :, -2:, :]
        _update_and_finalize(
            compressors[layer_idx],
            attentions[layer_idx],
            hidden,
            _position_embeddings(seq_len, head_dim),
            key,
            query,
            value,
            past,
            past.layers[layer_idx],
        )

    group_a_hist = sum(int((past.layers[layer_idx].kv_lengths - 2).sum().item()) for layer_idx in [0, 1])
    group_b_hist = sum(int((past.layers[layer_idx].kv_lengths - 2).sum().item()) for layer_idx in [2, 3])
    assert group_a_hist == 12
    assert group_b_hist == 4
    assert group_a_hist > group_b_hist
    assert group_a_hist + group_b_hist == 4 * kv_heads * (4 - 2)


def test_profile_missing_target_mix_falls_back_to_self_hidden(tmp_path):
    batch, kv_heads, seq_len, head_dim = 1, 2, 5, 2
    hidden_size = kv_heads * head_dim
    profile_path = _write_profile(
        tmp_path,
        {
            "metric": "cosine",
            "groups": [
                {
                    "layers": [0, 1],
                    "mix": {"0": {"sources": [1], "weights": [1.0]}},
                }
            ],
        },
    )
    config = _make_config(num_layers=2, compression=None)
    past = FakePast(num_layers=2)
    attentions = [FakeAttention(hidden_size, kv_heads, head_dim) for _ in range(2)]
    for layer_idx, attention in enumerate(attentions):
        attention.layer_idx = layer_idx
    compressors = [
        SnapKVNeighborShared(
            budget=3,
            window_size=2,
            kernel_size=1,
            layer_idx=i,
            model_config=config,
            hidden_mix_profile_path=profile_path,
        )
        for i in range(2)
    ]
    recorded_hidden = {}

    def record_project_query(attention, hidden_window, position_embeddings):
        recorded_hidden[int(attention.layer_idx)] = hidden_window.detach().clone()
        return hidden_window.reshape(batch, -1, kv_heads, head_dim).transpose(1, 2)

    compressors[-1]._project_query_window = record_project_query
    hidden_layers = [
        torch.zeros(batch, seq_len, hidden_size),
        torch.full((batch, seq_len, hidden_size), 3.0),
    ]

    for layer_idx, hidden in enumerate(hidden_layers):
        key = torch.randn(batch, kv_heads, seq_len, head_dim)
        value = torch.randn(batch, kv_heads, seq_len, head_dim)
        query = hidden.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
        past.layers[layer_idx].query_cache = query[:, :, -2:, :]
        _update_and_finalize(
            compressors[layer_idx],
            attentions[layer_idx],
            hidden,
            _position_embeddings(seq_len, head_dim),
            key,
            query,
            value,
            past,
            past.layers[layer_idx],
        )

    expected_self = hidden_layers[1][:, -2:, :]
    assert torch.equal(recorded_hidden[0], expected_self)
    assert torch.equal(recorded_hidden[1], expected_self)


def _require_cuda_flatten_deps():
    try:
        import pytest
    except ImportError:
        pytest = None

    if not torch.cuda.is_available():
        if pytest is not None:
            pytest.skip("CUDA is required for snapkv_neighbor_shared flatten cache tests.")
        raise RuntimeError("CUDA is required for snapkv_neighbor_shared flatten cache tests.")
    if pytest is not None:
        pytest.importorskip("tiny_api_cuda")
        pytest.importorskip("flash_attn")
    else:
        __import__("tiny_api_cuda")
        __import__("flash_attn")


def _build_flatten_cache(dtype=torch.float16):
    _require_cuda_flatten_deps()
    torch.manual_seed(37)
    device = torch.device("cuda")
    batch, kv_heads, seq_len, head_dim = 1, 2, 8, 8
    hidden_size = kv_heads * head_dim
    config = _make_config(num_layers=3, compression=None)
    past = FakePast(num_layers=3)
    attentions = [
        FakeAttention(hidden_size, kv_heads, head_dim, attn_implementation="flash_attention_2").to(device=device, dtype=dtype)
        for _ in range(3)
    ]
    compressors = [
        SnapKVNeighborShared(
            budget=4,
            window_size=2,
            kernel_size=1,
            layer_idx=i,
            model_config=config,
            flatten_cache=True,
        )
        for i in range(3)
    ]

    for layer_idx in range(3):
        hidden = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
        key = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        value = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        query = hidden.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
        past.layers[layer_idx].query_cache = query[:, :, -2:, :]
        _update_and_finalize(
            compressors[layer_idx],
            attentions[layer_idx],
            hidden,
            _position_embeddings(seq_len, head_dim, device=device, dtype=dtype),
            key,
            query,
            value,
            past,
            past.layers[layer_idx],
        )

    return past, compressors, attentions, (batch, kv_heads, head_dim, hidden_size), dtype


def test_flatten_cache_metadata_cuda():
    past, _, _, (_, kv_heads, _, _), _ = _build_flatten_cache()

    for layer in past.layers:
        assert layer.keys.ndim == 2
        assert layer.values.ndim == 2
        assert layer.kv_flatten_enabled is True
        assert layer.kv_num_heads == kv_heads
        assert not hasattr(layer, "kv_valid_mask")
        assert int(layer.kv_head_lens.sum().item()) == layer.keys.shape[0]
        assert int(layer.kv_cu_lens[-1].item()) == layer.keys.shape[0]
        assert layer.kv_max_seqlen == int(layer.kv_head_lens.max().item())
        assert layer.get_seq_length() == layer.kv_max_seqlen
        assert torch.all(layer.kv_head_lens >= 2)


def test_flatten_cache_decode_update_cuda():
    past, compressors, attentions, (batch, kv_heads, head_dim, hidden_size), dtype = _build_flatten_cache()
    device = torch.device("cuda")
    layer_idx = 0
    layer = past.layers[layer_idx]
    old_total = layer.keys.shape[0]
    old_lens = layer.kv_head_lens.clone()

    hidden = torch.randn(batch, 1, hidden_size, device=device, dtype=dtype)
    key = torch.randn(batch, kv_heads, 1, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch, kv_heads, 1, head_dim, device=device, dtype=dtype)
    query = hidden.view(batch, 1, kv_heads, head_dim).transpose(1, 2)

    _update_and_finalize(
        compressors[layer_idx],
        attentions[layer_idx],
        hidden,
        _position_embeddings(1, head_dim, device=device, dtype=dtype),
        key,
        query,
        value,
        past,
        layer,
    )

    expected_lens = old_lens + 1
    expected_cu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(expected_lens, dim=0, dtype=torch.int32),
        ]
    )
    assert layer.keys.shape[0] == old_total + kv_heads
    assert torch.equal(layer.kv_head_lens, expected_lens)
    assert torch.equal(layer.kv_cu_lens, expected_cu)
    assert layer.kv_max_seqlen == int(expected_lens.max().item())
    assert layer.get_seq_length() == layer.kv_max_seqlen


def test_flatten_varlen_attention_smoke_cuda():
    _require_cuda_flatten_deps()
    from models.compression.modeling import flatten_varlen_attention_forward

    device = torch.device("cuda")
    dtype = torch.float16
    kv_heads, groups, head_dim = 2, 2, 8
    cu_lens = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    layer_cache = types.SimpleNamespace(
        kv_num_heads=kv_heads,
        kv_cu_lens=cu_lens,
        kv_max_seqlen=3,
    )
    module = types.SimpleNamespace(
        head_dim=head_dim,
        training=False,
        is_causal=True,
        config=types.SimpleNamespace(_attn_implementation="flash_attention_2"),
    )
    query = torch.randn(1, kv_heads * groups, 1, head_dim, device=device, dtype=dtype)
    key = torch.randn(int(cu_lens[-1].item()), head_dim, device=device, dtype=dtype)
    value = torch.randn_like(key)

    output, attn_weights = flatten_varlen_attention_forward(module, query, key, value, layer_cache, scaling=1.0)

    assert output.shape == (1, 1, kv_heads * groups, head_dim)
    assert attn_weights is None


def test_hidden_mix_profile_rejects_invalid_json(tmp_path):
    invalid_profiles = [
        {"groups": [{"layers": [0, 0]}]},
        {"groups": [{"layers": [0, 1], "mix": {"0": {"sources": [0], "weights": [0.5, 0.5]}}}]},
        {"groups": [{"layers": [0, 1], "mix": {"0": {"sources": [2], "weights": [1.0]}}}]},
    ]
    for profile in invalid_profiles:
        profile_path = _write_profile(tmp_path, profile)
        try:
            SnapKVNeighborShared(
                budget=4,
                window_size=2,
                kernel_size=1,
                layer_idx=0,
                model_config=_make_config(num_layers=2, compression=None),
                hidden_mix_profile_path=profile_path,
            )
        except ValueError:
            continue
        raise AssertionError(f"Invalid hidden mix profile was accepted: {profile}")


def test_masked_attention_ignores_padded_values():
    module = types.SimpleNamespace(head_dim=1, training=False)
    query = torch.tensor([[[[1.0]]]])
    key = torch.tensor([[[[1.0], [100.0], [1.0]]]])
    value = torch.tensor([[[[1.0], [1000.0], [3.0]]]])
    mask = torch.tensor([[[True, False, True]]])

    output, _ = masked_eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=torch.zeros(1, 1, 1, 2),
        kv_valid_mask=mask,
        scaling=1.0,
    )

    assert output.shape == (1, 1, 1, 1)
    assert torch.allclose(output.squeeze(), torch.tensor(2.0), atol=1e-5)


def test_incomplete_tail_layer_uses_original_snapkv_shape():
    torch.manual_seed(11)
    batch, kv_heads, seq_len, head_dim = 1, 2, 8, 2
    hidden_size = kv_heads * head_dim
    config = _make_config(num_layers=4, compression=None)
    past = FakePast(num_layers=4)
    layer_idx = 3
    attention = FakeAttention(hidden_size, kv_heads, head_dim)
    compressor = SnapKVNeighborShared(
        budget=4,
        window_size=2,
        kernel_size=1,
        layer_idx=layer_idx,
        model_config=config,
    )
    hidden = torch.randn(batch, seq_len, hidden_size)
    key = torch.randn(batch, kv_heads, seq_len, head_dim)
    value = torch.randn(batch, kv_heads, seq_len, head_dim)
    query = hidden.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    past.layers[layer_idx].query_cache = query[:, :, -2:, :]

    compressor.update_kv_cache(
        attention,
        hidden,
        _position_embeddings(seq_len, head_dim),
        key,
        query,
        value,
        past,
        past.layers[layer_idx],
    )

    assert past.layers[layer_idx].keys.shape == (batch, kv_heads, 4, head_dim)
    assert past.layers[layer_idx].values.shape == (batch, kv_heads, 4, head_dim)
    assert not hasattr(past.layers[layer_idx], "kv_valid_mask")


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        test_single_sample_profile_builder_splits_groups_and_normalizes_weights(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_single_sample_profile_builder_adds_ratio_budget_weights(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_single_sample_profile_builder_rejects_missing_ratio_layers(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_single_sample_profile_builder_min_weight_fallbacks_to_self(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_single_sample_profile_builder_rejects_invalid_npz(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_builder_generated_profile_runs_neighbor_shared_runtime(Path(tmp))
    test_triplet_shared_topk_packs_variable_head_lengths_and_recent_window()
    test_shared_global_rank_scores_are_normalized_per_layer_head()
    with TemporaryDirectory() as tmp:
        test_attn_output_ratio_weights_historical_token_selection(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_profile_four_layer_group_uses_variable_shared_budget(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_profile_budget_weights_shift_historical_budget_between_groups(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_profile_missing_target_mix_falls_back_to_self_hidden(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_hidden_mix_profile_rejects_invalid_json(Path(tmp))
    test_masked_attention_ignores_padded_values()
    test_incomplete_tail_layer_uses_original_snapkv_shape()
