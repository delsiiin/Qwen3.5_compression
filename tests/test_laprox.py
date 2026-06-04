import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.compression.methods.laprox import LaProx
from models.compression.methods.snapkv_neighbor_shared import masked_eager_attention_forward


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
    def __init__(self, num_attention_heads=2, num_key_value_heads=2, head_dim=2):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.config = types.SimpleNamespace(num_attention_heads=num_attention_heads)
        self.o_proj = torch.nn.Linear(num_attention_heads * head_dim, num_attention_heads * head_dim, bias=False)
        self.training = False
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(num_attention_heads * head_dim))


def _make_config(num_layers=1, compression=None):
    return types.SimpleNamespace(update_kv=True, compression=compression, num_hidden_layers=num_layers)


def _run_prefill(compressors, attentions, past, keys, queries, values):
    for layer_idx, compressor in enumerate(compressors):
        past.layers[layer_idx].query_cache = queries[layer_idx][:, :, -compressor.window_size :, :]
        compressor.update_kv_cache(
            attentions[layer_idx],
            torch.empty(keys[layer_idx].shape[0], keys[layer_idx].shape[-2], 1),
            None,
            keys[layer_idx],
            queries[layer_idx],
            values[layer_idx],
            past,
            past.layers[layer_idx],
        )


def test_registry_can_instantiate_laprox():
    from models.compression.modeling import KV_COMPRESSION_MAP

    assert KV_COMPRESSION_MAP["laprox"] is LaProx
    compressor = KV_COMPRESSION_MAP["laprox"](budget=4, window_size=2, model_config=_make_config())
    assert compressor.requires_layer_coordination is True


def test_laprox_preserves_recent_window_and_selects_high_projected_value_token():
    config = _make_config(num_layers=1, compression=None)
    past = FakePast(num_layers=1)
    attention = FakeAttention(num_attention_heads=1, num_key_value_heads=1, head_dim=2)
    compressor = LaProx(budget=3, window_size=1, layer_idx=0, model_config=config)

    key = torch.tensor([[[[10.0, 0.0], [0.0, 1.0], [9.0, 2.0], [8.0, 3.0], [0.0, 4.0]]]])
    query = torch.zeros(1, 1, 5, 2)
    query[:, :, -1, 0] = 1.0
    value = torch.tensor([[[[1.0, 0.0], [0.0, 1_000_000.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]])

    _run_prefill([compressor], [attention], past, [key], [query], [value])

    kept_key_markers = past.layers[0].keys[0, 0, :, 1]
    assert 1.0 in kept_key_markers.tolist()
    assert 4.0 in kept_key_markers.tolist()
    assert torch.equal(past.layers[0].kv_lengths, torch.tensor([[3]]))
    assert torch.equal(past.layers[0].kv_valid_mask.sum(dim=-1), past.layers[0].kv_lengths)


def test_laprox_global_topk_allows_variable_head_lengths():
    torch.manual_seed(11)
    batch, num_layers, kv_heads, q_heads, seq_len, head_dim = 1, 3, 2, 2, 8, 2
    config = _make_config(num_layers=num_layers, compression=None)
    past = FakePast(num_layers=num_layers)
    attentions = [FakeAttention(q_heads, kv_heads, head_dim) for _ in range(num_layers)]
    compressors = [LaProx(budget=4, window_size=2, layer_idx=i, model_config=config) for i in range(num_layers)]

    keys = []
    queries = []
    values = []
    for layer_idx in range(num_layers):
        key = torch.randn(batch, kv_heads, seq_len, head_dim)
        query = torch.randn(batch, q_heads, seq_len, head_dim)
        value = torch.ones(batch, kv_heads, seq_len, head_dim)
        value[:, 0, : seq_len - 2, :] *= 100.0 + layer_idx
        value[:, 1, : seq_len - 2, :] *= 0.01
        keys.append(key)
        queries.append(query)
        values.append(value)

    _run_prefill(compressors, attentions, past, keys, queries, values)

    total_historical = 0
    all_lengths = []
    for layer in past.layers:
        assert hasattr(layer, "kv_valid_mask")
        assert torch.equal(layer.kv_valid_mask.sum(dim=-1), layer.kv_lengths)
        total_historical += int((layer.kv_lengths - 2).sum().item())
        all_lengths.extend(layer.kv_lengths.reshape(-1).tolist())

    assert total_historical == num_layers * kv_heads * (4 - 2)
    assert len(set(all_lengths)) > 1


def test_laprox_supports_gqa_scores_and_cache_packing():
    torch.manual_seed(5)
    batch, kv_heads, groups, seq_len, head_dim = 1, 2, 2, 6, 2
    q_heads = kv_heads * groups
    config = _make_config(num_layers=1, compression=None)
    past = FakePast(num_layers=1)
    attention = FakeAttention(q_heads, kv_heads, head_dim)
    compressor = LaProx(budget=4, window_size=2, layer_idx=0, model_config=config)

    key = torch.randn(batch, kv_heads, seq_len, head_dim)
    query = torch.randn(batch, q_heads, seq_len, head_dim)
    value = torch.randn(batch, kv_heads, seq_len, head_dim)

    _run_prefill([compressor], [attention], past, [key], [query], [value])

    assert past.layers[0].keys.shape[:2] == (batch, kv_heads)
    assert int((past.layers[0].kv_lengths - 2).sum().item()) == kv_heads * (4 - 2)
    assert torch.equal(past.layers[0].kv_valid_mask.sum(dim=-1), past.layers[0].kv_lengths)


def test_laprox_padding_mask_prevents_attention_to_padded_cache_values():
    torch.manual_seed(19)
    batch, num_layers, kv_heads, q_heads, seq_len, head_dim = 1, 2, 2, 2, 7, 2
    config = _make_config(num_layers=num_layers, compression=None)
    past = FakePast(num_layers=num_layers)
    attentions = [FakeAttention(q_heads, kv_heads, head_dim) for _ in range(num_layers)]
    compressors = [LaProx(budget=3, window_size=1, layer_idx=i, model_config=config) for i in range(num_layers)]

    keys = [torch.randn(batch, kv_heads, seq_len, head_dim) for _ in range(num_layers)]
    queries = [torch.randn(batch, q_heads, seq_len, head_dim) for _ in range(num_layers)]
    values = [torch.randn(batch, kv_heads, seq_len, head_dim) for _ in range(num_layers)]
    values[0][:, 0, : seq_len - 1, :] *= 100.0
    values[1][:, 0, : seq_len - 1, :] *= 100.0

    _run_prefill(compressors, attentions, past, keys, queries, values)

    layer = past.layers[0]
    assert (~layer.kv_valid_mask).any()
    poisoned_values = layer.values.clone()
    poisoned_values.masked_fill_(~layer.kv_valid_mask[..., None], 1_000_000.0)
    clean_values = poisoned_values.masked_fill(~layer.kv_valid_mask[..., None], 0.0)
    query = torch.randn(batch, q_heads, 1, head_dim)

    poisoned_output, _ = masked_eager_attention_forward(
        attentions[0],
        query,
        layer.keys,
        poisoned_values,
        None,
        layer.kv_valid_mask,
    )
    clean_output, _ = masked_eager_attention_forward(
        attentions[0],
        query,
        layer.keys,
        clean_values,
        None,
        layer.kv_valid_mask,
    )

    assert torch.allclose(poisoned_output, clean_output, atol=1e-5)
