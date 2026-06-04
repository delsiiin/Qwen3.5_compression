import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from attn_output_ratio import (
    build_layer_stats,
    compute_attn_output_hidden_state_ratios,
    parse_layer_spec,
)


class ToyAttention(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, hidden_states, **_kwargs):
        return hidden_states * self.scale, None


class ToyLayer(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.self_attn = ToyAttention(scale)

    def forward(self, hidden_states):
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        return hidden_states


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyLayer(2.0), ToyLayer(3.0)])


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()

    def forward(self, input_ids, **_kwargs):
        hidden_states = torch.stack((input_ids.float(), input_ids.float() + 1.0), dim=-1)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states}


def test_compute_attn_output_hidden_state_ratios_captures_per_layer_token_l2_ratio():
    model = ToyModel()
    inputs = {"input_ids": torch.tensor([[1, 2, 4]])}

    ratios, input_norms, output_norms, layer_indices, mixer_names = compute_attn_output_hidden_state_ratios(
        model,
        inputs,
    )

    assert layer_indices == [0, 1]
    assert mixer_names == ["self_attn", "self_attn"]
    assert ratios.shape == (2, 3)
    np.testing.assert_allclose(ratios[0], np.full(3, 2.0, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ratios[1], np.full(3, 3.0, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output_norms / input_norms, ratios, rtol=1e-6, atol=1e-6)


def test_parse_layer_spec_supports_all_auto_lists_and_ranges():
    available = [0, 1, 2, 3, 4, 5]

    assert parse_layer_spec("all", available) == available
    assert parse_layer_spec("auto", available) == [0, 2, 3, 5]
    assert parse_layer_spec("1,3-5", available) == [1, 3, 4, 5]


def test_build_layer_stats_records_distribution_summary():
    ratios = np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]], dtype=np.float32)

    stats = build_layer_stats(ratios, [5, 10])

    assert stats[0]["layer"] == 5
    assert stats[0]["token_count"] == 3
    assert stats[0]["sum"] == 6.0
    assert stats[0]["median"] == 2.0
    assert stats[1]["layer"] == 10
    assert stats[1]["sum"] == 6.0
    assert stats[1]["std"] == 0.0
