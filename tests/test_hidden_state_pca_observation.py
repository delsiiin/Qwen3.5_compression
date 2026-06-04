import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hidden_state_pca_observation import (
    HiddenStatePCARunWriter,
    compute_adjacent_layer_angles,
    collect_layer_key_value_states,
    collect_layer_hidden_states,
    compute_hidden_state_pca,
    compute_key_value_state_pca,
    compute_pca_layer_centers,
    compute_shared_pca,
    flatten_key_value_token_states,
    normalize_token_span,
)
from attn_output_ratio import parse_layer_spec


class ToyLayer(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)

    def forward(self, hidden_states):
        return hidden_states + self.offset


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyLayer(1.0), ToyLayer(2.0), ToyLayer(4.0)])


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()

    def forward(self, input_ids, **_kwargs):
        hidden_states = torch.stack(
            (
                input_ids.float(),
                input_ids.float() * 2.0,
                input_ids.float() * -1.0,
            ),
            dim=-1,
        )
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states}


class ToyTokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [str(token_id) for token_id in token_ids]

    def batch_decode(self, batch_token_ids, **_kwargs):
        return [str(token_ids[0]) for token_ids in batch_token_ids]


class ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.num_heads = 2
        self.num_key_value_heads = 2
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.v_proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            eye = torch.eye(4)
            self.q_proj.weight.copy_(eye)
            self.k_proj.weight.copy_(eye)
            self.v_proj.weight.copy_(eye)

    def forward(self, hidden_states, **_kwargs):
        return hidden_states, None


class ToyKVLayer(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)
        self.self_attn = ToyAttention()

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.offset
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        return hidden_states


class ToyKVBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyKVLayer(1.0), ToyKVLayer(3.0)])


class ToyKVModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyKVBackbone()

    def forward(self, input_ids, **_kwargs):
        hidden_states = torch.stack(
            (
                input_ids.float(),
                input_ids.float() * 2.0,
                input_ids.float() * -1.0,
                input_ids.float() + 1.0,
            ),
            dim=-1,
        )
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states}


def test_hidden_state_pca_layer_spec_supports_all_auto_lists_and_ranges():
    available = [0, 1, 2, 3, 4, 5]

    assert parse_layer_spec("all", available) == available
    assert parse_layer_spec("auto", available) == [0, 2, 3, 5]
    assert parse_layer_spec("1,3-5", available) == [1, 3, 4, 5]


def test_normalize_token_span_supports_positive_and_negative_ranges():
    assert normalize_token_span(10, 2, 5) == (2, 5)
    assert normalize_token_span(10, -4, None) == (6, 10)
    assert normalize_token_span(10, 0, -1) == (0, 9)


def test_collect_layer_key_value_states_captures_same_token_span_for_selected_layers():
    model = ToyKVModel()
    inputs = {"input_ids": torch.tensor([[1, 3, 5, 7]])}

    captured, layer_indices, token_start, token_end = collect_layer_key_value_states(
        model,
        inputs,
        layer_spec="all",
        token_start=1,
        token_end=3,
    )
    layer0 = flatten_key_value_token_states(
        captured[0]["key_states"],
        captured[0]["value_states"],
        token_start,
        token_end,
    )
    layer1 = flatten_key_value_token_states(
        captured[1]["key_states"],
        captured[1]["value_states"],
        token_start,
        token_end,
    )

    assert layer_indices == [0, 1]
    assert token_start == 1
    assert token_end == 3
    assert layer0.shape == (2, 8)
    expected_layer0 = torch.tensor(
        [
            [4.0, 7.0, -2.0, 5.0, 4.0, 7.0, -2.0, 5.0],
            [6.0, 11.0, -4.0, 7.0, 6.0, 11.0, -4.0, 7.0],
        ]
    )
    expected_layer1 = expected_layer0 + 3.0
    torch.testing.assert_close(layer0, expected_layer0)
    torch.testing.assert_close(layer1, expected_layer1)


def test_compute_key_value_state_pca_returns_all_key_value_token_points():
    model = ToyKVModel()
    inputs = {"input_ids": torch.tensor([[1, 3, 5, 7]])}

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
        token_start,
        token_end,
    ) = compute_key_value_state_pca(
        model,
        inputs,
        layer_spec="all",
        token_start=0,
        token_end=None,
    )

    assert layer_indices == [0, 1]
    assert token_start == 0
    assert token_end == 4
    assert key_pca_points.shape == (2, 4, 2)
    assert value_pca_points.shape == (2, 4, 2)
    assert key_components.shape == (2, 4)
    assert value_components.shape == (2, 4)
    assert key_mean.shape == (4,)
    assert value_mean.shape == (4,)
    assert key_explained.shape == (2,)
    assert value_explained.shape == (2,)


def test_collect_layer_hidden_states_captures_same_token_span_for_selected_layers():
    model = ToyModel()
    inputs = {"input_ids": torch.tensor([[1, 3, 5, 7]])}

    hidden_states, layer_indices, token_start, token_end = collect_layer_hidden_states(
        model,
        inputs,
        layer_spec="0,2",
        token_start=1,
        token_end=3,
    )

    assert layer_indices == [0, 2]
    assert token_start == 1
    assert token_end == 3
    assert hidden_states.shape == (2, 2, 3)
    expected_first_layer = np.asarray(
        [[4.0, 7.0, -2.0], [6.0, 11.0, -4.0]],
        dtype=np.float32,
    )
    expected_third_layer = expected_first_layer + 6.0
    np.testing.assert_allclose(hidden_states[0], expected_first_layer)
    np.testing.assert_allclose(hidden_states[1], expected_third_layer)


def test_compute_shared_pca_uses_one_coordinate_system_for_all_layers():
    hidden_states = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    pca_points, components, mean, explained = compute_shared_pca(hidden_states)

    assert pca_points.shape == (2, 2, 2)
    assert components.shape == (2, 3)
    assert mean.shape == (3,)
    assert explained.shape == (2,)
    np.testing.assert_allclose(
        pca_points.reshape(-1, 2),
        (hidden_states.reshape(-1, 3) - mean) @ components.T,
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_adjacent_layer_angles_uses_origin_to_layer_center_vectors():
    pca_points = np.asarray(
        [
            [[1.0, 0.0], [3.0, 0.0]],
            [[0.0, 2.0], [0.0, 4.0]],
            [[-2.0, 0.0], [-4.0, 0.0]],
        ],
        dtype=np.float32,
    )

    centers = compute_pca_layer_centers(pca_points)
    angles = compute_adjacent_layer_angles(centers)

    np.testing.assert_allclose(
        centers,
        np.asarray([[2.0, 0.0], [0.0, 3.0], [-3.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(angles, np.asarray([90.0, 90.0], dtype=np.float32), atol=1e-5)


def test_compute_hidden_state_pca_returns_shared_pca_points_for_toy_model():
    model = ToyModel()
    inputs = {"input_ids": torch.tensor([[1, 3, 5]])}

    pca_points, layer_indices, components, mean, explained, token_start, token_end = compute_hidden_state_pca(
        model,
        inputs,
        layer_spec="auto",
        token_start=-2,
    )

    assert layer_indices == [0, 1, 2]
    assert token_start == 1
    assert token_end == 3
    assert pca_points.shape == (3, 2, 2)
    assert components.shape == (2, 3)
    assert mean.shape == (3,)
    assert explained.shape == (2,)


def test_hidden_state_pca_writer_skips_over_prefill_cap(tmp_path):
    out_file = tmp_path / "result.jsonl"
    out_file.write_text("", encoding="utf-8")
    writer = HiddenStatePCARunWriter(
        root_dir=str(tmp_path / "pca"),
        model_name="toy",
        out_file=str(out_file),
        max_prefill_tokens=2,
        layer_spec="all",
        token_start=1,
        token_end=3,
    )
    sample_writer = writer.new_sample({"_id": "abc", "question": "q"})

    record = sample_writer.capture_prefill(
        model=ToyModel(),
        tokenizer=ToyTokenizer(),
        prompt_text="prompt",
        inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        label="response",
    )

    assert record["status"] == "skipped_over_cap"
    assert record["pca_file"] is None
    assert "exceeds cap 2" in record["reason"]


def test_hidden_state_pca_writer_outputs_separate_key_and_value_plots(tmp_path):
    out_file = tmp_path / "result.jsonl"
    out_file.write_text("", encoding="utf-8")
    writer = HiddenStatePCARunWriter(
        root_dir=str(tmp_path / "pca"),
        model_name="toy",
        out_file=str(out_file),
        max_prefill_tokens=None,
        layer_spec="all",
        token_start=0,
        token_end=None,
    )
    sample_writer = writer.new_sample({"_id": "abc", "question": "q"})

    record = sample_writer.capture_prefill(
        model=ToyKVModel(),
        tokenizer=ToyTokenizer(),
        prompt_text="prompt",
        inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        label="response",
    )

    assert record["status"] == "saved"
    assert record["key_pca_shape"] == [2, 3, 2]
    assert record["value_pca_shape"] == [2, 3, 2]
    assert record["key_plot_file"].endswith("_key_state_pca.png")
    assert record["value_plot_file"].endswith("_value_state_pca.png")
    assert record["key_angle_plot_file"].endswith("_key_state_layer_angle.png")
    assert record["value_angle_plot_file"].endswith("_value_state_layer_angle.png")
    assert record["adjacent_layer_pairs"] == [[0, 1]]
    assert len(record["key_adjacent_layer_angles_deg"]) == 1
    assert len(record["value_adjacent_layer_angles_deg"]) == 1
    sample_dir = Path(sample_writer.sample_dir)
    assert (sample_dir / record["key_plot_file"]).exists()
    assert (sample_dir / record["value_plot_file"]).exists()
    assert (sample_dir / record["key_angle_plot_file"]).exists()
    assert (sample_dir / record["value_angle_plot_file"]).exists()

    with np.load(sample_dir / record["pca_file"]) as data:
        assert data["key_pca_points"].shape == (2, 3, 2)
        assert data["value_pca_points"].shape == (2, 3, 2)
        assert data["key_pca_components"].shape == (2, 4)
        assert data["value_pca_components"].shape == (2, 4)
        assert data["key_layer_centers"].shape == (2, 2)
        assert data["value_layer_centers"].shape == (2, 2)
        assert data["adjacent_layer_pairs"].shape == (1, 2)
        assert data["key_adjacent_layer_angles_deg"].shape == (1,)
        assert data["value_adjacent_layer_angles_deg"].shape == (1,)
