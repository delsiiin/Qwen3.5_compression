import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from query_window_similarity import (
    SIMILARITY_METRIC_L2_DIFF,
    SIMILARITY_STATE_HIDDEN_L2_DIFF,
    SUPPORTED_SIMILARITY_STATES,
    build_l2_difference_from_layer_windows,
    get_adjacent_layer_scores,
    get_query_window_similarity_metric,
)


def test_hidden_state_l2_diff_builds_pairwise_layer_distance_matrix():
    layer_windows = [
        (2, torch.tensor([[0.0, 0.0], [1.0, 1.0]])),
        (4, torch.tensor([[1.0, 0.0], [1.0, 3.0]])),
        (7, torch.tensor([[2.0, 0.0], [1.0, 1.0]])),
    ]

    distances, layer_indices = build_l2_difference_from_layer_windows(layer_windows)
    vectors = np.stack([window.reshape(-1).numpy() for _layer_idx, window in layer_windows])
    expected = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=-1)

    assert layer_indices == [2, 4, 7]
    assert distances.dtype == np.float32
    np.testing.assert_allclose(distances, expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.diag(distances), np.zeros(3, dtype=np.float32))


def test_hidden_state_l2_diff_is_registered_as_query_window_submode():
    assert SIMILARITY_STATE_HIDDEN_L2_DIFF in SUPPORTED_SIMILARITY_STATES
    assert get_query_window_similarity_metric(SIMILARITY_STATE_HIDDEN_L2_DIFF) == SIMILARITY_METRIC_L2_DIFF


def test_adjacent_layer_scores_follow_layer_index_order():
    similarity = np.array(
        [
            [1.0, 0.91, 0.20],
            [0.91, 1.0, 0.83],
            [0.20, 0.83, 1.0],
        ],
        dtype=np.float32,
    )

    scores = get_adjacent_layer_scores(similarity, np.asarray([2, 4, 7], dtype=np.int16))

    assert scores == [
        {"from_layer": 2, "to_layer": 4, "score": float(np.float32(0.91))},
        {"from_layer": 4, "to_layer": 7, "score": float(np.float32(0.83))},
    ]
