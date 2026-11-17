import numpy as np
import pytest

from biointerp.ablation import (
    equalize_norms,
    global_norm,
    group_by_layer,
    random_sets,
    select_top_bottom,
    summarize_ablation,
)


def test_select_top_bottom_finds_signal():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    cett = rng.normal(size=(n, 6))
    cett[:, 4] += y * 3.0
    train = np.ones(n, dtype=bool)

    top, bottom = select_top_bottom(cett, y, train, k=2)

    assert 4 in top
    assert 4 not in bottom


def test_random_sets_are_deterministic_and_disjoint():
    first = random_sets(100, 5, 3, seed=0)
    second = random_sets(100, 5, 3, seed=0)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert len(set(left)) == 5


def test_group_by_layer_maps_global_indices():
    assert group_by_layer([0, 2, 5, 7], intermediate=3) == {0: [0, 2], 1: [2], 2: [1]}


def test_equalize_norms_uses_shared_target():
    vectors = {0: np.array([3.0, 4.0]), 1: np.array([0.0, 0.0])}
    scaled = equalize_norms(vectors, target_norm=10.0)
    assert global_norm(scaled) == 1.0 * 10.0
    assert np.allclose(scaled[0], [6.0, 8.0])


def test_summarize_ablation_uses_degradation_direction():
    rows = [
        {"condition": "baseline", "accuracy": 0.45, "p_true": 0.40},
        {"condition": "top", "accuracy": 0.44, "p_true": 0.36},
        {"condition": "bottom", "accuracy": 0.48, "p_true": 0.42},
        {"condition": "random-0", "accuracy": 0.45, "p_true": 0.39},
        {"condition": "random-1", "accuracy": 0.46, "p_true": 0.40},
        {"condition": "random-2", "accuracy": 0.45, "p_true": 0.395},
    ]
    summary = summarize_ablation(rows, equal_norm_target=4.0)
    assert summary["accuracy_top_drop"] == pytest.approx(0.01)
    assert summary["accuracy_top_exceeds_random_p95"] is True
    assert summary["p_true_top_exceeds_random_p95"] is True
    assert summary["metrics_supported"] == 2
