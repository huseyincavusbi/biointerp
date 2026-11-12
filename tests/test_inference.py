import numpy as np
import pytest

from biointerp.inference import (
    bh_fdr,
    bootstrap_auroc,
    bootstrap_tail_p,
    paired_bootstrap_delta,
    percentile_ci,
)


def test_bh_fdr_matches_known_example():
    q = bh_fdr([0.01, 0.02, 0.03, 0.04])
    np.testing.assert_allclose(q, [0.04, 0.04, 0.04, 0.04])


def test_bh_fdr_keeps_order():
    q = bh_fdr([0.5, 0.001])
    assert q[0] == pytest.approx(0.5)
    assert q[1] == pytest.approx(0.002)


def test_bh_fdr_is_monotone_in_p():
    p = np.array([0.001, 0.01, 0.2, 0.9])
    q = bh_fdr(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_bh_fdr_rejects_empty_and_nan():
    assert len(bh_fdr([])) == 0
    with pytest.raises(ValueError):
        bh_fdr([0.1, float("nan")])


def test_bootstrap_auroc_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = y + rng.normal(0, 0.5, 200)
    values = bootstrap_auroc(y, scores, n_boot=200, seed=0)
    low, high = percentile_ci(values)
    assert low <= high
    assert len(values) > 150


def test_bootstrap_auroc_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 80)
    scores = y + rng.normal(0, 0.5, 80)
    first = bootstrap_auroc(y, scores, n_boot=50, seed=3)
    second = bootstrap_auroc(y, scores, n_boot=50, seed=3)
    np.testing.assert_array_equal(first, second)


def test_paired_bootstrap_delta_zero_for_identical_scores():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 100)
    scores = y + rng.normal(0, 0.5, 100)
    delta = paired_bootstrap_delta(y, scores, scores, n_boot=50, seed=0)
    np.testing.assert_allclose(delta, 0.0)


def test_bootstrap_tail_p_is_small_for_clear_difference():
    delta = np.full(200, 0.1)
    assert bootstrap_tail_p(delta) == 0.0


def test_percentile_ci_returns_two_sided_interval():
    values = np.arange(101, dtype=float)
    assert percentile_ci(values) == [2.5, 97.5]
