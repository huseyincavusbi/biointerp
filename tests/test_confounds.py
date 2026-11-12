import numpy as np
import pytest

from biointerp.confounds import (
    aa_class,
    cluster_bootstrap,
    cluster_bootstrap_values,
    match_nearest,
    partial_spearman,
    percentile_ci,
    pooled_spearman,
    standardized_mean_diff,
)


def test_aa_class_groups_chemically_similar_residues():
    assert aa_class("D") == "negative"
    assert aa_class("K") == "positive"
    assert aa_class("S") == "polar"
    assert aa_class("L") == "hydrophobic"
    assert aa_class("P") == "special"


def test_aa_class_rejects_unknown_residue():
    with pytest.raises(ValueError):
        aa_class("X")


def test_standardized_mean_diff_zero_for_identical_groups():
    assert standardized_mean_diff([1, 2, 3], [1, 2, 3]) == 0.0


def test_standardized_mean_diff_positive_when_treated_is_larger():
    assert standardized_mean_diff([3, 4, 5], [1, 2, 3]) > 0


def test_partial_spearman_removes_common_confound():
    rng = np.random.default_rng(0)
    confound = rng.normal(size=500)
    x = confound + 0.3 * rng.normal(size=500)
    y = confound + 0.3 * rng.normal(size=500)

    raw, _ = partial_spearman(x, y, [])
    adjusted, _ = partial_spearman(x, y, [confound])

    assert raw > 0.7
    assert abs(adjusted) < 0.2


def test_partial_spearman_is_negative_when_relation_is_negative():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = -x + 0.1 * rng.normal(size=300)
    rho, _ = partial_spearman(x, y, [])
    assert rho < -0.8


def test_match_nearest_respects_strata_and_caliper():
    strata = ["a", "a", "a", "b", "b"]
    treated = [0, 3]
    controls = [1, 2, 4]
    distance = lambda left, right: abs(left - right) * 0.1  # noqa: E731

    pairs, unmatched = match_nearest(treated, controls, strata, distance, caliper=0.15, seed=0)

    assert not unmatched
    for left, right in pairs:
        assert strata[left] == strata[right]
        assert distance(left, right) <= 0.15


def test_match_nearest_reports_unmatched_when_pool_is_empty():
    pairs, unmatched = match_nearest(
        treated=[0], controls=[1], strata=["a", "b"], distance=lambda a, b: 0.0, caliper=1.0
    )
    assert pairs == []
    assert unmatched == [0]


def test_cluster_bootstrap_is_deterministic():
    groups = [0, 0, 1, 1, 2, 2]
    values = np.arange(6, dtype=float)
    stat = lambda index: float(values[index].mean())  # noqa: E731
    first = cluster_bootstrap(groups, stat, n_boot=50, seed=4)
    second = cluster_bootstrap(groups, stat, n_boot=50, seed=4)
    np.testing.assert_array_equal(first, second)


def test_cluster_bootstrap_values_brackets_mean():
    values = [1.0, 2.0, 3.0, 4.0]
    groups = [0, 0, 1, 1]
    values_dist = cluster_bootstrap_values(values, groups, n_boot=200, seed=0)
    low, high = percentile_ci(values_dist)
    assert low <= 2.5 <= high


def test_pooled_spearman_perfect_relation():
    assert pooled_spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
