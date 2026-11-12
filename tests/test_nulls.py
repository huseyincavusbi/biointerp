import numpy as np
import pytest

from biointerp.nulls import (
    empirical_p,
    null_summary,
    permute_labels,
    permute_rows,
    permute_within_groups,
    plddt_pairs,
    pooled_spearman,
)


def test_permute_labels_preserves_counts():
    labels = np.array([0, 0, 1, 1, 1])
    shuffled = permute_labels(labels, seed=0)
    np.testing.assert_array_equal(np.sort(shuffled), np.sort(labels))


def test_permute_labels_is_deterministic():
    labels = np.arange(10)
    np.testing.assert_array_equal(permute_labels(labels, seed=3), permute_labels(labels, seed=3))


def test_permute_rows_keeps_rows_intact():
    matrix = np.arange(12).reshape(4, 3)
    shuffled = permute_rows(matrix, seed=1)
    assert {tuple(row) for row in shuffled} == {tuple(row) for row in matrix}


def test_permute_within_groups_keeps_membership():
    values = np.array([1, 2, 3, 4, 5, 6])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    shuffled = permute_within_groups(values, groups, seed=2)
    np.testing.assert_array_equal(np.sort(shuffled[groups == "a"]), [1, 2, 3])
    np.testing.assert_array_equal(np.sort(shuffled[groups == "b"]), [4, 5, 6])
    assert not np.array_equal(shuffled, values)


def test_permute_within_groups_rejects_length_mismatch():
    with pytest.raises(ValueError):
        permute_within_groups([1, 2], ["a"])


def test_empirical_p_uses_plus_one_correction():
    assert empirical_p(10, [0, 1, 2], alternative="greater") == pytest.approx(0.25)
    assert empirical_p(-10, [0, 1, 2], alternative="less") == pytest.approx(0.25)


def test_empirical_p_rejects_unknown_alternative():
    with pytest.raises(ValueError):
        empirical_p(1, [0], alternative="two-sided")


def test_null_summary_reports_tail():
    summary = null_summary(0.8, [0.5, 0.6, 0.7], alternative="greater")
    assert summary["observed"] == 0.8
    assert summary["null_p95"] == pytest.approx(0.69, abs=0.01)
    assert summary["p_value"] == pytest.approx(0.25)


def test_plddt_pairs_flattens_records():
    records = [{"per_residue": [{"plddt": 0.9, "p_true": 0.8}, {"plddt": 0.5, "p_true": 0.5}]}]
    pairs = plddt_pairs(records)
    assert pairs[0][0] == pytest.approx(0.9)
    assert pairs[0][1] == pytest.approx(0.2)
    assert pairs[1][0] == pytest.approx(0.5)
    assert pairs[1][1] == pytest.approx(0.5)


def test_pooled_spearman_detects_monotone_relation():
    pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert pooled_spearman(pairs) == pytest.approx(1.0)
