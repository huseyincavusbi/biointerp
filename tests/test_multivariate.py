import numpy as np

from biointerp.multivariate import (
    fit_cluster_logit,
    likelihood_ratio_test,
    variance_inflation_factors,
)


def _simulate(n=4000, groups=200, effect=1.5, seed=0):
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(groups), n // groups)
    x = rng.normal(size=(len(group_ids), 1))
    intercept = rng.normal(scale=0.5, size=groups)[group_ids]
    logits = intercept + effect * x[:, 0]
    y = rng.random(len(group_ids)) < 1 / (1 + np.exp(-logits))
    return x, y.astype(int), group_ids


def test_cluster_logit_recovers_positive_effect():
    x, y, groups = _simulate()
    result = fit_cluster_logit(x, y, groups, ["x"])
    feature = result["features"]["x"]
    assert feature["odds_ratio"] > 1.5
    assert feature["p_value"] < 0.01


def test_cluster_se_exceeds_naive_under_within_group_correlation():
    x, y, groups = _simulate(effect=0.0, seed=2)
    result = fit_cluster_logit(x, y, groups, ["x"])
    feature = result["features"]["x"]
    assert feature["se"] > feature["naive_se"]


def test_vif_detects_collinearity():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=500)
    x2 = x1 + 0.05 * rng.normal(size=500)
    x3 = rng.normal(size=500)
    vif = variance_inflation_factors(np.column_stack([x1, x2, x3]), ["x1", "x2", "x3"])
    assert vif["x1"] > 5
    assert vif["x2"] > 5
    assert vif["x3"] < 2


def test_vif_single_predictor_is_one():
    assert variance_inflation_factors(np.ones((10, 1)), ["only"]) == {"only": 1.0}


def test_likelihood_ratio_test_is_significant_for_real_term():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 2))
    logits = 1.5 * x[:, 0]
    y = (rng.random(2000) < 1 / (1 + np.exp(-logits))).astype(int)
    groups = np.arange(2000)
    full = fit_cluster_logit(x, y, groups, ["signal", "noise"])
    reduced = fit_cluster_logit(x[:, [1]], y, groups, ["noise"])
    assert likelihood_ratio_test(full, reduced)["p_value"] < 0.01


def test_fit_rejects_shape_mismatch():
    try:
        fit_cluster_logit(np.ones((5, 2)), [0, 1, 0, 1, 0], [0] * 5, ["one"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
