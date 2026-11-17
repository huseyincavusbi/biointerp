"""Cluster-robust logistic regression and model diagnostics.

Positions within a protein are not independent, so standard errors are
clustered by protein. Coefficients are reported per standard deviation, which
makes odds ratios comparable across predictors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import chi2, norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


def _log_likelihood(y: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)))


def fit_cluster_logit(
    x: np.ndarray,
    y: Sequence[int],
    groups: Sequence[object],
    feature_names: Sequence[str],
    max_iter: int = 5000,
) -> dict[str, object]:
    """Fit logistic regression with cluster-robust sandwich covariance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(list(groups), dtype=object)
    if x.ndim != 2 or x.shape[1] != len(feature_names):
        raise ValueError("x must be (n, len(feature_names))")
    if len(y) != len(x) or len(groups) != len(x):
        raise ValueError("x, y, and groups must have the same length")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])

    model = LogisticRegression(
        C=float("inf"),
        solver="lbfgs",
        fit_intercept=False,
        max_iter=max_iter,
    )
    model.fit(design, y)
    coefficients = model.coef_[0]
    probabilities = model.predict_proba(design)[:, 1]
    weights = probabilities * (1.0 - probabilities)

    bread = np.linalg.pinv(design.T @ (weights[:, None] * design))
    residuals = y - probabilities
    meat = np.zeros_like(bread)
    unique_groups = np.unique(groups)
    for group in unique_groups:
        index = groups == group
        score = design[index].T @ residuals[index]
        meat += np.outer(score, score)

    n, k = design.shape
    g = len(unique_groups)
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 else 1.0
    covariance = bread @ meat @ bread * correction

    standard_errors = np.sqrt(np.diag(covariance))
    naive_errors = np.sqrt(np.diag(bread))
    z_values = coefficients / np.where(standard_errors > 0, standard_errors, np.nan)
    p_values = 2.0 * norm.sf(np.abs(z_values))

    features: dict[str, dict[str, float]] = {}
    for index, name in enumerate(feature_names, start=1):
        features[name] = {
            "coefficient": round(float(coefficients[index]), 4),
            "se": round(float(standard_errors[index]), 4),
            "naive_se": round(float(naive_errors[index]), 4),
            "z": round(float(z_values[index]), 4),
            "p_value": round(float(p_values[index]), 6),
            "odds_ratio": round(float(np.exp(coefficients[index])), 4),
            "ci95": [
                round(float(np.exp(coefficients[index] - 1.96 * standard_errors[index])), 4),
                round(float(np.exp(coefficients[index] + 1.96 * standard_errors[index])), 4),
            ],
        }

    return {
        "n": int(n),
        "n_groups": int(g),
        "n_parameters": int(k),
        "log_likelihood": round(_log_likelihood(y, probabilities), 4),
        "features": features,
        "intercept": round(float(coefficients[0]), 4),
    }


def variance_inflation_factors(x: np.ndarray, feature_names: Sequence[str]) -> dict[str, float]:
    """VIF from auxiliary linear regressions."""
    x = np.asarray(x, dtype=float)
    values: dict[str, float] = {}
    if x.shape[1] <= 1:
        return {name: 1.0 for name in feature_names}
    for index, name in enumerate(feature_names):
        others = np.delete(x, index, axis=1)
        model = LinearRegression().fit(others, x[:, index])
        r_squared = float(model.score(others, x[:, index]))
        values[name] = round(1.0 / max(1e-12, 1.0 - r_squared), 4)
    return values


def likelihood_ratio_test(
    full: dict[str, object], reduced: dict[str, object], df: int = 1
) -> dict[str, float]:
    """Likelihood-ratio test between nested fits."""
    statistic = 2.0 * (float(full["log_likelihood"]) - float(reduced["log_likelihood"]))
    statistic = max(0.0, statistic)
    return {
        "statistic": round(statistic, 4),
        "df": int(df),
        "p_value": round(float(chi2.sf(statistic, df)), 6),
    }
