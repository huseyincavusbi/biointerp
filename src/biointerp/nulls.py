"""Null-model utilities for permutation tests.

Pure functions used by the null suite: label permutation, row permutation,
within-group permutation, and empirical p-values. Keeping them separate makes
the statistical behaviour testable without loading a model or a feature matrix.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def permute_labels(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return a random permutation of ``labels``."""
    rng = np.random.default_rng(seed)
    return np.asarray(labels)[rng.permutation(len(labels))]


def permute_rows(matrix: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return ``matrix`` with rows permuted."""
    rng = np.random.default_rng(seed)
    return np.asarray(matrix)[rng.permutation(len(matrix))]


def permute_within_groups(
    values: np.ndarray,
    groups: Iterable[object],
    seed: int = 0,
) -> np.ndarray:
    """Permute ``values`` inside each group, preserving group membership."""
    values = np.asarray(values)
    groups = np.asarray(list(groups), dtype=object)
    if len(values) != len(groups):
        raise ValueError("values and groups must have the same length")

    rng = np.random.default_rng(seed)
    out = values.copy()
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        out[index] = values[index][rng.permutation(len(index))]
    return out


def empirical_p(
    observed: float,
    null: Iterable[float],
    alternative: str = "greater",
) -> float:
    """Empirical p-value with the +1 correction.

    ``alternative="greater"`` tests whether the observed value is larger than
    the null; ``"less"`` tests the opposite direction.
    """
    null = np.asarray(list(null), dtype=float)
    if len(null) == 0:
        raise ValueError("null distribution is empty")
    if alternative == "greater":
        count = int(np.sum(null >= observed))
    elif alternative == "less":
        count = int(np.sum(null <= observed))
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    return (count + 1) / (len(null) + 1)


def null_summary(
    observed: float,
    null: Iterable[float],
    alternative: str = "greater",
) -> dict[str, object]:
    """Summarize a null distribution against an observed statistic."""
    null = np.asarray(list(null), dtype=float)
    return {
        "observed": float(observed),
        "null_mean": round(float(null.mean()), 4),
        "null_sd": round(float(null.std(ddof=1)), 4) if len(null) > 1 else 0.0,
        "null_p95": round(float(np.percentile(null, 95)), 4),
        "null_min": round(float(null.min()), 4),
        "null_max": round(float(null.max()), 4),
        "n_permutations": int(len(null)),
        "p_value": round(empirical_p(observed, null, alternative), 6),
        "alternative": alternative,
    }


def pooled_spearman(pairs: Iterable[tuple[float, float]]) -> float:
    """Spearman correlation for pooled ``(x, y)`` pairs."""
    from scipy.stats import spearmanr

    values = np.asarray(list(pairs), dtype=float)
    if len(values) < 3:
        return float("nan")
    return float(spearmanr(values[:, 0], values[:, 1]).statistic)


def plddt_pairs(records: Iterable[Mapping[str, object]]) -> list[tuple[float, float]]:
    """Flatten ``structure_mapping`` records into ``(plddt, error)`` pairs."""
    pairs: list[tuple[float, float]] = []
    for record in records:
        for residue in record["per_residue"]:  # type: ignore[index]
            pairs.append((float(residue["plddt"]), 1.0 - float(residue["p_true"])))
    return pairs
