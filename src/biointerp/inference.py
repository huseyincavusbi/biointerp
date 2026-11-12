"""Statistical inference helpers: BH-FDR and cluster bootstrap.

The bootstrap resamples whole proteins, not individual positions, because
residues within a protein are not independent. BH-FDR is applied to the
per-assay and per-layer families.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values in the original order."""
    p = np.asarray(list(p_values), dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if len(p) == 0:
        return np.array([], dtype=float)
    if np.any(np.isnan(p)):
        raise ValueError("p_values must not contain NaN")

    order = np.argsort(p)
    ranks = p[order] * len(p) / np.arange(1, len(p) + 1)
    monotone = np.minimum.accumulate(ranks[::-1])[::-1]
    q = np.empty(len(p), dtype=float)
    q[order] = np.clip(monotone, 0.0, 1.0)
    return q


def _resample_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    chosen = rng.choice(unique, size=len(unique), replace=True)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    return np.concatenate([index_by_group[group] for group in chosen])


def _resample_rows(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, n)


def bootstrap_auroc(
    y: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[object] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap AUROC, resampling groups when provided."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores):
        raise ValueError("y and scores must have the same length")

    rng = np.random.default_rng(seed)
    groups_array = None if groups is None else np.asarray(list(groups), dtype=object)
    values: list[float] = []
    for _ in range(n_boot):
        index = (
            _resample_rows(len(y), rng)
            if groups_array is None
            else _resample_indices(groups_array, rng)
        )
        if len(np.unique(y[index])) < 2:
            continue
        values.append(float(roc_auc_score(y[index], scores[index])))
    return np.asarray(values)


def paired_bootstrap_delta(
    y: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    groups: Sequence[object] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap the paired AUROC difference ``a - b``."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if not (len(y) == len(scores_a) == len(scores_b)):
        raise ValueError("y and both score vectors must have the same length")

    rng = np.random.default_rng(seed)
    groups_array = None if groups is None else np.asarray(list(groups), dtype=object)
    values: list[float] = []
    for _ in range(n_boot):
        index = (
            _resample_rows(len(y), rng)
            if groups_array is None
            else _resample_indices(groups_array, rng)
        )
        if len(np.unique(y[index])) < 2:
            continue
        values.append(
            float(
                roc_auc_score(y[index], scores_a[index]) - roc_auc_score(y[index], scores_b[index])
            )
        )
    return np.asarray(values)


def bootstrap_tail_p(delta: Sequence[float]) -> float:
    """Two-sided bootstrap tail probability for a paired difference."""
    delta = np.asarray(list(delta), dtype=float)
    if len(delta) == 0:
        return float("nan")
    lower = float(np.mean(delta <= 0.0))
    upper = float(np.mean(delta >= 0.0))
    return float(min(1.0, 2.0 * min(lower, upper)))


def percentile_ci(values: Sequence[float], level: float = 95.0) -> list[float]:
    """Percentile interval for a bootstrap distribution."""
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    tail = (100.0 - level) / 2.0
    return [
        round(float(np.percentile(values, tail)), 4),
        round(float(np.percentile(values, 100.0 - tail)), 4),
    ]
