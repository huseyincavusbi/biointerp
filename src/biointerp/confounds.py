"""Confound-matching utilities for position-level error analysis.

Positions are matched on amino-acid class, protein length, and relative
position so a pLDDT/error contrast is not explained by composition or
position alone. Partial Spearman uses rank residuals to control the same
covariates.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

AA_CLASS = {
    "A": "hydrophobic",
    "V": "hydrophobic",
    "I": "hydrophobic",
    "L": "hydrophobic",
    "M": "hydrophobic",
    "F": "hydrophobic",
    "W": "hydrophobic",
    "Y": "hydrophobic",
    "C": "hydrophobic",
    "G": "special",
    "P": "special",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "D": "negative",
    "E": "negative",
    "K": "positive",
    "R": "positive",
    "H": "positive",
}


def aa_class(residue: str) -> str:
    """Return the coarse amino-acid class for a residue."""
    try:
        return AA_CLASS[residue.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown amino acid: {residue!r}") from exc


def standardized_mean_diff(treated: Sequence[float], control: Sequence[float]) -> float:
    """Standardized mean difference (treated minus control)."""
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return float((treated.mean() - control.mean()) / pooled)


def _rank_residuals(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    ranks = rankdata(values)
    coefficients, *_ = np.linalg.lstsq(design, ranks, rcond=None)
    return ranks - design @ coefficients


def partial_spearman(
    x: Sequence[float],
    y: Sequence[float],
    covariates: Iterable[Sequence[float]],
) -> tuple[float, float]:
    """Spearman correlation of ``x`` and ``y`` after rank-residualizing covariates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    columns = [np.asarray(c, dtype=float) for c in covariates]
    if any(len(c) != len(x) for c in columns):
        raise ValueError("covariates must have the same length as x")

    design = np.column_stack([rankdata(c) for c in columns] + [np.ones(len(x))])
    residual_x = _rank_residuals(x, design)
    residual_y = _rank_residuals(y, design)
    if residual_x.std() < 1e-12 or residual_y.std() < 1e-12:
        return float("nan"), float("nan")
    rho, p_value = pearsonr(residual_x, residual_y)
    return float(rho), float(p_value)


def match_nearest(
    treated: Sequence[int],
    controls: Sequence[int],
    strata: Sequence[object],
    distance: Callable[[int, int], float],
    caliper: float,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Greedy nearest-neighbour matching within exact strata.

    Every treated unit is matched to at most one unused control from the same
    stratum, provided the distance is within ``caliper``. Hardest strata are
    matched first so rare classes are not starved.
    """
    rng = np.random.default_rng(seed)
    pools: dict[object, list[int]] = defaultdict(list)
    for index in controls:
        pools[strata[index]].append(index)
    for pool in pools.values():
        rng.shuffle(pool)

    order = sorted(treated, key=lambda index: len(pools.get(strata[index], [])))
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    unmatched: list[int] = []

    for index in order:
        best: int | None = None
        best_distance = float("inf")
        for candidate in pools.get(strata[index], []):
            if candidate in used:
                continue
            current = distance(index, candidate)
            if current < best_distance:
                best, best_distance = candidate, current
        if best is None or best_distance > caliper:
            unmatched.append(index)
        else:
            used.add(best)
            pairs.append((index, best))

    return pairs, unmatched


def cluster_bootstrap(
    groups: Sequence[object],
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap whole groups, returning the statistic distribution."""
    groups = np.asarray(list(groups), dtype=object)
    unique = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([index_by_group[group] for group in chosen])
        values[i] = statistic(index)
    return values


def cluster_bootstrap_values(
    values: Sequence[float],
    groups: Sequence[object],
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap a mean by resampling groups."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(list(groups), dtype=object)
    unique = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([index_by_group[group] for group in chosen])
        out[i] = float(values[index].mean())
    return out


def percentile_ci(values: Sequence[float]) -> list[float]:
    """Two-sided 95% percentile interval."""
    return [
        round(float(np.percentile(values, 2.5)), 4),
        round(float(np.percentile(values, 97.5)), 4),
    ]


def pooled_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation for two aligned vectors."""
    return float(spearmanr(np.asarray(x), np.asarray(y)).statistic)
