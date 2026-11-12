"""Calibration and triage metrics for 0-1 error scores.

The error score is a predicted probability of model error. Calibration asks
whether a score of 0.7 really means 70% errors; triage asks whether the top
decile is enriched for errors compared with random ranking.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def reliability_curve(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Bin predictions and return mean score, observed rate, and count per bin."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p):
        raise ValueError("y_true and probabilities must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    curve: list[dict[str, float]] = []
    for bucket in range(n_bins):
        mask = index == bucket
        if not mask.any():
            continue
        curve.append(
            {
                "bin": bucket,
                "count": int(mask.sum()),
                "mean_score": round(float(p[mask].mean()), 4),
                "observed_rate": round(float(y[mask].mean()), 4),
            }
        )
    return curve


def expected_calibration_error(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Weighted mean absolute gap between score and observed error rate."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        return float("nan")
    curve = reliability_curve(y, p, n_bins=n_bins)
    total = sum(item["count"] for item in curve)
    return float(
        sum(
            item["count"] / total * abs(item["mean_score"] - item["observed_rate"])
            for item in curve
        )
    )


def brier_score(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mean squared error of probabilistic predictions."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def precision_at_fraction(
    y_true: Sequence[int],
    scores: Sequence[float],
    fraction: float = 0.1,
) -> float:
    """Fraction of errors among the top-scoring ``fraction`` of items."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if len(y) == 0:
        return float("nan")
    k = max(1, int(round(len(y) * fraction)))
    order = np.argsort(scores)[::-1][:k]
    return float(y[order].mean())


def enrichment_at_fraction(
    y_true: Sequence[int],
    scores: Sequence[float],
    fraction: float = 0.1,
) -> float:
    """Precision at the top fraction divided by the base error rate."""
    base = float(np.mean(np.asarray(y_true, dtype=float)))
    if base == 0.0:
        return float("nan")
    return float(precision_at_fraction(y_true, scores, fraction) / base)


def calibration_summary(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    fraction: float = 0.1,
    n_bins: int = 10,
) -> dict[str, float]:
    """ECE, Brier, precision@k, enrichment, and base rate."""
    base = float(np.mean(np.asarray(y_true, dtype=float)))
    return {
        "ece": round(expected_calibration_error(y_true, probabilities, n_bins=n_bins), 4),
        "brier": round(brier_score(y_true, probabilities), 4),
        "precision_at_fraction": round(precision_at_fraction(y_true, probabilities, fraction), 4),
        "enrichment": round(enrichment_at_fraction(y_true, probabilities, fraction), 4),
        "base_rate": round(base, 4),
        "fraction": fraction,
    }


def percentile_ci(values: Iterable[float]) -> list[float]:
    """Two-sided 95% percentile interval."""
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    return [
        round(float(np.percentile(values, 2.5)), 4),
        round(float(np.percentile(values, 97.5)), 4),
    ]
