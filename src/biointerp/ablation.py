"""Neuron selection and intervention helpers for ablation controls."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from biointerp.core import univariate_auroc


def select_top_bottom(
    cett: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k and bottom-k error neurons by training AUROC."""
    scores = univariate_auroc(cett[train], y[train])
    order = np.argsort(scores)
    return order[::-1][:k], order[:k]


def random_sets(n_features: int, k: int, draws: int, seed: int = 0) -> list[np.ndarray]:
    """Draw random neuron sets without replacement."""
    rng = np.random.default_rng(seed)
    return [rng.choice(n_features, size=k, replace=False) for _ in range(draws)]


def group_by_layer(indices: Iterable[int], intermediate: int) -> dict[int, list[int]]:
    """Map global neuron indices to per-layer local indices."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        grouped[int(index) // intermediate].append(int(index) % intermediate)
    return dict(grouped)


def equalize_norms(
    mean_by_layer: Mapping[int, np.ndarray],
    target_norm: float,
) -> dict[int, np.ndarray]:
    """Scale per-layer mean vectors to a shared concatenated norm."""
    total = np.sqrt(
        sum(float(np.sum(np.asarray(vector) ** 2)) for vector in mean_by_layer.values())
    )
    if total < 1e-12:
        return {layer: np.asarray(vector, dtype=float) for layer, vector in mean_by_layer.items()}
    scale = target_norm / total
    return {
        layer: np.asarray(vector, dtype=float) * scale for layer, vector in mean_by_layer.items()
    }


def condition_vectors(
    mean_z: Mapping[int, np.ndarray],
    indices: Sequence[int],
    intermediate: int,
    target_norm: float,
) -> dict[int, np.ndarray]:
    """Build equal-norm per-layer replacement vectors for a neuron set."""
    grouped = group_by_layer(indices, intermediate)
    raw = {layer: np.asarray(mean_z[layer])[np.asarray(local)] for layer, local in grouped.items()}
    return equalize_norms(raw, target_norm)


def global_norm(mean_by_layer: Mapping[int, np.ndarray]) -> float:
    return float(
        np.sqrt(sum(float(np.sum(np.asarray(vector) ** 2)) for vector in mean_by_layer.values()))
    )


def summarize_ablation(
    rows: Sequence[Mapping[str, object]],
    equal_norm_target: float,
) -> dict[str, object]:
    """Compare top-k degradation against the random-k drop distribution.

    Lower accuracy or p(true) means more damage. Degradation is measured
    relative to the baseline, never by comparing raw metric values.
    """
    baseline = next(row for row in rows if row.get("condition") == "baseline")
    top = next(row for row in rows if row.get("condition") == "top")
    bottom = next(row for row in rows if row.get("condition") == "bottom")
    random_rows = [row for row in rows if str(row.get("condition", "")).startswith("random-")]

    summary: dict[str, object] = {"equal_norm_target": round(float(equal_norm_target), 4)}
    supported = 0
    for metric in ("accuracy", "p_true"):
        base = float(baseline[metric])
        random_values = np.asarray([float(row[metric]) for row in random_rows])
        random_drops = base - random_values
        top_drop = base - float(top[metric])
        p95 = float(np.percentile(random_drops, 95))
        exceeds = bool(top_drop > p95)
        supported += int(exceeds)
        summary[f"{metric}_baseline"] = round(base, 4)
        summary[f"{metric}_top"] = round(float(top[metric]), 4)
        summary[f"{metric}_bottom"] = round(float(bottom[metric]), 4)
        summary[f"{metric}_random_mean"] = round(float(random_values.mean()), 4)
        summary[f"{metric}_random_p95_drop"] = round(p95, 4)
        summary[f"{metric}_top_drop"] = round(top_drop, 4)
        summary[f"{metric}_top_exceeds_random_p95"] = exceeds

    summary["metrics_supported"] = supported
    summary["top_exceeds_random_p95"] = supported >= 2
    return summary
