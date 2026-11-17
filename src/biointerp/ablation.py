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
        layer: np.asarray(vector, dtype=float) * scale
        for layer, vector in mean_by_layer.items()
    }


def condition_vectors(
    mean_z: Mapping[int, np.ndarray],
    indices: Sequence[int],
    intermediate: int,
    target_norm: float,
) -> dict[int, np.ndarray]:
    """Build equal-norm per-layer replacement vectors for a neuron set."""
    grouped = group_by_layer(indices, intermediate)
    raw = {
        layer: np.asarray(mean_z[layer])[np.asarray(local)]
        for layer, local in grouped.items()
    }
    return equalize_norms(raw, target_norm)


def global_norm(mean_by_layer: Mapping[int, np.ndarray]) -> float:
    return float(
        np.sqrt(sum(float(np.sum(np.asarray(vector) ** 2)) for vector in mean_by_layer.values()))
    )
