"""Steering condition definitions and direction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Condition:
    name: str
    direction: str = "none"  # none | true | negative | random | random_neurons
    layer: int = 24
    alpha: float = 0.3
    n_neurons: int = 25


VALID_DIRECTIONS = {"none", "true", "negative", "random", "random_neurons"}


def parse_conditions(payload: Mapping[str, object]) -> list[Condition]:
    """Parse a YAML payload into validated conditions."""
    conditions: list[Condition] = []
    for entry in payload.get("conditions", []):  # type: ignore[union-attr]
        if not isinstance(entry, dict):
            raise ValueError("each condition must be a mapping")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("condition without a name")
        direction = str(entry.get("direction", "none"))
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r} in {name!r}")
        conditions.append(
            Condition(
                name=name,
                direction=direction,
                layer=int(entry.get("layer", 24)),
                alpha=float(entry.get("alpha", 0.3)),
                n_neurons=int(entry.get("n_neurons", 25)),
            )
        )
    if not conditions:
        raise ValueError("no conditions configured")
    return conditions


def negate_direction(vector: Sequence[float]) -> np.ndarray:
    return -np.asarray(vector, dtype=float)


def random_neuron_direction(base: Sequence[float], n_neurons: int, seed: int = 0) -> np.ndarray:
    """Keep a random subset of direction components and preserve its norm."""
    base = np.asarray(base, dtype=float)
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if n_neurons >= len(base):
        return base.copy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(base), size=n_neurons, replace=False)
    sparse = np.zeros_like(base)
    sparse[chosen] = base[chosen]
    norm = float(np.linalg.norm(sparse))
    if norm < 1e-12:
        return base.copy()
    return sparse * (float(np.linalg.norm(base)) / norm)


def unit_random_direction(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimension)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero random direction")
    return vector / norm


def build_steer_map(
    condition: Condition,
    directions: Mapping[int, np.ndarray],
    mean_norms: Mapping[int, float],
    hidden_dim: int,
    seed: int = 0,
) -> dict[int, tuple[np.ndarray, float]] | None:
    """Build the per-layer steering map for one condition."""
    if condition.direction == "none":
        return None
    if condition.layer not in directions:
        raise KeyError(f"layer {condition.layer} has no calibrated direction")

    if condition.direction == "random":
        vector = unit_random_direction(hidden_dim, seed)
        scale = float(np.linalg.norm(directions[condition.layer]))
        vector = vector * scale
    elif condition.direction == "random_neurons":
        vector = random_neuron_direction(directions[condition.layer], condition.n_neurons, seed)
    else:
        vector = np.asarray(directions[condition.layer], dtype=float)
        if condition.direction == "negative":
            vector = negate_direction(vector)

    norm = float(np.linalg.norm(vector))
    effective_alpha = condition.alpha * mean_norms[condition.layer] / (norm + 1e-8)
    return {condition.layer: (vector, effective_alpha)}


def summarize_conditions(
    rows: Sequence[Mapping[str, object]],
    metric: str = "mean_naturalness",
) -> dict[str, object]:
    """Simple aggregation for condition rows."""
    baseline = next((row for row in rows if row.get("condition") == "baseline"), None)
    values = {
        str(row["condition"]): float(row[metric]) for row in rows if row.get(metric) is not None
    }
    deltas = {}
    if baseline is not None:
        base = float(baseline[metric])
        deltas = {name: round(value - base, 4) for name, value in values.items()}
    return {
        "metric": metric,
        "conditions": list(values),
        "scores": values,
        "naturalness": values,
        "delta_vs_baseline": deltas,
    }


HYDROPHOBIC = set("AVILMFWYC")


def sequence_features(sequence: str) -> tuple[float, float]:
    """Length and hydrophobic fraction, used for matching generated sets."""
    clean = [residue for residue in sequence if residue.isalpha()]
    if not clean:
        return 0.0, 0.0
    hydrophobic = sum(1 for residue in clean if residue in HYDROPHOBIC)
    return float(len(clean)), hydrophobic / len(clean)


def _bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=int)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.digitize(values, edges)


def match_by_bins(
    baseline: Sequence[str],
    condition: Sequence[str],
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Match two sequence sets on length and hydrophobic-fraction bins."""
    if not baseline or not condition:
        return np.array([], dtype=int), np.array([], dtype=int)
    base_features = np.asarray([sequence_features(seq) for seq in baseline])
    cond_features = np.asarray([sequence_features(seq) for seq in condition])
    combined_length = np.concatenate([base_features[:, 0], cond_features[:, 0]])
    combined_hydro = np.concatenate([base_features[:, 1], cond_features[:, 1]])
    length_bins = _bin(combined_length, n_bins)
    hydro_bins = _bin(combined_hydro, n_bins)
    base_keys = list(zip(length_bins[: len(baseline)], hydro_bins[: len(baseline)]))
    cond_keys = list(zip(length_bins[len(baseline) :], hydro_bins[len(baseline) :]))

    base_by_key: dict[tuple[int, int], list[int]] = {}
    cond_by_key: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(base_keys):
        base_by_key.setdefault(key, []).append(index)
    for index, key in enumerate(cond_keys):
        cond_by_key.setdefault(key, []).append(index)

    base_index: list[int] = []
    cond_index: list[int] = []
    for key in sorted(set(base_by_key) & set(cond_by_key)):
        count = min(len(base_by_key[key]), len(cond_by_key[key]))
        base_index.extend(base_by_key[key][:count])
        cond_index.extend(cond_by_key[key][:count])
    return np.asarray(base_index, dtype=int), np.asarray(cond_index, dtype=int)
