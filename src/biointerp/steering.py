"""Steering condition definitions and direction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Condition:
    name: str
    direction: str = "none"  # none | true | negative | random
    layer: int = 24
    alpha: float = 0.3


VALID_DIRECTIONS = {"none", "true", "negative", "random"}


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
            )
        )
    if not conditions:
        raise ValueError("no conditions configured")
    return conditions


def negate_direction(vector: Sequence[float]) -> np.ndarray:
    return -np.asarray(vector, dtype=float)


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
    else:
        vector = np.asarray(directions[condition.layer], dtype=float)
        if condition.direction == "negative":
            vector = negate_direction(vector)

    norm = float(np.linalg.norm(vector))
    effective_alpha = condition.alpha * mean_norms[condition.layer] / (norm + 1e-8)
    return {condition.layer: (vector, effective_alpha)}


def summarize_conditions(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Simple aggregation for condition rows."""
    baseline = next((row for row in rows if row.get("condition") == "baseline"), None)
    values = {
        str(row["condition"]): float(row["mean_naturalness"])
        for row in rows
        if row.get("mean_naturalness") is not None
    }
    deltas = {}
    if baseline is not None:
        base = float(baseline["mean_naturalness"])
        deltas = {name: round(value - base, 4) for name, value in values.items()}
    return {
        "conditions": list(values),
        "naturalness": values,
        "delta_vs_baseline": deltas,
    }
