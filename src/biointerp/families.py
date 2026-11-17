"""Family assignments and leave-one-family-out splits.

Families are a configuration input: a table with a protein id and a family
label. Pfam annotations, sequence clusters, or curated families can all be
plugged in without changing the experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def load_family_table(
    path: str | Path,
    protein_column: str = "protein_id",
    family_column: str = "family",
) -> dict[str, str]:
    """Read a CSV/TSV family table into ``protein -> family``."""
    table_path = Path(path)
    if not table_path.exists():
        raise SystemExit(f"family table not found: {table_path}")
    separator = "\t" if table_path.suffix in {".tsv", ".txt"} else ","
    lines = table_path.read_text().splitlines()
    if not lines:
        return {}
    header = [item.strip() for item in lines[0].split(separator)]
    if protein_column not in header or family_column not in header:
        raise SystemExit(f"family table needs columns {protein_column!r} and {family_column!r}")
    protein_index = header.index(protein_column)
    family_index = header.index(family_column)
    mapping: dict[str, str] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split(separator)
        if len(fields) <= max(protein_index, family_index):
            continue
        mapping[fields[protein_index].strip()] = fields[family_index].strip()
    return mapping


def family_splits(
    ids: Sequence[str],
    families: Mapping[str, str],
    min_members: int = 5,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return ``(family, train_mask, test_mask)`` for each eligible family.

    The test mask is the whole family; the train mask excludes it entirely.
    """
    ids_array = np.asarray(list(ids), dtype=object)
    labels = np.asarray([families.get(str(item), "") for item in ids_array], dtype=object)
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    for family in sorted({label for label in labels if label}):
        test = labels == family
        if int(test.sum()) < min_members:
            continue
        train = ~test
        if not train.any():
            continue
        splits.append((str(family), train, test))
    return splits
