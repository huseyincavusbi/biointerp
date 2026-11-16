import pandas as pd

from biointerp.proteingym import from_config
from biointerp.vep import parse_mutations, prepare_rows


def test_dataset_config_is_not_hardcoded():
    dataset = from_config("proteingym")
    assert dataset.repo
    assert "{shard}" in dataset.file_pattern
    assert dataset.label == "DMS_score_bin"


def test_parse_mutations():
    assert parse_mutations("A12C") == [("A", 12, "C")]
    assert parse_mutations("A12C:G14D") == [("A", 12, "C"), ("G", 14, "D")]
    assert parse_mutations("12C") is None


def test_prepare_rows_keeps_single_substitutions_and_samples():
    frame = pd.DataFrame(
        {
            "DMS_id": ["TEST"] * 6,
            "mutant": ["A1C", "A1C:G2D", "A3C", "A4C", "A5C", "A6C"],
            "DMS_score": [0.0] * 6,
            "DMS_score_bin": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "target_seq": ["ACDEFG"] * 6,
        }
    )
    rows = prepare_rows(frame, n_mutations=3, seed=0)
    assert len(rows) == 3
    assert all(len(row["muts"]) == 1 for row in rows)
