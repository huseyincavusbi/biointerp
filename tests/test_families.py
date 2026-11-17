import numpy as np

from biointerp.families import family_splits, load_family_table


def test_load_family_table_csv(tmp_path):
    path = tmp_path / "families.csv"
    path.write_text("protein_id,family\nseq0,A\nseq1,B\n")
    assert load_family_table(path) == {"seq0": "A", "seq1": "B"}


def test_family_splits_hold_out_entire_family():
    ids = [f"seq{i}" for i in range(12)]
    families = {f"seq{i}": ("A" if i < 6 else "B") for i in range(12)}
    splits = family_splits(ids, families, min_members=3)
    assert [name for name, _, _ in splits] == ["A", "B"]
    for _name, train, test in splits:
        train_index = set(np.flatnonzero(train))
        test_index = set(np.flatnonzero(test))
        assert not (train_index & test_index)
        assert len(train_index) + len(test_index) == len(ids)


def test_family_splits_skip_small_families():
    ids = [f"seq{i}" for i in range(5)]
    families = {"seq0": "A", "seq1": "A", "seq2": "B", "seq3": "C", "seq4": "C"}
    splits = family_splits(ids, families, min_members=2)
    assert [name for name, _, _ in splits] == ["A", "C"]
