from biointerp.assays import CORE_ASSAYS, core_names, get_assay, shard_for


def test_core_assays_are_unique():
    names = core_names()
    assert len(names) == 6
    assert len(set(names)) == 6


def test_shard_lookup():
    first = CORE_ASSAYS[0]
    assert get_assay(first.name) == first
    assert shard_for(first.name) == first.shard
    assert shard_for("not-an-assay") is None
