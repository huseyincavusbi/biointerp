import numpy as np
import pytest

from biointerp.clusters import (
    CORE_ASSAYS,
    cluster_sequences,
    exclude_clusters,
    merge_clusters,
    mmseqs_available,
    parse_cluster_tsv,
    parse_fasta_text,
    parse_m8,
    read_fasta,
    split_by_cluster,
    write_fasta,
)

requires_mmseqs = pytest.mark.skipif(not mmseqs_available(), reason="mmseqs2 not installed")


def test_parse_fasta_text_keeps_first_header_token():
    records = parse_fasta_text(">sp|P1|name extra words\nACDE\nFGHI\n\n>P2\nKLMN\n")
    assert records == [("sp|P1|name", "ACDEFGHI"), ("P2", "KLMN")]


def test_fasta_roundtrip(tmp_path):
    records = [("a", "ACDE"), ("b", "FGHI")]
    path = write_fasta(records, tmp_path / "seqs.fasta")
    assert read_fasta(path) == records


def test_parse_cluster_tsv_maps_members_and_representatives(tmp_path):
    path = tmp_path / "clusters_cluster.tsv"
    path.write_text("rep1\tmem1\nrep1\trep1\nrep2\tmem2\n")
    cluster_of = parse_cluster_tsv(path)
    assert cluster_of == {"mem1": "rep1", "rep1": "rep1", "mem2": "rep2", "rep2": "rep2"}


def test_parse_cluster_tsv_rejects_malformed(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("only_one_column\n")
    with pytest.raises(ValueError):
        parse_cluster_tsv(path)


def test_parse_m8(tmp_path):
    path = tmp_path / "hits.m8"
    path.write_text("q1\tt1\t42.5\nq2\tt2\t31.0\n")
    assert parse_m8(path) == [("q1", "t1", 42.5), ("q2", "t2", 31.0)]


def test_parse_m8_missing_file_is_empty(tmp_path):
    assert parse_m8(tmp_path / "missing.m8") == []


def test_split_by_cluster_never_splits_a_cluster():
    ids = [f"s{i}" for i in range(8)]
    cluster_of = {f"s{i}": f"c{i // 2}" for i in range(8)}
    train, test, info = split_by_cluster(ids, cluster_of, test_frac=0.25, seed=0)

    train_clusters = {cluster_of[ids[i]] for i in np.flatnonzero(train)}
    test_clusters = {cluster_of[ids[i]] for i in np.flatnonzero(test)}
    assert train_clusters.isdisjoint(test_clusters)
    assert info["n_train"] + info["n_test"] == 8


def test_split_by_cluster_is_deterministic():
    ids = [f"s{i}" for i in range(20)]
    cluster_of = {f"s{i}": f"c{i // 3}" for i in range(20)}
    first = split_by_cluster(ids, cluster_of, test_frac=0.2, seed=7)
    second = split_by_cluster(ids, cluster_of, test_frac=0.2, seed=7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_split_by_cluster_rejects_missing_ids():
    with pytest.raises(KeyError):
        split_by_cluster(["a", "b"], {"a": "c1"})


def test_split_by_cluster_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_by_cluster(["a"], {"a": "c1"}, test_frac=1.0)


def test_exclude_clusters_returns_whole_clusters():
    cluster_of = {"a": "c1", "b": "c1", "c": "c2"}
    assert exclude_clusters(cluster_of, {f"ASSAY_{CORE_ASSAYS[0]}", "a"}) == {"c1"}


def test_merge_clusters_unions_hit_pairs():
    cluster_of = {"a": "c1", "b": "c2", "c": "c3", "d": "c4"}
    merged = merge_clusters(cluster_of, [("a", "b"), ("c", "d")])
    assert merged["a"] == merged["b"]
    assert merged["c"] == merged["d"]
    assert merged["a"] != merged["c"]


def test_merge_clusters_is_idempotent():
    cluster_of = {"a": "c1", "b": "c1"}
    assert merge_clusters(merge_clusters(cluster_of, [("a", "b")]), [("a", "b")]) == cluster_of


@requires_mmseqs
@pytest.mark.slow
def test_mmseqs_clusters_identical_sequences(tmp_path):
    fasta = tmp_path / "toy.fasta"
    write_fasta(
        [
            ("h1", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"),
            ("h2", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"),
            ("s1", "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"),
            ("s2", "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"),
        ],
        fasta,
    )
    cluster_of = cluster_sequences(fasta, tmp_path / "work", min_seq_id=0.3, coverage=0.8)
    assert cluster_of["h1"] == cluster_of["h2"]
    assert cluster_of["s1"] != cluster_of["s2"]
