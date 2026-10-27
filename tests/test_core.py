import numpy as np

from biointerp.core import load_sequences, split_by_protein, top_probe_scores, univariate_auroc


def test_univariate_auroc_perfect():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    assert univariate_auroc(x, y)[0] == 1.0


def test_univariate_auroc_inverse():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1, 1, 0, 0])
    assert univariate_auroc(x, y)[0] == 0.0


def test_univariate_auroc_ties_are_chance():
    x = np.ones((4, 1))
    y = np.array([0, 1, 0, 1])
    assert univariate_auroc(x, y)[0] == 0.5


def test_univariate_auroc_partial_ties():
    x = np.array([[0.0], [1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    assert univariate_auroc(x, y)[0] == 0.875


def test_univariate_auroc_chunking_matches():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 25))
    y = rng.integers(0, 2, size=60)
    np.testing.assert_allclose(univariate_auroc(x, y, chunk=1), univariate_auroc(x, y, chunk=1000))


def test_split_by_protein_has_no_leakage():
    protein = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    train, test = split_by_protein(protein, test_frac=0.25, seed=0)
    assert set(protein[train]).isdisjoint(set(protein[test]))
    assert test.sum() > 0


def test_split_by_protein_fraction_and_determinism():
    protein = np.repeat(np.arange(10), 3)
    train, test = split_by_protein(protein, test_frac=0.2, seed=7)
    assert len(np.unique(protein[test])) == 2
    assert len(np.unique(protein[train])) == 8
    again = split_by_protein(protein, test_frac=0.2, seed=7)
    np.testing.assert_array_equal(test, again[1])


def test_split_by_protein_keeps_at_least_one_test_group():
    protein = np.zeros(6, dtype=int)
    train, test = split_by_protein(protein, test_frac=0.01, seed=0)
    assert test.sum() == 6
    assert train.sum() == 0


def test_load_sequences_multiline(tmp_path):
    path = tmp_path / "seqs.fasta"
    path.write_text(">a\nMKTAY\nIAK\n\n>b\nGAVLI\n")
    assert load_sequences(path) == ["MKTAYIAK", "GAVLI"]


def test_load_sequences_ignores_empty_file(tmp_path):
    path = tmp_path / "empty.fasta"
    path.write_text("\n\n")
    assert load_sequences(path) == []


def test_top_probe_scores_separable():
    rng = np.random.default_rng(0)
    x_train = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(3, 1, (20, 2))])
    y_train = np.array([0] * 20 + [1] * 20)
    x_test = np.vstack([rng.normal(0, 1, (5, 2)), rng.normal(3, 1, (5, 2))])
    scores, _ = top_probe_scores(x_train, y_train, x_test, top_k=1)
    assert scores[:5].mean() < scores[5:].mean()


def test_top_probe_scores_selects_top_k():
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(30, 6))
    y_train = np.array([0, 1] * 15)
    x_test = rng.normal(size=(8, 6))
    scores, idx = top_probe_scores(x_train, y_train, x_test, top_k=3)
    assert idx.shape == (3,)
    assert len(set(idx.tolist())) == 3
    assert scores.shape == (8,)
    assert np.all((scores >= 0) & (scores <= 1))
