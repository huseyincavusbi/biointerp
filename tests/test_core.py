import numpy as np

from biointerp.core import split_by_protein, top_probe_scores, univariate_auroc


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


def test_split_by_protein_has_no_leakage():
    protein = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    train, test = split_by_protein(protein, test_frac=0.25, seed=0)
    assert set(protein[train]).isdisjoint(set(protein[test]))
    assert test.sum() > 0


def test_top_probe_scores_separable():
    rng = np.random.default_rng(0)
    x_train = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(3, 1, (20, 2))])
    y_train = np.array([0] * 20 + [1] * 20)
    x_test = np.vstack([rng.normal(0, 1, (5, 2)), rng.normal(3, 1, (5, 2))])
    scores, _ = top_probe_scores(x_train, y_train, x_test, top_k=1)
    assert scores[:5].mean() < scores[5:].mean()
