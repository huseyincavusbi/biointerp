import numpy as np

from biointerp.experiments import layer_disorder_resolution as ld


def test_stratified_split_keeps_disordered_sequences_on_both_sides():
    seq_id = np.repeat(np.arange(6), 4)
    pos_mask = np.zeros(len(seq_id), dtype=bool)
    pos_mask[(seq_id == 0) | (seq_id == 2)] = True

    train, test = ld.stratified_seq_split(seq_id, pos_mask, test_frac=0.5, seed=0)

    train_seqs = set(seq_id[train].tolist())
    test_seqs = set(seq_id[test].tolist())
    assert train_seqs.isdisjoint(test_seqs)
    assert any(pos_mask[seq_id == s].any() for s in train_seqs)
    assert any(pos_mask[seq_id == s].any() for s in test_seqs)


def test_stratified_split_has_no_position_leakage():
    seq_id = np.repeat(np.arange(8), 3)
    pos_mask = seq_id == 3
    train, test = ld.stratified_seq_split(seq_id, pos_mask, test_frac=0.25, seed=1)
    assert train.shape == test.shape == seq_id.shape
    assert np.array_equal(train, ~test)
    assert set(seq_id[train]).isdisjoint(set(seq_id[test]))


def test_grouped_probe_auc_separable():
    rng = np.random.default_rng(0)
    n_seq, per_seq = 10, 4
    groups = np.repeat(np.arange(n_seq), per_seq)
    y = np.tile([0, 0, 1, 1], n_seq)
    x = np.column_stack([3.0 * y + rng.normal(0, 0.1, len(y)), rng.normal(size=(len(y), 2))])

    auc, scores, te_idx, te_y = ld.grouped_probe_auc(x, y, groups, seed=0)

    assert auc > 0.95
    assert scores.shape == (len(te_idx),)
    assert set(te_y.tolist()) == {0, 1}
    assert set(groups[te_idx].tolist()).isdisjoint(
        set(groups[np.setdiff1d(np.arange(len(y)), te_idx)].tolist())
    )


def test_grouped_probe_auc_single_class_returns_nan():
    groups = np.repeat(np.arange(6), 3)
    x = np.random.default_rng(1).normal(size=(len(groups), 2))
    y = np.zeros(len(groups), dtype=int)

    auc, scores, te_idx, te_y = ld.grouped_probe_auc(x, y, groups, seed=0)

    assert np.isnan(auc)
    assert np.all(scores == 0)
    assert np.all(te_y == 0)
    assert len(te_idx) == scores.shape[0]


def test_bootstrap_first_crossing_finds_later_layer():
    n_seq, per_seq = 10, 4
    groups = np.repeat(np.arange(n_seq), per_seq)
    y = np.tile([0, 1, 0, 1], n_seq)
    scores = [
        np.where(y == 1, 0.4, 0.6),  # AUC 0
        np.where(y == 1, 0.9, 0.1),  # AUC 1
    ]

    median, ci = ld.bootstrap_first_crossing(scores, y, groups, thresh=0.65, n_boot=20, seed=0)

    assert median == 1.0
    assert ci == [1.0, 1.0]


def test_bootstrap_first_crossing_returns_none_without_crossing():
    n_seq, per_seq = 8, 3
    groups = np.repeat(np.arange(n_seq), per_seq)
    y = np.tile([0, 1, 0], n_seq)
    scores = [np.zeros_like(y)]

    median, ci = ld.bootstrap_first_crossing(scores, y, groups, thresh=0.65, n_boot=10, seed=0)

    assert median is None
    assert ci is None
