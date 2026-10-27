"""Batched CETT extraction must equal per-sample extraction."""

from _fake_models import FakeEsmModel, FakeEsmTokenizer
from equivalence_utils import assert_batch_matches_singles

from biointerp import esm
from biointerp.task import make_masked_batch


def test_esm_cett_batch_matches_single_rows():
    model = FakeEsmModel(hidden=16, intermediate=24, n_layers=4, seed=0).eval()
    tokenizer = FakeEsmTokenizer()
    sequences = ["ACACACACAC", "ACACACACAC", "ACACACACAC"]
    tokens, positions, _ = make_masked_batch(sequences, tokenizer, masks_per_seq=2, seed=5)

    layers = [0, 2, 3]
    norms = esm.precompute_col_norms(model, layers)
    aa_ids = esm.aa_token_ids(tokenizer)
    batched = esm.extract_cett_batch(model, tokens, positions, layers, norms, aa_ids)

    cett_singles, prob_singles = [], []
    for row in range(len(sequences)):
        single_tokens = {k: v[row : row + 1] for k, v in tokens.items()}
        single_positions = positions[row : row + 1]
        single = esm.extract_cett_batch(
            model, single_tokens, single_positions, layers, norms, aa_ids
        )
        cett_singles.extend(single["cett"])
        prob_singles.extend(single["probs"])

    assert_batch_matches_singles(batched["cett"], cett_singles)
    assert_batch_matches_singles(batched["probs"], prob_singles)


def test_esm_cett_rows_match_individual_positions():
    model = FakeEsmModel(hidden=16, intermediate=24, n_layers=2, seed=1).eval()
    tokenizer = FakeEsmTokenizer()
    tokens, positions, _ = make_masked_batch(["ACACACAC"], tokenizer, masks_per_seq=2, seed=2)
    layers = [0, 1]
    norms = esm.precompute_col_norms(model, layers)
    aa_ids = esm.aa_token_ids(tokenizer)

    batched = esm.extract_cett_batch(model, tokens, positions, layers, norms, aa_ids)
    singles = []
    for m in range(positions.shape[1]):
        single = esm.extract_cett_batch(
            model, tokens, positions[:, m : m + 1], layers, norms, aa_ids
        )
        singles.append(single["cett"][0])

    assert_batch_matches_singles(batched["cett"], singles)
