import numpy as np
import torch
from _fake_models import AA, AA_TO_ID, FakeEsmModel, FakeEsmTokenizer

from biointerp import esm
from biointerp.task import make_masked_batch


def _model_and_batch(seqs, masks_per_seq=1, seed=0):
    model = FakeEsmModel(hidden=16, intermediate=24, n_layers=4, seed=0).eval()
    tokenizer = FakeEsmTokenizer()
    tokens, positions, _ = make_masked_batch(seqs, tokenizer, masks_per_seq, seed=seed)
    return model, tokenizer, tokens, positions


def test_pick_device_explicit():
    assert esm.pick_device("cpu") == "cpu"


def test_layer_and_projection_discovery():
    model = FakeEsmModel(n_layers=3)
    layers = esm.get_transformer_layers(model)
    assert len(layers) == 3
    assert esm.get_ffn_output_proj(layers[0]) is layers[0].output.dense


def test_precompute_col_norms_matches_manual():
    model = FakeEsmModel(hidden=8, intermediate=12, n_layers=3)
    norms = esm.precompute_col_norms(model, [0, 2])
    assert sorted(norms) == [0, 2]
    for layer in (0, 2):
        expected = torch.norm(model.esm.encoder.layer[layer].output.dense.weight, dim=0)
        assert norms[layer].shape == (12,)
        torch.testing.assert_close(norms[layer], expected)


def test_aa_token_ids_cover_alphabet():
    tokenizer = FakeEsmTokenizer()
    ids = esm.aa_token_ids(tokenizer)
    assert len(ids) == 20
    assert len(set(ids)) == 20
    assert ids == [AA_TO_ID[aa] for aa in AA]


def test_extract_cett_batch_shapes_and_probabilities():
    model, tokenizer, tokens, positions = _model_and_batch(["ACACACAC", "ACACACAC"])
    layers = [0, 2]
    norms = esm.precompute_col_norms(model, layers)
    res = esm.extract_cett_batch(
        model, tokens, positions, layers, norms, esm.aa_token_ids(tokenizer)
    )

    assert res["cett"].shape == (2, 2 * 24)
    assert res["logits_aa"].shape == (2, 20)
    assert res["probs"].shape == (2, 20)
    assert torch.isfinite(res["cett"]).all()
    assert (res["cett"] >= 0).all()
    torch.testing.assert_close(res["probs"].sum(-1), torch.ones(2))
    for layer in layers:
        assert len(model.esm.encoder.layer[layer].output.dense._forward_hooks) == 0


def test_extract_cett_matches_manual_formula():
    model, tokenizer, tokens, positions = _model_and_batch(["ACACACAC"], masks_per_seq=2, seed=1)
    layer = 1
    norms = esm.precompute_col_norms(model, [layer])
    res = esm.extract_cett_batch(
        model, tokens, positions, [layer], norms, esm.aa_token_ids(tokenizer)
    )

    captured = {}
    handle = model.esm.encoder.layer[layer].output.dense.register_forward_hook(
        lambda module, inputs, output: captured.update(z=inputs[0], h=output)
    )
    try:
        model(**tokens)
    finally:
        handle.remove()

    for m in range(2):
        pos = int(positions[0, m])
        z = captured["z"][0, pos].float()
        h = captured["h"][0, pos].float()
        expected = z.abs() * norms[layer] / (h.norm() + 1e-8)
        torch.testing.assert_close(res["cett"][m], expected, atol=1e-6, rtol=1e-5)


def test_extract_cett_flattens_masks_in_row_major_order():
    model, tokenizer, tokens, positions = _model_and_batch(
        ["ACACACACAC", "ACACACACAC"], masks_per_seq=2, seed=2
    )
    norms = esm.precompute_col_norms(model, [0])
    res = esm.extract_cett_batch(model, tokens, positions, [0], norms, esm.aa_token_ids(tokenizer))

    for row in range(2):
        single_tokens = {k: v[row : row + 1] for k, v in tokens.items()}
        single_positions = positions[row : row + 1]
        single = esm.extract_cett_batch(
            model, single_tokens, single_positions, [0], norms, esm.aa_token_ids(tokenizer)
        )
        torch.testing.assert_close(res["cett"][row * 2 : row * 2 + 2], single["cett"])


def test_extract_cett_probabilities_are_finite():
    model, tokenizer, tokens, positions = _model_and_batch(["ACACACAC"], masks_per_seq=2, seed=3)
    norms = esm.precompute_col_norms(model, [0, 1, 2, 3])
    res = esm.extract_cett_batch(
        model, tokens, positions, [0, 1, 2, 3], norms, esm.aa_token_ids(tokenizer)
    )
    assert np.isfinite(res["cett"].numpy()).all()
    assert np.isfinite(res["probs"].numpy()).all()
