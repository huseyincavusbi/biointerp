import torch
import torch.nn.functional as F
from _fake_models import AA, FakeEsmcModel, FakeEsmcTokenizer

from biointerp import esmc


def test_get_blocks_and_col_norms():
    model = FakeEsmcModel(hidden=8, intermediate=12, n_layers=3)
    blocks = esmc.get_blocks(model)
    assert len(blocks) == 3

    norms = esmc.precompute_col_norms(model)
    assert sorted(norms) == [0, 1, 2]
    for i, block in enumerate(blocks):
        expected = torch.norm(block.ffn.fc2_weight.float(), dim=0)
        assert norms[i].shape == (12,)
        torch.testing.assert_close(norms[i], expected)


def test_aa_token_ids_match_esmc_vocabulary():
    ids = esmc.aa_token_ids()
    assert ids == [esmc.ESMC_AA_IDS[aa] for aa in AA]
    assert len(set(ids)) == 20


def test_pick_device_explicit():
    assert esmc.pick_device("cpu") == "cpu"


def test_extract_cett_sequence_shapes_and_probabilities():
    model = FakeEsmcModel(hidden=16, intermediate=24, n_layers=3).eval()
    tokenizer = FakeEsmcTokenizer()
    norms = esmc.precompute_col_norms(model)

    cett, probs = esmc.extract_cett_sequence(model, tokenizer, "ACDEFG", [1, 3], norms)

    assert cett.shape == (2, 3 * 24)
    assert probs.shape == (2, 20)
    assert torch.isfinite(cett).all()
    assert (cett >= 0).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(2))
    for block in esmc.get_blocks(model):
        assert len(block.ffn._forward_hooks) == 0


def test_extract_cett_sequence_matches_manual_formula():
    model = FakeEsmcModel(hidden=16, intermediate=24, n_layers=2).eval()
    tokenizer = FakeEsmcTokenizer()
    norms = esmc.precompute_col_norms(model)
    sequence = "ACDEFG"
    mask_positions = [1, 4]

    cett, _ = esmc.extract_cett_sequence(model, tokenizer, sequence, mask_positions, norms)

    masked = list(sequence)
    for p in mask_positions:
        masked[p] = "<mask>"
    inputs = tokenizer("".join(masked), return_tensors="pt")
    token_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero().flatten()

    captured_x, captured_h = {}, {}
    handles = []
    for i, block in enumerate(esmc.get_blocks(model)):

        def hook(module, inps, out, i=i):
            captured_x[i] = inps[0][0].detach().float()
            captured_h[i] = out[0].detach().float()

        handles.append(block.ffn.register_forward_hook(hook))
    try:
        model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    for m, pos in enumerate(token_positions.tolist()):
        parts = []
        for i, block in enumerate(esmc.get_blocks(model)):
            ffn = block.ffn
            x = captured_x[i][pos]
            xn = F.layer_norm(
                x,
                (ffn.hidden_size,),
                ffn.layer_norm_weight.float(),
                ffn.layer_norm_bias.float(),
                ffn.eps,
            )
            x12 = F.linear(xn, ffn.fc1_weight.float())
            z = F.silu(x12[: ffn.ffn_hidden_size]) * x12[ffn.ffn_hidden_size :]
            h_norm = float(captured_h[i][pos].norm()) + 1e-8
            parts.append(z.abs() * norms[i] / h_norm)
        expected = torch.cat(parts)
        torch.testing.assert_close(cett[m], expected, atol=1e-6, rtol=1e-5)
