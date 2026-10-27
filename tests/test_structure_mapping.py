import numpy as np
import torch
from _fake_models import MASK_ID, FakeEsmModel, FakeEsmTokenizer

from biointerp.esm import AA_ALPHABET, aa_token_ids
from biointerp.experiments.structure_mapping import leave_one_out_p_true


def test_leave_one_out_matches_manual_forward():
    model = FakeEsmModel(hidden=16, intermediate=24, n_layers=3, seed=0).eval()
    tokenizer = FakeEsmTokenizer()
    sequence = "ACDEFG"
    aa_t = torch.tensor(aa_token_ids(tokenizer))
    aa_index = {aa: i for i, aa in enumerate(AA_ALPHABET)}

    p_true = leave_one_out_p_true(model, tokenizer, sequence, "cpu", aa_t, aa_index)

    assert p_true.shape == (len(sequence),)
    assert np.all((p_true >= 0.0) & (p_true <= 1.0))

    j = 3
    ids = tokenizer(sequence, add_special_tokens=True)["input_ids"]
    row = list(ids)
    row[j] = MASK_ID
    out = model(
        input_ids=torch.tensor([row]),
        attention_mask=torch.ones(1, len(row), dtype=torch.long),
    )
    probs = torch.softmax(out.logits[0, j, aa_t].float(), dim=-1)
    expected = float(probs[aa_index[sequence[j - 1]]].detach())
    assert abs(p_true[j - 1] - expected) < 1e-6


def test_leave_one_out_length_tracks_sequence():
    model = FakeEsmModel(hidden=8, intermediate=12, n_layers=2).eval()
    tokenizer = FakeEsmTokenizer()
    aa_index = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    p_true = leave_one_out_p_true(
        model,
        tokenizer,
        "ACDEFGHIKL",
        "cpu",
        torch.tensor(aa_token_ids(tokenizer)),
        aa_index,
    )
    assert p_true.shape == (10,)
