import torch
from _fake_dna_models import FakeCarbon

from biointerp import causal


def test_hidden_states_at_shapes():
    model = FakeCarbon(hidden=8, intermediate=12, n_layers=2).eval()
    tokens = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    positions = torch.tensor([[2]])
    states = causal.hidden_states_at(model, tokens, positions, [0, 1], "carbon")
    assert states[0].shape == (1, 1, 8)
    assert states[1].shape == (1, 1, 8)


def test_patch_hidden_changes_logits():
    model = FakeCarbon(hidden=8, intermediate=12, n_layers=2).eval()
    tokens = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    positions = torch.tensor([[2]])
    donor = torch.full((1, 1, 8), 0.25)

    base = model(**tokens).logits
    patched = causal.patch_hidden(model, tokens, 1, positions, donor, "carbon").logits

    assert patched.shape == base.shape
    assert torch.isfinite(patched).all()
    assert not torch.allclose(base, patched)
