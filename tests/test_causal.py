from types import SimpleNamespace

import torch
import torch.nn as nn
from _fake_dna_models import FakeCarbon

from biointerp import causal


class HeadLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.self_attn.o_proj(x)


class HeadModel(nn.Module):
    def __init__(self, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden,
            num_attention_heads=4,
            head_dim=hidden // 4,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(HeadLayer(hidden) for _ in range(n_layers))

    def forward(self, input_ids, attention_mask=None):
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(logits=x)


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


def test_ablate_heads_changes_logits():
    model = HeadModel().eval()
    tokens = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    base = model(**tokens).logits
    ablated = causal.ablate_heads(model, tokens, 0, [0, 2], "carbon").logits
    assert ablated.shape == base.shape
    assert torch.isfinite(ablated).all()
    assert not torch.allclose(base, ablated)


def test_ablate_heads_rejects_out_of_range():
    model = HeadModel().eval()
    try:
        with causal.ablate_heads_context(model, 0, [4], "carbon"):
            pass
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range head should fail")


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
