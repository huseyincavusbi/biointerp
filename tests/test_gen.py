from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from _fake_models import (
    EOS_ID,
    FakeGpt2Model,
    FakeGpt2Tokenizer,
    LayerOnly,
)

from biointerp import gen


def test_get_transformer_layers_gpt2_style():
    model = FakeGpt2Model(n_layers=2)
    assert len(gen.get_transformer_layers(model)) == 2


def test_get_transformer_layers_decoder_style():
    model = SimpleNamespace(model=SimpleNamespace(decoder=SimpleNamespace(layers=[1, 2, 3])))
    assert gen.get_transformer_layers(model) == [1, 2, 3]


def test_get_transformer_layers_llama_style():
    model = SimpleNamespace(model=SimpleNamespace(layers=[1, 2]))
    assert gen.get_transformer_layers(model) == [1, 2]


def test_get_transformer_layers_unsupported():
    with pytest.raises(ValueError, match="Unsupported generative architecture"):
        gen.get_transformer_layers(SimpleNamespace())


@pytest.mark.parametrize("name", ["c_proj", "down_proj", "fc2", "o_proj", "fc_out"])
def test_get_ffn_output_proj_discovers_names(name):
    proj = nn.Linear(3, 3)
    layer = SimpleNamespace(mlp=SimpleNamespace(**{name: proj}))
    assert gen.get_ffn_output_proj(layer) is proj


def test_get_ffn_output_proj_missing_raises():
    with pytest.raises(ValueError, match="No FFN output projection"):
        gen.get_ffn_output_proj(SimpleNamespace(mlp=SimpleNamespace()))


def test_precompute_col_norms_linear_uses_output_dim():
    layer = LayerOnly(SimpleNamespace(c_proj=nn.Linear(12, 8)))
    model = SimpleNamespace(transformer=SimpleNamespace(h=[layer]))
    norms = gen.precompute_col_norms(model, [0])
    assert norms[0].shape == (12,)
    torch.testing.assert_close(norms[0], torch.norm(layer.mlp.c_proj.weight, dim=0))


def test_precompute_col_norms_conv1d_uses_input_dim():
    class Conv1D(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_features, out_features))

    layer = LayerOnly(SimpleNamespace(c_proj=Conv1D(12, 8)))
    model = SimpleNamespace(transformer=SimpleNamespace(h=[layer]))
    norms = gen.precompute_col_norms(model, [0])
    assert norms[0].shape == (12,)
    torch.testing.assert_close(norms[0], torch.norm(layer.mlp.c_proj.weight, dim=1))


def test_ensure_head_mask_adds_compat_shim():
    class Transformer:
        pass

    gen._ensure_head_mask(SimpleNamespace(transformer=Transformer()))
    assert hasattr(Transformer, "get_head_mask")
    assert Transformer().get_head_mask(None, 3) == [None, None, None]
    sentinel = [[1.0]]
    assert Transformer().get_head_mask(sentinel, 3) == sentinel


def test_ensure_head_mask_preserves_existing():
    class Transformer:
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            return "original"

    gen._ensure_head_mask(SimpleNamespace(transformer=Transformer()))
    assert Transformer().get_head_mask(None, 3) == "original"


def test_generate_with_cett_shapes():
    model = FakeGpt2Model(hidden=12, intermediate=20, n_layers=2, eos_bias=-100.0)
    tokenizer = FakeGpt2Tokenizer()

    ids, cetts, texts, confs, ents = gen.generate_with_cett(
        model, tokenizer, n_steps=4, seed=0, greedy=True, device="cpu"
    )

    assert len(ids) == len(cetts) == len(texts) == len(confs) == len(ents) == 4
    assert all(isinstance(i, int) and i != EOS_ID for i in ids)
    for cett in cetts:
        assert cett.shape == (2 * 20,)
        assert torch.isfinite(cett).all()
    assert all(0.0 <= c <= 1.0 for c in confs)
    assert all(e >= 0.0 for e in ents)
    for layer in model.transformer.h:
        assert len(layer.mlp.c_proj._forward_hooks) == 0


def test_generate_with_cett_stops_at_eos():
    model = FakeGpt2Model(eos_bias=100.0)
    tokenizer = FakeGpt2Tokenizer()
    ids, cetts, texts, confs, ents = gen.generate_with_cett(
        model, tokenizer, n_steps=4, seed=0, greedy=True, device="cpu"
    )
    assert ids == cetts == texts == confs == ents == []


def test_generate_with_cett_is_seed_deterministic():
    model = FakeGpt2Model(seed=0)
    tokenizer = FakeGpt2Tokenizer()
    first = gen.generate_with_cett(model, tokenizer, n_steps=5, seed=7, greedy=False, device="cpu")
    second = gen.generate_with_cett(model, tokenizer, n_steps=5, seed=7, greedy=False, device="cpu")
    assert first[0] == second[0]
    for a, b in zip(first[1], second[1]):
        torch.testing.assert_close(a, b)


def test_generate_with_cett_captures_hidden_states():
    model = FakeGpt2Model(hidden=12, n_layers=2)
    tokenizer = FakeGpt2Tokenizer()
    out = gen.generate_with_cett(
        model,
        tokenizer,
        n_steps=3,
        seed=1,
        greedy=True,
        device="cpu",
        capture_layers=[0, 1],
    )
    assert len(out) == 6
    ids, _, _, _, _, hidden = out
    assert hidden.keys() == {0, 1}
    assert len(hidden[0]) == len(ids) == len(hidden[1])
    assert hidden[0][0].shape == (12,)


def test_generate_with_cett_zero_steering_is_identity():
    model = FakeGpt2Model(hidden=12, n_layers=2)
    tokenizer = FakeGpt2Tokenizer()
    base = gen.generate_with_cett(model, tokenizer, n_steps=4, seed=3, greedy=True, device="cpu")
    steered = gen.generate_with_cett(
        model,
        tokenizer,
        n_steps=4,
        seed=3,
        greedy=True,
        device="cpu",
        steer_map={0: (np.zeros(12, dtype=np.float32), 0.0)},
    )
    assert base[0] == steered[0]
    for a, b in zip(base[1], steered[1]):
        torch.testing.assert_close(a, b)
