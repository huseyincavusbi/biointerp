from types import SimpleNamespace

import numpy as np
import torch
from _fake_models import FakeEsmModel, FakeEsmTokenizer, FakeGpt2Model, FakeGpt2Tokenizer

from biointerp.esm import AA_ALPHABET, aa_token_ids
from biointerp.experiments.generative_causal import generate_and_label, step_labels


def test_step_labels_skips_non_amino_acids():
    aa_index = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    texts = ["A", "C", "X", "D"]
    naturalness = np.array([0.1, 0.3, 0.7])

    labels, valid = step_labels(texts, naturalness, aa_index)

    assert valid == [0, 1, 3]
    np.testing.assert_allclose(labels, [0.1, 0.3, 0.7])


def test_generate_and_label_returns_hidden_states():
    gen_model = FakeGpt2Model(hidden=12, intermediate=20, n_layers=2, eos_bias=-100.0)
    gen_tok = FakeGpt2Tokenizer()
    esm_model = FakeEsmModel(hidden=16, intermediate=24, n_layers=3).eval()
    esm_tok = FakeEsmTokenizer()
    aa_t = torch.tensor(aa_token_ids(esm_tok))
    aa_index = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    args = SimpleNamespace(n_steps=12, temperature=1.0, seed=0)

    result = generate_and_label(
        gen_model,
        gen_tok,
        esm_model,
        esm_tok,
        aa_t,
        aa_index,
        args,
        "cpu",
        seed_offset=0,
        capture_layers=[0, 1],
    )

    assert result is not None
    assert len(result["seq"]) == 12
    assert len(result["labels"]) == len(result["valid"]) == 12
    assert set(result["hidden"]) == {0, 1}
    assert len(result["hidden"][0]) == 12
    assert result["hidden"][0][0].shape == (12,)
    assert all(np.isfinite(v) for v in result["labels"])


def test_generate_and_label_rejects_short_sequences():
    gen_model = FakeGpt2Model(eos_bias=-100.0)
    gen_tok = FakeGpt2Tokenizer()
    esm_model = FakeEsmModel().eval()
    esm_tok = FakeEsmTokenizer()
    aa_index = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    args = SimpleNamespace(n_steps=5, temperature=1.0, seed=0)

    result = generate_and_label(
        gen_model,
        gen_tok,
        esm_model,
        esm_tok,
        torch.tensor(aa_token_ids(esm_tok)),
        aa_index,
        args,
        "cpu",
        seed_offset=0,
        capture_layers=[0],
    )

    assert result is None
