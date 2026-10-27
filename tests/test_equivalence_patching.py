"""Activation-patching equivalence: self-patching must be an identity."""

import numpy as np
import torch
from _fake_models import FakeEsmModel, FakeEsmTokenizer
from equivalence_utils import assert_tensor_close

from biointerp.esm import aa_token_ids
from biointerp.experiments import activation_patching as ap


def test_self_patch_reproduces_baseline():
    model = FakeEsmModel(hidden=16, intermediate=24, n_layers=3, seed=0).eval()
    tokenizer = FakeEsmTokenizer()
    sequences = ["ACACACACAC"] * 6
    layers = [0, 2]
    aa_t = torch.tensor(aa_token_ids(tokenizer))

    hidden, correct, base_probs = ap.capture(
        model,
        tokenizer,
        sequences,
        layers,
        batch_size=3,
        seed=0,
        device="cpu",
        aa_t=aa_t,
    )
    assert hidden[0].shape == (6, 16)
    assert hidden[2].shape == (6, 16)
    assert base_probs.shape == (6, 20)
    np.testing.assert_allclose(base_probs.sum(axis=1), 1.0, rtol=1e-5)

    for layer in layers:
        patched_probs, patched_correct = ap.run_patch(
            model,
            tokenizer,
            sequences,
            layer,
            hidden[layer],
            batch_size=3,
            seed=0,
            device="cpu",
            aa_t=aa_t,
        )
        assert_tensor_close(patched_probs, base_probs, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(patched_correct, correct)
