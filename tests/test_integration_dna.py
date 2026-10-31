"""Real DNABERT-2 smoke test (marked ``real_model``; runs on Linux CI)."""

import pytest
import torch

from biointerp.dna import extract_cett_masked, load_dna_model, precompute_col_norms
from biointerp.dna_task import masked_dna_batch, tokenize_dna

pytestmark = pytest.mark.real_model


def test_dnabert2_masked_forward_and_cett():
    spec = {"adapter": "dnabert", "hf_id": "zhihan1996/DNABERT-2-117M"}
    model, tokenizer, device = load_dna_model(spec, device="cpu")
    sequence = "ACGT" * 8
    ids, _ = tokenize_dna(sequence, tokenizer, "dnabert")
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    assert out.logits.shape[0] == 1
    assert out.logits.shape[-1] > 4
    assert len(ids) == len(inputs["input_ids"][0])
    norms = precompute_col_norms(model, "dnabert", [0])
    assert norms[0].ndim == 1

    tokens, positions, true_ids, _ = masked_dna_batch(
        [sequence], tokenizer, "dnabert", masks_per_seq=1, seed=0
    )
    result = extract_cett_masked(
        model, tokens, positions, [0], norms, "dnabert", capture_hidden=True
    )
    assert result["cett"].shape[0] == 1
    assert result["logits"].shape[0] == 1
    assert result["hidden_states"][0].shape[0] == 1
    assert true_ids.shape == (1, 1)
    assert device == "cpu"
