"""Real DNABERT-2 smoke test (marked ``real_model``; runs on Linux CI)."""

import pytest
import torch

from biointerp.dna import load_dna_model, precompute_col_norms
from biointerp.dna_task import tokenize_dna

pytestmark = pytest.mark.real_model


def test_dnabert2_masked_forward():
    spec = {"adapter": "dnabert", "hf_id": "zhihan1996/DNABERT-2-117M"}
    model, tokenizer, device = load_dna_model(spec, device="cpu")
    ids, _ = tokenize_dna("ACGT" * 8, tokenizer, "dnabert")
    inputs = tokenizer("ACGT" * 8, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    assert out.logits.shape[0] == 1
    assert out.logits.shape[-1] > 4
    assert len(ids) == len(inputs["input_ids"][0])
    assert precompute_col_norms(model, "dnabert")[0].ndim == 1
    assert device == "cpu"
