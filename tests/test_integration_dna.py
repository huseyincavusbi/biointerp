"""Real DNABERT-2 smoke tests (marked ``real_model``; run on Linux CI)."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from biointerp.dna import extract_cett_masked, load_dna_model, precompute_col_norms
from biointerp.dna_task import masked_dna_batch, tokenize_dna
from biointerp.experiments import dna_vep

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


def test_dnabert2_vep_smoke(monkeypatch, tmp_path):
    rows = []
    for chrom in (1, 2):
        for i in range(2):
            ref = "ACGTACGTACGT"
            alt = ref[:3] + ("A" if ref[3] != "A" else "C") + ref[4:]
            rows.append(
                {
                    "ref_forward_sequence": ref,
                    "alt_forward_sequence": alt,
                    "label": i,
                    "chromosome": str(chrom),
                }
            )
    path = tmp_path / "vep.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = tmp_path / "vep.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dna_vep",
            "--model-key",
            "dna_bert2",
            "--data",
            str(path),
            "--n",
            "4",
            "--batch-size",
            "2",
            "--test-frac",
            "0.5",
            "--controls",
            "none",
            "--out",
            str(out),
        ],
    )

    dna_vep.main()

    data = np.load(out)
    assert data["cett"].shape[0] == 4
    assert data["score"].shape == (4,)
    assert np.isfinite(data["score"]).all()
    summary = json.loads((tmp_path / "vep.summary.json").read_text())
    assert summary["model"] == "dna_bert2"
    assert summary["results"]["n"] == 4
