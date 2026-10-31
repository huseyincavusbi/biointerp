import json
import sys

import numpy as np
import pandas as pd
import pytest
from _fake_dna_models import FakeCarbon, FakeDnaBert, FakeDnaTokenizer, FakeKmerTokenizer

from biointerp.experiments import dna_vep as exp


def _frame() -> pd.DataFrame:
    rows = []
    for chrom in range(1, 6):
        for i in range(8):
            ref = "ACGTACGTACGT"
            alt = ref[:3] + ("A" if ref[3] != "A" else "C") + ref[4:]
            rows.append(
                {
                    "ref_forward_sequence": ref,
                    "alt_forward_sequence": alt,
                    "label": i % 2,
                    "chromosome": str(chrom),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "model_key,model,tokenizer",
    [
        ("carbon_500m", FakeCarbon(hidden=16, intermediate=24, n_layers=3), FakeKmerTokenizer()),
        ("dna_bert2", FakeDnaBert(hidden=16, intermediate=24, n_layers=3), FakeDnaTokenizer()),
    ],
)
def test_dna_vep_writes_outputs(monkeypatch, tmp_path, model_key, model, tokenizer):
    path = tmp_path / "vep.parquet"
    _frame().to_parquet(path, index=False)
    monkeypatch.setattr(exp, "load_model", lambda *a, **k: (model.eval(), tokenizer, "cpu"))
    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dna_vep",
            "--model-key",
            model_key,
            "--data",
            str(path),
            "--n",
            "40",
            "--batch-size",
            "8",
            "--test-frac",
            "0.2",
        ],
    )

    exp.main()

    out = tmp_path / f"dna_vep_{model_key}.npz"
    assert out.exists()
    data = np.load(out)
    assert data["cett"].shape == (40, 3 * 24)
    assert data["score"].shape == (40,)
    assert data["label"].shape == (40,)

    summary = json.loads((tmp_path / f"dna_vep_{model_key}.summary.json").read_text())
    assert summary["modality"] == "dna"
    assert summary["results"]["n"] == 40
    assert summary["results"]["n_test"] == 8
    assert 0.0 <= summary["results"]["sign_accuracy"] <= 1.0
    assert summary["results"]["auroc_score"] is not None
    assert summary["results"]["auroc_kmer"] is not None

    preds = (tmp_path / f"dna_vep_{model_key}.predictions.jsonl").read_text().splitlines()
    assert len(preds) == 40
    assert set(json.loads(preds[0])) == {
        "chrom",
        "offset",
        "ref",
        "alt",
        "label",
        "score",
        "pred",
        "correct",
        "confidence",
    }
