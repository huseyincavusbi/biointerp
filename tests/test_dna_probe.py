import json
import sys

import numpy as np
from _fake_dna_models import FakeDnaBert, FakeDnaTokenizer

from biointerp.experiments import dna_probe as exp


def test_dna_probe_writes_outputs(monkeypatch, tmp_path):
    model = FakeDnaBert(hidden=16, intermediate=24, n_layers=3, seed=0).eval()
    tokenizer = FakeDnaTokenizer()
    monkeypatch.setattr(exp, "load_model", lambda *a, **k: (model, tokenizer, "cpu"))
    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dna_probe",
            "--model-key",
            "dna_bert2",
            "--n",
            "60",
            "--length",
            "12",
            "--batch-size",
            "10",
            "--test-frac",
            "0.5",
        ],
    )

    exp.main()

    out = tmp_path / "dna_probe_dna_bert2.npz"
    assert out.exists()
    data = np.load(out)
    assert data["cett"].shape == (60, 3 * 24)
    assert data["correct"].shape == (60,)
    assert data["true_id"].shape == (60,)

    summary = json.loads((tmp_path / "dna_probe_dna_bert2.summary.json").read_text())
    assert summary["modality"] == "dna"
    assert summary["tokenizer"] == "bpe"
    assert summary["results"]["n"] == 60
    assert 0.0 <= summary["results"]["token_accuracy"] <= 1.0
    assert len(summary["rows"]) == 3
    assert [row["layer"] for row in summary["rows"]] == [0, 1, 2]

    preds = (tmp_path / "dna_probe_dna_bert2.predictions.jsonl").read_text().splitlines()
    assert len(preds) == 60
    assert set(json.loads(preds[0])) == {
        "sequence",
        "position",
        "true_token",
        "pred_token",
        "correct",
        "confidence",
        "entropy",
    }
