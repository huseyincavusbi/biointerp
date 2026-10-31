import json
import sys

import numpy as np
from _fake_dna_models import FakeCarbon, FakeKmerTokenizer

from biointerp.experiments import dna_cett as exp


def test_dna_cett_writes_outputs(monkeypatch, tmp_path):
    model = FakeCarbon(hidden=16, intermediate=24, n_layers=3, seed=0).eval()
    tokenizer = FakeKmerTokenizer()
    monkeypatch.setattr(exp, "load_model", lambda *a, **k: (model, tokenizer, "cpu"))
    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dna_cett",
            "--model-key",
            "carbon_500m",
            "--n",
            "4",
            "--length",
            "12",
            "--batch-size",
            "2",
        ],
    )

    exp.main()

    out = tmp_path / "dna_cett_carbon_500m.npz"
    assert out.exists()
    data = np.load(out)
    assert data["cett"].shape == (4, 3 * 24)
    assert data["correct"].shape == (4,)
    assert data["true_id"].shape == (4,)
    assert data["model"].item() == "carbon_500m"

    summary = json.loads((tmp_path / "dna_cett_carbon_500m.summary.json").read_text())
    assert summary["modality"] == "dna"
    assert summary["tokenizer"] == "6mer"
    assert summary["results"]["n"] == 4
    assert 0.0 <= summary["results"]["next_token_accuracy"] <= 1.0
    assert summary["features"]["total"] == 3 * 24

    preds = (tmp_path / "dna_cett_carbon_500m.predictions.jsonl").read_text().splitlines()
    assert len(preds) == 4
    assert set(json.loads(preds[0])) == {
        "sequence",
        "position",
        "true_token",
        "pred_token",
        "correct",
        "confidence",
        "entropy",
    }
