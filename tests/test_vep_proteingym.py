import json
import sys

import pandas as pd
from _fake_models import FakeEsmModel, FakeEsmTokenizer

from biointerp.experiments import vep_proteingym as exp

TARGET = "ACACACACAC"


def _frame(n_pairs: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n_pairs):
        pos = 1 + (i % 8)
        wt = TARGET[pos]
        mut = "C" if wt == "A" else "A"
        for bin_value in (0.0, 1.0):
            rows.append(
                {
                    "DMS_id": "TEST",
                    "mutant": f"{wt}{pos}{mut}",
                    "DMS_score": float(i - n_pairs / 2),
                    "DMS_score_bin": bin_value,
                    "target_seq": TARGET,
                }
            )
    return pd.DataFrame(rows)


def test_main_writes_metrics_and_predictions(monkeypatch, tmp_path):
    frame = _frame()
    model = FakeEsmModel(hidden=16, intermediate=24, n_layers=3, seed=0).eval()
    tokenizer = FakeEsmTokenizer()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "unused.parquet")
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    monkeypatch.setattr(exp, "load_model", lambda *a, **k: (model, tokenizer, "cpu"))
    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vep",
            "--assay",
            "TEST",
            "--n-mutations",
            str(len(frame)),
            "--test-frac",
            "0.3",
            "--top-k",
            "10",
            "--batch-size",
            "8",
        ],
    )

    exp.main()

    out = json.loads((tmp_path / "vep_proteingym_TEST.json").read_text())
    assert out["assay"] == "TEST"
    assert out["n_mutations"] == len(frame)
    assert 0.0 <= out["auroc_probe"] <= 1.0
    assert 0.0 <= out["auroc_confidence"] <= 1.0
    assert -1.0 <= out["spearman"] <= 1.0
    assert 0.0 <= out["sign_accuracy"] <= 1.0

    preds = (tmp_path / "vep_proteingym_TEST.predictions.jsonl").read_text().splitlines()
    assert len(preds) == len(frame)
    record = json.loads(preds[0])
    assert set(record) == {"mutant", "score", "dms_score", "correct", "confidence"}


def test_main_skips_assay_with_too_few_substitutions(monkeypatch, tmp_path):
    frame = _frame(3)  # 6 rows, below the minimum of 10
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "unused.parquet")
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["vep", "--assay", "TEST", "--n-mutations", "0"])

    exp.main()

    assert not (tmp_path / "vep_proteingym_TEST.json").exists()
