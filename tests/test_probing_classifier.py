import json
import sys

from _fake_models import FakeEsmModel, FakeEsmTokenizer

from biointerp.experiments import probing_classifier as exp


def test_main_writes_per_layer_metrics(monkeypatch, tmp_path):
    model = FakeEsmModel(hidden=16, intermediate=24, n_layers=4, seed=0).eval()
    tokenizer = FakeEsmTokenizer()
    sequences = ["ACACACACAC"] * 80

    monkeypatch.setattr(exp, "fetch_swissprot", lambda n, seed=0: sequences)
    monkeypatch.setattr(exp, "load_model", lambda *a, **k: (model, tokenizer, "cpu"))
    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--n", "80", "--batch-size", "16", "--test-frac", "0.3"],
    )

    exp.main()

    out = json.loads((tmp_path / "probing_classifier.json").read_text())
    assert out["model"] == "facebook/esm2_t33_650M_UR50D"
    assert out["n"] == 80
    assert len(out["rows"]) == 5
    assert [row["layer"] for row in out["rows"]] == [0, 1, 2, 3, 4]
    assert all(row["auroc"] is None or 0.0 <= row["auroc"] <= 1.0 for row in out["rows"])
    assert 0 <= out["best_layer"] <= 4
    assert 0.0 <= out["best_auroc"] <= 1.0


def test_main_respects_requested_sample_count(monkeypatch, tmp_path):
    model = FakeEsmModel(hidden=8, intermediate=12, n_layers=2, seed=1).eval()
    tokenizer = FakeEsmTokenizer()
    sequences = ["ACACACACAC"] * 40

    monkeypatch.setattr(exp, "fetch_swissprot", lambda n, seed=0: sequences)
    monkeypatch.setattr(exp, "load_model", lambda *a, **k: (model, tokenizer, "cpu"))
    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["probe", "--n", "20", "--batch-size", "5"])

    exp.main()

    out = json.loads((tmp_path / "probing_classifier.json").read_text())
    assert out["n"] == 20
    assert len(out["rows"]) == 3
