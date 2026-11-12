import json
from argparse import Namespace

from biointerp.experiments import headline_table as table


def _args(tmp_path):
    return Namespace(n_boot=50, seed=0, top_k=25)


def test_render_markdown_marks_missing_ci():
    rows = [
        {"claim": "a", "estimate": 0.7, "ci95": [0.6, 0.8], "n": 10, "family": "core"},
        {"claim": "b", "estimate": 0.1, "ci95": None, "n": 20, "family": "assay"},
    ]
    text = table.render_markdown(rows)
    assert "| a | 0.7 | [0.6, 0.8] | 10 | core |" in text
    assert "| b | 0.1 | pending | 20 | assay |" in text


def test_assay_row_without_predictions_is_summary_only(monkeypatch, tmp_path):
    monkeypatch.setattr(table, "RESULTS_DIR", tmp_path)
    payload = {"n_mutations": 100, "auroc_probe": 0.8, "auroc_confidence": 0.6}
    (tmp_path / "vep_proteingym_test.json").write_text(json.dumps(payload))

    row = table.assay_row("test", _args(tmp_path))

    assert row["status"] == "summary_only"
    assert row["estimate"] == 0.2
    assert row["ci95"] is None


def test_assay_row_bootstraps_paired_delta(monkeypatch, tmp_path):
    monkeypatch.setattr(table, "RESULTS_DIR", tmp_path)
    payload = {"n_mutations": 120, "auroc_probe": 0.8, "auroc_confidence": 0.5}
    (tmp_path / "vep_proteingym_test.json").write_text(json.dumps(payload))

    records = []
    for i in range(120):
        correct = i % 2 == 0
        records.append(
            {
                "mutant": f"A{i}C",
                "score": 0.0,
                "dms_score": 0.0,
                "correct": correct,
                "confidence": 0.5,
                "probe_score": 0.1 if correct else 0.9,
                "split": "test" if i < 60 else "train",
            }
        )
    path = tmp_path / "vep_proteingym_test.predictions.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))

    row = table.assay_row("test", _args(tmp_path))

    assert row["status"] == "ok"
    assert row["n"] == 60
    assert row["estimate"] == 0.5
    assert row["ci95"] is not None
