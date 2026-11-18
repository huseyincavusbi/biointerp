import json

import numpy as np
from _fake_models import FakeEsmModel, FakeEsmTokenizer

from biointerp.experiments import ablation_controls as exp


def test_main_runs_small_condition_matrix(monkeypatch, tmp_path):
    n, features = 40, 24
    rng = np.random.default_rng(0)
    npz = tmp_path / "features.npz"
    np.savez(
        npz,
        cett=rng.normal(size=(n, features)).astype(np.float16),
        correct=np.arange(n) % 2 == 0,
        protein=np.arange(n),
    )
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text("".join(f">seq{i}\nACDEFGHIKLMNPQRSTVWY\n" for i in range(n)))
    assignments = tmp_path / "assignments.tsv"
    lines = ["id\tcluster\tsplit"]
    for i in range(n):
        lines.append(f"seq{i}\tseq{i}\t{'train' if i < 30 else 'test'}")
    assignments.write_text("\n".join(lines) + "\n")
    out = tmp_path / "ablation.json"

    model = FakeEsmModel(hidden=8, intermediate=12, n_layers=2).eval()
    tokenizer = FakeEsmTokenizer()
    monkeypatch.setattr(exp, "load_model", lambda *a, **k: (model, tokenizer, "cpu"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "ablation",
            "--npz",
            str(npz),
            "--fasta",
            str(fasta),
            "--assignments",
            str(assignments),
            "--calib-n",
            "8",
            "--eval-n",
            "8",
            "--k",
            "3",
            "--draws",
            "2",
            "--batch-size",
            "4",
            "--out",
            str(out),
            "--figure",
            str(tmp_path / "ablation.png"),
        ],
    )

    exp.main()

    result = json.loads(out.read_text())
    assert result["k"] == 3
    assert len(result["conditions"]) == 5
    assert "accuracy_top_drop" in result["summary"]
    assert "accuracy_random_p95_drop" in result["summary"]
    assert "metrics_supported" in result["summary"]
    assert (tmp_path / "ablation.json").exists()
    assert "meta" in result
