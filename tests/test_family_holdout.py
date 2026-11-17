import json

import numpy as np

from biointerp.experiments import family_holdout as exp


def test_main_holds_out_each_family(monkeypatch, tmp_path):
    n, features = 90, 12
    rng = np.random.default_rng(0)
    npz = tmp_path / "features.npz"
    np.savez(
        npz,
        cett=rng.normal(size=(n, features)).astype(np.float16),
        correct=np.arange(n) % 2 == 0,
        protein=np.arange(n),
        confidence=np.full(n, 0.7),
    )
    assignments = tmp_path / "assignments.tsv"
    lines = ["id\tcluster\tsplit"] + [f"seq{i}\tseq{i}\ttrain" for i in range(n)]
    assignments.write_text("\n".join(lines) + "\n")
    families = tmp_path / "families.tsv"
    lines = ["protein_id\tfamily"]
    for i in range(n):
        lines.append(f"seq{i}\t{'ABC'[i // 30]}")
    families.write_text("\n".join(lines) + "\n")
    out = tmp_path / "family.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "family",
            "--npz",
            str(npz),
            "--assignments",
            str(assignments),
            "--families",
            str(families),
            "--min-members",
            "20",
            "--top-k",
            "2",
            "--out",
            str(out),
        ],
    )
    exp.main()

    result = json.loads(out.read_text())
    assert result["n_families"] == 3
    assert {row["family"] for row in result["per_family"]} == {"A", "B", "C"}
    assert all(row["n_test"] == 30 for row in result["per_family"])
    assert all(row["n_train"] == 60 for row in result["per_family"])
    assert "mean_auroc" in result["summary"]
    assert "meta" in result
