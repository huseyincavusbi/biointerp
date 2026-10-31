import json

import numpy as np
import pandas as pd

from biointerp.experiments import carbonx_report


def test_build_report_writes_tables(tmp_path):
    run_dir = tmp_path / "run"
    features = run_dir / "features"
    features.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"config": {"layers": [0, 1]}}))
    rows = []
    for variant in (0, 1):
        path = features / f"{variant}_dna_only.npz"
        np.savez_compressed(
            path,
            cett=np.arange(8, dtype=float),
            attention_dna=np.ones((2, 1, 3), dtype=float),
        )
        for condition, label, probability in (
            ("dna_only", "benign", 0.2),
            ("correct_gene", "pathogenic", 0.8),
        ):
            rows.append(
                {
                    "variant": variant,
                    "chrom": "1",
                    "label": 1,
                    "condition": condition,
                    "parsed": {"label": label},
                    "p_pathogenic": probability,
                    "features": str(path),
                }
            )
    (run_dir / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = carbonx_report.build_report(run_dir, run_dir / "report")

    assert summary["n_records"] == 4
    assert summary["n_variants"] == 2
    assert summary["n_flipped"] == 2
    for name in ("answers", "probabilities", "flips", "cett_norms", "attention"):
        assert (run_dir / "report" / f"{name}.csv").exists()
    answers = pd.read_csv(run_dir / "report" / "answers.csv")
    assert set(answers["condition"]) == {"dna_only", "correct_gene"}
    cett = pd.read_csv(run_dir / "report" / "cett_norms.csv")
    assert set(cett["layer"]) == {0, 1}
