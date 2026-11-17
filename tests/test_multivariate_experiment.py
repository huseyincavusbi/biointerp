import json

import pandas as pd

from biointerp.experiments import multivariate_model as exp


def test_structure_table_flattens_residues(tmp_path):
    payload = {
        "records": [
            {
                "seq": 0,
                "length": 3,
                "per_residue": [
                    {"plddt": 0.9, "p_true": 0.8},
                    {"plddt": 0.5, "p_true": 0.4},
                    {"plddt": 0.2, "p_true": 0.1},
                ],
            }
        ]
    }
    path = tmp_path / "structure.json"
    path.write_text(json.dumps(payload))
    frame = exp.structure_table(str(path))
    assert len(frame) == 3
    assert {"protein_id", "length", "relative_position", "error", "plddt"} <= set(frame.columns)
    assert frame.loc[2, "relative_position"] == 1.0


def test_read_table_maps_configured_columns(tmp_path):
    path = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "prot": ["a", "a"],
            "err": [0, 1],
            "plddt_score": [0.9, 0.2],
        }
    ).to_csv(path, index=False)
    frame = exp.read_table(
        str(path),
        {"protein_id": "prot", "error": "err", "plddt": "plddt_score"},
    )
    assert {"protein_id", "error", "plddt"} <= set(frame.columns)


def test_main_from_structure_mapping(monkeypatch, tmp_path):
    records = []
    for protein in range(4):
        residues = [
            {"plddt": (position % 10) / 10, "p_true": 0.9 if position % 2 else 0.2}
            for position in range(30)
        ]
        records.append({"seq": protein, "length": 30, "per_residue": residues})
    structure = tmp_path / "structure.json"
    structure.write_text(json.dumps({"records": records}))
    out = tmp_path / "model.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "multivariate",
            "--from-structure-mapping",
            "--structure",
            str(structure),
            "--predictors",
            "plddt,length",
            "--test-predictor",
            "plddt",
            "--out",
            str(out),
        ],
    )
    exp.main()

    result = json.loads(out.read_text())
    assert result["n"] == 120
    assert result["n_proteins"] == 4
    assert result["test_predictor"] == "plddt"
    assert "lr_test" in result
    assert "meta" in result
