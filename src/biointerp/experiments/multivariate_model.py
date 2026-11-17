"""Conservation-controlled multivariate error model.

The experiment reads a position table whose path and column names come from
``conf/dataset/positions.yaml``. Predictors are standardized, standard errors
are clustered by protein, and the tested term is compared with a nested model
by likelihood-ratio test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from biointerp.core import RESULTS_DIR
from biointerp.multivariate import (
    fit_cluster_logit,
    likelihood_ratio_test,
    variance_inflation_factors,
)
from biointerp.repro import metadata
from biointerp.runner import load_dataset_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-config", default="positions")
    p.add_argument("--table", default=None, help="override the configured table path")
    p.add_argument(
        "--from-structure-mapping",
        action="store_true",
        help="build a reduced table from results/structure_mapping.json",
    )
    p.add_argument("--structure", default=str(RESULTS_DIR / "structure_mapping.json"))
    p.add_argument("--predictors", default=None, help="comma-separated predictor list")
    p.add_argument("--test-predictor", default=None, help="term removed in the reduced model")
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--out", default=str(RESULTS_DIR / "multivariate_model.json"))
    return p.parse_args()


def structure_table(path: str) -> pd.DataFrame:
    """Build a reduced position table from cached structure mapping records."""
    payload = json.loads(Path(path).read_text())
    rows = []
    for record in payload["records"]:
        length = int(record["length"])
        for position, residue in enumerate(record["per_residue"]):
            rows.append(
                {
                    "protein_id": record["seq"],
                    "position": position,
                    "length": length,
                    "relative_position": position / (length - 1) if length > 1 else 0.0,
                    "error": 1.0 if float(residue["p_true"]) < 0.5 else 0.0,
                    "plddt": float(residue["plddt"]),
                }
            )
    return pd.DataFrame(rows)


def read_table(path: str, columns: dict[str, str]) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise SystemExit(f"position table not found: {table_path}")
    if table_path.suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(table_path)
    elif table_path.suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(table_path, sep="\t" if table_path.suffix == ".tsv" else ",")
    else:
        raise SystemExit(f"unsupported table format: {table_path.suffix}")

    rename = {
        actual: canonical
        for canonical, actual in columns.items()
        if actual in frame.columns and actual != canonical
    }
    return frame.rename(columns=rename)


def main() -> None:
    args = parse_args()
    started = time.time()
    spec = load_dataset_config(args.dataset_config)
    columns = dict(spec.get("columns", {}))

    if args.from_structure_mapping:
        frame = structure_table(args.structure)
        predictors = (
            [item.strip() for item in args.predictors.split(",")]
            if args.predictors
            else ["plddt", "length", "relative_position"]
        )
        test_predictor = args.test_predictor or "plddt"
    else:
        path = args.table or str(spec["path"])
        frame = read_table(path, columns)
        predictors = (
            [item.strip() for item in args.predictors.split(",")]
            if args.predictors
            else list(spec.get("predictors", []))
        )
        test_predictor = args.test_predictor or str(spec.get("test_predictor", "idr"))

    if not predictors:
        raise SystemExit("no predictors configured")
    if test_predictor not in predictors:
        raise SystemExit(f"test predictor {test_predictor!r} not in {predictors}")

    required = ["protein_id", "error", *predictors]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"position table is missing columns: {missing}")

    clean = frame.dropna(subset=required).reset_index(drop=True)
    if len(clean) < 50:
        raise SystemExit(f"too few complete rows: {len(clean)}")
    dropped = len(frame) - len(clean)

    outcome = clean["error"].to_numpy(dtype=float)
    if not set(np.unique(outcome)).issubset({0.0, 1.0}):
        raise SystemExit("error column must be binary (0/1)")

    x = clean[predictors].to_numpy(dtype=float)
    y = outcome.astype(int)
    groups = clean["protein_id"].to_numpy()

    full = fit_cluster_logit(x, y, groups, predictors, max_iter=args.max_iter)
    kept = [index for index, name in enumerate(predictors) if name != test_predictor]
    reduced = fit_cluster_logit(
        x[:, kept],
        y,
        groups,
        [predictors[index] for index in kept],
        max_iter=args.max_iter,
    )
    vif = variance_inflation_factors(x, predictors)
    lr_test = likelihood_ratio_test(full, reduced)

    result = {
        "meta": metadata(),
        "dataset": spec.get("name", args.dataset_config),
        "source": args.structure if args.from_structure_mapping else args.table,
        "outcome": "error",
        "predictors": predictors,
        "test_predictor": test_predictor,
        "group": "protein_id",
        "n": int(len(clean)),
        "n_proteins": int(len(np.unique(groups))),
        "dropped_rows": int(dropped),
        "full": full,
        "reduced": reduced,
        "vif": vif,
        "lr_test": lr_test,
        "elapsed_s": round(time.time() - started, 1),
    }

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"rows: {result['n']} | proteins: {result['n_proteins']} | dropped: {dropped}")
    print(f"full model log-likelihood: {full['log_likelihood']}")
    for name in predictors:
        feature = full["features"][name]
        print(
            f"  {name:20s} OR {feature['odds_ratio']:.3f} "
            f"CI {feature['ci95']} p={feature['p_value']:.3g} VIF {vif[name]:.2f}"
        )
    print(
        f"LR test for {test_predictor}: chi2 {lr_test['statistic']:.3f} "
        f"df {lr_test['df']} p={lr_test['p_value']:.3g}"
    )
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
