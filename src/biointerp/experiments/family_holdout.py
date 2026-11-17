"""Leave-one-family-out generalization for the error-neuron probe.

The family table is a configuration input; this experiment never assumes Pfam
or any specific annotation source. Each eligible family is held out entirely,
the probe is trained on the rest, and the held-out AUROC is reported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, apply_probe, fit_top_probe
from biointerp.families import family_splits, load_family_table
from biointerp.repro import metadata
from biointerp.runner import load_dataset_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--assignments", default=str(RESULTS_DIR / "split_assignments.tsv"))
    p.add_argument("--dataset-config", default="families")
    p.add_argument("--families", default=None, help="override the configured family table")
    p.add_argument("--min-members", type=int, default=None)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(RESULTS_DIR / "family_holdout.json"))
    return p.parse_args()


def load_rows(npz: str, assignments: str):
    data = np.load(npz, allow_pickle=True)
    x = data["cett"]
    y = (~data["correct"]).astype(int)
    protein = data["protein"]
    confidence = data["confidence"].astype(float)

    split_of: dict[str, str] = {}
    for line in Path(assignments).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        record_id, _cluster, split = line.split("\t")
        split_of[record_id] = split

    ids = np.asarray([f"seq{int(index)}" for index in protein])
    keep = np.asarray([split_of.get(record_id) in {"train", "test"} for record_id in ids])
    return x[keep], y[keep], confidence[keep], ids[keep]


def main() -> None:
    args = parse_args()
    started = time.time()
    spec = load_dataset_config(args.dataset_config)
    columns = dict(spec.get("columns", {}))
    family_path = args.families or str(spec["path"])
    min_members = (
        args.min_members if args.min_members is not None else int(spec.get("min_members", 5))
    )

    families = load_family_table(
        family_path,
        protein_column=str(columns.get("protein_id", "protein_id")),
        family_column=str(columns.get("family", "family")),
    )
    x, y, confidence, ids = load_rows(args.npz, args.assignments)

    labelled = np.asarray([str(item) in families for item in ids])
    x = x[labelled]
    y = y[labelled]
    confidence = confidence[labelled]
    ids = ids[labelled]
    if len(ids) < 20:
        raise SystemExit(f"too few labelled rows: {len(ids)}")

    splits = family_splits(ids, families, min_members=min_members)
    per_family: list[dict[str, object]] = []
    for family, train, test in splits:
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        scaler, clf, idx = fit_top_probe(x[train], y[train], top_k=args.top_k, seed=args.seed)
        scores = apply_probe(scaler, clf, idx, x[test])
        per_family.append(
            {
                "family": family,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "auroc_probe": round(float(roc_auc_score(y[test], scores)), 4),
                "auroc_confidence": round(float(roc_auc_score(y[test], -confidence[test])), 4),
            }
        )

    if not per_family:
        raise SystemExit("no family had both classes and enough members")

    values = np.asarray([row["auroc_probe"] for row in per_family], dtype=float)
    result = {
        "meta": metadata(),
        "dataset": spec.get("name", args.dataset_config),
        "source": family_path,
        "n_rows": int(len(ids)),
        "n_families": len(per_family),
        "min_members": min_members,
        "per_family": per_family,
        "summary": {
            "mean_auroc": round(float(values.mean()), 4),
            "median_auroc": round(float(np.median(values)), 4),
            "worst_auroc": round(float(values.min()), 4),
            "best_auroc": round(float(values.max()), 4),
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"rows: {result['n_rows']} | families: {result['n_families']}")
    for row in per_family:
        print(
            f"  {row['family']:20s} n_test {row['n_test']:4d} "
            f"AUROC {row['auroc_probe']:.4f} (confidence {row['auroc_confidence']:.4f})"
        )
    print(f"summary: {result['summary']}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
