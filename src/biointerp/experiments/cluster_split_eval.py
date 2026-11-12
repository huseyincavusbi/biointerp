"""Compare the cluster split against the old random protein split.

Loads cached CETT features, selects the top-k error neurons on the training
side of each split, fits the same logistic probe, and reports test AUROC with a
bootstrap interval. This isolates the effect of the split itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, split_by_protein, top_probe_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--assignments", default=str(RESULTS_DIR / "split_assignments.tsv"))
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--out", default=None)
    return p.parse_args()


def bootstrap_auroc(
    y: np.ndarray, scores: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        if y[index].min() == y[index].max():
            continue
        values.append(roc_auc_score(y[index], scores[index]))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    top_k: int,
    seed: int,
    n_boot: int,
) -> dict[str, object]:
    scores, idx = top_probe_scores(x[train], y[train], x[test], top_k=top_k, seed=seed)
    auroc = float(roc_auc_score(y[test], scores))
    low, high = bootstrap_auroc(y[test], scores, n_boot=n_boot, seed=seed)
    return {
        "auroc": round(auroc, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "n_error_train": int(y[train].sum()),
        "n_error_test": int(y[test].sum()),
        "top_features": [int(i) for i in idx],
    }


def main() -> None:
    args = parse_args()
    data = np.load(args.npz, allow_pickle=True)
    x = data["cett"]
    y = (~data["correct"]).astype(int)
    protein = data["protein"]

    split_of: dict[str, str] = {}
    for line in Path(args.assignments).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        record_id, _cluster, split = line.split("\t")
        split_of[record_id] = split

    ids = np.asarray([f"seq{int(p)}" for p in protein])
    keep = np.asarray([split_of.get(record_id) in {"train", "test"} for record_id in ids])
    if not keep.any():
        raise SystemExit("no rows matched the assignments file")

    x_keep = x[keep]
    y_keep = y[keep]
    protein_keep = protein[keep]
    cluster_split = np.asarray([split_of[record_id] == "test" for record_id in ids[keep]])

    random_train, random_test = split_by_protein(
        protein_keep, test_frac=args.test_frac, seed=args.seed
    )

    cluster_train = ~cluster_split
    cluster_result = evaluate(
        x_keep,
        y_keep,
        cluster_train,
        cluster_split,
        args.top_k,
        args.seed,
        args.bootstrap,
    )
    random_result = evaluate(
        x_keep,
        y_keep,
        random_train,
        random_test,
        args.top_k,
        args.seed,
        args.bootstrap,
    )

    result = {
        "npz": str(args.npz),
        "assignments": str(args.assignments),
        "top_k": args.top_k,
        "n_rows": int(keep.sum()),
        "cluster_split": cluster_result,
        "random_split": random_result,
        "delta_auroc": round(cluster_result["auroc"] - random_result["auroc"], 4),
    }

    out = Path(args.out) if args.out else RESULTS_DIR / "cluster_split_eval.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"rows: {result['n_rows']} | top-k: {args.top_k}")
    print(f"random split:  AUROC {random_result['auroc']:.4f} CI {random_result['ci95']}")
    print(f"cluster split: AUROC {cluster_result['auroc']:.4f} CI {cluster_result['ci95']}")
    print(f"delta: {result['delta_auroc']:+.4f}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
