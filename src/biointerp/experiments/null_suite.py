"""Permutation and shuffled-label null suite for the error-neuron claims.

Three nulls are reported:

- N1: shuffle error labels on the training side and rerun the top-k probe.
- N2: shuffle CETT feature rows relative to labels and rerun the same probe.
- N3: permute pLDDT values within each protein and recompute the pooled
  Spearman correlation between pLDDT and error.

The observed statistic is computed once; every null uses the same pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, top_probe_scores
from biointerp.nulls import (
    null_summary,
    permute_labels,
    permute_rows,
    plddt_pairs,
    pooled_spearman,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--assignments", default=str(RESULTS_DIR / "split_assignments.tsv"))
    p.add_argument("--structure", default=str(RESULTS_DIR / "structure_mapping.json"))
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--n-label", type=int, default=200)
    p.add_argument("--n-feature", type=int, default=50)
    p.add_argument("--n-plddt", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--figure", default=str(RESULTS_DIR / "null_suite.png"))
    p.add_argument("--quick", action="store_true", help="50/20/200 permutations")
    return p.parse_args()


def evaluate_auroc(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    top_k: int,
    seed: int,
) -> float:
    scores, _ = top_probe_scores(x[train], y[train], x[test], top_k=top_k, seed=seed)
    return float(roc_auc_score(y[test], scores))


def load_rows(npz: str, assignments: str):
    data = np.load(npz, allow_pickle=True)
    x = data["cett"]
    y = (~data["correct"]).astype(int)
    protein = data["protein"]

    split_of: dict[str, str] = {}
    for line in Path(assignments).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        record_id, _cluster, split = line.split("\t")
        split_of[record_id] = split

    ids = np.asarray([f"seq{int(p)}" for p in protein])
    keep = np.asarray([split_of.get(record_id) in {"train", "test"} for record_id in ids])
    if not keep.any():
        raise SystemExit("no rows matched the assignments file")

    is_test = np.asarray([split_of[record_id] == "test" for record_id in ids[keep]])
    return x[keep], y[keep], ~is_test, is_test


def plddt_permutation(records, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    observed = pooled_spearman(plddt_pairs(records))
    null: list[float] = []
    for _ in range(n):
        pairs: list[tuple[float, float]] = []
        for record in records:
            plddt = np.asarray([r["plddt"] for r in record["per_residue"]], dtype=float)
            error = np.asarray([1.0 - r["p_true"] for r in record["per_residue"]], dtype=float)
            shuffled = plddt[rng.permutation(len(plddt))]
            pairs.extend(zip(shuffled, error))
        null.append(pooled_spearman(pairs))
    return observed, null


def plot_nulls(
    path: str,
    label_null: list[float],
    feature_null: list[float],
    observed: float,
    plddt_null: list[float],
    plddt_observed: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(label_null, bins=25, alpha=0.7, label="N1 label shuffle")
    if feature_null:
        axes[0].hist(feature_null, bins=25, alpha=0.5, label="N2 feature shuffle")
    axes[0].axvline(observed, color="crimson", label=f"observed {observed:.3f}")
    axes[0].set_xlabel("test AUROC")
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(plddt_null, bins=35, alpha=0.7, color="steelblue")
    axes[1].axvline(plddt_observed, color="crimson", label=f"observed {plddt_observed:.3f}")
    axes[1].set_xlabel("Spearman(pLDDT, error)")
    axes[1].set_ylabel("count")
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_label, args.n_feature, args.n_plddt = 50, 20, 200

    started = time.time()
    x, y, train, test = load_rows(args.npz, args.assignments)
    print(f"rows: {len(y)} | train {train.sum()} / test {test.sum()}")

    observed = evaluate_auroc(x, y, train, test, args.top_k, args.seed)
    print(f"observed AUROC: {observed:.4f}")

    label_null: list[float] = []
    for i in range(args.n_label):
        y_perm = permute_labels(y[train], seed=args.seed + i)
        scores, _ = top_probe_scores(
            x[train], y_perm, x[test], top_k=args.top_k, seed=args.seed + i
        )
        label_null.append(float(roc_auc_score(y[test], scores)))
        if (i + 1) % 10 == 0 or i + 1 == args.n_label:
            print(f"  N1 {i + 1}/{args.n_label} | {time.time() - started:.0f}s")

    feature_null: list[float] = []
    for i in range(args.n_feature):
        x_perm = permute_rows(x[train], seed=args.seed + i)
        scores, _ = top_probe_scores(
            x_perm, y[train], x[test], top_k=args.top_k, seed=args.seed + i
        )
        feature_null.append(float(roc_auc_score(y[test], scores)))

    structure = json.loads(Path(args.structure).read_text())
    plddt_observed, plddt_null = plddt_permutation(structure["records"], args.n_plddt, args.seed)
    print(f"pLDDT observed: {plddt_observed:.4f} | null mean {np.mean(plddt_null):.4f}")

    result = {
        "npz": str(args.npz),
        "assignments": str(args.assignments),
        "structure": str(args.structure),
        "top_k": args.top_k,
        "rows": {
            "n": int(len(y)),
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
        },
        "label_shuffle": null_summary(observed, label_null, alternative="greater"),
        "feature_shuffle": null_summary(observed, feature_null, alternative="greater"),
        "plddt_permutation": null_summary(plddt_observed, plddt_null, alternative="less"),
        "elapsed_s": round(time.time() - started, 1),
    }

    out = Path(args.out) if args.out else RESULTS_DIR / "null_suite.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    plot_nulls(args.figure, label_null, feature_null, observed, plddt_null, plddt_observed)

    print(
        f"N1 p={result['label_shuffle']['p_value']:.4f} "
        f"(null p95 {result['label_shuffle']['null_p95']:.4f})"
    )
    print(
        f"N2 p={result['feature_shuffle']['p_value']:.4f} "
        f"(null p95 {result['feature_shuffle']['null_p95']:.4f})"
    )
    print(
        f"N3 p={result['plddt_permutation']['p_value']:.4f} "
        f"(null p95 {result['plddt_permutation']['null_p95']:.4f})"
    )
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
