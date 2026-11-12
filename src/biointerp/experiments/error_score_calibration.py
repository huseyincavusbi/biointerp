"""Calibration and triage analysis for the error-neuron score.

Fits the top-k CETT probe on the cluster-split training set, scores held-out
positions, and compares the resulting 0-1 error score with the model's own
confidence. Reports ECE, Brier, precision@10%, enrichment, and a reliability
diagram with cluster-bootstrap intervals over proteins.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biointerp.calibration import (
    calibration_summary,
    percentile_ci,
    reliability_curve,
)
from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, top_probe_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--assignments", default=str(RESULTS_DIR / "split_assignments.tsv"))
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--figure", default=str(RESULTS_DIR / "error_score_calibration.png"))
    return p.parse_args()


def load_test_rows(npz: str, assignments: str):
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

    ids = np.asarray([f"seq{int(p)}" for p in protein])
    keep = np.asarray([split_of.get(record_id) in {"train", "test"} for record_id in ids])
    is_test = np.asarray([split_of[record_id] == "test" for record_id in ids[keep]])

    x_keep = x[keep]
    y_keep = y[keep]
    protein_keep = protein[keep]
    confidence_keep = confidence[keep]
    train = ~is_test
    return (
        x_keep,
        y_keep,
        protein_keep,
        confidence_keep,
        train,
        is_test,
    )


def bootstrap_summary(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    n_boot: int,
    seed: int,
) -> dict[str, list[float]]:
    unique = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([index_by_group[group] for group in chosen])
        if len(np.unique(y[index])) < 2:
            continue
        summary = calibration_summary(y[index], scores[index], fraction=fraction)
        for key, value in summary.items():
            if key == "fraction":
                continue
            collected.setdefault(key, []).append(value)
    return {key: percentile_ci(values) for key, values in collected.items()}


def reliability_payload(y: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    return reliability_curve(y, scores, n_bins=10)


def plot_reliability(
    path: str,
    y: np.ndarray,
    error_score: np.ndarray,
    confidence: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    for scores, label, color in (
        (error_score, "error score", "crimson"),
        (confidence, "confidence baseline", "steelblue"),
    ):
        curve = reliability_curve(y, scores, n_bins=10)
        axis.plot(
            [item["mean_score"] for item in curve],
            [item["observed_rate"] for item in curve],
            marker="o",
            label=label,
            color=color,
        )
    axis.set_xlabel("predicted error probability")
    axis.set_ylabel("observed error rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()

    x, y, protein, confidence, train, test = load_test_rows(args.npz, args.assignments)
    scores, _ = top_probe_scores(x[train], y[train], x[test], top_k=args.top_k, seed=args.seed)
    y_test = y[test]
    groups = protein[test]
    error_score = scores
    confidence_error = 1.0 - confidence[test]

    error_summary = calibration_summary(y_test, error_score, fraction=args.fraction)
    confidence_summary = calibration_summary(y_test, confidence_error, fraction=args.fraction)

    error_ci = bootstrap_summary(y_test, error_score, groups, args.fraction, args.n_boot, args.seed)
    confidence_ci = bootstrap_summary(
        y_test, confidence_error, groups, args.fraction, args.n_boot, args.seed
    )

    result = {
        "npz": str(args.npz),
        "assignments": str(args.assignments),
        "n_test": int(test.sum()),
        "top_k": args.top_k,
        "error_score": {**error_summary, "ci95": error_ci},
        "confidence_baseline": {**confidence_summary, "ci95": confidence_ci},
        "delta": {
            "ece": round(error_summary["ece"] - confidence_summary["ece"], 4),
            "brier": round(error_summary["brier"] - confidence_summary["brier"], 4),
            "precision_at_fraction": round(
                error_summary["precision_at_fraction"]
                - confidence_summary["precision_at_fraction"],
                4,
            ),
        },
        "reliability": {
            "error_score": reliability_payload(y_test, error_score),
            "confidence": reliability_payload(y_test, confidence_error),
        },
        "elapsed_s": round(time.time() - started, 1),
    }

    out = Path(args.out) if args.out else RESULTS_DIR / "error_score_calibration.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    plot_reliability(args.figure, y_test, error_score, confidence_error)

    print(f"test rows: {result['n_test']} | base error rate {error_summary['base_rate']:.3f}")
    print(
        f"error score: ECE {error_summary['ece']:.3f} | Brier {error_summary['brier']:.3f} "
        f"| P@10 {error_summary['precision_at_fraction']:.3f} "
        f"| enrichment {error_summary['enrichment']:.2f}"
    )
    print(
        f"confidence:  ECE {confidence_summary['ece']:.3f} | Brier {confidence_summary['brier']:.3f} "
        f"| P@10 {confidence_summary['precision_at_fraction']:.3f} "
        f"| enrichment {confidence_summary['enrichment']:.2f}"
    )
    print(f"delta: {result['delta']}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
