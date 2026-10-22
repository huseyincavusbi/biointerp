"""Per-protein reliability / triage on ESM-2 (PLM-native).

Risk-coverage analysis: rank test samples by probe score (trust = low score); report the
error rate of the trusted set at 50/70/90% coverage vs the baseline error rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, split_by_protein, top_probe_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--coverages", default="0.5,0.7,0.9")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)
    x = data["cett"]
    y = (~data["correct"]).astype(int)
    conf = data["confidence"]
    protein = data["protein"]

    tr, te = split_by_protein(protein, seed=args.seed)
    scores, _ = top_probe_scores(x[tr], y[tr], x[te], args.top_k, args.seed)
    y_te = y[te]
    conf_te = conf[te]
    n = len(y_te)
    base_err = float(y_te.mean())

    order = np.argsort(scores)  # low score = trusted first
    print(f"test samples: {n} | baseline error rate {base_err:.4f}\n")
    print(
        f"{'coverage':>9s} {'n trusted':>10s} {'error in trusted':>16s} {'errors missed':>13s} {'error in dropped':>16s}"
    )
    rows = []
    for cov in [float(c) for c in args.coverages.split(",")]:
        k = int(round(n * cov))
        trusted = order[:k]
        dropped = order[k:]
        err_t = float(y_te[trusted].mean())
        err_d = float(y_te[dropped].mean()) if len(dropped) else float("nan")
        missed = int(y_te[trusted].sum())
        rows.append(
            {
                "coverage": cov,
                "n_trusted": k,
                "error_trusted": round(err_t, 4),
                "errors_missed": missed,
                "error_dropped": round(err_d, 4),
            }
        )
        print(f"{cov:9.2f} {k:10d} {err_t:16.4f} {missed:13d} {err_d:16.4f}")

    order_conf = np.argsort(-conf_te)
    print("\nconfidence-based triage (baseline comparison):")
    for cov in [float(c) for c in args.coverages.split(",")]:
        k = int(round(n * cov))
        trusted = order_conf[:k]
        print(f"  coverage {cov:.2f}: error in trusted {float(y_te[trusted].mean()):.4f}")

    out = RESULTS_DIR / "per_protein_reliability.json"
    out.write_text(
        json.dumps(
            {
                "model": str(data["model"]),
                "n_test": n,
                "baseline_error": round(base_err, 4),
                "top_k": args.top_k,
                "triage": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
