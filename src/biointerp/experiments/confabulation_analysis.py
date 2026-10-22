"""Confidently-wrong analysis on ESM-2 (CoTLab confabulation_analysis concept adapted).

Categories: high-conf correct / high-conf wrong (confabulation-like) /
low-conf wrong (uncertainty) / low-conf correct. Plus within-confidence AUROC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, ttest_ind
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, split_by_protein, top_probe_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--high", type=float, default=0.5)
    p.add_argument("--very-high", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


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

    results = {"model": str(data["model"]), "top_k": args.top_k, "high_threshold": args.high}

    def report(name, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"{name:26s} n=0")
            return None
        mean_score = float(scores[mask].mean())
        err = float(y_te[mask].mean())
        print(f"{name:26s} n={n:4d} | probe score {mean_score:.4f} | error rate {err:.4f}")
        results[name] = {
            "n": n,
            "mean_probe_score": round(mean_score, 4),
            "error_rate": round(err, 4),
        }
        return mean_score

    print(f"test samples: {len(y_te)} | overall error rate {y_te.mean():.4f}\n")
    hc = y_te == 0
    hw = y_te == 1
    high = conf_te >= args.high
    report("high-conf correct", high & hc)
    report("high-conf wrong", high & hw)
    report("low-conf correct", (~high) & hc)
    report("low-conf wrong", (~high) & hw)

    print()
    for label, mask in [
        (f"conf >= {args.high}", high),
        (f"conf >= {args.very_high}", conf_te >= args.very_high),
        (f"conf < {args.high}", ~high),
    ]:
        if mask.sum() > 10 and len(np.unique(y_te[mask])) == 2:
            a = roc_auc_score(y_te[mask], scores[mask])
            print(f"within {label:14s} (n={mask.sum():3d}): probe AUROC {a:.4f}")
            results[f"auroc_within_{label.replace(' ', '_')}"] = round(float(a), 4)
        else:
            print(f"within {label:14s}: n={mask.sum()} (skipped)")

    hw_scores = scores[high & hw]
    lw_scores = scores[(~high) & hw]
    if len(hw_scores) > 2 and len(lw_scores) > 2:
        t, p = ttest_ind(hw_scores, lw_scores, equal_var=False)
        d = cohens_d(hw_scores, lw_scores)
        print(f"\nhigh-conf-wrong vs low-conf-wrong: t={t:.3f} p={p:.4f} Cohen's d={d:.3f}")
        results["ttest_high_wrong_vs_low_wrong"] = {
            "t": round(float(t), 3),
            "p": round(float(p), 4),
            "cohens_d": round(d, 3),
        }

    rho, _ = spearmanr(scores, conf_te)
    print(f"Spearman(probe score, confidence): {rho:.4f}")
    results["spearman_probe_conf"] = round(float(rho), 4)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "confabulation_analysis.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
