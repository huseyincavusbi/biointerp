"""Probe-confidence formalization on ESM-2 (CoTLab probe_confidence adapted).

Probe score vs output confidence: Spearman, partial Spearman (controlling length),
AUROC comparison, bootstrap CI, verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import (
    DEFAULT_FASTA,
    DEFAULT_NPZ,
    RESULTS_DIR,
    load_sequences,
    split_by_protein,
    top_probe_scores,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--fasta", default=None)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def partial_spearman(a, b, c):
    from numpy import argsort

    def rank(v):
        return argsort(argsort(v)) + 1.0

    ra, rb, rc = rank(a), rank(b), rank(c)
    res_a = ra - np.polyval(np.polyfit(rc, ra, 1), rc)
    res_b = rb - np.polyval(np.polyfit(rc, rb, 1), rc)
    rho, _ = spearmanr(res_a, res_b)
    return float(rho)


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)
    x = data["cett"]
    y = (~data["correct"]).astype(int)
    conf = data["confidence"]
    ent = data["entropy"]
    protein = data["protein"]

    tr, te = split_by_protein(protein, seed=args.seed)
    scores, idx = top_probe_scores(x[tr], y[tr], x[te], args.top_k, args.seed)
    y_te = y[te]
    conf_te = conf[te]

    fasta = args.fasta or str(DEFAULT_FASTA)
    try:
        seqs = load_sequences(fasta)
        length = np.array([len(seqs[p]) for p in protein], dtype=float)
    except Exception:
        length = None

    auroc_probe = float(roc_auc_score(y_te, scores))
    auroc_conf = float(roc_auc_score(y_te, -conf_te))
    rho_conf, _ = spearmanr(scores, conf_te)
    rho_ent, _ = spearmanr(scores, ent[te])

    rng = np.random.default_rng(args.seed)
    boot = []
    n = len(y_te)
    for _ in range(args.boot):
        take = rng.integers(0, n, n)
        if len(np.unique(y_te[take])) < 2:
            continue
        boot.append(roc_auc_score(y_te[take], scores[take]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    print(f"model: {str(data['model'])}")
    print(f"test samples: {n} | top-k: {args.top_k} | probe features: {len(idx)}")
    print(f"AUROC probe:      {auroc_probe:.4f} (95% CI {ci[0]:.4f}-{ci[1]:.4f})")
    print(f"AUROC confidence: {auroc_conf:.4f}")
    print(f"Spearman(probe score, confidence): {rho_conf:.4f}")
    print(f"Spearman(probe score, entropy):    {rho_ent:.4f}")
    results = {
        "model": str(data["model"]),
        "test_n": int(n),
        "top_k": args.top_k,
        "auroc_probe": round(auroc_probe, 4),
        "auroc_probe_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "auroc_confidence": round(auroc_conf, 4),
        "spearman_probe_conf": round(float(rho_conf), 4),
        "spearman_probe_entropy": round(float(rho_ent), 4),
    }
    if length is not None:
        ps = partial_spearman(scores, conf_te, length[te])
        rho_len, _ = spearmanr(length[te], y_te)
        print(f"partial Spearman(probe, conf | length): {ps:.4f}")
        print(f"Spearman(length, incorrect): {rho_len:.4f}")
        results["partial_spearman_probe_conf_length"] = round(ps, 4)
        results["spearman_length_incorrect"] = round(float(rho_len), 4)

    verdict = (
        "probe adds nothing beyond confidence"
        if auroc_probe < auroc_conf
        else "probe beats confidence"
    )
    print(f"\nverdict: {verdict}")
    results["verdict"] = verdict

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "probe_confidence.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
