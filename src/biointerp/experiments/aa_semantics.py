"""AA semantics of error neurons on ESM-2 (PLM-native).

For the top error neurons: what do they respond to? Correlation of activation with
confidence/entropy; AA profiles of samples where they are most active.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, split_by_protein, univariate_auroc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)
    x = data["cett"]
    y = (~data["correct"]).astype(int)
    conf = data["confidence"]
    ent = data["entropy"]
    true_aa = data["true_aa"]
    protein = data["protein"]
    inter = int(data["intermediate_dim"])

    tr, te = split_by_protein(protein, seed=args.seed)
    au_tr = univariate_auroc(x[tr], y[tr])
    order = np.argsort(au_tr)[::-1][: args.top_k]

    print(f"top-{args.top_k} error neurons (train AUROC):")
    rows = []
    for rank, j in enumerate(order):
        layer = int(j) // inter
        neuron = int(j) % inter
        act = x[:, j]
        rho_conf, _ = spearmanr(act, conf)
        rho_ent, _ = spearmanr(act, ent)
        hi = act >= np.quantile(act, 0.9)
        lo = act <= np.quantile(act, 0.1)
        conf_hi, conf_lo = float(conf[hi].mean()), float(conf[lo].mean())
        err_hi, err_lo = float(y[hi].mean()), float(y[lo].mean())
        vals, cnts = np.unique(true_aa[hi], return_counts=True)
        top_aa = vals[np.argsort(cnts)[::-1][:3]].tolist()
        row = {
            "rank": rank + 1,
            "layer": layer,
            "neuron": neuron,
            "train_auroc": round(float(au_tr[j]), 4),
            "spearman_confidence": round(float(rho_conf), 4),
            "spearman_entropy": round(float(rho_ent), 4),
            "confidence_top_decile": round(conf_hi, 4),
            "confidence_bottom_decile": round(conf_lo, 4),
            "error_top_decile": round(err_hi, 4),
            "error_bottom_decile": round(err_lo, 4),
            "top_true_aa_high_activation": top_aa,
        }
        rows.append(row)
        if rank < 10:
            print(
                f"  #{rank + 1} l{layer:2d} n{neuron:5d} | AUROC {au_tr[j]:.4f} | rho(conf) {rho_conf:+.3f} rho(ent) {rho_ent:+.3f} | err top/bottom decile {err_hi:.2f}/{err_lo:.2f}"
            )

    rhos_conf = np.array([r["spearman_confidence"] for r in rows])
    rhos_ent = np.array([r["spearman_entropy"] for r in rows])
    print(
        f"\nacross top-{args.top_k}: rho(conf) mean {rhos_conf.mean():+.4f} | rho(ent) mean {rhos_ent.mean():+.4f}"
    )

    out = RESULTS_DIR / "aa_semantics.json"
    out.write_text(
        json.dumps(
            {
                "model": str(data["model"]),
                "top_k": args.top_k,
                "mean_spearman_confidence": round(float(rhos_conf.mean()), 4),
                "mean_spearman_entropy": round(float(rhos_ent.mean()), 4),
                "neurons": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
