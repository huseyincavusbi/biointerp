"""Ladder scaling analysis (Bio-Interp).

Compares the CETT-probe error AUROC across ESM-2 sizes (and other registry models).
Reads extraction npz files from pilot/results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import PILOT_RESULTS, RESULTS_DIR, split_by_protein, univariate_auroc
from biointerp.models import get_spec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="esm2_8m,esm2_35m,esm2_150m,esm2_650m")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    print(
        f"{'model':>12s} {'params':>10s} {'acc':>7s} {'probe AUROC':>11s} {'conf AUROC':>10s} {'features':>10s}"
    )
    for key in args.models.split(","):
        spec = get_spec(key)
        name = Path(spec["hf_id"]).name
        npz = PILOT_RESULTS / f"{name}_n{args.n}_m1_seed0.npz"
        if not npz.exists():
            print(f"{key:>12s} {'missing npz: ' + npz.name}")
            continue
        data = np.load(npz)
        x = data["cett"]
        y = (~data["correct"]).astype(int)
        conf = data["confidence"]
        protein = data["protein"]
        tr, te = split_by_protein(protein, seed=args.seed)
        au = univariate_auroc(x[tr], y[tr])
        idx = np.argsort(au)[::-1][: args.top_k]
        scaler = StandardScaler()
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
        clf.fit(scaler.fit_transform(x[tr][:, idx]), y[tr])
        auroc_probe = float(
            roc_auc_score(y[te], clf.predict_proba(scaler.transform(x[te][:, idx]))[:, 1])
        )
        auroc_conf = float(roc_auc_score(y[te], -conf[te]))
        acc = float(data["correct"].mean())
        rows.append(
            {
                "key": key,
                "hf_id": spec["hf_id"],
                "features": int(x.shape[1]),
                "accuracy": round(acc, 4),
                "auroc_probe": round(auroc_probe, 4),
                "auroc_confidence": round(auroc_conf, 4),
            }
        )
        print(
            f"{key:>12s} {'':>10s} {acc:7.4f} {auroc_probe:11.4f} {auroc_conf:10.4f} {x.shape[1]:10d}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ladder_scaling.json"
    out.write_text(json.dumps({"n": args.n, "top_k": args.top_k, "rows": rows}, indent=2) + "\n")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
