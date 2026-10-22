"""Probing classifier on ESM-2 (CoTLab probing_classifier adapted).

Per-layer linear probes on masked-position hidden states -> correctness AUROC per layer,
compared against the CETT-feature probe baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET, aa_token_ids, get_transformer_layers, load_model
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sequences = fetch_swissprot(2000, seed=args.seed)[: args.n]
    model, tokenizer, device = load_model(args.model, args.device)
    n_layers = len(get_transformer_layers(model))
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    per_layer: list[list[np.ndarray]] = [[] for _ in range(n_layers + 1)]
    correct_all: list[bool] = []

    t0 = time.time()
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = sequences[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, true_aa = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=args.seed + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = model(**tokens, output_hidden_states=True)
        pos = positions.to(device)
        b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)
        logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
        pred = logits.argmax(dim=-1).cpu().numpy()
        true_flat = [aa for row in true_aa for aa in row]
        true_idx = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        correct_all.extend((pred == true_idx).tolist())
        for i, h in enumerate(out.hidden_states):
            per_layer[i].append(h[b_idx, pos].squeeze(1).float().cpu().numpy())
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"batch {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    y = (~np.array(correct_all)).astype(int)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y))
    n_test = int(round(len(y) * args.test_frac))
    test, train = perm[:n_test], perm[n_test:]
    print(f"\nsamples: {len(y)} (train {len(train)} / test {len(test)})\n")
    print(f"{'layer':>5s} {'hidden AUROC':>13s}")
    rows = []
    for i in range(n_layers + 1):
        x = np.concatenate(per_layer[i])
        if x.std() < 1e-9:
            rows.append({"layer": i, "auroc": None})
            print(f"{i:5d} {'degenerate':>13s}")
            continue
        scaler = StandardScaler()
        xt = scaler.fit_transform(x[train])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
        clf.fit(xt, y[train])
        a = float(roc_auc_score(y[test], clf.predict_proba(scaler.transform(x[test]))[:, 1]))
        rows.append({"layer": i, "auroc": round(a, 4)})
        print(f"{i:5d} {a:13.4f}")

    valid = [r for r in rows if r["auroc"] is not None]
    best = max(valid, key=lambda r: r["auroc"])
    print(f"\nbest layer: {best['layer']} AUROC {best['auroc']}")
    out = RESULTS_DIR / "probing_classifier.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": int(len(y)),
                "best_layer": best["layer"],
                "best_auroc": best["auroc"],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
