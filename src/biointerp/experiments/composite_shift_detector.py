"""Composite shift detector on ESM-2 (CoTLab composite_shift_detector adapted).

Single pass combining last-layer masked-position residual norm + masked attention entropy
at a chosen layer into a Mahalanobis anomaly score (calibration split); tests whether the
score tracks accuracy degradation (quantile bins) and detects errors (AUROC).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET, aa_token_ids, pick_device
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--attn-layer", type=int, default=3)
    p.add_argument("--calib-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    sequences = [s for s in fetch_swissprot(2000, seed=args.seed) if len(s) <= args.max_len][
        : args.n
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, attn_implementation="eager")
    model.eval()
    device = pick_device(args.device)
    model.to(device)
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    norms: list[float] = []
    ents: list[float] = []
    correct_all: list[bool] = []

    t0 = time.time()
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = sequences[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, true_aa = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=args.seed + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = model(**tokens, output_attentions=True, output_hidden_states=True)
        pos = positions.to(device)
        b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)

        logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
        pred = logits.argmax(dim=-1).cpu().numpy()
        true_flat = [aa for row in true_aa for aa in row]
        true_idx = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        correct_all.extend((pred == true_idx).tolist())

        h_last = out.hidden_states[-1][b_idx, pos].squeeze(1).float()
        norms.extend(torch.linalg.norm(h_last, dim=-1).cpu().numpy().tolist())

        attn = out.attentions[args.attn_layer].float()
        p = attn + 1e-12
        ent = -(p * p.log()).sum(dim=-1)
        rows = ent[torch.arange(pos.shape[0], device=device), :, pos.squeeze(-1)]
        ents.extend(rows.mean(dim=-1).cpu().numpy().tolist())
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"batch {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    y = (~np.array(correct_all)).astype(int)
    X = np.column_stack([norms, ents])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y))
    n_calib = int(len(y) * args.calib_frac)
    calib, test = perm[:n_calib], perm[n_calib:]

    mu = X[calib].mean(axis=0)
    cov = np.cov(X[calib].T) + np.eye(2) * 1e-6
    inv = np.linalg.inv(cov)

    def mahalanobis(Xs):
        d = Xs - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))

    score = mahalanobis(X[test])
    y_test = y[test]
    auroc_combined = float(roc_auc_score(y_test, score))
    auroc_norm = float(roc_auc_score(y_test, X[test, 0]))
    auroc_ent = float(roc_auc_score(y_test, X[test, 1]))
    rho, pval = spearmanr(score, y_test)

    print(
        f"\nsamples: {len(y)} (calib {len(calib)} / test {len(test)}) | incorrect rate {y.mean():.4f}"
    )
    print(f"attention entropy layer: {args.attn_layer} | norm layer: last\n")
    print(f"AUROC Mahalanobis (combined): {auroc_combined:.4f}")
    print(f"AUROC norm alone:             {auroc_norm:.4f}")
    print(f"AUROC entropy alone:          {auroc_ent:.4f}")
    print(f"Spearman(score, incorrect):   {rho:.4f} (p={pval:.2e})\n")

    q = np.quantile(score, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = []
    print("accuracy by anomaly-score quintile (test):")
    for i in range(5):
        lo, hi = q[i], q[i + 1]
        m = (score >= lo) & (score <= hi) if i == 4 else (score >= lo) & (score < hi)
        acc = float(1 - y_test[m].mean())
        bins.append({"bin": i + 1, "n": int(m.sum()), "accuracy": round(acc, 4)})
        print(f"  bin {i + 1} [{lo:.2f},{hi:.2f}]: n={m.sum():3d} accuracy {acc:.4f}")

    out = RESULTS_DIR / "composite_shift_detector.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": int(len(y)),
                "attn_layer": args.attn_layer,
                "auroc_combined": round(auroc_combined, 4),
                "auroc_norm": round(auroc_norm, 4),
                "auroc_entropy": round(auroc_ent, 4),
                "spearman_score_incorrect": round(float(rho), 4),
                "bins": bins,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
