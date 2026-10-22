"""Residual-norm OOD on ESM-2 (CoTLab residual_norm_ood adapted).

Masked-position residual L2 norm per layer vs AA-logit entropy as error/OOD detectors:
AUROC for incorrectness, best balanced-accuracy threshold, layer profile.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET, aa_token_ids, get_transformer_layers, load_model
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
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

    norms: list[list[float]] = [[] for _ in range(n_layers + 1)]
    entropies: list[float] = []
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
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        ent = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        entropies.extend(ent.tolist())
        pred = probs.argmax(axis=1)
        true_flat = [aa for row in true_aa for aa in row]
        true_idx = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        correct_all.extend((pred == true_idx).tolist())

        for i, h in enumerate(out.hidden_states):
            h_at = h[b_idx, pos].squeeze(1).float()
            norms[i].extend(torch.linalg.norm(h_at, dim=-1).cpu().numpy().tolist())
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"batch {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    y = (~np.array(correct_all)).astype(int)
    ent_arr = np.array(entropies)
    print(f"\nsamples: {len(y)} | incorrect rate {y.mean():.4f}\n")
    print(f"entropy baseline AUROC: {roc_auc_score(y, ent_arr):.4f}\n")

    rows = []
    print(f"{'layer':>5s} {'norm AUROC':>11s} {'mean||h|| c':>12s} {'mean||h|| i':>12s}")
    for i in range(n_layers + 1):
        nrm = np.array(norms[i])
        if nrm.std() < 1e-9:
            print(f"{i:5d} {'degenerate':>11s}")
            rows.append({"layer": i, "norm_auroc": None})
            continue
        a = float(roc_auc_score(y, nrm))
        mean_c = float(nrm[y == 0].mean())
        mean_i = float(nrm[y == 1].mean())
        rows.append(
            {
                "layer": i,
                "norm_auroc": round(a, 4),
                "mean_norm_correct": round(mean_c, 4),
                "mean_norm_incorrect": round(mean_i, 4),
            }
        )
        print(f"{i:5d} {a:11.4f} {mean_c:12.4f} {mean_i:12.4f}")

    valid = [r for r in rows if r["norm_auroc"] is not None]
    best = max(valid, key=lambda r: abs(r["norm_auroc"] - 0.5))
    nrm_best = np.array(norms[best["layer"]])
    thresholds = np.quantile(nrm_best, np.linspace(0.05, 0.95, 91))
    bacc = [(balanced_accuracy_score(y, (nrm_best >= t).astype(int)), t) for t in thresholds]
    best_bacc, best_t = max(bacc, key=lambda x: x[0])
    print(f"\nbest norm layer: {best['layer']} (AUROC {best['norm_auroc']})")
    print(f"threshold {best_t:.4f} -> balanced accuracy {best_bacc:.4f}")

    out = RESULTS_DIR / "residual_norm_ood.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": int(len(y)),
                "incorrect_rate": round(float(y.mean()), 4),
                "entropy_auroc": round(float(roc_auc_score(y, ent_arr)), 4),
                "best_norm_layer": best["layer"],
                "best_norm_auroc": best["norm_auroc"],
                "best_threshold": round(float(best_t), 4),
                "best_balanced_acc": round(float(best_bacc), 4),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
