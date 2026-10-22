"""Attention analysis on ESM-2 (CoTLab attention_analysis adapted).

Per-layer mean attention entropy at the masked query position and over all positions;
per-head statistics; AUROC for incorrectness. Loads with eager attention.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET, aa_token_ids, pick_device
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
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
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    masked_ent: list[list[float]] = [[] for _ in range(n_layers)]
    all_ent: list[list[float]] = [[] for _ in range(n_layers)]
    correct_all: list[bool] = []

    t0 = time.time()
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = sequences[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, true_aa = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=args.seed + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = model(**tokens, output_attentions=True)
        pos = positions.to(device)
        b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)

        logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
        pred = logits.argmax(dim=-1).cpu().numpy()
        true_flat = [aa for row in true_aa for aa in row]
        true_idx = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        correct_all.extend((pred == true_idx).tolist())

        for i, attn in enumerate(out.attentions):
            a = attn.float()
            p = a + 1e-12
            ent = -(p * p.log()).sum(dim=-1)
            rows = ent[torch.arange(pos.shape[0], device=device), :, pos.squeeze(-1)]
            masked_ent[i].extend(rows.mean(dim=-1).cpu().numpy().tolist())
            all_ent[i].extend(ent.mean(dim=(1, 2)).cpu().numpy().tolist())
        if (bi + 1) % 5 == 0 or bi + 1 == n_batches:
            print(f"batch {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    y = (~np.array(correct_all)).astype(int)
    print(f"\nsamples: {len(y)} | heads: {n_heads} | incorrect rate {y.mean():.4f}\n")
    print(
        f"{'layer':>5s} {'masked AUROC':>13s} {'all-pos AUROC':>14s} {'masked ent c':>13s} {'masked ent i':>13s}"
    )
    rows = []
    for i in range(n_layers):
        me = np.array(masked_ent[i])
        ae = np.array(all_ent[i])
        a_m = float(roc_auc_score(y, me))
        a_a = float(roc_auc_score(y, ae))
        rows.append(
            {
                "layer": i,
                "masked_auroc": round(a_m, 4),
                "all_pos_auroc": round(a_a, 4),
                "masked_entropy_correct": round(float(me[y == 0].mean()), 4),
                "masked_entropy_incorrect": round(float(me[y == 1].mean()), 4),
            }
        )
        print(f"{i:5d} {a_m:13.4f} {a_a:14.4f} {me[y == 0].mean():13.4f} {me[y == 1].mean():13.4f}")

    best = max(rows, key=lambda r: abs(r["masked_auroc"] - 0.5))
    print(f"\nbest layer (masked entropy): {best['layer']} AUROC {best['masked_auroc']}")

    out = RESULTS_DIR / "attention_analysis.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": int(len(y)),
                "n_heads": n_heads,
                "best_layer": best["layer"],
                "best_masked_auroc": best["masked_auroc"],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
