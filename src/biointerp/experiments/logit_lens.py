"""Logit lens on ESM-2 (CoTLab logit_lens adapted to the masked position).

Project every layer's masked-position hidden state through final LayerNorm + AA head.
Measures when the correct residue emerges across depth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET, aa_token_ids, get_transformer_layers, load_model
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sequences = fetch_swissprot(2000, seed=args.seed)[: args.n]
    model, tokenizer, device = load_model(args.model, args.device)
    layers = get_transformer_layers(model)
    lm_head = model.lm_head
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)
    n_lens = len(layers) + 1

    per_layer_correct = np.zeros(n_lens, dtype=np.int64)
    emergence = []
    final_correct = []
    true_all: list[str] = []
    total = 0

    t0 = time.time()
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = sequences[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, true_aa = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=args.seed + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = model(**tokens, output_hidden_states=True)
        hs = out.hidden_states
        pos = positions.to(device)
        b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)
        true_flat = [aa for row in true_aa for aa in row]
        true_all.extend(true_flat)
        true_idx = np.array([AA_ALPHABET.index(aa) for aa in true_flat])

        if bi == 0:
            h_last = hs[-1][b_idx, pos]
            ref = out.logits[b_idx, pos][..., aa_t]
            direct = lm_head(h_last)[..., aa_t]
            d_direct = (direct - ref).abs().max().item()
            print(f"sanity | lm_head(h_last) vs logits: {d_direct:.6f}\n")

        sample_emergence = np.full(len(true_flat), -1, dtype=np.int64)
        for i in range(n_lens):
            h = hs[i][b_idx, pos]
            logits = lm_head(h)[..., aa_t]
            pred = logits.argmax(dim=-1).squeeze(-1).cpu().numpy()
            correct = pred == true_idx
            per_layer_correct[i] += int(correct.sum())
            first = sample_emergence < 0
            sample_emergence[first & correct] = i
        emergence.extend(sample_emergence.tolist())
        final_correct.extend((sample_emergence >= 0).tolist())
        total += len(true_flat)
        if (bi + 1) % 5 == 0 or bi + 1 == n_batches:
            print(f"batch {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    print()
    print(f"{'lens layer':>10s} {'top1 acc':>9s}")
    for i in range(n_lens):
        print(f"{i:10d} {per_layer_correct[i] / total:9.4f}")

    emerged = np.array(emergence)
    never = int((emerged < 0).sum())
    print()
    print(f"final accuracy (last lens layer): {per_layer_correct[-1] / total:.4f}")
    print(f"never emerged: {never}/{total} ({never / total:.4f})")
    if (emerged >= 0).sum() > 0:
        print(f"mean emergence layer (of emerged): {emerged[emerged >= 0].mean():.2f}")
        hist = np.bincount(emerged[emerged >= 0], minlength=n_lens)
        print(f"emergence histogram: {hist.tolist()}")

    results = {
        "model": args.model,
        "n": total,
        "per_layer_top1": [round(per_layer_correct[i] / total, 4) for i in range(n_lens)],
        "final_accuracy": round(per_layer_correct[-1] / total, 4),
        "never_emerged": never,
        "mean_emergence_layer": round(float(emerged[emerged >= 0].mean()), 2)
        if (emerged >= 0).sum()
        else None,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "logit_lens.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    records = RESULTS_DIR / "logit_lens.samples.jsonl"
    with records.open("w") as fh:
        for i, e in enumerate(emergence):
            fh.write(
                json.dumps({"sample": i, "emergence_layer": int(e), "true_aa": true_all[i]}) + "\n"
            )
    print(f"\nsaved: {out}")
    print(f"saved: {records}")


if __name__ == "__main__":
    main()
