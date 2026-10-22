"""Activation compare on ESM-2 (CoTLab activation_compare adapted).

Mean residual-stream vector per layer for correct vs incorrect masked predictions;
cosine/L2 divergence profile + split-half reliability control.
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
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sequences = fetch_swissprot(2000, seed=args.seed)[: args.n]
    model, tokenizer, device = load_model(args.model, args.device)
    n_layers = len(get_transformer_layers(model))
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    per_layer_vectors: list[list[np.ndarray]] = [[] for _ in range(n_layers + 1)]
    labels: list[bool] = []

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
        correct = pred == true_idx
        labels.extend(correct.tolist())
        for i, h in enumerate(out.hidden_states):
            h_at = h[b_idx, pos].squeeze(1).float().cpu().numpy()
            for row in h_at:
                per_layer_vectors[i].append(row)
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"batch {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    labels_arr = np.array(labels)
    print(
        f"\nsamples: {len(labels_arr)} | correct {int(labels_arr.sum())} | incorrect {int((~labels_arr).sum())}\n"
    )

    rows = []
    print(
        f"{'layer':>5s} {'cos(c,i)':>9s} {'L2':>8s} {'||c||':>8s} {'||i||':>8s} {'splithalf cos c':>15s} {'splithalf cos i':>15s}"
    )
    for i in range(n_layers + 1):
        vecs = np.stack(per_layer_vectors[i])
        c = vecs[labels_arr]
        ic = vecs[~labels_arr]
        mean_c, mean_i = c.mean(axis=0), ic.mean(axis=0)
        cos_ci = cosine(mean_c, mean_i)
        l2 = float(np.linalg.norm(mean_c - mean_i))
        rng = np.random.default_rng(args.seed)

        def split_half_cos(group):
            perm = rng.permutation(len(group))
            h1, h2 = group[perm[: len(perm) // 2]], group[perm[len(perm) // 2 :]]
            return cosine(h1.mean(axis=0), h2.mean(axis=0))

        sh_c = split_half_cos(c) if len(c) > 4 else float("nan")
        sh_i = split_half_cos(ic) if len(ic) > 4 else float("nan")
        rows.append(
            {
                "layer": i,
                "cos_correct_incorrect": round(cos_ci, 4),
                "l2": round(l2, 4),
                "norm_correct": round(float(np.linalg.norm(mean_c)), 4),
                "norm_incorrect": round(float(np.linalg.norm(mean_i)), 4),
                "splithalf_cos_correct": round(sh_c, 4),
                "splithalf_cos_incorrect": round(sh_i, 4),
            }
        )
        print(
            f"{i:5d} {cos_ci:9.4f} {l2:8.4f} {np.linalg.norm(mean_c):8.4f} {np.linalg.norm(mean_i):8.4f} {sh_c:15.4f} {sh_i:15.4f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "activation_compare.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": int(len(labels_arr)),
                "n_correct": int(labels_arr.sum()),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
