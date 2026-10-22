"""Jacobian lens on ESM-2 (CoTLab jacobian_lens adapted, scoped).

Ridge-regression estimate of J_l = E[d h_final / d h_l] at the masked position.
Tests: (a) causal readout — decode J_l h_l through the AA head; (b) transport alignment —
does J_l map the error direction at layer l toward the final error direction?
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

from biointerp.core import RESULTS_DIR, split_by_protein
from biointerp.esm import AA_ALPHABET, aa_token_ids, load_model
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--layers", default="16,24,30,32")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    sequences = fetch_swissprot(2000, seed=args.seed)[: args.n]
    model, tokenizer, device = load_model(args.model, args.device)
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)
    lm_head = model.lm_head

    captured = {layer: [] for layer in layers}
    final_h = []
    correct_all = []
    true_idx_all = []

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
        true_idx_all.extend(true_idx.tolist())
        for layer in layers:
            captured[layer].append(
                out.hidden_states[layer + 1][b_idx, pos].squeeze(1).float().cpu().numpy()
            )
        final_h.append(out.hidden_states[-1][b_idx, pos].squeeze(1).float().cpu().numpy())
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"batch {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    y = (~np.array(correct_all)).astype(int)
    Y = np.concatenate(final_h)
    tr, te = split_by_protein(np.arange(len(y)), seed=args.seed)
    true_idx_all = np.array(true_idx_all)
    print(f"\nsamples: {len(y)} | layers: {layers}\n")

    print(
        f"{'layer':>5s} {'readout acc (T h)':>17s} {'transport cos':>14s} {'null cos (mean)':>16s} {'null cos (max)':>15s}"
    )
    rows = []
    rng = np.random.default_rng(args.seed)
    for layer in layers:
        H = np.concatenate(captured[layer])
        Htr, Hte = H[tr], H[te]
        Ytr = Y[tr]
        lam = args.ridge * np.trace(Htr.T @ Htr) / Htr.shape[1]
        A = Htr.T @ Htr + lam * np.eye(Htr.shape[1])
        T = np.linalg.solve(A, Htr.T @ Ytr)

        with torch.no_grad():
            h_te = torch.tensor(Hte, dtype=torch.float32, device=device)
            decoded = lm_head(h_te)[:, aa_t].argmax(dim=-1).cpu().numpy()
        readout_acc = float((decoded == true_idx_all[te]).mean())

        d_l = H[tr][y[tr] == 1].mean(axis=0) - H[tr][y[tr] == 0].mean(axis=0)
        d_final = Y[te][y[te] == 1].mean(axis=0) - Y[te][y[te] == 0].mean(axis=0)
        u = T @ d_l
        cos_transport = cosine(u, d_final)
        null_cos = []
        for _ in range(1000):
            v = rng.normal(size=len(d_final))
            v /= np.linalg.norm(v)
            null_cos.append(abs(cosine(v, d_final)))
        rows.append(
            {
                "layer": layer,
                "readout_accuracy": round(readout_acc, 4),
                "transport_cosine": round(cos_transport, 4),
                "null_cosine_mean": round(float(np.mean(null_cos)), 4),
                "null_cosine_max": round(float(np.max(null_cos)), 4),
            }
        )
        print(
            f"{layer:5d} {readout_acc:17.4f} {cos_transport:14.4f} {np.mean(null_cos):16.4f} {np.max(null_cos):15.4f}"
        )

    out = RESULTS_DIR / "jacobian_lens.json"
    out.write_text(
        json.dumps(
            {"model": args.model, "n": int(len(y)), "ridge": args.ridge, "rows": rows}, indent=2
        )
        + "\n"
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
