"""Steering vectors on ESM-2 (CoTLab steering_vectors redesigned for PLMs).

Direction: d_l = mean(h_l | incorrect) - mean(h_l | correct) at the masked position
(calibration set). Intervene: h_l += alpha * d_l at the masked position only.
Endpoint: p(true residue), accuracy, KL, paired flips. Controls: random, reversed.

Redesign notes (see docs/CAUSAL_REDESIGN.md): one condition per batched forward (no packed
rows — invalid for bidirectional attention); intervention and endpoint at the masked position.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET, aa_token_ids, get_transformer_layers, load_model
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--calib-n", type=int, default=500)
    p.add_argument("--eval-n", type=int, default=300)
    p.add_argument("--layers", default="24,30,32")
    p.add_argument("--alphas", default="-3,-1,1,3")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--only-controls", action="store_true")
    p.add_argument("--control-layers", default="24,30")
    p.add_argument("--control-alphas", default="1,-1")
    return p.parse_args()


@torch.inference_mode()
def run_pass(model, tokenizer, sequences, layers, directions, alpha, batch_size, seed, device):
    blocks = get_transformer_layers(model)
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)
    probs_all, correct_all = [], []
    handles = []

    if directions:
        for layer, direction in directions.items():
            d = torch.tensor(direction, dtype=torch.float32, device=device)

            def make_hook(dvec, a):
                def hook(module, inputs, output):
                    is_tuple = isinstance(output, tuple)
                    h = output[0] if is_tuple else output
                    h = h.clone()
                    h[b_idx, pos] = h[b_idx, pos] + a * dvec
                    return (h,) + output[1:] if is_tuple else h

                return hook

            handles.append(blocks[layer].register_forward_hook(make_hook(d, alpha)))

    try:
        n_batches = (len(sequences) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            batch = sequences[bi * batch_size : (bi + 1) * batch_size]
            tokens, positions, true_aa = make_masked_batch(
                batch, tokenizer, masks_per_seq=1, seed=seed + bi
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            pos = positions.to(device)
            b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)
            out = model(**tokens)
            logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            true_flat = [aa for row in true_aa for aa in row]
            true_idx = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
            probs_all.append(probs)
            correct_all.append(probs.argmax(axis=1) == true_idx)
    finally:
        for handle in handles:
            handle.remove()

    probs = np.concatenate(probs_all)
    correct = np.concatenate(correct_all)
    return probs, correct


def kl(p, q):
    return (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(axis=1)


def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    all_seqs = fetch_swissprot(2000, seed=args.seed)
    calib_seqs = all_seqs[500 : 500 + args.calib_n]
    eval_seqs = all_seqs[: args.eval_n]

    model, tokenizer, device = load_model(args.model, args.device)
    get_transformer_layers(model)
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    # --- calibration: directions ---
    sums = {
        layer: {
            "c": np.zeros(model.config.hidden_size),
            "i": np.zeros(model.config.hidden_size),
            "nc": 0,
            "ni": 0,
        }
        for layer in layers
    }
    t0 = time.time()
    n_batches = (len(calib_seqs) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = calib_seqs[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, true_aa = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=args.seed + 100 + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        pos = positions.to(device)
        b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
        pred = logits.argmax(dim=-1).cpu().numpy()
        true_flat = [aa for row in true_aa for aa in row]
        true_idx = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        correct = pred == true_idx
        for layer in layers:
            h = out.hidden_states[layer + 1][b_idx, pos].squeeze(1).float().cpu().numpy()
            sums[layer]["c"] += h[correct].sum(axis=0)
            sums[layer]["nc"] += int(correct.sum())
            sums[layer]["i"] += h[~correct].sum(axis=0)
            sums[layer]["ni"] += int((~correct).sum())
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"calibration {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    directions = {}
    for layer in layers:
        s = sums[layer]
        directions[layer] = s["i"] / max(s["ni"], 1) - s["c"] / max(s["nc"], 1)
        print(f"layer {layer}: ||d|| = {np.linalg.norm(directions[layer]):.3f}")

    # --- baseline + conditions ---
    def run_condition(directions_map, alpha, name):
        t1 = time.time()
        probs, correct = run_pass(
            model,
            tokenizer,
            eval_seqs,
            layers,
            directions_map,
            alpha,
            args.batch_size,
            args.seed,
            device,
        )
        return probs, correct, time.time() - t1

    base_probs, base_correct, dt = run_condition(None, 1.0, "baseline")
    n = len(eval_seqs)

    true_idx_list = []
    for bi in range((n + args.batch_size - 1) // args.batch_size):
        batch = eval_seqs[bi * args.batch_size : (bi + 1) * args.batch_size]
        _, _, true_aa = make_masked_batch(batch, tokenizer, masks_per_seq=1, seed=args.seed + bi)
        true_idx_list.extend(AA_ALPHABET.index(aa) for row in true_aa for aa in row)
    true_idx = np.array(true_idx_list)
    p_true_base = base_probs[np.arange(n), true_idx]

    print(
        f"\nbaseline: acc {base_correct.mean():.4f} | p(true) {p_true_base.mean():.4f} | {dt:.0f}s\n"
    )
    print(
        f"{'condition':>18s} {'acc':>7s} {'p(true)':>8s} {'dp(true)':>9s} {'wilcoxon p':>11s} {'KL':>8s} {'w->r':>5s} {'r->w':>5s}"
    )

    rows = [
        {
            "condition": "baseline",
            "acc": round(float(base_correct.mean()), 4),
            "p_true": round(float(p_true_base.mean()), 4),
            "d_p_true": 0.0,
            "wilcoxon_p": None,
            "kl": 0.0,
            "wrong_to_right": 0,
            "right_to_wrong": 0,
        }
    ]
    conditions = []
    if not args.only_controls:
        for layer in layers:
            for alpha in alphas:
                conditions.append((f"l{layer}_a{alpha:g}", {layer: directions[layer]}, alpha))
    rng = np.random.default_rng(args.seed)
    for cl in [int(x) for x in args.control_layers.split(",")]:
        rand_dir = rng.normal(size=model.config.hidden_size).astype(np.float32)
        rand_dir = rand_dir / np.linalg.norm(rand_dir) * np.linalg.norm(directions[cl])
        for ca in [float(x) for x in args.control_alphas.split(",")]:
            conditions.append((f"l{cl}_random_a{ca:g}", {cl: rand_dir}, ca))
            conditions.append((f"l{cl}_reversed_a{ca:g}", {cl: -directions[cl]}, ca))

    for name, dmap, alpha in conditions:
        probs, correct, dt = run_condition(dmap, alpha, name)
        p_true = probs[np.arange(n), true_idx]
        d = p_true - p_true_base
        try:
            pval = float(wilcoxon(p_true, p_true_base).pvalue)
        except Exception:
            pval = float("nan")
        k = float(kl(base_probs, probs).mean())
        w2r = int(((~base_correct) & correct).sum())
        r2w = int((base_correct & (~correct)).sum())
        rows.append(
            {
                "condition": name,
                "acc": round(float(correct.mean()), 4),
                "p_true": round(float(p_true.mean()), 4),
                "d_p_true": round(float(d.mean()), 5),
                "wilcoxon_p": pval,
                "kl": round(k, 5),
                "wrong_to_right": w2r,
                "right_to_wrong": r2w,
            }
        )
        print(
            f"{name:>18s} {correct.mean():7.4f} {p_true.mean():8.4f} {d.mean():+9.5f} {pval:11.4f} {k:8.5f} {w2r:5d} {r2w:5d}"
        )

    np.savez(
        RESULTS_DIR / "steering_vectors.directions.npz", **{f"l{l}": directions[l] for l in layers}
    )
    suffix = "_controls" if args.only_controls else ""
    out = RESULTS_DIR / f"steering_vectors{suffix}.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "calib_n": len(calib_seqs),
                "eval_n": n,
                "direction_norms": {
                    str(l): round(float(np.linalg.norm(directions[l])), 3) for l in layers
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
