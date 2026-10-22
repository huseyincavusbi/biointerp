"""Safety-neuron concept on ESM-2 (CoTLab safety_neurons redesigned).

Contrast two behavioral states (correct vs incorrect) at the masked position; select neurons
by raw-activation effect size; patch them toward one state's mean and measure behavior change.

Redesign notes: no checkpoint pair exists for PLMs, so the contrast is correct vs incorrect
states; per-condition forwards; masked-position endpoint.
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
from biointerp.esm import (
    AA_ALPHABET,
    aa_token_ids,
    get_ffn_output_proj,
    get_transformer_layers,
    load_model,
)
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--model-key", default=None, help="registry key (overrides --model)")
    p.add_argument("--ks", default="25,100")
    p.add_argument("--calib-n", type=int, default=500)
    p.add_argument("--eval-n", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_key:
        from biointerp.models import resolve_id

        args.model = resolve_id(args.model_key)
    all_seqs = fetch_swissprot(2000, seed=args.seed)
    calib_seqs = all_seqs[500 : 500 + args.calib_n]
    eval_seqs = all_seqs[: args.eval_n]

    model, tokenizer, device = load_model(args.model, args.device)
    blocks = get_transformer_layers(model)
    n_layers = len(blocks)
    inter = model.config.intermediate_size
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    sums_c = [torch.zeros(inter) for _ in range(n_layers)]
    sums_i = [torch.zeros(inter) for _ in range(n_layers)]
    sumsq_c = [torch.zeros(inter) for _ in range(n_layers)]
    sumsq_i = [torch.zeros(inter) for _ in range(n_layers)]
    counts = {"c": 0, "i": 0}

    def calib_hook(layer):
        def hook(module, inputs, output):
            z = inputs[0][b_idx, pos].squeeze(1).float().cpu()
            for row, is_correct in zip(z, correct_batch):
                if is_correct:
                    sums_c[layer] += row
                    sumsq_c[layer] += row**2
                else:
                    sums_i[layer] += row
                    sumsq_i[layer] += row**2
            return output

        return hook

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
            out = model(**tokens)
        logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
        pred = logits.argmax(dim=-1).cpu().numpy()
        true_flat = [aa for row in true_aa for aa in row]
        ti = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        correct_batch = (pred == ti).tolist()
        counts["c"] += int(sum(correct_batch))
        counts["i"] += int(len(correct_batch) - sum(correct_batch))
        handles = [
            get_ffn_output_proj(blocks[l]).register_forward_hook(calib_hook(l))
            for l in range(n_layers)
        ]
        try:
            with torch.no_grad():
                model(**tokens)
        finally:
            for h in handles:
                h.remove()
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"calibration {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    nc, ni = max(counts["c"], 1), max(counts["i"], 1)
    mean_c = [(sums_c[l] / nc) for l in range(n_layers)]
    mean_i = [(sums_i[l] / ni) for l in range(n_layers)]
    var_c = [(sumsq_c[l] / nc - mean_c[l] ** 2).clamp(min=0) for l in range(n_layers)]
    var_i = [(sumsq_i[l] / ni - mean_i[l] ** 2).clamp(min=0) for l in range(n_layers)]
    effect = torch.cat(
        [
            (mean_c[l] - mean_i[l]).abs() / (torch.sqrt((var_c[l] + var_i[l]) / 2) + 1e-6)
            for l in range(n_layers)
        ]
    ).numpy()
    print(
        f"\ncalibration: {counts['c']} correct / {counts['i']} incorrect | max effect size {effect.max():.3f}"
    )

    def run_condition(ids, target):
        group = {}
        for j in ids:
            group.setdefault(j // inter, []).append(j % inter)
        handles = []
        for layer, idxs in group.items():
            idx_t = torch.tensor(idxs, dtype=torch.long)
            target_t = target[layer][idx_t]

            def make_hook(it, tt):
                def hook(module, inputs, output):
                    zz = inputs[0]
                    ids_dev = it.to(zz.device)
                    delta = (tt.to(zz.device) - zz[..., ids_dev]) @ module.weight[:, ids_dev].T
                    return output + delta

                return hook

            handles.append(
                get_ffn_output_proj(blocks[layer]).register_forward_hook(make_hook(idx_t, target_t))
            )
        probs_all, correct_all = [], []
        try:
            for bi in range((len(eval_seqs) + args.batch_size - 1) // args.batch_size):
                batch = eval_seqs[bi * args.batch_size : (bi + 1) * args.batch_size]
                tokens, positions, true_aa = make_masked_batch(
                    batch, tokenizer, masks_per_seq=1, seed=args.seed + bi
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                pos = positions.to(device)
                b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)
                with torch.no_grad():
                    out = model(**tokens)
                logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
                p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                true_flat = [aa for row in true_aa for aa in row]
                ti = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
                probs_all.append(p)
                correct_all.append(p.argmax(axis=1) == ti)
        finally:
            for h in handles:
                h.remove()
        return np.concatenate(probs_all), np.concatenate(correct_all)

    true_idx = []
    for bi in range((len(eval_seqs) + args.batch_size - 1) // args.batch_size):
        batch = eval_seqs[bi * args.batch_size : (bi + 1) * args.batch_size]
        _, _, true_aa = make_masked_batch(batch, tokenizer, masks_per_seq=1, seed=args.seed + bi)
        true_idx.extend(AA_ALPHABET.index(aa) for row in true_aa for aa in row)
    true_idx = np.array(true_idx)

    base_probs, base_correct = run_condition([], mean_c)
    p_base = base_probs[np.arange(len(true_idx)), true_idx]
    print(f"baseline: acc {base_correct.mean():.4f} | p(true) {p_base.mean():.4f}\n")

    rng = np.random.default_rng(args.seed)
    order = np.argsort(effect)[::-1]
    rows = [
        {
            "condition": "baseline",
            "k": 0,
            "acc": round(float(base_correct.mean()), 4),
            "p_true": round(float(p_base.mean()), 4),
            "d_p_true": 0.0,
            "kl": 0.0,
        }
    ]
    print(f"{'condition':>20s} {'k':>5s} {'acc':>7s} {'p(true)':>8s} {'dp(true)':>9s} {'KL':>8s}")
    for k in [int(x) for x in args.ks.split(",")]:
        top_ids = order[:k].tolist()
        rand_ids = rng.choice(effect.shape[0], size=k, replace=False).tolist()
        for name, ids, target in (
            (f"top-{k}->correct", top_ids, mean_c),
            (f"top-{k}->incorrect", top_ids, mean_i),
            (f"rand-{k}->correct", rand_ids, mean_c),
            (f"rand-{k}->incorrect", rand_ids, mean_i),
        ):
            probs, correct = run_condition(ids, target)
            p_true = probs[np.arange(len(true_idx)), true_idx]
            kls = float(
                (base_probs * (np.log(base_probs + 1e-12) - np.log(probs + 1e-12)))
                .sum(axis=1)
                .mean()
            )
            d = float(p_true.mean() - p_base.mean())
            rows.append(
                {
                    "condition": name,
                    "k": k,
                    "acc": round(float(correct.mean()), 4),
                    "p_true": round(float(p_true.mean()), 4),
                    "d_p_true": round(d, 5),
                    "kl": round(kls, 5),
                }
            )
            print(
                f"{name:>20s} {k:5d} {correct.mean():7.4f} {p_true.mean():8.4f} {d:+9.5f} {kls:8.5f}"
            )

    out = RESULTS_DIR / "safety_neurons.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n_correct": counts["c"],
                "n_incorrect": counts["i"],
                "max_effect": round(float(effect.max()), 4),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
