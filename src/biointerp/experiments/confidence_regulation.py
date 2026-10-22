"""Confidence regulation on ESM-2 (CoTLab confidence_regulation adapted).

Mode `identify`: weight-space criteria on FFN output-projection neurons —
L2 norm, output-logit variance, null-space fraction of the output head — and
overlap with error neurons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, split_by_protein, univariate_auroc
from biointerp.esm import get_ffn_output_proj, get_transformer_layers, load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="identify", choices=["identify", "intervene"])
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--ks", default="25,100,500")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calib-n", type=int, default=500)
    p.add_argument("--eval-n", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    return p.parse_args()


def run_intervene(args) -> None:
    """Mean-ablation of error-neuron sets: replace z_j with its calibration mean."""
    import time as _time

    import torch as _torch

    from biointerp.esm import get_ffn_output_proj as _proj
    from biointerp.esm import get_transformer_layers as _layers
    from biointerp.esm import load_model as _load
    from biointerp.task import fetch_swissprot as _fetch
    from biointerp.task import make_masked_batch as _mask

    data = np.load(args.npz)
    y = (~data["correct"]).astype(int)
    protein = data["protein"]
    inter = int(data["intermediate_dim"])
    n_layers = int(data["n_layers"])
    tr, _ = split_by_protein(protein, seed=args.seed)
    au_tr = univariate_auroc(data["cett"][tr], y[tr])
    order = np.argsort(au_tr)[::-1]

    all_seqs = _fetch(2000, seed=args.seed)
    calib_seqs = all_seqs[500 : 500 + args.calib_n]
    eval_seqs = all_seqs[: args.eval_n]

    model, tokenizer, device = _load(args.model, args.device)
    blocks = _layers(model)

    z_sums = [_torch.zeros(inter, dtype=_torch.float32) for _ in range(n_layers)]
    count = 0

    def make_calib_hook(layer):
        def hook(module, inputs, output):
            nonlocal count
            z = inputs[0]
            z_sums[layer] += z[b_idx, pos].squeeze(1).sum(dim=0).float().cpu()
            return output

        return hook

    t0 = _time.time()
    n_batches = (len(calib_seqs) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = calib_seqs[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, _ = _mask(batch, tokenizer, masks_per_seq=1, seed=args.seed + 100 + bi)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        pos = positions.to(device)
        b_idx = _torch.arange(pos.shape[0], device=device).unsqueeze(1)
        handles = [
            _proj(blocks[l]).register_forward_hook(make_calib_hook(l)) for l in range(n_layers)
        ]
        try:
            with _torch.no_grad():
                model(**tokens)
        finally:
            for h in handles:
                h.remove()
        count += pos.shape[0]
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"calibration {bi + 1}/{n_batches} | {_time.time() - t0:.0f}s")

    mean_z = [(z_sums[l] / max(count, 1)).float() for l in range(n_layers)]

    def run_condition(ids):
        group = {}
        for j in ids:
            group.setdefault(j // inter, []).append(j % inter)
        handles = []
        for layer, idxs in group.items():
            idx_t = _torch.tensor(idxs, dtype=_torch.long)
            mean_t = mean_z[layer][idx_t]

            def make_hook(module, it, mt):
                def hook(module, inputs, output):
                    z = inputs[0]
                    ids_dev = it.to(z.device)
                    delta = (mt.to(z.device) - z[..., ids_dev]) @ module.weight[:, ids_dev].T
                    return output + delta

                return hook

            handles.append(
                _proj(blocks[layer]).register_forward_hook(make_hook(blocks[layer], idx_t, mean_t))
            )
        probs_all, correct_all = [], []
        try:
            for bi in range((len(eval_seqs) + args.batch_size - 1) // args.batch_size):
                batch = eval_seqs[bi * args.batch_size : (bi + 1) * args.batch_size]
                tokens, positions, true_aa = _mask(
                    batch, tokenizer, masks_per_seq=1, seed=args.seed + bi
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                pos = positions.to(device)
                b_idx = _torch.arange(pos.shape[0], device=device).unsqueeze(1)
                with _torch.no_grad():
                    out = model(**tokens)
                logits = out.logits[b_idx, pos][
                    ..., _torch.tensor(aa_ids_local, device=device)
                ].squeeze(1)
                p = _torch.softmax(logits.float(), dim=-1).cpu().numpy()
                true_flat = [aa for row in true_aa for aa in row]
                ti = np.array([_AA.index(aa) for aa in true_flat])
                probs_all.append(p)
                correct_all.append(p.argmax(axis=1) == ti)
        finally:
            for h in handles:
                h.remove()
        return np.concatenate(probs_all), np.concatenate(correct_all)

    from biointerp.esm import AA_ALPHABET as _AA
    from biointerp.esm import aa_token_ids

    aa_ids_local = aa_token_ids(tokenizer)

    true_idx = []
    for bi in range((len(eval_seqs) + args.batch_size - 1) // args.batch_size):
        batch = eval_seqs[bi * args.batch_size : (bi + 1) * args.batch_size]
        _, _, true_aa = _mask(batch, tokenizer, masks_per_seq=1, seed=args.seed + bi)
        true_idx.extend(_AA.index(aa) for row in true_aa for aa in row)
    true_idx = np.array(true_idx)

    base_probs, base_correct = run_condition([])
    p_base = base_probs[np.arange(len(true_idx)), true_idx]
    print(f"\nbaseline: acc {base_correct.mean():.4f} | p(true) {p_base.mean():.4f}\n")

    ks = [int(k) for k in args.ks.split(",")]
    rng = np.random.default_rng(args.seed)

    def layer_matched_random(top_ids):
        counts = {}
        for j in top_ids:
            counts[j // inter] = counts.get(j // inter, 0) + 1
        chosen = []
        for layer, c in counts.items():
            pool = np.arange(layer * inter, (layer + 1) * inter)
            chosen.extend(rng.choice(pool, size=c, replace=False).tolist())
        return chosen

    rows = [
        {
            "condition": "baseline",
            "k": 0,
            "acc": round(float(base_correct.mean()), 4),
            "p_true": round(float(p_base.mean()), 4),
            "d_p_true": 0.0,
            "wrong_to_right": 0,
            "right_to_wrong": 0,
            "kl": 0.0,
        }
    ]
    print(
        f"{'condition':>14s} {'k':>5s} {'acc':>7s} {'p(true)':>8s} {'dp(true)':>9s} {'w->r':>5s} {'r->w':>5s} {'KL':>8s}"
    )
    for k in ks:
        top_ids = order[:k].tolist()
        rand_ids = layer_matched_random(top_ids)
        for name, ids in ((f"top-{k}", top_ids), (f"random-{k}", rand_ids)):
            probs, correct = run_condition(ids)
            p_true = probs[np.arange(len(true_idx)), true_idx]
            d = float(p_true.mean() - p_base.mean())
            kls = float(
                (base_probs * (np.log(base_probs + 1e-12) - np.log(probs + 1e-12)))
                .sum(axis=1)
                .mean()
            )
            w2r = int(((~base_correct) & correct).sum())
            r2w = int((base_correct & (~correct)).sum())
            rows.append(
                {
                    "condition": name,
                    "k": k,
                    "acc": round(float(correct.mean()), 4),
                    "p_true": round(float(p_true.mean()), 4),
                    "d_p_true": round(d, 5),
                    "wrong_to_right": w2r,
                    "right_to_wrong": r2w,
                    "kl": round(kls, 5),
                }
            )
            print(
                f"{name:>14s} {k:5d} {correct.mean():7.4f} {p_true.mean():8.4f} {d:+9.5f} {w2r:5d} {r2w:5d} {kls:8.5f}"
            )

    out = RESULTS_DIR / "confidence_regulation_intervene.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "calib_n": len(calib_seqs),
                "eval_n": len(eval_seqs),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {out}")


def run_identify(args) -> None:
    data = np.load(args.npz)
    y = (~data["correct"]).astype(int)
    protein = data["protein"]
    n_layers = int(data["n_layers"])
    inter = int(data["intermediate_dim"])

    tr, te = split_by_protein(protein, seed=args.seed)
    au_tr = univariate_auroc(data["cett"][tr], y[tr])

    model, _, _ = load_model(args.model, device="cpu")
    blocks = get_transformer_layers(model)
    head = model.lm_head.decoder.weight.detach().float().numpy()
    u, s, vh = np.linalg.svd(head, full_matrices=False)
    rank = int((s > 1e-6 * s[0]).sum())
    row_space = vh[:rank]

    norms = np.empty(n_layers * inter)
    logit_vars = np.empty(n_layers * inter)
    rhos = np.empty(n_layers * inter)
    for layer in range(n_layers):
        sl = slice(layer * inter, (layer + 1) * inter)
        w_out = get_ffn_output_proj(blocks[layer]).weight.detach().float().numpy()
        norms[sl] = np.linalg.norm(w_out, axis=0)
        logits = head @ w_out
        logit_vars[sl] = logits.var(axis=0)
        proj = row_space.T @ (row_space @ w_out)
        rhos[sl] = 1.0 - (proj**2).sum(axis=0) / (np.maximum((w_out**2).sum(axis=0), 1e-12))
    del model

    print(f"model: {args.model} | output-head rank: {rank} (vocab {head.shape[0]})\n")
    results = {"model": args.model, "head_rank": rank}

    criteria = {"norm": norms, "logit_variance": logit_vars, "null_space_fraction": rhos}
    print("Spearman(criterion, error-AUROC) per layer:")
    print(f"{'layer':>5s} {'norm':>8s} {'logitvar':>9s} {'rho':>8s}")
    layer_corr = {"norm": [], "logit_variance": [], "null_space_fraction": []}
    for layer in range(n_layers):
        sl = slice(layer * inter, (layer + 1) * inter)
        row = {}
        for name, values in criteria.items():
            r, _ = spearmanr(values[sl], au_tr[sl])
            layer_corr[name].append(round(float(r), 4))
            row[name] = float(r)
        print(
            f"{layer:5d} {row['norm']:8.4f} {row['logit_variance']:9.4f} {row['null_space_fraction']:8.4f}"
        )
    results["layer_correlations"] = layer_corr

    print()
    print(f"{'k':>5s} " + " ".join(f"{name:>20s}" for name in criteria))
    ks = [int(k) for k in args.ks.split(",")]
    overlap = {}
    for k in ks:
        err = set(np.argsort(au_tr)[::-1][:k].tolist())
        row = {}
        for name, values in criteria.items():
            top = set(np.argsort(values)[::-1][:k].tolist())
            row[name] = round(len(err & top) / len(err | top), 4)
        overlap[k] = row
        print(f"{k:5d} " + " ".join(f"{row[name]:20.4f}" for name in criteria))
    results["jaccard_by_k"] = overlap

    global_corr = {}
    for name, values in criteria.items():
        r, _ = spearmanr(values, au_tr)
        global_corr[name] = round(float(r), 4)
    print()
    print(
        f"global Spearman: norm {global_corr['norm']} | logitvar {global_corr['logit_variance']} | rho {global_corr['null_space_fraction']}"
    )
    results["global_correlations"] = global_corr

    out = RESULTS_DIR / "confidence_regulation.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nsaved: {out}")


def main() -> None:
    args = parse_args()
    if args.mode == "identify":
        run_identify(args)
    else:
        run_intervene(args)


if __name__ == "__main__":
    main()
