"""Cluster-aware ablation on ESM-2 (redesign; CoTLab #91 concept).

Cluster correlated error neurons (correlation of activations across samples), then
mean-ablate clusters jointly / separately vs size-matched random sets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, split_by_protein, univariate_auroc
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
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--model-key", default=None, help="registry key (overrides --model)")
    p.add_argument("--top-n", type=int, default=1000)
    p.add_argument("--corr-threshold", type=float, default=0.5)
    p.add_argument("--n-clusters", type=int, default=3)
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
    data = np.load(args.npz)
    y = (~data["correct"]).astype(int)
    protein = data["protein"]
    inter = int(data["intermediate_dim"])
    n_layers = int(data["n_layers"])
    tr, _ = split_by_protein(protein, seed=args.seed)
    au_tr = univariate_auroc(data["cett"][tr], y[tr])
    order = np.argsort(au_tr)[::-1][: args.top_n]
    acts = data["cett"][tr][:, order].astype(np.float64)

    corr = np.corrcoef(acts, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(z, t=1.0 - args.corr_threshold, criterion="distance")
    sizes = np.bincount(labels)[1:]
    print(
        f"clustered {args.top_n} error neurons -> {len(sizes)} clusters (threshold |r|>={args.corr_threshold})"
    )
    print(f"largest cluster sizes: {sorted(sizes, reverse=True)[:10]}")

    cluster_ids = [int(c + 1) for c in np.argsort(sizes)[::-1][: args.n_clusters]]

    all_seqs = fetch_swissprot(2000, seed=args.seed)
    calib_seqs = all_seqs[500 : 500 + args.calib_n]
    eval_seqs = all_seqs[: args.eval_n]
    model, tokenizer, device = load_model(args.model, args.device)
    blocks = get_transformer_layers(model)
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    z_sums = [torch.zeros(inter) for _ in range(n_layers)]
    count = 0

    def calib_hook(layer):
        def hook(module, inputs, output):
            nonlocal count
            z_sums[layer] += inputs[0][b_idx, pos].squeeze(1).sum(dim=0).float().cpu()
            return output

        return hook

    t0 = time.time()
    n_batches = (len(calib_seqs) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = calib_seqs[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, _ = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=args.seed + 100 + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        pos = positions.to(device)
        b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)
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
        count += pos.shape[0]
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"calibration {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")
    mean_z = [(s / max(count, 1)).float() for s in z_sums]

    def run_condition(global_ids):
        group = {}
        for j in global_ids:
            group.setdefault(j // inter, []).append(j % inter)
        handles = []
        for layer, idxs in group.items():
            idx_t = torch.tensor(idxs, dtype=torch.long)
            mean_t = mean_z[layer][idx_t]

            def make_hook(it, mt):
                def hook(module, inputs, output):
                    zz = inputs[0]
                    ids_dev = it.to(zz.device)
                    delta = (mt.to(zz.device) - zz[..., ids_dev]) @ module.weight[:, ids_dev].T
                    return output + delta

                return hook

            handles.append(
                get_ffn_output_proj(blocks[layer]).register_forward_hook(make_hook(idx_t, mean_t))
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

    base_probs, base_correct = run_condition([])
    p_base = base_probs[np.arange(len(true_idx)), true_idx]
    print(f"\nbaseline: acc {base_correct.mean():.4f} | p(true) {p_base.mean():.4f}\n")

    rng = np.random.default_rng(args.seed)
    rows = [
        {
            "condition": "baseline",
            "n_neurons": 0,
            "acc": round(float(base_correct.mean()), 4),
            "p_true": round(float(p_base.mean()), 4),
            "d_p_true": 0.0,
            "kl": 0.0,
        }
    ]
    print(f"{'condition':>18s} {'n':>5s} {'acc':>7s} {'p(true)':>8s} {'dp(true)':>9s} {'KL':>8s}")

    conditions = []
    for cid in cluster_ids:
        members = order[labels == cid].tolist()
        conditions.append((f"cluster-{cid}", members))
        rand_members = []
        counts = {}
        for j in members:
            counts[j // inter] = counts.get(j // inter, 0) + 1
        for layer, c in counts.items():
            pool = np.arange(layer * inter, (layer + 1) * inter)
            rand_members.extend(rng.choice(pool, size=c, replace=False).tolist())
        conditions.append((f"random-c{cid}", rand_members))
    all_members = order.tolist()
    conditions.append((f"all-{args.top_n}", all_members))

    for name, ids in conditions:
        probs, correct = run_condition(ids)
        p_true = probs[np.arange(len(true_idx)), true_idx]
        kls = float(
            (base_probs * (np.log(base_probs + 1e-12) - np.log(probs + 1e-12))).sum(axis=1).mean()
        )
        d = float(p_true.mean() - p_base.mean())
        rows.append(
            {
                "condition": name,
                "n_neurons": len(ids),
                "acc": round(float(correct.mean()), 4),
                "p_true": round(float(p_true.mean()), 4),
                "d_p_true": round(d, 5),
                "kl": round(kls, 5),
            }
        )
        print(
            f"{name:>18s} {len(ids):5d} {correct.mean():7.4f} {p_true.mean():8.4f} {d:+9.5f} {kls:8.5f}"
        )

    out = RESULTS_DIR / "cluster_ablation.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "top_n": args.top_n,
                "threshold": args.corr_threshold,
                "n_clusters": int(len(sizes)),
                "largest_sizes": [int(s) for s in sorted(sizes, reverse=True)[:10]],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
