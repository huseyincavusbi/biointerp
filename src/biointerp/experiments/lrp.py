"""LRP attribution on ESM-2 (CoTLab lrp rules adapted).

Layer-wise Relevance Propagation rules (GELU identity rule, LayerNorm detached-scale rule)
installed as custom autograd modules; relevance from the model's predicted-residue logit
back to (a) input residue positions and (b) FFN neurons at a chosen layer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, split_by_protein, univariate_auroc
from biointerp.esm import AA_ALPHABET, aa_token_ids, get_transformer_layers, load_model
from biointerp.task import fetch_swissprot, make_masked_batch


class GELUIdentityRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return F.gelu(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GELUIdentityModule(nn.Module):
    def forward(self, x):
        return GELUIdentityRule.apply(x)


class LayerNormDetachRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)
        xhat = (x - mean) / std
        ctx.save_for_backward(xhat, std, weight)
        return xhat * weight + bias

    @staticmethod
    def backward(ctx, grad_output):
        xhat, std, weight = ctx.saved_tensors
        grad_x = grad_output * weight / std
        reduce_dims = tuple(range(grad_output.dim() - 1))
        grad_weight = (grad_output * xhat).sum(dim=reduce_dims)
        grad_bias = grad_output.sum(dim=reduce_dims)
        return grad_x, grad_weight, grad_bias, None


class LayerNormDetachModule(nn.Module):
    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps

    def forward(self, x):
        return LayerNormDetachRule.apply(x, self.weight, self.bias, self.eps)


def install_lrp_rules(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, GELUIdentityModule())
        elif isinstance(child, nn.LayerNorm):
            setattr(module, name, LayerNormDetachModule(child))
        else:
            install_lrp_rules(child)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--layer", type=int, default=30)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)
    y = (~data["correct"]).astype(int)
    protein = data["protein"]
    inter = int(data["intermediate_dim"])
    tr, _ = split_by_protein(protein, seed=args.seed)
    au_layer = univariate_auroc(data["cett"][tr], y[tr])[
        args.layer * inter : (args.layer + 1) * inter
    ]

    sequences = fetch_swissprot(2000, seed=args.seed)[: args.n]
    model, tokenizer, device = load_model(args.model, args.device)
    install_lrp_rules(model)
    blocks = get_transformer_layers(model)
    aa_ids = aa_token_ids(tokenizer)

    z_captured = {}

    def hook(module, inputs, output):
        z_captured["z"] = output
        return output

    handle = blocks[args.layer].intermediate.register_forward_hook(hook)

    masked_share, ent_share, dist_profile = [], [], []
    neuron_rel_correct, neuron_rel_incorrect = [], []
    correct_all = []
    t0 = time.time()
    for idx, seq in enumerate(sequences):
        tokens, positions, true_aa = make_masked_batch(
            [seq], tokenizer, masks_per_seq=1, seed=args.seed + idx
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        pos = positions.to(device)
        out = model(**tokens, output_hidden_states=True)
        logits = out.logits[0, pos[0, 0]]
        pred = int(logits[aa_ids].argmax().item())
        true_idx = AA_ALPHABET.index(true_aa[0][0])
        correct_all.append(pred == true_idx)

        loss = logits[aa_ids[pred]]
        z = z_captured["z"]
        emb = out.hidden_states[0]
        grads = torch.autograd.grad(loss, [z, emb])
        rel_z = (z * grads[0]).squeeze(0)[pos[0, 0]]
        rel_pos = (emb * grads[1]).sum(dim=-1).squeeze(0)

        rel_pos = rel_pos.abs()
        total = rel_pos.sum().item() + 1e-12
        masked_share.append(rel_pos[pos[0, 0]].item() / total)
        p = rel_pos / total
        ent_share.append(float(-(p * (p + 1e-12).log()).sum().item()))
        L = rel_pos.shape[0]
        profile = np.zeros(51)
        for d in range(51):
            lo, hi = max(0, pos[0, 0].item() - d), min(L, pos[0, 0].item() + d + 1)
            profile[d] = rel_pos[lo:hi].sum().item() / total
        dist_profile.append(profile)
        if correct_all[-1]:
            neuron_rel_correct.append(rel_z.detach().cpu().numpy())
        else:
            neuron_rel_incorrect.append(rel_z.detach().cpu().numpy())
        if (idx + 1) % 20 == 0:
            print(f"sample {idx + 1}/{len(sequences)} | {time.time() - t0:.0f}s")
    handle.remove()

    y_arr = (~np.array(correct_all)).astype(int)
    print(f"\nsamples: {len(y_arr)} | incorrect {int(y_arr.sum())}\n")

    ms = np.array(masked_share)
    print(
        f"masked-position relevance share: correct {ms[y_arr == 0].mean():.4f} | incorrect {ms[y_arr == 1].mean():.4f}"
    )
    es = np.array(ent_share)
    print(
        f"relevance entropy over positions: correct {es[y_arr == 0].mean():.4f} | incorrect {es[y_arr == 1].mean():.4f}"
    )

    prof = np.stack(dist_profile).mean(axis=0)
    print(
        f"distance profile (share within ±d of mask): d=0 {prof[0]:.4f} | d=1 {prof[1]:.4f} | d=5 {prof[5]:.4f} | d=20 {prof[20]:.4f} | d=50 {prof[50]:.4f}"
    )

    rel_inc = (
        np.stack(neuron_rel_incorrect).mean(axis=0) if neuron_rel_incorrect else np.zeros(inter)
    )
    rho, _ = spearmanr(rel_inc, au_layer)
    print(
        f"\nmean neuron relevance (incorrect) vs error-AUROC at layer {args.layer}: Spearman {rho:.4f}"
    )
    top_rel = set(np.argsort(rel_inc)[::-1][:100].tolist())
    top_err = set(np.argsort(au_layer)[::-1][:100].tolist())
    jac = len(top_rel & top_err) / len(top_rel | top_err)
    print(f"top-100 overlap (LRP relevance vs error neurons): Jaccard {jac:.4f}")

    out = RESULTS_DIR / "lrp.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "layer": args.layer,
                "n": int(len(y_arr)),
                "masked_relevance_share_correct": round(float(ms[y_arr == 0].mean()), 4),
                "masked_relevance_share_incorrect": round(float(ms[y_arr == 1].mean()), 4),
                "relevance_entropy_correct": round(float(es[y_arr == 0].mean()), 4),
                "relevance_entropy_incorrect": round(float(es[y_arr == 1].mean()), 4),
                "distance_profile": [round(float(v), 4) for v in prof[:51]],
                "neuron_relevance_vs_error_auroc_spearman": round(float(rho), 4),
                "top100_jaccard": round(jac, 4),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
