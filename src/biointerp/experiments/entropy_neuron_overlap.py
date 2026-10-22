"""Entropy-neuron overlap on ESM-2 (CoTLab entropy_neuron_overlap adapted).

Entropy neurons = high-L2-norm columns of the FFN output projection (per layer).
Question: are our error neurons just high-norm neurons?
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
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--percentile", type=float, default=99.0, help="norm percentile per layer")
    p.add_argument("--ks", default="25,100,500")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)
    x = data["cett"]
    y = (~data["correct"]).astype(int)
    protein = data["protein"]
    n_layers = int(data["n_layers"])
    inter = int(data["intermediate_dim"])

    tr, te = split_by_protein(protein, seed=args.seed)
    au_tr = univariate_auroc(x[tr], y[tr])
    order = np.argsort(au_tr)[::-1]

    model, _, _ = load_model(args.model, device="cpu")
    blocks = get_transformer_layers(model)
    norms = np.empty(x.shape[1], dtype=np.float64)
    for layer in range(n_layers):
        weight = get_ffn_output_proj(blocks[layer]).weight.detach().float()
        norms[layer * inter : (layer + 1) * inter] = np.linalg.norm(weight.cpu().numpy(), axis=0)
    del model

    entropy_mask = np.zeros(x.shape[1], dtype=bool)
    layer_spearman = {}
    for layer in range(n_layers):
        sl = slice(layer * inter, (layer + 1) * inter)
        threshold = np.percentile(norms[sl], args.percentile)
        entropy_mask[sl] = norms[sl] >= threshold
        rho, _ = spearmanr(norms[sl], au_tr[sl])
        layer_spearman[layer] = round(float(rho), 4)

    print(f"model: {args.model}")
    print(
        f"entropy neurons (top {100 - args.percentile:.0f}% norm per layer): {entropy_mask.sum()}"
    )
    print(f"norm range: {norms.min():.4f} - {norms.max():.4f}")
    print()

    results = {
        "model": args.model,
        "percentile": args.percentile,
        "n_entropy_neurons": int(entropy_mask.sum()),
        "norm_min": round(float(norms.min()), 4),
        "norm_max": round(float(norms.max()), 4),
        "layer_spearman_norm_vs_auroc": layer_spearman,
    }
    print(f"{'k':>6s} {'Jaccard':>9s} {'overlap':>8s} {'expected':>9s} {'enrichment':>11s}")
    ks = [int(k) for k in args.ks.split(",")]
    jaccards = {}
    for k in ks:
        err = set(order[:k].tolist())
        ent = set(np.where(entropy_mask)[0].tolist())
        inter_set = err & ent
        union = err | ent
        jaccard = len(inter_set) / len(union)
        expected = k * len(ent) / x.shape[1]
        enrichment = len(inter_set) / expected if expected > 0 else float("nan")
        jaccards[k] = {
            "jaccard": round(jaccard, 4),
            "overlap": len(inter_set),
            "expected": round(expected, 1),
            "enrichment": round(enrichment, 2),
        }
        print(f"{k:6d} {jaccard:9.4f} {len(inter_set):8d} {expected:9.1f} {enrichment:11.2f}")

    global_rho, _ = spearmanr(norms, au_tr)
    print()
    print(f"global Spearman(norm, error-AUROC): {global_rho:.4f}")
    results["jaccard_by_k"] = jaccards
    results["global_spearman_norm_vs_auroc"] = round(float(global_rho), 4)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "entropy_neuron_overlap.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
