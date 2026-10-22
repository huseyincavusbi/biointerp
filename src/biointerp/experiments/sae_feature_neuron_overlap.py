"""SAE-feature/neuron overlap on ESM-2 (CoTLab sae_feature_neuron_overlap adapted).

For each InterPLM SAE feature (layer 30), find the FFN neurons contributing most to its
decoder direction; measure overlap/enrichment with layer-30 error neurons. Random-direction
baseline included.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, split_by_protein, univariate_auroc
from biointerp.esm import get_ffn_output_proj, get_transformer_layers, load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--sae-repo", default="Elana/InterPLM-esm2-650m")
    p.add_argument("--sae-file", default="layer_30/ae_unnormalized.pt")
    p.add_argument("--layer", type=int, default=30)
    p.add_argument("--top-n", type=int, default=25, help="top neurons per SAE feature")
    p.add_argument("--ks", default="25,100")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main() -> None:
    args = parse_args()
    from huggingface_hub import hf_hub_download

    data = np.load(args.npz)
    y = (~data["correct"]).astype(int)
    protein = data["protein"]
    inter = int(data["intermediate_dim"])
    tr, _ = split_by_protein(protein, seed=args.seed)
    au_tr = univariate_auroc(data["cett"][tr], y[tr])
    au_layer = au_tr[args.layer * inter : (args.layer + 1) * inter]

    state = torch.load(
        hf_hub_download(args.sae_repo, args.sae_file), map_location="cpu", weights_only=True
    )
    w_dec = state["decoder.weight"].float().numpy()
    n_features = w_dec.shape[1]

    model, _, _ = load_model(args.model, device="cpu")
    blocks = get_transformer_layers(model)
    w_out = get_ffn_output_proj(blocks[args.layer]).weight.detach().float().numpy()
    del model

    contrib = w_dec.T @ w_out
    top_n = min(args.top_n, contrib.shape[1])

    print(f"SAE: {args.sae_repo} {args.sae_file} | features: {n_features} | layer: {args.layer}")
    print(f"neurons per layer: {contrib.shape[1]} | top-n per feature: {top_n}\n")

    results = {
        "model": args.model,
        "sae": f"{args.sae_repo}/{args.sae_file}",
        "layer": args.layer,
        "n_features": n_features,
        "top_n": top_n,
    }

    for k in [int(x) for x in args.ks.split(",")]:
        err = set(np.argsort(au_layer)[::-1][:k].tolist())
        top_idx = np.argsort(np.abs(contrib), axis=1)[:, ::-1][:, :top_n]
        jac = np.array([jaccard(set(row.tolist()), err) for row in top_idx])
        expected = top_n * k / contrib.shape[1]
        float(
            jac.mean() * (top_n + k - top_n * k / contrib.shape[1]) / expected
        ) if expected else float("nan")
        frac_overlap = float((jac > 0).mean())
        best = int(jac.argmax())
        print(
            f"error neurons k={k}: mean Jaccard {jac.mean():.5f} | max {jac.max():.4f} (feature {best}) | "
            f"features with overlap {frac_overlap:.4f} | expected overlap/feature {expected:.3f}"
        )
        results[f"k{k}"] = {
            "mean_jaccard": round(float(jac.mean()), 5),
            "max_jaccard": round(float(jac.max()), 4),
            "best_feature": best,
            "frac_features_with_overlap": round(frac_overlap, 4),
            "expected_overlap_per_feature": round(expected, 3),
        }

    rng = np.random.default_rng(args.seed)
    rand_dirs = rng.normal(size=w_dec.shape).astype(np.float32)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=0, keepdims=True)
    contrib_rand = rand_dirs.T @ w_out
    err25 = set(np.argsort(au_layer)[::-1][:25].tolist())
    top_rand = np.argsort(np.abs(contrib_rand), axis=1)[:, ::-1][:, :top_n]
    jac_rand = np.array([jaccard(set(row.tolist()), err25) for row in top_rand])
    print(
        f"\nrandom-direction baseline (k=25): mean Jaccard {jac_rand.mean():.5f} | max {jac_rand.max():.4f}"
    )
    results["random_baseline_k25"] = {
        "mean_jaccard": round(float(jac_rand.mean()), 5),
        "max_jaccard": round(float(jac_rand.max()), 4),
    }

    agg = np.abs(contrib).sum(axis=0)
    rho, _ = spearmanr(agg, au_layer)
    print(f"aggregate neuron contribution vs error-AUROC (layer {args.layer}): Spearman {rho:.4f}")
    results["aggregate_spearman"] = round(float(rho), 4)

    out = RESULTS_DIR / "sae_feature_neuron_overlap.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
