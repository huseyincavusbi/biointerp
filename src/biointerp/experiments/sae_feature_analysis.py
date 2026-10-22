"""SAE feature analysis on ESM-2 (CoTLab sae_feature_analysis adapted).

Encode masked-position layer-30 activations through the InterPLM SAE; probe feature
activations for misprediction (train-selected, test-validated); compare with error neurons.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR, split_by_protein, univariate_auroc
from biointerp.esm import AA_ALPHABET, aa_token_ids, get_transformer_layers, load_model
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--sae-repo", default="Elana/InterPLM-esm2-650m")
    p.add_argument("--sae-file", default="layer_30/ae_unnormalized.pt")
    p.add_argument("--layer", type=int, default=30)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    from huggingface_hub import hf_hub_download

    sequences = fetch_swissprot(2000, seed=args.seed)[: args.n]
    model, tokenizer, device = load_model(args.model, args.device)
    len(get_transformer_layers(model)) + 1
    hidden_idx = args.layer + 1
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    acts: list[np.ndarray] = []
    correct_all: list[bool] = []
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
        h = out.hidden_states[hidden_idx][b_idx, pos].squeeze(1).float().cpu().numpy()
        acts.append(h)
        if (bi + 1) % 8 == 0 or bi + 1 == n_batches:
            print(f"batch {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    a = np.concatenate(acts)
    y = (~np.array(correct_all)).astype(int)

    state = torch.load(
        hf_hub_download(args.sae_repo, args.sae_file), map_location="cpu", weights_only=True
    )
    w_enc = state["encoder.weight"].float().numpy()
    b_enc = state["encoder.bias"].float().numpy()
    b_dec = state["bias"].float().numpy()
    features = np.maximum((a - b_dec) @ w_enc.T + b_enc, 0.0)
    print(f"\nactivations: {a.shape} | SAE features: {features.shape}\n")

    tr, te = split_by_protein(np.arange(len(y)), seed=args.seed)
    au_tr = univariate_auroc(features[tr], y[tr])
    order = np.argsort(au_tr)[::-1]
    n_feat = features.shape[1]
    print(
        f"features with train AUROC >0.60: {(au_tr > 0.60).sum()}/{n_feat} ({(au_tr > 0.60).mean():.4f})"
    )

    print(f"\n{'rank':>4s} {'feature':>8s} {'train AUROC':>11s} {'test AUROC':>10s}")
    top = []
    for rank, f in enumerate(order[:10]):
        test_au = float(roc_auc_score(y[te], features[te, f]))
        top.append(
            {
                "feature": int(f),
                "train_auroc": round(float(au_tr[f]), 4),
                "test_auroc": round(test_au, 4),
            }
        )
        print(f"{rank + 1:4d} {f:8d} {au_tr[f]:11.4f} {test_au:10.4f}")

    rng = np.random.default_rng(args.seed)
    rand_feats = rng.choice(n_feat, size=10, replace=False)
    rand_test = [float(roc_auc_score(y[te], features[te, f])) for f in rand_feats]
    print(
        f"\nrandom-feature test AUROC (10 draws): mean {np.mean(rand_test):.4f} | max {np.max(rand_test):.4f}"
    )

    out = RESULTS_DIR / "sae_feature_analysis.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "sae": f"{args.sae_repo}/{args.sae_file}",
                "layer": args.layer,
                "n": int(len(y)),
                "n_features": int(n_feat),
                "features_train_auroc_gt_060": int((au_tr > 0.60).sum()),
                "top_features": top,
                "random_feature_test_auroc": {
                    "mean": round(float(np.mean(rand_test)), 4),
                    "max": round(float(np.max(rand_test)), 4),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
