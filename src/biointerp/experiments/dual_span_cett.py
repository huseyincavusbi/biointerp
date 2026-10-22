"""Dual-span CETT on ESM-2 (hprobes dual-span adapted).

Compare CETT features pooled over two spans: the masked position (decision point) vs all
other (context) positions. Probe both for misprediction; compare overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR, split_by_protein, univariate_auroc
from biointerp.esm import (
    AA_ALPHABET,
    aa_token_ids,
    get_ffn_output_proj,
    get_transformer_layers,
    load_model,
    precompute_col_norms,
)
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def fit_probe(x_tr, y_tr, x_te, y_te, top_k, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    au = univariate_auroc(x_tr, y_tr)
    idx = np.argsort(au)[::-1][:top_k]
    scaler = StandardScaler()
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(scaler.fit_transform(x_tr[:, idx]), y_tr)
    a = float(roc_auc_score(y_te, clf.predict_proba(scaler.transform(x_te[:, idx]))[:, 1]))
    return a, idx, au


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    sequences = fetch_swissprot(2000, seed=args.seed)[: args.n]
    model, tokenizer, device = load_model(args.model, args.device)
    blocks = get_transformer_layers(model)
    n_layers = len(blocks)
    col_norms = precompute_col_norms(model, list(range(n_layers)))
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    masked_all: list[np.ndarray] = []
    ctx_all: list[np.ndarray] = []
    correct_all: list[bool] = []

    t0 = time.time()
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = sequences[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, true_aa = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=args.seed + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        pos = positions.to(device)
        b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)
        valid = tokens["attention_mask"].float().clone()
        valid.scatter_(1, pos, 0.0)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)

        store_masked = {}
        store_ctx = {}
        handles = []

        def make_hook(layer: int):
            def hook(module, inputs, output):
                z = inputs[0]
                h = output
                h_norm = h.norm(dim=-1, keepdim=True) + 1e-8
                cett = torch.abs(z) * col_norms[layer].to(z.device) / h_norm
                store_masked[layer] = cett[b_idx, pos].squeeze(1).float().cpu()
                ctx = (cett * valid.unsqueeze(-1)).sum(dim=1) / denom
                store_ctx[layer] = ctx.float().cpu()
                return output

            return hook

        for layer in range(n_layers):
            handles.append(
                get_ffn_output_proj(blocks[layer]).register_forward_hook(make_hook(layer))
            )
        try:
            out = model(**tokens)
        finally:
            for handle in handles:
                handle.remove()

        logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
        pred = logits.argmax(dim=-1).cpu().numpy()
        true_flat = [aa for row in true_aa for aa in row]
        true_idx = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        correct_all.extend((pred == true_idx).tolist())

        masked_all.append(
            np.concatenate([store_masked[l].numpy() for l in range(n_layers)], axis=1)
        )
        ctx_all.append(np.concatenate([store_ctx[l].numpy() for l in range(n_layers)], axis=1))
        if (bi + 1) % 4 == 0 or bi + 1 == n_batches:
            print(f"batch {bi + 1}/{n_batches} | {time.time() - t0:.0f}s")

    x_masked = np.concatenate(masked_all)
    x_ctx = np.concatenate(ctx_all)
    y = (~np.array(correct_all)).astype(int)
    tr, te = split_by_protein(np.arange(len(y)), seed=args.seed)
    print(f"\nsamples: {len(y)} | features per span: {x_masked.shape[1]}\n")

    a_m, idx_m, au_m = fit_probe(x_masked[tr], y[tr], x_masked[te], y[te], args.top_k, args.seed)
    a_c, idx_c, au_c = fit_probe(x_ctx[tr], y[tr], x_ctx[te], y[te], args.top_k, args.seed)
    x_comb = np.concatenate([x_masked, x_ctx], axis=1)
    a_both, _, _ = fit_probe(x_comb[tr], y[tr], x_comb[te], y[te], args.top_k, args.seed)

    print(f"probe (top-{args.top_k}) test AUROC:")
    print(f"  masked-position span: {a_m:.4f}")
    print(f"  context span:         {a_c:.4f}")
    print(f"  combined:             {a_both:.4f}")

    jac = len(set(idx_m.tolist()) & set(idx_c.tolist())) / len(
        set(idx_m.tolist()) | set(idx_c.tolist())
    )
    rho, _ = spearmanr(au_m, au_c)
    print(f"\ntop-{args.top_k} overlap (masked vs context): Jaccard {jac:.4f}")
    print(f"per-neuron AUROC correlation between spans: Spearman {rho:.4f}")

    out = RESULTS_DIR / "dual_span_cett.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": int(len(y)),
                "top_k": args.top_k,
                "auroc_masked": round(a_m, 4),
                "auroc_context": round(a_c, 4),
                "auroc_combined": round(a_both, 4),
                "jaccard_topk": round(jac, 4),
                "spearman_neuron_auroc": round(float(rho), 4),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
