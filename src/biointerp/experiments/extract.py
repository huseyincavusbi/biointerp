"""Extract CETT features for the masked-residue task -> .npz

Usage:
    python extract.py --model facebook/esm2_t33_650M_UR50D --n 100 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from biointerp.core import PILOT_RESULTS
from biointerp.esm import (
    AA_ALPHABET,
    aa_token_ids,
    extract_cett_batch,
    get_transformer_layers,
    load_model,
    precompute_col_norms,
)
from biointerp.task import fetch_swissprot, make_masked_batch

OUT_DIR = PILOT_RESULTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--model-key", default=None, help="registry key (overrides --model)")
    p.add_argument("--dtype", default=None, help="bf16 for large models on small GPUs")
    p.add_argument("--n", type=int, default=100, help="number of sequences")
    p.add_argument("--masks", type=int, default=1, help="masks per sequence")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-len", type=int, default=50)
    p.add_argument("--max-len", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_key:
        from biointerp.models import resolve_id

        args.model = resolve_id(args.model_key)
    sequences = fetch_swissprot(args.n, min_len=args.min_len, max_len=args.max_len, seed=args.seed)
    print(
        f"sequences: {len(sequences)} (len {min(map(len, sequences))}-{max(map(len, sequences))})"
    )

    model, tokenizer, device = load_model(args.model, args.device, args.dtype)
    layers: List[int] = list(range(len(get_transformer_layers(model))))
    intermediate_dim = model.config.intermediate_size
    print(
        f"model: {args.model} | device: {device} | layers: {len(layers)} | intermediate: {intermediate_dim}"
    )

    col_norms = precompute_col_norms(model, layers)
    aa_ids = aa_token_ids(tokenizer)
    aa_chars = np.array(list(AA_ALPHABET))

    cett_all: List[np.ndarray] = []
    correct_all: List[np.ndarray] = []
    conf_all: List[np.ndarray] = []
    ent_all: List[np.ndarray] = []
    protein_all: List[np.ndarray] = []
    pos_all: List[np.ndarray] = []
    true_all: List[str] = []
    pred_all: List[str] = []

    t0 = time.time()
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = sequences[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, true_aa = make_masked_batch(
            batch, tokenizer, masks_per_seq=args.masks, seed=args.seed + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        positions = positions.to(device)
        res = extract_cett_batch(model, tokens, positions, layers, col_norms, aa_ids)

        probs = res["probs"].numpy()
        pred_idx = probs.argmax(axis=1)
        true_flat = [aa for row in true_aa for aa in row]
        true_idx = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        correct = pred_idx == true_idx
        confidence = probs.max(axis=1)
        entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)

        b, m = positions.shape
        cett_all.append(res["cett"].numpy())
        correct_all.append(correct)
        conf_all.append(confidence)
        ent_all.append(entropy)
        protein_all.append(np.repeat(np.arange(bi * args.batch_size, bi * args.batch_size + b), m))
        pos_all.append(positions.cpu().numpy().reshape(-1))
        true_all.extend(true_flat)
        pred_all.extend(aa_chars[pred_idx].tolist())

        if (bi + 1) % 5 == 0 or bi + 1 == n_batches:
            elapsed = time.time() - t0
            print(
                f"  batch {bi + 1}/{n_batches} | {elapsed:.1f}s | {elapsed / (bi + 1):.2f}s/batch"
            )

    out_path = (
        Path(args.out)
        if args.out
        else (OUT_DIR / f"{Path(args.model).name}_n{args.n}_m{args.masks}_seed{args.seed}.npz")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct_cat = np.concatenate(correct_all)
    conf_cat = np.concatenate(conf_all)
    ent_cat = np.concatenate(ent_all)
    protein_cat = np.concatenate(protein_all)
    pos_cat = np.concatenate(pos_all)
    cett_cat = np.concatenate(cett_all)

    np.savez_compressed(
        out_path,
        cett=cett_cat,
        correct=correct_cat,
        confidence=conf_cat,
        entropy=ent_cat,
        protein=protein_cat,
        position=pos_cat,
        true_aa=np.array(true_all),
        pred_aa=np.array(pred_all),
        model=np.array(args.model),
        n_layers=np.array(len(layers)),
        intermediate_dim=np.array(intermediate_dim),
        layers=np.array(layers),
        masks_per_seq=np.array(args.masks),
    )

    records_path = out_path.with_suffix(".predictions.jsonl")
    with records_path.open("w") as fh:
        for i in range(len(true_all)):
            fh.write(
                json.dumps(
                    {
                        "protein": int(protein_cat[i]),
                        "pos": int(pos_cat[i]),
                        "true_aa": true_all[i],
                        "pred_aa": pred_all[i],
                        "correct": bool(correct_cat[i]),
                        "confidence": round(float(conf_cat[i]), 4),
                        "entropy": round(float(ent_cat[i]), 4),
                    }
                )
                + "\n"
            )

    n_samples = int(len(correct_cat))
    acc = float(correct_cat.mean())
    elapsed = time.time() - t0
    summary = {
        "model": args.model,
        "config": {
            "n_sequences": len(sequences),
            "masks_per_seq": args.masks,
            "batch_size": args.batch_size,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "seed": args.seed,
            "device": device,
        },
        "results": {
            "n_samples": n_samples,
            "masked_residue_accuracy": round(acc, 4),
            "mean_confidence": round(float(conf_cat.mean()), 4),
            "mean_entropy": round(float(ent_cat.mean()), 4),
            "extraction_seconds": round(elapsed, 1),
        },
        "features": {
            "total": int(cett_cat.shape[1]),
            "n_layers": len(layers),
            "intermediate_dim": intermediate_dim,
        },
        "files": {
            "npz": str(out_path),
            "predictions": str(records_path),
        },
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nsamples: {n_samples} | masked-residue accuracy: {acc:.4f}")
    print(f"features: {cett_cat.shape[1]} ({len(layers)} layers x {intermediate_dim})")
    print(f"saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"saved: {records_path} ({n_samples} records)")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
