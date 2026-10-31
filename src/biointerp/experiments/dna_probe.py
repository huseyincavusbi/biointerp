"""Per-layer probing of masked-token correctness in a DNA language model."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from biointerp.core import RESULTS_DIR, load_sequences, split_by_protein
from biointerp.dna import extract_cett_masked, get_layers, precompute_col_norms
from biointerp.dna_task import masked_dna_batch
from biointerp.models import get_spec, load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="zhihan1996/DNABERT-2-117M")
    p.add_argument("--model-key", default=None, help="registry key (overrides --model)")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--masks", type=int, default=1, help="masks per sequence")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--length", type=int, default=60, help="synthetic sequence length")
    p.add_argument("--fasta", default=None, help="optional FASTA of DNA sequences")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _load_sequences(args: argparse.Namespace) -> list[str]:
    if args.fasta:
        sequences = load_sequences(args.fasta)[: args.n]
        if not sequences:
            raise SystemExit(f"no sequences in {args.fasta}")
        return sequences
    rng = random.Random(args.seed)
    return ["".join(rng.choice("ACGT") for _ in range(args.length)) for _ in range(args.n)]


def _auroc(x_train, y_train, x_test, y_test, seed: int):
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    if float(np.asarray(x_train).std()) < 1e-9:
        return None
    scaler = StandardScaler()
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(scaler.fit_transform(x_train), y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(scaler.transform(x_test))[:, 1]))


def main() -> None:
    args = parse_args()
    model_name = args.model_key or args.model
    spec = get_spec(model_name)
    adapter = spec["adapter"]
    sequences = _load_sequences(args)
    model, tokenizer, device = load_model(model_name, args.device)
    model.eval()
    layers = list(range(len(get_layers(model, adapter))))
    col_norms = precompute_col_norms(model, adapter, layers)

    print(
        f"model: {model_name} | adapter: {adapter} | device: {device} | "
        f"layers: {len(layers)} | sequences: {len(sequences)}"
    )

    cett_all: list[np.ndarray] = []
    hidden_all: list[list[np.ndarray]] | None = None
    correct_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []
    ent_all: list[np.ndarray] = []
    seq_all: list[np.ndarray] = []
    pos_all: list[np.ndarray] = []
    true_ids: list[int] = []
    pred_ids: list[int] = []

    t0 = time.time()
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = sequences[bi * args.batch_size : (bi + 1) * args.batch_size]
        tokens, positions, true, _ = masked_dna_batch(
            batch,
            tokenizer,
            adapter,
            masks_per_seq=args.masks,
            seed=args.seed + bi,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        positions = positions.to(device)
        res = extract_cett_masked(
            model, tokens, positions, layers, col_norms, adapter, capture_hidden=True
        )
        probs = torch.softmax(res["logits"].float(), dim=-1).numpy()
        pred = probs.argmax(axis=1)
        true_flat = true.reshape(-1).numpy()

        cett_all.append(res["cett"].numpy())
        if hidden_all is None:
            hidden_all = [[] for _ in res["hidden_states"]]
        for i, hidden in enumerate(res["hidden_states"]):
            hidden_all[i].append(hidden.numpy())
        correct_all.append(pred == true_flat)
        conf_all.append(probs.max(axis=1))
        ent_all.append(-(probs * np.log(probs + 1e-12)).sum(axis=1))
        seq_all.append(
            np.repeat(
                np.arange(bi * args.batch_size, bi * args.batch_size + len(batch)),
                positions.shape[1],
            )
        )
        pos_all.append(positions.reshape(-1).cpu().numpy())
        true_ids.extend(true_flat.tolist())
        pred_ids.extend(pred.tolist())

        if (bi + 1) % 5 == 0 or bi + 1 == n_batches:
            print(f"  batch {bi + 1}/{n_batches} | {time.time() - t0:.1f}s")

    cett = np.concatenate(cett_all)
    hidden = [np.concatenate(blocks) for blocks in hidden_all or []]
    correct = np.concatenate(correct_all)
    confidence = np.concatenate(conf_all)
    entropy = np.concatenate(ent_all)
    seq_ids = np.concatenate(seq_all)
    positions = np.concatenate(pos_all)
    intermediate_dim = int(cett.shape[1] // len(layers))
    cett_by_layer = cett.reshape(len(correct), len(layers), intermediate_dim)

    y = (~correct).astype(int)
    train, test = split_by_protein(seq_ids, test_frac=args.test_frac, seed=args.seed)

    rows = []
    for layer in range(len(hidden)):
        hidden_auroc = _auroc(
            hidden[layer][train], y[train], hidden[layer][test], y[test], args.seed
        )
        cett_auroc = None
        if layer < len(layers):
            cett_auroc = _auroc(
                cett_by_layer[train, layer],
                y[train],
                cett_by_layer[test, layer],
                y[test],
                args.seed,
            )
        rows.append(
            {
                "layer": layer,
                "hidden_auroc": None if hidden_auroc is None else round(hidden_auroc, 4),
                "cett_auroc": None if cett_auroc is None else round(cett_auroc, 4),
            }
        )

    valid = [
        row for row in rows if row["hidden_auroc"] is not None or row["cett_auroc"] is not None
    ]
    if valid:
        best = max(
            valid,
            key=lambda row: max(
                value for value in (row["hidden_auroc"], row["cett_auroc"]) if value is not None
            ),
        )
        best_layer = best["layer"]
        best_auroc = max(
            value for value in (best["hidden_auroc"], best["cett_auroc"]) if value is not None
        )
    else:
        best_layer = None
        best_auroc = None

    stem = model_name.replace("/", "_").replace(":", "_")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"dna_probe_{stem}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        cett=cett,
        correct=correct,
        confidence=confidence,
        entropy=entropy,
        sequence=seq_ids,
        position=positions,
        true_id=np.array(true_ids),
        pred_id=np.array(pred_ids),
        model=np.array(model_name),
        adapter=np.array(adapter),
        layers=np.array(layers),
        n_layers=np.array(len(layers)),
        intermediate_dim=np.array(intermediate_dim),
    )

    true_tokens = tokenizer.convert_ids_to_tokens(true_ids)
    pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids)
    records_path = out_path.with_suffix(".predictions.jsonl")
    with records_path.open("w") as fh:
        for i in range(len(true_ids)):
            fh.write(
                json.dumps(
                    {
                        "sequence": int(seq_ids[i]),
                        "position": int(positions[i]),
                        "true_token": true_tokens[i],
                        "pred_token": pred_tokens[i],
                        "correct": bool(correct[i]),
                        "confidence": round(float(confidence[i]), 4),
                        "entropy": round(float(entropy[i]), 4),
                    }
                )
                + "\n"
            )

    summary = {
        "model": model_name,
        "adapter": adapter,
        "modality": "dna",
        "tokenizer": spec.get("tokenizer"),
        "task": "masked_token",
        "config": {
            "n": len(sequences),
            "masks": args.masks,
            "batch_size": args.batch_size,
            "length": args.length,
            "fasta": args.fasta,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "device": device,
        },
        "results": {
            "n": int(len(correct)),
            "token_accuracy": round(float(correct.mean()), 4),
            "mean_confidence": round(float(confidence.mean()), 4),
            "mean_entropy": round(float(entropy.mean()), 4),
            "seconds": round(time.time() - t0, 1),
            "best_layer": best_layer,
            "best_auroc": None if best_auroc is None else round(float(best_auroc), 4),
        },
        "features": {
            "total": int(cett.shape[1]),
            "n_layers": len(layers),
            "intermediate_dim": intermediate_dim,
        },
        "rows": rows,
        "files": {"npz": str(out_path), "predictions": str(records_path)},
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\ntoken accuracy: {float(correct.mean()):.4f}")
    print(f"best layer: {best_layer} AUROC {best_auroc}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
