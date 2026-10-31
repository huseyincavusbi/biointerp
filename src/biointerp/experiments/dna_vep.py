"""DNA variant-effect prediction with ClinVar sequences and k-mer controls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from biointerp.core import RESULTS_DIR, split_by_protein
from biointerp.dna import (
    extract_cett_causal,
    extract_cett_masked,
    get_layers,
    precompute_col_norms,
)
from biointerp.dna_task import token_spans, tokenize_dna, variant_token_index
from biointerp.models import get_spec, load_model
from biointerp.runner import load_dataset_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceBio/Carbon-500M")
    p.add_argument("--model-key", default=None, help="registry key (overrides --model)")
    p.add_argument("--dataset", default="clinvar_vep", help="conf/dataset/<name>.yaml")
    p.add_argument("--data", default=None, help="optional local parquet/CSV override")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _load_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    cfg = load_dataset_config(args.dataset) if args.dataset else {}
    if args.data:
        path = Path(args.data)
    else:
        source = cfg.get("source", "hf")
        if source == "hf":
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(cfg["hf_id"], cfg["file"], repo_type="dataset"))
        else:
            path = Path(cfg["path"])
    if path.suffix == ".parquet":
        return pd.read_parquet(path), cfg
    return pd.read_csv(path), cfg


def _prepare_variants(
    frame: pd.DataFrame,
    cfg: dict,
    tokenizer,
    adapter: str,
    n: int,
    seed: int,
) -> list[dict]:
    ref_col = cfg.get("ref_column", "ref_forward_sequence")
    alt_col = cfg.get("alt_column", "alt_forward_sequence")
    label_col = cfg.get("label_column", "label")
    chrom_col = cfg.get("chrom_column", "chromosome")

    if ref_col not in frame or alt_col not in frame or label_col not in frame:
        raise SystemExit(f"missing columns in VEP frame: {list(frame.columns)}")

    n = min(n, len(frame))
    sample = frame.sample(n=n, random_state=seed)
    variants = []
    for _, row in sample.iterrows():
        ref = str(row[ref_col]).upper()
        alt = str(row[alt_col]).upper()
        if len(ref) != len(alt) or len(ref) < 2:
            continue
        offset = next((i for i, (a, b) in enumerate(zip(ref, alt)) if a != b), None)
        if offset is None:
            continue

        ref_ids, _ = tokenize_dna(ref, tokenizer, adapter)
        alt_ids, _ = tokenize_dna(alt, tokenizer, adapter)
        if len(ref_ids) != len(alt_ids):
            continue
        spans = token_spans(ref, tokenizer, adapter)
        try:
            token_index = variant_token_index(spans, offset)
        except ValueError:
            continue
        ref_id = int(ref_ids[token_index])
        alt_id = int(alt_ids[token_index])
        if ref_id == alt_id:
            continue

        variant = {
            "chrom": str(row.get(chrom_col, len(variants))),
            "offset": offset,
            "ref": ref,
            "alt": alt,
            "label": int(row[label_col]),
            "ref_id": ref_id,
            "alt_id": alt_id,
            "token_index": token_index,
            "ref_ids": ref_ids,
            "alt_ids": alt_ids,
        }
        if adapter == "dnabert":
            mask_id = getattr(tokenizer, "mask_token_id", None)
            if mask_id is None:
                raise SystemExit("DNABERT tokenizer has no mask_token_id")
            masked = list(ref_ids)
            masked[token_index] = int(mask_id)
            variant["masked_ids"] = masked
        variants.append(variant)
    if not variants:
        raise SystemExit("no usable single-nucleotide variants found")
    return variants


def _pad(rows: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(row) for row in rows)
    ids = [row + [pad_id] * (max_len - len(row)) for row in rows]
    mask = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
    )


def _safe_auroc(y, scores):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def main() -> None:
    args = parse_args()
    model_name = args.model_key or args.model
    spec = get_spec(model_name)
    adapter = spec["adapter"]
    frame, cfg = _load_frame(args)
    model, tokenizer, device = load_model(model_name, args.device)
    model.eval()
    layers = list(range(len(get_layers(model, adapter))))
    col_norms = precompute_col_norms(model, adapter, layers)
    variants = _prepare_variants(frame, cfg, tokenizer, adapter, args.n, args.seed)

    print(
        f"model: {model_name} | adapter: {adapter} | device: {device} | "
        f"variants: {len(variants)} | layers: {len(layers)}"
    )

    cett_all, score_all, conf_all, ent_all = [], [], [], []
    chrom_all, offset_all, label_all = [], [], []
    t0 = time.time()
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = 0 if pad_id is None else int(pad_id)

    for start in range(0, len(variants), args.batch_size):
        batch = variants[start : start + args.batch_size]
        if adapter == "carbon":
            input_ids, attention = _pad([row["ref_ids"] for row in batch], pad_id)
            positions = torch.tensor(
                [[max(row["token_index"] - 1, 0)] for row in batch], dtype=torch.long
            )
            tokens = {"input_ids": input_ids.to(device), "attention_mask": attention.to(device)}
            positions = positions.to(device)
            res = extract_cett_causal(model, tokens, positions, layers, col_norms, "carbon")
        else:
            input_ids, attention = _pad([row["masked_ids"] for row in batch], pad_id)
            positions = torch.tensor([[row["token_index"]] for row in batch], dtype=torch.long)
            tokens = {"input_ids": input_ids.to(device), "attention_mask": attention.to(device)}
            positions = positions.to(device)
            res = extract_cett_masked(model, tokens, positions, layers, col_norms, "dnabert")

        logits = res["logits"].float()
        logp = torch.log_softmax(logits, dim=-1)
        rows = torch.arange(len(batch))
        ref_ids = torch.tensor([row["ref_id"] for row in batch])
        alt_ids = torch.tensor([row["alt_id"] for row in batch])
        scores = (logp[rows, alt_ids] - logp[rows, ref_ids]).numpy()
        probs = torch.softmax(logits, dim=-1).numpy()

        cett_all.append(res["cett"].numpy())
        score_all.append(scores)
        conf_all.append(probs.max(axis=1))
        ent_all.append(-(probs * np.log(probs + 1e-12)).sum(axis=1))
        chrom_all.extend(row["chrom"] for row in batch)
        offset_all.extend(row["offset"] for row in batch)
        label_all.extend(row["label"] for row in batch)
        print(
            f"  scored {min(start + len(batch), len(variants))}/{len(variants)} | {time.time() - t0:.0f}s"
        )

    cett = np.concatenate(cett_all)
    scores = np.concatenate(score_all)
    confidence = np.concatenate(conf_all)
    entropy = np.concatenate(ent_all)
    labels = np.array(label_all)
    chroms = np.array(chrom_all)
    offsets = np.array(offset_all)
    pred = (scores > 0).astype(int)
    correct = pred == labels

    train, test = split_by_protein(chroms, test_frac=args.test_frac, seed=args.seed)
    auroc_score = _safe_auroc(labels[test], scores[test])

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False)
    # Align the vectorizer rows to the original train/test masks.
    train_rows = [i for i in np.flatnonzero(train)]
    test_rows = [i for i in np.flatnonzero(test)]
    x_kmer_train = vectorizer.fit_transform([variants[i]["ref"] for i in train_rows])
    x_kmer_test = vectorizer.transform([variants[i]["ref"] for i in test_rows])
    auroc_kmer = None
    if len(np.unique(labels[train])) > 1 and len(np.unique(labels[test])) > 1:
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
        clf.fit(x_kmer_train, labels[train])
        auroc_kmer = _safe_auroc(labels[test], clf.predict_proba(x_kmer_test)[:, 1])

    error = (~correct).astype(int)
    auroc_probe_error = None
    if len(np.unique(error[train])) > 1 and len(np.unique(error[test])) > 1:
        scaler = StandardScaler()
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
        clf.fit(scaler.fit_transform(cett[train]), error[train])
        auroc_probe_error = _safe_auroc(
            error[test], clf.predict_proba(scaler.transform(cett[test]))[:, 1]
        )
    auroc_confidence_error = None
    if len(np.unique(error[test])) > 1:
        auroc_confidence_error = _safe_auroc(error[test], -confidence[test])

    stem = model_name.replace("/", "_").replace(":", "_")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"dna_vep_{stem}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        cett=cett,
        score=scores,
        label=labels,
        confidence=confidence,
        entropy=entropy,
        chrom=chroms,
        offset=offsets,
        correct=correct,
        model=np.array(model_name),
        adapter=np.array(adapter),
        n_layers=np.array(len(layers)),
        intermediate_dim=np.array(cett.shape[1] // len(layers)),
    )

    records_path = out_path.with_suffix(".predictions.jsonl")
    with records_path.open("w") as fh:
        for i, row in enumerate(variants):
            fh.write(
                json.dumps(
                    {
                        "chrom": str(chroms[i]),
                        "offset": int(offsets[i]),
                        "ref": row["ref"][row["offset"]],
                        "alt": row["alt"][row["offset"]],
                        "label": int(labels[i]),
                        "score": round(float(scores[i]), 6),
                        "pred": int(pred[i]),
                        "correct": bool(correct[i]),
                        "confidence": round(float(confidence[i]), 6),
                    }
                )
                + "\n"
            )

    summary = {
        "model": model_name,
        "adapter": adapter,
        "modality": "dna",
        "tokenizer": spec.get("tokenizer"),
        "task": "variant_effect",
        "dataset": args.dataset if not args.data else str(args.data),
        "config": {
            "n": len(variants),
            "batch_size": args.batch_size,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "device": device,
        },
        "results": {
            "n": int(len(labels)),
            "n_test": int(test.sum()),
            "sign_accuracy": round(float(correct.mean()), 4),
            "auroc_score": None if auroc_score is None else round(auroc_score, 4),
            "auroc_kmer": None if auroc_kmer is None else round(auroc_kmer, 4),
            "auroc_probe_error": None if auroc_probe_error is None else round(auroc_probe_error, 4),
            "auroc_confidence_error": None
            if auroc_confidence_error is None
            else round(auroc_confidence_error, 4),
            "seconds": round(time.time() - t0, 1),
        },
        "features": {
            "total": int(cett.shape[1]),
            "n_layers": len(layers),
            "intermediate_dim": int(cett.shape[1] // len(layers)),
        },
        "files": {"npz": str(out_path), "predictions": str(records_path)},
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nsign accuracy: {float(correct.mean()):.4f}")
    print(f"AUROC score: {auroc_score} | k-mer: {auroc_kmer}")
    print(f"AUROC probe error: {auroc_probe_error} | confidence: {auroc_confidence_error}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
