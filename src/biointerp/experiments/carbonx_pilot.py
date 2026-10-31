"""Carbon-X pilot: paired prompts over ClinVar variants.

This runner is intentionally small and resumable: each variant/condition writes
one JSONL record and one feature file, so a long 4-bit run can be restarted
without repeating completed work.
"""

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

from biointerp.carbonx import (
    CONDITIONS,
    build_messages,
    context_token_positions,
    dna_segments,
    dna_token_positions,
    encode_chat,
    parse_answer,
    reduce_attention,
    variant_indices,
)
from biointerp.core import RESULTS_DIR
from biointerp.dna import extract_cett_causal, get_layers, precompute_col_norms
from biointerp.experiments.carbonx_context import parse_layers
from biointerp.models import get_spec, load_model
from biointerp.runner import load_dataset_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", default="carbon_x_4bit")
    p.add_argument("--dataset", default="clinvar_vep", help="conf/dataset/<name>.yaml")
    p.add_argument("--data", default=None, help="optional local parquet/CSV override")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="all", choices=("all", *CONDITIONS))
    p.add_argument("--layers", default="all", help="'all', 'last:N', or indices")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--correct-gene", default="BRCA1")
    p.add_argument("--wrong-gene", default="TP53")
    p.add_argument("--attention", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="output directory")
    return p.parse_args()


def load_variants(
    dataset: str,
    data: str | None,
    n: int,
    seed: int,
) -> tuple[list[dict], dict]:
    cfg = load_dataset_config(dataset) if dataset else {}
    if data:
        path = Path(data)
    else:
        source = cfg.get("source", "hf")
        if source == "hf":
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(cfg["hf_id"], cfg["file"], repo_type="dataset"))
        else:
            path = Path(cfg["path"])

    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    ref_col = cfg.get("ref_column", "ref_forward_sequence")
    alt_col = cfg.get("alt_column", "alt_forward_sequence")
    label_col = cfg.get("label_column", "label")
    chrom_col = cfg.get("chrom_column", "chromosome")
    gene_col = cfg.get("gene_column")

    sample = frame.sample(n=min(n, len(frame)), random_state=seed)
    variants = []
    for i, (_, row) in enumerate(sample.iterrows()):
        ref = str(row[ref_col]).upper()
        alt = str(row[alt_col]).upper()
        if len(ref) != len(alt) or len(ref) < 2:
            continue
        offset = next((j for j, (a, b) in enumerate(zip(ref, alt)) if a != b), None)
        if offset is None:
            continue
        variants.append(
            {
                "index": i,
                "chrom": str(row.get(chrom_col, i)),
                "offset": offset,
                "ref": ref,
                "alt": alt,
                "label": int(row[label_col]),
                "gene": None if gene_col is None else str(row[gene_col]),
            }
        )
    if not variants:
        raise SystemExit("no usable single-nucleotide variants found")
    return variants, cfg


def _read_done(path: Path) -> set[tuple[int, str]]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        done.add((int(record["variant"]), str(record["condition"])))
    return done


def main() -> None:
    args = parse_args()
    conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    variants, _ = load_variants(args.dataset, args.data, args.n, args.seed)

    spec = get_spec(args.model_key)
    started = time.time()
    print(f"loading {args.model_key}...", flush=True)
    model, tokenizer, device = load_model(args.model_key, args.device)
    model.eval()
    if args.attention:
        try:
            model.set_attn_implementation("eager")
        except AttributeError:
            model.config._attn_implementation = "eager"
        print("attention implementation: eager", flush=True)
    print(f"loaded in {time.time() - started:.1f}s on {device}", flush=True)

    n_layers = len(get_layers(model, spec["adapter"]))
    layers = parse_layers(args.layers, n_layers)
    print(f"cett layers: {layers} ({len(layers)}/{n_layers})", flush=True)
    started = time.time()
    col_norms = precompute_col_norms(model, spec["adapter"], layers)
    print(f"column norms ready in {time.time() - started:.1f}s", flush=True)

    stem = args.model_key.replace("/", "_").replace(":", "_")
    out_dir = Path(args.out) if args.out else RESULTS_DIR / f"carbonx_pilot_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = out_dir / "features"
    features_dir.mkdir(exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = _read_done(records_path) if args.resume else set()
    if done:
        print(f"resuming with {len(done)} completed records", flush=True)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = 0 if pad_id is None else int(pad_id)
    generated_records = 0
    with records_path.open("a" if args.resume else "w") as fh:
        for variant in variants:
            gene = variant["gene"] or args.correct_gene
            for condition in conditions:
                key = (variant["index"], condition)
                if key in done:
                    continue
                messages = build_messages(
                    variant["ref"],
                    variant["alt"],
                    condition,
                    correct_gene=gene,
                    wrong_gene=args.wrong_gene,
                )
                encoded = encode_chat(tokenizer, messages)
                input_ids = encoded["input_ids"]
                segments = dna_segments(tokenizer, input_ids)
                ref_token = alt_token = None
                cett = None
                attention_dna = None
                attention_variant = None
                context_tokens = []
                if len(segments) >= 2:
                    ref_token, alt_token = variant_indices(segments, variant["offset"])
                    token_positions = torch.tensor([[ref_token]], dtype=torch.long, device=device)
                    tokens = {k: v.to(device) for k, v in encoded.items()}
                    result = extract_cett_causal(
                        model,
                        tokens,
                        token_positions,
                        layers,
                        col_norms,
                        spec["adapter"],
                        capture_attention=args.attention,
                    )
                    cett = result["cett"][0].numpy()
                    if args.attention:
                        dna_positions = dna_token_positions(segments)
                        context_positions = context_token_positions(tokenizer, input_ids, segments)
                        attention_dna = reduce_attention(
                            result["attentions"], dna_positions, context_positions
                        )
                        attention_variant = reduce_attention(
                            result["attentions"], [ref_token], context_positions
                        )
                        ids = input_ids.tolist()
                        if ids and isinstance(ids[0], list):
                            ids = ids[0]
                        tokens_text = tokenizer.convert_ids_to_tokens(ids)
                        context_tokens = [str(tokens_text[i]) for i in context_positions]

                tokens = {k: v.to(device) for k, v in encoded.items()}
                with torch.no_grad():
                    generated = model.generate(
                        **tokens,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                answer = tokenizer.decode(
                    generated[0][input_ids.shape[1] :], skip_special_tokens=True
                ).strip()
                parsed = parse_answer(answer)
                feature_path = features_dir / f"{variant['index']:04d}_{condition}.npz"
                feature_arrays = {}
                if cett is not None:
                    feature_arrays["cett"] = cett
                if attention_dna is not None:
                    feature_arrays["attention_dna"] = attention_dna
                if attention_variant is not None:
                    feature_arrays["attention_variant"] = attention_variant
                if context_tokens:
                    feature_arrays["context_tokens"] = np.array(context_tokens)
                if feature_arrays:
                    np.savez_compressed(feature_path, **feature_arrays)

                record = {
                    "variant": variant["index"],
                    "chrom": variant["chrom"],
                    "offset": variant["offset"],
                    "ref": variant["ref"][variant["offset"]],
                    "alt": variant["alt"][variant["offset"]],
                    "label": variant["label"],
                    "gene": gene,
                    "condition": condition,
                    "answer": answer,
                    "parsed": parsed,
                    "ref_token_index": ref_token,
                    "alt_token_index": alt_token,
                    "cett_shape": None if cett is None else list(cett.shape),
                    "attention_shape": None if attention_dna is None else list(attention_dna.shape),
                    "features": str(feature_path) if feature_arrays else None,
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                generated_records += 1
                print(
                    f"[{variant['index'] + 1}/{len(variants)}] {condition}: "
                    f"{parsed['label']} | {answer[:80]}",
                    flush=True,
                )

    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    answers = {
        condition: {
            label: sum(
                1
                for record in records
                if record["condition"] == condition and record["parsed"]["label"] == label
            )
            for label in ("pathogenic", "benign", "uncertain", "ambiguous", "unparsed")
        }
        for condition in conditions
    }
    flips = 0
    for variant_id in sorted({record["variant"] for record in records}):
        labels = {
            record["parsed"]["label"]
            for record in records
            if record["variant"] == variant_id
            and record["parsed"]["label"] not in ("unparsed", "ambiguous")
        }
        if len(labels) > 1:
            flips += 1
    summary = {
        "model": args.model_key,
        "dataset": args.dataset if not args.data else str(args.data),
        "config": {
            "n_variants": len(variants),
            "conditions": list(conditions),
            "layers": layers,
            "max_new_tokens": args.max_new_tokens,
            "attention": args.attention,
            "seed": args.seed,
            "device": device,
        },
        "records": len(records),
        "new_records": generated_records,
        "answers": answers,
        "flips": flips,
        "files": {
            "records": str(records_path),
            "features": str(features_dir),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"records: {len(records)} | flips: {flips}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
