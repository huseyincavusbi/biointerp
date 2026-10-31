"""Multi-variant hidden-state patching sweep for Carbon-X."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from biointerp.carbonx import CONDITIONS, parse_answer
from biointerp.core import RESULTS_DIR
from biointerp.experiments.carbonx_patch import run_patch
from biointerp.experiments.carbonx_pilot import load_variants
from biointerp.models import get_spec, load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", default="carbon_x_4bit")
    p.add_argument("--dataset", default="clinvar_vep")
    p.add_argument("--data", default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", default="correct_gene", choices=CONDITIONS)
    p.add_argument("--target", default="dna_only", choices=CONDITIONS)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--correct-gene", default="BRCA1")
    p.add_argument("--wrong-gene", default="TP53")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(json.loads(line)["variant"]) for line in path.read_text().splitlines() if line.strip()
    }


def main() -> None:
    args = parse_args()
    variants, _ = load_variants(args.dataset, args.data, args.n, args.seed)
    spec = get_spec(args.model_key)
    model, tokenizer, device = load_model(args.model_key, args.device)
    model.eval()

    stem = args.model_key.replace("/", "_").replace(":", "_")
    out_dir = Path(args.out) if args.out else RESULTS_DIR / f"carbonx_patch_sweep_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = _done(records_path) if args.resume else set()
    records = []
    with records_path.open("a" if args.resume else "w") as handle:
        for variant in variants:
            if variant["index"] in done:
                continue
            gene = variant["gene"] or args.correct_gene
            result = run_patch(
                model,
                tokenizer,
                spec,
                variant["ref"],
                variant["alt"],
                args.source,
                args.target,
                args.layer,
                gene,
                args.wrong_gene,
                args.max_new_tokens,
                device,
            )
            record = {
                "variant": variant["index"],
                "chrom": variant["chrom"],
                "offset": variant["offset"],
                "ref": variant["ref"][variant["offset"]],
                "alt": variant["alt"][variant["offset"]],
                "label": variant["label"],
                "gene": gene,
                **result,
                "parsed": {name: parse_answer(value) for name, value in result["answers"].items()},
            }
            record["answer_changed"] = record["answers"]["target"] != record["answers"]["patched"]
            record["label_changed"] = (
                record["parsed"]["target"]["label"] != record["parsed"]["patched"]["label"]
            )
            record["control_label_changed"] = (
                record["parsed"]["target"]["label"] != record["parsed"]["random_donor"]["label"]
            )
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"[{len(records)}/{len(variants)}] delta_l2="
                f"{record['logits']['delta_l2']:.2f} "
                f"answer_changed={record['answer_changed']}",
                flush=True,
            )

    if not records:
        records = [
            json.loads(line) for line in records_path.read_text().splitlines() if line.strip()
        ]
    deltas = [record["logits"]["delta_l2"] for record in records]
    summary = {
        "model": args.model_key,
        "dataset": args.dataset if not args.data else str(args.data),
        "config": {
            "n": len(variants),
            "source": args.source,
            "target": args.target,
            "layer": records[0]["layer"] if records else args.layer,
            "seed": args.seed,
        },
        "records": len(records),
        "mean_delta_l2": round(float(np.mean(deltas)), 4) if deltas else None,
        "answer_change_rate": round(
            float(np.mean([record["answer_changed"] for record in records])), 4
        )
        if records
        else None,
        "label_change_rate": round(
            float(np.mean([record["label_changed"] for record in records])), 4
        )
        if records
        else None,
        "control_label_change_rate": round(
            float(np.mean([record["control_label_changed"] for record in records])),
            4,
        )
        if records
        else None,
        "files": {"records": str(records_path)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
