"""Variant-effect evaluation on ProteinGym assays.

Models come from ``conf/model``, datasets from ``conf/dataset``, and the core
assay list from ``biointerp.assays``. The experiment loops over assays and
writes per-assay metrics plus a BH-corrected summary.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from biointerp.assays import core_names, get_assay
from biointerp.core import RESULTS_DIR
from biointerp.inference import bh_fdr
from biointerp.models import get_spec, load_model
from biointerp.proteingym import from_config
from biointerp.repro import metadata
from biointerp.vep import Esm2Scorer, prepare_rows, run_vep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-config", default="proteingym", help="conf/dataset key")
    p.add_argument("--assay", default=None, help="single assay id")
    p.add_argument("--assays", default=None, help="comma-separated assay ids")
    p.add_argument("--shard", type=int, default=None, help="override the assay shard")
    p.add_argument("--n-mutations", type=int, default=None, help="override dataset default")
    p.add_argument("--model-key", default=None, help="conf/model key (defaults to config)")
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild the summary from saved per-assay JSONs without a model",
    )
    return p.parse_args()


def resolve_model_key(value: str | None) -> str:
    if value:
        return value
    from biointerp.runner import load_config

    return str(load_config()["model"])


def selected_assays(args: argparse.Namespace) -> list[str]:
    if args.assay:
        return [args.assay]
    if args.assays:
        return [name.strip() for name in args.assays.split(",") if name.strip()]
    return core_names()


def assay_shard(assay: str, override: int | None) -> int:
    if override is not None:
        return override
    shard = get_assay(assay)
    if shard is None:
        raise SystemExit(f"unknown assay {assay!r}; pass --shard explicitly")
    return shard.shard


def write_summary(
    dataset,
    model_key: str,
    args: argparse.Namespace,
    summaries: list[dict[str, object]],
    started: float,
) -> None:
    """Merge this run's rows into the persisted summary and recompute BH."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "vep_summary.json"
    merged: dict[str, dict[str, object]] = {}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            for row in existing.get("assays", []):
                if isinstance(row, dict) and row.get("assay"):
                    merged[str(row["assay"])] = row
        except (json.JSONDecodeError, OSError):
            merged = {}
    for row in summaries:
        merged[str(row["assay"])] = row

    rows = list(merged.values())
    valid = [row for row in rows if row.get("p_value") is not None]
    if valid:
        q_values = bh_fdr([float(row["p_value"]) for row in valid])
        for row, q_value in zip(valid, q_values):
            row["q_value"] = round(float(q_value), 6)

    summary = {
        "meta": metadata(),
        "dataset": dataset.name,
        "dataset_repo": dataset.repo,
        "model": model_key,
        "top_k": args.top_k,
        "test_frac": args.test_frac,
        "assays": rows,
        "wins": sum(1 for row in rows if row["delta_ci95"][0] > 0),
        "losses": sum(1 for row in rows if row["delta_ci95"][1] < 0),
        "ties": sum(1 for row in rows if row["delta_ci95"][0] <= 0 <= row["delta_ci95"][1]),
        "elapsed_s": round(time.time() - started, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved: {summary_path}")


def aggregate_only(dataset, model_key: str, args: argparse.Namespace) -> None:
    """Rebuild the persisted summary from saved per-assay result JSONs."""
    started = time.time()
    summaries: list[dict[str, object]] = []
    for assay in selected_assays(args):
        path = RESULTS_DIR / f"vep_proteingym_{assay}.json"
        if not path.exists():
            print(f"[missing] {assay}: no saved result")
            continue
        payload = json.loads(path.read_text())
        if not {"delta", "delta_ci95", "p_value"} <= payload.keys():
            print(f"[skip] {assay}: no paired interval stored")
            continue
        summaries.append(
            {
                "assay": assay,
                "n_mutations": payload["n_mutations"],
                "auroc_probe": payload["auroc_probe"],
                "auroc_confidence": payload["auroc_confidence"],
                "delta": payload["delta"],
                "delta_ci95": payload["delta_ci95"],
                "p_value": payload["p_value"],
            }
        )
    write_summary(dataset, model_key, args, summaries, started)


def main() -> None:
    args = parse_args()
    started = time.time()
    dataset = from_config(args.dataset_config)
    n_mutations = args.n_mutations if args.n_mutations is not None else dataset.n_mutations
    model_key = resolve_model_key(args.model_key)
    spec = get_spec(model_key)
    if spec.get("adapter") != "esm":
        raise SystemExit(
            f"VEP scorer for adapter {spec.get('adapter')!r} is not implemented; "
            "add a scorer in biointerp.vep and keep the runner unchanged"
        )

    if args.aggregate_only:
        aggregate_only(dataset, model_key, args)
        return

    model, tokenizer, device = load_model(model_key, args.device)
    scorer = Esm2Scorer(model, tokenizer, device)
    shard_cache: dict[int, object] = {}
    summaries: list[dict[str, object]] = []

    for assay in selected_assays(args):
        shard = assay_shard(assay, args.shard)
        if shard not in shard_cache:
            shard_cache[shard] = dataset.load_shard(shard)
        frame = shard_cache[shard]
        frame = frame[frame["DMS_id"] == assay].reset_index(drop=True)
        rows = prepare_rows(frame, n_mutations, args.seed)
        print(f"assay: {assay} | substitutions used: {len(rows)} (single)")
        if len(rows) < 10:
            print(f"[skip] {assay}: only {len(rows)} single substitutions")
            continue

        try:
            result = run_vep(
                rows,
                scorer,
                top_k=args.top_k,
                test_frac=args.test_frac,
                seed=args.seed,
                batch_size=args.batch_size,
                n_boot=args.bootstrap,
            )
        except ValueError as exc:
            print(f"[skip] {assay}: {exc}")
            continue

        payload = {
            "dataset": dataset.name,
            "dataset_repo": dataset.repo,
            "model": model_key,
            "assay": assay,
            "n_mutations": result["n_mutations"],
            "spearman": result["spearman"],
            "sign_accuracy": result["sign_accuracy"],
            "auroc_probe": result["auroc_probe"],
            "auroc_confidence": result["auroc_confidence"],
            "delta": result["delta"],
            "delta_ci95": result["delta_ci95"],
            "p_value": result["p_value"],
            "top_k": args.top_k,
            "test_frac": args.test_frac,
        }
        out = RESULTS_DIR / f"vep_proteingym_{assay}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")

        predictions_path = RESULTS_DIR / f"vep_proteingym_{assay}.predictions.jsonl"
        with predictions_path.open("w") as handle:
            for index, row in enumerate(rows):
                mutation = ":".join(
                    f"{wild_type}{position}{mutant}" for wild_type, position, mutant in row["muts"]
                )
                probe_score = result["probe_scores"][index]
                handle.write(
                    json.dumps(
                        {
                            "mutant": mutation,
                            "score": round(float(result["scores"][index]), 4),
                            "dms_score": round(float(result["dms"][index]), 4),
                            "correct": bool(result["y"][index] == 0),
                            "confidence": round(float(result["confidence"][index]), 4),
                            "probe_score": (
                                None if np.isnan(probe_score) else round(float(probe_score), 6)
                            ),
                            "split": ("test" if result["test_mask"][index] else "train"),
                        }
                    )
                    + "\n"
                )

        summaries.append(
            {
                "assay": assay,
                "n_mutations": result["n_mutations"],
                "auroc_probe": result["auroc_probe"],
                "auroc_confidence": result["auroc_confidence"],
                "delta": result["delta"],
                "delta_ci95": result["delta_ci95"],
                "p_value": result["p_value"],
            }
        )
        print(
            f"AUROC probe {result['auroc_probe']:.4f} | "
            f"confidence {result['auroc_confidence']:.4f} | "
            f"delta {result['delta']:+.4f} {result['delta_ci95']}"
        )
        write_summary(dataset, model_key, args, summaries, started)

    write_summary(dataset, model_key, args, summaries, started)


if __name__ == "__main__":
    main()
