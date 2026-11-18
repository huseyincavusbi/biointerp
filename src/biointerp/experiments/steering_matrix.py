"""Causal steering control matrix with naturalness or pLDDT endpoints.

Conditions are configuration, not code. Calibration directions come from ESM-2
naturalness; the endpoint can be naturalness (local) or ESMFold2 pLDDT
(API/GPU). Generated sets are matched on length and hydrophobic fraction before
condition means are compared.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET, aa_token_ids
from biointerp.esm import load_model as load_esm
from biointerp.experiments.generative_causal import generate_and_label
from biointerp.experiments.viability_scorer import fold_plddt
from biointerp.gen import generate_with_cett, load_generative
from biointerp.models import get_spec
from biointerp.repro import metadata
from biointerp.steering import (
    build_steer_map,
    match_by_bins,
    parse_conditions,
    summarize_conditions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-model-key", default="protgpt2")
    p.add_argument("--scorer-model-key", default="esm2_650m")
    p.add_argument("--conditions", default="conf/steering/core.yaml")
    p.add_argument("--evaluator", choices=("naturalness", "plddt"), default="naturalness")
    p.add_argument("--api-key-env", default="BIOHUB_API_KEY")
    p.add_argument("--fold-model", default="esmfold2")
    p.add_argument("--calib-seqs", type=int, default=4)
    p.add_argument("--eval-seqs", type=int, default=3)
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--layers", default="24,30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(RESULTS_DIR / "steering_matrix.json"))
    p.add_argument("--figure", default=str(RESULTS_DIR / "steering_matrix.png"))
    return p.parse_args()


def plot_conditions(path: str, rows: list[dict[str, object]], metric: str) -> None:
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["condition"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    baseline = next(float(row[metric]) for row in rows if row["condition"] == "baseline")
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.bar(range(len(rows)), values, color="steelblue")
    axis.axhline(baseline, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel(metric)
    axis.set_ylim(min(values) - 0.02, max(values) + 0.02)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.time()
    layers = [int(value) for value in args.layers.split(",")]
    conditions = parse_conditions(yaml.safe_load(Path(args.conditions).read_text()) or {})

    gen_spec = get_spec(args.gen_model_key)
    scorer_spec = get_spec(args.scorer_model_key)
    gen_model, gen_tok, device = load_generative(gen_spec["hf_id"], args.device)
    esm_model, esm_tok, _ = load_esm(scorer_spec["hf_id"], args.device)
    aa_index = {aa: index for index, aa in enumerate(AA_ALPHABET)}
    aa_t = torch.tensor(aa_token_ids(esm_tok), device=device)
    run_args = SimpleNamespace(n_steps=args.n_steps, temperature=args.temperature, seed=args.seed)

    fold_client = None
    if args.evaluator == "plddt":
        token = os.environ.get(args.api_key_env, "")
        if not token:
            raise SystemExit(f"missing {args.api_key_env} for the pLDDT evaluator")
        from esm.sdk.forge import SequenceStructureForgeInferenceClient

        fold_client = SequenceStructureForgeInferenceClient(model=args.fold_model, token=token)

    hidden_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    labels: list[float] = []
    for index in range(args.calib_seqs):
        result = generate_and_label(
            gen_model,
            gen_tok,
            esm_model,
            esm_tok,
            aa_t,
            aa_index,
            run_args,
            device,
            seed_offset=index,
            capture_layers=layers,
        )
        if result is None:
            continue
        for row, step in enumerate(result["valid"]):
            labels.append(result["labels"][row])
            for layer in layers:
                hidden_by_layer[layer].append(result["hidden"][layer][step].numpy())
        print(f"calib {index + 1}/{args.calib_seqs} | steps {len(labels)}")

    if len(labels) < 4:
        raise SystemExit("calibration produced too few labelled steps")
    labels_array = np.asarray(labels)
    threshold = float(np.median(labels_array))
    directions: dict[int, np.ndarray] = {}
    mean_norms: dict[int, float] = {}
    for layer in layers:
        matrix = np.stack(hidden_by_layer[layer])
        directions[layer] = matrix[labels_array < threshold].mean(axis=0) - matrix[
            labels_array >= threshold
        ].mean(axis=0)
        mean_norms[layer] = float(np.linalg.norm(matrix, axis=1).mean())

    def generate_sequence(steer_map, seed_offset: int) -> str:
        _, _, texts, _, _ = generate_with_cett(
            gen_model,
            gen_tok,
            n_steps=args.n_steps,
            seed=args.seed + seed_offset,
            temperature=args.temperature,
            greedy=False,
            device=device,
            steer_map=steer_map,
        )
        return "".join(char for char in "".join(texts) if char in aa_index)

    metric_key = "mean_naturalness" if args.evaluator == "naturalness" else "mean_plddt"

    def evaluate(condition):
        steer_map = build_steer_map(
            condition,
            directions,
            mean_norms,
            hidden_dim=int(gen_model.config.n_embd),
            seed=args.seed,
        )
        sequences: list[str] = []
        per_sequence: list[float] = []
        values: list[float] = []
        for index in range(args.eval_seqs):
            if args.evaluator == "naturalness":
                result = generate_and_label(
                    gen_model,
                    gen_tok,
                    esm_model,
                    esm_tok,
                    aa_t,
                    aa_index,
                    run_args,
                    device,
                    seed_offset=1000 + index,
                    steer_map=steer_map,
                )
                if result is None:
                    continue
                sequence = result["seq"]
                score = float(np.mean(result["labels"]))
                values.extend(result["labels"])
            else:
                sequence = generate_sequence(steer_map, 1000 + index)
                plddt = fold_plddt(sequence, fold_client)
                score = float(np.mean(plddt))
                values.extend(plddt.tolist())
            sequences.append(sequence)
            per_sequence.append(score)

        mean = float(np.mean(values)) if values else float("nan")
        print(f"{condition.name:20s} {args.evaluator} {mean:.4f} (n={len(values)})")
        return (
            {
                "condition": condition.name,
                "direction": condition.direction,
                "layer": condition.layer,
                "alpha": condition.alpha,
                "n_neurons": condition.n_neurons,
                metric_key: round(mean, 4),
                "n_points": len(values),
            },
            sequences,
            per_sequence,
        )

    rows: list[dict[str, object]] = []
    sequence_sets: list[list[str]] = []
    per_sequence_sets: list[list[float]] = []
    for condition in conditions:
        row, sequences, per_sequence = evaluate(condition)
        rows.append(row)
        sequence_sets.append(sequences)
        per_sequence_sets.append(per_sequence)

    baseline_sequences = sequence_sets[0]
    baseline_scores = per_sequence_sets[0]
    for row, sequences, per_sequence in zip(rows, sequence_sets, per_sequence_sets):
        base_index, cond_index = match_by_bins(baseline_sequences, sequences)
        if len(base_index) > 0:
            row[f"matched_{metric_key}"] = round(
                float(np.mean(np.asarray(per_sequence)[cond_index])), 4
            )
            row["matched_n"] = int(len(cond_index))
            row["matched_baseline"] = round(
                float(np.mean(np.asarray(baseline_scores)[base_index])), 4
            )

    result = {
        "meta": metadata(),
        "gen_model": gen_spec["hf_id"],
        "scorer_model": scorer_spec["hf_id"],
        "evaluator": args.evaluator,
        "calib_seqs": args.calib_seqs,
        "eval_seqs": args.eval_seqs,
        "n_steps": args.n_steps,
        "threshold": round(threshold, 4),
        "direction_norms": {
            str(layer): round(float(np.linalg.norm(directions[layer])), 4) for layer in layers
        },
        "rows": rows,
        "summary": summarize_conditions(rows, metric=metric_key),
        "elapsed_s": round(time.time() - started, 1),
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    plot_conditions(args.figure, rows, metric_key)
    print(f"summary: {result['summary']}")
    print(f"wrote: {out}")
    print(f"wrote: {args.figure}")


if __name__ == "__main__":
    main()
