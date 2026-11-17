"""Causal steering control matrix.

Conditions are configuration, not code: true direction, negative direction,
random direction, or no-op. The small local run uses ESM-2 naturalness as the
endpoint; pLDDT/ESMFold2 is a later compute job.
"""

from __future__ import annotations

import argparse
import json
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
from biointerp.gen import load_generative
from biointerp.models import get_spec
from biointerp.repro import metadata
from biointerp.steering import (
    build_steer_map,
    parse_conditions,
    summarize_conditions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-model-key", default="protgpt2")
    p.add_argument("--scorer-model-key", default="esm2_650m")
    p.add_argument("--conditions", default="conf/steering/core.yaml")
    p.add_argument("--calib-seqs", type=int, default=4)
    p.add_argument("--eval-seqs", type=int, default=3)
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--layers", default="24,30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(RESULTS_DIR / "steering_matrix.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    layers = [int(value) for value in args.layers.split(",")]
    payload = yaml.safe_load(Path(args.conditions).read_text()) or {}
    conditions = parse_conditions(payload)

    gen_spec = get_spec(args.gen_model_key)
    scorer_spec = get_spec(args.scorer_model_key)
    gen_model, gen_tok, device = load_generative(gen_spec["hf_id"], args.device)
    esm_model, esm_tok, _ = load_esm(scorer_spec["hf_id"], args.device)
    aa_index = {aa: index for index, aa in enumerate(AA_ALPHABET)}
    aa_t = torch.tensor(aa_token_ids(esm_tok), device=device)
    run_args = SimpleNamespace(n_steps=args.n_steps, temperature=args.temperature, seed=args.seed)

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

    def evaluate(condition) -> dict[str, object]:
        steer_map = build_steer_map(
            condition,
            directions,
            mean_norms,
            hidden_dim=int(gen_model.config.n_embd),
            seed=args.seed,
        )
        values: list[float] = []
        for index in range(args.eval_seqs):
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
            if result is not None:
                values.extend(result["labels"])
        mean = float(np.mean(values)) if values else float("nan")
        print(f"{condition.name:20s} naturalness {mean:.4f} (n={len(values)})")
        return {
            "condition": condition.name,
            "direction": condition.direction,
            "layer": condition.layer,
            "alpha": condition.alpha,
            "mean_naturalness": round(mean, 4),
            "n_steps": len(values),
        }

    rows = [evaluate(condition) for condition in conditions]
    result = {
        "meta": metadata(),
        "gen_model": gen_spec["hf_id"],
        "scorer_model": scorer_spec["hf_id"],
        "calib_seqs": args.calib_seqs,
        "eval_seqs": args.eval_seqs,
        "n_steps": args.n_steps,
        "threshold": round(threshold, 4),
        "direction_norms": {
            str(layer): round(float(np.linalg.norm(directions[layer])), 4) for layer in layers
        },
        "rows": rows,
        "summary": summarize_conditions(rows),
        "elapsed_s": round(time.time() - started, 1),
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"summary: {result['summary']}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
