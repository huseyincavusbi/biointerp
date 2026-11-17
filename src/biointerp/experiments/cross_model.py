"""Cross-model replication of error-neuron probes.

This is a thin CLI over ``biointerp.replication`` and ``biointerp.adapters``.
The runner is modality-agnostic; new modalities only need a ModelAdapter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biointerp.adapters import build_adapter
from biointerp.core import DEFAULT_FASTA, DEFAULT_NPZ, RESULTS_DIR, load_sequences
from biointerp.replication import (
    analyze,
    extract_features,
    load_reference,
    load_split,
)
from biointerp.repro import metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", default="esmc_600m", help="registry key or HF id")
    p.add_argument("--reference-npz", default=str(DEFAULT_NPZ))
    p.add_argument("--assignments", default=str(RESULTS_DIR / "split_assignments.tsv"))
    p.add_argument("--n", type=int, default=0, help="0 uses every reference row")
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--device-map", default=None, help="e.g. auto for multi-GPU")
    p.add_argument("--cached-features", default=None, help="skip extraction and reuse an npz")
    p.add_argument("--out", default=str(RESULTS_DIR / "cross_model_cluster.json"))
    p.add_argument("--features-out", default=str(RESULTS_DIR / "cross_model_features.npz"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    reference = load_reference(args.reference_npz, args.n)
    n = len(reference["position"])  # type: ignore[arg-type]
    sequences = load_sequences(DEFAULT_FASTA)[:n]
    ids = [f"seq{index}" for index in range(n)]
    keep, is_test = load_split(args.assignments, ids)

    adapter = build_adapter(args.model_key, args.device, args.dtype, args.device_map)
    print(
        f"adapter: {adapter.name} | layers: {adapter.n_layers} | "
        f"intermediate: {adapter.intermediate} | rows: {n}"
    )

    if args.cached_features:
        cached = np.load(args.cached_features, allow_pickle=True)
        if not np.array_equal(cached["position"], reference["position"]):
            raise SystemExit("cached features use different masked positions")
        extracted = {
            "cett": cached["cett"].astype(np.float32),
            "correct": cached["correct"].astype(bool),
            "confidence": cached["confidence"].astype(float),
            "pred_aa": cached["pred_aa"],
        }
        print(f"using cached features: {args.cached_features}")
    else:
        extracted = extract_features(reference, sequences, adapter)

    result = analyze(
        reference,
        extracted,
        keep,
        is_test,
        adapter,
        top_k=args.top_k,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    result["meta"] = metadata()
    result["positions"] = "same masked positions as the reference"
    result["features_out"] = args.features_out
    result["elapsed_s"] = round(time.time() - started, 1)

    Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.features_out,
        cett=extracted["cett"].astype(np.float16),
        correct=extracted["correct"],
        confidence=extracted["confidence"],
        position=reference["position"],
        true_aa=np.asarray(reference["true_token"]),
        pred_aa=np.asarray(extracted["pred_aa"]),
        protein=reference["protein"],
        model=adapter.name,
    )

    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{adapter.name} AUROC {result['adapter']['auroc_probe']:.4f} "
        f"{result['adapter']['ci95']} vs reference "
        f"{result['reference']['auroc_probe']:.4f} {result['reference']['ci95']}"
    )
    print(
        f"score rho {result['comparison']['score_spearman']:.3f} | "
        f"layer profile r {result['comparison']['layer_profile_pearson']:.3f} | "
        f"layer histogram r {result['comparison']['layer_histogram_pearson']:.3f}"
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
