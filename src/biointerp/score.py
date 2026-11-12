"""Score one protein variant with a saved error-neuron probe."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from biointerp.core import apply_probe
from biointerp.esm import (
    aa_token_ids,
    extract_cett_batch,
    load_model,
    precompute_col_norms,
)

VARIANT_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_variant(text: str) -> tuple[str, int, str]:
    """Parse ``E30T`` into ``("E", 30, "T")``; positions are 1-based."""
    match = VARIANT_PATTERN.match(text.strip().upper())
    if not match:
        raise ValueError(f"invalid variant {text!r}; expected format like E30T")
    wild_type, position, mutant = match.groups()
    position = int(position)
    if position < 1:
        raise ValueError("variant position must be >= 1")
    if wild_type == mutant:
        raise ValueError("wild-type and mutant residues are identical")
    return wild_type, position, mutant


def validate_variant(sequence: str, wild_type: str, position: int) -> None:
    """Check that the sequence carries the expected wild-type residue."""
    if position > len(sequence):
        raise ValueError(f"position {position} exceeds sequence length {len(sequence)}")
    observed = sequence[position - 1].upper()
    if observed != wild_type:
        raise ValueError(
            f"wild-type mismatch at position {position}: expected {wild_type}, found {observed}"
        )


@torch.inference_mode()
def score_variant(
    model,
    tokenizer,
    probe: Mapping[str, object],
    sequence: str,
    wild_type: str,
    position: int,
    device: str | torch.device,
) -> float:
    """Return the probe's error probability for one masked variant position."""
    validate_variant(sequence, wild_type, position)

    ids = list(tokenizer(sequence, add_special_tokens=True)["input_ids"])
    token_index = position  # residue 1 sits at token 1 (token 0 is <cls>)
    if token_index >= len(ids) - 1:
        raise ValueError("variant position falls outside the tokenized sequence")
    ids[token_index] = tokenizer.mask_token_id

    tokens = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
    }
    mask_positions = torch.tensor([[token_index]], dtype=torch.long, device=device)

    layers = list(probe["layers"])  # type: ignore[arg-type]
    col_norms = precompute_col_norms(model, layers)
    result = extract_cett_batch(
        model,
        tokens,
        mask_positions,
        layers,
        col_norms,
        aa_token_ids(tokenizer),
    )
    features = result["cett"][0].float().cpu().numpy()[None, :]
    score = apply_probe(probe["scaler"], probe["clf"], probe["idx"], features)
    return float(score[0])


def load_sequence(args: argparse.Namespace) -> str:
    if args.sequence:
        return args.sequence.strip().upper()
    if args.fasta and args.id:
        from biointerp.clusters import read_fasta

        records = dict(read_fasta(args.fasta))
        if args.id not in records:
            raise SystemExit(f"sequence id not found in FASTA: {args.id}")
        return records[args.id].upper()
    raise SystemExit("provide --sequence or both --fasta and --id")


def score_main(args: argparse.Namespace) -> int:
    import joblib

    sequence = load_sequence(args)
    try:
        wild_type, position, mutant = parse_variant(args.mutation)
        validate_variant(sequence, wild_type, position)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    probe_path = Path(args.probe)
    if not probe_path.exists():
        raise SystemExit(f"probe artifact not found: {probe_path}")
    probe = joblib.load(probe_path)

    model_name = args.model or str(probe.get("model", "facebook/esm2_t33_650M_UR50D"))
    model, tokenizer, device = load_model(model_name, args.device)
    score = score_variant(model, tokenizer, probe, sequence, wild_type, position, device)

    reference = np.asarray(probe.get("reference_scores", []), dtype=float)
    percentile = float(np.mean(reference <= score)) if len(reference) else None

    payload = {
        "mutation": f"{wild_type}{position}{mutant}",
        "position": position,
        "wild_type": wild_type,
        "mutant": mutant,
        "sequence_length": len(sequence),
        "error_score": round(score, 4),
        "percentile": None if percentile is None else round(percentile, 4),
        "model": model_name,
        "probe_top_k": int(probe.get("top_k", len(probe["idx"]))),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        rank = "n/a" if percentile is None else f"{percentile * 100:.1f}th percentile"
        print(f"{payload['mutation']} | error score {payload['error_score']:.4f} | {rank}")
    return 0
