"""Viability scorer for generated protein sequences (Bio-Interp generative phase).

Backends:
  - api: Biohub Platform ESMFold2 fold endpoint -> per-residue pLDDT (default)
  - naturalness: ESM-2 masked leave-one-out p(true residue) proxy (local)

Input: sequences generated with ProtGPT2 (gen.py). Output: per-sequence mean viability +
per-residue values (JSON/JSONL). The API key is read from an environment variable and is
never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET, aa_token_ids
from biointerp.esm import load_model as load_esm
from biointerp.gen import generate_with_cett, load_generative


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="api", choices=["api", "naturalness"])
    p.add_argument("--gen-model", default="nferruz/ProtGPT2")
    p.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--fold-model", default="esmfold2")
    p.add_argument("--api-key-env", default="BIOHUB_API_KEY")
    p.add_argument("--n-seqs", type=int, default=3)
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def fold_plddt(sequence: str, client) -> np.ndarray:
    res = client.fold(sequence=sequence)
    if type(res).__name__ == "ESMProteinError":
        raise RuntimeError(f"fold error: {res}")
    plddt = getattr(res, "plddt", None)
    if plddt is None:
        raise RuntimeError("fold response has no plddt")
    return np.asarray(plddt.detach().cpu() if torch.is_tensor(plddt) else plddt, dtype=float)


def main() -> None:
    args = parse_args()
    gen_model, gen_tok, device = load_generative(args.gen_model, args.device)
    aa_index = {c: i for i, c in enumerate(AA_ALPHABET)}

    client = None
    esm_model = esm_tok = aa_t = None
    if args.backend == "api":
        from esm.sdk.forge import SequenceStructureForgeInferenceClient

        token = os.environ.get(args.api_key_env, "")
        if not token:
            raise SystemExit(f"missing API key in ${args.api_key_env}")
        client = SequenceStructureForgeInferenceClient(model=args.fold_model, token=token)
    else:
        from biointerp.experiments.generative_error_neurons import esm_naturalness

        esm_model, esm_tok, _ = load_esm(args.esm_model, args.device)
        aa_t = torch.tensor(aa_token_ids(esm_tok), device=device)

    rows = []
    t0 = time.time()
    for si in range(args.n_seqs):
        ids, cetts, texts, confs, ents = generate_with_cett(
            gen_model,
            gen_tok,
            n_steps=args.n_steps,
            seed=args.seed + si,
            temperature=args.temperature,
            greedy=False,
            device=device,
        )
        seq = "".join(c for c in "".join(texts) if c in aa_index)
        if len(seq) < 10:
            continue
        if args.backend == "api":
            vals = fold_plddt(seq, client)
        else:
            vals = esm_naturalness(esm_model, esm_tok, seq, device, aa_t, aa_index)
        rows.append(
            {
                "seq": si,
                "sequence": seq,
                "length": len(seq),
                "mean_viability": round(float(np.mean(vals)), 4),
                "min_viability": round(float(np.min(vals)), 4),
                "per_residue": [round(float(v), 4) for v in vals],
            }
        )
        print(
            f"seq {si + 1}/{args.n_seqs}: len {len(seq)} | mean {rows[-1]['mean_viability']} | "
            f"range [{rows[-1]['min_viability']}, {round(float(np.max(vals)), 4)}] | {time.time() - t0:.0f}s"
        )

    suffix = args.backend
    out = RESULTS_DIR / f"viability_scorer_{suffix}.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = RESULTS_DIR / f"viability_scorer_{suffix}.json"
    summary.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "fold_model": args.fold_model if args.backend == "api" else None,
                "n_sequences": len(rows),
                "n_steps": args.n_steps,
                "mean_viability": round(float(np.mean([r["mean_viability"] for r in rows])), 4)
                if rows
                else None,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {out}")
    print(f"saved: {summary}")


if __name__ == "__main__":
    main()
