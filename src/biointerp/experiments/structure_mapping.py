"""Structure mapping on ESM-2 (Bio-Interp PLM-native).

Do prediction errors cluster in disordered / low-confidence structural regions?
For each sequence: fold with ESMFold2 (Biohub API) -> per-residue pLDDT; per-position
correctness via batched leave-one-out masking with ESM-2; correlate error with pLDDT.
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import DEFAULT_FASTA, RESULTS_DIR, load_sequences
from biointerp.esm import AA_ALPHABET, aa_token_ids
from biointerp.experiments.viability_scorer import fold_plddt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-seqs", type=int, default=10)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--model-key", default="esm2_650m")
    p.add_argument("--fold-model", default="esmfold2")
    p.add_argument("--api-key-env", default="BIOHUB_API_KEY")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.inference_mode()
def leave_one_out_p_true(model, tokenizer, seq, device, aa_t, aa_index):
    """One batched forward: L rows, each masking a different position -> p(true residue)."""
    ids = tokenizer(seq, add_special_tokens=True)["input_ids"]
    L = len(seq)
    batch = []
    for j in range(1, L + 1):
        row = list(ids)
        row[j] = tokenizer.mask_token_id
        batch.append(row)
    input_ids = torch.tensor(batch, device=device)
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    rows = torch.arange(L, device=device)
    logits = out.logits[rows, rows + 1][:, aa_t]
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    true_idx = np.array([aa_index[c] for c in seq])
    return probs[np.arange(L), true_idx]


def main() -> None:
    args = parse_args()
    from esm.sdk.forge import SequenceStructureForgeInferenceClient

    from biointerp.models import load_model as load_by_key

    token = os.environ.get(args.api_key_env, "")
    if not token:
        raise SystemExit(f"missing API key in ${args.api_key_env}")
    client = SequenceStructureForgeInferenceClient(model=args.fold_model, token=token)

    seqs = [s for s in load_sequences(DEFAULT_FASTA) if len(s) <= args.max_len][: args.n_seqs]
    model, tokenizer, device = load_by_key(args.model_key, args.device)
    aa_t = torch.tensor(aa_token_ids(tokenizer), device=device)
    aa_index = {c: i for i, c in enumerate(AA_ALPHABET)}

    records = []
    t0 = time.time()
    for i, seq in enumerate(seqs):
        plddt = fold_plddt(seq, client)
        p_true = leave_one_out_p_true(model, tokenizer, seq, device, aa_t, aa_index)
        L = min(len(plddt), len(p_true))
        plddt, p_true = plddt[:L], p_true[:L]
        err = 1.0 - p_true
        rho, _ = spearmanr(plddt, err)
        records.append(
            {
                "seq": i,
                "length": L,
                "spearman_plddt_error": round(float(rho), 4),
                "mean_plddt": round(float(plddt.mean()), 4),
                "mean_p_true": round(float(p_true.mean()), 4),
                "per_residue": [
                    {"plddt": round(float(a), 4), "p_true": round(float(b), 4)}
                    for a, b in zip(plddt, p_true)
                ],
            }
        )
        print(
            f"seq {i + 1}/{len(seqs)}: len {L} | pLDDT {plddt.mean():.3f} | p(true) {p_true.mean():.3f} | "
            f"rho(pLDDT, err) {rho:+.3f} | {time.time() - t0:.0f}s"
        )

    all_plddt = np.concatenate(
        [np.array([r["plddt"] for r in rec["per_residue"]]) for rec in records]
    )
    all_err = np.concatenate(
        [np.array([1 - r["p_true"] for r in rec["per_residue"]]) for rec in records]
    )
    rho_all, pval = spearmanr(all_plddt, all_err)
    structured = all_err[all_plddt >= 0.7]
    disordered = all_err[all_plddt < 0.5]
    print(f"\npooled positions: {len(all_err)}")
    print(f"pooled Spearman(pLDDT, error): {rho_all:.4f} (p={pval:.2e})")
    print(
        f"mean error | structured (pLDDT>=0.7, n={len(structured)}): {structured.mean() if len(structured) else float('nan'):.4f}"
    )
    print(
        f"mean error | disordered (pLDDT<0.5,  n={len(disordered)}): {disordered.mean() if len(disordered) else float('nan'):.4f}"
    )

    out = RESULTS_DIR / "structure_mapping.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model_key,
                "fold_model": args.fold_model,
                "n_seqs": len(records),
                "pooled_spearman_plddt_error": round(float(rho_all), 4),
                "mean_error_structured": round(float(structured.mean()), 4)
                if len(structured)
                else None,
                "mean_error_disordered": round(float(disordered.mean()), 4)
                if len(disordered)
                else None,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
