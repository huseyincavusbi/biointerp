"""Confound-matched analysis of the pLDDT/error relation.

Rebuilds residue-level rows from ``structure_mapping.json`` and the source
FASTA, then reports:

- pooled Spearman(pLDDT, error);
- partial Spearman controlling protein length and relative position;
- matched disordered-vs-structured error difference, matching on amino-acid
  class, length, and relative position.

All intervals are cluster bootstraps over proteins.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biointerp.confounds import (
    aa_class,
    cluster_bootstrap,
    cluster_bootstrap_values,
    match_nearest,
    partial_spearman,
    percentile_ci,
    pooled_spearman,
    standardized_mean_diff,
)
from biointerp.core import DEFAULT_FASTA, RESULTS_DIR, load_sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", default=str(RESULTS_DIR / "structure_mapping.json"))
    p.add_argument("--fasta", default=str(DEFAULT_FASTA))
    p.add_argument("--max-len", type=int, default=250)
    p.add_argument("--n-seqs", type=int, default=30)
    p.add_argument("--disordered", type=float, default=0.5)
    p.add_argument("--structured", type=float, default=0.7)
    p.add_argument("--caliper", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def build_rows(structure: dict, sequences: list[str]) -> dict[str, np.ndarray]:
    proteins: list[int] = []
    positions: list[int] = []
    lengths: list[int] = []
    relative: list[float] = []
    classes: list[str] = []
    plddt: list[float] = []
    error: list[float] = []

    records = structure["records"]
    if len(records) > len(sequences):
        raise SystemExit("not enough sequences for the structure records")

    for record in records:
        index = int(record["seq"])
        sequence = sequences[index]
        if len(sequence) != int(record["length"]):
            raise SystemExit(
                f"sequence {index} length mismatch: {len(sequence)} != {record['length']}"
            )
        length = len(sequence)
        for position, residue in enumerate(record["per_residue"]):
            proteins.append(index)
            positions.append(position)
            lengths.append(length)
            relative.append(position / (length - 1) if length > 1 else 0.0)
            classes.append(aa_class(sequence[position]))
            plddt.append(float(residue["plddt"]))
            error.append(1.0 - float(residue["p_true"]))

    class_names = sorted(set(classes))
    class_codes = np.asarray([class_names.index(c) for c in classes], dtype=int)
    return {
        "protein": np.asarray(proteins, dtype=int),
        "position": np.asarray(positions, dtype=int),
        "length": np.asarray(lengths, dtype=float),
        "relative": np.asarray(relative, dtype=float),
        "class_code": class_codes,
        "class_name": np.asarray(classes, dtype=object),
        "plddt": np.asarray(plddt, dtype=float),
        "error": np.asarray(error, dtype=float),
    }


def main() -> None:
    args = parse_args()
    started = time.time()

    structure = json.loads(Path(args.structure).read_text())
    sequences = [
        sequence for sequence in load_sequences(args.fasta) if len(sequence) <= args.max_len
    ][: args.n_seqs]
    rows = build_rows(structure, sequences)

    plddt = rows["plddt"]
    error = rows["error"]
    protein = rows["protein"]
    length = rows["length"]
    relative = rows["relative"]
    class_name = rows["class_name"]

    log_length = np.log(length)
    z_length = (log_length - log_length.mean()) / log_length.std(ddof=1)

    disordered = plddt < args.disordered
    structured = plddt >= args.structured
    if disordered.sum() == 0 or structured.sum() == 0:
        raise SystemExit("no positions in one of the pLDDT groups")

    raw_rho = pooled_spearman(plddt, error)
    raw_null = cluster_bootstrap(
        protein,
        lambda index: pooled_spearman(plddt[index], error[index]),
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    partial_rho, partial_p = partial_spearman(plddt, error, [z_length, relative])
    partial_null = cluster_bootstrap(
        protein,
        lambda index: partial_spearman(
            plddt[index], error[index], [z_length[index], relative[index]]
        )[0],
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    treated = np.flatnonzero(disordered)
    controls = np.flatnonzero(structured)

    def distance(left: int, right: int) -> float:
        return float(abs(z_length[left] - z_length[right]) + abs(relative[left] - relative[right]))

    pairs, unmatched = match_nearest(
        treated,
        controls,
        class_name,
        distance,
        caliper=args.caliper,
        seed=args.seed,
    )
    differences = np.asarray([error[left] - error[right] for left, right in pairs])
    pair_groups = np.asarray([protein[left] for left, _ in pairs], dtype=int)
    matched_null = cluster_bootstrap_values(
        differences, pair_groups, n_boot=args.bootstrap, seed=args.seed
    )

    matched_controls = np.asarray([right for _, right in pairs], dtype=int)
    balance = {
        "length_smd_before": round(
            standardized_mean_diff(z_length[treated], z_length[controls]), 4
        ),
        "length_smd_after": round(
            standardized_mean_diff(z_length[treated], z_length[matched_controls]), 4
        ),
        "relative_smd_before": round(
            standardized_mean_diff(relative[treated], relative[controls]), 4
        ),
        "relative_smd_after": round(
            standardized_mean_diff(relative[treated], relative[matched_controls]), 4
        ),
    }

    proteins_disordered = int(len(np.unique(protein[disordered])))
    proteins_structured = int(len(np.unique(protein[structured])))
    proteins_both = int(len(set(protein[disordered].tolist()) & set(protein[structured].tolist())))

    result = {
        "structure": str(args.structure),
        "fasta": str(args.fasta),
        "n_proteins": int(len(np.unique(protein))),
        "n_positions": int(len(plddt)),
        "groups": {
            "disordered_threshold": args.disordered,
            "structured_threshold": args.structured,
            "n_disordered": int(disordered.sum()),
            "n_structured": int(structured.sum()),
            "n_proteins_disordered": proteins_disordered,
            "n_proteins_structured": proteins_structured,
            "n_proteins_both": proteins_both,
        },
        "inference": (
            "Intervals are cluster bootstraps over proteins. Positional p-values "
            "assume independent residues and are reported only for reference."
        ),
        "raw": {
            "spearman": round(raw_rho, 4),
            "ci95": percentile_ci(raw_null),
            "mean_error_disordered": round(float(error[disordered].mean()), 4),
            "mean_error_structured": round(float(error[structured].mean()), 4),
        },
        "partial": {
            "spearman": round(partial_rho, 4),
            "p_value_positional": round(partial_p, 6),
            "ci95": percentile_ci(partial_null),
            "controls": ["log_length", "relative_position"],
        },
        "matched": {
            "n_pairs": len(pairs),
            "n_unmatched": len(unmatched),
            "match_rate": round(len(pairs) / max(1, len(treated)), 4),
            "caliper": args.caliper,
            "mean_difference": round(float(differences.mean()), 4) if differences.size else None,
            "ci95": percentile_ci(matched_null) if differences.size else None,
            "balance": balance,
        },
        "params": {
            "max_len": args.max_len,
            "n_seqs": args.n_seqs,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        "elapsed_s": round(time.time() - started, 1),
    }

    out = Path(args.out) if args.out else RESULTS_DIR / "confound_matching.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"positions: {result['n_positions']} | proteins: {result['n_proteins']}")
    print(
        f"raw rho: {raw_rho:+.4f} CI {result['raw']['ci95']} | "
        f"errors {result['raw']['mean_error_disordered']:.3f} vs "
        f"{result['raw']['mean_error_structured']:.3f}"
    )
    print(f"partial rho: {partial_rho:+.4f} (p={partial_p:.2g}) CI {result['partial']['ci95']}")
    print(
        f"matched diff: {result['matched']['mean_difference']} "
        f"CI {result['matched']['ci95']} | pairs {len(pairs)}/{len(treated)}"
    )
    print(f"balance: {balance}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
