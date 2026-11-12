"""Homology-aware split for the Swiss-Prot sample."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from biointerp.clusters import (
    ASSAY_PREFIX,
    CORE_ASSAYS,
    cluster_sequences,
    exclude_clusters,
    fetch_uniprot_fasta,
    merge_clusters,
    read_fasta,
    search_hits,
    split_by_cluster,
    write_fasta,
)
from biointerp.core import DATA_DIR, DEFAULT_FASTA, RESULTS_DIR

MAX_REPAIR_ROUNDS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", default=str(DEFAULT_FASTA))
    p.add_argument("--assay-fasta", default=str(DATA_DIR / "core_assays.fasta"))
    p.add_argument("--no-assays", action="store_true", help="skip assay exclusion")
    p.add_argument("--min-seq-id", type=float, default=0.3)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    records = read_fasta(args.fasta)
    ids = [record_id for record_id, _ in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate sequence ids in input FASTA")

    assay_records: list[tuple[str, str]] = []
    if not args.no_assays:
        assay_path = fetch_uniprot_fasta(CORE_ASSAYS, args.assay_fasta)
        assay_records = read_fasta(assay_path)
        expected = {f"{ASSAY_PREFIX}{entry}" for entry in CORE_ASSAYS}
        missing = expected - {record_id for record_id, _ in assay_records}
        if missing:
            raise SystemExit(f"missing assay sequences: {sorted(missing)}")

    combined = records + assay_records
    work_root = Path(tempfile.mkdtemp(prefix="cluster_split_", dir=RESULTS_DIR))
    cluster_fasta = write_fasta(combined, work_root / "combined.fasta")

    cluster_of = cluster_sequences(
        cluster_fasta,
        work_root,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        threads=args.threads,
    )

    assay_ids = [record_id for record_id in cluster_of if record_id.startswith(ASSAY_PREFIX)]
    assay_fasta = None
    if assay_records:
        assay_fasta = write_fasta(assay_records, work_root / "assays.fasta")
        swiss_fasta = write_fasta(records, work_root / "swiss.fasta")
        initial_assay_hits = search_hits(
            swiss_fasta,
            assay_fasta,
            work_root / "audit_assay",
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
            threads=args.threads,
        )
        cluster_of = merge_clusters(cluster_of, [(hit[0], hit[1]) for hit in initial_assay_hits])

    def current_kept() -> tuple[list[tuple[str, str]], set[str]]:
        excluded = exclude_clusters(cluster_of, assay_ids)
        kept = [
            (record_id, sequence)
            for record_id, sequence in records
            if cluster_of[record_id] not in excluded
        ]
        return kept, excluded

    kept, excluded_clusters = current_kept()
    split_info: dict[str, object] = {}
    test_hits: list[tuple[str, str, float]] = []
    repair_rounds = 0

    for repair_rounds in range(MAX_REPAIR_ROUNDS):
        train_mask, test_mask, split_info = split_by_cluster(
            [record_id for record_id, _ in kept],
            cluster_of,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        train_records = [kept[i] for i in np.flatnonzero(train_mask)]
        test_records = [kept[i] for i in np.flatnonzero(test_mask)]
        train_fasta = write_fasta(train_records, work_root / "train.fasta")
        test_fasta = write_fasta(test_records, work_root / "test.fasta")

        test_hits = search_hits(
            train_fasta,
            test_fasta,
            work_root / f"audit_test_{repair_rounds}",
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
            threads=args.threads,
        )
        if not test_hits:
            break

        cluster_of = merge_clusters(cluster_of, [(hit[0], hit[1]) for hit in test_hits])
        kept, excluded_clusters = current_kept()

    assay_hits: list[tuple[str, str, float]] = []
    if assay_fasta is not None:
        assay_hits = search_hits(
            train_fasta,
            assay_fasta,
            work_root / "audit_assay_final",
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
            threads=args.threads,
        )

    max_identity = max((hit[2] for hit in test_hits + assay_hits), default=0.0)

    split_of: dict[str, str] = {}
    if split_info:
        for index, record_id in enumerate([record_id for record_id, _ in kept]):
            split_of[record_id] = "test" if test_mask[index] else "train"

    assignments = RESULTS_DIR / "split_assignments.tsv"
    with assignments.open("w") as handle:
        handle.write("id\tcluster\tsplit\n")
        for record_id in ids:
            split = split_of.get(record_id, "excluded_assay")
            handle.write(f"{record_id}\t{cluster_of[record_id]}\t{split}\n")

    audit = {
        "params": {
            "fasta": str(args.fasta),
            "min_seq_id": args.min_seq_id,
            "coverage": args.coverage,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "threads": args.threads,
            "assay_exclusion": not args.no_assays,
        },
        "counts": {
            "input_sequences": len(records),
            "assay_sequences": len(assay_records),
            "excluded_sequences": len(records) - len(kept),
            "clusters_total": len(set(cluster_of.values())),
            "clusters_excluded_assay": len(excluded_clusters),
            "clusters_kept": len({cluster_of[record_id] for record_id, _ in kept}),
        },
        "split": split_info,
        "leakage": {
            "train_vs_test_hits": len(test_hits),
            "train_vs_assay_hits": len(assay_hits),
            "max_identity": round(max_identity, 4),
            "repair_rounds": repair_rounds,
            "passed": not test_hits and not assay_hits,
        },
        "artifacts": {
            "work_dir": str(work_root),
            "assignments": str(assignments),
        },
        "elapsed_s": round(time.time() - started, 1),
    }

    out_path = Path(args.out) if args.out else RESULTS_DIR / "split_audit.json"
    out_path.write_text(json.dumps(audit, indent=2))

    print(f"input sequences: {len(records)}")
    print(f"assay sequences: {len(assay_records)}")
    print(f"clusters total: {audit['counts']['clusters_total']}")
    print(f"clusters excluded (assay): {len(excluded_clusters)}")
    print(f"train/test: {split_info.get('n_train')}/{split_info.get('n_test')}")
    print(f"repair rounds: {repair_rounds}")
    print(f"train vs test hits: {len(test_hits)}")
    print(f"train vs assay hits: {len(assay_hits)}")
    print(f"wrote: {out_path}")

    if test_hits or assay_hits:
        print("LEAKAGE DETECTED", file=sys.stderr)
        for hit in (test_hits + assay_hits)[:10]:
            print(f"  {hit[0]} -> {hit[1]} ({hit[2]:.3f})", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
