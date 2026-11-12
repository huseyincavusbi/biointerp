"""Homology-aware clustering and splits for protein sequence datasets.

Wraps MMseqs2 ``easy-cluster`` and ``easy-search`` so experiments can group
homologous sequences before splitting. The split unit is a cluster, never a
single sequence, which prevents near-duplicate leakage across train and test.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

CORE_ASSAYS = (
    "SDA_BACSU",
    "MTHR_HUMAN",
    "S22A1_HUMAN",
    "MSH2_HUMAN",
    "PPARG_HUMAN",
    "PAI1_HUMAN",
)

ASSAY_PREFIX = "ASSAY_"


def parse_fasta_text(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into ``(id, sequence)`` records.

    Only the first whitespace-delimited token of a header is kept.
    """
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header = line[1:].split()[0]
            parts = []
        elif header is not None:
            parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read a FASTA file into ``(id, sequence)`` records."""
    return parse_fasta_text(Path(path).read_text())


def write_fasta(records: Iterable[tuple[str, str]], path: str | Path) -> Path:
    """Write ``(id, sequence)`` records to a FASTA file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f">{record_id}\n{sequence}\n" for record_id, sequence in records))
    return out


def parse_cluster_tsv(path: str | Path) -> dict[str, str]:
    """Parse an MMseqs2 cluster TSV into ``member -> representative``.

    Representatives map to themselves so every input id is present.
    """
    cluster_of: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"malformed cluster row: {line!r}")
        representative, member = fields[0], fields[1]
        cluster_of[member] = representative
        cluster_of.setdefault(representative, representative)
    return cluster_of


def parse_m8(path: str | Path) -> list[tuple[str, str, float]]:
    """Parse MMseqs2 ``--format-output query,target,pident`` hits."""
    hits: list[tuple[str, str, float]] = []
    path = Path(path)
    if not path.exists():
        return hits
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError(f"malformed m8 row: {line!r}")
        hits.append((fields[0], fields[1], float(fields[2])))
    return hits


def mmseqs_available(binary: str = "mmseqs") -> bool:
    """Return whether an MMseqs2 executable is on ``PATH``."""
    return shutil.which(binary) is not None


def _run(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(cmd)}):\n{proc.stderr[-2000:]}")


def cluster_sequences(
    fasta: str | Path,
    work_dir: str | Path,
    *,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    threads: int = 4,
    binary: str = "mmseqs",
) -> dict[str, str]:
    """Cluster sequences with MMseqs2 connected components.

    Returns ``member -> representative`` for every sequence in ``fasta``.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / "clusters"
    tmp = work / "tmp_cluster"
    _run(
        [
            binary,
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(tmp),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "--cluster-mode",
            "1",
            "--threads",
            str(threads),
            "--remove-tmp-files",
            "1",
        ]
    )
    return parse_cluster_tsv(Path(f"{prefix}_cluster.tsv"))


def search_hits(
    query_fasta: str | Path,
    target_fasta: str | Path,
    work_dir: str | Path,
    *,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    threads: int = 4,
    binary: str = "mmseqs",
) -> list[tuple[str, str, float]]:
    """Search query against target and return ``(query, target, pident)`` hits."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_m8 = work / "hits.m8"
    tmp = work / "tmp_search"
    _run(
        [
            binary,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(out_m8),
            str(tmp),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "--format-output",
            "query,target,pident",
            "--threads",
            str(threads),
            "--remove-tmp-files",
            "1",
        ]
    )
    return parse_m8(out_m8)


def split_by_cluster(
    ids: Sequence[str],
    cluster_of: Mapping[str, str],
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Split rows by cluster, never by individual sequence.

    Whole clusters are assigned to either train or test. Returns boolean masks
    aligned with ``ids`` plus a summary dictionary.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1")

    ids_array = np.asarray(list(ids), dtype=object)
    if len(ids_array) == 0:
        raise ValueError("cannot split an empty sequence list")

    missing = [str(item) for item in ids_array if item not in cluster_of]
    if missing:
        raise KeyError(f"{len(missing)} ids missing from cluster map, e.g. {missing[:3]}")

    clusters = np.asarray([cluster_of[str(item)] for item in ids_array], dtype=object)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    sizes = Counter(clusters.tolist())
    target = max(1, int(round(len(ids_array) * test_frac)))

    test_clusters: set[str] = set()
    n_test = 0
    for cluster in unique:
        if n_test >= target:
            break
        test_clusters.add(str(cluster))
        n_test += sizes[cluster]

    is_test = np.asarray([str(cluster) in test_clusters for cluster in clusters])
    is_train = ~is_test

    overlap = set(clusters[is_train].tolist()) & set(clusters[is_test].tolist())
    if overlap:
        raise AssertionError(f"cluster leakage between splits: {sorted(overlap)[:3]}")

    info: dict[str, object] = {
        "n_sequences": int(len(ids_array)),
        "n_clusters": int(len(unique)),
        "n_train": int(is_train.sum()),
        "n_test": int(is_test.sum()),
        "clusters_train": int(len(unique) - len(test_clusters)),
        "clusters_test": int(len(test_clusters)),
        "test_frac_requested": float(test_frac),
        "test_frac_achieved": float(is_test.mean()),
        "seed": int(seed),
    }
    return is_train, is_test, info


def exclude_clusters(
    cluster_of: Mapping[str, str],
    ids: Iterable[str],
) -> set[str]:
    """Return clusters containing any of ``ids``."""
    return {cluster_of[item] for item in ids if item in cluster_of}


def merge_clusters(
    cluster_of: Mapping[str, str],
    pairs: Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Merge clusters connected by ``pairs`` using union-find.

    Used to repair approximate clustering: any cross-split search hit means the
    two sequences belong to the same homology cluster, so their clusters are
    unioned before the split is recomputed.
    """
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            parent[item], item = root, parent[item]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left, right in pairs:
        if left in cluster_of and right in cluster_of:
            union(cluster_of[left], cluster_of[right])

    return {sequence: find(cluster) for sequence, cluster in cluster_of.items()}


def fetch_uniprot_fasta(
    entries: Sequence[str],
    out_path: str | Path,
    *,
    timeout: int = 30,
    opener: Callable[[str, int], object] | None = None,
) -> Path:
    """Fetch entry-name FASTA records from UniProt and cache them locally.

    Records are renamed to ``ASSAY_<entry>`` so they can be clustered together
    with Swiss-Prot sequences and excluded as whole clusters.
    """
    out = Path(out_path)
    if out.exists():
        cached = {record_id for record_id, _ in read_fasta(out)}
        if all(f"{ASSAY_PREFIX}{entry}" in cached for entry in entries):
            return out

    if opener is None:

        def opener(url: str, timeout_s: int):  # type: ignore[no-redef]
            return urllib.request.urlopen(url, timeout=timeout_s)

    records: list[tuple[str, str]] = []
    for entry in entries:
        url = f"https://rest.uniprot.org/uniprotkb/{entry}.fasta"
        try:
            response = opener(url, timeout)
            raw = response.read().decode("utf-8")  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - surface the entry name
            raise RuntimeError(f"failed to fetch UniProt entry {entry}: {exc}") from exc
        parsed = parse_fasta_text(raw)
        if not parsed:
            raise RuntimeError(f"empty FASTA returned for UniProt entry {entry}")
        records.append((f"{ASSAY_PREFIX}{entry}", parsed[0][1]))

    return write_fasta(records, out)
