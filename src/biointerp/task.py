"""Masked-residue task: Swiss-Prot sequences, random masking, correctness labels."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

UNIPROT_STREAM = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?format=fasta&query=(reviewed:true)%20AND%20(length:[{min_len}%20TO%20{max_len}])"
)


def _read_fasta(path: Path) -> List[str]:
    sequences: List[str] = []
    parts: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
                parts = []
        else:
            parts.append(line)
    if parts:
        sequences.append("".join(parts))
    return sequences


def fetch_swissprot(
    n: int,
    min_len: int = 50,
    max_len: int = 500,
    seed: int = 0,
    pool: int = 20000,
    cache: bool = True,
) -> List[str]:
    """Fetch n reviewed UniProt sequences, reservoir-sampled over a streamed pool."""
    cache_path = DATA_DIR / f"swissprot_n{n}_len{min_len}-{max_len}_seed{seed}.fasta"
    if cache and cache_path.exists():
        return _read_fasta(cache_path)

    import requests

    url = UNIPROT_STREAM.format(min_len=min_len, max_len=max_len)
    rng = random.Random(seed)
    reservoir: List[str] = []
    eligible = 0
    parts: List[str] = []

    def consider(seq: str) -> bool:
        nonlocal eligible
        if not (min_len <= len(seq) <= max_len):
            return False
        eligible += 1
        if len(reservoir) < n:
            reservoir.append(seq)
        else:
            j = rng.randint(0, eligible - 1)
            if j < n:
                reservoir[j] = seq
        return eligible >= pool

    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        done = False
        for raw in resp.iter_lines(decode_unicode=True):
            line = (raw or "").strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    done = consider("".join(parts))
                    parts = []
                    if done:
                        break
            else:
                parts.append(line)
        if parts and not done:
            consider("".join(parts))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if cache:
        with cache_path.open("w") as fh:
            for i, seq in enumerate(reservoir):
                fh.write(f">seq{i}\n{seq}\n")
    return reservoir


def make_masked_batch(
    sequences: List[str],
    tokenizer,
    masks_per_seq: int = 1,
    seed: int = 0,
) -> Tuple[dict, torch.Tensor, List[List[str]]]:
    """Tokenize sequences and mask random interior positions.

    Returns tokens (padded dict), mask_positions (B, M) token indices,
    true_aa (B, M) residue characters.
    """
    rng = random.Random(seed)
    batch_ids: List[List[int]] = []
    positions: List[List[int]] = []
    true_aa: List[List[str]] = []
    for seq in sequences:
        ids = list(tokenizer(seq, add_special_tokens=True)["input_ids"])
        n_masks = min(masks_per_seq, max(0, len(seq) - 2))
        pos = sorted(rng.sample(range(1, len(seq) - 1), n_masks))
        for p in pos:
            ids[p + 1] = tokenizer.mask_token_id
        batch_ids.append(ids)
        positions.append([p + 1 for p in pos])
        true_aa.append([seq[p] for p in pos])

    max_len = max(len(x) for x in batch_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = [x + [pad_id] * (max_len - len(x)) for x in batch_ids]
    attention = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch_ids]
    tokens = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }
    return tokens, torch.tensor(positions, dtype=torch.long), true_aa
