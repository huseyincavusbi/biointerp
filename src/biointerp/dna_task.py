"""DNA tokenization, token→base spans, and batch builders for CETT tasks."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import torch

DNA_CHARS = set("ACGTN")


def _as_ids(encoded) -> List[int]:
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def tokenize_dna(sequence: str, tokenizer, adapter: str) -> Tuple[List[int], List[str]]:
    """Tokenize a DNA sequence and return ``(ids, token_strings)``."""
    if adapter == "carbon":
        encoded = tokenizer(f"<dna>{sequence}</dna>", add_special_tokens=False)
    else:
        encoded = tokenizer(sequence, add_special_tokens=True)
    ids = _as_ids(encoded)
    tokens = list(tokenizer.convert_ids_to_tokens(ids))
    return ids, tokens


def _dna_length(token: str) -> int:
    text = str(token)
    if (text.startswith("<") and text.endswith(">")) or (
        text.startswith("[") and text.endswith("]")
    ):
        return 0
    if text.startswith("##"):
        text = text[2:]
    if text.startswith("Ġ"):
        text = text[1:]
    return sum(char in DNA_CHARS for char in text.upper())


def token_spans(sequence: str, tokenizer, adapter: str) -> List[Tuple[int, int] | None]:
    """Map each token to its half-open base span; special tokens map to ``None``."""
    _, tokens = tokenize_dna(sequence, tokenizer, adapter)
    spans: List[Tuple[int, int] | None] = []
    cursor = 0
    for token in tokens:
        length = _dna_length(token)
        if length:
            spans.append((cursor, cursor + length))
            cursor += length
        else:
            spans.append(None)
    if cursor != len(sequence):
        raise ValueError(
            f"tokenizer covered {cursor} bases but sequence has {len(sequence)}; "
            "check the tokenizer/adapter pairing"
        )
    return spans


def dna_token_positions(spans: Sequence[Tuple[int, int] | None]) -> List[int]:
    return [i for i, span in enumerate(spans) if span is not None]


def variant_token_index(spans: Sequence[Tuple[int, int] | None], position: int) -> int:
    """Return the token index containing a 0-based base position."""
    for i, span in enumerate(spans):
        if span is not None and span[0] <= position < span[1]:
            return i
    raise ValueError(f"base position {position} is not covered by any DNA token")


def masked_dna_batch(
    sequences: Sequence[str],
    tokenizer,
    adapter: str,
    masks_per_seq: int = 1,
    seed: int = 0,
):
    """Mask random DNA tokens and return ``(tokens, positions, true_ids, spans)``."""
    if not sequences:
        raise ValueError("sequences must not be empty")
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("tokenizer has no mask_token_id")

    encoded = [tokenize_dna(sequence, tokenizer, adapter) for sequence in sequences]
    span_lists = [token_spans(sequence, tokenizer, adapter) for sequence in sequences]
    dna_positions = [dna_token_positions(spans) for spans in span_lists]
    if any(not positions for positions in dna_positions):
        raise ValueError("every sequence must contain at least one DNA token")

    n_masks = min(masks_per_seq, *(len(positions) for positions in dna_positions))
    if n_masks < 1:
        raise ValueError("masks_per_seq must be at least 1")

    rng = random.Random(seed)
    rows: List[List[int]] = []
    positions: List[List[int]] = []
    true_ids: List[List[int]] = []
    for (ids, _), candidates in zip(encoded, dna_positions):
        chosen = sorted(rng.sample(candidates, n_masks))
        rows.append(list(ids))
        positions.append(chosen)
        true_ids.append([int(ids[p]) for p in chosen])
        for p in chosen:
            rows[-1][p] = int(mask_id)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = 0 if pad_id is None else int(pad_id)
    max_len = max(len(row) for row in rows)
    input_ids = [row + [pad_id] * (max_len - len(row)) for row in rows]
    attention = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]

    tokens = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }
    return (
        tokens,
        torch.tensor(positions, dtype=torch.long),
        torch.tensor(true_ids, dtype=torch.long),
        span_lists,
    )


def causal_dna_batch(sequences: Sequence[str], tokenizer, adapter: str):
    """Build next-token prefixes and targets for a causal DNA model."""
    if not sequences:
        raise ValueError("sequences must not be empty")
    rows: List[List[int]] = []
    positions: List[int] = []
    targets: List[int] = []
    for sequence in sequences:
        ids, _ = tokenize_dna(sequence, tokenizer, adapter)
        spans = token_spans(sequence, tokenizer, adapter)
        candidates = dna_token_positions(spans)
        if not candidates:
            raise ValueError("sequence has no DNA tokens")
        target_pos = candidates[-1]
        prefix = ids[:target_pos]
        if not prefix:
            raise ValueError("sequence has no prefix before the target token")
        rows.append(prefix)
        positions.append(len(prefix) - 1)
        targets.append(int(ids[target_pos]))

    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = 0 if pad_id is None else int(pad_id)
    max_len = max(len(row) for row in rows)
    input_ids = [row + [pad_id] * (max_len - len(row)) for row in rows]
    attention = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
    tokens = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }
    return (
        tokens,
        torch.tensor(positions, dtype=torch.long).unsqueeze(1),
        torch.tensor(targets, dtype=torch.long).unsqueeze(1),
    )
