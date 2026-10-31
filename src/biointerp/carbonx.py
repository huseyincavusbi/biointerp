"""Carbon-X prompt harness: English instructions + DNA in one chat stream."""

from __future__ import annotations

import re
from typing import Iterable, List

import torch

from .dna_task import variant_token_index

DNA_TOKEN = re.compile(r"^[ACGTN]{1,6}$")
CONDITIONS = (
    "dna_only",
    "correct_gene",
    "wrong_gene",
    "irrelevant_text",
    "english_only",
)


def build_messages(
    ref: str,
    alt: str,
    condition: str = "dna_only",
    correct_gene: str = "BRCA1",
    wrong_gene: str = "TP53",
) -> List[dict]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    dna = f"Reference: <dna>{ref}</dna>\nVariant: <dna>{alt}</dna>"
    question = "Is this variant likely benign or pathogenic? Answer briefly."
    if condition == "english_only":
        content = f"Is this {correct_gene} variant likely benign or pathogenic? Answer briefly."
    elif condition == "dna_only":
        content = f"{dna}\n\n{question}"
    elif condition == "correct_gene":
        content = f"Gene: {correct_gene}\n{dna}\n\n{question}"
    elif condition == "wrong_gene":
        content = f"Gene: {wrong_gene}\n{dna}\n\n{question}"
    else:
        content = f"Context: the weather is mild today.\n{dna}\n\n{question}"
    return [{"role": "user", "content": content}]


def encode_chat(tokenizer, messages: List[dict], **kwargs):
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
        **kwargs,
    )


def _token_length(token: str) -> int:
    text = str(token)
    if (text.startswith("<") and text.endswith(">")) or (
        text.startswith("[") and text.endswith("]")
    ):
        return 0
    return len(DNA_TOKEN.match(text.upper()).group(0)) if DNA_TOKEN.match(text.upper()) else 0


def dna_segments(tokenizer, input_ids) -> List[dict]:
    """Return DNA segments with token positions and base spans.

    Carbon-X wraps every DNA run as ``<dna>...</dna>``. A prompt with a
    reference and alternate sequence therefore yields two segments.
    """
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    segments: List[dict] = []
    current = None
    cursor = 0
    for index, token in enumerate(tokens):
        text = str(token)
        if text == "<dna>":
            current = {"start_token": index + 1, "end_token": index + 1, "spans": []}
            cursor = 0
            continue
        if text == "</dna>":
            if current is not None:
                current["end_token"] = index
                segments.append(current)
                current = None
            continue
        if current is None:
            continue
        length = _token_length(text)
        if length:
            current["spans"].append((cursor, cursor + length))
            cursor += length
    if current is not None:
        current["end_token"] = len(tokens)
        segments.append(current)
    return segments


def variant_indices(segments: Iterable[dict], offset: int) -> tuple[int, int]:
    """Return ``(ref_token_index, alt_token_index)`` for a variant offset."""
    segments = list(segments)
    if len(segments) < 2:
        raise ValueError("expected reference and alternate DNA segments")
    return (
        variant_token_index(segments[0]["spans"], offset),
        variant_token_index(segments[1]["spans"], offset),
    )


def dna_token_positions(segments: Iterable[dict]) -> List[int]:
    """Return all token positions that belong to DNA segments."""
    positions = []
    for segment in segments:
        positions.extend(range(segment["start_token"], segment["end_token"]))
    return positions


def context_token_positions(tokenizer, input_ids, segments: Iterable[dict]) -> List[int]:
    """Return non-DNA, non-special token positions (English context)."""
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    dna_positions = set(dna_token_positions(segments))
    positions = []
    for index, (token_id, token) in enumerate(zip(input_ids, tokens)):
        if index in dna_positions or int(token_id) in special:
            continue
        if str(token) in ("<dna>", "</dna>"):
            continue
        positions.append(index)
    return positions


def reduce_attention(
    attentions: Iterable[torch.Tensor],
    query_positions: Iterable[int],
    key_positions: Iterable[int],
):
    """Average attention from query positions to keys, per layer and head."""
    query_positions = list(query_positions)
    key_positions = list(key_positions)
    if not query_positions or not key_positions:
        return None
    parts = []
    for attention in attentions:
        queries = torch.tensor(query_positions, dtype=torch.long, device=attention.device)
        keys = torch.tensor(key_positions, dtype=torch.long, device=attention.device)
        selected = attention[0][:, queries][:, :, keys]
        parts.append(selected.mean(dim=1))
    return torch.stack(parts).float().numpy()
