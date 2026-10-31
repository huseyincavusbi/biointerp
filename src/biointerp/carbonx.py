"""Carbon-X prompt harness: English instructions + DNA in one chat stream."""

from __future__ import annotations

import math
import re
from typing import Iterable, List

import torch

from .dna_task import variant_token_index

DNA_TOKEN = re.compile(r"^[ACGTN]{1,6}$")
PATHOGENIC = re.compile(r"\b(likely pathogenic|pathogenic)\b")
BENIGN = re.compile(r"\b(likely benign|benign)\b")
UNCERTAIN = re.compile(r"\b(uncertain|vus|conflicting)\b")
NEGATION = re.compile(r"\b(not|no|never|without|cannot|unlikely)\b")
CONDITIONS = (
    "dna_only",
    "correct_gene",
    "wrong_gene",
    "irrelevant_text",
    "english_only",
)
QUESTION = "Is this variant likely benign or pathogenic? Answer briefly."
FORCED_CHOICE = "Answer with exactly one word: benign or pathogenic."


def _negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 28) : start]
    clause = re.split(r"[,.;:]|\bbut\b", prefix)[-1]
    return NEGATION.search(clause) is not None


def parse_answer(text: str) -> dict:
    """Map a free-form Carbon-X answer to a deterministic label."""
    lower = " ".join(str(text).lower().split())
    matches = []
    for label, pattern in (
        ("pathogenic", PATHOGENIC),
        ("benign", BENIGN),
        ("uncertain", UNCERTAIN),
    ):
        for match in pattern.finditer(lower):
            matches.append((match.start(), label, match.group(0)))
    if not matches:
        return {"label": "unparsed", "matched": None, "negated": False}

    labels = set()
    matched = None
    negated = False
    for start, label, phrase in sorted(matches):
        flipped = _negated(lower, start)
        final = label
        if flipped and label == "pathogenic":
            final = "benign"
        elif flipped and label == "benign":
            final = "pathogenic"
        labels.add(final)
        if matched is None:
            matched = phrase
            negated = flipped
    if len(labels) > 1:
        return {"label": "ambiguous", "matched": matched, "negated": negated}
    return {"label": labels.pop(), "matched": matched, "negated": negated}


def build_messages(
    ref: str,
    alt: str,
    condition: str = "dna_only",
    correct_gene: str = "BRCA1",
    wrong_gene: str = "TP53",
    question: str = QUESTION,
) -> List[dict]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    dna = f"Reference: <dna>{ref}</dna>\nVariant: <dna>{alt}</dna>"
    if condition == "english_only":
        content = f"Gene: {correct_gene}\n\n{question}"
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
    if not getattr(tokenizer, "chat_template", None):
        text = "\n\n".join(message["content"] for message in messages if message["role"] == "user")
        fallback = {
            key: value
            for key, value in kwargs.items()
            if key not in ("add_generation_prompt", "enable_thinking")
        }
        return tokenizer(text, return_tensors="pt", return_dict=True, **fallback)
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
        **kwargs,
    )


@torch.inference_mode()
def continuation_logprob(model, tokens: dict, continuation_ids) -> float:
    """Return log p(continuation | prompt) for one token sequence."""
    input_ids = tokens["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = tokens.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    continuation = torch.tensor([list(continuation_ids)], dtype=torch.long, device=input_ids.device)
    full_ids = torch.cat([input_ids, continuation], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(continuation)], dim=1)
    forward = dict(tokens)
    forward["input_ids"] = full_ids
    forward["attention_mask"] = full_mask
    logits = model(**forward).logits.float()
    start = input_ids.shape[1]
    selected = logits[:, start - 1 : start - 1 + continuation.shape[1], :]
    logp = torch.log_softmax(selected, dim=-1)
    indices = torch.arange(continuation.shape[1], device=input_ids.device)
    return float(logp[0, indices, continuation[0]].sum())


def pathogenic_probability(model, tokenizer, tokens: dict) -> dict:
    """Compare the log-probability of ``pathogenic`` and ``benign``."""
    positive = tokenizer.encode(" pathogenic", add_special_tokens=False)
    negative = tokenizer.encode(" benign", add_special_tokens=False)
    if not positive or not negative:
        raise ValueError("tokenizer produced an empty answer continuation")
    logp_positive = continuation_logprob(model, tokens, positive)
    logp_negative = continuation_logprob(model, tokens, negative)
    high = max(logp_positive, logp_negative)
    log_denom = high + math.log(math.exp(logp_positive - high) + math.exp(logp_negative - high))
    return {
        "p_pathogenic": math.exp(logp_positive - log_denom),
        "logp_pathogenic": logp_positive,
        "logp_benign": logp_negative,
    }


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
