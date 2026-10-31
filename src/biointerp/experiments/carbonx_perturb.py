"""Perturb a DNA feature span and measure Carbon-X answer changes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from biointerp.carbonx import (
    CONDITIONS,
    build_messages,
    encode_chat,
    parse_answer,
)
from biointerp.core import RESULTS_DIR
from biointerp.models import load_model

MUTATIONS = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "A"}


def perturb_sequence(
    sequence: str,
    start: int,
    end: int,
    mode: str,
    rng: random.Random,
    keep: set[int] | None = None,
) -> str | None:
    keep = set() if keep is None else set(keep)
    positions = [i for i in range(start, end) if i not in keep]
    if not positions:
        return None
    chars = list(sequence)
    if mode == "shuffle":
        values = [sequence[i] for i in positions]
        rng.shuffle(values)
        for position, value in zip(positions, values):
            chars[position] = value
    elif mode == "gc_matched":
        for position in positions:
            chars[position] = MUTATIONS.get(sequence[position].upper(), "A")
    elif mode == "random":
        for position in positions:
            choices = [base for base in "ACGT" if base != sequence[position].upper()]
            chars[position] = rng.choice(choices)
    else:
        raise ValueError(f"unknown perturbation mode {mode!r}")
    result = "".join(chars)
    return None if result == sequence else result


def _generate(model, tokenizer, tokens, max_new_tokens):
    with torch.no_grad():
        generated = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        generated[0][tokens["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def _answer(
    model,
    tokenizer,
    ref,
    alt,
    condition,
    args,
    max_new_tokens,
):
    messages = build_messages(
        ref,
        alt,
        condition,
        correct_gene=args.correct_gene,
        wrong_gene=args.wrong_gene,
    )
    encoded = encode_chat(tokenizer, messages)
    tokens = {key: value.to(args.device or "cpu") for key, value in encoded.items()}
    return _generate(model, tokenizer, tokens, max_new_tokens)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", required=True)
    p.add_argument("--alt", required=True)
    p.add_argument("--condition", default="dna_only", choices=CONDITIONS)
    p.add_argument("--start", type=int, required=True, help="0-based span start")
    p.add_argument("--end", type=int, required=True, help="exclusive span end")
    p.add_argument("--mode", default="random", choices=("random", "shuffle", "gc_matched"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-key", default="carbon_x_4bit")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--correct-gene", default="BRCA1")
    p.add_argument("--wrong-gene", default="TP53")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ref = args.ref.upper()
    alt = args.alt.upper()
    if len(ref) != len(alt):
        raise SystemExit("reference and alternate sequences must have equal length")
    offset = next((i for i, (a, b) in enumerate(zip(ref, alt)) if a != b), None)
    if offset is None:
        raise SystemExit("reference and alternate sequences are identical")

    model, tokenizer, device = load_model(args.model_key, args.device)
    model.eval()
    args.device = device

    baseline = _answer(model, tokenizer, ref, alt, args.condition, args, args.max_new_tokens)
    baseline_parsed = parse_answer(baseline)
    rng = random.Random(args.seed)
    perturbations = []
    for i in range(args.n):
        new_ref = perturb_sequence(ref, args.start, args.end, args.mode, rng, keep={offset})
        if new_ref is None:
            continue
        new_alt = list(alt)
        for position in range(args.start, args.end):
            if position != offset:
                new_alt[position] = new_ref[position]
        answer = _answer(
            model,
            tokenizer,
            new_ref,
            "".join(new_alt),
            args.condition,
            args,
            args.max_new_tokens,
        )
        parsed = parse_answer(answer)
        perturbations.append(
            {
                "index": i,
                "ref": new_ref,
                "answer": answer,
                "parsed": parsed,
                "answer_changed": answer.strip() != baseline.strip(),
                "label_changed": parsed["label"] != baseline_parsed["label"],
            }
        )

    summary = {
        "model": args.model_key,
        "condition": args.condition,
        "span": [args.start, args.end],
        "mode": args.mode,
        "baseline": {"answer": baseline, "parsed": baseline_parsed},
        "n": len(perturbations),
        "answer_change_rate": 0.0
        if not perturbations
        else round(
            sum(item["answer_changed"] for item in perturbations) / len(perturbations),
            4,
        ),
        "label_change_rate": 0.0
        if not perturbations
        else round(
            sum(item["label_changed"] for item in perturbations) / len(perturbations),
            4,
        ),
        "perturbations": perturbations,
    }
    out = Path(args.out) if args.out else RESULTS_DIR / "carbonx_perturb.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"baseline: {baseline_parsed['label']} | {baseline[:120]}")
    print(
        f"perturbations: {summary['n']} | "
        f"answer change={summary['answer_change_rate']} | "
        f"label change={summary['label_change_rate']}"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
