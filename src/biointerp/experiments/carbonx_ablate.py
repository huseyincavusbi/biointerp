"""Ablate Carbon-X attention heads and measure answer changes."""

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
    dna_segments,
    encode_chat,
    parse_answer,
    variant_indices,
)
from biointerp.causal import ablate_heads, ablate_heads_context
from biointerp.core import RESULTS_DIR
from biointerp.dna import get_layers
from biointerp.models import get_spec, load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", required=True)
    p.add_argument("--alt", required=True)
    p.add_argument("--condition", default="dna_only", choices=CONDITIONS)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--heads", required=True, help="comma-separated head indices")
    p.add_argument("--correct-gene", default="BRCA1")
    p.add_argument("--wrong-gene", default="TP53")
    p.add_argument("--model-key", default="carbon_x_4bit")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _encode(tokenizer, ref, alt, condition, args):
    messages = build_messages(
        ref,
        alt,
        condition,
        correct_gene=args.correct_gene,
        wrong_gene=args.wrong_gene,
    )
    encoded = encode_chat(tokenizer, messages)
    segments = dna_segments(tokenizer, encoded["input_ids"])
    if len(segments) < 2:
        raise SystemExit(f"condition {condition!r} has no reference/alternate DNA")
    offset = next((i for i, (a, b) in enumerate(zip(ref, alt)) if a != b), None)
    if offset is None:
        raise SystemExit("reference and alternate sequences are identical")
    ref_token, _ = variant_indices(segments, offset)
    return encoded, ref_token


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


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model_key)
    model, tokenizer, device = load_model(args.model_key, args.device)
    model.eval()

    encoded, position = _encode(tokenizer, args.ref, args.alt, args.condition, args)
    tokens = {key: value.to(device) for key, value in encoded.items()}
    n_layers = len(get_layers(model, spec["adapter"]))
    layer = args.layer % n_layers
    heads = [int(item) for item in args.heads.split(",") if item.strip()]
    if not heads:
        raise SystemExit("no heads selected")

    with torch.no_grad():
        base_logits = model(**tokens).logits[0, position].detach().float().cpu()
    ablated_logits = (
        ablate_heads(model, tokens, layer, heads, spec["adapter"])
        .logits[0, position]
        .detach()
        .float()
        .cpu()
    )

    answer = _generate(model, tokenizer, tokens, args.max_new_tokens)
    with ablate_heads_context(model, layer, heads, spec["adapter"]):
        ablated_answer = _generate(model, tokenizer, tokens, args.max_new_tokens)

    n_heads = getattr(model.config, "num_attention_heads", None)
    rng = random.Random(args.seed)
    if n_heads:
        candidates = [head for head in range(n_heads) if head not in heads]
        random_heads = sorted(rng.sample(candidates, min(len(heads), len(candidates))))
    else:
        random_heads = []
    if random_heads:
        with ablate_heads_context(model, layer, random_heads, spec["adapter"]):
            random_answer = _generate(model, tokenizer, tokens, args.max_new_tokens)
    else:
        random_answer = None

    result = {
        "model": args.model_key,
        "condition": args.condition,
        "layer": layer,
        "heads": heads,
        "random_heads": random_heads,
        "logits": {
            "base_norm": float(base_logits.norm()),
            "ablated_norm": float(ablated_logits.norm()),
            "delta_l2": float((ablated_logits - base_logits).norm()),
        },
        "answers": {
            "base": answer,
            "ablated": ablated_answer,
            "random_heads": random_answer,
        },
        "parsed": {
            "base": parse_answer(answer),
            "ablated": parse_answer(ablated_answer),
            "random_heads": None if random_answer is None else parse_answer(random_answer),
        },
    }
    out = Path(args.out) if args.out else RESULTS_DIR / "carbonx_ablate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"base: {answer[:160]}")
    print(f"ablated: {ablated_answer[:160]}")
    print(f"random: {random_answer}")
    print(f"delta_l2: {result['logits']['delta_l2']:.4f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
