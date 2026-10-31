"""Patch Carbon-X DNA hidden states between prompt conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from biointerp.carbonx import (
    CONDITIONS,
    build_messages,
    dna_segments,
    encode_chat,
    variant_indices,
)
from biointerp.causal import hidden_states_at, patch_hidden, patch_hidden_context
from biointerp.core import RESULTS_DIR
from biointerp.dna import get_layers
from biointerp.models import get_spec, load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", required=True)
    p.add_argument("--alt", required=True)
    p.add_argument("--source", default="correct_gene", choices=CONDITIONS)
    p.add_argument("--target", default="dna_only", choices=CONDITIONS)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--correct-gene", default="BRCA1")
    p.add_argument("--wrong-gene", default="TP53")
    p.add_argument("--model-key", default="carbon_x_4bit")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _encode(tokenizer, ref, alt, condition, correct_gene, wrong_gene):
    messages = build_messages(
        ref,
        alt,
        condition,
        correct_gene=correct_gene,
        wrong_gene=wrong_gene,
    )
    encoded = encode_chat(tokenizer, messages)
    segments = dna_segments(tokenizer, encoded["input_ids"])
    if len(segments) < 2:
        raise SystemExit(f"condition {condition!r} has no reference/alternate DNA")
    offset = next((i for i, (a, b) in enumerate(zip(ref, alt)) if a != b), None)
    if offset is None:
        raise SystemExit("reference and alternate sequences are identical")
    ref_token, _ = variant_indices(segments, offset)
    return encoded, segments, ref_token, offset


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


def run_patch(
    model,
    tokenizer,
    spec,
    ref,
    alt,
    source_condition,
    target_condition,
    layer,
    correct_gene,
    wrong_gene,
    max_new_tokens,
    device,
):
    source, _, source_token, offset = _encode(
        tokenizer, ref, alt, source_condition, correct_gene, wrong_gene
    )
    target, _, target_token, target_offset = _encode(
        tokenizer, ref, alt, target_condition, correct_gene, wrong_gene
    )
    if offset != target_offset:
        raise ValueError("source and target variant offsets differ")

    n_layers = len(get_layers(model, spec["adapter"]))
    layer = layer % n_layers
    source_tokens = {k: v.to(device) for k, v in source.items()}
    target_tokens = {k: v.to(device) for k, v in target.items()}
    source_positions = torch.tensor([[source_token]], dtype=torch.long, device=device)
    target_positions = torch.tensor([[target_token]], dtype=torch.long, device=device)

    donor = hidden_states_at(model, source_tokens, source_positions, [layer], spec["adapter"])[
        layer
    ]
    with torch.no_grad():
        base_logits = model(**target_tokens).logits[0, target_token].detach().float().cpu()
    patched_logits = (
        patch_hidden(
            model,
            target_tokens,
            layer,
            target_positions,
            donor,
            spec["adapter"],
        )
        .logits[0, target_token]
        .detach()
        .float()
        .cpu()
    )

    answer = _generate(model, tokenizer, target_tokens, max_new_tokens)
    with patch_hidden_context(
        model, target_tokens, layer, target_positions, donor, spec["adapter"]
    ):
        patched_answer = _generate(model, tokenizer, target_tokens, max_new_tokens)
    random_donor = torch.randn_like(donor)
    with patch_hidden_context(
        model, target_tokens, layer, target_positions, random_donor, spec["adapter"]
    ):
        random_answer = _generate(model, tokenizer, target_tokens, max_new_tokens)

    return {
        "offset": offset,
        "source": source_condition,
        "target": target_condition,
        "layer": layer,
        "ref_token_index": target_token,
        "logits": {
            "base_norm": float(base_logits.norm()),
            "patched_norm": float(patched_logits.norm()),
            "delta_l2": float((patched_logits - base_logits).norm()),
        },
        "answers": {
            "target": answer,
            "patched": patched_answer,
            "random_donor": random_answer,
        },
    }


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model_key)
    model, tokenizer, device = load_model(args.model_key, args.device)
    model.eval()
    result = run_patch(
        model,
        tokenizer,
        spec,
        args.ref,
        args.alt,
        args.source,
        args.target,
        args.layer,
        args.correct_gene,
        args.wrong_gene,
        args.max_new_tokens,
        device,
    )
    result["model"] = args.model_key
    result["ref"] = args.ref
    result["alt"] = args.alt
    out = Path(args.out) if args.out else RESULTS_DIR / "carbonx_patch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"base: {result['answers']['target'][:160]}")
    print(f"patched: {result['answers']['patched'][:160]}")
    print(f"random: {result['answers']['random_donor'][:160]}")
    print(f"delta_l2: {result['logits']['delta_l2']:.4f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
