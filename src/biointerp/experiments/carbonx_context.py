"""Prompt harness for Carbon-X: paired English/DNA conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from biointerp.carbonx import CONDITIONS, build_messages, dna_segments, encode_chat, variant_indices
from biointerp.core import RESULTS_DIR
from biointerp.dna import extract_cett_causal, get_layers, precompute_col_norms
from biointerp.models import get_spec, load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", required=True)
    p.add_argument("--alt", required=True)
    p.add_argument("--condition", default="all", choices=("all", *CONDITIONS))
    p.add_argument("--correct-gene", default="BRCA1")
    p.add_argument("--wrong-gene", default="TP53")
    p.add_argument("--model", default="HuggingFaceBio/Carbon-X")
    p.add_argument("--model-key", default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    if args.dry_run:
        for condition in conditions:
            messages = build_messages(
                args.ref,
                args.alt,
                condition,
                correct_gene=args.correct_gene,
                wrong_gene=args.wrong_gene,
            )
            print(f"=== {condition} ===")
            print(messages[0]["content"])
            print()
        return

    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major < 5:
        raise SystemExit("Carbon-X requires transformers>=5; use a separate environment")

    model_name = args.model_key or args.model
    spec = get_spec(model_name)
    model, tokenizer, device = load_model(model_name, args.device)
    model.eval()
    layers = list(range(len(get_layers(model, spec["adapter"]))))
    col_norms = precompute_col_norms(model, spec["adapter"], layers)

    offset = next((i for i, (a, b) in enumerate(zip(args.ref, args.alt)) if a != b), None)
    if offset is None:
        raise SystemExit("reference and alternate sequences are identical")

    rows = []
    cett_store = {}
    for condition in conditions:
        messages = build_messages(
            args.ref,
            args.alt,
            condition,
            correct_gene=args.correct_gene,
            wrong_gene=args.wrong_gene,
        )
        encoded = encode_chat(tokenizer, messages)
        input_ids = encoded["input_ids"]
        segments = dna_segments(tokenizer, input_ids)
        ref_token = alt_token = None
        cett = None
        if len(segments) >= 2:
            ref_token, alt_token = variant_indices(segments, offset)
            token_positions = torch.tensor([[ref_token]], dtype=torch.long, device=device)
            tokens = {key: value.to(device) for key, value in encoded.items()}
            cett = extract_cett_causal(
                model, tokens, token_positions, layers, col_norms, spec["adapter"]
            )["cett"][0].numpy()
            cett_store[condition] = cett

        tokens = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            generated[0][input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        rows.append(
            {
                "condition": condition,
                "prompt": messages[0]["content"],
                "answer": answer,
                "ref_token_index": ref_token,
                "alt_token_index": alt_token,
                "n_dna_segments": len(segments),
                "cett_shape": None if cett is None else list(cett.shape),
            }
        )
        print(f"{condition:>16s}: {answer[:160]}")

    out = Path(args.out) if args.out else RESULTS_DIR / "carbonx_context.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if cett_store:
        np.savez_compressed(out.with_suffix(".cett.npz"), **cett_store)
    out.write_text(
        json.dumps(
            {
                "model": model_name,
                "ref": args.ref,
                "alt": args.alt,
                "offset": offset,
                "conditions": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
