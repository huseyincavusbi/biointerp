"""Forced-choice p(pathogenic) for Carbon-X prompt conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from biointerp.carbonx import (
    CONDITIONS,
    FORCED_CHOICE,
    build_messages,
    encode_chat,
    pathogenic_probability,
)
from biointerp.core import RESULTS_DIR
from biointerp.models import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", required=True)
    p.add_argument("--alt", required=True)
    p.add_argument("--condition", default="all", choices=("all", *CONDITIONS))
    p.add_argument("--question", default=FORCED_CHOICE)
    p.add_argument("--correct-gene", default="BRCA1")
    p.add_argument("--wrong-gene", default="TP53")
    p.add_argument("--model-key", default="carbon_x_4bit")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    model, tokenizer, device = load_model(args.model_key, args.device)
    model.eval()

    rows = []
    for condition in conditions:
        messages = build_messages(
            args.ref,
            args.alt,
            condition,
            correct_gene=args.correct_gene,
            wrong_gene=args.wrong_gene,
            question=args.question,
        )
        encoded = encode_chat(tokenizer, messages)
        tokens = {key: value.to(device) for key, value in encoded.items()}
        score = pathogenic_probability(model, tokenizer, tokens)
        score["condition"] = condition
        rows.append(score)
        print(
            f"{condition:>16s}: p(pathogenic)={score['p_pathogenic']:.4f} "
            f"logp+={score['logp_pathogenic']:.3f} "
            f"logp-={score['logp_benign']:.3f}",
            flush=True,
        )

    out = Path(args.out) if args.out else RESULTS_DIR / "carbonx_probability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model_key,
                "ref": args.ref,
                "alt": args.alt,
                "question": args.question,
                "conditions": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
