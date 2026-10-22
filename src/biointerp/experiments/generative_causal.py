"""Generative causal steering on ProtGPT2 (Bio-Interp generative phase).

Calibration: direction d_l = mean(h_l | bad residues) - mean(h_l | good residues) at the
generated position. Intervention: steer h_l += alpha * d_l at every generation step;
measure the mean ESM-2 naturalness of generated sequences vs baseline / random control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET, aa_token_ids
from biointerp.esm import load_model as load_esm
from biointerp.experiments.generative_error_neurons import esm_naturalness
from biointerp.gen import generate_with_cett, load_generative


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-model", default="nferruz/ProtGPT2")
    p.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--calib-seqs", type=int, default=8)
    p.add_argument("--eval-seqs", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--layers", default="24,30")
    p.add_argument("--alphas", default="-0.3,-0.1,0.1,0.3", help="relative to hidden norm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def step_labels(texts, nat, aa_index):
    labels, valid = [], []
    pos = 0
    for i, text in enumerate(texts):
        res = [c for c in text if c in aa_index]
        if not res:
            continue
        vals = nat[pos : pos + len(res)]
        pos += len(res)
        if len(vals) == 0:
            continue
        labels.append(float(vals.mean()))
        valid.append(i)
    return labels, valid


def generate_and_label(
    gen_model,
    gen_tok,
    esm_model,
    esm_tok,
    aa_t,
    aa_index,
    args,
    device,
    seed_offset,
    steer_map=None,
    capture_layers=None,
):
    out = generate_with_cett(
        gen_model,
        gen_tok,
        n_steps=args.n_steps,
        seed=args.seed + seed_offset,
        temperature=args.temperature,
        greedy=False,
        device=device,
        steer_map=steer_map,
        capture_layers=capture_layers,
    )
    if capture_layers:
        ids, cetts, texts, confs, ents, hidden = out
    else:
        ids, cetts, texts, confs, ents = out
        hidden = None
    seq = "".join(texts)
    clean = "".join(c for c in seq if c in aa_index)
    if len(clean) < 10:
        return None
    nat = esm_naturalness(esm_model, esm_tok, clean, device, aa_t, aa_index)
    labels, valid = step_labels(texts, nat, aa_index)
    return {"labels": labels, "valid": valid, "hidden": hidden, "seq": clean}


def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    aa_index = {c: i for i, c in enumerate(AA_ALPHABET)}

    gen_model, gen_tok, device = load_generative(args.gen_model, args.device)
    esm_model, esm_tok, _ = load_esm(args.esm_model, args.device)
    aa_t = torch.tensor(aa_token_ids(esm_tok), device=device)

    t0 = time.time()
    hidden_by_layer = {l: [] for l in layers}
    all_labels = []
    for si in range(args.calib_seqs):
        res = generate_and_label(
            gen_model,
            gen_tok,
            esm_model,
            esm_tok,
            aa_t,
            aa_index,
            args,
            device,
            seed_offset=si,
            capture_layers=layers,
        )
        if res is None:
            continue
        for li, step_i in enumerate(res["valid"]):
            all_labels.append(res["labels"][li])
            for l in layers:
                hidden_by_layer[l].append(res["hidden"][l][step_i].numpy())
        print(
            f"calib {si + 1}/{args.calib_seqs} | steps {len(all_labels)} | {time.time() - t0:.0f}s"
        )

    labels_arr = np.array(all_labels)
    threshold = float(np.median(labels_arr))
    directions = {}
    mean_norms = {}
    for l in layers:
        H = np.stack(hidden_by_layer[l])
        bad = H[labels_arr < threshold].mean(axis=0)
        good = H[labels_arr >= threshold].mean(axis=0)
        directions[l] = bad - good
        mean_norms[l] = float(np.linalg.norm(H, axis=1).mean())
        print(
            f"layer {l}: ||d|| = {np.linalg.norm(directions[l]):.4f} | mean ||h|| = {mean_norms[l]:.4f} (n={len(H)})"
        )
    print(f"naturalness threshold (median): {threshold:.4f}\n")

    def evaluate(name, steer_map=None):
        vals = []
        for si in range(args.eval_seqs):
            res = generate_and_label(
                gen_model,
                gen_tok,
                esm_model,
                esm_tok,
                aa_t,
                aa_index,
                args,
                device,
                seed_offset=1000 + si,
                steer_map=steer_map,
            )
            if res is not None:
                vals.extend(res["labels"])
        mean_nat = float(np.mean(vals)) if vals else float("nan")
        print(f"{name:>22s} mean naturalness {mean_nat:.4f} (n={len(vals)})")
        return {"condition": name, "mean_naturalness": round(mean_nat, 4), "n_steps": len(vals)}

    rows = [evaluate("baseline")]
    for l in layers:
        for a_rel in alphas:
            a_eff = a_rel * mean_norms[l] / (np.linalg.norm(directions[l]) + 1e-8)
            rows.append(evaluate(f"l{l}_rel{a_rel:g}", {l: (directions[l], a_eff)}))
    rng = np.random.default_rng(args.seed)
    rand_dir = rng.normal(size=gen_model.config.n_embd).astype(np.float32)
    rand_dir = rand_dir / np.linalg.norm(rand_dir) * np.linalg.norm(directions[layers[-1]])
    a_eff_rand = 0.3 * mean_norms[layers[-1]] / (np.linalg.norm(rand_dir) + 1e-8)
    rows.append(evaluate(f"l{layers[-1]}_random_rel0.3", {layers[-1]: (rand_dir, a_eff_rand)}))

    out = RESULTS_DIR / "generative_causal.json"
    out.write_text(
        json.dumps(
            {
                "gen_model": args.gen_model,
                "calib_seqs": args.calib_seqs,
                "eval_seqs": args.eval_seqs,
                "n_steps": args.n_steps,
                "direction_norms": {
                    str(l): round(float(np.linalg.norm(directions[l])), 4) for l in layers
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
