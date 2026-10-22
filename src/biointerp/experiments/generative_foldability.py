"""Foldability-scored generative causal steering (Bio-Interp, Exp 2 in RQ).

Replica of generative_causal (ProtGPT2 steering) with ESMFold pLDDT as THE label
for calibration AND evaluation (was: ESM-2 naturalness proxy).

Calibration: direction d_l = mean(h_l | low-pLDDT residues) - mean(h_l |
high-pLDDT residues) at the generated position (per-residue pLDDT from folding
the generated sequence). Intervention: steer h_l += alpha * d_l every step;
report mean pLDDT vs baseline / random control. Pre-registered: steer-away
raises pLDDT dose-response; random direction ~= baseline.

Speed: generate-all-then-fold-batched. Folds run in a small thread pool
(default 4 workers) with per-attempt timeouts, retries, and rate-limit
backoff (429 -> >=60s). Generation order (RNG) is untouched by fold order.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR
from biointerp.esm import AA_ALPHABET
from biointerp.experiments.viability_scorer import fold_plddt
from biointerp.gen import generate_with_cett, load_generative


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-model", default="nferruz/ProtGPT2")
    p.add_argument("--fold-model", default="esmfold2")
    p.add_argument("--api-key-env", default="BIOHUB_API_KEY")
    p.add_argument("--calib-seqs", type=int, default=8)
    p.add_argument("--eval-seqs", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--layers", default="24,30")
    p.add_argument("--alphas", default="-0.3,-0.1,0.1,0.3", help="relative to hidden norm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--fold-workers", type=int, default=4)
    p.add_argument("--out", default=str(RESULTS_DIR / "generative_foldability.json"))
    return p.parse_args()


def step_labels(texts, per_res, aa_index):
    labels, valid = [], []
    pos = 0
    for i, text in enumerate(texts):
        res = [c for c in text if c in aa_index]
        if not res:
            continue
        vals = per_res[pos : pos + len(res)]
        pos += len(res)
        if len(vals) == 0:
            continue
        labels.append(float(vals.mean()))
        valid.append(i)
    return labels, valid


def fold_retry(clean, client, timeout_s=300, tries=4):
    """fold_plddt with per-attempt timeout + retries + rate-limit backoff."""
    last = None
    for a in range(tries):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(fold_plddt, clean, client).result(timeout=timeout_s)
        except Exception as e:
            last = e
            wait = min(180, 15 * 2**a)
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "too many" in msg:
                wait = max(wait, 60)
            print(
                f"  fold attempt {a + 1}/{tries} failed ({type(e).__name__}), "
                f"waiting {wait}s (rate-limit respectful)...",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"fold failed after {tries} tries: {last}")


def fold_many(cleans, client, workers=4):
    """Fold a list of sequences in a thread pool; None for sequences that
    fail after retries. Order-preserving."""

    def one(s):
        try:
            return fold_retry(s, client)
        except RuntimeError as e:
            print(f"  giving up on one sequence: {e}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, cleans))


def gen_one(
    gen_model, gen_tok, aa_index, args, device, seed_offset, steer_map=None, capture_layers=None
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
    return {"texts": texts, "hidden": hidden, "clean": clean}


def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    aa_index = {c: i for i, c in enumerate(AA_ALPHABET)}

    from esm.sdk.forge import SequenceStructureForgeInferenceClient

    token = os.environ.get(args.api_key_env, "")
    if not token:
        raise SystemExit(f"missing API key in ${args.api_key_env}")
    client = SequenceStructureForgeInferenceClient(model=args.fold_model, token=token)
    gen_model, gen_tok, device = load_generative(args.gen_model, args.device)

    t0 = time.time()
    # --- calibration: generate all (fast, local), then fold batched ---
    print("calibration: generating...", flush=True)
    gens = []
    for si in range(args.calib_seqs):
        g = gen_one(
            gen_model, gen_tok, aa_index, args, device, seed_offset=si, capture_layers=layers
        )
        gens.append(g)
    print(
        f"calibration: generated {sum(g is not None for g in gens)}/{args.calib_seqs}, "
        f"folding batched (workers={args.fold_workers})...",
        flush=True,
    )
    cleans = [g["clean"] if g else "" for g in gens]
    folds = fold_many([c for c in cleans if c], client, args.fold_workers)
    it = iter(folds)
    hidden_by_layer = {l: [] for l in layers}
    all_labels = []
    for g, c in zip(gens, cleans):
        if g is None or not c:
            continue
        plddt = next(it)
        if plddt is None:
            continue
        L = min(len(c), len(plddt))
        labels, valid = step_labels(g["texts"], plddt[:L], aa_index)
        for li, step_i in enumerate(valid):
            all_labels.append(labels[li])
            for l in layers:
                hidden_by_layer[l].append(g["hidden"][l][step_i].numpy())
    print(f"calib done | steps {len(all_labels)} | {time.time() - t0:.0f}s", flush=True)

    labels_arr = np.array(all_labels)
    threshold = float(np.median(labels_arr))
    directions, mean_norms = {}, {}
    for l in layers:
        H = np.stack(hidden_by_layer[l])
        bad = H[labels_arr < threshold].mean(axis=0)
        good = H[labels_arr >= threshold].mean(axis=0)
        directions[l] = bad - good
        mean_norms[l] = float(np.linalg.norm(H, axis=1).mean())
        print(
            f"layer {l}: ||d|| = {np.linalg.norm(directions[l]):.4f} | "
            f"mean ||h|| = {mean_norms[l]:.4f} (n={len(H)})"
        )
    print(f"pLDDT threshold (median): {threshold:.4f}\n")

    def evaluate(name, steer_map=None):
        gs = []
        for si in range(args.eval_seqs):
            g = gen_one(
                gen_model,
                gen_tok,
                aa_index,
                args,
                device,
                seed_offset=1000 + si,
                steer_map=steer_map,
            )
            if g is not None:
                gs.append(g)
        vals = []
        for g, pl in zip(gs, fold_many([g["clean"] for g in gs], client, args.fold_workers)):
            if pl is None:
                continue
            L = min(len(g["clean"]), len(pl))
            labels, _ = step_labels(g["texts"], pl[:L], aa_index)
            vals.extend(labels)
        mean_pl = float(np.mean(vals)) if vals else float("nan")
        print(
            f"{name:>22s} mean pLDDT {mean_pl:.4f} (n={len(vals)}) | {time.time() - t0:.0f}s",
            flush=True,
        )
        return {"condition": name, "mean_plddt": round(mean_pl, 4), "n_steps": len(vals)}

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

    open(args.out, "w").write(
        json.dumps(
            {
                "gen_model": args.gen_model,
                "fold_model": args.fold_model,
                "calib_seqs": args.calib_seqs,
                "eval_seqs": args.eval_seqs,
                "n_steps": args.n_steps,
                "fold_workers": args.fold_workers,
                "plddt_threshold": round(threshold, 4),
                "direction_norms": {
                    str(l): round(float(np.linalg.norm(directions[l])), 4) for l in layers
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved: {args.out} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
