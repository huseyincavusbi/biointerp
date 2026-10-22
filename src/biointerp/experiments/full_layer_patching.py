"""Full-layer (cumulative prefix) patching on ESM-2 (CoTLab full_layer_patching redesigned).

Patch the masked-position residual cumulatively across layer prefixes (0..k) from a donor
(correct / incorrect / self). Measures p(true) and prediction transfer.

Redesign notes: per-condition batched forwards; masked-position intervention; p(true) endpoint.
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
from biointerp.esm import AA_ALPHABET, aa_token_ids, get_transformer_layers, load_model
from biointerp.task import fetch_swissprot, make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--model-key", default=None, help="registry key (overrides --model)")
    p.add_argument("--prefixes", default="11,23,29,32")
    p.add_argument("--donor-n", type=int, default=300)
    p.add_argument("--eval-n", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.inference_mode()
def capture_all(model, tokenizer, sequences, n_layers, batch_size, seed, device, aa_t):
    h_all = [[] for _ in range(n_layers + 1)]
    correct_all, probs_all = [], []
    n_batches = (len(sequences) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch = sequences[bi * batch_size : (bi + 1) * batch_size]
        tokens, positions, true_aa = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=seed + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        pos = positions.to(device)
        b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)
        out = model(**tokens, output_hidden_states=True)
        logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        true_flat = [aa for row in true_aa for aa in row]
        ti = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        probs_all.append(probs)
        correct_all.append(probs.argmax(axis=1) == ti)
        for i in range(n_layers + 1):
            h_all[i].append(out.hidden_states[i][b_idx, pos].squeeze(1).float().cpu().numpy())
    return (
        [np.concatenate(h) for h in h_all],
        np.concatenate(correct_all),
        np.concatenate(probs_all),
    )


@torch.inference_mode()
def run_prefix_patch(
    model, tokenizer, sequences, prefix, donor_vecs, batch_size, seed, device, aa_t
):
    blocks = get_transformer_layers(model)
    probs_all, correct_all = [], []
    n_batches = (len(sequences) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch = sequences[bi * batch_size : (bi + 1) * batch_size]
        tokens, positions, true_aa = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=seed + bi
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        pos = positions.to(device)
        b_idx = torch.arange(pos.shape[0], device=device).unsqueeze(1)
        handles = []
        for layer in range(prefix + 1):
            dv = torch.tensor(
                donor_vecs[layer + 1][bi * batch_size : (bi + 1) * batch_size],
                dtype=torch.float32,
                device=device,
            )

            def make_hook(vecs):
                def hook(module, inputs, output):
                    is_tuple = isinstance(output, tuple)
                    h = output[0] if is_tuple else output
                    h = h.clone()
                    h[b_idx, pos] = vecs.unsqueeze(1)
                    return (h,) + output[1:] if is_tuple else h

                return hook

            handles.append(blocks[layer].register_forward_hook(make_hook(dv)))
        try:
            out = model(**tokens)
        finally:
            for h in handles:
                h.remove()
        logits = out.logits[b_idx, pos][..., aa_t].squeeze(1)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        true_flat = [aa for row in true_aa for aa in row]
        ti = np.array([AA_ALPHABET.index(aa) for aa in true_flat])
        probs_all.append(probs)
        correct_all.append(probs.argmax(axis=1) == ti)
    return np.concatenate(probs_all), np.concatenate(correct_all)


def main() -> None:
    args = parse_args()
    if args.model_key:
        from biointerp.models import resolve_id

        args.model = resolve_id(args.model_key)
    prefixes = [int(x) for x in args.prefixes.split(",")]
    all_seqs = fetch_swissprot(2000, seed=args.seed)
    donor_seqs = all_seqs[500 : 500 + args.donor_n]
    eval_seqs = all_seqs[: args.eval_n]

    model, tokenizer, device = load_model(args.model, args.device)
    n_layers = len(get_transformer_layers(model))
    aa_t = torch.tensor(aa_token_ids(tokenizer), device=device)

    t0 = time.time()
    donor_h, donor_correct, _ = capture_all(
        model, tokenizer, donor_seqs, n_layers, args.batch_size, args.seed + 100, device, aa_t
    )
    print(f"donors: {len(donor_seqs)} ({donor_correct.sum()} correct) | {time.time() - t0:.0f}s")
    recip_h, recip_correct, base_probs = capture_all(
        model, tokenizer, eval_seqs, n_layers, args.batch_size, args.seed, device, aa_t
    )
    print(f"recipients: {len(eval_seqs)} ({recip_correct.sum()} correct) | {time.time() - t0:.0f}s")

    true_idx = []
    for bi in range((len(eval_seqs) + args.batch_size - 1) // args.batch_size):
        batch = eval_seqs[bi * args.batch_size : (bi + 1) * args.batch_size]
        _, _, true_aa = make_masked_batch(batch, tokenizer, masks_per_seq=1, seed=args.seed + bi)
        true_idx.extend(AA_ALPHABET.index(aa) for row in true_aa for aa in row)
    true_idx = np.array(true_idx)

    lens = np.array([len(s) for s in eval_seqs], dtype=float)
    donor_lens = np.array([len(s) for s in donor_seqs], dtype=float)
    rng = np.random.default_rng(args.seed)

    def assign(pool_mask):
        pool_idx = np.where(pool_mask)[0]
        out = []
        for i in range(len(eval_seqs)):
            cand = pool_idx[np.abs(donor_lens[pool_idx] - lens[i]) <= 0.2 * lens[i]]
            if len(cand) == 0:
                cand = pool_idx[np.argsort(np.abs(donor_lens[pool_idx] - lens[i]))[:10]]
            out.append(int(rng.choice(cand)))
        return np.array(out)

    correct_donor = assign(donor_correct)
    incorrect_donor = assign(~donor_correct)
    base_p_true = base_probs[np.arange(len(true_idx)), true_idx]
    incorrect_mask = ~recip_correct
    print(f"incorrect recipients: {int(incorrect_mask.sum())}\n")

    rows = []
    print(
        f"{'condition':>22s} {'p(true) all':>11s} {'p(true) wrong':>13s} {'w->r':>5s} {'r->w':>5s}"
    )
    for prefix in prefixes:
        for name, donor_idx in (
            ("correct-donor", correct_donor),
            ("incorrect-donor", incorrect_donor),
            ("self-patch", None),
        ):
            if name == "self-patch":
                vecs = recip_h
            else:
                vecs = [h[donor_idx] for h in donor_h]
            probs, correct = run_prefix_patch(
                model, tokenizer, eval_seqs, prefix, vecs, args.batch_size, args.seed, device, aa_t
            )
            p_true = probs[np.arange(len(true_idx)), true_idx]
            w2r = int((incorrect_mask & correct).sum())
            r2w = int(((~incorrect_mask) & (~correct)).sum())
            cond = f"prefix0-{prefix}_{name}"
            rows.append(
                {
                    "condition": cond,
                    "p_true_all": round(float(p_true.mean()), 4),
                    "p_true_incorrect": round(float(p_true[incorrect_mask].mean()), 4),
                    "wrong_to_right": w2r,
                    "right_to_wrong": r2w,
                }
            )
            print(
                f"{cond:>22s} {p_true.mean():11.4f} {p_true[incorrect_mask].mean():13.4f} {w2r:5d} {r2w:5d}"
            )

    base_wrong = float(base_p_true[incorrect_mask].mean())
    print(f"\nbaseline p(true) on incorrect recipients: {base_wrong:.4f}")
    out = RESULTS_DIR / "full_layer_patching.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n_eval": len(eval_seqs),
                "n_incorrect": int(incorrect_mask.sum()),
                "baseline_p_true_incorrect": round(base_wrong, 4),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
