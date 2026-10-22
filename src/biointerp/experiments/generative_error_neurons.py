"""Generative error neurons on ProtGPT2 (Bio-Interp generative phase).

Generate protein sequences with per-step CETT; label each emitted residue by its
ESM-2 masked naturalness (p(true residue) under leave-one-out masking); probe CETT for
"bad residue" prediction; compare against ProtGPT2's own confidence/entropy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR, univariate_auroc
from biointerp.esm import AA_ALPHABET, aa_token_ids
from biointerp.esm import load_model as load_esm
from biointerp.gen import generate_with_cett, load_generative
from biointerp.gen import get_transformer_layers as gen_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-model", default="nferruz/ProtGPT2")
    p.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--n-seqs", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.inference_mode()
def esm_naturalness(esm_model, tokenizer, seq, device, aa_t, aa_index, chunk=64):
    """Per-residue p(true) under leave-one-out masking, chunked to bound memory."""
    ids = tokenizer(seq, add_special_tokens=True)["input_ids"]
    L = len(seq)
    true_idx = np.array([aa_index[c] for c in seq])
    out_probs = np.empty(L, dtype=np.float64)
    for start in range(0, L, chunk):
        end = min(start + chunk, L)
        batch = []
        for j in range(start + 1, end + 1):
            row = list(ids)
            row[j] = tokenizer.mask_token_id
            batch.append(row)
        input_ids = torch.tensor(batch, device=device)
        out = esm_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        rows = torch.arange(len(batch), device=device)
        logits = out.logits[rows, rows + start + 1][:, aa_t]
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        out_probs[start:end] = probs[np.arange(len(batch)), true_idx[start:end]]
    return out_probs


def main() -> None:
    args = parse_args()
    gen_model, gen_tok, device = load_generative(args.gen_model, args.device)
    n_layers = len(gen_layers(gen_model))
    esm_model, esm_tok, esm_dev = load_esm(args.esm_model, args.device)
    aa_t = torch.tensor(aa_token_ids(esm_tok), device=esm_dev)
    aa_index = {c: i for i, c in enumerate(AA_ALPHABET)}

    feats: list[np.ndarray] = []
    labels: list[float] = []
    confs: list[float] = []
    ents: list[float] = []
    seq_ids: list[int] = []
    records: list[dict] = []

    t0 = time.time()
    for si in range(args.n_seqs):
        ids, cetts, texts, confs_s, ents_s = generate_with_cett(
            gen_model,
            gen_tok,
            n_steps=args.n_steps,
            seed=args.seed + si,
            temperature=args.temperature,
            greedy=False,
            device=device,
        )
        if not ids:
            continue
        seq = "".join(texts)
        clean = "".join(c for c in seq if c in aa_index)
        if len(clean) < 10:
            continue
        nat = esm_naturalness(esm_model, esm_tok, clean, esm_dev, aa_t, aa_index)
        pos = 0
        for step_i, text in enumerate(texts):
            res = [c for c in text if c in aa_index]
            if not res:
                continue
            vals = nat[pos : pos + len(res)]
            pos += len(res)
            if len(vals) == 0:
                continue
            feats.append(cetts[step_i].numpy().astype(np.float16))
            labels.append(float(vals.mean()))
            confs.append(confs_s[step_i])
            ents.append(ents_s[step_i])
            seq_ids.append(si)
            if len(records) < 200:
                records.append(
                    {
                        "seq": si,
                        "step": step_i,
                        "token": text,
                        "residues": "".join(res),
                        "naturalness": round(float(vals.mean()), 4),
                        "gen_confidence": round(confs_s[step_i], 4),
                        "gen_entropy": round(ents_s[step_i], 4),
                    }
                )
        print(
            f"seq {si + 1}/{args.n_seqs}: len {len(clean)} | steps used {len(feats)} | {time.time() - t0:.0f}s"
        )

    X = np.stack(feats)
    y_nat = np.array(labels)
    conf = np.array(confs)
    ent = np.array(ents)
    seq_arr = np.array(seq_ids)
    print(f"\nsteps: {len(X)} | features: {X.shape[1]} | mean naturalness {y_nat.mean():.4f}")

    unique = np.unique(seq_arr)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * args.test_frac)))
    test_seqs = set(unique[:n_test].tolist())
    te = np.array([s in test_seqs for s in seq_arr])
    tr = ~te

    threshold = float(np.median(y_nat[tr]))
    y = (y_nat < threshold).astype(int)
    print(
        f"bad threshold (train median naturalness): {threshold:.4f} | bad rate train {y[tr].mean():.4f} test {y[te].mean():.4f}"
    )

    Xf = X.astype(np.float32)
    au = univariate_auroc(Xf[tr], y[tr])
    idx = np.argsort(au)[::-1][: args.top_k]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
    clf.fit(scaler.fit_transform(Xf[tr][:, idx]), y[tr])
    scores = clf.predict_proba(scaler.transform(Xf[te][:, idx]))[:, 1]
    auroc_probe = (
        float(roc_auc_score(y[te], scores)) if len(np.unique(y[te])) == 2 else float("nan")
    )
    auroc_conf = (
        float(roc_auc_score(y[te], -conf[te])) if len(np.unique(y[te])) == 2 else float("nan")
    )
    auroc_ent = float(roc_auc_score(y[te], ent[te])) if len(np.unique(y[te])) == 2 else float("nan")

    print(f"\nAUROC probe (CETT top-{args.top_k}): {auroc_probe:.4f}")
    print(f"AUROC ProtGPT2 confidence:          {auroc_conf:.4f}")
    print(f"AUROC ProtGPT2 entropy:             {auroc_ent:.4f}")

    slug = args.gen_model.replace("/", "_")
    out = RESULTS_DIR / f"generative_error_neurons_{slug}.json"
    out.write_text(
        json.dumps(
            {
                "gen_model": args.gen_model,
                "esm_model": args.esm_model,
                "n_seqs": args.n_seqs,
                "n_steps": args.n_steps,
                "temperature": args.temperature,
                "steps": int(len(X)),
                "features": int(X.shape[1]),
                "n_layers": n_layers,
                "mean_naturalness": round(float(y_nat.mean()), 4),
                "bad_threshold": round(threshold, 4),
                "auroc_probe": round(auroc_probe, 4),
                "auroc_gen_confidence": round(auroc_conf, 4),
                "auroc_gen_entropy": round(auroc_ent, 4),
            },
            indent=2,
        )
        + "\n"
    )
    records_path = RESULTS_DIR / f"generative_error_neurons_{slug}.samples.jsonl"
    with records_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nsaved: {out}")
    print(f"saved: {records_path}")


if __name__ == "__main__":
    main()
