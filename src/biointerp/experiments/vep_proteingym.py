"""VEP on ProteinGym (Bio-Interp utility task).

Masked-marginal zero-shot scoring with ESM-2 650M on a DMS substitution assay.
Per-mutation correctness = sign agreement with DMS_score_bin; CETT features at the
mutated position are probed for misprediction. Reports per-assay Spearman as sanity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import RESULTS_DIR, univariate_auroc
from biointerp.esm import (
    AA_ALPHABET,
    aa_token_ids,
    extract_cett_batch,
    get_transformer_layers,
    load_model,
    precompute_col_norms,
)

MUT_SEG = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="OATML-Markslab/ProteinGym_v1")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--assay", default="SDA_BACSU_Tsuboyama_2023_1PV0")
    p.add_argument("--assays", default=None, help="comma-separated assay list (overrides --assay)")
    p.add_argument("--n-mutations", type=int, default=200)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--model-key", default=None, help="registry key (overrides --model)")
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    args = parse_args()
    if args.model_key:
        from biointerp.models import resolve_id

        args.model = resolve_id(args.model_key)
    path = hf_hub_download(
        args.dataset,
        f"DMS_substitutions/train-0000{args.shard}-of-00005.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    df = df[df["DMS_id"] == args.assay].reset_index(drop=True)

    def parse_muts(mutant):
        muts = []
        for seg in str(mutant).split(":"):
            m = MUT_SEG.match(seg.strip())
            if not m:
                return None
            muts.append((m.group(1), int(m.group(2)), m.group(3)))
        return muts

    rows = []
    for _, r in df.iterrows():
        muts = parse_muts(r["mutant"])
        if not muts or len(muts) != 1:
            continue
        rows.append(
            {
                "muts": muts,
                "score": float(r["DMS_score"]),
                "bin": float(r["DMS_score_bin"]),
                "target": str(r["target_seq"]),
            }
        )
    # Sample *after* filtering single substitutions: some assays are dominated
    # by multi-mutants (e.g. HIS7: 146 single out of 475k), so sampling first
    # leaves ~0 usable rows.
    if args.n_mutations and args.n_mutations < len(rows):
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(rows), size=args.n_mutations, replace=False)
        rows = [rows[i] for i in sorted(pick)]
    print(f"assay: {args.assay} | substitutions used: {len(rows)} (single)")
    if len(rows) < 10:
        print(f"[skip] {args.assay}: only {len(rows)} single substitutions")
        return

    model, tokenizer, device = load_model(args.model, args.device)
    layers = list(range(len(get_transformer_layers(model))))
    col_norms = precompute_col_norms(model, layers)
    aa_ids = aa_token_ids(tokenizer)
    aa_index = {aa: i for i, aa in enumerate(AA_ALPHABET)}

    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        groups[len(r["muts"])].append(r)

    cett_all, score_all, conf_all, correct_all = [], [], [], []
    t0 = time.time()
    done = 0
    for m_count, group_rows in sorted(groups.items()):
        for start in range(0, len(group_rows), args.batch_size):
            batch = group_rows[start : start + args.batch_size]
            ids_list, positions = [], []
            for r in batch:
                ids = list(tokenizer(r["target"], add_special_tokens=True)["input_ids"])
                pos_list = []
                for _wt, pos, _mut in r["muts"]:
                    ids[pos] = tokenizer.mask_token_id
                    pos_list.append(pos)
                ids_list.append(ids)
                positions.append(pos_list)
            tokens = {
                "input_ids": torch.tensor(ids_list),
                "attention_mask": torch.ones(len(ids_list), len(ids_list[0]), dtype=torch.long),
            }
            tokens = {k: v.to(device) for k, v in tokens.items()}
            pos_t = torch.tensor(positions, dtype=torch.long, device=device)
            res = extract_cett_batch(model, tokens, pos_t, layers, col_norms, aa_ids)
            logits_aa = res["logits_aa"].numpy()
            probs = res["probs"].numpy()
            cett_rows = res["cett"].numpy().astype(np.float16)
            for i, r in enumerate(batch):
                s_score = 0.0
                for k, (wt, _pos, mut) in enumerate(r["muts"]):
                    s_score += float(
                        logits_aa[i * m_count + k, aa_index[mut]]
                        - logits_aa[i * m_count + k, aa_index[wt]]
                    )
                score_all.append(s_score)
                conf_all.append(float(probs[i * m_count : (i + 1) * m_count].max()))
                correct_all.append((s_score < 0) == (r["bin"] < 0.5))
                cett_all.append(cett_rows[i * m_count : (i + 1) * m_count].mean(axis=0))
            done += len(batch)
            if done % (args.batch_size * 5) < args.batch_size:
                print(f"  scored {done}/{len(rows)} | {time.time() - t0:.0f}s")

    X = np.stack(cett_all).astype(np.float32)
    scores = np.array(score_all)
    conf = np.array(conf_all)
    y = (~np.array(correct_all)).astype(int)
    if len(np.unique(y)) < 2:
        print(f"[skip] {args.assay}: single-class labels (n={len(y)})")
        return
    dms = np.array([r["score"] for r in rows])
    rho, _ = spearmanr(scores, dms)
    print(f"\nscored {len(rows)} mutations in {time.time() - t0:.0f}s")
    print(f"zero-shot Spearman(score, DMS_score): {rho:.4f}")
    print(f"sign-agreement accuracy: {1 - y.mean():.4f} | incorrect rate {y.mean():.4f}")

    n = len(y)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * args.test_frac)))
    te = np.zeros(n, dtype=bool)
    te[perm[:n_test]] = True
    tr = ~te
    au = univariate_auroc(X[tr], y[tr])
    idx = np.argsort(au)[::-1][: args.top_k]
    scaler = StandardScaler()
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
    clf.fit(scaler.fit_transform(X[tr][:, idx]), y[tr])
    auroc_probe = float(
        roc_auc_score(y[te], clf.predict_proba(scaler.transform(X[te][:, idx]))[:, 1])
    )
    auroc_conf = float(roc_auc_score(y[te], -conf[te]))
    print(f"AUROC probe (top-{args.top_k} CETT): {auroc_probe:.4f}")
    print(f"AUROC confidence:                   {auroc_conf:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"vep_proteingym_{args.assay}.json"
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "assay": args.assay,
                "n_mutations": n,
                "spearman": round(float(rho), 4),
                "sign_accuracy": round(float(1 - y.mean()), 4),
                "auroc_probe": round(auroc_probe, 4),
                "auroc_confidence": round(auroc_conf, 4),
            },
            indent=2,
        )
        + "\n"
    )
    preds = RESULTS_DIR / f"vep_proteingym_{args.assay}.predictions.jsonl"
    with preds.open("w") as fh:
        for i, r in enumerate(rows):
            mut_str = ":".join(f"{wt}{pos}{mut}" for wt, pos, mut in r["muts"])
            fh.write(
                json.dumps(
                    {
                        "mutant": mut_str,
                        "score": round(scores[i], 4),
                        "dms_score": round(dms[i], 4),
                        "correct": bool(correct_all[i]),
                        "confidence": round(conf[i], 4),
                    }
                )
                + "\n"
            )
    print(f"\nsaved: {out}")
    print(f"saved: {preds}")


if __name__ == "__main__":
    main()
