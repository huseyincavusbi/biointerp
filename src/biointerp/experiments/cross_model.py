"""Cross-model replication on ESMC-600M (Bio-Interp).

Runs the core pipeline (masked-residue task -> CETT -> probe) on the 2026 frontier open
model via the official esm SDK. Compares against the ESM-2 650M results.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import DEFAULT_FASTA, RESULTS_DIR, load_sequences, univariate_auroc
from biointerp.esmc import AA_ALPHABET, extract_cett_sequence, load_esmc, precompute_col_norms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="biohub/ESMC-600M")
    p.add_argument("--model-key", default=None, help="registry key (overrides --model)")
    p.add_argument("--dtype", default=None, help="bf16 for large models")
    p.add_argument("--device-map", default=None, help="e.g. auto for multi-GPU")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_key:
        from biointerp.models import resolve_id

        args.model = resolve_id(args.model_key)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sequences = load_sequences(DEFAULT_FASTA)[: args.n]
    model, tokenizer, device = load_esmc(args.model, args.device, args.dtype, args.device_map)
    col_norms = precompute_col_norms(model)
    rng = random.Random(args.seed)

    feats, correct, conf = [], [], []
    t0 = time.time()
    for i, seq in enumerate(sequences):
        p = rng.randrange(1, len(seq) - 1)
        cett, probs = extract_cett_sequence(model, tokenizer, seq, [p], col_norms)
        probs_np = probs.numpy()[0]
        pred = AA_ALPHABET[int(probs_np.argmax())]
        feats.append(cett[0].numpy().astype(np.float16))
        correct.append(pred == seq[p])
        conf.append(float(probs_np.max()))
        if (i + 1) % 20 == 0 or i + 1 == len(sequences):
            print(
                f"seq {i + 1}/{len(sequences)} | acc so far {np.mean(correct):.4f} | {time.time() - t0:.0f}s"
            )

    X = np.stack(feats).astype(np.float32)
    y = (~np.array(correct)).astype(int)
    conf = np.array(conf)
    n = len(y)
    rng_np = np.random.default_rng(args.seed)
    perm = rng_np.permutation(n)
    n_test = max(1, int(round(n * args.test_frac)))
    te = np.zeros(n, dtype=bool)
    te[perm[:n_test]] = True
    tr = ~te
    print(f"\nsamples: {n} | accuracy {1 - y.mean():.4f} | features {X.shape[1]} | device {device}")

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

    out = RESULTS_DIR / "cross_model.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": n,
                "accuracy": round(float(1 - y.mean()), 4),
                "features": int(X.shape[1]),
                "top_k": args.top_k,
                "device": device,
                "auroc_probe": round(auroc_probe, 4),
                "auroc_confidence": round(auroc_conf, 4),
                "reference_esm2_650m_probe": 0.7542,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
