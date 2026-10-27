"""Shared utilities for Bio-Interp experiments."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
BIOINTERP_DIR = REPO_ROOT
PILOT_DIR = REPO_ROOT / "pilot"
PILOT_RESULTS = PILOT_DIR / "results"
RESULTS_DIR = Path(os.environ.get("BIOINTERP_RESULTS_DIR", REPO_ROOT / "results"))
DATA_DIR = Path(os.environ.get("BIOINTERP_DATA_DIR", REPO_ROOT / "data"))

DEFAULT_NPZ = PILOT_RESULTS / "esm2_t33_650M_UR50D_n2000_m1_seed0.npz"
DEFAULT_FASTA = DATA_DIR / "swissprot_n2000_len50-500_seed0.fasta"


def split_by_protein(protein, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    unique = np.unique(protein)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_frac)))
    test_prot = set(unique[:n_test].tolist())
    te = np.array([p in test_prot for p in protein])
    return ~te, te


def univariate_auroc(x, y, chunk: int = 20000):
    out = np.empty(x.shape[1], dtype=np.float64)
    npos = int(y.sum())
    nneg = len(y) - npos
    for s in range(0, x.shape[1], chunk):
        e = min(s + chunk, x.shape[1])
        # Average ranks for ties, so a tied positive/negative pair contributes 0.5.
        r = rankdata(x[:, s:e], axis=0)
        out[s:e] = (r[y == 1].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
    return out


def load_sequences(path) -> list[str]:
    sequences: list[str] = []
    parts: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
                parts = []
        else:
            parts.append(line)
    if parts:
        sequences.append("".join(parts))
    return sequences


def top_probe_scores(x_tr, y_tr, x_te, top_k: int, seed: int = 0):
    """Univariate-AUROC top-k selection + logistic probe; returns test scores + idx."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    au = univariate_auroc(x_tr, y_tr)
    idx = np.argsort(au)[::-1][:top_k]
    scaler = StandardScaler()
    xt = scaler.fit_transform(x_tr[:, idx])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(xt, y_tr)
    scores = clf.predict_proba(scaler.transform(x_te[:, idx]))[:, 1]
    return scores, idx
