"""Layer-resolved disorder analysis (Bio-Interp PLM-native, exp #8).

Question: does structural non-resolution in disordered regions precede the
late-layer error signal? I.e. at which layer do ordered vs disordered positions
diverge (L_div), and does that layer come BEFORE the error signal fires (L_err)?

Inputs: results/structure_mapping.json per-residue {plddt, p_true}.
Method: one forward/seq with output_hidden_states; per layer: residual norms,
error-probe AUROC (sequence split), disorder-separability AUROC + group-mean
cosine. Bootstrap CIs over sequences. Controls: length-matched, terminus-cut,
per-AA stratification, permutation null.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from biointerp.core import DEFAULT_FASTA, RESULTS_DIR, load_sequences, split_by_protein


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(RESULTS_DIR / "structure_mapping.json"))
    p.add_argument("--n-seqs", type=int, default=-1, help="-1 = all records")
    p.add_argument("--start", type=int, default=0, help="first record index")
    p.add_argument("--model-key", default="esm2_650m")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--out", default=str(RESULTS_DIR / "layer_disorder_resolution.json"))
    return p.parse_args()


def stratified_seq_split(seq_id, pos_mask, test_frac=0.2, seed=0):
    """Sequence split guaranteeing disordered-containing seqs on both sides."""
    rng = np.random.default_rng(seed)
    seq_id = np.asarray(seq_id)
    uniq = np.unique(seq_id)
    has = np.array([pos_mask[seq_id == s].any() for s in uniq])
    n_total = max(2, int(round(len(uniq) * test_frac)))
    n_has = min(max(1, int(round(int(has.sum()) * test_frac))), n_total - 1)
    g1 = rng.permutation(uniq[has])[:n_has]
    g0 = rng.permutation(uniq[~has])[: max(0, n_total - n_has)]
    te = set(g1.tolist() + g0.tolist())
    return np.array([s not in te for s in seq_id]), np.array([s in te for s in seq_id])


def grouped_probe_auc(X, y, groups, seed=0, stratify=None):
    """Logistic probe with sequence split; returns test AUC + test scores/idx."""
    if stratify is None:
        tr, te = split_by_protein(np.asarray(groups), test_frac=0.2, seed=seed)
    else:
        tr, te = stratified_seq_split(groups, np.asarray(stratify), 0.2, seed)
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return float("nan"), np.zeros(int(te.sum())), np.flatnonzero(te), y[te]
    scaler = StandardScaler()
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(scaler.fit_transform(X[tr]), y[tr])
    scores = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    auc = float(roc_auc_score(y[te], scores))
    return auc, scores, np.flatnonzero(te), y[te]


def bootstrap_first_crossing(test_scores, test_y, test_groups, thresh, n_boot, seed):
    """CI for first layer crossing thresh, resampling test sequences."""
    rng = np.random.default_rng(seed)
    seqs = np.array(sorted(set(test_groups.tolist())))
    firsts = []
    for _ in range(n_boot):
        pick = set(rng.choice(seqs, size=len(seqs), replace=True).tolist())
        m = np.array([g in pick for g in test_groups])
        if len(np.unique(test_y[m])) < 2:
            continue
        aucs = [roc_auc_score(test_y[m], test_scores[l][m]) for l in range(len(test_scores))]
        hit = next((l for l, a in enumerate(aucs) if a > thresh), None)
        if hit is not None:
            firsts.append(hit)
    if not firsts:
        return None, None
    return float(np.median(firsts)), [
        float(np.quantile(firsts, 0.025)),
        float(np.quantile(firsts, 0.975)),
    ]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    t0 = time.time()
    data = json.loads(open(args.input).read())
    records = (
        data["records"]
        if args.n_seqs < 0
        else data["records"][args.start : args.start + args.n_seqs]
    )
    seqs_all = [s for s in load_sequences(DEFAULT_FASTA) if len(s) <= 250]
    idx = [r["seq"] for r in records]

    from biointerp.models import load_model as load_by_key

    model, tokenizer, device = load_by_key(args.model_key, args.device)
    model.eval()

    H, meta = [], []  # per-layer hidden states, per-position meta
    for k, (i, rec) in enumerate(zip(idx, records)):
        seq = seqs_all[i]
        ids = tokenizer(seq, add_special_tokens=True)["input_ids"]
        L = len(seq)
        inp = torch.tensor([ids], device=device)
        out = model(input_ids=inp, attention_mask=torch.ones_like(inp), output_hidden_states=True)
        hs = [h[0, 1 : L + 1, :].float().cpu().numpy() for h in out.hidden_states]
        H.append(np.stack(hs))  # (34, L, d)
        for j, pr in enumerate(rec["per_residue"]):
            meta.append((k, j / max(L - 1, 1), seq[j], pr["plddt"], 1.0 - pr["p_true"]))
        print(f"seq {k + 1}/{len(records)}: len {L} | {time.time() - t0:.0f}s", flush=True)
    H = np.concatenate(H, axis=1)  # (34, N, d)
    n_layers = H.shape[0]
    meta = np.array(meta, dtype=object)
    seq_id = np.array([m[0] for m in meta], dtype=int)
    relpos = np.array([m[1] for m in meta], dtype=float)
    aas = np.array([m[2] for m in meta])
    plddt = np.array([m[3] for m in meta], dtype=float)
    err = np.array([m[4] for m in meta], dtype=float)
    ordered = plddt >= 0.7
    disordered = plddt < 0.5
    print(f"positions: {len(err)} | ordered {ordered.sum()} | disordered {disordered.sum()}")

    err_bin = (err > np.median(err)).astype(int)
    layers = []
    err_scores, sep_scores = [], []
    err_te_y, sep_te_y, _te_groups = None, None, None
    for l in range(n_layers):
        X = H[l]
        norms = np.linalg.norm(X, axis=1)
        auc_err, sc_err, te_idx, te_y = grouped_probe_auc(
            X, err_bin, seq_id, args.seed, stratify=(err_bin == 1)
        )
        m = ordered | disordered
        auc_sep, sc_sep, te_idx_s, te_y_s = grouped_probe_auc(
            X[m], ordered[m].astype(int), seq_id[m], args.seed, stratify=disordered[m]
        )
        mu_o, mu_d = X[m][ordered[m]].mean(0), X[m][disordered[m]].mean(0)
        cos = float(mu_o @ mu_d / (np.linalg.norm(mu_o) * np.linalg.norm(mu_d)))
        layers.append(
            {
                "layer": l,
                "mean_norm_ordered": round(float(norms[ordered].mean()), 4),
                "mean_norm_disordered": round(float(norms[disordered].mean()), 4),
                "err_probe_auc": round(auc_err, 4),
                "disorder_sep_auc": round(auc_sep, 4),
                "group_mean_cos": round(cos, 4),
            }
        )
        err_scores.append(sc_err)
        if l == 0:
            _te_groups, err_te_y = seq_id[te_idx], te_y
        sep_scores.append(sc_sep)
        if l == 0:
            sep_te_y = te_y_s
        print(
            f"layer {l}: norm o/d {norms[ordered].mean():.2f}/{norms[disordered].mean():.2f} "
            f"| errAUC {auc_err:.4f} | sepAUC {auc_sep:.4f} | cos {cos:.4f}",
            flush=True,
        )

    # test-seq bookkeeping for bootstrap (err probe and sep probe share seq split seed;
    # rebuild group ids for sep probe test positions)
    _, _, te_idx_e, _ = grouped_probe_auc(H[0], err_bin, seq_id, args.seed, stratify=(err_bin == 1))
    m = ordered | disordered
    _, _, te_idx_s, _ = grouped_probe_auc(
        H[0][m], ordered[m].astype(int), seq_id[m], args.seed, stratify=disordered[m]
    )
    te_groups_e = seq_id[te_idx_e]
    te_groups_s = seq_id[m][te_idx_s]
    # align stored scores to these test sets (same split -> same order)
    L_err = next((l for l in range(n_layers) if layers[l]["err_probe_auc"] > 0.65), None)
    L_div = next((l for l in range(n_layers) if layers[l]["disorder_sep_auc"] > 0.7), None)
    L_err_med, L_err_ci = bootstrap_first_crossing(
        err_scores, err_te_y, te_groups_e, 0.65, args.n_bootstrap, args.seed
    )
    L_div_med, L_div_ci = bootstrap_first_crossing(
        sep_scores, sep_te_y, te_groups_s, 0.7, args.n_bootstrap, args.seed + 1
    )

    # final-layer norm correlations
    n_fin = H[-1]
    nf = np.linalg.norm(n_fin, axis=1)
    rho_n_p, _ = spearmanr(nf, plddt)
    rho_n_e, _ = spearmanr(nf, err)

    # controls
    rng = np.random.default_rng(args.seed)
    # length-matched: subsample majority group per length decile
    lens = np.array([len(seqs_all[i]) for i in seq_id])
    bins = np.quantile(lens, np.linspace(0, 1, 6))
    keep = np.zeros(len(err), dtype=bool)
    for b in range(5):
        mb = (lens >= bins[b]) & (lens <= bins[b + 1]) & (ordered | disordered)
        io = np.flatnonzero(mb & ordered)
        idd = np.flatnonzero(mb & disordered)
        n = min(len(io), len(idd))
        if n:
            keep[rng.choice(io, n, replace=False)] = True
            keep[rng.choice(idd, n, replace=False)] = True
    auc_lm = grouped_probe_auc(
        H[-1][keep], ordered[keep].astype(int), seq_id[keep], args.seed, stratify=disordered[keep]
    )[0]
    # terminus-cut: middle 80% only
    mid = (relpos >= 0.1) & (relpos <= 0.9) & (ordered | disordered)
    auc_mid = grouped_probe_auc(
        H[-1][mid], ordered[mid].astype(int), seq_id[mid], args.seed, stratify=disordered[mid]
    )[0]
    # per-AA stratification: ordered-vs-disordered error gap within each AA
    aa_gap = {}
    for aa in sorted(set(aas.tolist())):
        ma = (aas == aa) & (ordered | disordered)
        if ma.sum() >= 20 and ordered[ma].sum() >= 5 and disordered[ma].sum() >= 5:
            aa_gap[aa] = round(float(err[ma & disordered].mean() - err[ma & ordered].mean()), 4)
    # permutation null on separability (final layer)
    yo = ordered[m].astype(int)
    auc_null = grouped_probe_auc(
        H[-1][m], rng.permutation(yo), seq_id[m], args.seed, stratify=disordered[m]
    )[0]

    print(f"\nL_err(first>0.65): {L_err} | boot median {L_err_med} CI {L_err_ci}")
    print(f"L_div(first>0.70): {L_div} | boot median {L_div_med} CI {L_div_ci}")
    print(f"final-layer norm vs pLDDT rho {rho_n_p:.4f} | vs err rho {rho_n_e:.4f}")
    print(
        f"controls sepAUC: length-matched {auc_lm:.4f} | mid-only {auc_mid:.4f} "
        f"| perm-null {auc_null:.4f}"
    )
    print(f"per-AA disordered-minus-ordered err gap (n={len(aa_gap)} AAs): " + str(aa_gap))

    verdict = (
        "supported"
        if (L_div is not None and L_err is not None and L_div < L_err)
        else "not-supported"
        if (L_div is not None and L_err is not None)
        else "undetermined"
    )
    open(args.out, "w").write(
        json.dumps(
            {
                "input": args.input,
                "model": args.model_key,
                "n_seqs": len(records),
                "n_positions": len(err),
                "n_ordered": int(ordered.sum()),
                "n_disordered": int(disordered.sum()),
                "L_err": L_err,
                "L_err_boot": [L_err_med, L_err_ci],
                "L_div": L_div,
                "L_div_boot": [L_div_med, L_div_ci],
                "final_norm_rho_plddt": round(float(rho_n_p), 4),
                "final_norm_rho_err": round(float(rho_n_e), 4),
                "controls": {
                    "length_matched_sep_auc": round(auc_lm, 4),
                    "mid_only_sep_auc": round(auc_mid, 4),
                    "perm_null_sep_auc": round(auc_null, 4),
                    "per_aa_err_gap": aa_gap,
                },
                "verdict": verdict,
                "layers": layers,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nverdict: {verdict} | saved: {args.out} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
