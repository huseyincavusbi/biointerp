"""Evolutionary mapping (Bio-Interp PLM-native, exp #9 in NEXT, Exp 1 in RQ).

Question: do errors arise where evolution is unconstrained, and does disorder
add information beyond conservation?

Two independent axes (no MSA building, no GPU):
  1. metapredict disorder (local, CPU) -- independent validator of the pLDDT
     proxy + a second disorder axis vs error.
  2. Pfam domains via EBI hmmscan (needs HMMER_EMAIL env) -- in-domain vs
     out-of-domain error; logistic err ~ disorder + in_domain (residual test).
     Per-seq results cached in results/hmmscan_cache/; skipped if no email.

Stats are CPU-only. Designed for sequence-split probes (no position leakage).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from biointerp.core import DEFAULT_FASTA, RESULTS_DIR, load_sequences, split_by_protein

EBI_RUN = "https://www.ebi.ac.uk/Tools/services/rest/hmmer3_hmmscan/run"
EBI_STATUS = "https://www.ebi.ac.uk/Tools/services/rest/hmmer3_hmmscan/status/"
EBI_RESULT = "https://www.ebi.ac.uk/Tools/services/rest/hmmer3_hmmscan/result/"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(RESULTS_DIR / "structure_mapping.json"))
    p.add_argument("--n-seqs", type=int, default=-1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--email-env", default="HMMER_EMAIL")
    p.add_argument("--no-hmmscan", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(RESULTS_DIR / "evolutionary_mapping.json"))
    return p.parse_args()


def meta_disorder(seq: str) -> np.ndarray:
    import metapredict

    out = metapredict.predict(seq)
    for attr in ("disorder", "disorder_scores", "scores", "values"):
        if hasattr(out, attr):
            return np.asarray(getattr(out, attr), dtype=float)
    return np.asarray(out, dtype=float)


def hmmscan_submit(seq: str, email: str, title: str) -> str:
    data = urllib.parse.urlencode(
        {"email": email, "title": title, "database": "pfam", "sequence": seq}
    ).encode()
    req = urllib.request.Request(EBI_RUN, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode().strip()


def hmmscan_status(job: str) -> str:
    with urllib.request.urlopen(EBI_STATUS + job, timeout=30) as r:
        return r.read().decode().strip()


def hmmscan_fetch(job: str) -> str:
    with urllib.request.urlopen(EBI_RESULT + job + "/out", timeout=120) as r:
        return r.read().decode()


def _is_transient(e: Exception) -> bool:
    import urllib.error

    if isinstance(e, urllib.error.HTTPError):
        return e.code in (429, 500, 502, 503, 504)
    return isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError))


def call_with_retry(fn, what: str, tries: int = 6):
    import urllib.error

    last = None
    for a in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if not _is_transient(e) or a == tries - 1:
                raise
            wait = min(120, 10 * 2**a)
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                wait = max(wait, 60)
            print(
                f"  {what}: transient {type(e).__name__} (try {a + 1}/{tries}), "
                f"waiting {wait}s (rate-limit respectful)...",
                flush=True,
            )
            time.sleep(wait)
    raise last  # unreachable


def hmmscan_poll(job: str, timeout_s: int = 900) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = call_with_retry(lambda: hmmscan_status(job), f"status {job}")
        if st == "FINISHED":
            return call_with_retry(lambda: hmmscan_fetch(job), f"fetch {job}")
        if st in ("ERROR", "FAILURE", "NOT_FOUND"):
            raise RuntimeError(f"hmmscan job {job}: {st}")
        time.sleep(10)
    raise TimeoutError(f"hmmscan job {job} not finished in {timeout_s}s")


def hmmscan_batch(items: list, email: str) -> dict:
    """Submit ALL jobs first, then poll together. Rate-limit respectful:
    server queues jobs; polling sweeps sleep 15s; transient errors back off."""
    jobs = {}
    for i, seq in items:
        jobs[i] = call_with_retry(
            lambda: hmmscan_submit(seq, email, f"biointerp_seq{i}"), f"submit seq{i}"
        )
    print(f"submitted {len(jobs)} hmmscan jobs; polling together...", flush=True)
    out, pending = {}, dict(jobs)
    while pending:
        for i, job in list(pending.items()):
            st = call_with_retry(lambda: hmmscan_status(job), f"status {job}")
            if st == "FINISHED":
                out[i] = call_with_retry(lambda: hmmscan_fetch(job), f"fetch {job}")
                del pending[i]
                print(f"  job seq{i} finished ({len(out)}/{len(jobs)})", flush=True)
            elif st in ("ERROR", "FAILURE", "NOT_FOUND"):
                raise RuntimeError(f"hmmscan job {job}: {st}")
        if pending:
            time.sleep(15)
    return out


def parse_domains(out_text: str, L: int) -> list:
    """Union of trusted domain envelopes as (start, end) 0-based inclusive.

    Parses hmmscan text domain rows:
      ` 1 !  238.2  0.2  8.3e-75  5.3e-71  1  177 []  19  189 .. ...`
    i.e. parts = [num, mark, score, bias, ceval, ieval, hmmfrom, hmmto,
    hmmbounds, alifrom, alito, ...]. Trusted '!' hits with i-Evalue < 0.01.
    """
    import re

    row = re.compile(
        r"^\s*(\d+)\s+([!?])\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+"
        r"(\d+)\s+(\d+)\s+\S+\s+(\d+)\s+(\d+)"
    )
    ivs = []
    for line in out_text.splitlines():
        m = row.match(line)
        if not m:
            continue
        mark, ieval = m.group(2), float(m.group(6))
        a0, a1 = int(m.group(9)) - 1, int(m.group(10)) - 1
        if mark == "!" and ieval < 0.01 and 0 <= a0 < L and 0 <= a1 < L:
            ivs.append((min(a0, a1), max(a0, a1)))
    return ivs


def in_domain_mask(L: int, ivs: list) -> np.ndarray:
    m = np.zeros(L, dtype=bool)
    for a, b in ivs:
        m[a : b + 1] = True
    return m


def probe_auc(X, y, groups, seed, stratify=None):
    from biointerp.experiments.layer_disorder_resolution import stratified_seq_split

    groups = np.asarray(groups)
    if stratify is None:
        tr, te = split_by_protein(groups, test_frac=0.2, seed=seed)
    else:
        tr, te = stratified_seq_split(groups, np.asarray(stratify), 0.2, seed)
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return float("nan")
    sc = StandardScaler()
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(sc.fit_transform(X[tr]), y[tr])
    return float(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))


def main() -> None:
    args = parse_args()
    t0 = time.time()
    data = json.loads(open(args.input).read())
    recs = (
        data["records"]
        if args.n_seqs < 0
        else data["records"][args.start : args.start + args.n_seqs]
    )
    seqs_all = [s for s in load_sequences(DEFAULT_FASTA) if len(s) <= 250]

    email = os.environ.get(args.email_env, "")
    do_hmm = not args.no_hmmscan and bool(email)
    cache = RESULTS_DIR / "hmmscan_cache"
    cache.mkdir(exist_ok=True)
    if not args.no_hmmscan and not email:
        print(
            f"WARN: ${args.email_env} unset -> Pfam/hmmscan step skipped (disorder part still runs)"
        )

    md_all, err_all, pl_all, dom_all, grp_all, len_all = [], [], [], [], [], []
    n_dom_hits = 0
    if do_hmm:
        todo = [rec["seq"] for rec in recs if not (cache / f"seq{rec['seq']}.json").exists()]
        if todo:
            batch = hmmscan_batch([(i, seqs_all[i]) for i in todo], email)
            for i, txt in batch.items():
                L = len(seqs_all[i])
                (cache / f"seq{i}.raw.txt").write_text(txt)
                (cache / f"seq{i}.json").write_text(
                    json.dumps({"intervals": parse_domains(txt, L)})
                )
    for k, rec in enumerate(recs):
        seq = seqs_all[rec["seq"]]
        L = len(seq)
        md = meta_disorder(seq)
        assert len(md) == L, f"metapredict len {len(md)} != seq len {L}"
        pr = rec["per_residue"]
        err = np.array([1.0 - p["p_true"] for p in pr])
        pl = np.array([p["plddt"] for p in pr])
        dom = np.zeros(L, dtype=bool)
        if do_hmm:
            ck = cache / f"seq{rec['seq']}.json"
            ivs = json.loads(ck.read_text())["intervals"]  # batched above; cache always hit
            dom = in_domain_mask(L, ivs)
            n_dom_hits += len(ivs)
        md_all.append(md)
        err_all.append(err)
        pl_all.append(pl)
        dom_all.append(dom)
        grp_all.append(np.full(L, k))
        len_all.append(np.full(L, L))
        print(
            f"seq {k + 1}/{len(recs)} (rec {rec['seq']}, len {L}): "
            f"meta-dis frac {float((md > 0.5).mean()):.3f} | domains {int(dom.sum())} | "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )

    md_all = np.concatenate(md_all)
    err_all = np.concatenate(err_all)
    pl_all = np.concatenate(pl_all)
    dom_all = np.concatenate(dom_all)
    grp_all = np.concatenate(grp_all)
    len_all = np.concatenate(len_all)
    N = len(err_all)
    err_bin = (err_all > np.median(err_all)).astype(int)
    dis_meta = (md_all > 0.5).astype(int)

    rho_meta_pl, _ = spearmanr(md_all, pl_all)
    rho_meta_err, _ = spearmanr(md_all, err_all)
    e_dis = err_all[dis_meta == 1].mean()
    e_ord = err_all[dis_meta == 0].mean()
    auc_meta = probe_auc(
        np.column_stack([md_all, len_all]), err_bin, grp_all, args.seed, stratify=(dis_meta == 1)
    )
    auc_meta_only = probe_auc(
        md_all[:, None], err_bin, grp_all, args.seed, stratify=(dis_meta == 1)
    )
    auc_len_only = probe_auc(
        len_all[:, None], err_bin, grp_all, args.seed, stratify=(dis_meta == 1)
    )
    print(f"\npositions: {N}")
    print(f"metapredict vs pLDDT rho {rho_meta_pl:.4f} (proxy validation)")
    print(f"metapredict-disorder vs err rho {rho_meta_err:.4f}")
    print(f"mean err | meta-disordered {e_dis:.4f} vs meta-ordered {e_ord:.4f}")
    print(
        f"err probe from [meta-disorder, len] AUC {auc_meta:.4f} "
        f"(meta-only {auc_meta_only:.4f}, len-only {auc_len_only:.4f})"
    )

    out = {
        "input": args.input,
        "n_seqs": len(recs),
        "n_positions": N,
        "hmmscan_run": do_hmm,
        "meta_vs_plddt_rho": round(float(rho_meta_pl), 4),
        "meta_vs_err_rho": round(float(rho_meta_err), 4),
        "mean_err_meta_disordered": round(float(e_dis), 4),
        "mean_err_meta_ordered": round(float(e_ord), 4),
        "err_probe_auc_meta_len": round(float(auc_meta), 4),
        "err_probe_auc_meta_only": round(float(auc_meta_only), 4),
        "err_probe_auc_len_only": round(float(auc_len_only), 4),
    }
    if do_hmm:
        cov = float(dom_all.mean())
        e_in, e_out = err_all[dom_all].mean(), err_all[~dom_all].mean()
        X = np.column_stack([md_all, dom_all.astype(float), len_all])
        auc_full = probe_auc(X, err_bin, grp_all, args.seed, stratify=(dis_meta == 1))
        auc_nodom = probe_auc(
            np.column_stack([md_all, len_all]),
            err_bin,
            grp_all,
            args.seed,
            stratify=(dis_meta == 1),
        )
        print(f"Pfam coverage {cov:.3f} | intervals {n_dom_hits}")
        print(f"mean err | in-domain {e_in:.4f} vs out {e_out:.4f}")
        print(f"err probe AUC [meta, in-domain, len] {auc_full:.4f} vs [meta, len] {auc_nodom:.4f}")
        out.update(
            {
                "pfam_coverage": round(cov, 4),
                "n_intervals": n_dom_hits,
                "mean_err_in_domain": round(float(e_in), 4),
                "mean_err_out_domain": round(float(e_out), 4),
                "err_probe_auc_full": round(float(auc_full), 4),
                "err_probe_auc_no_domain": round(float(auc_nodom), 4),
            }
        )
    open(args.out, "w").write(json.dumps(out, indent=2) + "\n")
    print(f"\nsaved: {args.out} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
