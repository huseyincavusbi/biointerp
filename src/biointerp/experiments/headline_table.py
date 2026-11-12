"""Build one canonical headline table with intervals and FDR correction.

Every claim that appears in the paper or thesis should be generated here:
cluster-split AUROC, random-split AUROC, pLDDT correlations, cross-model
replication, and the six-assay probe-versus-confidence deltas. Assay deltas use
the saved per-mutation probe scores and a paired bootstrap; the family is
corrected with Benjamini-Hochberg.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from biointerp.core import DEFAULT_NPZ, RESULTS_DIR, split_by_protein, top_probe_scores
from biointerp.inference import (
    bh_fdr,
    bootstrap_auroc,
    bootstrap_tail_p,
    paired_bootstrap_delta,
    percentile_ci,
)

CORE_ASSAYS = (
    "SDA_BACSU_Tsuboyama_2023_1PV0",
    "MTHR_HUMAN_Weile_2021",
    "S22A1_HUMAN_Yee_2023_activity",
    "MSH2_HUMAN_Jia_2020",
    "PPARG_HUMAN_Majithia_2016",
    "PAI1_HUMAN_Huttinger_2021",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--assignments", default=str(RESULTS_DIR / "split_assignments.tsv"))
    p.add_argument("--confound", default=str(RESULTS_DIR / "confound_matching.json"))
    p.add_argument("--assays", default=",".join(CORE_ASSAYS))
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--markdown", default=None)
    return p.parse_args()


def load_split(npz: str, assignments: str):
    data = np.load(npz, allow_pickle=True)
    x = data["cett"]
    y = (~data["correct"]).astype(int)
    protein = data["protein"]

    split_of: dict[str, str] = {}
    for line in Path(assignments).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        record_id, _cluster, split = line.split("\t")
        split_of[record_id] = split

    ids = np.asarray([f"seq{int(p)}" for p in protein])
    keep = np.asarray([split_of.get(record_id) in {"train", "test"} for record_id in ids])
    is_test = np.asarray([split_of[record_id] == "test" for record_id in ids[keep]])
    return x[keep], y[keep], protein[keep], ~is_test, is_test


def core_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    x, y, protein, train, test = load_split(args.npz, args.assignments)
    rows: list[dict[str, object]] = []

    scores, _ = top_probe_scores(x[train], y[train], x[test], top_k=args.top_k, seed=args.seed)
    cluster_auroc = float(roc_auc_score(y[test], scores))
    cluster_boot = bootstrap_auroc(
        y[test], scores, groups=protein[test], n_boot=args.n_boot, seed=args.seed
    )
    rows.append(
        {
            "claim": "cluster_split_probe_auroc",
            "family": "core",
            "estimate": round(cluster_auroc, 4),
            "ci95": percentile_ci(cluster_boot),
            "n": int(test.sum()),
            "source": "cluster split, ESM-2 650M CETT",
        }
    )

    random_train, random_test = split_by_protein(protein, test_frac=args.test_frac, seed=args.seed)
    random_scores, _ = top_probe_scores(
        x[random_train], y[random_train], x[random_test], top_k=args.top_k, seed=args.seed
    )
    random_auroc = float(roc_auc_score(y[random_test], random_scores))
    random_boot = bootstrap_auroc(
        y[random_test],
        random_scores,
        groups=protein[random_test],
        n_boot=args.n_boot,
        seed=args.seed,
    )
    rows.append(
        {
            "claim": "random_split_probe_auroc",
            "family": "core",
            "estimate": round(random_auroc, 4),
            "ci95": percentile_ci(random_boot),
            "n": int(random_test.sum()),
            "source": "random protein split, ESM-2 650M CETT",
        }
    )

    confound_path = Path(args.confound)
    if confound_path.exists():
        confound = json.loads(confound_path.read_text())
        rows.append(
            {
                "claim": "plddt_spearman_raw",
                "family": "core",
                "estimate": confound["raw"]["spearman"],
                "ci95": confound["raw"]["ci95"],
                "n": confound["n_positions"],
                "source": "confound_matching.json",
            }
        )
        rows.append(
            {
                "claim": "plddt_spearman_partial",
                "family": "core",
                "estimate": confound["partial"]["spearman"],
                "ci95": confound["partial"]["ci95"],
                "n": confound["n_positions"],
                "source": "confound_matching.json",
            }
        )

    cross_model = RESULTS_DIR / "cross_model.json"
    if cross_model.exists():
        payload = json.loads(cross_model.read_text())
        rows.append(
            {
                "claim": "cross_model_esmc600m_auroc",
                "family": "core",
                "estimate": round(float(payload["auroc_probe"]), 4),
                "ci95": None,
                "n": int(payload["n"]),
                "source": "cross_model.json (no per-row scores)",
            }
        )
    return rows


def assay_row(assay: str, args: argparse.Namespace) -> dict[str, object]:
    result_path = RESULTS_DIR / f"vep_proteingym_{assay}.json"
    predictions_path = RESULTS_DIR / f"vep_proteingym_{assay}.predictions.jsonl"
    payload = json.loads(result_path.read_text())

    row: dict[str, object] = {
        "claim": f"assay:{assay}:probe_minus_confidence",
        "family": "assay",
        "assay": assay,
        "estimate": round(payload["auroc_probe"] - payload["auroc_confidence"], 4),
        "ci95": None,
        "p_value": None,
        "n": int(payload["n_mutations"]),
        "auroc_probe": payload["auroc_probe"],
        "auroc_confidence": payload["auroc_confidence"],
        "source": str(result_path),
        "status": "summary_only",
    }
    if not predictions_path.exists():
        return row

    records = [json.loads(line) for line in predictions_path.read_text().splitlines() if line]
    test_records = [record for record in records if record.get("split") == "test"]
    if not test_records:
        return row

    y = np.asarray([0 if record["correct"] else 1 for record in test_records], dtype=int)
    probe = np.asarray([record["probe_score"] for record in test_records], dtype=float)
    confidence = np.asarray([record["confidence"] for record in test_records], dtype=float)
    if len(np.unique(y)) < 2:
        return row

    delta_null = paired_bootstrap_delta(y, probe, -confidence, n_boot=args.n_boot, seed=args.seed)
    row.update(
        {
            "estimate": round(float(roc_auc_score(y, probe) - roc_auc_score(y, -confidence)), 4),
            "ci95": percentile_ci(delta_null),
            "p_value": round(bootstrap_tail_p(delta_null), 6),
            "n": int(len(test_records)),
            "status": "ok",
        }
    )
    return row


def render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Claim | Estimate | 95% CI | n | Family |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        ci = row.get("ci95")
        ci_text = f"[{ci[0]}, {ci[1]}]" if ci else "pending"
        lines.append(
            f"| {row['claim']} | {row['estimate']} | {ci_text} | {row['n']} | {row['family']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    started = time.time()

    rows = core_rows(args)
    assay_names = [name.strip() for name in args.assays.split(",") if name.strip()]
    assay_rows = [assay_row(name, args) for name in assay_names]

    p_values = [row["p_value"] for row in assay_rows if row.get("p_value") is not None]
    if p_values:
        q_values = bh_fdr(p_values)
        cursor = 0
        for row in assay_rows:
            if row.get("p_value") is not None:
                row["q_value"] = round(float(q_values[cursor]), 6)
                cursor += 1

    wins = sum(1 for row in assay_rows if row.get("ci95") and row["ci95"][0] > 0)
    losses = sum(1 for row in assay_rows if row.get("ci95") and row["ci95"][1] < 0)
    ties = sum(1 for row in assay_rows if row.get("ci95") and row["ci95"][0] <= 0 <= row["ci95"][1])
    pending = sum(1 for row in assay_rows if row.get("ci95") is None)

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {
            "top_k": args.top_k,
            "test_frac": args.test_frac,
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "rows": rows + assay_rows,
        "assay_summary": {
            "n": len(assay_rows),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "pending": pending,
        },
        "elapsed_s": round(time.time() - started, 1),
    }

    out = Path(args.out) if args.out else RESULTS_DIR / "headline_table.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    markdown = render_markdown(result["rows"])
    md_path = Path(args.markdown) if args.markdown else RESULTS_DIR / "headline_table.md"
    md_path.write_text(markdown)

    print(markdown)
    print(f"assays: {wins} wins / {ties} ties / {losses} losses / {pending} pending")
    print(f"wrote: {out}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
