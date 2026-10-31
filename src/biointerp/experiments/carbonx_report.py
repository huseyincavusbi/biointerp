"""Build CSV/JSON summaries from a Carbon-X pilot run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="pilot output directory")
    p.add_argument("--out", default=None, help="report directory")
    return p.parse_args()


def _records(run_dir: Path) -> list[dict]:
    path = run_dir / "records.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _answers(records: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "condition": record["condition"],
            "label": record["parsed"]["label"],
        }
        for record in records
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["condition", "label", "count"])
    return (
        frame.groupby(["condition", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["condition", "label"])
    )


def _probabilities(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "condition": record["condition"],
                "variant": record["variant"],
                "p_pathogenic": record.get("p_pathogenic"),
            }
            for record in records
        ]
    )
    if frame.empty or frame["p_pathogenic"].isna().all():
        return pd.DataFrame(columns=["condition", "n", "mean_p", "std_p"])
    frame = frame.dropna(subset=["p_pathogenic"])
    return (
        frame.groupby("condition")["p_pathogenic"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "mean_p", "std": "std_p"})
    )


def _flips(records: list[dict]) -> pd.DataFrame:
    conditions = sorted({record["condition"] for record in records})
    rows = []
    for variant in sorted({record["variant"] for record in records}):
        subset = [record for record in records if record["variant"] == variant]
        labels = {record["condition"]: record["parsed"]["label"] for record in subset}
        informative = {label for label in labels.values() if label not in ("unparsed", "ambiguous")}
        row = {
            "variant": variant,
            "chrom": subset[0]["chrom"],
            "label": subset[0]["label"],
        }
        row.update({condition: labels.get(condition) for condition in conditions})
        row["n_labels"] = len(informative)
        row["flipped"] = len(informative) > 1
        rows.append(row)
    return pd.DataFrame(rows)


def _features(run_dir: Path, records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = run_dir / "summary.json"
    layers = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        layers = summary.get("config", {}).get("layers", [])
    cett_rows = []
    attention_rows = []
    for record in records:
        path = record.get("features")
        if not path:
            continue
        feature = np.load(path)
        if "cett" in feature and layers:
            values = feature["cett"]
            if len(values) % len(layers) == 0:
                width = len(values) // len(layers)
                for index, layer in enumerate(layers):
                    part = values[index * width : (index + 1) * width]
                    cett_rows.append(
                        {
                            "variant": record["variant"],
                            "condition": record["condition"],
                            "layer": layer,
                            "norm": float(np.linalg.norm(part)),
                            "mean_abs": float(np.mean(np.abs(part))),
                        }
                    )
        if "attention_dna" in feature:
            values = feature["attention_dna"]
            if values.ndim == 3:
                means = values.mean(axis=(1, 2))
                for layer, value in enumerate(means):
                    attention_rows.append(
                        {
                            "variant": record["variant"],
                            "condition": record["condition"],
                            "layer": layer,
                            "mean_attention": float(value),
                        }
                    )
    return (
        pd.DataFrame(cett_rows),
        pd.DataFrame(attention_rows),
    )


def build_report(run_dir: Path, out_dir: Path) -> dict:
    records = _records(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers = _answers(records)
    probabilities = _probabilities(records)
    flips = _flips(records)
    cett, attention = _features(run_dir, records)
    files = {
        "answers": out_dir / "answers.csv",
        "probabilities": out_dir / "probabilities.csv",
        "flips": out_dir / "flips.csv",
        "cett": out_dir / "cett_norms.csv",
        "attention": out_dir / "attention.csv",
    }
    answers.to_csv(files["answers"], index=False)
    probabilities.to_csv(files["probabilities"], index=False)
    flips.to_csv(files["flips"], index=False)
    cett.to_csv(files["cett"], index=False)
    attention.to_csv(files["attention"], index=False)

    summary = {
        "run": str(run_dir),
        "n_records": len(records),
        "n_variants": len({record["variant"] for record in records}),
        "conditions": sorted({record["condition"] for record in records}),
        "n_flipped": int(flips["flipped"].sum()) if not flips.empty else 0,
        "probabilities": probabilities.to_dict(orient="records"),
        "files": {name: str(path) for name, path in files.items()},
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir / "report"
    summary = build_report(run_dir, out_dir)
    print(
        f"records={summary['n_records']} variants={summary['n_variants']} "
        f"flipped={summary['n_flipped']}"
    )
    for name, path in summary["files"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
