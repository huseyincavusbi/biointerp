"""Build a decontaminated ClinVar VEP manifest for Carbon-X pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from biointerp.core import RESULTS_DIR, split_by_protein
from biointerp.runner import load_dataset_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="clinvar_vep")
    p.add_argument("--train", default=None, help="optional local train parquet")
    p.add_argument("--test", default=None, help="optional local test parquet")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _download(dataset: str, train: str | None, test: str | None):
    cfg = load_dataset_config(dataset)
    if train is None or test is None:
        from huggingface_hub import hf_hub_download

        if train is None:
            train = hf_hub_download(cfg["hf_id"], cfg["file"], repo_type="dataset")
        if test is None:
            test = hf_hub_download(cfg["hf_id"], cfg["test_file"], repo_type="dataset")
    return Path(train), Path(test), cfg


def _normalize(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ref_col = cfg.get("ref_column", "ref_forward_sequence")
    alt_col = cfg.get("alt_column", "alt_forward_sequence")
    label_col = cfg.get("label_column", "label")
    chrom_col = cfg.get("chrom_column", "chromosome")
    out = pd.DataFrame(
        {
            "chrom": frame[chrom_col].astype(str),
            "position": frame["position"].astype(int),
            "ref_base": frame["ref"].astype(str).str.upper(),
            "alt_base": frame["alt"].astype(str).str.upper(),
            "ref": frame[ref_col].astype(str).str.upper(),
            "alt": frame[alt_col].astype(str).str.upper(),
            "label": frame[label_col].astype(int),
        }
    )
    out["key"] = (
        out["chrom"]
        + ":"
        + out["position"].astype(str)
        + ":"
        + out["ref_base"]
        + ":"
        + out["alt_base"]
    )
    return out


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    mask = []
    for ref, alt, ref_base, alt_base in zip(
        frame["ref"], frame["alt"], frame["ref_base"], frame["alt_base"]
    ):
        if len(ref) != len(alt) or len(ref) < 2:
            mask.append(False)
            continue
        if len(ref_base) != 1 or len(alt_base) != 1 or ref_base == alt_base:
            mask.append(False)
            continue
        if ref_base not in "ACGT" or alt_base not in "ACGT":
            mask.append(False)
            continue
        differences = [i for i, (a, b) in enumerate(zip(ref, alt)) if a != b]
        mask.append(len(differences) == 1 and ref[differences[0]] == ref_base)
    return frame[np.array(mask, dtype=bool)]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_balanced(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.sample(frac=1, random_state=seed)
    per_label = n // 2
    selected = []
    remaining = frame
    for label in (0, 1):
        pool = remaining[remaining["label"] == label]
        take = min(per_label, len(pool))
        if take:
            picked = pool.sample(n=take, random_state=seed)
            selected.append(picked)
            remaining = remaining.drop(picked.index)
    selected_frame = pd.concat(selected) if selected else frame.iloc[0:0]
    missing = n - len(selected_frame)
    if missing > 0:
        selected_frame = pd.concat([selected_frame, remaining.sample(n=missing, random_state=seed)])
    return selected_frame.sample(frac=1, random_state=seed)


def build_manifest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict,
    n: int,
    test_frac: float,
    seed: int,
) -> dict:
    train = _normalize(train, cfg)
    test = _normalize(test, cfg)
    raw_train, raw_test = len(train), len(test)
    train = _valid(train)
    test = _valid(test)
    train = train.drop_duplicates("key", keep="first")
    test_keys = set(test["key"])
    overlap = train["key"].isin(test_keys)
    excluded = int(overlap.sum())
    train = train[~overlap]
    eligible = len(train)
    sampled = _sample_balanced(train, n, seed).copy()
    train_mask, test_mask = split_by_protein(
        sampled["chrom"].to_numpy(), test_frac=test_frac, seed=seed
    )
    sampled["split"] = np.where(test_mask, "test", "train")
    variants = [
        {
            "chrom": row.chrom,
            "position": int(row.position),
            "ref": row.ref_base,
            "alt": row.alt_base,
            "label": int(row.label),
            "split": row.split,
        }
        for row in sampled.itertuples()
    ]
    variants.sort(key=lambda row: (row["chrom"], row["position"], row["ref"], row["alt"]))
    return {
        "filters": {
            "single_nucleotide": True,
            "equal_length": True,
            "acgt_only": True,
            "deduplicate": True,
            "exclude_test_overlap": True,
        },
        "counts": {
            "train_raw": raw_train,
            "test_raw": raw_test,
            "train_valid_unique": eligible,
            "excluded_test_overlap": excluded,
            "sampled": len(variants),
        },
        "sampling": {
            "n": n,
            "seed": seed,
            "label_counts": {
                str(label): int((sampled["label"] == label).sum()) for label in (0, 1)
            },
        },
        "split": {
            "test_frac": test_frac,
            "seed": seed,
            "train": int((sampled["split"] == "train").sum()),
            "test": int((sampled["split"] == "test").sum()),
            "test_chromosomes": sorted(set(sampled.loc[sampled["split"] == "test", "chrom"])),
        },
        "variants": variants,
    }


def main() -> None:
    args = parse_args()
    train_path, test_path, cfg = _download(args.dataset, args.train, args.test)
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    manifest = build_manifest(train, test, cfg, args.n, args.test_frac, args.seed)
    manifest["source"] = {
        "hf_id": cfg.get("hf_id"),
        "train_file": cfg.get("file"),
        "test_file": cfg.get("test_file"),
        "train_sha256": _hash(train_path),
        "test_sha256": _hash(test_path),
    }
    out = Path(args.out) if args.out else RESULTS_DIR / "clinvar_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"eligible={manifest['counts']['train_valid_unique']} "
        f"excluded_test_overlap={manifest['counts']['excluded_test_overlap']} "
        f"sampled={manifest['counts']['sampled']}"
    )
    print(
        f"split: train={manifest['split']['train']} "
        f"test={manifest['split']['test']} "
        f"test_chromosomes={manifest['split']['test_chromosomes']}"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
