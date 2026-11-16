"""ProteinGym dataset access driven by ``conf/dataset/<name>.yaml``.

The repository, shard pattern, and label column are configuration, not code.
"""

from __future__ import annotations

from dataclasses import dataclass

from biointerp.runner import load_dataset_config


@dataclass(frozen=True)
class ProteinGymDataset:
    name: str
    repo: str
    file_pattern: str
    label: str
    n_mutations: int

    def load_shard(self, shard: int):
        import pandas as pd
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            self.repo,
            self.file_pattern.format(shard=shard),
            repo_type="dataset",
        )
        return pd.read_parquet(path)

    def assay_frame(self, assay: str, shard: int):
        frame = self.load_shard(shard)
        return frame[frame["DMS_id"] == assay].reset_index(drop=True)


def from_config(name: str = "proteingym") -> ProteinGymDataset:
    spec = load_dataset_config(name)
    missing = [key for key in ("repo", "file_pattern", "label") if not spec.get(key)]
    if missing:
        raise ValueError(f"dataset config {name!r} is missing: {missing}")
    return ProteinGymDataset(
        name=name,
        repo=str(spec["repo"]),
        file_pattern=str(spec["file_pattern"]),
        label=str(spec["label"]),
        n_mutations=int(spec.get("n_mutations", 2000)),
    )
