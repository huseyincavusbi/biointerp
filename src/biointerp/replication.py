"""Modality-agnostic replication runner for error-neuron probes.

The runner owns the reference positions, split, probe, bootstrap, and report
schema. Modalities provide a small adapter that masks one position and returns
CETT features plus token probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from biointerp.core import apply_probe, fit_top_probe, univariate_auroc
from biointerp.inference import bootstrap_auroc, percentile_ci


class ModelAdapter(Protocol):
    """Minimal interface a modality must provide for replication."""

    name: str
    n_layers: int
    intermediate: int

    def extract(self, sequence: str, position: int) -> tuple[np.ndarray, np.ndarray]:
        """Return CETT features and token probabilities at a 1-based position."""

    def decode(self, probabilities: np.ndarray) -> str:
        """Map token probabilities to the residue/base token."""


@dataclass
class ReplicationInputs:
    reference_npz: str
    assignments: str
    sequences: Sequence[str]
    adapter: ModelAdapter
    top_k: int = 25
    bootstrap: int = 1000
    seed: int = 0
    n: int = 0


def load_reference(path: str, n: int = 0) -> dict[str, object]:
    """Load the reference extraction, optionally truncated to ``n`` rows."""
    data = np.load(path, allow_pickle=True)
    total = len(data["position"])
    limit = total if n <= 0 else min(n, total)
    return {
        "cett": data["cett"][:limit].astype(np.float32),
        "correct": data["correct"].astype(bool)[:limit],
        "position": data["position"].astype(int)[:limit],
        "true_token": np.asarray([str(value) for value in data["true_aa"]])[:limit],
        "protein": data["protein"].astype(int)[:limit],
        "n_layers": int(data["n_layers"]),
        "intermediate": int(data["intermediate_dim"]),
        "model": str(data["model"]),
    }


def load_split(assignments: str, ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return keep and test masks from a split-assignments TSV."""
    split_of: dict[str, str] = {}
    for line in Path(assignments).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        record_id, _cluster, split = line.split("\t")
        split_of[record_id] = split
    keep = np.asarray([split_of.get(record_id) in {"train", "test"} for record_id in ids])
    is_test = np.asarray([split_of.get(record_id) == "test" for record_id in ids])
    return keep, is_test


def layer_profile(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    n_layers: int,
    intermediate: int,
) -> np.ndarray:
    """Best single-unit AUROC per layer on the training split."""
    profile = np.empty(n_layers)
    for layer in range(n_layers):
        start = layer * intermediate
        block = x[train, start : start + intermediate]
        profile[layer] = float(univariate_auroc(block, y[train]).max())
    return profile


def neuron_layer_histogram(
    indices: np.ndarray, n_layers: int, intermediate: int, bins: int = 10
) -> np.ndarray:
    """Histogram of selected units over normalized depth."""
    layers = np.asarray(indices) // intermediate
    normalized = layers / max(1, n_layers - 1)
    counts, _ = np.histogram(normalized, bins=bins, range=(0.0, 1.0))
    return counts / max(1, counts.sum())


def extract_features(
    reference: dict[str, object],
    sequences: Sequence[str],
    adapter: ModelAdapter,
    progress_every: int = 20,
) -> dict[str, np.ndarray]:
    """Run the adapter at every reference position."""
    positions = reference["position"]  # type: ignore[index]
    true_tokens = reference["true_token"]  # type: ignore[index]
    if len(sequences) < len(positions):
        raise ValueError("not enough sequences for the reference positions")

    import time

    features: list[np.ndarray] = []
    correct: list[bool] = []
    confidence: list[float] = []
    predictions: list[str] = []
    start = time.time()
    for index, sequence in enumerate(sequences[: len(positions)]):
        cett, probs = adapter.extract(sequence, int(positions[index]))
        prediction = adapter.decode(probs)
        features.append(np.asarray(cett, dtype=np.float16))
        predictions.append(prediction)
        correct.append(prediction == str(true_tokens[index]))
        confidence.append(float(np.max(probs)))
        if (index + 1) % progress_every == 0 or index + 1 == len(positions):
            print(
                f"seq {index + 1}/{len(positions)} | acc {np.mean(correct):.4f} | "
                f"{time.time() - start:.0f}s"
            )

    return {
        "cett": np.stack(features),
        "correct": np.asarray(correct),
        "confidence": np.asarray(confidence),
        "pred_aa": np.asarray(predictions),
    }


def analyze(
    reference: dict[str, object],
    extracted: dict[str, np.ndarray],
    keep: np.ndarray,
    is_test: np.ndarray,
    adapter: ModelAdapter,
    top_k: int,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Fit both probes and compare them on the same held-out rows."""
    train = keep & ~is_test
    test = keep & is_test
    x_new = extracted["cett"].astype(np.float32)
    y_new = (~extracted["correct"]).astype(int)
    confidence = extracted["confidence"]

    x_ref = reference["cett"]
    y_ref = (~reference["correct"]).astype(int)
    protein_ref = reference["protein"]

    scaler_new, clf_new, idx_new = fit_top_probe(x_new[train], y_new[train], top_k=top_k, seed=seed)
    scores_new = apply_probe(scaler_new, clf_new, idx_new, x_new[test])
    auroc_new = float(roc_auc_score(y_new[test], scores_new))

    scaler_ref, clf_ref, idx_ref = fit_top_probe(x_ref[train], y_ref[train], top_k=top_k, seed=seed)
    scores_ref = apply_probe(scaler_ref, clf_ref, idx_ref, x_ref[test])
    auroc_ref = float(roc_auc_score(y_ref[test], scores_ref))

    groups = protein_ref[test]
    new_ci = percentile_ci(
        bootstrap_auroc(y_new[test], scores_new, groups=groups, n_boot=bootstrap, seed=seed)
    )
    ref_ci = percentile_ci(
        bootstrap_auroc(y_ref[test], scores_ref, groups=groups, n_boot=bootstrap, seed=seed)
    )

    ref_profile = layer_profile(
        x_ref,
        y_ref,
        train,
        int(reference["n_layers"]),
        int(reference["intermediate"]),
    )
    new_profile = layer_profile(x_new, y_new, train, adapter.n_layers, adapter.intermediate)
    ref_grid = np.linspace(0.0, 1.0, int(reference["n_layers"]))
    new_profile_interp = np.interp(ref_grid, np.linspace(0.0, 1.0, adapter.n_layers), new_profile)

    ref_histogram = neuron_layer_histogram(
        idx_ref,
        int(reference["n_layers"]),
        int(reference["intermediate"]),
    )
    new_histogram = neuron_layer_histogram(idx_new, adapter.n_layers, adapter.intermediate)

    return {
        "adapter": {
            "name": adapter.name,
            "n": int(keep.sum()),
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "accuracy": round(float(extracted["correct"][keep].mean()), 4),
            "auroc_probe": round(auroc_new, 4),
            "ci95": new_ci,
            "auroc_confidence": round(float(roc_auc_score(y_new[test], -confidence[test])), 4),
        },
        "reference": {
            "name": str(reference["model"]),
            "accuracy": round(float(reference["correct"][keep].mean()), 4),
            "auroc_probe": round(auroc_ref, 4),
            "ci95": ref_ci,
        },
        "comparison": {
            "probe_delta": round(auroc_new - auroc_ref, 4),
            "score_spearman": round(float(spearmanr(scores_ref, scores_new).statistic), 4),
            "layer_profile_pearson": round(
                float(pearsonr(ref_profile, new_profile_interp).statistic), 4
            ),
            "layer_histogram_pearson": round(
                float(pearsonr(ref_histogram, new_histogram).statistic), 4
            ),
            "unit_overlap": None,
            "note": ("feature widths differ; unit identity is not comparable across models"),
        },
        "split": {
            "n": int(len(keep)),
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "test_frac": round(float(test.sum() / max(1, keep.sum())), 4),
        },
    }
