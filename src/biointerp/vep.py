"""Reusable variant-effect scoring for ProteinGym-style assays.

The scorer owns model-specific masking and CETT extraction. The runner owns
splitting, probe fitting, bootstrap intervals, and metrics. Assays supply rows;
models and datasets are resolved elsewhere from configuration.
"""

from __future__ import annotations

import re
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from biointerp.core import apply_probe, fit_top_probe
from biointerp.esm import (
    AA_ALPHABET,
    aa_token_ids,
    extract_cett_batch,
    get_transformer_layers,
    precompute_col_norms,
)
from biointerp.inference import (
    bootstrap_tail_p,
    paired_bootstrap_delta,
    percentile_ci,
)

MUT_SEG = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_mutations(mutant: str) -> list[tuple[str, int, str]] | None:
    """Parse ``A12C`` or ``A12C:G14D``; return None on malformed input."""
    mutations = []
    for segment in str(mutant).split(":"):
        match = MUT_SEG.match(segment.strip())
        if not match:
            return None
        mutations.append((match.group(1), int(match.group(2)), match.group(3)))
    return mutations


def prepare_rows(frame, n_mutations: int, seed: int) -> list[dict[str, object]]:
    """Keep single substitutions, then sample up to ``n_mutations``."""
    rows: list[dict[str, object]] = []
    for _, record in frame.iterrows():
        mutations = parse_mutations(record["mutant"])
        if not mutations or len(mutations) != 1:
            continue
        rows.append(
            {
                "muts": mutations,
                "score": float(record["DMS_score"]),
                "bin": float(record["DMS_score_bin"]),
                "target": str(record["target_seq"]),
            }
        )

    if n_mutations and n_mutations < len(rows):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(rows), size=n_mutations, replace=False)
        rows = [rows[index] for index in sorted(pick)]
    return rows


class Esm2Scorer:
    """Batched masked-marginal scorer for the ESM-2 adapter family."""

    name = "esm2"

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_layers = len(get_transformer_layers(model))
        config = getattr(model, "config", None)
        intermediate = getattr(config, "intermediate_size", None)
        if intermediate is None:
            intermediate = getattr(model, "intermediate_size")
        self.intermediate = int(intermediate)
        self._layers = list(range(self.n_layers))
        self._col_norms = precompute_col_norms(model, self._layers)
        self._aa_ids = aa_token_ids(tokenizer)
        self._aa_index = {aa: index for index, aa in enumerate(AA_ALPHABET)}

    def score_rows(
        self, rows: Sequence[Mapping[str, object]], batch_size: int = 16
    ) -> dict[str, np.ndarray]:
        groups: dict[int, list[Mapping[str, object]]] = {}
        for row in rows:
            groups.setdefault(len(row["muts"]), []).append(row)  # type: ignore[arg-type]

        scores: list[float] = []
        confidence: list[float] = []
        correct: list[bool] = []
        cett_all: list[np.ndarray] = []
        started = time.time()
        done = 0

        for mutation_count, group_rows in sorted(groups.items()):
            for start in range(0, len(group_rows), batch_size):
                batch = group_rows[start : start + batch_size]
                ids_list, positions = [], []
                for row in batch:
                    ids = list(self.tokenizer(row["target"], add_special_tokens=True)["input_ids"])
                    position_list = []
                    for _wild_type, position, _mutant in row["muts"]:  # type: ignore[index]
                        ids[position] = self.tokenizer.mask_token_id
                        position_list.append(position)
                    ids_list.append(ids)
                    positions.append(position_list)

                tokens = {
                    "input_ids": torch.tensor(ids_list, device=self.device),
                    "attention_mask": torch.ones(
                        len(ids_list), len(ids_list[0]), dtype=torch.long, device=self.device
                    ),
                }
                position_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
                result = extract_cett_batch(
                    self.model,
                    tokens,
                    position_tensor,
                    self._layers,
                    self._col_norms,
                    self._aa_ids,
                )
                logits = result["logits_aa"].cpu().numpy()
                probabilities = result["probs"].cpu().numpy()
                cett = result["cett"].cpu().numpy().astype(np.float16)

                for index, row in enumerate(batch):
                    score = 0.0
                    for offset, (wild_type, _position, mutant) in enumerate(row["muts"]):  # type: ignore[index]
                        score += float(
                            logits[index * mutation_count + offset, self._aa_index[mutant]]
                            - logits[index * mutation_count + offset, self._aa_index[wild_type]]
                        )
                    scores.append(score)
                    confidence.append(
                        float(
                            probabilities[
                                index * mutation_count : (index + 1) * mutation_count
                            ].max()
                        )
                    )
                    correct.append((score < 0) == (row["bin"] < 0.5))  # type: ignore[operator]
                    cett_all.append(
                        cett[index * mutation_count : (index + 1) * mutation_count].mean(axis=0)
                    )

                done += len(batch)
                if done % (batch_size * 5) < batch_size:
                    print(f"  scored {done}/{len(rows)} | {time.time() - started:.0f}s")

        return {
            "scores": np.asarray(scores),
            "confidence": np.asarray(confidence),
            "correct": np.asarray(correct),
            "cett": np.stack(cett_all).astype(np.float32),
        }


def run_vep(
    rows: Sequence[Mapping[str, object]],
    scorer,
    *,
    top_k: int = 25,
    test_frac: float = 0.3,
    seed: int = 0,
    batch_size: int = 16,
    n_boot: int = 1000,
) -> dict[str, object]:
    """Score one assay, fit the error probe, and return metrics plus arrays."""
    scored = scorer.score_rows(rows, batch_size=batch_size)
    y = (~scored["correct"]).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("single-class labels")

    dms = np.asarray([row["score"] for row in rows], dtype=float)
    rho = float(spearmanr(scored["scores"], dms).statistic)

    n = len(y)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test = np.zeros(n, dtype=bool)
    test[permutation[:n_test]] = True
    train = ~test

    scaler, clf, idx = fit_top_probe(scored["cett"][train], y[train], top_k=top_k, seed=seed)
    probe_scores = np.full(n, np.nan)
    probe_scores[test] = apply_probe(scaler, clf, idx, scored["cett"][test])
    auroc_probe = float(roc_auc_score(y[test], probe_scores[test]))
    auroc_confidence = float(roc_auc_score(y[test], -scored["confidence"][test]))

    delta_null = paired_bootstrap_delta(
        y[test],
        probe_scores[test],
        -scored["confidence"][test],
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "n_mutations": n,
        "spearman": round(rho, 4),
        "sign_accuracy": round(float(1 - y.mean()), 4),
        "auroc_probe": round(auroc_probe, 4),
        "auroc_confidence": round(auroc_confidence, 4),
        "delta": round(auroc_probe - auroc_confidence, 4),
        "delta_ci95": percentile_ci(delta_null),
        "p_value": round(bootstrap_tail_p(delta_null), 6),
        "probe_scores": probe_scores,
        "test_mask": test,
        "y": y,
        "scores": scored["scores"],
        "confidence": scored["confidence"],
        "dms": dms,
    }
