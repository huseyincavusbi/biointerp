"""Model adapters for the modality-agnostic replication runner.

Each adapter owns masking, feature extraction, and decoding for one model
family. The runner never sees tokenizers or layer layouts.
"""

from __future__ import annotations

import numpy as np
import torch

from biointerp import esm as esm_module
from biointerp import esmc as esmc_module
from biointerp.esm import AA_ALPHABET, aa_token_ids, extract_cett_batch, get_transformer_layers
from biointerp.models import get_spec


class Esm2Adapter:
    """ESM-2 adapter: residue-level masking with the ESM tokenizer."""

    name = "esm2"

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str | None = None,
        model=None,
        tokenizer=None,
    ):
        if (model is None) != (tokenizer is None):
            raise ValueError("pass both model and tokenizer, or neither")
        if model is None:
            model, tokenizer, device = esm_module.load_model(model_id, device, dtype)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or "cpu"
        self.n_layers = len(get_transformer_layers(model))
        config = getattr(model, "config", None)
        intermediate = getattr(config, "intermediate_size", None)
        if intermediate is None:
            intermediate = getattr(model, "intermediate_size")
        self.intermediate = int(intermediate)
        self._col_norms = esm_module.precompute_col_norms(model, list(range(self.n_layers)))
        self._aa_ids = aa_token_ids(tokenizer)

    def extract(self, sequence: str, position: int) -> tuple[np.ndarray, np.ndarray]:
        ids = list(self.tokenizer(sequence, add_special_tokens=True)["input_ids"])
        if not 1 <= position < len(ids) - 1:
            raise ValueError(f"position {position} outside tokenized sequence")
        ids[position] = self.tokenizer.mask_token_id
        tokens = {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=self.device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
        }
        mask_positions = torch.tensor([[position]], dtype=torch.long, device=self.device)
        result = extract_cett_batch(
            self.model,
            tokens,
            mask_positions,
            list(range(self.n_layers)),
            self._col_norms,
            self._aa_ids,
        )
        return (
            result["cett"][0].float().cpu().numpy(),
            result["probs"][0].float().cpu().numpy(),
        )

    def decode(self, probabilities: np.ndarray) -> str:
        return AA_ALPHABET[int(np.argmax(probabilities))]


class EsmcAdapter:
    """ESMC adapter: residue-level masking with the fast tokenizer fallback."""

    name = "esmc"

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str | None = None,
        device_map: str | None = None,
        model=None,
        tokenizer=None,
    ):
        if (model is None) != (tokenizer is None):
            raise ValueError("pass both model and tokenizer, or neither")
        if model is None:
            model, tokenizer, device = esmc_module.load_esmc(model_id, device, dtype, device_map)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or "cpu"
        self._col_norms = esmc_module.precompute_col_norms(model)
        self.n_layers = len(self._col_norms)
        self.intermediate = int(self._col_norms[0].shape[0])

    def extract(self, sequence: str, position: int) -> tuple[np.ndarray, np.ndarray]:
        cett, probs = esmc_module.extract_cett_sequence(
            self.model,
            self.tokenizer,
            sequence,
            [position - 1],
            self._col_norms,
        )
        return (
            cett[0].float().cpu().numpy(),
            probs[0].float().cpu().numpy(),
        )

    def decode(self, probabilities: np.ndarray) -> str:
        return AA_ALPHABET[int(np.argmax(probabilities))]


def build_adapter(
    key_or_id: str,
    device: str | None = None,
    dtype: str | None = None,
    device_map: str | None = None,
):
    """Build the adapter declared in ``conf/model/<key>.yaml``."""
    spec = get_spec(key_or_id)
    adapter = spec.get("adapter", "esm")
    if adapter == "esm":
        return Esm2Adapter(spec["hf_id"], device, dtype)
    if adapter == "esmc":
        return EsmcAdapter(spec["hf_id"], device, dtype, device_map)
    if spec.get("modality") == "dna":
        raise NotImplementedError(
            "DNA replication needs a DnaBertAdapter/CarbonAdapter implementing "
            "ModelAdapter; the runner itself is modality-agnostic"
        )
    raise ValueError(f"unknown adapter {adapter!r} for model {key_or_id!r}")
