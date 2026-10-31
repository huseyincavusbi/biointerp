"""DNA adapter core: tokenization-agnostic CETT and variant scoring.

The adapters cover two architectures:

- ``carbon``  — Llama/Qwen-style causal decoder (``model.model.layers``)
- ``dnabert`` — BERT-style masked encoder (``model.bert.encoder.layer``)

Both expose a standard FFN output projection, so the CETT formula is the same:

    CETT_j = |z_j| * ||W_out[:, j]|| / (||h|| + eps)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

DNA_ALPHABET = "ACGT"
DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(DNA_COMPLEMENT)[::-1]


def pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_layers(model, adapter: str):
    """Return the transformer block list for a DNA adapter."""
    if adapter == "carbon":
        base = getattr(model, "model", model)
        layers = getattr(base, "layers", None)
        if layers is not None:
            return layers
    elif adapter == "dnabert":
        for path in (("bert", "encoder", "layer"), ("encoder", "layer")):
            obj = model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
    raise ValueError(f"cannot find transformer layers for adapter {adapter!r}")


def get_ffn_output_proj(layer, adapter: str):
    """Return the FFN output projection that CETT hooks."""
    if adapter == "carbon":
        mlp = getattr(layer, "mlp", None)
        proj = getattr(mlp, "down_proj", None) if mlp is not None else None
        if proj is not None:
            return proj
    elif adapter == "dnabert":
        output = getattr(layer, "output", None)
        proj = getattr(output, "dense", None) if output is not None else None
        if proj is not None:
            return proj
    raise ValueError(f"cannot find FFN output projection for adapter {adapter!r}")


def precompute_col_norms(
    model, adapter: str, layers: Optional[List[int]] = None
) -> Dict[int, torch.Tensor]:
    blocks = get_layers(model, adapter)
    indices = list(range(len(blocks))) if layers is None else layers
    norms: Dict[int, torch.Tensor] = {}
    for i in indices:
        module = get_ffn_output_proj(blocks[i], adapter)
        weight = module.weight.detach().float()
        dim = 1 if type(module).__name__ == "Conv1D" else 0
        norms[i] = torch.norm(weight, dim=dim).cpu()
    return norms


def load_dna_model(spec: dict, device: Optional[str] = None):
    """Load a DNA model from a registry spec and return ``(model, tokenizer, device)``."""
    adapter = spec.get("adapter")
    hf_id = spec["hf_id"]
    dev = pick_device(device)

    if adapter == "carbon":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True)
    elif adapter == "dnabert":
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        try:
            model = AutoModelForMaskedLM.from_pretrained(hf_id, trust_remote_code=True)
        except (ValueError, OSError):
            from transformers.models.bert.configuration_bert import BertConfig

            config = BertConfig.from_pretrained(hf_id)
            model = AutoModelForMaskedLM.from_pretrained(
                hf_id, trust_remote_code=True, config=config
            )
    else:
        raise ValueError(f"unknown DNA adapter {adapter!r} for model {hf_id!r}")

    model.eval()
    model.to(dev)
    return model, tokenizer, dev


@torch.inference_mode()
def extract_cett(
    model,
    tokens: dict,
    positions: torch.Tensor,
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    adapter: str,
):
    """Run one forward pass and return CETT + logits at the given token positions."""
    blocks = get_layers(model, adapter)
    positions = positions.to(tokens["input_ids"].device)
    batch = positions.shape[0]
    b_idx = torch.arange(batch, device=positions.device).unsqueeze(1)

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            z_cache[idx] = inputs[0].detach().float().cpu()
            h_cache[idx] = output.detach().float().cpu()
            return output

        return hook

    for i in layers:
        handles.append(get_ffn_output_proj(blocks[i], adapter).register_forward_hook(make_hook(i)))
    try:
        out = model(**tokens)
    finally:
        for handle in handles:
            handle.remove()

    cett_parts = []
    for i in layers:
        z = z_cache[i][b_idx, positions]
        h = h_cache[i][b_idx, positions]
        cett_parts.append(torch.abs(z) * col_norms[i] / (h.norm(dim=-1, keepdim=True) + 1e-8))
    cett = torch.cat(cett_parts, dim=-1)

    logits = out.logits[b_idx, positions]
    return {
        "cett": cett.reshape(batch * positions.shape[1], -1),
        "logits": logits.reshape(batch * positions.shape[1], -1),
    }


def extract_cett_masked(model, tokens, positions, layers, col_norms, adapter):
    return extract_cett(model, tokens, positions, layers, col_norms, adapter)


def extract_cett_causal(model, tokens, positions, layers, col_norms, adapter):
    return extract_cett(model, tokens, positions, layers, col_norms, adapter)


@torch.inference_mode()
def score_ref_alt(model, tokens_ref, tokens_alt, position: int, ref_id: int, alt_id: int) -> float:
    """Return log p(alt) - log p(ref) at one token position."""
    ref_logits = model(**tokens_ref).logits[0, position].float()
    alt_logits = model(**tokens_alt).logits[0, position].float()
    ref_logp = torch.log_softmax(ref_logits, dim=-1)
    alt_logp = torch.log_softmax(alt_logits, dim=-1)
    return float(alt_logp[alt_id] - ref_logp[ref_id])
