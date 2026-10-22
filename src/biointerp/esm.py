"""ESM-2 adapter for CETT / error-neuron extraction (Bio-Interp pilot).

Methodology adapted from hprobes `cett.py`:
    hook the FFN output projection, capture z (its input) and h (its output), and
    compute CETT_j = |z_j| * ||W_out[:, j]|| / (||h|| + eps)

ESM-2 structure (HF): EsmForMaskedLM.esm.encoder.layer[i].output.dense
    z = GELU(intermediate.dense(hidden))   (input to output.dense)
    h = output.dense(z)                    (FFN contribution, pre-residual)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: Optional[str] = None,
    dtype: Optional[str] = None,
):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if dtype in ("bf16", "bfloat16"):
        try:
            model = AutoModelForMaskedLM.from_pretrained(model_name, dtype=torch.bfloat16)
        except TypeError:
            model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    dev = pick_device(device)
    model.to(dev)
    return model, tokenizer, dev


def get_transformer_layers(model):
    base = getattr(model, "esm", model)
    return base.encoder.layer


def get_ffn_output_proj(layer):
    return layer.output.dense


def precompute_col_norms(model, layers: List[int]) -> Dict[int, torch.Tensor]:
    blocks = get_transformer_layers(model)
    norms: Dict[int, torch.Tensor] = {}
    for i in layers:
        weight = get_ffn_output_proj(blocks[i]).weight.detach().float()
        norms[i] = torch.norm(weight, dim=0).cpu()
    return norms


def aa_token_ids(tokenizer) -> List[int]:
    return [tokenizer.convert_tokens_to_ids(aa) for aa in AA_ALPHABET]


@torch.inference_mode()
def extract_cett_batch(
    model,
    tokens: Dict[str, torch.Tensor],
    mask_positions: torch.Tensor,
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    aa_ids: List[int],
) -> Dict[str, torch.Tensor]:
    """One batched forward pass; CETT + amino-acid distribution at masked positions.

    mask_positions: (B, M) token indices (same device as model inputs).
    Returns flattened over B*M:
        cett      (B*M, n_layers * intermediate_dim)
        logits_aa (B*M, 20)
        probs     (B*M, 20)
    """
    blocks = get_transformer_layers(model)
    b = mask_positions.shape[0]
    b_idx = torch.arange(b, device=mask_positions.device).unsqueeze(1)
    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            z = inputs[0]
            h = output
            z_cache[idx] = z[b_idx, mask_positions].detach().float().cpu()
            h_cache[idx] = h[b_idx, mask_positions].detach().float().cpu()
            return output

        return hook

    for i in layers:
        handles.append(get_ffn_output_proj(blocks[i]).register_forward_hook(make_hook(i)))
    try:
        out = model(**tokens)
    finally:
        for handle in handles:
            handle.remove()

    cett_parts = []
    for i in layers:
        z = z_cache[i]
        h = h_cache[i]
        h_norm = h.norm(dim=-1, keepdim=True) + 1e-8
        cett_parts.append(torch.abs(z) * col_norms[i] / h_norm)
    cett = torch.cat(cett_parts, dim=-1)
    cett = cett.reshape(b * mask_positions.shape[1], -1)

    aa_ids_t = torch.tensor(aa_ids, device=out.logits.device)
    logits = out.logits[b_idx, mask_positions]
    logits = logits.reshape(b * mask_positions.shape[1], -1)
    logits_aa = logits[:, aa_ids_t].float().cpu()
    probs = torch.softmax(logits_aa, dim=-1)
    return {"cett": cett, "logits_aa": logits_aa, "probs": probs}
