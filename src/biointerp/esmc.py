"""ESMC adapter for CETT / error-neuron extraction (cross-model experiment).

Uses the HuggingFace path from the model card (AutoTokenizer + AutoModelForMaskedLM with
trust_remote_code). Masking = insert ``<mask>`` into the sequence string.

CETT follows the same formula as the ESM-2 adapter: hook the FFN module, recover the
SwiGLU activation z and the FFN output h,
    CETT_j = |z_j| * ||W_out[:, j]|| / (||h|| + eps)
with W_out = ffn.fc2_weight.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
ESMC_AA_IDS = {
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
}
ESMC_MASK_STR = "<mask>"


def pick_device(device=None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_esmc(model_id: str = "biohub/ESMC-600M", device=None, dtype=None, device_map=None):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if dtype in ("bf16", "bfloat16"):
        kwargs["dtype"] = torch.bfloat16
    if device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForMaskedLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if device_map:
        dev = next(model.parameters()).device
        if dev.type == "meta":
            dev = torch.device("cuda:0")
        return model, tokenizer, str(dev)
    dev = pick_device(device)
    model.to(dev)
    return model, tokenizer, dev


def get_blocks(model):
    return model.esmc.transformer.blocks


def precompute_col_norms(model) -> Dict[int, torch.Tensor]:
    blocks = get_blocks(model)
    norms = {}
    for i, block in enumerate(blocks):
        weight = block.ffn.fc2_weight.detach().float().cpu()
        norms[i] = torch.norm(weight, dim=0)
    return norms


def aa_token_ids() -> List[int]:
    return [ESMC_AA_IDS[aa] for aa in AA_ALPHABET]


@torch.inference_mode()
def extract_cett_sequence(
    model,
    tokenizer,
    sequence: str,
    mask_positions: List[int],
    col_norms: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One sequence with one or more masked positions (0-based residue indices).

    Returns cett (n_masks, n_layers * intermediate), probs (n_masks, 20).
    """
    import torch.nn.functional as F

    blocks = get_blocks(model)
    masked = list(sequence)
    for p in mask_positions:
        masked[p] = ESMC_MASK_STR
    inputs = tokenizer("".join(masked), return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    ids = inputs["input_ids"][0]
    mask_token_id = tokenizer.mask_token_id
    token_positions = [int(i) for i in (ids == mask_token_id).nonzero(as_tuple=True)[0].tolist()]

    x_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(i: int):
        def hook(module, inputs_, output):
            x_cache[i] = inputs_[0][0].detach().float()
            h_cache[i] = output[0].detach().float()
            return output

        return hook

    for i, block in enumerate(blocks):
        handles.append(block.ffn.register_forward_hook(make_hook(i)))
    try:
        out = model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    logits = out.logits[0]
    aa_ids = torch.tensor(aa_token_ids(), device=logits.device)
    probs = torch.softmax(logits[token_positions][:, aa_ids].float(), dim=-1).cpu()

    cetts = []
    for pos in token_positions:
        parts = []
        for i, block in enumerate(blocks):
            ffn = block.ffn
            x = x_cache[i][pos]
            xn = F.layer_norm(
                x,
                (ffn.hidden_size,),
                ffn.layer_norm_weight.float(),
                ffn.layer_norm_bias.float(),
                ffn.eps,
            )
            x12 = F.linear(xn, ffn.fc1_weight.float())
            z = F.silu(x12[: ffn.ffn_hidden_size]) * x12[ffn.ffn_hidden_size :]
            h_norm = float(h_cache[i][pos].norm()) + 1e-8
            parts.append((z.abs().cpu() * col_norms[i]) / h_norm)
        cetts.append(torch.cat(parts))
    return torch.stack(cetts), probs
