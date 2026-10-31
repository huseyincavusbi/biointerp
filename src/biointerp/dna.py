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
        mlp = getattr(layer, "mlp", None)
        proj = getattr(mlp, "wo", None) if mlp is not None else None
        if proj is not None:
            return proj
        output = getattr(layer, "output", None)
        proj = getattr(output, "dense", None) if output is not None else None
        if proj is not None:
            return proj
    raise ValueError(f"cannot find FFN output projection for adapter {adapter!r}")


def dequantize_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return a dense float copy of a weight, including bitsandbytes 4-bit."""
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None:
        return weight.detach().float()
    from bitsandbytes.functional import dequantize_4bit

    # Params4bit.dequantize() can return the packed tensor on some ROCm
    # builds; the functional path uses the stored quant state directly.
    return dequantize_4bit(weight.data, quant_state).detach().float()


def precompute_col_norms(
    model, adapter: str, layers: Optional[List[int]] = None
) -> Dict[int, torch.Tensor]:
    blocks = get_layers(model, adapter)
    indices = list(range(len(blocks))) if layers is None else layers
    norms: Dict[int, torch.Tensor] = {}
    for i in indices:
        module = get_ffn_output_proj(blocks[i], adapter)
        weight = dequantize_weight(module.weight)
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
        kwargs = {"trust_remote_code": True}
        quantized = bool(spec.get("load_in_4bit"))
        if quantized:
            if not dev.startswith("cuda"):
                raise ValueError("4-bit loading requires a CUDA/ROCm device")
            from transformers import BitsAndBytesConfig

            compute = spec.get("compute_dtype", spec.get("dtype", "bf16"))
            compute_dtype = torch.bfloat16 if compute in ("bf16", "bfloat16") else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": dev}
            kwargs["low_cpu_mem_usage"] = True
        elif spec.get("dtype") in ("bf16", "bfloat16"):
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
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
    if not (adapter == "carbon" and spec.get("load_in_4bit")):
        model.to(dev)
    if adapter == "dnabert":
        # DNABERT-2 selects its Triton kernel when attention dropout is zero,
        # including on CPU/MPS where the kernel cannot run. A nonzero p_dropout
        # keeps inference on the standard PyTorch path (dropout is inactive in
        # eval mode), making CPU/MPS and CUDA behavior consistent.
        for block in get_layers(model, "dnabert"):
            attention = getattr(block, "attention", None)
            self_attention = getattr(attention, "self", None)
            if hasattr(self_attention, "p_dropout"):
                self_attention.p_dropout = max(float(self_attention.p_dropout), 1e-6)
    return model, tokenizer, dev


def _gather_token_states(
    tensor: torch.Tensor, positions: torch.Tensor, attention_mask: torch.Tensor
):
    """Gather token states from padded ``(B, L, D)`` or packed ``(N, D)`` tensors."""
    if tensor.dim() == 3:
        batch = positions.shape[0]
        b_idx = torch.arange(batch).unsqueeze(1)
        return tensor[b_idx, positions]
    lengths = attention_mask.sum(dim=1).tolist()
    offsets = []
    total = 0
    for length in lengths:
        offsets.append(total)
        total += int(length)
    flat = [offsets[b] + p for b in range(positions.shape[0]) for p in positions[b].tolist()]
    return tensor[torch.tensor(flat, dtype=torch.long)]


@torch.inference_mode()
def extract_cett(
    model,
    tokens: dict,
    positions: torch.Tensor,
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    adapter: str,
    capture_hidden: bool = False,
):
    """Run one forward pass and return CETT + logits at the given token positions."""
    blocks = get_layers(model, adapter)
    positions = positions.to(tokens["input_ids"].device)
    batch = positions.shape[0]
    b_idx = torch.arange(batch, device=positions.device).unsqueeze(1)
    positions_cpu = positions.cpu()
    attention_mask = tokens.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(tokens["input_ids"])
    attention_cpu = attention_mask.cpu()

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    hidden_cache: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            z_cache[idx] = inputs[0].detach().float().cpu()
            h_cache[idx] = output.detach().float().cpu()
            return output

        return hook

    def make_hidden_hook(idx: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_cache[idx] = hidden.detach().float().cpu()
            return output

        return hook

    for i in layers:
        handles.append(get_ffn_output_proj(blocks[i], adapter).register_forward_hook(make_hook(i)))
    if capture_hidden:
        for i in range(len(blocks)):
            handles.append(blocks[i].register_forward_hook(make_hidden_hook(i)))
    try:
        out = model(**tokens)
    finally:
        for handle in handles:
            handle.remove()

    cett_parts = []
    for i in layers:
        z = _gather_token_states(z_cache[i], positions_cpu, attention_cpu).reshape(
            batch * positions.shape[1], -1
        )
        h = _gather_token_states(h_cache[i], positions_cpu, attention_cpu).reshape(
            batch * positions.shape[1], -1
        )
        cett_parts.append(torch.abs(z) * col_norms[i] / (h.norm(dim=-1, keepdim=True) + 1e-8))
    cett = torch.cat(cett_parts, dim=-1)

    logits = out.logits[b_idx, positions]
    result = {
        "cett": cett,
        "logits": logits.reshape(batch * positions.shape[1], -1).float().cpu(),
    }
    if capture_hidden:
        result["hidden_states"] = [
            _gather_token_states(hidden_cache[i], positions_cpu, attention_cpu).reshape(
                batch * positions.shape[1], -1
            )
            for i in range(len(blocks))
        ]
    return result


def extract_cett_masked(model, tokens, positions, layers, col_norms, adapter, capture_hidden=False):
    return extract_cett(model, tokens, positions, layers, col_norms, adapter, capture_hidden)


def extract_cett_causal(model, tokens, positions, layers, col_norms, adapter, capture_hidden=False):
    return extract_cett(model, tokens, positions, layers, col_norms, adapter, capture_hidden)


@torch.inference_mode()
def score_ref_alt(model, tokens_ref, tokens_alt, position: int, ref_id: int, alt_id: int) -> float:
    """Return log p(alt) - log p(ref) at one token position."""
    ref_logits = model(**tokens_ref).logits[0, position].float()
    alt_logits = model(**tokens_alt).logits[0, position].float()
    ref_logp = torch.log_softmax(ref_logits, dim=-1)
    alt_logp = torch.log_softmax(alt_logits, dim=-1)
    return float(alt_logp[alt_id] - ref_logp[ref_id])
