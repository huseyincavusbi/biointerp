"""Generative PLM adapter: ProtGPT2 with generation-time CETT extraction.

Adapts the hprobes CETT methodology to autoregressive protein generation:
    hook the FFN output projection (c_proj), capture z (input) and h (output),
    compute CETT_j = |z_j| * ||W_out[:, j]|| / (||h|| + eps) at the current token
    position of every generation step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
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


def _install_multimolecule_stub() -> None:
    """Bypass multimolecule's heavy __init__ (training runner deps) to import model classes."""
    import sys
    import types

    import transformers

    sp = Path(transformers.__file__).parent.parent
    if "multimolecule" not in sys.modules:
        pkg = types.ModuleType("multimolecule")
        pkg.__path__ = [str(sp / "multimolecule")]
        sys.modules["multimolecule"] = pkg
    if "multimolecule.models" not in sys.modules:
        models_pkg = types.ModuleType("multimolecule.models")
        models_pkg.__path__ = [str(sp / "multimolecule" / "models")]
        sys.modules["multimolecule.models"] = models_pkg


def load_generative(model_name: str = "nferruz/ProtGPT2", device: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name.startswith("multimolecule/progen2"):
        _install_multimolecule_stub()
        from multimolecule.models.progen2 import ProGen2ForCausalLM, ProteinTokenizer

        tokenizer = ProteinTokenizer.from_pretrained(model_name)
        model = ProGen2ForCausalLM.from_pretrained(model_name)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except (ValueError, OSError):
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    _ensure_head_mask(model)
    dev = pick_device(device)
    model.to(dev)
    return model, tokenizer, dev


def _ensure_head_mask(model) -> None:
    """Compat shim: transformers 5.x removed PreTrainedModel.get_head_mask, which older
    remote-code checkpoints (e.g. ProGen2) still call."""
    target = getattr(model, "transformer", model)
    cls = type(target)
    if not hasattr(cls, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            return head_mask

        cls.get_head_mask = get_head_mask


def get_transformer_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(
        "Unsupported generative architecture: no transformer.h / decoder.layers / model.layers"
    )


def get_ffn_output_proj(layer):
    for name in ("c_proj", "down_proj", "fc2", "o_proj", "fc_out"):
        if hasattr(layer.mlp, name):
            return getattr(layer.mlp, name)
    raise ValueError("No FFN output projection found on layer.mlp")


def precompute_col_norms(model, layers: List[int]) -> Dict[int, torch.Tensor]:
    blocks = get_transformer_layers(model)
    norms = {}
    for i in layers:
        module = get_ffn_output_proj(blocks[i])
        weight = module.weight.detach().float()
        if type(module).__name__ == "Conv1D":
            norms[i] = torch.norm(weight, dim=1).cpu()
        else:
            norms[i] = torch.norm(weight, dim=0).cpu()
    return norms


@torch.inference_mode()
def generate_with_cett(
    model,
    tokenizer,
    n_steps: int = 100,
    seed: int = 0,
    temperature: float = 1.0,
    greedy: bool = True,
    device: Optional[str] = None,
    steer_map: Optional[Dict[int, Tuple[np.ndarray, float]]] = None,
    capture_layers: Optional[List[int]] = None,
):
    """Greedy/sampled generation with per-step CETT at the emitted position.

    Returns token ids, per-step CETT vectors (n_layers * intermediate), and the
    decoded text of each emitted token.
    """
    dev = device or pick_device()
    blocks = get_transformer_layers(model)
    n_layers = len(blocks)
    col_norms = precompute_col_norms(model, list(range(n_layers)))

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    hidden_cache: Dict[int, list] = {l: [] for l in (capture_layers or [])}
    handles = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            z = inputs[0]
            h = output
            z_cache[idx] = z[0, -1, :].detach().float().cpu()
            h_cache[idx] = h[0, -1, :].detach().float().cpu()
            return output

        return hook

    for i in range(n_layers):
        handles.append(get_ffn_output_proj(blocks[i]).register_forward_hook(make_hook(i)))

    def make_capture_hook(layer: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_cache[layer].append(h[0, -1, :].detach().float().cpu())
            return output

        return hook

    for layer in capture_layers or []:
        handles.append(blocks[layer].register_forward_hook(make_capture_hook(layer)))

    def make_steer_hook(layer: int, direction, alpha: float):
        d = torch.tensor(np.asarray(direction), dtype=torch.float32)

        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            h = h.clone()
            h[0, -1, :] = h[0, -1, :] + alpha * d.to(h.device)
            return (h,) + output[1:] if is_tuple else h

        return hook

    for layer, (direction, alpha) in (steer_map or {}).items():
        handles.append(
            blocks[layer].register_forward_hook(make_steer_hook(layer, direction, alpha))
        )

    torch.manual_seed(seed)
    start_id = (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.bos_token_id
    )
    tokens = torch.tensor([[start_id]], device=dev)
    past = None
    step_cetts: List[torch.Tensor] = []
    step_texts: List[str] = []
    step_ids: List[int] = []
    step_confs: List[float] = []
    step_ents: List[float] = []

    try:
        for _ in range(n_steps):
            input_ids = tokens if past is None else tokens[:, -1:]
            out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :]
            if greedy:
                next_id = int(logits.argmax().item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            if next_id == tokenizer.eos_token_id:
                break
            probs = torch.softmax(logits.float(), dim=-1)
            step_confs.append(float(probs.max().item()))
            step_ents.append(float(-(probs * (probs + 1e-12).log()).sum().item()))
            cett = torch.cat(
                [
                    torch.abs(z_cache[i]) * col_norms[i] / (float(h_cache[i].norm()) + 1e-8)
                    for i in range(n_layers)
                ]
            )
            step_cetts.append(cett)
            step_ids.append(next_id)
            step_texts.append(tokenizer.decode([next_id]))
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=dev)], dim=1)
    finally:
        for handle in handles:
            handle.remove()

    if capture_layers:
        return (
            step_ids,
            step_cetts,
            step_texts,
            step_confs,
            step_ents,
            {l: hidden_cache[l] for l in capture_layers},
        )
    return step_ids, step_cetts, step_texts, step_confs, step_ents
