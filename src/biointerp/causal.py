"""Causal interventions for DNA language models."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .dna import _gather_token_states, get_layers


def _positions(tokens, positions):
    positions = positions.to(tokens["input_ids"].device)
    attention_mask = tokens.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(tokens["input_ids"])
    return positions, attention_mask


@torch.inference_mode()
def hidden_states_at(model, tokens: dict, positions: torch.Tensor, layers, adapter: str):
    """Capture block outputs at selected token positions."""
    blocks = get_layers(model, adapter)
    positions, attention_mask = _positions(tokens, positions)
    cache = {}
    handles = []

    def make_hook(index):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            cache[index] = _gather_token_states(hidden.detach(), positions, attention_mask).float()
            return output

        return hook

    for index in layers:
        handles.append(blocks[index].register_forward_hook(make_hook(index)))
    try:
        model(**tokens)
    finally:
        for handle in handles:
            handle.remove()
    return cache


@contextmanager
def patch_hidden_context(
    model,
    tokens: dict,
    layer: int,
    positions: torch.Tensor,
    replacement: torch.Tensor,
    adapter: str,
):
    """Temporarily replace hidden states at selected positions."""
    blocks = get_layers(model, adapter)
    positions = positions.to(tokens["input_ids"].device)
    replacement = replacement.to(tokens["input_ids"].device)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() != 3:
            raise ValueError("patch_hidden requires padded (B, L, D) hidden states")
        if hidden.shape[1] <= int(positions.max()):
            return output
        patched = hidden.clone()
        batch = positions.shape[0]
        batch_index = torch.arange(batch, device=patched.device).unsqueeze(1)
        patched[batch_index, positions] = replacement.to(patched.dtype)
        return (patched, *output[1:]) if isinstance(output, tuple) else patched

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def patch_hidden(
    model,
    tokens: dict,
    layer: int,
    positions: torch.Tensor,
    replacement: torch.Tensor,
    adapter: str,
):
    """Run a forward pass with donor hidden states at selected positions."""
    with patch_hidden_context(model, tokens, layer, positions, replacement, adapter):
        return model(**tokens)
