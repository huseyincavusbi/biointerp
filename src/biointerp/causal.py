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


def get_attn_out_proj(layer, adapter: str):
    """Return the projection that consumes concatenated attention heads."""
    if adapter == "carbon":
        attention = getattr(layer, "self_attn", None)
        projection = getattr(attention, "o_proj", None) if attention is not None else None
        if projection is not None:
            return projection
    raise ValueError(f"cannot find attention output projection for adapter {adapter!r}")


@contextmanager
def ablate_heads_context(model, layer: int, heads, adapter: str):
    """Zero selected attention-head slices at a layer."""
    blocks = get_layers(model, adapter)
    projection = get_attn_out_proj(blocks[layer], adapter)
    config = model.config
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    heads = sorted({int(head) for head in heads})
    for head in heads:
        if head < 0 or head >= config.num_attention_heads:
            raise ValueError(f"head {head} outside [0, {config.num_attention_heads})")

    def hook(module, args):
        hidden = args[0].clone()
        for head in heads:
            start = head * head_dim
            hidden[..., start : start + head_dim] = 0
        return (hidden, *args[1:])

    handle = projection.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def ablate_heads(model, tokens: dict, layer: int, heads, adapter: str):
    """Run a forward pass with selected attention heads removed."""
    with ablate_heads_context(model, layer, heads, adapter):
        return model(**tokens)
