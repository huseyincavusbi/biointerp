"""Deterministic CPU-only stand-ins for the protein language models.

These are deliberately tiny and position-wise so that numerical-equivalence tests
can compare batched and per-sample code paths exactly. They mimic only the
structural surface the adapters touch: ESM-2 ``esm.encoder.layer[i].output.dense``,
ESMC ``esmc.transformer.blocks[i].ffn`` and GPT-2 ``transformer.h[i].mlp.c_proj``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIALS = ["<cls>", "<pad>", "<mask>", "<eos>"]
AA_TO_ID = {aa: len(SPECIALS) + i for i, aa in enumerate(AA)}
ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}
CLS_ID, PAD_ID, MASK_ID, EOS_ID = 0, 1, 2, 3
VOCAB = len(SPECIALS) + len(AA)


class FakeEsmTokenizer:
    """Character-level tokenizer with ESM special tokens."""

    mask_token_id = MASK_ID
    pad_token_id = PAD_ID
    eos_token_id = EOS_ID
    cls_token_id = CLS_ID

    def __call__(self, sequence: str, add_special_tokens: bool = True) -> dict:
        ids = [AA_TO_ID[c] for c in sequence]
        if add_special_tokens:
            ids = [CLS_ID, *ids, EOS_ID]
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in AA_TO_ID:
            return AA_TO_ID[token]
        return {"<cls>": CLS_ID, "<pad>": PAD_ID, "<mask>": MASK_ID, "<eos>": EOS_ID}[token]


class _FakeEsmLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.intermediate = nn.Linear(hidden, intermediate)
        self.output = nn.Module()
        self.output.dense = nn.Linear(intermediate, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        z = F.gelu(self.intermediate(x))
        return self.norm(x + self.output.dense(z))


class FakeEsmModel(nn.Module):
    """Minimal ``EsmForMaskedLM`` look-alike.

    The lm-head gets a deterministic position bias (A/C in two-position blocks), so
    a repeated ``ACAC...`` sequence yields a balanced mix of correct/incorrect
    predictions without relying on random weights.
    """

    def __init__(
        self,
        hidden: int = 16,
        intermediate: int = 24,
        n_layers: int = 4,
        seed: int = 0,
        bias_scale: float = 8.0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, hidden)
        self.esm = nn.Module()
        self.esm.encoder = nn.Module()
        self.esm.encoder.layer = nn.ModuleList(
            [_FakeEsmLayer(hidden, intermediate) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(hidden, VOCAB, bias=False)
        self.bias_scale = bias_scale
        self.hidden_size = hidden
        self.intermediate_size = intermediate

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        x = self.embed(input_ids)
        hidden_states = [x]
        for layer in self.esm.encoder.layer:
            x = layer(x)
            hidden_states.append(x)
        logits = self.lm_head(x)
        if self.bias_scale:
            for j in range(1, input_ids.shape[1]):
                aa = "A" if ((j - 1) // 2) % 2 == 0 else "C"
                logits[:, j, AA_TO_ID[aa]] += self.bias_scale
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states if output_hidden_states else None,
        )


class FakeEsmcTokenizer:
    """Character tokenizer that understands a literal ``<mask>`` token."""

    mask_token_id = MASK_ID
    pad_token_id = PAD_ID

    def __call__(self, text: str, return_tensors: str | None = None) -> dict:
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith("<mask>", i):
                ids.append(MASK_ID)
                i += len("<mask>")
            else:
                ids.append(AA_TO_ID[text[i]])
                i += 1
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


class FakeEsmcFfn(nn.Module):
    """SwiGLU FFN with the attribute surface ``esmc.extract_cett_sequence`` reads."""

    def __init__(self, hidden: int = 16, intermediate: int = 24):
        super().__init__()
        self.hidden_size = hidden
        self.ffn_hidden_size = intermediate
        self.eps = 1e-5
        self.layer_norm_weight = nn.Parameter(torch.ones(hidden))
        self.layer_norm_bias = nn.Parameter(torch.zeros(hidden))
        self.fc1_weight = nn.Parameter(torch.randn(2 * intermediate, hidden) * 0.1)
        self.fc2_weight = nn.Parameter(torch.randn(hidden, intermediate) * 0.1)

    def forward(self, x):
        xn = F.layer_norm(
            x,
            (self.hidden_size,),
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.eps,
        )
        x12 = F.linear(xn, self.fc1_weight)
        z = F.silu(x12[..., : self.ffn_hidden_size]) * x12[..., self.ffn_hidden_size :]
        return F.linear(z, self.fc2_weight)


class _FakeEsmcBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.ffn = FakeEsmcFfn(hidden, intermediate)

    def forward(self, x):
        return x + self.ffn(x)


class FakeEsmcModel(nn.Module):
    """Minimal ESMC look-alike with ``esmc.transformer.blocks``."""

    def __init__(self, hidden: int = 16, intermediate: int = 24, n_layers: int = 3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, hidden)
        self.esmc = nn.Module()
        self.esmc.transformer = nn.Module()
        self.esmc.transformer.blocks = nn.ModuleList(
            [_FakeEsmcBlock(hidden, intermediate) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(hidden, VOCAB, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        for block in self.esmc.transformer.blocks:
            x = block(x)
        return SimpleNamespace(logits=self.lm_head(x))


class _FakeProj(nn.Module):
    """Linear projection shaped like GPT-2's ``Conv1D`` (out, in)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class _FakeGpt2Layer(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.fc_in = nn.Linear(hidden, intermediate)
        self.mlp.c_proj = _FakeProj(intermediate, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        return self.norm(x + self.mlp.c_proj(F.gelu(self.mlp.fc_in(x))))


class FakeGpt2Tokenizer:
    eos_token_id = EOS_ID
    bos_token_id = 0
    pad_token_id = PAD_ID

    def decode(self, ids) -> str:
        return "".join(ID_TO_AA.get(int(i), "") for i in ids)


class FakeGpt2Model(nn.Module):
    """Minimal GPT-2 look-alike for ``biointerp.gen.generate_with_cett``."""

    def __init__(
        self,
        hidden: int = 12,
        intermediate: int = 20,
        n_layers: int = 2,
        seed: int = 0,
        eos_bias: float = -100.0,
        special_bias: float = -100.0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, hidden)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(
            [_FakeGpt2Layer(hidden, intermediate) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(hidden, VOCAB)
        self.eos_bias = eos_bias
        self.special_bias = special_bias
        self.config = SimpleNamespace(n_embd=hidden, n_layer=n_layers)

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        x = self.embed(input_ids)
        for layer in self.transformer.h:
            x = layer(x)
        logits = self.lm_head(x)
        logits[..., :EOS_ID] += self.special_bias
        logits[..., EOS_ID] += self.eos_bias
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class LayerOnly(nn.Module):
    """Standalone layer used to exercise FFN-projection discovery."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp
