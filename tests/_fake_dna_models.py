"""Tiny CPU-only DNA model/tokenizer stand-ins for unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

DNA = "ACGTN"
SPECIALS = ["<pad>", "<dna>", "</dna>", "<mask>", "<cls>", "<sep>"]
PAD, DNA_OPEN, DNA_CLOSE, MASK, CLS, SEP = range(6)
DNA_TO_ID = {base: len(SPECIALS) + i for i, base in enumerate(DNA)}


class FakeDnaTokenizer:
    """Character-level DNA tokenizer with BERT-style specials."""

    mask_token_id = MASK
    pad_token_id = PAD
    cls_token_id = CLS
    sep_token_id = SEP

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        ids = [DNA_TO_ID[char] for char in text]
        if add_special_tokens:
            ids = [CLS, *ids, SEP]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids) -> list[str]:
        reverse = {v: k for k, v in DNA_TO_ID.items()}
        reverse.update(
            {
                PAD: "<pad>",
                DNA_OPEN: "<dna>",
                DNA_CLOSE: "</dna>",
                MASK: "<mask>",
                CLS: "<cls>",
                SEP: "<sep>",
            }
        )
        return [reverse[int(i)] for i in ids]

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in DNA_TO_ID:
            return DNA_TO_ID[token]
        return {
            "<pad>": PAD,
            "<dna>": DNA_OPEN,
            "</dna>": DNA_CLOSE,
            "<mask>": MASK,
            "<cls>": CLS,
            "<sep>": SEP,
        }[token]


class FakeBpeTokenizer(FakeDnaTokenizer):
    """Fixed tokenization of ``ACGT`` into A + ##C + ##G + ##T."""

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        if text != "ACGT":
            raise ValueError("FakeBpeTokenizer only supports ACGT")
        return {"input_ids": [CLS, 6, 7, 8, 9, SEP]}

    def convert_ids_to_tokens(self, ids) -> list[str]:
        mapping = {CLS: "<cls>", SEP: "<sep>", 6: "A", 7: "##C", 8: "##G", 9: "##T"}
        return [mapping[int(i)] for i in ids]


class FakeKmerTokenizer:
    """Dynamic non-overlapping 6-mer tokenizer with <dna> markers."""

    mask_token_id = MASK
    pad_token_id = PAD
    eos_token_id = SEP

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self._token_to_id = {"<pad>": PAD, "<dna>": DNA_OPEN, "</dna>": DNA_CLOSE, "<mask>": MASK}
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        self._next_id = len(SPECIALS)

    def _id(self, token: str) -> int:
        if token not in self._token_to_id:
            if self._next_id >= self.vocab_size:
                raise ValueError("fake k-mer vocabulary exhausted")
            self._token_to_id[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1
        return self._token_to_id[token]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        if "<dna>" in text and "</dna>" in text:
            start = text.index("<dna>") + len("<dna>")
            end = text.index("</dna>", start)
            sequence = text[start:end]
            ids = [DNA_OPEN]
            ids.extend(self._id(sequence[i : i + 6]) for i in range(0, len(sequence), 6))
            ids.append(DNA_CLOSE)
            return {"input_ids": ids}
        return {"input_ids": [self._id(text[i : i + 6]) for i in range(0, len(text), 6)]}

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [self._id_to_token.get(int(i), f"<unk:{int(i)}>") for i in ids]


class FakePaddedKmerTokenizer(FakeKmerTokenizer):
    """Six-mer tokenizer that pads a short tail into a full token."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        if "<dna>" not in text or "</dna>" not in text:
            return super().__call__(text, add_special_tokens)
        start = text.index("<dna>") + len("<dna>")
        end = text.index("</dna>", start)
        sequence = text[start:end]
        ids = [DNA_OPEN]
        for i in range(0, len(sequence), 6):
            chunk = sequence[i : i + 6]
            ids.append(self._id(chunk.ljust(6, "A")))
        ids.append(DNA_CLOSE)
        return {"input_ids": ids}


class _FakeBertLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.intermediate = nn.Linear(hidden, intermediate)
        self.output = nn.Module()
        self.output.dense = nn.Linear(intermediate, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        z = F.gelu(self.intermediate(x))
        return self.norm(x + self.output.dense(z))


class FakeDnaBert(nn.Module):
    def __init__(self, hidden: int = 16, intermediate: int = 24, n_layers: int = 3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(len(SPECIALS) + len(DNA), hidden)
        self.bert = nn.Module()
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList(
            [_FakeBertLayer(hidden, intermediate) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(hidden, len(SPECIALS) + len(DNA))

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.embed(input_ids)
        hidden_states = [x]
        for layer in self.bert.encoder.layer:
            x = layer(x)
            hidden_states.append(x)
        return SimpleNamespace(
            logits=self.lm_head(x),
            hidden_states=hidden_states if output_hidden_states else None,
        )


class _FakeLlamaLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        z = F.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        return self.norm(x + self.mlp.down_proj(z))


class FakeCarbon(nn.Module):
    def __init__(
        self,
        hidden: int = 16,
        intermediate: int = 24,
        n_layers: int = 3,
        seed: int = 0,
        vocab: int = 128,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, hidden)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_FakeLlamaLayer(hidden, intermediate) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.embed(input_ids)
        hidden_states = [x]
        for layer in self.model.layers:
            x = layer(x)
            hidden_states.append(x)
        return SimpleNamespace(
            logits=self.lm_head(x),
            hidden_states=hidden_states if output_hidden_states else None,
        )
