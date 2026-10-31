from types import SimpleNamespace

import torch
import torch.nn as nn

from biointerp.carbonx import (
    FORCED_CHOICE,
    build_messages,
    continuation_logprob,
    pathogenic_probability,
)


class FixedModel(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros(1, input_ids.shape[1], 10)
        logits[..., 2] = 5.0
        logits[..., 3] = 1.0
        return SimpleNamespace(logits=logits)


class FixedTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [2] if "pathogenic" in text else [3]


def test_forced_choice_prompt_is_not_duplicated():
    content = build_messages("ACGT", "ACGA", "dna_only", question=FORCED_CHOICE)[0]["content"]
    assert content.count(FORCED_CHOICE) == 1


def test_continuation_logprob_prefers_high_logit():
    model = FixedModel()
    tokens = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    high = continuation_logprob(model, tokens, [2])
    low = continuation_logprob(model, tokens, [3])
    assert high > low
    assert high <= 0


def test_pathogenic_probability_in_range():
    model = FixedModel()
    tokenizer = FixedTokenizer()
    tokens = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    result = pathogenic_probability(model, tokenizer, tokens)
    assert 0.0 <= result["p_pathogenic"] <= 1.0
    assert result["p_pathogenic"] > 0.5
    assert result["logp_pathogenic"] > result["logp_benign"]
