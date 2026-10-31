import sys

import torch

from biointerp.carbonx import (
    CONDITIONS,
    build_messages,
    context_token_positions,
    dna_segments,
    dna_token_positions,
    reduce_attention,
    variant_indices,
)
from biointerp.experiments.carbonx_context import parse_layers


class _Tok:
    def convert_ids_to_tokens(self, ids):
        return ["<dna>", "ACGTAC", "GT", "</dna>", "<dna>", "ACGTAC", "GA", "</dna>"]


def test_build_messages_conditions():
    dna_only = build_messages("ACGT", "ACGA", "dna_only")[0]["content"]
    assert "<dna>ACGT</dna>" in dna_only
    assert "Gene:" not in dna_only

    correct = build_messages("ACGT", "ACGA", "correct_gene")[0]["content"]
    assert "Gene: BRCA1" in correct
    assert "<dna>" in correct

    wrong = build_messages("ACGT", "ACGA", "wrong_gene")[0]["content"]
    assert "Gene: TP53" in wrong

    irrelevant = build_messages("ACGT", "ACGA", "irrelevant_text")[0]["content"]
    assert "weather" in irrelevant
    assert "<dna>" in irrelevant

    english = build_messages("ACGT", "ACGA", "english_only")[0]["content"]
    assert "BRCA1" in english
    assert "<dna>" not in english

    assert set(CONDITIONS) == {
        "dna_only",
        "correct_gene",
        "wrong_gene",
        "irrelevant_text",
        "english_only",
    }


def test_parse_layers():
    assert parse_layers("all", 40) == list(range(40))
    assert parse_layers("last:2", 40) == [38, 39]
    assert parse_layers("0,39", 40) == [0, 39]
    assert parse_layers("last:50", 40) == list(range(40))
    for value in ("", "40", "-1", "last:0"):
        try:
            parse_layers(value, 40)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{value!r} should be rejected")


def test_dna_segments_and_variant_indices():
    segments = dna_segments(_Tok(), list(range(8)))
    assert len(segments) == 2
    assert segments[0]["spans"] == [(0, 6), (6, 8)]
    assert segments[1]["spans"] == [(0, 6), (6, 8)]
    assert variant_indices(segments, 7) == (1, 1)
    assert variant_indices(segments, 2) == (0, 0)


def test_dna_and_context_positions():
    class Tok:
        all_special_ids = [0, 8]

        def convert_ids_to_tokens(self, ids):
            return [
                "<s>",
                "Gene:",
                "BRCA1",
                "<dna>",
                "ACGTAC",
                "</dna>",
                "Is",
                "?",
                "</s>",
            ]

    segments = [{"start_token": 4, "end_token": 5}]
    assert dna_token_positions(segments) == [4]
    assert context_token_positions(Tok(), list(range(9)), segments) == [1, 2, 6, 7]


def test_reduce_attention_averages_queries():
    attention = torch.zeros(1, 2, 5, 4)
    attention[:, :, 1, 0] = 1.0
    attention[:, :, 3, 0] = 3.0
    attention[:, :, 1, 2] = 2.0
    attention[:, :, 3, 2] = 4.0
    reduced = reduce_attention([attention], [1, 3], [0, 2])
    assert reduced.shape == (1, 2, 2)
    assert reduced[0, 0, 0] == 2.0
    assert reduced[0, 0, 1] == 3.0


def test_dry_run_prints_all_conditions(monkeypatch, capsys):
    from biointerp.experiments import carbonx_context

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "carbonx_context",
            "--ref",
            "ACGTACGT",
            "--alt",
            "ACGTACGA",
            "--condition",
            "all",
            "--dry-run",
        ],
    )
    carbonx_context.main()
    out = capsys.readouterr().out
    for condition in CONDITIONS:
        assert f"=== {condition} ===" in out
    assert "<dna>ACGTACGT</dna>" in out
