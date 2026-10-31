import pytest
import torch
from _fake_dna_models import (
    DNA_TO_ID,
    FakeBpeTokenizer,
    FakeCarbon,
    FakeDnaBert,
    FakeDnaTokenizer,
    FakeKmerTokenizer,
)

from biointerp import dna, models
from biointerp.dna import reverse_complement
from biointerp.dna_task import (
    causal_dna_batch,
    dna_token_positions,
    masked_dna_batch,
    token_spans,
    variant_token_index,
)


def test_registry_dispatches_dna(monkeypatch, tmp_path):
    (tmp_path / "custom.yaml").write_text(
        "adapter: carbon\nhf_id: example/dna-model\nmodality: dna\ntokenizer: 6mer\n"
    )
    monkeypatch.setattr(models, "CONFIG_DIR", tmp_path)
    calls = {}

    def fake_load(spec, device=None):
        calls.update(spec=spec, device=device)
        return "model", "tokenizer", "cpu"

    monkeypatch.setattr(dna, "load_dna_model", fake_load)
    assert models.load_model("custom", device="cpu") == ("model", "tokenizer", "cpu")
    assert calls["spec"]["modality"] == "dna"
    assert calls["device"] == "cpu"


def test_layer_and_ffn_discovery():
    carbon = FakeCarbon(n_layers=3)
    bert = FakeDnaBert(n_layers=3)

    assert dna.get_layers(carbon, "carbon") is carbon.model.layers
    assert dna.get_layers(bert, "dnabert") is bert.bert.encoder.layer
    assert (
        dna.get_ffn_output_proj(carbon.model.layers[0], "carbon")
        is carbon.model.layers[0].mlp.down_proj
    )
    assert (
        dna.get_ffn_output_proj(bert.bert.encoder.layer[0], "dnabert")
        is bert.bert.encoder.layer[0].output.dense
    )


def test_unknown_adapter_raises():
    with pytest.raises(ValueError):
        dna.get_layers(object(), "bogus")


def test_col_norms_shapes():
    carbon = FakeCarbon(hidden=8, intermediate=12, n_layers=2)
    bert = FakeDnaBert(hidden=8, intermediate=12, n_layers=2)
    assert {k: tuple(v.shape) for k, v in dna.precompute_col_norms(carbon, "carbon").items()} == {
        0: (12,),
        1: (12,),
    }
    assert {k: tuple(v.shape) for k, v in dna.precompute_col_norms(bert, "dnabert").items()} == {
        0: (12,),
        1: (12,),
    }


def test_kmer_spans_and_variant_index():
    spans = token_spans("ACGTACGTAA", FakeKmerTokenizer(), "carbon")
    assert spans == [None, (0, 6), (6, 10), None]
    assert dna_token_positions(spans) == [1, 2]
    assert variant_token_index(spans, 0) == 1
    assert variant_token_index(spans, 7) == 2
    with pytest.raises(ValueError):
        variant_token_index(spans, 10)


def test_single_nucleotide_spans():
    spans = token_spans("ACGT", FakeDnaTokenizer(), "dnabert")
    assert spans == [None, (0, 1), (1, 2), (2, 3), (3, 4), None]


def test_bpe_spans_strip_continuation_prefix():
    spans = token_spans("ACGT", FakeBpeTokenizer(), "dnabert")
    assert spans == [None, (0, 1), (1, 2), (2, 3), (3, 4), None]


def test_masked_batch_alignment():
    tokenizer = FakeDnaTokenizer()
    sequences = ["ACGTAC", "ACGTAC"]
    tokens, positions, true_ids, spans = masked_dna_batch(
        sequences, tokenizer, "dnabert", masks_per_seq=1, seed=3
    )

    assert positions.shape == (2, 1)
    assert true_ids.shape == (2, 1)
    assert tokens["input_ids"].shape[0] == 2
    for i, sequence in enumerate(sequences):
        token_index = int(positions[i, 0])
        span = spans[i][token_index]
        assert tokens["input_ids"][i, token_index] == tokenizer.mask_token_id
        assert int(true_ids[i, 0]) == DNA_TO_ID[sequence[span[0]]]
        assert int(tokens["attention_mask"][i].sum()) == len(sequence) + 2


def test_causal_batch_targets_last_dna_token():
    tokenizer = FakeKmerTokenizer()
    tokens, positions, targets = causal_dna_batch(["ACGTACGTAA", "ACGTACGTAC"], tokenizer, "carbon")
    target_tokens = tokenizer.convert_ids_to_tokens(targets[:, 0].tolist())
    assert target_tokens == ["GTAA", "GTAC"]
    assert positions.shape == (2, 1)
    assert tokens["input_ids"].shape[0] == 2


def test_masked_cett_shapes_and_cleanup():
    model = FakeDnaBert(hidden=16, intermediate=24, n_layers=3).eval()
    tokenizer = FakeDnaTokenizer()
    tokens, positions, _, _ = masked_dna_batch(["ACGTAC", "ACGTAC"], tokenizer, "dnabert", seed=0)
    layers = [0, 2]
    norms = dna.precompute_col_norms(model, "dnabert", layers)
    result = dna.extract_cett_masked(model, tokens, positions, layers, norms, "dnabert")

    assert result["cett"].shape == (2, 2 * 24)
    assert result["logits"].shape == (2, 11)
    assert (result["cett"] >= 0).all()
    for layer in layers:
        assert len(model.bert.encoder.layer[layer].output.dense._forward_hooks) == 0


def test_causal_cett_shapes_and_cleanup():
    model = FakeCarbon(hidden=16, intermediate=24, n_layers=3).eval()
    tokenizer = FakeKmerTokenizer()
    tokens, positions, _ = causal_dna_batch(["ACGTACGTAA", "ACGTACGTAC"], tokenizer, "carbon")
    layers = [0, 2]
    norms = dna.precompute_col_norms(model, "carbon", layers)
    result = dna.extract_cett_causal(model, tokens, positions, layers, norms, "carbon")

    assert result["cett"].shape == (2, 2 * 24)
    assert result["logits"].shape == (2, 128)
    assert (result["cett"] >= 0).all()
    for layer in layers:
        assert len(model.model.layers[layer].mlp.down_proj._forward_hooks) == 0


def test_cett_batch_equals_single_rows():
    model = FakeDnaBert(hidden=8, intermediate=12, n_layers=2).eval()
    tokenizer = FakeDnaTokenizer()
    tokens, positions, _, _ = masked_dna_batch(["ACGTAC", "ACGTAC"], tokenizer, "dnabert", seed=4)
    layers = [0, 1]
    norms = dna.precompute_col_norms(model, "dnabert", layers)
    batched = dna.extract_cett_masked(model, tokens, positions, layers, norms, "dnabert")

    for i in range(2):
        single_tokens = {key: value[i : i + 1] for key, value in tokens.items()}
        single = dna.extract_cett_masked(
            model, single_tokens, positions[i : i + 1], layers, norms, "dnabert"
        )
        torch.testing.assert_close(batched["cett"][i : i + 1], single["cett"])
        torch.testing.assert_close(batched["logits"][i : i + 1], single["logits"])


def test_score_ref_alt_is_finite():
    model = FakeDnaBert().eval()
    tokens = {"input_ids": torch.tensor([[0, 6, 7, 3]])}
    score = dna.score_ref_alt(model, tokens, tokens, position=1, ref_id=6, alt_id=7)
    assert isinstance(score, float)
    assert torch.isfinite(torch.tensor(score))


def test_reverse_complement():
    assert reverse_complement("ACGTN") == "NACGT"
    assert reverse_complement("AACCGGTT") == "AACCGGTT"
