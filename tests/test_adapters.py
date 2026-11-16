import pytest
from _fake_models import AA, FakeEsmcModel, FakeEsmcTokenizer, FakeEsmModel, FakeEsmTokenizer

from biointerp.adapters import Esm2Adapter, EsmcAdapter, build_adapter


def test_esm2_adapter_extracts_and_decodes():
    model = FakeEsmModel(hidden=8, intermediate=12, n_layers=2).eval()
    tokenizer = FakeEsmTokenizer()
    adapter = Esm2Adapter("fake", model=model, tokenizer=tokenizer, device="cpu")

    features, probabilities = adapter.extract("ACDEFGHIKL", 2)

    assert features.shape == (2 * 12,)
    assert probabilities.shape == (20,)
    assert adapter.decode(probabilities) in AA


def test_esmc_adapter_extracts_and_decodes():
    model = FakeEsmcModel(hidden=8, intermediate=12, n_layers=3).eval()
    tokenizer = FakeEsmcTokenizer()
    adapter = EsmcAdapter("fake", model=model, tokenizer=tokenizer, device="cpu")

    features, probabilities = adapter.extract("ACDEFGHIKL", 2)

    assert features.shape == (3 * 12,)
    assert probabilities.shape == (20,)
    assert adapter.decode(probabilities) in AA


def test_build_adapter_rejects_unknown_key():
    with pytest.raises(KeyError):
        build_adapter("not-a-model")


def test_build_adapter_rejects_missing_modality_adapter():
    with pytest.raises((NotImplementedError, ValueError)):
        build_adapter("dna_bert2")
