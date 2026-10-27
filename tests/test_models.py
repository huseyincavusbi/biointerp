import pytest

from biointerp import models


def test_registry_has_expected_models():
    reg = models.registry()
    for key in ("esm2_8m", "esm2_650m", "esmc_600m", "protgpt2"):
        assert key in reg
    assert reg["esm2_650m"]["hf_id"] == "facebook/esm2_t33_650M_UR50D"


def test_resolve_id():
    assert models.resolve_id("esm2_8m") == "facebook/esm2_t6_8M_UR50D"
    assert models.resolve_id("facebook/esm2_t6_8M_UR50D") == "facebook/esm2_t6_8M_UR50D"


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="unknown model key"):
        models.resolve_id("not-a-model")


def test_get_spec_by_hf_id_includes_key():
    spec = models.get_spec("facebook/esm2_t6_8M_UR50D")
    assert spec["key"] == "esm2_8m"
    assert spec["adapter"] == "esm"


def test_list_models_sorted():
    names = models.list_models()
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_load_model_dispatches_esm(monkeypatch):
    calls = []

    def fake(hf_id, device):
        calls.append((hf_id, device))
        return "model", "tokenizer", "cpu"

    monkeypatch.setattr("biointerp.esm.load_model", fake)
    assert models.load_model("esm2_8m", "cpu") == ("model", "tokenizer", "cpu")
    assert calls == [("facebook/esm2_t6_8M_UR50D", "cpu")]


def test_load_model_dispatches_esmc(monkeypatch):
    calls = []

    def fake(hf_id, device):
        calls.append((hf_id, device))
        return "model", "tokenizer", "cpu"

    monkeypatch.setattr("biointerp.esmc.load_esmc", fake)
    models.load_model("esmc_600m")
    assert calls == [("biohub/ESMC-600M", None)]


def test_load_model_dispatches_generative(monkeypatch):
    calls = []

    def fake(hf_id, device):
        calls.append((hf_id, device))
        return "model", "tokenizer", "cpu"

    monkeypatch.setattr("biointerp.gen.load_generative", fake)
    models.load_model("protgpt2")
    assert calls == [("nferruz/ProtGPT2", None)]


def test_load_model_unknown_adapter(monkeypatch):
    monkeypatch.setattr(models, "registry", lambda: {"odd": {"adapter": "bogus", "hf_id": "x"}})
    with pytest.raises(ValueError, match="unknown adapter"):
        models.load_model("odd")
