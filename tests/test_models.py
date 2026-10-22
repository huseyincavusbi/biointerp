import pytest

from biointerp.models import registry, resolve_id


def test_registry_has_expected_models():
    reg = registry()
    for key in ("esm2_8m", "esm2_650m", "esmc_600m", "esm3", "protgpt2"):
        assert key in reg


def test_resolve_id():
    assert resolve_id("esm2_650m") == "facebook/esm2_t33_650M_UR50D"


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        resolve_id("nope")
