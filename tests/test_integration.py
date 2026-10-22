"""Real-model smoke test (marked ``real_model``; runs in the integration CI job)."""

import pytest

pytestmark = pytest.mark.real_model


def test_esm2_small_masked_forward():
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name).eval()
    inputs = tokenizer("MKTAYIAK<mask>RQISFVK", return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    assert output.logits.shape[0] == 1
    assert output.logits.shape[-1] > 20
