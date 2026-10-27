import json
import sys

import torch
from _fake_models import FakeEsmModel, FakeEsmTokenizer

from biointerp.experiments import activation_patching as exp


def _sequences(n):
    return ["ACACACACAC"] * n


def test_capture_shapes_and_probabilities():
    model = FakeEsmModel(hidden=16, intermediate=24, n_layers=3, seed=0).eval()
    tokenizer = FakeEsmTokenizer()
    from biointerp.esm import aa_token_ids

    hidden, correct, probs = exp.capture(
        model,
        tokenizer,
        _sequences(6),
        layers=[0, 2],
        batch_size=3,
        seed=0,
        device="cpu",
        aa_t=torch.tensor(aa_token_ids(tokenizer)),
    )

    assert hidden[0].shape == (6, 16)
    assert hidden[2].shape == (6, 16)
    assert correct.shape == (6,)
    assert correct.dtype == bool
    assert probs.shape == (6, 20)
    assert torch.allclose(torch.from_numpy(probs).sum(-1), torch.ones(6))


def test_main_writes_all_conditions(monkeypatch, tmp_path):
    model = FakeEsmModel(hidden=16, intermediate=24, n_layers=3, seed=0).eval()
    tokenizer = FakeEsmTokenizer()
    all_sequences = _sequences(620)

    monkeypatch.setattr(exp, "fetch_swissprot", lambda n, seed=0: all_sequences)
    monkeypatch.setattr(exp, "load_model", lambda *a, **k: (model, tokenizer, "cpu"))
    monkeypatch.setattr(exp, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "patch",
            "--layers",
            "0,2",
            "--donor-n",
            "40",
            "--eval-n",
            "20",
            "--batch-size",
            "10",
        ],
    )

    exp.main()

    out = json.loads((tmp_path / "activation_patching.json").read_text())
    assert out["n_eval"] == 20
    assert out["n_incorrect"] > 0
    assert out["n_incorrect"] < 20
    assert len(out["rows"]) == 6
    conditions = [row["condition"] for row in out["rows"]]
    assert conditions == [
        "l0_correct-donor",
        "l0_incorrect-donor",
        "l0_self-patch",
        "l2_correct-donor",
        "l2_incorrect-donor",
        "l2_self-patch",
    ]
    for row in out["rows"]:
        assert 0.0 <= row["p_true_all"] <= 1.0
        assert 0.0 <= row["p_true_incorrect"] <= 1.0
        assert row["wrong_to_right"] >= 0
        assert row["right_to_wrong"] >= 0
    assert 0.0 <= out["baseline_p_true_incorrect"] <= 1.0
