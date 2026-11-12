import json

import joblib
import numpy as np
import pytest
from _fake_models import FakeEsmModel, FakeEsmTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from biointerp import cli
from biointerp.score import parse_variant, score_variant, validate_variant

SEQUENCE = "ACDEFGHIKL"


def _probe(n_features: int = 24, n_selected: int = 3):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, n_features))
    y = rng.integers(0, 2, 40)
    scaler = StandardScaler().fit(x[:, :n_selected])
    clf = LogisticRegression().fit(scaler.transform(x[:, :n_selected]), y)
    return {
        "model": "fake",
        "layers": [0, 1],
        "top_k": n_selected,
        "idx": np.arange(n_selected),
        "scaler": scaler,
        "clf": clf,
        "reference_scores": [0.2, 0.4, 0.6, 0.8],
    }


def test_parse_variant_valid():
    assert parse_variant("E30T") == ("E", 30, "T")
    assert parse_variant(" e30t ") == ("E", 30, "T")


@pytest.mark.parametrize("text", ["30T", "E30", "E0T", "E30T extra", "E30*"])
def test_parse_variant_invalid(text):
    with pytest.raises(ValueError):
        parse_variant(text)


def test_validate_variant_rejects_mismatch():
    with pytest.raises(ValueError, match="wild-type mismatch"):
        validate_variant(SEQUENCE, "W", 2)


def test_validate_variant_rejects_out_of_range():
    with pytest.raises(ValueError, match="exceeds sequence length"):
        validate_variant(SEQUENCE, "A", 99)


def test_score_variant_with_fake_model():
    model = FakeEsmModel(hidden=8, intermediate=12, n_layers=2).eval()
    tokenizer = FakeEsmTokenizer()
    score = score_variant(model, tokenizer, _probe(), SEQUENCE, "C", 2, "cpu")
    assert 0.0 <= score <= 1.0


def test_cli_score_prints_json(monkeypatch, tmp_path, capsys):
    model = FakeEsmModel(hidden=8, intermediate=12, n_layers=2).eval()
    tokenizer = FakeEsmTokenizer()
    probe_path = tmp_path / "probe.joblib"
    joblib.dump(_probe(), probe_path)

    monkeypatch.setattr("biointerp.score.load_model", lambda *a, **k: (model, tokenizer, "cpu"))

    rc = cli.main(
        [
            "score",
            "--sequence",
            SEQUENCE,
            "--mutation",
            "C2T",
            "--probe",
            str(probe_path),
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mutation"] == "C2T"
    assert 0.0 <= payload["error_score"] <= 1.0
    assert payload["percentile"] is not None


def test_cli_score_rejects_bad_variant(monkeypatch, tmp_path):
    probe_path = tmp_path / "probe.joblib"
    joblib.dump(_probe(), probe_path)
    with pytest.raises(SystemExit):
        cli.main(
            [
                "score",
                "--sequence",
                SEQUENCE,
                "--mutation",
                "W2T",
                "--probe",
                str(probe_path),
            ]
        )
