from pathlib import Path

import pytest
import transformers

from biointerp.experiments import carbonx_context


def test_carbonx_requirements_use_transformers_5():
    text = Path("requirements/carbonx.txt").read_text()
    assert "transformers>=5" in text
    assert "torch" in text


def test_carbonx_dry_run_prints_conditions(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "carbonx",
            "--ref",
            "ACGTACGTACGTACGTACGT",
            "--alt",
            "ACGTACGTACGTACGTACGA",
            "--dry-run",
        ],
    )
    carbonx_context.main()
    output = capsys.readouterr().out
    for condition in ("dna_only", "correct_gene", "wrong_gene", "irrelevant_text", "english_only"):
        assert f"=== {condition} ===" in output


def test_carbonx_model_run_is_gated_below_transformers_5(monkeypatch):
    if int(transformers.__version__.split(".")[0]) >= 5:
        pytest.skip("main environment has transformers>=5")
    monkeypatch.setattr(
        "sys.argv",
        [
            "carbonx",
            "--ref",
            "ACGTACGTACGTACGTACGT",
            "--alt",
            "ACGTACGTACGTACGTACGA",
        ],
    )
    with pytest.raises(SystemExit):
        carbonx_context.main()
