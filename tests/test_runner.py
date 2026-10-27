import re
from types import SimpleNamespace

import yaml

from biointerp import runner
from biointerp.runner import (
    _args_to_overrides,
    _load,
    _out_dir,
    build_argv,
    load_config,
    run,
    run_grid,
)


def test_defaults_resolve():
    cfg = load_config()
    assert cfg["experiment"] == "probing_classifier"
    assert cfg["model"] == "esm2_650m"
    assert cfg["dataset"] == "swissprot"
    assert cfg["model_cfg"]["hf_id"] == "facebook/esm2_t33_650M_UR50D"


def test_model_key_flag_toggle():
    with_key = load_config("extract", "esm2_8m", "swissprot")
    assert build_argv(with_key)[:2] == ["--model-key", "esm2_8m"]

    probing = load_config("probing_classifier", "esm2_650m", "swissprot")
    assert "--model-key" not in build_argv(probing)

    ladder = load_config("ladder_scaling", "esm2_8m")
    assert "--model-key" not in build_argv(ladder)


def test_dry_run_builds_command():
    result = run("probing_classifier", model="esm2_650m", overrides={"n": "10"}, dry_run=True)
    assert result["dry_run"] is True
    assert result["cmd"][-2:] == ["--n", "10"]


def test_load_missing_file_returns_empty(tmp_path):
    assert _load(tmp_path / "missing.yaml") == {}


def test_build_argv_order_and_overrides():
    cfg = load_config("extract", "esm2_8m", "swissprot")
    argv = build_argv(cfg, {"n": "5", "device": True, "dtype": ""})
    assert argv == ["--model-key", "esm2_8m", "--n", "5", "--device", "--dtype"]


def test_build_argv_keeps_negative_values():
    cfg = load_config("ladder_scaling", "esm2_8m")
    assert build_argv(cfg, {"start": "-3"}) == ["--start", "-3"]


def test_out_dir_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    cfg = {"experiment": "probing_classifier", "base": {"output_dir": "outputs"}}
    out = _out_dir(cfg)
    assert out.parent == tmp_path / "outputs" / "probing_classifier"
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", out.name)


def test_out_dir_absolute(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path / "repo")
    cfg = {"experiment": "extract", "base": {"output_dir": str(tmp_path / "abs")}}
    assert _out_dir(cfg).parent == tmp_path / "abs" / "extract"


def test_dry_run_does_not_create_output_dir(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"
    monkeypatch.setattr(runner, "_out_dir", lambda cfg: out)
    result = run("probing_classifier", overrides={"n": "10"}, dry_run=True)
    assert result["rc"] == 0
    assert not out.exists()
    assert "BIOINTERP_RESULTS_DIR" in capsys.readouterr().out


def test_run_writes_config_and_results_env(monkeypatch, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setattr(runner, "_out_dir", lambda cfg: out)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = run("extract", model="esm2_8m", overrides={"n": "5"})

    assert result["rc"] == 0
    assert result["out_dir"] == str(out)
    assert captured["cmd"][1:3] == ["-m", "biointerp.experiments.extract"]
    assert captured["cwd"] == runner.REPO_ROOT
    assert captured["env"]["BIOINTERP_RESULTS_DIR"] == str(out)

    saved = yaml.safe_load((out / "config.yaml").read_text())
    assert saved["experiment"] == "extract"
    assert saved["model"] == "esm2_8m"
    assert saved["argv"] == ["--model-key", "esm2_8m", "--n", "5"]
    assert (out / "run.log").exists()


def test_run_propagates_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_out_dir", lambda cfg: tmp_path / "out")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=3))
    assert run("extract", dry_run=False)["rc"] == 3


def test_args_to_overrides_mixed():
    overrides = _args_to_overrides(["--n", "10", "--flag", "--alpha", "-0.3", "positional"])
    assert overrides == {"n": "10", "flag": True, "alpha": "-0.3"}


def test_run_grid_dispatches(monkeypatch, tmp_path):
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {
                        "experiment": "biointerp/experiments/extract.py",
                        "model_key": "esm2_8m",
                        "args": ["--n", "5"],
                    },
                    {
                        "experiment": "biointerp/experiments/probing_classifier.py",
                        "dataset": "swissprot",
                        "args": ["--n", "10"],
                    },
                ]
            }
        )
    )
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {"rc": 0}

    monkeypatch.setattr(runner, "run", fake_run)
    results = run_grid(str(grid))

    assert len(results) == 2
    assert calls[0] == {
        "experiment": "extract",
        "model": "esm2_8m",
        "dataset": None,
        "overrides": {"n": "5"},
        "dry_run": False,
    }
    assert calls[1]["experiment"] == "probing_classifier"
    assert calls[1]["model"] is None
    assert calls[1]["dataset"] == "swissprot"
    assert calls[1]["overrides"] == {"n": "10"}


def test_run_grid_empty(monkeypatch, tmp_path):
    grid = tmp_path / "grid.yaml"
    grid.write_text("runs: []\n")
    monkeypatch.setattr(runner, "run", lambda **kwargs: {"rc": 0})
    assert run_grid(str(grid)) == []
