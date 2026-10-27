import argparse
import importlib
import sys
from pathlib import Path

import pytest
import yaml

from biointerp.runner import CONF_DIR, build_argv, load_config

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = "biointerp.experiments"

EXPERIMENTS = sorted(
    p.stem for p in (SRC / "biointerp" / "experiments").glob("*.py") if p.stem != "__init__"
)
CONFIGS = sorted(p.stem for p in (CONF_DIR / "experiment").glob("*.yaml"))
GRIDS = sorted((CONF_DIR / "grids").glob("*.yaml"))


def _module(name: str):
    return importlib.import_module(f"{PACKAGE}.{name}")


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_modules_import(name):
    assert _module(name) is not None


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_help_exits_zero(name, monkeypatch, capsys):
    module = _module(name)
    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    with pytest.raises(SystemExit) as exc:
        module.parse_args()
    assert exc.value.code == 0
    capsys.readouterr()


@pytest.mark.parametrize("name", CONFIGS)
def test_config_argv_parses(name, monkeypatch, capsys):
    cfg = load_config(name, model="esm2_8m", dataset="swissprot")
    module = importlib.import_module(cfg["experiment_cfg"]["module"])
    argv = build_argv(cfg)
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    parsed = module.parse_args()
    assert isinstance(parsed, argparse.Namespace)
    capsys.readouterr()


def test_every_config_maps_to_module():
    missing = [name for name in CONFIGS if name not in EXPERIMENTS]
    assert missing == []


def test_config_modules_match_experiment_names():
    for name in CONFIGS:
        cfg = load_config(name)
        assert cfg["experiment_cfg"]["module"] == f"{PACKAGE}.{name}"


@pytest.mark.parametrize("path", GRIDS, ids=lambda p: p.stem)
def test_grid_entries_parse(path, monkeypatch, capsys):
    grid = yaml.safe_load(path.read_text()) or {}
    for entry in grid.get("runs", []):
        name = Path(str(entry["experiment"])).stem
        assert name in EXPERIMENTS, f"{path.name} references unknown experiment {name}"
        module = _module(name)
        monkeypatch.setattr(sys, "argv", ["prog", *[str(a) for a in entry.get("args", [])]])
        parsed = module.parse_args()
        assert isinstance(parsed, argparse.Namespace)
        capsys.readouterr()
