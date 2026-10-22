"""Unified experiment runner (plain YAML config; no Hydra).

Configuration lives in ``conf/``:

    conf/config.yaml        base defaults
    conf/model/*.yaml       adapter + HuggingFace id
    conf/dataset/*.yaml     dataset source/parameters (no data files)
    conf/experiment/*.yaml  module path + default args
    conf/grids/*.yaml       sweep definitions

Experiments keep their own argparse ``main()``; the runner resolves the
configuration, redirects outputs via ``BIOINTERP_RESULTS_DIR`` and runs the
module. This is the single entry point for runs and grids.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_config(
    experiment: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve base + experiment + model + dataset configuration."""
    base = _load(CONF_DIR / "config.yaml")
    defaults = base.get("defaults", {})
    experiment = experiment or defaults.get("experiment")
    model = model or defaults.get("model")
    dataset = dataset or defaults.get("dataset")
    if not experiment:
        raise ValueError("no experiment given and no default in conf/config.yaml")
    exp_cfg = _load(CONF_DIR / "experiment" / f"{experiment}.yaml")
    model_cfg = _load(CONF_DIR / "model" / f"{model}.yaml") if model else {}
    data_cfg = _load(CONF_DIR / "dataset" / f"{dataset}.yaml") if dataset else {}
    return {
        "base": base,
        "experiment": experiment,
        "model": model,
        "dataset": dataset,
        "experiment_cfg": exp_cfg,
        "model_cfg": model_cfg,
        "dataset_cfg": data_cfg,
    }


def build_argv(cfg: Dict[str, Any], overrides: Optional[Dict[str, str]] = None) -> List[str]:
    """Build the experiment argv from its config plus CLI overrides."""
    argv: List[str] = [str(a) for a in cfg["experiment_cfg"].get("args", [])]
    if cfg["experiment_cfg"].get("model_key", True) and cfg.get("model"):
        argv += ["--model-key", str(cfg["model"])]
    for key, value in (overrides or {}).items():
        flag = f"--{key.replace('_', '-')}"
        if value is True or value == "":
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def _out_dir(cfg: Dict[str, Any]) -> Path:
    root = Path(cfg["base"].get("output_dir", "outputs"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / cfg["experiment"] / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def run(
    experiment: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    overrides: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one experiment; returns {rc, out_dir, cmd}."""
    cfg = load_config(experiment, model, dataset)
    module = cfg["experiment_cfg"].get("module", f"biointerp.experiments.{cfg['experiment']}")
    argv = build_argv(cfg, overrides)
    out_dir = _out_dir(cfg)
    cmd = [sys.executable, "-m", module, *argv]

    if dry_run:
        print("cwd:", REPO_ROOT)
        print("cmd:", " ".join(cmd))
        print("BIOINTERP_RESULTS_DIR:", out_dir)
        return {"rc": 0, "out_dir": str(out_dir), "cmd": cmd, "dry_run": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": cfg["experiment"],
                "model": cfg["model"],
                "dataset": cfg["dataset"],
                "module": module,
                "argv": argv,
                "overrides": overrides or {},
            },
            sort_keys=False,
        )
    )
    env = os.environ.copy()
    env["BIOINTERP_RESULTS_DIR"] = str(out_dir)
    log_path = out_dir / "run.log"
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[rc={proc.returncode}] {log_path}")
    return {"rc": proc.returncode, "out_dir": str(out_dir), "cmd": cmd}


def _args_to_overrides(args: List[Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(args):
        item = str(args[i])
        if item.startswith("--"):
            key = item[2:].replace("-", "_")
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                overrides[key] = args[i + 1]
                i += 2
            else:
                overrides[key] = True
                i += 1
        else:
            i += 1
    return overrides


def run_grid(grid_path: str, dry_run: bool = False) -> List[Dict[str, Any]]:
    """Run every entry of a grid YAML sequentially."""
    grid = _load(Path(grid_path))
    results = []
    for entry in grid.get("runs", []):
        experiment = Path(str(entry["experiment"])).stem
        overrides = _args_to_overrides(entry.get("args", []))
        results.append(
            run(
                experiment=experiment,
                model=entry.get("model_key"),
                dataset=entry.get("dataset"),
                overrides=overrides,
                dry_run=dry_run,
            )
        )
    return results
