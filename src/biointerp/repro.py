"""Reproducibility metadata and result contracts.

The goal is to make every canonical result self-describing: which config was
used, which package versions produced it, and what schema it must satisfy.
Containerization is intentionally out of scope here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "uv.lock",
    REPO_ROOT / "conf" / "config.yaml",
)
PACKAGES = (
    "torch",
    "transformers",
    "scikit-learn",
    "numpy",
    "scipy",
    "pandas",
)


def file_digest(paths: Iterable[str | Path] | None = None) -> str:
    """SHA-256 over the given files, in order."""
    digest = hashlib.sha256()
    for path in paths or DEFAULT_INPUTS:
        path = Path(path)
        if not path.exists():
            continue
        digest.update(str(path.name).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_hash(paths: Iterable[str | Path] | None = None) -> str:
    """Short digest of the environment-defining inputs."""
    return file_digest(paths)[:16]


def package_versions() -> dict[str, str]:
    """Installed versions of the core scientific stack."""
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    return versions


def metadata(paths: Iterable[str | Path] | None = None) -> dict[str, object]:
    """Metadata block attached to canonical results."""
    return {
        "config_hash": config_hash(paths),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def write_with_metadata(
    path: str | Path,
    payload: Mapping[str, object],
    paths: Iterable[str | Path] | None = None,
) -> dict[str, object]:
    """Write JSON with a ``meta`` block and return the payload."""
    result = dict(payload)
    result["meta"] = metadata(paths)
    Path(path).write_text(json.dumps(result, indent=2) + "\n")
    return result


def missing_keys(path: str | Path, required: Sequence[str]) -> list[str]:
    """Return required top-level keys missing from a JSON result."""
    result_path = Path(path)
    if not result_path.exists():
        return ["<file missing>"]
    try:
        payload = json.loads(result_path.read_text())
    except json.JSONDecodeError:
        return ["<invalid json>"]
    return [key for key in required if key not in payload]


def check_contract(spec: Mapping[str, object]) -> list[str]:
    """Validate a repro spec; return human-readable failures."""
    failures: list[str] = []
    for entry in spec.get("results", []):  # type: ignore[union-attr]
        if isinstance(entry, str):
            path, required = entry, []
        else:
            path = str(entry.get("path", ""))
            required = list(entry.get("required", []))
        if not path:
            failures.append("result entry without a path")
            continue
        missing = missing_keys(REPO_ROOT / path, required)
        if missing:
            failures.append(f"{path}: missing {missing}")
    return failures


def repro_main(args) -> int:
    """CLI entry point for the reproducibility contract."""
    import yaml

    spec_path = Path(args.config)
    if not spec_path.exists():
        raise SystemExit(f"repro config not found: {spec_path}")
    spec = yaml.safe_load(spec_path.read_text()) or {}
    meta = metadata()

    if args.dry_run:
        print(json.dumps(meta, indent=2))
        return 0

    failures = check_contract(spec)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"spec": str(spec_path), "meta": meta, "failures": failures}, indent=2) + "\n"
    )

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"wrote: {output}")
        return 1

    print(f"repro contract OK ({len(spec.get('results', []))} results)")
    print(f"config hash: {meta['config_hash']} | python {meta['python']}")
    print(f"wrote: {output}")
    return 0
