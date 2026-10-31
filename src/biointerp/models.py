"""Model registry for Bio-Interp.

Model keys resolve to adapter + HuggingFace id from ``conf/model/*.yaml``::

    esm2_8m, esm2_35m, esm2_150m, esm2_650m, esm2_3b, esmc_300m, esmc_600m, protgpt2

``load_model`` dispatches to the right adapter and always returns
``(model, tokenizer, device)``. Accepts a registry key or a raw HF id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

CONFIG_DIR = Path(__file__).resolve().parents[2] / "conf" / "model"


def registry() -> dict:
    import yaml

    reg = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text()) or {}
        spec.setdefault("modality", "protein")
        reg[path.stem] = spec
    return reg


def get_spec(key_or_id: str) -> dict:
    reg = registry()
    if key_or_id in reg:
        return {"key": key_or_id, **reg[key_or_id]}
    for key, spec in reg.items():
        if spec.get("hf_id") == key_or_id:
            return {"key": key, **spec}
    raise KeyError(f"unknown model key or HF id: {key_or_id!r} (known: {sorted(reg)})")


def resolve_id(key_or_id: str) -> str:
    return get_spec(key_or_id)["hf_id"]


def list_models(modality: Optional[str] = None) -> list[str]:
    reg = registry()
    if modality is not None:
        reg = {key: spec for key, spec in reg.items() if spec.get("modality") == modality}
    return sorted(reg)


def load_model(key_or_id: str, device: Optional[str] = None) -> Tuple[object, object, str]:
    spec = get_spec(key_or_id)
    if spec.get("modality") == "dna":
        from biointerp.dna import load_dna_model

        return load_dna_model(spec, device)

    adapter = spec.get("adapter", "esm")
    if adapter == "esm":
        from biointerp.esm import load_model as _load
    elif adapter == "esmc":
        from biointerp.esmc import load_esmc as _load
    elif adapter == "gen":
        from biointerp.gen import load_generative as _load
    else:
        raise ValueError(f"unknown adapter {adapter!r} for model {key_or_id!r}")
    return _load(spec["hf_id"], device)
