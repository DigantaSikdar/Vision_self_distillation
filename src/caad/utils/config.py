"""
caad/utils/config.py
====================
Plain-YAML config loading with dotted CLI overrides.

Training recipes under configs/train/ are SELF-CONTAINED — one complete file per
run, nothing merged in (so a recipe reads top-to-bottom as the full description
of that run). To start a new experiment, copy a recipe and edit it.

CLI overrides use dotted paths and win over the file:
    caad.lambda_l2=0.5 lora.enabled=false

An optional ``defaults:`` key (list of YAML files, relative to the file or to
configs/) is still supported for deep-merge composition if you ever want it, but
the shipped recipes don't use it.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(path: Path, ref: str, root: Path) -> Path:
    """Resolve a `defaults:` entry relative to the file, then to configs root."""
    for cand in (path.parent / ref, root / ref):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"config default '{ref}' not found from {path}")


def load_config(path, *, overrides=None, _root=None, _seen=None) -> dict:
    """Load a YAML config, resolving `defaults:` inheritance and CLI overrides."""
    path = Path(path).resolve()
    root = Path(_root) if _root else path.parent
    seen = _seen if _seen is not None else set()
    if path in seen:
        raise ValueError(f"circular config inheritance at {path}")
    seen.add(path)

    raw = yaml.safe_load(path.read_text()) or {}
    defaults = raw.pop("defaults", []) or []

    merged: dict = {}
    for ref in defaults:
        parent = _resolve(path, ref, root)
        merged = _deep_merge(merged, load_config(parent, _root=root, _seen=seen))
    merged = _deep_merge(merged, raw)

    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_dotted(merged, key.strip(), _coerce(val.strip()))
    return merged


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _set_dotted(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def freeze_config(cfg: dict, out_dir) -> Path:
    """Write the fully-resolved config next to a run's artifacts (reproducibility)."""
    out = Path(out_dir) / "config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
