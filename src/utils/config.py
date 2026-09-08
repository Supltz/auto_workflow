"""Configuration loading, path resolution, and reproducibility hashes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    # Optional machine settings live outside version control. Merge only the
    # explicitly named config, preserving the effective runtime values and hashes.
    local = Path(os.environ.get("WORKFLOW_LOCAL_CONFIG", PROJECT_ROOT / ".local/config.yaml"))
    if local.exists():
        with local.open(encoding="utf-8") as handle:
            overrides = yaml.safe_load(handle)
        if not isinstance(overrides, dict):
            raise TypeError("Local configuration must be a mapping")
        key = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
        patch = overrides.get(key, {})
        if not isinstance(patch, dict):
            raise TypeError(f"Local override must be a mapping: {key}")
        def merge(base, extra):
            for name, item in extra.items():
                if isinstance(item, dict) and isinstance(base.get(name), dict):
                    merge(base[name], item)
                else:
                    base[name] = item
        merge(value, patch)
    elif "WORKFLOW_LOCAL_CONFIG" in os.environ:
        raise FileNotFoundError(f"Local configuration does not exist: {local}")
    return value


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    result = Path(path).expanduser()
    return result if result.is_absolute() else (root / result).resolve()


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_keys(config: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"{context} missing keys: {', '.join(missing)}")
