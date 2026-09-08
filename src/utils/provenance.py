"""Model/resource provenance helpers."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import config_hash, resolve_path
from src.utils.io import JsonlWriter


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: str | Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_name() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return output.splitlines()[0] if output else None
    except (OSError, subprocess.CalledProcessError):
        return None


def model_metadata(
    model_config: dict[str, Any],
    effective_config: dict[str, Any],
    prompt_version: str,
) -> dict[str, Any]:
    repo_path = model_config.get("repo_path")
    return {
        "model_name": model_config["name"],
        "model_revision": model_config.get("revision"),
        "checkpoint": str(resolve_path(model_config["local_path"])),
        "git_commit": git_commit(resolve_path(repo_path)) if repo_path else None,
        "dtype": model_config.get("dtype", "unknown"),
        "gpu": gpu_name(),
        "prompt_version": prompt_version,
        "config_hash": config_hash(effective_config),
    }


def record_run(
    path: str | Path,
    component: str,
    effective_config: dict[str, Any],
    **metadata: Any,
) -> None:
    JsonlWriter(resolve_path(path)).append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "config_hash": config_hash(effective_config),
            "gpu": gpu_name(),
            **metadata,
        }
    )
