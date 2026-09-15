"""Route B orchestration over JSONL and isolated model subprocesses."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any

from src.grounding.cache import reconcile_worker_cache
from src.utils.config import PROJECT_ROOT, resolve_path
from src.utils.io import read_jsonl, rewrite_jsonl_atomic

WORKER_MODULES = {
    "egm": "src.grounding.egm",
    "rex": "src.grounding.rex",
    "sam31": "src.grounding.sam31",
    "groundingdino": "src.grounding.groundingdino",
}


def selected_output(configured: str, output_dir: str | None) -> Path:
    configured_path = resolve_path(configured)
    if not output_dir:
        return configured_path
    output_root = resolve_path("outputs")
    try:
        relative = configured_path.relative_to(output_root)
    except ValueError:
        relative = Path(configured).name
    return resolve_path(output_dir) / relative


def manifest_index(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    return {record["image_id"]: record for record in read_jsonl(resolve_path(manifest_path))}


def sampled_manifest_rows(
    manifest_path: str | Path, start_index: int, end_index: int | None, seed: int = 42
) -> list[dict[str, Any]]:
    """Slice a reproducibly shuffled manifest without modifying the source file.

    Sorting before shuffling makes selection independent of physical JSONL order.
    The manifest contents and seed must remain fixed throughout a resumed run.
    """
    if start_index < 0 or (end_index is not None and end_index < start_index):
        raise ValueError("invalid manifest sample range")
    rows = sorted(read_jsonl(resolve_path(manifest_path)), key=lambda r: r["image_id"])
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("manifest contains duplicate image IDs")
    random.Random(seed).shuffle(rows)
    return rows[start_index:end_index]


def manifest_slice_ids(
    manifest_path: str | Path, start_index: int, end_index: int | None, seed: int = 42
) -> set[str]:
    return {r["image_id"] for r in sampled_manifest_rows(
        manifest_path, start_index, end_index, seed)}


def invoke_grounders(
    requests: list[dict[str, Any]],
    grounders: list[str],
    models_config_path: str,
    models_config: dict[str, Any],
    combined_output: Path,
    output_root: Path,
    failures_file: Path,
    resume: bool,
    overwrite: bool,
    save_crops: bool = True,
    save_overlays: bool = True,
    save_masks: bool = True,
    worker_environment: dict[str, Any] | None = None,
) -> None:
    request_path = combined_output.with_name(f"{combined_output.stem}_requests.jsonl")
    # Keep the prior manifest until ALL workers have rebound their old detections.
    # A durable worker-specific request manifest survives interruption between models.
    prior_requests = list(read_jsonl(request_path)) if request_path.exists() else []
    request_ids = {request["request_id"] for request in requests}
    worker_outputs: list[Path] = []
    for grounder in grounders:
        if grounder not in WORKER_MODULES:
            raise ValueError(f"unknown grounder: {grounder}")
        model_config = models_config[grounder]
        python = Path(model_config["env_python"])
        if not python.exists():
            raise FileNotFoundError(
                f"isolated environment for {grounder} is missing: {python}; see docs/MODEL_SETUP.md"
            )
        worker_output = combined_output.with_name(f"{combined_output.stem}_{grounder}.jsonl")
        worker_outputs.append(worker_output)
        progress_path = worker_output.with_name(f"{worker_output.stem}.progress.jsonl")
        worker_requests = worker_output.with_name(f"{worker_output.stem}.requests.jsonl")
        prior = list(read_jsonl(worker_requests)) if worker_requests.exists() else prior_requests
        signature, completed_ids = reconcile_worker_cache(
            requests=requests, prior_requests=prior, output=worker_output,
            progress_path=progress_path, model_config=model_config, grounder=grounder,
            environment=worker_environment, save_masks=save_masks,
            resume=resume, overwrite=overwrite)
        rewrite_jsonl_atomic(worker_requests, requests)
        if request_ids.issubset(completed_ids):
            print(f"Skipping completed grounder {grounder}: {len(completed_ids)} requests")
            continue
        command = [
            str(python),
            "-m",
            WORKER_MODULES[grounder],
            "--model-config",
            str(resolve_path(models_config_path)),
            "--input-jsonl",
            str(worker_requests),
            "--output-jsonl",
            str(worker_output),
            "--failures-file",
            str(failures_file),
            "--output-dir",
            str(output_root),
            "--resume" if resume else "--no-resume",
            "--save-crops" if save_crops else "--no-save-crops",
            "--save-overlays" if save_overlays else "--no-save-overlays",
            "--save-masks" if save_masks else "--no-save-masks",
            "--context-signature", signature,
        ]
        environment = {
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
            **{key: str(value) for key, value in (worker_environment or {}).items()},
        }
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        completed_ids = {item["request_id"] for item in read_jsonl(progress_path)}
        if not request_ids.issubset(completed_ids):
            raise RuntimeError(f"{grounder}: incomplete requests; resume required")
    def unique_worker_records():
        """Merge resumable worker files without double-counting interrupted writes."""
        seen: set[tuple[str, str, int]] = set()
        for worker_output in worker_outputs:
            if not worker_output.exists():
                continue
            for record in read_jsonl(worker_output):
                metadata = record.get("metadata", {})
                if metadata.get("request_id") not in request_ids:
                    continue
                key = (
                    str(record["grounder"]),
                    str(metadata["request_id"]),
                    int(metadata["detection_index"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                yield record

    rewrite_jsonl_atomic(combined_output, unique_worker_records())
    rewrite_jsonl_atomic(request_path, requests)
