"""Reuse completed spatial queries by image/query/model, rebinding current metadata."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import yaml

from src.routes.route_b_checkpoint import fingerprint
from src.utils.config import PROJECT_ROOT, config_hash
from src.utils.io import read_jsonl, rewrite_jsonl_atomic


@lru_cache(maxsize=1)
def legacy_contract():
    """Optional local metadata establishes the legacy cache source."""

    root = Path(os.environ.get("WORKFLOW_LEGACY_ROOT", PROJECT_ROOT / ".local/legacy"))
    def committed(path):
        return (root / path).read_text(encoding="utf-8")

    try:
        return (
            {
                r["image_id"]: r
                for r in map(json.loads, committed("data/manifest.jsonl").splitlines())
            },
            yaml.safe_load(committed("configs/models.yaml")),
            yaml.safe_load(committed("configs/route_b.yaml")),
        )
    except (OSError, ValueError, yaml.YAMLError):
        return {}, {}, {}


def query_identity(request):
    return (
        request["image_id"],
        request["image_path"],
        request["phrase"],
        request["image_width"],
        request["image_height"],
    )


def reconcile_worker_cache(
    *,
    requests,
    prior_requests,
    output,
    progress_path,
    model_config,
    grounder,
    environment,
    save_masks,
    resume,
    overwrite,
):
    """Only a full, durable success marker can commit detections (including zero)."""
    context = fingerprint([model_config, environment or {}, save_masks])
    rows = list(read_jsonl(output)) if output.exists() and resume and not overwrite else []
    progress = (
        list(read_jsonl(progress_path))
        if progress_path.exists() and resume and not overwrite
        else []
    )
    grouped = defaultdict(dict)
    for row in rows:
        meta = row["metadata"]
        grouped[meta["request_id"]][meta["detection_index"]] = row
    old_by_id = {r["request_id"]: r for r in prior_requests}
    reusable = {}
    legacy_manifest, legacy_models, legacy_config = legacy_contract()
    for entry in progress:
        old = old_by_id.get(entry["request_id"])
        if old is None:
            continue
        detections = list(grouped[entry["request_id"]].values())
        if len(detections) != entry["detections"]:
            continue
        if any(r.get("mask_path") and not Path(r["mask_path"]).is_file() for r in detections):
            continue
        image_hash = old.get("metadata", {}).get("image_sha256")
        if entry.get("context_signature") != context:
            # Import only verifiable initial-run results. Unverifiable checkpoints
            # (including changed model settings) are recomputed, not guessed safe.
            source = legacy_manifest.get(old["image_id"], {})
            if entry.get("context_signature") or model_config != legacy_models.get(grounder):
                continue
            if (environment or {}) != legacy_config.get("grounder_environment", {}):
                continue
            if old["image_path"] != source.get("image_path"):
                continue
            if any(
                r["metadata"].get("config_hash") != config_hash(model_config) for r in detections
            ):
                continue
            image_hash = source.get("image_sha256")
        reusable[(query_identity(old), image_hash)] = detections
    retained, committed = [], []
    for request in requests:
        key = (query_identity(request), request.get("metadata", {}).get("image_sha256"))
        if key not in reusable or not key[1]:
            continue
        records = reusable[key]
        for row in records:
            retained.append(
                {
                    **row,
                    "rank": request["rank"],
                    "phrase": request["phrase"],
                    "source_method": request["metadata"]["source_method"],
                    "metadata": {
                        **row["metadata"],
                        **request["metadata"],
                        "request_id": request["request_id"],
                    },
                }
            )
        committed.append(
            {
                "request_id": request["request_id"],
                "image_id": request["image_id"],
                "detections": len(records),
                "context_signature": context,
            }
        )
    rewrite_jsonl_atomic(output, retained)
    rewrite_jsonl_atomic(progress_path, committed)
    return context, {r["request_id"] for r in committed}
