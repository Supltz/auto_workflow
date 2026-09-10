"""Per-input checkpoints: JSONL records, not row counts, determine completion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

from src.utils.io import atomic_write_json, read_jsonl, rewrite_jsonl_atomic

_SIGNATURES: ContextVar[dict[tuple[str, ...], str] | None] = ContextVar("signatures", default=None)
CHECK_ONLY: ContextVar[bool] = ContextVar("check_only", default=False)


class StagePending(Exception):
    """A read-only stage probe found unfinished or incompatible records."""


def semantic(value: Any) -> Any:
    """Ignore timing/model response logs while retaining actual input evidence."""
    if isinstance(value, dict):
        return {
            k: semantic(v)
            for k, v in value.items()
            if k != "_input_signature"
            and k != "raw_response"
            and not k.endswith("runtime_ms")
            and not k.endswith("metadata")
        }
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def record_key(record: dict[str, Any]) -> tuple[str, ...]:
    if "expression_id" in record:
        return (record["image_id"], record["expression_id"])
    if "entity_id" in record:
        return (record["image_id"], record["entity_id"])
    return (record["image_id"],)


def stamp(record: dict[str, Any]) -> dict[str, Any]:
    signature = signature_for(record, _SIGNATURES.get() or {})
    return {**record, "_input_signature": signature} if signature else record


def signature_for(record, signatures):
    key = record_key(record)
    if key not in signatures and "entity_id" in record:
        key = (record["image_id"], record["entity_id"])
    return signatures.get(key)


def archive_and_replace(path: Path, records: list[dict[str, Any]], *, history_limit: int | None = None) -> None:
    """Replace the active checkpoint atomically; full historical copies are opt-in.

    The bounded update summary is diagnostic only, never a recovery input.
    Existing archives are left alone when history is disabled.
    """
    if history_limit is None:
        history_limit = int(os.environ.get("ROUTE_B_CHECKPOINT_HISTORY_LIMIT", "0"))
    if history_limit < 0:
        raise ValueError("checkpoint history limit must be non-negative")
    old = None
    archive = None
    if path.exists():
        old = list(read_jsonl(path))
        if old == records:
            return
        if old and history_limit:
            archive = (
                path.parent / "checkpoint_archive" / f"{path.stem}_{fingerprint(old)[:16]}.jsonl"
            )
            if not archive.exists():
                rewrite_jsonl_atomic(archive, old)
    rewrite_jsonl_atomic(path, records)
    if old is not None:
        atomic_write_json(path.parent / "checkpoint_updates" / f"{path.name}.json", {
            "stage_file": path.name, "updated_at": time.time(),
            "before_digest": fingerprint(old), "after_digest": fingerprint(records),
            "before_count": len(old), "after_count": len(records),
            "status": "replaced", "history_limit": history_limit,
        })
    # Prune only this stage's recognized history, and only after successful replace.
    if archive is not None:
        pattern = re.compile(re.escape(path.stem) + r"_[0-9a-f]{16}\.jsonl")
        histories = [p for p in archive.parent.iterdir()
                     if pattern.fullmatch(p.name) and p.is_file() and not p.is_symlink()]
        others = sorted((p for p in histories if p != archive),
                        key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
        for obsolete in others[max(0, history_limit - 1):]:
            obsolete.unlink()


def checkpointed(input_name: str, *, accepted_only: bool = False):
    """Keep only outputs matching exact semantic inputs and current prompts/config.

    Stamping happens on every successful append, so interruption preserves completed
    items. Old unstamped outputs are invalidated, never guessed to be compatible.
    Unchanged items retain their checkpoints when another source image changes.
    """

    def decorate(function):
        @wraps(function)
        def run(**kwargs):
            inputs = kwargs[input_name]
            if function.__name__ == "align_entities_to_instances":
                inputs = [
                    {"image_id": row["image_id"], "entity_id": e["entity_id"], "entity": e}
                    for row in inputs
                    for e in row["entities"]
                ]
            if accepted_only:
                inputs = [r for r in inputs if r["accepted"]]
            context = {
                "stage": function.__name__,
                "contract": "visual_discovery_native_png_1",
                "config": kwargs["config"],
                "model": kwargs.get("qwen_config"),
                "prompt": kwargs["prompt_path"].read_text(),
            }
            signatures = {}
            for record in inputs:
                image_id = record["image_id"]
                extras = {}
                for name in ("candidates", "ocr_records"):
                    if name in kwargs:
                        extras[name] = sorted(
                            [semantic(r) for r in kwargs[name] if r["image_id"] == image_id],
                            key=lambda r: json.dumps(r, sort_keys=True),
                        )
                if "raw_records" in kwargs:
                    extras["grounding"] = [
                        semantic(r)
                        for r in kwargs["raw_records"]
                        if r.get("metadata", {}).get("expression_id") == record.get("expression_id")
                    ]
                if "manifest" in kwargs:
                    extras["source"] = kwargs["manifest"][image_id]
                signatures[record_key(record)] = fingerprint([context, semantic(record), extras])
            path = kwargs["output_path"]
            existing = []
            if path.exists() and kwargs.get("resume") and not kwargs.get("overwrite"):
                existing = [
                    r
                    for r in read_jsonl(path)
                    if r.get("_input_signature") is not None
                    and r.get("_input_signature") == signature_for(r, signatures)
                ]
                existing = list({record_key(r): r for r in existing}.values())
            if CHECK_ONLY.get():
                if (
                    not path.exists()
                    or len(existing) != len(signatures)
                    or len(list(read_jsonl(path))) != len(existing)
                ):
                    raise StagePending(function.__name__)
                return None
            if existing:
                from src.routes.model_failures import report_model_failures
                invalid = sum(bool(r.get("model_output_invalid")) for r in existing)
                if invalid:
                    report_model_failures(function.__name__ + " cached", invalid, len(existing))
            if path.exists() and kwargs.get("resume") and not kwargs.get("overwrite"):
                archive_and_replace(path, existing)
            if path.exists() and len(existing) == len(signatures) and not kwargs.get("overwrite"):
                return None
            token = _SIGNATURES.set(signatures)
            try:
                return function(**kwargs)
            finally:
                _SIGNATURES.reset(token)

        return run

    return decorate
