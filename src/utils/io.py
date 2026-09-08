"""Crash-safe JSON/JSONL helpers and common long-job CLI flags."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from src.schema import FailureRecord


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record is not an object at {path}:{line_number}")
            yield value


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class JsonlWriter:
    """Append complete records under an advisory lock, flushing each record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def rewrite_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def completed_keys(path: str | Path, fields: tuple[str, ...]) -> set[tuple[Any, ...]]:
    source = Path(path)
    if not source.exists():
        return set()
    return {tuple(item.get(field) for field in fields) for item in read_jsonl(source)}


def slice_records(
    records: Iterable[dict[str, Any]], start_index: int, end_index: int | None
) -> Iterator[dict[str, Any]]:
    for index, record in enumerate(records):
        if index < start_index:
            continue
        if end_index is not None and index >= end_index:
            break
        yield record


def failure_writer(path: str | Path, **kwargs: Any) -> None:
    failure = FailureRecord(**kwargs)
    JsonlWriter(path).append(failure.model_dump(mode="json"))


def add_job_arguments(parser: argparse.ArgumentParser, default_config: str) -> None:
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir")


def prepare_output(path: Path, overwrite: bool, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite:
        path.unlink()
    elif path.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume or --overwrite: {path}")
