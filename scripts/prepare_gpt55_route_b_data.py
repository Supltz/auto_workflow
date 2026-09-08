#!/usr/bin/env python3
"""Download GPT-5.5-captioned images that overlap the local ModelScope shard."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow.parquet as pq
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.schema import DatasetRecord
from src.utils.config import load_yaml, resolve_path
from src.utils.io import atomic_write_json, atomic_write_text, rewrite_jsonl_atomic

_THREAD_LOCAL = threading.local()


def _image_id_from_url(url: str) -> str:
    return Path(urlparse(url).path).stem


def _load_annotations(paths: list[Path]) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                caption = record.get("caption")
                if not caption:
                    continue
                image_id = f"sa_{record['image_id']}"
                expected_name = f"{image_id}.jpg"
                for key in ("image_name", "file_name", "member_name"):
                    actual_name = Path(str(record.get(key, ""))).name
                    if actual_name != expected_name:
                        raise ValueError(
                            f"{key} mismatch for {image_id}: {actual_name!r} != "
                            f"{expected_name!r} at {path}:{line_number}"
                        )
                if image_id in annotations:
                    raise ValueError(f"duplicate annotation for {image_id}: {path}:{line_number}")
                annotations[image_id] = record
    return annotations


def _match_source_rows(
    parquet_path: Path, annotations: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    table = pq.read_table(parquet_path, columns=["opensource_url", "global_caption"])
    matches: list[dict[str, Any]] = []
    for source_index, (url, legacy_caption) in enumerate(
        zip(table["opensource_url"].to_pylist(), table["global_caption"].to_pylist())
    ):
        if not url:
            continue
        image_id = _image_id_from_url(url)
        annotation = annotations.get(image_id)
        if annotation is None:
            continue
        matches.append(
            {
                "image_id": image_id,
                "url": url,
                "source_index": source_index,
                "legacy_caption": legacy_caption,
                "annotation": annotation,
            }
        )
    if len({item["image_id"] for item in matches}) != len(matches):
        raise ValueError("the Parquet shard contains duplicate matching image IDs")
    return matches


def _session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "route-b-gpt55-data-preparation/1.0"
        _THREAD_LOCAL.session = session
    return session


def _inspect_image(data: bytes, expected_width: int, expected_height: int) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"dimension mismatch: downloaded={width}x{height}, "
                f"annotation={expected_width}x{expected_height}"
            )
        rgb = image.convert("RGB")
        pixel_hash = hashlib.sha256(
            f"{width}x{height}:RGB:".encode() + rgb.tobytes()
        ).hexdigest()
    return width, height, pixel_hash


def _inspect_file(
    path: Path, expected_width: int, expected_height: int
) -> tuple[int, int, str]:
    return _inspect_image(path.read_bytes(), expected_width, expected_height)


def _download_one(
    item: dict[str, Any], images_dir: Path, timeout: float, retries: int
) -> dict[str, Any]:
    annotation = item["annotation"]
    destination = images_dir / f"{item['image_id']}.jpg"
    expected_width = int(annotation["width"])
    expected_height = int(annotation["height"])
    if destination.exists():
        try:
            width, height, pixel_hash = _inspect_file(
                destination, expected_width, expected_height
            )
            return {**item, "width": width, "height": height, "pixel_hash": pixel_hash}
        except (OSError, UnidentifiedImageError, ValueError):
            pass

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = _session().get(item["url"], timeout=(15, timeout))
            response.raise_for_status()
            payload = response.content
            width, height, pixel_hash = _inspect_image(
                payload, expected_width, expected_height
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".jpg.part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return {**item, "width": width, "height": height, "pixel_hash": pixel_hash}
        except (OSError, requests.RequestException, UnidentifiedImageError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"{item['image_id']}: {last_error}")


def _manifest_record(item: dict[str, Any], source_shard: str) -> dict[str, Any]:
    annotation = item["annotation"]
    record = DatasetRecord(
        image_id=item["image_id"],
        image_path=f"data/images/{item['image_id']}.jpg",
        caption=annotation["caption"],
        width=item["width"],
        height=item["height"],
        source_index=item["source_index"],
        dataset="Tongyi-DataEngine/SA1B-Paired-Captions-Images",
        source_shard=source_shard,
        image_sha256=item["pixel_hash"],
        metadata={
            "source_locator": item["url"],
            "image_column": "opensource_url",
            "caption_source": "external_gpt55_jsonl",
            "caption_model": annotation.get("model"),
            "caption_prompt": annotation.get("prompt"),
            "caption_archive": annotation.get("archive"),
            "caption_member_name": annotation.get("member_name"),
            "caption_attempts": annotation.get("caption_attempts"),
            "caption_token_usage": annotation.get("token_usage"),
            "annotation_count": annotation.get("annotation_count"),
            "legacy_global_caption": item["legacy_caption"],
        },
    )
    return record.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=["sa_000074.jsonl", "sa_000523.jsonl"],
    )
    parser.add_argument("--output-dir", default="data_gpt55")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    experiment = load_yaml("configs/experiment.yaml")
    dataset_config = experiment["dataset"]
    parquet_path = Path(dataset_config["download_dir"]) / dataset_config["shard"]
    output_dir = resolve_path(args.output_dir)
    images_dir = output_dir / "images"
    annotations = _load_annotations([resolve_path(path) for path in args.annotations])
    matches = _match_source_rows(parquet_path, annotations)
    print(
        f"Matched {len(matches)} images in {parquet_path.name} "
        f"from {len(annotations)} successful GPT-5.5 captions",
        flush=True,
    )

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_items = {
            executor.submit(
                _download_one, item, images_dir, args.timeout, args.retries
            ): item
            for item in matches
        }
        for future in tqdm(
            as_completed(future_items), total=len(future_items), unit="image"
        ):
            item = future_items[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # preserve all failures for a resumable rerun
                failures.append(
                    {
                        "image_id": item["image_id"],
                        "url": item["url"],
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    completed.sort(key=lambda item: item["source_index"])
    failures.sort(key=lambda item: item["image_id"])
    rewrite_jsonl_atomic(output_dir / "download_failures.jsonl", failures)
    summary = {
        "annotation_files": args.annotations,
        "successful_caption_count": len(annotations),
        "matched_image_count": len(matches),
        "downloaded_and_validated_count": len(completed),
        "failure_count": len(failures),
        "source_parquet": str(parquet_path),
    }
    atomic_write_json(output_dir / "download_summary.json", summary)
    if failures:
        print(json.dumps(summary, indent=2), flush=True)
        raise SystemExit(1)

    manifest = [
        _manifest_record(item, dataset_config["shard"])
        for item in completed
    ]
    rewrite_jsonl_atomic(output_dir / "manifest.jsonl", manifest)
    atomic_write_text(
        output_dir / "sample_ids.txt",
        "".join(f"{item['image_id']}\n" for item in completed),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
