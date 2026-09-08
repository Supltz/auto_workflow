"""Export exactly the verified Route B expression and box for human review."""

from __future__ import annotations

import csv
import re
import shutil
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any

from src.regions.target_scope import scope_verified
from src.utils.images import open_rgb, save_referring_expression_overlay
from src.utils.io import atomic_write_text, read_jsonl


def _select(
    records: list[dict[str, Any]],
    max_per_source_image: int,
) -> list[dict[str, Any]]:
    """Keep every verified target, subject only to the per-source safety cap."""
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_image[record["image_id"]].append(record)
    selected = []
    for image_id in sorted(by_image):
        image_records = sorted(
            by_image[image_id],
            key=lambda item: (
                int(item.get("entity_rank", 0)),
                str(item["entity_id"]),
                str(item["instance_id"]),
            ),
        )
        selected.extend(image_records[:max_per_source_image])
    return selected


def _write_index(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "review_file",
        "image_id",
        "entity_id",
        "instance_id",
        "region_id",
        "category",
        "origin",
        "caption_supported",
        "caption_span",
        "initial_locator_query",
        "final_referring_expression",
        "visible_evidence",
        "bbox_supporting_grounders",
        "bbox_grounder_support",
        "expression_grounder_support",
        "reground_iou",
        "bbox_median_pairwise_iou",
        "revision",
        "distractor_instance_ids",
        "bbox_area_ratio",
        "size_band",
        "representative_method",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "tight_crop",
        "context_crop",
        "source_image",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    atomic_write_text(path, buffer.getvalue())


def _visible_evidence(record: dict[str, Any]) -> str:
    values = list(record.get("used_attributes", []))
    values.extend(
        value
        for value in (record.get("used_action"), record.get("used_relation"))
        if value
    )
    values.extend(record.get("used_visible_text", []))
    return " | ".join(values)


def export_route_b_review(
    *,
    verified_path: Path,
    review_root: Path,
    selected_ids: set[str],
    max_per_source_image: int,
    overwrite: bool,
) -> None:
    verified = [
        item
        for item in read_jsonl(verified_path)
        if item["image_id"] in selected_ids
        and item["status"] == "verified_unique_referring_expression"
    ]
    if any(not scope_verified(record) for record in verified):
        raise ValueError("Review inputs lack current target-scope verification; rerun bbox QA onward")
    # Validate before replacing any existing human-review files.
    if overwrite and review_root.exists():
        shutil.rmtree(review_root)
    review_root.mkdir(parents=True, exist_ok=True)
    selected = _select(verified, max_per_source_image)

    rows = []
    for index, record in enumerate(selected, 1):
        if index == 1 or index % 10 == 0 or index == len(selected):
            print(f"[review] rendering={index}/{len(selected)}", flush=True)
        safe_entity = re.sub(r"[^A-Za-z0-9_-]+", "_", record["entity_id"])
        filename = f"{index:04d}_{record['image_id']}_{safe_entity}.jpg"
        destination = review_root / filename
        image = open_rgb(record["source_image"])
        try:
            save_referring_expression_overlay(
                image,
                tuple(record["bbox_xyxy"]),
                record["final_referring_expression"],
                destination,
            )
        finally:
            image.close()
        x1, y1, x2, y2 = record["bbox_xyxy"]
        reground = record["reground_audit"]
        rows.append(
            {
                "review_file": str(destination),
                "image_id": record["image_id"],
                "entity_id": record["entity_id"],
                "instance_id": record["instance_id"],
                "region_id": record["region_id"],
                "category": record["category"],
                "origin": record.get("origin", "legacy_caption"),
                "caption_supported": record.get("caption_supported", True),
                "caption_span": record["caption_span"],
                "initial_locator_query": record["initial_locator_query"],
                "final_referring_expression": record["final_referring_expression"],
                "visible_evidence": _visible_evidence(record),
                "bbox_supporting_grounders": "|".join(
                    record["bbox_supporting_grounders"]
                ),
                "bbox_grounder_support": record["bbox_grounder_support"],
                "expression_grounder_support": reground[
                    "expression_grounder_support"
                ],
                "reground_iou": reground["reground_iou"],
                "bbox_median_pairwise_iou": record["bbox_median_pairwise_iou"],
                "revision": record["revision"],
                "distractor_instance_ids": "|".join(
                    record["distractor_instance_ids"]
                ),
                "bbox_area_ratio": record["bbox_area_ratio"],
                "size_band": record["size_band"],
                "representative_method": record["representative_method"],
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "tight_crop": record["tight_crop_path"],
                "context_crop": record["context_crop_path"],
                "source_image": record["source_image"],
            }
        )
    _write_index(review_root / "index.csv", rows)
    print(f"Route B human review: {len(rows)} verified overlays -> {review_root}")
