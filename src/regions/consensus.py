"""Multi-grounder consensus and size filtering used by Route B."""

from __future__ import annotations

import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.utils.geometry import BBox, area, area_ratio, containment, iou, valid_box


def cross_grounder_match(
    first: dict[str, Any], second: dict[str, Any], rules: dict[str, Any]
) -> bool:
    """Return whether detections from different grounders represent one instance."""
    if first["grounder"] == second["grounder"]:
        return False
    first_box = tuple(first["bbox_xyxy"])
    second_box = tuple(second["bbox_xyxy"])
    if iou(first_box, second_box) >= float(rules["iou_threshold"]):
        return True
    first_area, second_area = area(first_box), area(second_box)
    if min(first_area, second_area) <= 0:
        return False
    smaller, larger = (
        (first_box, second_box)
        if first_area <= second_area
        else (second_box, first_box)
    )
    return (
        containment(smaller, larger) >= float(rules["containment_threshold"])
        and max(first_area, second_area) / min(first_area, second_area)
        <= float(rules["max_area_ratio_between_boxes"])
    )


def consensus_components(
    records: list[dict[str, Any]], rules: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    """Build transitive components from pairwise cross-grounder matches."""
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first, second in combinations(range(len(records)), 2):
        if cross_grounder_match(records[first], records[second], rules):
            union(first, second)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[find(index)].append(record)
    return list(grouped.values())


def cluster_stats(
    cluster: list[dict[str, Any]], rules: dict[str, Any]
) -> dict[str, Any]:
    cross_pairs = [
        (first, second)
        for first, second in combinations(cluster, 2)
        if first["grounder"] != second["grounder"]
    ]
    pair_ious = [
        iou(tuple(first["bbox_xyxy"]), tuple(second["bbox_xyxy"]))
        for first, second in cross_pairs
    ]
    matched_pairs = sum(
        cross_grounder_match(first, second, rules)
        for first, second in cross_pairs
    )
    scores = [
        float(record["grounding_score"])
        for record in cluster
        if record.get("grounding_score") is not None
    ]
    return {
        "supporting_grounders": sorted({record["grounder"] for record in cluster}),
        "grounder_support": len({record["grounder"] for record in cluster}),
        "matched_cross_grounder_pairs": matched_pairs,
        "median_pairwise_iou": statistics.median(pair_ious) if pair_ious else 0.0,
        "mean_model_score": statistics.mean(scores) if scores else None,
    }


def cluster_sort_key(
    cluster: list[dict[str, Any]], rules: dict[str, Any]
) -> tuple[float, ...]:
    stats = cluster_stats(cluster, rules)
    return (
        float(stats["grounder_support"]),
        float(stats["matched_cross_grounder_pairs"]),
        float(stats["median_pairwise_iou"]),
        float(stats["mean_model_score"] or 0.0),
        -float(len(cluster)),
    )


def _mask_tight_box(mask_path: str | None, width: int, height: int) -> BBox | None:
    if not mask_path or not Path(mask_path).exists():
        return None
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L")) > 0
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    box = (
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    )
    return box if valid_box(box, width, height) else None


def _mean_iou_to_other_grounders(
    box: BBox, grounder: str, cluster: list[dict[str, Any]]
) -> float:
    values = [
        iou(box, tuple(other["bbox_xyxy"]))
        for other in cluster
        if other["grounder"] != grounder
    ]
    return statistics.mean(values) if values else 0.0


def representative_box(
    cluster: list[dict[str, Any]],
    width: int,
    height: int,
    rules: dict[str, Any],
) -> tuple[dict[str, Any], BBox, str]:
    """Prefer a supported SAM mask-tight box, otherwise select a detection medoid."""
    sam_options = []
    for record in cluster:
        if record["grounder"] != "sam31":
            continue
        tight_box = _mask_tight_box(record.get("mask_path"), width, height)
        if tight_box is None:
            continue
        other_grounders = [item for item in cluster if item["grounder"] != "sam31"]
        if any(
            cross_grounder_match({**record, "bbox_xyxy": tight_box}, other, rules)
            for other in other_grounders
        ):
            sam_options.append(
                (
                    _mean_iou_to_other_grounders(tight_box, "sam31", cluster),
                    record,
                    tight_box,
                )
            )
    if sam_options:
        _, record, box = max(sam_options, key=lambda item: item[0])
        return record, box, "sam31_mask_tight"

    record = max(
        cluster,
        key=lambda item: (
            _mean_iou_to_other_grounders(
                tuple(item["bbox_xyxy"]), item["grounder"], cluster
            ),
            item.get("grounding_score") is not None,
            float(item.get("grounding_score") or 0.0),
        ),
    )
    return record, tuple(record["bbox_xyxy"]), "detection_medoid"


def size_assessment(
    box: BBox, width: int, height: int, rules: dict[str, Any]
) -> dict[str, Any]:
    """Apply area limits and size labels; short-side length is diagnostic only."""
    shortest_side = min(box[2] - box[0], box[3] - box[1])
    ratio = area_ratio(box, width, height)
    min_area_ratio = float(rules["min_area_ratio"])
    max_area_ratio = float(rules["max_area_ratio"])

    def threshold_label(value: float) -> str:
        return f"{value:g}".replace(".", "_")

    reject_reason = None
    if ratio < min_area_ratio:
        reject_reason = (
            f"bbox_area_below_{threshold_label(min_area_ratio * 100)}_percent"
        )
    elif ratio > max_area_ratio:
        reject_reason = (
            f"bbox_area_above_{threshold_label(max_area_ratio * 100)}_percent"
        )

    if ratio < float(rules["preferred_min_area_ratio"]):
        size_band = "borderline_small"
    elif ratio <= float(rules["preferred_max_area_ratio"]):
        size_band = "preferred"
    else:
        size_band = "large_acceptable" if reject_reason is None else "too_large"
    return {
        "bbox_short_side_px": shortest_side,
        "bbox_area_ratio": ratio,
        "size_band": size_band,
        "size_pass": reject_reason is None,
        "size_reject_reason": reject_reason,
    }
