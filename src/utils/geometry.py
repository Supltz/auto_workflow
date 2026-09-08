"""Geometry operations use inclusive-exclusive XYXY pixel coordinates."""

from __future__ import annotations

from collections.abc import Iterable

BBox = tuple[float, float, float, float]


def clip_box(box: Iterable[float], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        max(0.0, min(x1, float(width))),
        max(0.0, min(y1, float(height))),
        max(0.0, min(x2, float(width))),
        max(0.0, min(y2, float(height))),
    )


def valid_box(box: BBox, width: int, height: int, min_side: float = 1.0) -> bool:
    x1, y1, x2, y2 = box
    return (
        0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
        and x2 - x1 >= min_side
        and y2 - y1 >= min_side
    )


def area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def area_ratio(box: BBox, width: int, height: int) -> float:
    return area(box) / float(width * height)


def iou(first: BBox, second: BBox) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    union = area(first) + area(second) - intersection
    return intersection / union if union > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    intersection = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0])) * max(
        0.0, min(inner[3], outer[3]) - max(inner[1], outer[1])
    )
    denominator = area(inner)
    return intersection / denominator if denominator > 0 else 0.0


def normalized_xywh_to_xyxy(box: Iterable[float], width: int, height: int) -> BBox:
    x, y, box_width, box_height = (float(value) for value in box)
    return (x * width, y * height, (x + box_width) * width, (y + box_height) * height)
