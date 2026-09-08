"""Image artifact creation with atomic finalization."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.config import resolve_path
from src.utils.geometry import BBox


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(resolve_path(path)) as image:
        image.load()
        return image.convert("RGB")


def _atomic_image_save(image: Image.Image, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    try:
        image.save(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_crop(image: Image.Image, box: BBox, path: str | Path) -> None:
    integer_box = tuple(round(value) for value in box)
    _atomic_image_save(image.crop(integer_box), Path(path), quality=95)


def save_overlay(
    image: Image.Image, box: BBox, label: str, path: str | Path, color: str = "#ff2d55"
) -> None:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle(box, outline=color, width=max(2, round(min(image.size) / 300)))
    text_box = draw.textbbox((box[0], box[1]), label)
    draw.rectangle(text_box, fill=color)
    draw.text((box[0], box[1]), label, fill="white")
    _atomic_image_save(output, Path(path), quality=92)


def padded_box(box: BBox, width: int, height: int, padding: float = 0.20) -> BBox:
    """Expand a box by a fraction of its width and height on every side."""
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width), x2 + pad_x),
        min(float(height), y2 + pad_y),
    )


def save_bbox_only_overlay(
    image: Image.Image, box: BBox, path: str | Path, color: str = "#ff2d55"
) -> None:
    """Save an original-resolution image containing one unlabelled bbox."""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle(box, outline=color, width=max(4, round(min(image.size) / 260)))
    _atomic_image_save(output, Path(path), quality=94)


def save_referring_expression_overlay(
    image: Image.Image,
    box: BBox,
    expression: str,
    path: str | Path,
    color: str = "#ff2d55",
) -> None:
    """Save a review overlay whose top-left panel contains only the expression."""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font_size = max(20, round(min(output.size) / 55))
    font = _review_font(font_size)
    padding = max(10, round(font_size * 0.45))
    line_spacing = max(4, round(font_size * 0.2))
    panel_width = min(output.width - 2 * padding, max(round(output.width * 0.72), 480))
    lines = _wrap_text_pixels(draw, expression, font, panel_width - 2 * padding)
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3]
    panel_height = 2 * padding + len(lines) * line_height + (len(lines) - 1) * line_spacing
    draw.rounded_rectangle(
        (padding, padding, padding + panel_width, padding + panel_height),
        radius=max(8, padding // 2),
        fill=(0, 0, 0, 205),
    )
    y = 2 * padding
    for line in lines:
        draw.text((2 * padding, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_height + line_spacing
    rgb = Image.new("RGB", (1, 1), color).getpixel((0, 0))
    draw.rectangle(box, outline=(*rgb, 255), width=max(4, round(min(output.size) / 260)))
    _atomic_image_save(output, Path(path), quality=94)


def _review_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use a scalable, widely available font with a Pillow fallback."""
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _wrap_text_pixels(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> list[str]:
    """Wrap text using rendered pixel width instead of an approximate character count."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    binary = (np.asarray(mask, dtype=bool) * 255).astype(np.uint8)
    _atomic_image_save(Image.fromarray(binary, mode="L"), Path(path))


def save_numbered_bbox_overlay(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    path: str | Path,
    *,
    id_key: str = "instance_id",
    highlight_id: str | None = None,
) -> None:
    """Save a full-image instance map with stable labels and an optional target highlight."""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font = _review_font(max(18, round(min(output.size) / 60)))
    box_width = max(4, round(min(output.size) / 260))
    for index, candidate in enumerate(candidates, 1):
        identifier = str(candidate.get(id_key) or candidate.get("region_id") or index)
        is_target = highlight_id is not None and identifier == highlight_id
        color = (255, 45, 85, 255) if is_target else (0, 210, 255, 255)
        box = tuple(float(value) for value in candidate["bbox_xyxy"])
        draw.rectangle(box, outline=color, width=box_width + (2 if is_target else 0))
        tag = f"TARGET {identifier}" if is_target else identifier
        text_box = draw.textbbox((box[0], box[1]), tag, font=font)
        panel = (
            text_box[0] - 5,
            text_box[1] - 3,
            text_box[2] + 5,
            text_box[3] + 3,
        )
        draw.rectangle(panel, fill=(*color[:3], 225))
        draw.text((box[0], box[1]), tag, fill=(255, 255, 255, 255), font=font)
    _atomic_image_save(output, Path(path), quality=94)


def create_labeled_crop_montage(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    path: str | Path,
    *,
    id_key: str = "instance_id",
    highlight_id: str | None = None,
    cell_size: int = 360,
    columns: int = 4,
) -> None:
    """Save same-category candidate crops with readable, stable instance labels."""
    rows = max(1, (len(candidates) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_size, rows * cell_size), "#202020")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _review_font(max(18, round(cell_size / 16)))
    for index, candidate in enumerate(candidates):
        identifier = str(candidate.get(id_key) or candidate.get("region_id") or index + 1)
        is_target = highlight_id is not None and identifier == highlight_id
        color = (255, 45, 85, 255) if is_target else (0, 210, 255, 255)
        crop = image.crop(tuple(round(float(value)) for value in candidate["bbox_xyxy"]))
        crop.thumbnail((cell_size - 16, cell_size - 54), Image.Resampling.LANCZOS)
        cell_x = (index % columns) * cell_size
        cell_y = (index // columns) * cell_size
        x = cell_x + (cell_size - crop.width) // 2
        y = cell_y + 46 + (cell_size - 54 - crop.height) // 2
        canvas.paste(crop, (x, y))
        label = f"TARGET {identifier}" if is_target else identifier
        draw.rectangle(
            (cell_x, cell_y, cell_x + cell_size, cell_y + 42),
            fill=(*color[:3], 230),
        )
        draw.text((cell_x + 8, cell_y + 7), label, fill="white", font=font)
    _atomic_image_save(canvas, Path(path), quality=94)
