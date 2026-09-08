"""Pluggable target-region OCR interface used by Route B."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from src.models.qwen38_client import Qwen38Client
from src.schema import RegionOCRResult


class RegionOCR(ABC):
    @abstractmethod
    def extract(
        self,
        images: list[Image.Image],
        *,
        entity_id: str,
        instance_id: str,
        caption_visible_text: list[str],
    ) -> tuple[RegionOCRResult, str]:
        """Extract text visible on the target entity and return raw model output."""


class QwenRegionOCR(RegionOCR):
    """Use the already configured Qwen VLM as the default OCR provider."""

    def __init__(self, qwen_config: dict[str, Any], prompt_path: Path) -> None:
        self.client = Qwen38Client(qwen_config)
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def extract(
        self,
        images: list[Image.Image],
        *,
        entity_id: str,
        instance_id: str,
        caption_visible_text: list[str],
    ) -> tuple[RegionOCRResult, str]:
        context = {
            "entity_id": entity_id,
            "instance_id": instance_id,
            "caption_reported_visible_text_for_verification_only": caption_visible_text,
        }
        return self.client.generate_json(
            self.prompt,
            images,
            RegionOCRResult,
            extra_text="OCR target card:\n" + json.dumps(context, ensure_ascii=False),
            expected_fields={"entity_id": entity_id, "instance_id": instance_id},
        )


def build_region_ocr(
    backend: str,
    *,
    qwen_config: dict[str, Any],
    prompt_path: Path,
) -> RegionOCR:
    """Build an OCR provider while leaving room for a dedicated OCR fallback."""
    if backend == "qwen":
        return QwenRegionOCR(qwen_config, prompt_path)
    raise ValueError(f"unsupported Route B OCR backend: {backend}")
