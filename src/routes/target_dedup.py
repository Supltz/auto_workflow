"""Shared target identity checks for discovery and final selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models.qwen38_client import Qwen38Client
from src.routes.route_b_checkpoint import CHECK_ONLY, StagePending, fingerprint
from src.utils.geometry import area, iou
from src.utils.images import open_rgb, save_numbered_bbox_overlay
from src.utils.io import atomic_write_json


class TargetIdentityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["same_object", "different_objects", "uncertain"]
    reason: str = Field(min_length=1)


def target(record):
    candidate = record.get("selected_candidate") or record
    entity = record.get("entity") or {}
    return {
        "image_id": record["image_id"],
        "source_image": candidate.get("source_image", record.get("source_image")),
        "bbox_xyxy": list(candidate["bbox_xyxy"]),
        "category": str(entity.get("category_query") or candidate.get("category_query")
                        or record.get("category") or "").strip().casefold(),
        "scope": entity.get("scope", candidate.get("scope", record.get("scope"))),
    }


class TargetDeduplicator:
    def __init__(self, *, config=None, qwen_config=None, output_root=None):
        config = config or {}
        self.exact_iou = float(config.get("final_target_dedup_iou", 0.98))
        self.suspect_iou = float(config.get("target_dedup_suspect_iou", 0.75))
        self.min_area_ratio = float(config.get("target_dedup_min_area_ratio", 0.7))
        self.max_center_offset = float(config.get("target_dedup_max_center_offset", 0.2))
        if not 0 < self.suspect_iou <= self.exact_iou <= 1:
            raise ValueError("target dedup IoU thresholds must satisfy 0 < suspect <= exact <= 1")
        if not 0 < self.min_area_ratio <= 1 or self.max_center_offset < 0:
            raise ValueError("invalid target dedup area ratio or center offset")
        self.qwen_config = qwen_config
        self.root = Path(output_root) / "route_b" / "target_dedup" if output_root else None
        self.prompt_path = Path(config.get("target_dedup_prompt", "prompts/route_b_target_identity.txt"))
        self.memory = {}
        self.client = None

    def compare(self, first, second):
        a, b = target(first), target(second)
        first_text = str(first.get("final_referring_expression") or "").strip().casefold()
        second_text = str(second.get("final_referring_expression") or "").strip().casefold()
        if a["image_id"] == b["image_id"] and first_text and first_text == second_text:
            return {"same_object": True, "method": "exact_expression",
                    "reason": "duplicate_referring_expression",
                    "iou": iou(tuple(a["bbox_xyxy"]), tuple(b["bbox_xyxy"]))}
        overlap = iou(tuple(a["bbox_xyxy"]), tuple(b["bbox_xyxy"]))
        base = {"iou": overlap, "same_object": False, "method": "geometry"}
        if (a["image_id"] != b["image_id"] or not a["category"]
                or a["category"] != b["category"] or not a["scope"]
                or a["scope"] != b["scope"]):
            return {**base, "reason": "different_image_category_or_scope"}
        if overlap >= self.exact_iou:
            return {**base, "same_object": True, "reason": "near_identical_boxes"}
        boxes = [a["bbox_xyxy"], b["bbox_xyxy"]]
        areas = [area(tuple(box)) for box in boxes]
        if min(areas) <= 0 or overlap < self.suspect_iou:
            return {**base, "reason": "insufficient_overlap"}
        centers = [[(box[i] + box[i + 2]) / 2 for i in (0, 1)] for box in boxes]
        offset = max(abs(centers[0][i] - centers[1][i]) /
                     min(box[i + 2] - box[i] for box in boxes) for i in (0, 1))
        if min(areas) / max(areas) < self.min_area_ratio or offset > self.max_center_offset:
            return {**base, "reason": "different_extent_or_center"}
        if self.qwen_config is None or self.root is None:
            raise ValueError("suspected duplicate requires target identity verifier configuration")
        pair = sorted([a, b], key=fingerprint)
        prompt = self.prompt_path.read_text()
        source = Path(pair[0]["source_image"])
        stat = source.stat()
        signature = fingerprint({"contract": "target_identity_v1", "pair": pair,
                                 "source": [str(source.resolve()), stat.st_size, stat.st_mtime_ns],
                                 "prompt": prompt, "model": self.qwen_config})
        cache_path = self.root / "decisions" / f"{signature}.json"
        if signature in self.memory:
            return self.memory[signature]
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            decision = TargetIdentityDecision.model_validate(cached["decision"])
        else:
            if CHECK_ONLY.get():
                raise StagePending("target_identity")
            cards = [{"instance_id": str(i), "bbox_xyxy": item["bbox_xyxy"]}
                     for i, item in enumerate(pair, 1)]
            overlay = self.root / "overlays" / f"{signature}.jpg"
            with open_rgb(source) as original:
                save_numbered_bbox_overlay(original, cards, overlay, highlight_id="1")
                # A shared context crop preserves the spatial relationship for tiny targets.
                x1 = min(box[0] for box in boxes)
                y1 = min(box[1] for box in boxes)
                x2 = max(box[2] for box in boxes)
                y2 = max(box[3] for box in boxes)
                pad = max(x2 - x1, y2 - y1) * 0.5
                crop_box = (max(0, x1-pad), max(0, y1-pad),
                            min(original.width, x2+pad), min(original.height, y2+pad))
                with open_rgb(overlay) as numbered, original.crop(crop_box) as context:
                    if self.client is None:
                        self.client = Qwen38Client(self.qwen_config)
                    decision, raw = self.client.generate_json(
                        prompt, [original, numbered, context], TargetIdentityDecision,
                        extra_text=json.dumps({"targets": pair, "context_crop_xyxy": crop_box}),
                    )
            atomic_write_json(cache_path, {"signature": signature, "pair": pair,
                                          "decision": decision.model_dump(), "raw_response": raw})
            print(f"[target_dedup] image={a['image_id']} iou={overlap:.4f} "
                  f"decision={decision.decision}", flush=True)
        result = {**base, "same_object": decision.decision == "same_object",
                  "method": "vlm", "decision": decision.decision,
                  "reason": decision.reason, "audit_path": str(cache_path)}
        self.memory[signature] = result
        return result
