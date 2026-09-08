#!/usr/bin/env python3
"""Fast non-inference acceptance checks for the object-centric Route B."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from PIL import Image

from src.regions.route_b import (
    aggregate_route_b_entities,
    entity_grounding_requests,
    evaluate_expression_reground,
    validate_referring_expression,
)
from src.regions.target_scope import BBOX_FLAGS, SCOPE_FLAGS
from src.schema import (
    CaptionEntitySet,
    EntityAlignmentDecision,
    EntityBBoxVerification,
    ExpressionVerification,
    ReferringExpressionDraft,
    RegionOCRResult,
)
from src.utils.io import read_jsonl, rewrite_jsonl_atomic
from src.visualization.route_b_review import export_route_b_review


def _entity(entity_id: str, rank: int, locator: str) -> dict:
    return {
        "entity_id": entity_id,
        "rank": rank,
        "caption_span": locator,
        "category": "person",
        "scope": "whole_object",
        "attributes": ["brown striped shirt"],
        "action": "climbing",
        "relations": ["climbing the wooden ramp"],
        "visible_text": [],
        "category_query": "person",
        "locator_query": locator,
    }


def _raw(
    grounder: str,
    box: list[int],
    *,
    query_type: str = "category",
    entity_ids: list[str] | None = None,
) -> dict:
    return {
        "image_id": "example",
        "rank": 1,
        "phrase": "person",
        "grounder": grounder,
        "bbox_xyxy": box,
        "grounding_score": 0.9,
        "mask_path": None,
        "metadata": {
            "query_type": query_type,
            "entity_ids": entity_ids or ["person_01"],
            "request_id": f"{grounder}:{query_type}:{box[0]}",
            "detection_index": 0,
        },
    }


def main() -> None:
    locator = "the person in the brown striped shirt climbing the wooden ramp"
    entities = CaptionEntitySet.model_validate(
        {
            "entities": [
                _entity("person_01", 1, locator),
                _entity("person_02", 2, "the person in the white shirt"),
            ]
        }
    ).model_dump(mode="json")
    EntityAlignmentDecision.model_validate(
        {
            "entity_id": "person_01",
            "target_instance_id": "person_01_i01",
            "target_matches_caption": True,
            "target_is_unique_match": True,
            "reject_reason": None,
        }
    )
    EntityBBoxVerification.model_validate(
        {
            "entity_id": "person_01",
            "instance_id": "person_01_i01",
            "target_matches_entity": True,
            "bbox_covers_whole_entity": True,
            "crop_is_clear": True,
            "detail_is_sufficient": True,
            "caption_attributes_visible": True,
            "bbox_is_tight": True,
            "bbox_excludes_unnecessary_neighbors": True,
            "target_is_single_entity": True,
            "preserves_target_identity": True,
            "target_attributes": [], "target_actions": [], "reference_objects": [],
            "supported_relations": [], "uncertain_attributes": [],
            "reject_reason": None,
        }
    )
    RegionOCRResult.model_validate(
        {
            "entity_id": "person_01",
            "instance_id": "person_01_i01",
            "observations": [],
            "no_legible_text": True,
        }
    )
    expression_payload = {
        "entity_id": "person_01",
        "instance_id": "person_01_i01",
        "expression": locator,
        "head_category": "person",
        "used_attributes": ["brown striped shirt"],
        "used_action": "climbing",
        "used_relation": "climbing the wooden ramp",
        "used_visible_text": [],
        "comparison_basis": ["the other person is not climbing the ramp"],
    }
    ReferringExpressionDraft.model_validate(expression_payload)
    ExpressionVerification.model_validate(
        {
            "entity_id": "person_01",
            "instance_id": "person_01_i01",
            "expression": locator,
            "target_matches": True,
            "describes_whole_entity": True,
            "attributes_visible": True,
            "grammatically_valid": True,
            "target_is_unique": True,
            "reground_matches_target": True,
            "single_target_scope": True,
            "attributes_belong_to_target": True,
            "references_only_relational": True,
            "bbox_tight_and_complete": True,
            "confusable_instance_ids": [],
            "suggested_discriminator": None,
            "reject_reason": None,
        }
    )

    manifest = {
        "example": {
            "image_id": "example",
            "image_path": "unused.jpg",
            "width": 1000,
            "height": 1000,
        }
    }
    requests = entity_grounding_requests(
        [{"image_id": "example", **entities}], manifest, "B_caption"
    )
    assert sum(item["metadata"]["query_type"] == "category" for item in requests) == 1
    assert sum(item["metadata"]["query_type"] == "locator" for item in requests) == 2

    config = {
        "route": "B_caption",
        "aggregation": {
            "min_grounder_support": 2,
            "iou_threshold": 0.5,
            "containment_threshold": 0.8,
            "max_area_ratio_between_boxes": 3.0,
            "within_grounder_dedup_iou": 0.95,
            "instance_dedup_iou": 0.85,
            "min_short_side_px": 32,
            "min_area_ratio": 0.00001,
            "preferred_min_area_ratio": 0.001,
            "preferred_max_area_ratio": 0.10,
            "max_area_ratio": 0.10,
            "context_padding": 0.20,
        },
    }
    with tempfile.TemporaryDirectory(prefix="route_b_check_") as temporary:
        root = Path(temporary)
        image_path = root / "source.jpg"
        Image.new("RGB", (1000, 1000), "white").save(image_path)
        manifest["example"]["image_path"] = str(image_path)
        entities_path = root / "entities.jsonl"
        raw_path = root / "raw.jsonl"
        candidates_path = root / "candidates.jsonl"
        rejected_path = root / "rejected.jsonl"
        rewrite_jsonl_atomic(
            entities_path,
            [{"image_id": "example", "entities": [entities["entities"][0]]}],
        )
        raw = [
            _raw("rex_omni", [100, 100, 300, 400]),
            _raw("sam31", [102, 102, 302, 402]),
            _raw("groundingdino", [98, 101, 298, 401]),
            # Near-identical locator result from Rex must not inflate one grounder.
            _raw("rex_omni", [101, 101, 301, 401], query_type="locator"),
            _raw("rex_omni", [600, 120, 780, 410]),
            _raw("sam31", [602, 122, 782, 412]),
            _raw("groundingdino", [598, 118, 778, 408]),
        ]
        rewrite_jsonl_atomic(raw_path, raw)
        aggregate_route_b_entities(
            entities_path=entities_path,
            raw_grounding_path=raw_path,
            candidates_path=candidates_path,
            rejections_path=rejected_path,
            output_root=root / "outputs",
            manifest=manifest,
            selected_ids={"example"},
            config=config,
            overwrite=True,
        )
        candidates = list(read_jsonl(candidates_path))
        assert len(candidates) == 2, candidates
        assert all(item["grounder_support"] == 3 for item in candidates)
        assert all(item["size_pass"] for item in candidates)
        assert {item["instance_id"] for item in candidates} == {
            "person_01_i01",
            "person_01_i02",
        }

        draft = {
            **expression_payload,
            "image_width": 1000,
            "image_height": 1000,
            "bbox_xyxy": candidates[0]["bbox_xyxy"],
        }
        assert not validate_referring_expression(
            expression_payload,
            category="person",
            distractor_count=1,
            min_words=8,
            max_words=30,
            min_discriminative_cues=2,
        )
        simple = {**expression_payload, "expression": "the person", "used_attributes": []}
        assert validate_referring_expression(
            simple,
            category="person",
            distractor_count=1,
            min_words=8,
            max_words=30,
            min_discriminative_cues=2,
        )
        target_box = [round(value) for value in candidates[0]["bbox_xyxy"]]
        reground = [
            _raw("rex_omni", target_box),
            _raw("sam31", [value + 1 for value in target_box]),
            _raw("groundingdino", [value - 1 for value in target_box]),
        ]
        audit = evaluate_expression_reground(draft, reground, config["aggregation"])
        assert audit["passed"] is True
        ambiguous = evaluate_expression_reground(draft, raw, config["aggregation"])
        assert ambiguous["passed"] is False
        assert ambiguous["reject_reason"] == "expression_has_multiple_consensus_targets"

        candidate = candidates[0]
        verified = {
            "status": "verified_unique_referring_expression",
            "image_id": "example",
            "entity_id": "person_01",
            "instance_id": candidate["instance_id"],
            "region_id": candidate["region_id"],
            "category": "person",
            "caption_span": locator,
            "initial_locator_query": locator,
            "final_referring_expression": locator,
            "bbox_xyxy": candidate["bbox_xyxy"],
            "source_image": str(image_path),
            "bbox_supporting_grounders": candidate["supporting_grounders"],
            "bbox_grounder_support": candidate["grounder_support"],
            "bbox_median_pairwise_iou": candidate["median_pairwise_iou"],
            "bbox_area_ratio": candidate["bbox_area_ratio"],
            "size_band": candidate["size_band"],
            "representative_method": candidate["representative_method"],
            "tight_crop_path": candidate["tight_crop_path"],
            "context_crop_path": candidate["context_crop_path"],
            "distractor_instance_ids": ["person_01_i02"],
            "revision": 0,
            "used_attributes": ["brown striped shirt"],
            "used_action": "climbing",
            "used_relation": "climbing the wooden ramp",
            "used_visible_text": [],
            "reground_audit": audit,
            "bbox_verification": dict.fromkeys(BBOX_FLAGS, True),
            "expression_verification": dict.fromkeys(SCOPE_FLAGS, True),
            "target_evidence": {},
        }
        verified_path = root / "verified.jsonl"
        rewrite_jsonl_atomic(verified_path, [verified])
        review_root = root / "human_review" / "route_b"
        export_route_b_review(
            verified_path=verified_path,
            review_root=review_root,
            selected_ids={"example"},
            max_per_source_image=10,
            overwrite=True,
        )
        review_images = list(review_root.glob("*.jpg"))
        assert len(review_images) == 1
        with (review_root / "index.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["final_referring_expression"] == locator
    print("Route B object-centric acceptance checks: PASS")


if __name__ == "__main__":
    main()
