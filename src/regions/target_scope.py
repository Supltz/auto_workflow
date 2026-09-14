"""Category-independent target scope checks and bounded detector-backed repair."""

from copy import deepcopy
from pathlib import Path
import os

from src.models.qwen38_client import ModelContractError
from src.regions.consensus import _mask_tight_box, cluster_stats, size_assessment
from src.regions.route_b_artifacts import plan_artifacts
from src.utils.geometry import area, containment, iou
from src.utils.images import create_labeled_crop_montage, open_rgb, save_numbered_bbox_overlay

BBOX_FLAGS = ("target_matches_entity", "bbox_covers_whole_entity", "crop_is_clear",
              "detail_is_sufficient", "caption_attributes_visible", "bbox_is_tight",
              "bbox_excludes_unnecessary_neighbors", "target_is_single_entity",
              "preserves_target_identity")
SCOPE_FLAGS = ("single_target_scope", "attributes_belong_to_target",
               "references_only_relational", "bbox_tight_and_complete")


def bbox_accepted(decision):
    return all(getattr(decision, key) for key in BBOX_FLAGS)


def scope_verified(record):
    return (
        all(record.get("bbox_verification", {}).get(key, False) for key in BBOX_FLAGS)
        and all(record.get("expression_verification", {}).get(key, False) for key in SCOPE_FLAGS)
        and record.get("reground_audit", {}).get("geometry_contract") == "tight_target_iou_area_v1"
        and "target_evidence" in record
    )


def evidence_from(decision):
    return {key: getattr(decision, key) for key in (
        "target_attributes", "target_actions", "reference_objects",
        "supported_relations", "uncertain_attributes")}


def validate_evidence(draft, evidence, ocr):
    """Reject unapproved declared facts; full phrase ownership is checked by the verifier."""
    for attribute in draft.used_attributes:
        if attribute not in evidence["target_attributes"]:
            raise ModelContractError(f"used_attributes entry is not approved target evidence: {attribute}")
    if draft.used_action and draft.used_action not in evidence["target_actions"]:
        raise ModelContractError("used_action must come from target_actions")
    if draft.used_relation and draft.used_relation not in evidence["supported_relations"]:
        raise ModelContractError("used_relation must come from supported_relations")
    texts = {str(r[k]).casefold() for r in ocr for k in ("text", "normalized_text") if k in r}
    if any(text.casefold() not in texts for text in draft.used_visible_text):
        raise ModelContractError("used_visible_text must come from verified target OCR")


def repair_candidates(candidate, rules, limit=2):
    """Only existing detector boxes/mask envelopes, never VLM coordinates or unions."""
    original = tuple(candidate["bbox_xyxy"])
    width, height = candidate["image_width"], candidate["image_height"]
    members = candidate.get("cluster_members", [])
    options = []
    for member in members:
        options.append((tuple(member["bbox_xyxy"]), member["grounder"], "detector_tight_repair"))
        if member.get("mask_path"):
            box = _mask_tight_box(member["mask_path"], width, height)
            if box:
                options.append((box, member["grounder"], "mask_tight_repair"))
    result = []
    seen = {original}
    for box, grounder, method in options:
        if box in seen:
            continue
        seen.add(box)
        if area(box) >= area(original) or containment(box, original) < 0.98:
            continue
        assessment = size_assessment(box, width, height, rules)
        support = [m for m in members if iou(box, tuple(m["bbox_xyxy"])) >= rules["iou_threshold"]]
        if not assessment["size_pass"] or len({m["grounder"] for m in support}) < rules["min_grounder_support"]:
            continue
        repaired = {**candidate, **cluster_stats(support, rules), **assessment,
                    "bbox_xyxy": list(box), "cluster_members": support,
                    "representative_grounder": grounder, "representative_method": method}
        # Resumed candidates can belong to immutable, read-only asset shards.
        # New repairs belong to the current stage's output, not the old shard.
        output_root = os.environ.get("OPD_STAGE_OUTPUT_ROOT")
        if output_root:
            root = Path(output_root)
            if not root.is_absolute():
                raise ValueError("OPD_STAGE_OUTPUT_ROOT must be absolute")
            root = root / "grounded_entities/artifacts/route_b/repairs"
        else:
            root = Path(candidate["verifier_overlay_path"]).parent.parent / "repairs"
        plan_artifacts(repaired, root, float(rules["context_padding"]))
        result.append(repaired)
    result.sort(key=lambda r: (-r["grounder_support"], area(tuple(r["bbox_xyxy"]))))
    return result[:max(0, min(limit, 2))]


def update_alignment(alignment, candidate):
    """Rebuild target views after accepting new geometry; retain query identity."""
    result = deepcopy(alignment)
    result["selected_candidate"] = candidate
    for card in result["candidate_instances"]:
        if card["instance_id"] == candidate["instance_id"]:
            for key in list(card):
                if key in candidate:
                    card[key] = candidate[key]
    image = open_rgb(candidate["source_image"])
    root = Path(candidate["verifier_overlay_path"]).parent.parent / "aligned"
    stem = candidate["shared_instance_id"]
    numbered, montage = root / f"{stem}_numbered.jpg", root / f"{stem}_montage.jpg"
    try:
        save_numbered_bbox_overlay(image, result["candidate_instances"], numbered,
                                  highlight_id=candidate["instance_id"])
        create_labeled_crop_montage(image, result["candidate_instances"], montage,
                                   highlight_id=candidate["instance_id"])
    finally:
        image.close()
    result.update(target_numbered_overlay_path=str(numbered), target_montage_path=str(montage))
    return result
