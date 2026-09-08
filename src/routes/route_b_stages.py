"""Qwen-driven object-centric stages for the current Route B pipeline."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from src.models.qwen38_client import ModelContractError, Qwen38Client, QwenValidationError
from src.models.region_ocr import build_region_ocr
from src.regions.route_b import evaluate_expression_reground, validate_referring_expression
from src.regions.route_b_artifacts import ensure_artifacts
from src.regions.target_scope import (
    BBOX_FLAGS,
    SCOPE_FLAGS,
    bbox_accepted,
    evidence_from,
    repair_candidates,
    update_alignment,
    validate_evidence,
)
from src.routes.model_failures import report_model_failures, skipped_record
from src.routes.route_b_checkpoint import checkpointed, fingerprint, semantic, stamp
from src.schema import (
    EntityAlignmentDecision,
    EntityBBoxVerification,
    ExpressionVerification,
    ReferringExpressionDraft,
    VisualEntitySet,
)
from src.utils.geometry import iou
from src.utils.images import (
    create_labeled_crop_montage,
    open_rgb,
    save_numbered_bbox_overlay,
)
from src.utils.io import (
    JsonlWriter,
    failure_writer,
    prepare_output,
    read_jsonl,
    rewrite_jsonl_atomic,
)
from src.utils.progress import completed_with_progress
from src.utils.provenance import model_metadata


def _qwen_copy(image: Image.Image) -> Image.Image:
    """Preserve the decoded input pixels; model preprocessing is separate."""
    return image.convert("RGB").copy()


def _open_qwen_images(paths: list[str | Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        opened = open_rgb(path)
        try:
            images.append(_qwen_copy(opened))
        finally:
            opened.close()
    return images


def _close_images(images: list[Image.Image]) -> None:
    for image in images:
        image.close()




def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate.get(key)
        for key in (
            "region_id",
            "instance_id",
            "entity_id",
            "category",
            "bbox_xyxy",
            "grounder_support",
            "supporting_grounders",
            "median_pairwise_iou",
            "bbox_area_ratio",
            "size_band",
            "query_types",
            "grounding_queries",
            "verifier_overlay_path",
            "tight_crop_path",
            "context_crop_path",
        )
    }


def _parallel_write(
    *,
    items: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    output_path: Path,
    workers: int,
    failures_path: Path,
    route: str,
    stage: str,
    request_key: Callable[[dict[str, Any]], str],
) -> None:
    if not items:
        if not output_path.exists():
            rewrite_jsonl_atomic(output_path, [])
        return
    writer = JsonlWriter(output_path)
    invalid = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_item = {executor.submit(worker, item): item for item in items}
        for future in completed_with_progress(future_to_item, stage):
            item = future_to_item[future]
            request_id = request_key(item)
            try:
                writer.append(stamp(future.result()))
            except Exception as exc:
                failure_writer(
                    failures_path,
                    image_id=item["image_id"],
                    route=route,
                    stage=stage,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    request_id=request_id,
                )
                if isinstance(exc, QwenValidationError):
                    invalid += 1
                    writer.append(stamp(skipped_record(item, stage, exc)))
                    print(f"[{stage}] SKIP {request_id}: {exc}", flush=True)
                else:
                    for pending_future in future_to_item:
                        pending_future.cancel()
                    print(f"[{stage}] FATAL {request_id}: {exc}", flush=True)
                    raise
    report_model_failures(stage, invalid, len(items), output_path)


@checkpointed("manifest_rows")
def extract_caption_entities(
    *,
    manifest_rows: list[dict[str, Any]],
    output_path: Path,
    failures_path: Path,
    prompt_path: Path,
    qwen_config: dict[str, Any],
    config: dict[str, Any],
    workers: int,
    resume: bool,
    overwrite: bool,
) -> None:
    """Discover visual targets from the full image with caption as optional context."""
    prepare_output(output_path, overwrite, resume)
    existing = list(read_jsonl(output_path)) if resume and output_path.exists() else []
    # An empty proposal list is a completed result; do not retry to fill a quota.
    done = {item["image_id"] for item in existing if "entities" in item}
    pending = [item for item in manifest_rows if item["image_id"] not in done]
    prompt = prompt_path.read_text(encoding="utf-8")
    client = Qwen38Client({**qwen_config, "max_output_tokens":
                          int(config.get("entity_max_output_tokens", 12288))})
    provenance = model_metadata(qwen_config, {**config, "model": qwen_config}, "route_b_entity_v1")

    def worker(record: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        image = open_rgb(record["image_path"])
        qwen_image = _qwen_copy(image)
        image.close()
        try:
            result, raw = client.generate_json(
                prompt,
                [qwen_image],
                VisualEntitySet,
                extra_text=json.dumps({"caption_reference": record["caption"],
                                       "proposal_cap": config.get("max_entities_per_image", 30),
                                       "width": record["width"], "height": record["height"]}),
            )
        finally:
            qwen_image.close()
        entities = result.model_dump(mode="json")["entities"]
        entities = entities[: int(config.get("max_entities_per_image", 30))]
        caption_span_adjustments = []
        for rank, entity in enumerate(entities, 1):
            entity["rank"] = rank
            entity["source"] = "visual_proposal"
            span = entity["caption_span"]
            entity["caption_supported"] = bool(
                entity["caption_supported"] and span and span in record["caption"])
            if not entity["caption_supported"]:
                entity["caption_span"] = ""
            identity = {k: v for k, v in entity.items() if k not in {"entity_id", "rank"}}
            entity["entity_id"] = "v_" + fingerprint([record["image_id"], identity])[:20]
        # Exact duplicate semantic proposals are collapsed before any grounder work.
        entities = list({e["entity_id"]: e for e in entities}.values())
        for rank, entity in enumerate(entities, 1):
            entity["rank"] = rank
        validated = VisualEntitySet.model_validate({"entities": entities})
        return {
            "image_id": record["image_id"],
            "entities": validated.model_dump(mode="json")["entities"],
            "runtime_ms": (time.perf_counter() - started) * 1000,
            "metadata": {
                **provenance,
                "raw_response": raw,
                "caption_span_adjustments": caption_span_adjustments,
            },
        }

    _parallel_write(
        items=pending,
        worker=worker,
        output_path=output_path,
        workers=workers,
        failures_path=failures_path,
        route=config["route"],
        stage="visual_entity_proposal",
        request_key=lambda item: item["image_id"],
    )
    latest = {item["image_id"]: item for item in read_jsonl(output_path)}
    ordered = [latest[item["image_id"]] for item in manifest_rows]
    rewrite_jsonl_atomic(output_path, ordered)


@checkpointed("entity_rows")
def align_entities_to_instances(
    *,
    entity_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    output_path: Path,
    failures_path: Path,
    prompt_path: Path,
    output_root: Path,
    qwen_config: dict[str, Any],
    config: dict[str, Any],
    workers: int,
    resume: bool,
    overwrite: bool,
) -> None:
    """Map each proposal uniquely, retaining size-invalid peers as distractors."""
    from src.routes.route_b_discovery import peer_candidates
    prepare_output(output_path, overwrite, resume)
    existing = list(read_jsonl(output_path)) if resume and output_path.exists() else []
    done = {(item["image_id"], item["entity_id"]) for item in existing}
    candidate_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if not candidate.get("discovery_category_only"):
            candidate_map[(candidate["image_id"], candidate["entity_id"])].append(candidate)
    tasks = []
    for row in entity_rows:
        for entity in row.get("entities", []):
            key = (row["image_id"], entity["entity_id"])
            if key in done:
                continue
            tasks.append(
                {
                    "image_id": row["image_id"],
                    "entity_id": entity["entity_id"],
                    "entity": entity,
                    "candidates": candidate_map.get(key, []),
                }
            )
    client = Qwen38Client(qwen_config)
    prompt = prompt_path.read_text(encoding="utf-8")
    provenance = model_metadata(qwen_config, {**config, "model": qwen_config}, "route_b_align_v1")
    artifact_root = output_root / "entity_alignment" / "artifacts" / "route_b"

    def worker(task: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        image_id = task["image_id"]
        entity = task["entity"]
        source = manifest[image_id]
        candidates_for_entity = [c for c in task["candidates"] if c["size_pass"]]
        peers = peer_candidates(image_id, entity["category_query"], candidates, task["candidates"])
        cards = [_compact_candidate(item) for item in peers]
        if not candidates_for_entity:
            decision = EntityAlignmentDecision(
                entity_id=entity["entity_id"],
                target_instance_id=None,
                target_matches_caption=False,
                target_is_unique_match=False,
                reject_reason="no size-valid multi-grounder instance is available",
            )
            return {
                "image_id": image_id,
                "entity_id": entity["entity_id"],
                "entity_rank": entity["rank"],
                "entity": entity,
                "candidate_instances": cards,
                "eligible_target_instance_ids": [c["instance_id"] for c in candidates_for_entity],
                "selected_candidate": None,
                "distractor_instance_ids": [],
                "alignment": decision.model_dump(mode="json"),
                "accepted": False,
                "runtime_ms": (time.perf_counter() - started) * 1000,
                "metadata": provenance,
            }

        image = open_rgb(source["image_path"])
        stem = f"{image_id}_{entity['entity_id']}"
        numbered_path = artifact_root / "numbered" / f"{stem}.jpg"
        montage_path = artifact_root / "montages" / f"{stem}.jpg"
        save_numbered_bbox_overlay(image, cards, numbered_path)
        create_labeled_crop_montage(image, cards, montage_path)
        image.close()
        images = _open_qwen_images(
            [source["image_path"], numbered_path, montage_path]
        )
        by_id = {item["instance_id"]: item for item in candidates_for_entity}

        def validate_alignment(decision: EntityAlignmentDecision) -> None:
            if decision.entity_id != entity["entity_id"]:
                raise ModelContractError(f"entity_id must be exactly {entity['entity_id']}")
            if decision.target_instance_id is not None and decision.target_instance_id not in by_id:
                raise ModelContractError(
                    f"target_instance_id {decision.target_instance_id!r} is not eligible; "
                    f"choose only from {list(by_id)} or reject with null target and a reason"
                )
            if (
                not (decision.target_matches_caption and decision.target_is_unique_match)
                and decision.target_instance_id is not None
            ):
                raise ModelContractError("rejected alignment must have null target_instance_id")

        try:
            card = {
                "caption": source["caption"],
                "entity": entity,
                "candidate_instances": cards,
                "eligible_target_instance_ids": [c["instance_id"] for c in candidates_for_entity],
            }
            decision, raw = client.generate_json(
                prompt,
                images,
                EntityAlignmentDecision,
                extra_text="Alignment card:\n" + json.dumps(card, ensure_ascii=False),
                validate_result=validate_alignment,
            )
        finally:
            _close_images(images)
        accepted = (
            decision.target_matches_caption
            and decision.target_is_unique_match
            and decision.target_instance_id is not None
        )
        selected = by_id.get(decision.target_instance_id or "")
        distractors = [
            item["instance_id"]
            for item in cards
            if item["instance_id"] != decision.target_instance_id
        ]
        highlighted_path = None
        highlighted_montage_path = None
        if accepted and selected is not None:
            image = open_rgb(source["image_path"])
            highlighted_path = artifact_root / "target_numbered" / f"{stem}.jpg"
            highlighted_montage_path = artifact_root / "target_montages" / f"{stem}.jpg"
            save_numbered_bbox_overlay(
                image,
                cards,
                highlighted_path,
                highlight_id=decision.target_instance_id,
            )
            create_labeled_crop_montage(
                image,
                cards,
                highlighted_montage_path,
                highlight_id=decision.target_instance_id,
            )
            image.close()
        return {
            "image_id": image_id,
            "entity_id": entity["entity_id"],
            "entity_rank": entity["rank"],
            "entity": entity,
            "candidate_instances": cards,
            "selected_candidate": selected,
            "distractor_instance_ids": distractors,
            "alignment": decision.model_dump(mode="json"),
            "accepted": accepted,
            "numbered_overlay_path": str(numbered_path),
            "candidate_montage_path": str(montage_path),
            "target_numbered_overlay_path": (
                str(highlighted_path) if highlighted_path is not None else None
            ),
            "target_montage_path": (
                str(highlighted_montage_path)
                if highlighted_montage_path is not None
                else None
            ),
            "runtime_ms": (time.perf_counter() - started) * 1000,
            "metadata": {**provenance, "raw_response": raw},
        }

    _parallel_write(
        items=tasks,
        worker=worker,
        output_path=output_path,
        workers=workers,
        failures_path=failures_path,
        route=config["route"],
        stage="entity_instance_alignment",
        request_key=lambda item: f"{item['image_id']}:{item['entity_id']}",
    )


@checkpointed("alignments", accepted_only=True)
def verify_entity_bboxes(
    *,
    alignments: list[dict[str, Any]],
    output_path: Path,
    failures_path: Path,
    prompt_path: Path,
    qwen_config: dict[str, Any],
    config: dict[str, Any],
    workers: int,
    resume: bool,
    overwrite: bool,
) -> None:
    """Reject part boxes and truncated objects before expression writing."""
    prepare_output(output_path, overwrite, resume)
    existing = list(read_jsonl(output_path)) if resume and output_path.exists() else []
    done = {(item["image_id"], item["entity_id"]) for item in existing}
    pending = [
        item
        for item in alignments
        if item["accepted"] and (item["image_id"], item["entity_id"]) not in done
    ]
    client = Qwen38Client(qwen_config)
    prompt = prompt_path.read_text(encoding="utf-8")
    provenance = model_metadata(qwen_config, {**config, "model": qwen_config}, "route_b_bbox_v1")

    def worker(alignment: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        candidate = alignment["selected_candidate"]
        def assess(target):
            ensure_artifacts(target)
            paths = [target["verifier_overlay_path"],
                     target["tight_crop_path"], target["context_crop_path"]]
            is_repair = target["bbox_xyxy"] != candidate["bbox_xyxy"]
            if is_repair:
                paths.append(candidate["verifier_overlay_path"])
            images = _open_qwen_images(paths)
            card = {"entity": alignment["entity"],
                    "is_repair_attempt": is_repair,
                    "original_target_bbox": candidate["bbox_xyxy"],
                    "selected_candidate": {**_compact_candidate(target),
                                           "entity_id": alignment["entity_id"]},
                    "distractor_instance_ids": alignment["distractor_instance_ids"]}
            try:
                return client.generate_json(
                    prompt, images, EntityBBoxVerification,
                    extra_text="Entity bbox card:\n" + json.dumps(card, ensure_ascii=False),
                    expected_fields={"entity_id": alignment["entity_id"],
                                     "instance_id": target["instance_id"]},
                )
            finally:
                _close_images(images)

        decision, raw = assess(candidate)
        repair_audit = [{"bbox_xyxy": candidate["bbox_xyxy"],
                         "decision": decision.model_dump(mode="json"), "raw_response": raw}]
        if (decision.target_matches_entity and decision.bbox_covers_whole_entity
                and (not decision.bbox_is_tight or not decision.bbox_excludes_unnecessary_neighbors)):
            for alternative in repair_candidates(candidate, config["aggregation"]):
                alternative_decision, alternative_raw = assess(alternative)
                repair_audit.append({"bbox_xyxy": alternative["bbox_xyxy"],
                                     "decision": alternative_decision.model_dump(mode="json"),
                                     "raw_response": alternative_raw})
                if bbox_accepted(alternative_decision):
                    candidate, decision, raw = alternative, alternative_decision, alternative_raw
                    alignment = update_alignment(alignment, candidate)
                    break
        accepted = bbox_accepted(decision)
        return {
            **alignment,
            "target_evidence": evidence_from(decision),
            "bbox_repair_audit": repair_audit,
            "bbox_verification": decision.model_dump(mode="json"),
            "accepted": accepted,
            "bbox_verification_runtime_ms": (time.perf_counter() - started) * 1000,
            "bbox_verification_metadata": {**provenance, "raw_response": raw},
        }

    _parallel_write(
        items=pending,
        worker=worker,
        output_path=output_path,
        workers=workers,
        failures_path=failures_path,
        route=config["route"],
        stage="whole_entity_bbox_verification",
        request_key=lambda item: f"{item['image_id']}:{item['entity_id']}",
    )


@checkpointed("bbox_records", accepted_only=True)
def extract_region_text(
    *,
    bbox_records: list[dict[str, Any]],
    output_path: Path,
    failures_path: Path,
    prompt_path: Path,
    qwen_config: dict[str, Any],
    config: dict[str, Any],
    workers: int,
    resume: bool,
    overwrite: bool,
) -> None:
    """Run the configured target-associated OCR provider on accepted whole entities."""
    prepare_output(output_path, overwrite, resume)
    existing = list(read_jsonl(output_path)) if resume and output_path.exists() else []
    done = {(item["image_id"], item["entity_id"]) for item in existing}
    pending = [
        item
        for item in bbox_records
        if item["accepted"] and (item["image_id"], item["entity_id"]) not in done
    ]
    provider = build_region_ocr(
        str(config.get("ocr_backend", "qwen")),
        qwen_config=qwen_config,
        prompt_path=prompt_path,
    )
    provenance = model_metadata(qwen_config, {**config, "model": qwen_config}, "route_b_ocr_v1")

    def worker(record: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        candidate = record["selected_candidate"]
        images = _open_qwen_images(
            [
                candidate["verifier_overlay_path"],
                candidate["tight_crop_path"],
                candidate["context_crop_path"],
            ],
        )
        try:
            result, raw = provider.extract(
                images,
                entity_id=record["entity_id"],
                instance_id=candidate["instance_id"],
                caption_visible_text=record["entity"]["visible_text"],
            )
        finally:
            _close_images(images)
        if result.entity_id != record["entity_id"]:
            raise ValueError("OCR response changed entity_id")
        if result.instance_id != candidate["instance_id"]:
            raise ValueError("OCR response changed instance_id")
        return {
            "image_id": record["image_id"],
            "entity_id": record["entity_id"],
            "instance_id": candidate["instance_id"],
            "ocr": result.model_dump(mode="json"),
            "runtime_ms": (time.perf_counter() - started) * 1000,
            "metadata": {**provenance, "raw_response": raw},
        }

    _parallel_write(
        items=pending,
        worker=worker,
        output_path=output_path,
        workers=workers,
        failures_path=failures_path,
        route=config["route"],
        stage="target_region_ocr",
        request_key=lambda item: f"{item['image_id']}:{item['entity_id']}",
    )


def _expression_base(
    bbox_record: dict[str, Any], manifest: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    candidate = bbox_record["selected_candidate"]
    source = manifest[bbox_record["image_id"]]
    return {
        "origin": bbox_record["entity"].get("source", "visual_proposal"),
        "caption_supported": bbox_record["entity"].get("caption_supported", False),
        "target_fingerprint": fingerprint(semantic(bbox_record)),
        "image_id": bbox_record["image_id"],
        "entity_id": bbox_record["entity_id"],
        "entity_rank": int(bbox_record["entity_rank"]),
        "instance_id": candidate["instance_id"],
        "region_id": candidate["region_id"],
        "category": bbox_record["entity"]["category"],
        "scope": bbox_record["entity"]["scope"],
        "caption": source["caption"],
        "caption_span": bbox_record["entity"]["caption_span"],
        "initial_locator_query": bbox_record["entity"]["locator_query"],
        "entity": bbox_record["entity"],
        "bbox_xyxy": candidate["bbox_xyxy"],
        "image_width": int(candidate["image_width"]),
        "image_height": int(candidate["image_height"]),
        "source_image": candidate["source_image"],
        "bbox_grounder_support": int(candidate["grounder_support"]),
        "bbox_supporting_grounders": candidate["supporting_grounders"],
        "bbox_median_pairwise_iou": candidate["median_pairwise_iou"],
        "bbox_area_ratio": candidate["bbox_area_ratio"],
        "size_band": candidate["size_band"],
        "representative_method": candidate["representative_method"],
        "target_overlay_path": candidate["verifier_overlay_path"],
        "tight_crop_path": candidate["tight_crop_path"],
        "context_crop_path": candidate["context_crop_path"],
        "candidate_instances": bbox_record["candidate_instances"],
        "distractor_instance_ids": bbox_record["distractor_instance_ids"],
        "target_numbered_overlay_path": bbox_record["target_numbered_overlay_path"],
        "target_montage_path": bbox_record["target_montage_path"],
        "bbox_verification": bbox_record["bbox_verification"],
        "target_evidence": bbox_record["target_evidence"],
        "bbox_repair_audit": bbox_record["bbox_repair_audit"],
    }


def _materialize_draft(
    *,
    base: dict[str, Any],
    draft: ReferringExpressionDraft,
    revision: int,
    ocr_record: dict[str, Any],
    config: dict[str, Any],
    runtime_ms: float,
    raw_response: str,
    provenance: dict[str, Any],
    previous_expression_id: str | None = None,
) -> dict[str, Any]:
    if draft.entity_id != base["entity_id"]:
        raise ValueError("expression writer changed entity_id")
    if draft.instance_id != base["instance_id"]:
        raise ValueError("expression writer changed instance_id")
    payload = draft.model_dump(mode="json")
    surface_errors = validate_referring_expression(
        payload,
        category=base["category"],
        distractor_count=len(base["distractor_instance_ids"]),
        min_words=int(config.get("expression_min_words", 8)),
        max_words=int(config.get("expression_max_words", 30)),
        min_discriminative_cues=int(config.get("min_discriminative_cues", 2)),
    )
    base_expression_id = f"{base['region_id']}_expr"
    expression_id = f"{base_expression_id}_r{revision}"
    return {
        **base,
        "base_expression_id": base_expression_id,
        "expression_id": expression_id,
        "previous_expression_id": previous_expression_id,
        "revision": revision,
        "final_referring_expression": payload["expression"],
        "head_category": payload["head_category"],
        "used_attributes": payload["used_attributes"],
        "used_action": payload["used_action"],
        "used_relation": payload["used_relation"],
        "used_visible_text": payload["used_visible_text"],
        "comparison_basis": payload["comparison_basis"],
        "ocr_observations": ocr_record["ocr"]["observations"],
        "surface_validation": {
            "passed": not surface_errors,
            "errors": surface_errors,
        },
        "expression_runtime_ms": runtime_ms,
        "expression_metadata": {**provenance, "raw_response": raw_response},
    }


def _refresh_surface_validation(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    refreshed = []
    for record in records:
        if record.get("model_output_invalid"):
            refreshed.append(record)
            continue
        errors = validate_referring_expression(
            {
                "expression": record["final_referring_expression"],
                "head_category": record["head_category"],
                "used_attributes": record["used_attributes"],
                "used_action": record["used_action"],
                "used_relation": record["used_relation"],
                "used_visible_text": record["used_visible_text"],
                "comparison_basis": record["comparison_basis"],
            },
            category=record["category"],
            distractor_count=len(record["distractor_instance_ids"]),
            min_words=int(config.get("expression_min_words", 8)),
            max_words=int(config.get("expression_max_words", 30)),
            min_discriminative_cues=int(config.get("min_discriminative_cues", 2)),
        )
        refreshed.append(
            {
                **record,
                "surface_validation": {"passed": not errors, "errors": errors},
            }
        )
    return refreshed


@checkpointed("bbox_records", accepted_only=True)
def describe_grounded_entities(
    *,
    bbox_records: list[dict[str, Any]],
    ocr_records: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    output_path: Path,
    failures_path: Path,
    prompt_path: Path,
    qwen_config: dict[str, Any],
    config: dict[str, Any],
    workers: int,
    resume: bool,
    overwrite: bool,
) -> None:
    """Generate target-conditioned expressions after comparing same-category instances."""
    prepare_output(output_path, overwrite, resume)
    existing = list(read_jsonl(output_path)) if resume and output_path.exists() else []
    done = {(item["image_id"], item["entity_id"]) for item in existing}
    ocr_by_key = {
        (item["image_id"], item["entity_id"]): item for item in ocr_records
    }
    pending = [
        item
        for item in bbox_records
        if item["accepted"] and (item["image_id"], item["entity_id"]) not in done
    ]
    missing_ocr = [
        (item["image_id"], item["entity_id"])
        for item in pending
        if (item["image_id"], item["entity_id"]) not in ocr_by_key
    ]
    if missing_ocr:
        raise RuntimeError(f"OCR is incomplete for {len(missing_ocr)} accepted entities")
    client = Qwen38Client(qwen_config)
    prompt = prompt_path.read_text(encoding="utf-8")
    provenance = model_metadata(
        qwen_config, {**config, "model": qwen_config}, "route_b_expression_writer_v1"
    )

    def worker(record: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        base = _expression_base(record, manifest)
        ocr_record = ocr_by_key[(record["image_id"], record["entity_id"])]
        if ocr_record.get("model_output_invalid"):
            raise QwenValidationError("upstream OCR exhausted validation; expression skipped")
        paths = [
            base["source_image"],
            base["target_numbered_overlay_path"],
            base["target_overlay_path"],
            base["tight_crop_path"],
            base["context_crop_path"],
            base["target_montage_path"],
        ]
        images = _open_qwen_images(paths)
        card = {
            "caption": base["caption"],
            "entity": base["entity"],
            "target_instance_id": base["instance_id"],
            "candidate_instances": base["candidate_instances"],
            "distractor_instance_ids": base["distractor_instance_ids"],
            "verified_target_ocr": ocr_record["ocr"],
            "target_evidence": base["target_evidence"],
            "expression_constraints": {
                "min_words": int(config.get("expression_min_words", 8)),
                "max_words": int(config.get("expression_max_words", 30)),
                "min_discriminative_cues": int(
                    config.get("min_discriminative_cues", 2)
                ),
            },
        }
        try:
            draft, raw = client.generate_json(
                prompt,
                images,
                ReferringExpressionDraft,
                extra_text="Expression target card:\n" + json.dumps(card, ensure_ascii=False),
                expected_fields={"entity_id": base["entity_id"], "instance_id": base["instance_id"]},
                validate_result=lambda result: validate_evidence(
                    result, base["target_evidence"], ocr_record["ocr"]["observations"]),
            )
        finally:
            _close_images(images)
        return _materialize_draft(
            base=base,
            draft=draft,
            revision=0,
            ocr_record=ocr_record,
            config=config,
            runtime_ms=(time.perf_counter() - started) * 1000,
            raw_response=raw,
            provenance=provenance,
        )

    _parallel_write(
        items=pending,
        worker=worker,
        output_path=output_path,
        workers=workers,
        failures_path=failures_path,
        route=config["route"],
        stage="fine_grained_expression_generation",
        request_key=lambda item: f"{item['image_id']}:{item['entity_id']}",
    )
    rewrite_jsonl_atomic(
        output_path,
        _refresh_surface_validation(list(read_jsonl(output_path)), config),
    )


def _grounding_by_expression(
    raw_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        expression_id = record.get("metadata", {}).get("expression_id")
        if expression_id:
            indexed[str(expression_id)].append(record)
    return indexed


def _verify_expression_one(
    *,
    draft: dict[str, Any],
    raw_records: list[dict[str, Any]],
    client: Qwen38Client,
    prompt: str,
    output_root: Path,
    config: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    audit = evaluate_expression_reground(draft, raw_records, config["aggregation"])
    reground_candidates = [
        {"instance_id": f"G{index}", "bbox_xyxy": cluster["bbox_xyxy"]}
        for index, cluster in enumerate(audit["consensus_clusters"], 1)
    ]
    reground_overlay = (
        output_root
        / "expression_verification"
        / "artifacts"
        / "route_b"
        / "reground_overlays"
        / f"{draft['expression_id']}.jpg"
    )
    image = open_rgb(draft["source_image"])
    save_numbered_bbox_overlay(image, reground_candidates, reground_overlay)
    image.close()
    images = _open_qwen_images(
        [
            draft["source_image"],
            draft["target_numbered_overlay_path"],
            draft["target_overlay_path"],
            draft["tight_crop_path"],
            draft["context_crop_path"],
            reground_overlay,
        ],
    )
    card = {
        "entity": draft["entity"],
        "target_instance_id": draft["instance_id"],
        "candidate_instances": draft["candidate_instances"],
        "distractor_instance_ids": draft["distractor_instance_ids"],
        "expression": draft["final_referring_expression"],
        "target_evidence": draft["target_evidence"],
        "declared_expression_evidence": {
            "used_attributes": draft["used_attributes"],
            "used_action": draft["used_action"],
            "used_relation": draft["used_relation"],
            "used_visible_text": draft["used_visible_text"],
        },
        "deterministic_surface_validation": draft["surface_validation"],
        "deterministic_reground_audit": audit,
    }
    try:
        verification, raw = client.generate_json(
            prompt,
            images,
            ExpressionVerification,
            extra_text="Expression verification card:\n"
            + json.dumps(card, ensure_ascii=False),
            expected_fields={"entity_id": draft["entity_id"], "instance_id": draft["instance_id"],
                             "expression": draft["final_referring_expression"]},
        )
    finally:
        _close_images(images)
    if verification.entity_id != draft["entity_id"]:
        raise ValueError("expression verifier changed entity_id")
    if verification.instance_id != draft["instance_id"]:
        raise ValueError("expression verifier changed instance_id")
    if " ".join(verification.expression.split()) != draft["final_referring_expression"]:
        raise ValueError("expression verifier changed the expression under review")
    vlm_passed = all(
        (
            verification.target_matches,
            verification.describes_whole_entity,
            verification.attributes_visible,
            verification.grammatically_valid,
            verification.target_is_unique,
            verification.reground_matches_target,
        )
    )
    vlm_passed = vlm_passed and all(getattr(verification, key) for key in SCOPE_FLAGS)
    accepted = bool(draft["surface_validation"]["passed"] and audit["passed"] and vlm_passed)
    if not draft["surface_validation"]["passed"]:
        reject_reason = ",".join(draft["surface_validation"]["errors"])
    elif not audit["passed"]:
        reject_reason = audit["reject_reason"]
    elif not vlm_passed:
        reject_reason = verification.reject_reason
    else:
        reject_reason = None
    return {
        **draft,
        "reground_audit": audit,
        "reground_overlay_path": str(reground_overlay),
        "expression_verification": verification.model_dump(mode="json"),
        "accepted": accepted,
        "reject_reason": reject_reason,
        "expression_verification_runtime_ms": (time.perf_counter() - started) * 1000,
        "expression_verification_metadata": {**provenance, "raw_response": raw},
    }


@checkpointed("drafts")
def verify_generated_expressions(
    *,
    drafts: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    output_path: Path,
    failures_path: Path,
    prompt_path: Path,
    output_root: Path,
    qwen_config: dict[str, Any],
    config: dict[str, Any],
    workers: int,
    resume: bool,
    overwrite: bool,
) -> None:
    """Combine deterministic re-grounding with an independent visual-language audit."""
    drafts = _refresh_surface_validation(drafts, config)
    prepare_output(output_path, overwrite, resume)
    existing = list(read_jsonl(output_path)) if resume and output_path.exists() else []
    done = {item["expression_id"] for item in existing}
    pending = [item for item in drafts if item["expression_id"] not in done]
    grounding = _grounding_by_expression(raw_records)
    client = Qwen38Client(qwen_config)
    prompt = prompt_path.read_text(encoding="utf-8")
    provenance = model_metadata(
        qwen_config, {**config, "model": qwen_config}, "route_b_expression_verifier_v1"
    )

    def worker(draft: dict[str, Any]) -> dict[str, Any]:
        return _verify_expression_one(
            draft=draft,
            raw_records=grounding.get(draft["expression_id"], []),
            client=client,
            prompt=prompt,
            output_root=output_root,
            config=config,
            provenance=provenance,
        )

    _parallel_write(
        items=pending,
        worker=worker,
        output_path=output_path,
        workers=workers,
        failures_path=failures_path,
        route=config["route"],
        stage="expression_verification",
        request_key=lambda item: item["expression_id"],
    )


def refinement_eligible(record: dict[str, Any], max_rounds: int) -> bool:
    """Only revise a basically factual target description that remains weak or ambiguous."""
    if record["accepted"] or int(record["revision"]) >= max_rounds:
        return False
    verification = record["expression_verification"]
    if not verification.get("bbox_tight_and_complete", False):
        return False
    return bool(
        verification["target_matches"]
        and verification["describes_whole_entity"]
        and verification["attributes_visible"]
    )


def refine_expression_batch(
    *,
    previous_records: list[dict[str, Any]],
    output_path: Path,
    failures_path: Path,
    prompt_path: Path,
    qwen_config: dict[str, Any],
    config: dict[str, Any],
    workers: int,
) -> list[dict[str, Any]]:
    """Generate and persist one next-round expression for every eligible failed record."""
    client = Qwen38Client(qwen_config)
    prompt = prompt_path.read_text(encoding="utf-8")
    provenance = model_metadata(
        qwen_config, {**config, "model": qwen_config}, "route_b_expression_refiner_v1"
    )
    writer = JsonlWriter(output_path)
    generated: list[dict[str, Any]] = []
    errors: list[str] = []

    def worker(previous: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        images = _open_qwen_images(
            [
                previous["source_image"],
                previous["target_numbered_overlay_path"],
                previous["target_overlay_path"],
                previous["tight_crop_path"],
                previous["context_crop_path"],
                previous["target_montage_path"],
            ],
        )
        card = {
            "entity": previous["entity"],
            "target_instance_id": previous["instance_id"],
            "candidate_instances": previous["candidate_instances"],
            "distractor_instance_ids": previous["distractor_instance_ids"],
            "verified_target_ocr": previous["ocr_observations"],
            "target_evidence": previous["target_evidence"],
            "prior_expression": previous["final_referring_expression"],
            "prior_surface_validation": previous["surface_validation"],
            "prior_reground_audit": previous["reground_audit"],
            "prior_vlm_verification": previous["expression_verification"],
            "expression_constraints": {
                "min_words": int(config.get("expression_min_words", 8)),
                "max_words": int(config.get("expression_max_words", 30)),
                "min_discriminative_cues": int(
                    config.get("min_discriminative_cues", 2)
                ),
            },
        }
        try:
            draft, raw = client.generate_json(
                prompt,
                images,
                ReferringExpressionDraft,
                extra_text="Expression refinement card:\n"
                + json.dumps(card, ensure_ascii=False),
                expected_fields={"entity_id": previous["entity_id"],
                                 "instance_id": previous["instance_id"]},
                validate_result=lambda result: validate_evidence(
                    result, previous["target_evidence"], previous["ocr_observations"]),
            )
        finally:
            _close_images(images)
        base = {
            key: previous[key]
            for key in (
                "image_id",
                "entity_id",
                "entity_rank",
                "instance_id",
                "region_id",
                "category",
                "scope",
                "caption",
                "caption_span",
                "initial_locator_query",
                "entity",
                "bbox_xyxy",
                "image_width",
                "image_height",
                "source_image",
                "bbox_grounder_support",
                "bbox_supporting_grounders",
                "bbox_median_pairwise_iou",
                "bbox_area_ratio",
                "size_band",
                "representative_method",
                "target_overlay_path",
                "tight_crop_path",
                "context_crop_path",
                "candidate_instances",
                "distractor_instance_ids",
                "target_numbered_overlay_path",
                "target_montage_path",
                "bbox_verification",
                "target_evidence",
                "bbox_repair_audit",
            )
        }
        base.update({key: previous.get(key) for key in
                     ("origin", "caption_supported", "target_fingerprint")})
        base["refinement_root"] = (previous.get("refinement_root")
                                   or fingerprint(semantic(previous)))
        return _materialize_draft(
            base=base,
            draft=draft,
            revision=int(previous["revision"]) + 1,
            ocr_record={"ocr": {"observations": previous["ocr_observations"]}},
            config=config,
            runtime_ms=(time.perf_counter() - started) * 1000,
            raw_response=raw,
            provenance=provenance,
            previous_expression_id=previous["expression_id"],
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_record = {
            executor.submit(worker, previous): previous for previous in previous_records
        }
        for future in completed_with_progress(future_to_record, "expression_refinement"):
            previous = future_to_record[future]
            try:
                result = future.result()
                writer.append(result)
                generated.append(result)
            except Exception as exc:
                failure_writer(
                    failures_path,
                    image_id=previous["image_id"],
                    route=config["route"],
                    stage="expression_refinement",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    request_id=previous["expression_id"],
                )
                if isinstance(exc, QwenValidationError):
                    writer.append(skipped_record(previous, "expression_refinement", exc))
                    errors.append(previous["expression_id"])
                else:
                    for pending_future in future_to_record:
                        pending_future.cancel()
                    print(f"[expression_refinement] FATAL: {exc}", flush=True)
                    raise
    report_model_failures("expression_refinement", len(errors), len(previous_records), output_path)
    return generated


def final_expression_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten the final accepted audit into the review and downstream contract."""
    return {
        key: record[key]
        for key in (
            "image_id",
            "entity_id",
            "entity_rank",
            "instance_id",
            "region_id",
            "category",
            "scope",
            "caption_span",
            "initial_locator_query",
            "bbox_xyxy",
            "image_width",
            "image_height",
            "source_image",
            "bbox_grounder_support",
            "bbox_supporting_grounders",
            "bbox_median_pairwise_iou",
            "bbox_area_ratio",
            "size_band",
            "representative_method",
            "tight_crop_path",
            "context_crop_path",
            "distractor_instance_ids",
            "base_expression_id",
            "expression_id",
            "revision",
            "final_referring_expression",
            "head_category",
            "used_attributes",
            "used_action",
            "used_relation",
            "used_visible_text",
            "comparison_basis",
            "ocr_observations",
            "surface_validation",
            "reground_audit",
            "expression_verification",
        )
    } | {"status": "verified_unique_referring_expression",
         "bbox_verification": record["bbox_verification"],
         "target_evidence": record["target_evidence"],
         "bbox_repair_audit": record["bbox_repair_audit"],
         "origin": record.get("origin", "legacy_caption"),
         "caption_supported": record.get("caption_supported", True)}


def _final_record_quality(record: dict[str, Any]) -> tuple[float, ...]:
    """Rank duplicate target records by independently verified evidence strength."""
    audit = record["reground_audit"]
    visible_cues = (
        len(record.get("used_attributes", []))
        + len(record.get("used_visible_text", []))
        + int(bool(record.get("used_action")))
        + int(bool(record.get("used_relation")))
    )
    return (
        float(audit["expression_grounder_support"]),
        float(audit["reground_iou"]),
        float(record["bbox_grounder_support"]),
        float(record["bbox_median_pairwise_iou"]),
        float(visible_cues),
        -float(record["revision"]),
        -float(record["entity_rank"]),
    )


def _duplicate_target_components(
    candidates: list[dict[str, Any]], iou_threshold: float
) -> list[list[dict[str, Any]]]:
    """Group duplicate expressions or near-identical target boxes within one image."""
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    normalized = [
        item["final_referring_expression"].strip().casefold()
        for item in candidates
    ]
    for first in range(len(candidates)):
        first_box = tuple(candidates[first]["bbox_xyxy"])
        for second in range(first + 1, len(candidates)):
            same_expression = normalized[first] == normalized[second]
            same_target = (
                iou(first_box, tuple(candidates[second]["bbox_xyxy"]))
                >= iou_threshold
            )
            if same_expression or same_target:
                union(first, second)

    components: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(candidates):
        components[find(index)].append(record)
    return list(components.values())


def _deduplicate_final_records(
    records: list[dict[str, Any]], iou_threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one best expression for each same-image spatial or textual target.

    Upstream caption entities can independently rediscover the same physical object.
    A very high IoU threshold removes only near-identical target boxes. Identical
    referring expressions within one source image are also duplicates even if the
    selected boxes differ, because the expression would not uniquely select one.
    """
    by_image: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_image[record["image_id"]].append(record)

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for image_id in sorted(by_image):
        candidates = sorted(by_image[image_id], key=lambda item: item["region_id"])
        for component in _duplicate_target_components(candidates, iou_threshold):
            # Sorting by region ID first gives deterministic ascending tie-breaking.
            ranked = sorted(component, key=lambda item: item["region_id"])
            ranked.sort(key=_final_record_quality, reverse=True)
            winner = ranked[0]
            kept.append(winner)
            winner_expression = winner["final_referring_expression"].strip().casefold()
            for duplicate in ranked[1:]:
                overlap = iou(
                    tuple(winner["bbox_xyxy"]), tuple(duplicate["bbox_xyxy"])
                )
                same_expression = (
                    duplicate["final_referring_expression"].strip().casefold()
                    == winner_expression
                )
                rejected.append(
                    {
                        **duplicate,
                        "stage": "final_target_deduplication",
                        "reject_reason": (
                            "duplicate_referring_expression"
                            if same_expression
                            else "duplicate_target_region"
                        ),
                        "duplicate_of_region_id": winner["region_id"],
                        "duplicate_iou": overlap,
                    }
                )
    return kept, rejected


def materialize_final_route_b_outputs(
    *,
    initial_verifications: list[dict[str, Any]],
    refinement_verifications: list[dict[str, Any]],
    grounding_rejections: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    bbox_records: list[dict[str, Any]],
    verified_path: Path,
    rejected_path: Path,
    dedup_iou_threshold: float = 0.98,
    max_per_source_image: int = 30,
) -> None:
    """Select the latest attempt for each target and preserve every rejection stage."""
    latest: dict[str, dict[str, Any]] = {}
    for record in [*initial_verifications, *refinement_verifications]:
        key = record["base_expression_id"]
        previous = latest.get(key)
        if previous is None or int(record["revision"]) > int(previous["revision"]):
            latest[key] = record
    for key, record in list(latest.items()):
        if record["accepted"] and not (
            all(record.get("bbox_verification", {}).get(flag, False) for flag in BBOX_FLAGS)
            and all(record.get("expression_verification", {}).get(flag, False) for flag in SCOPE_FLAGS)
            and record.get("reground_audit", {}).get("geometry_contract") == "tight_target_iou_area_v1"
            and "target_evidence" in record
        ):
            latest[key] = {**record, "accepted": False,
                           "reject_reason": "target_scope_contract_missing_or_failed"}
    accepted_records = [record for record in latest.values() if record["accepted"]]
    accepted_records, dedup_rejections = _deduplicate_final_records(
        accepted_records, dedup_iou_threshold
    )
    capped = []
    per_source = defaultdict(list)
    for record in accepted_records:
        per_source[record["image_id"]].append(record)
    for records in per_source.values():
        ranked = sorted(records, key=lambda r: r["region_id"])
        ranked.sort(key=_final_record_quality, reverse=True)
        capped.extend(ranked[:max_per_source_image])
        dedup_rejections.extend({**r, "stage": "final_cap", "reject_reason": "per_source_cap"}
                                for r in ranked[max_per_source_image:])
    verified = [final_expression_record(record) for record in capped]
    rejected: list[dict[str, Any]] = list(grounding_rejections)
    rejected.extend(dedup_rejections)
    rejected.extend(
        {
            **record,
            "stage": "entity_instance_alignment",
            "reject_reason": record["alignment"]["reject_reason"],
        }
        for record in alignments
        if not record["accepted"]
    )
    rejected.extend(
        {
            **record,
            "stage": "whole_entity_bbox_verification",
            "reject_reason": record["bbox_verification"]["reject_reason"],
        }
        for record in bbox_records
        if not record["accepted"]
    )
    rejected.extend(
        {
            **record,
            "stage": "final_expression_verification",
            "reject_reason": record["reject_reason"],
        }
        for record in latest.values()
        if not record["accepted"]
    )
    verified.sort(key=lambda item: (item["image_id"], item["entity_rank"]))
    rejected.sort(
        key=lambda item: (
            str(item.get("image_id", "")),
            int(item.get("entity_rank", 0)),
            str(item.get("stage", "")),
        )
    )
    rewrite_jsonl_atomic(verified_path, verified)
    rewrite_jsonl_atomic(rejected_path, rejected)
    print(
        f"Route B finalization: {len(verified)} verified unique expressions; "
        f"{len(rejected)} rejected audit records"
    )
