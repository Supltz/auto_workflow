"""Object-centric entity aggregation and referring-expression round-trip checks."""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from src.regions.consensus import (
    cluster_sort_key,
    cluster_stats,
    consensus_components,
    representative_box,
    size_assessment,
)
from src.regions.route_b_artifacts import plan_artifacts
from src.schema import GroundingRequest
from src.utils.geometry import BBox, area, containment, iou
from src.utils.io import read_jsonl, rewrite_jsonl_atomic


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return normalized or "entity"


def _request_suffix(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def entity_grounding_requests(
    entity_rows: Iterable[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    route: str,
) -> list[dict[str, Any]]:
    """Build one shared category query plus one discriminative locator query per entity."""
    requests: list[dict[str, Any]] = []
    for row in entity_rows:
        image_id = row["image_id"]
        source = manifest[image_id]
        entities = row.get("entities", [])
        category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in entities:
            category_groups[entity["category_query"].casefold()].append(entity)

        for grouped in category_groups.values():
            query = grouped[0]["category_query"]
            entity_ids = [entity["entity_id"] for entity in grouped]
            request = GroundingRequest(
                request_id=(
                    f"{route}:{image_id}:category:{_request_suffix(query.casefold())}"
                ),
                image_id=image_id,
                image_path=source["image_path"],
                phrase=query,
                rank=min(int(entity["rank"]) for entity in grouped),
                route=route,
                image_width=int(source["width"]),
                image_height=int(source["height"]),
                metadata={
                    "source_method": "visual_entity_grounding",
                    "image_sha256": source.get("image_sha256"),
                    "query_type": "category",
                    "entity_ids": entity_ids,
                    "categories": sorted({entity["category"] for entity in grouped}),
                },
            )
            requests.append(request.model_dump(mode="json"))

        for entity in entities:
            locator_query = entity["locator_query"]
            if locator_query.casefold() == entity["category_query"].casefold():
                continue
            request = GroundingRequest(
                request_id=(
                    f"{route}:{image_id}:{entity['entity_id']}:locator:"
                    f"{_request_suffix(locator_query.casefold())}"
                ),
                image_id=image_id,
                image_path=source["image_path"],
                phrase=locator_query,
                rank=int(entity["rank"]),
                route=route,
                image_width=int(source["width"]),
                image_height=int(source["height"]),
                metadata={
                    "source_method": "visual_entity_grounding",
                    "image_sha256": source.get("image_sha256"),
                    "query_type": "locator",
                    "entity_id": entity["entity_id"],
                    "entity_ids": [entity["entity_id"]],
                    "category": entity["category"],
                },
            )
            requests.append(request.model_dump(mode="json"))
    return requests


def expression_grounding_requests(
    drafts: Iterable[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    route: str,
    source_method: str,
) -> list[dict[str, Any]]:
    """Build stable three-grounder requests for generated or refined expressions."""
    requests = []
    for draft in drafts:
        source = manifest[draft["image_id"]]
        expression = draft["final_referring_expression"]
        request = GroundingRequest(
            request_id=(
                f"{route}:{draft['expression_id']}:"
                f"{_request_suffix(expression.casefold())}"
            ),
            image_id=draft["image_id"],
            image_path=source["image_path"],
            phrase=expression,
            rank=int(draft["entity_rank"]),
            route=route,
            image_width=int(source["width"]),
            image_height=int(source["height"]),
            metadata={
                "source_method": source_method,
                "image_sha256": source.get("image_sha256"),
                "expression_id": draft["expression_id"],
                "base_expression_id": draft["base_expression_id"],
                "entity_id": draft["entity_id"],
                "instance_id": draft["instance_id"],
                "revision": int(draft["revision"]),
                "original_bbox_xyxy": draft["bbox_xyxy"],
            },
        )
        requests.append(request.model_dump(mode="json"))
    return requests


def _record_priority(record: dict[str, Any]) -> tuple[float, ...]:
    metadata = record.get("metadata", {})
    return (
        float(metadata.get("query_type") == "locator"),
        float(record.get("grounding_score") is not None),
        float(record.get("grounding_score") or 0.0),
        float(record.get("mask_path") is not None),
    )


def deduplicate_grounder_records(
    records: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Suppress near-identical boxes from one grounder without merging real instances."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["grounder"])].append(record)
    kept: list[dict[str, Any]] = []
    for grounder_records in grouped.values():
        grounder_kept: list[dict[str, Any]] = []
        for record in sorted(grounder_records, key=_record_priority, reverse=True):
            box = tuple(record["bbox_xyxy"])
            if any(iou(box, tuple(other["bbox_xyxy"])) >= threshold for other in grounder_kept):
                continue
            grounder_kept.append(record)
        kept.extend(grounder_kept)
    return kept


def boxes_match(first: BBox, second: BBox, rules: dict[str, Any]) -> bool:
    """Apply Route B IoU/containment geometry without a grounder-identity constraint."""
    if iou(first, second) >= float(rules["iou_threshold"]):
        return True
    first_area, second_area = area(first), area(second)
    if min(first_area, second_area) <= 0:
        return False
    smaller, larger = (first, second) if first_area <= second_area else (second, first)
    return (
        containment(smaller, larger) >= float(rules["containment_threshold"])
        and max(first_area, second_area) / min(first_area, second_area)
        <= float(rules["max_area_ratio_between_boxes"])
    )


def _deduplicate_instances(
    instances: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for instance in sorted(instances, key=lambda item: item["_sort_key"], reverse=True):
        box = tuple(instance["bbox_xyxy"])
        if any(iou(box, tuple(other["bbox_xyxy"])) >= threshold for other in kept):
            continue
        kept.append(instance)
    return kept


def _aggregate_route_b_entities_serial(
    *,
    entities_path: Path,
    raw_grounding_path: Path,
    candidates_path: Path,
    rejections_path: Path,
    output_root: Path,
    manifest: dict[str, dict[str, Any]],
    selected_ids: set[str],
    config: dict[str, Any],
    overwrite: bool,
) -> None:
    """Preserve every multi-grounder entity instance instead of one winner per query."""
    started = time.monotonic()
    print("[aggregate] Loading grounding records...", flush=True)
    rules = config["aggregation"]
    records_by_entity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    category_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(raw_grounding_path):
        image_id = record["image_id"]
        if image_id not in selected_ids:
            continue
        metadata = record.get("metadata", {})
        if metadata.get("query_type") == "category":
            category_records[(image_id, record["phrase"].casefold())].append(record)
        entity_ids = metadata.get("entity_ids") or [metadata.get("entity_id")]
        for entity_id in entity_ids:
            if entity_id:
                records_by_entity[(image_id, str(entity_id))].append(record)

    artifact_root = output_root / "grounded_entities" / "artifacts" / "route_b"
    if overwrite and artifact_root.exists():
        shutil.rmtree(artifact_root)
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    image_cache: dict[str, Image.Image] = {}
    rows = [row for row in read_jsonl(entities_path) if row["image_id"] in selected_ids]
    completed_images = 0
    last_report = 0.0

    def progress(detail: str, *, force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if force or now - last_report >= 30:
            print(
                f"[aggregate] images={completed_images}/{len(rows)} completed "
                f"candidates={len(candidates)} rejected_records={len(rejections)} "
                f"elapsed={now - started:.0f}s {detail}",
                flush=True,
            )
            last_report = now

    progress("Starting consensus and visual artifacts", force=True)
    try:
        for entity_row in rows:
            image_id = entity_row["image_id"]
            if image_id not in selected_ids:
                continue
            source = manifest[image_id]
            entities = list(entity_row.get("entities", []))
            if config.get("candidate_promotion"):
                categories = {e["category_query"].casefold(): e for e in entities}
                for query, example in categories.items():
                    pool_id = "category_" + _request_suffix(query)
                    entities.append({**example, "entity_id": pool_id,
                                     "discovery_category_only": True})
                    records_by_entity[(image_id, pool_id)] = category_records[(image_id, query)]
            for entity_number, entity in enumerate(entities, 1):
                entity_id = entity["entity_id"]
                progress(f"image={image_id} entity={entity_number}/{len(entities)} consensus")
                raw_records = records_by_entity.get((image_id, entity_id), [])
                records = deduplicate_grounder_records(
                    raw_records,
                    float(rules.get("within_grounder_dedup_iou", 0.95)),
                )
                provisional: list[dict[str, Any]] = []
                unsupported = 0
                for cluster in consensus_components(records, rules):
                    stats = cluster_stats(cluster, rules)
                    if stats["grounder_support"] < int(rules["min_grounder_support"]):
                        unsupported += 1
                        continue
                    representative, box, method = representative_box(
                        cluster, int(source["width"]), int(source["height"]), rules
                    )
                    size = size_assessment(
                        box, int(source["width"]), int(source["height"]), rules
                    )
                    provisional.append(
                        {
                            "bbox_xyxy": box,
                            "representative_grounder": representative["grounder"],
                            "representative_method": method,
                            **stats,
                            **size,
                            "query_types": sorted(
                                {
                                    str(item.get("metadata", {}).get("query_type", "unknown"))
                                    for item in cluster
                                }
                            ),
                            "grounding_queries": sorted({item["phrase"] for item in cluster}),
                            "cluster_members": [
                                {
                                    "grounder": item["grounder"],
                                    "bbox_xyxy": item["bbox_xyxy"],
                                    "grounding_score": item.get("grounding_score"),
                                    "mask_path": item.get("mask_path"),
                                    "query_type": item.get("metadata", {}).get("query_type"),
                                    "request_id": item.get("metadata", {}).get("request_id"),
                                }
                                for item in cluster
                            ],
                            "_sort_key": cluster_sort_key(cluster, rules),
                        }
                    )

                provisional = _deduplicate_instances(
                    provisional, float(rules.get("instance_dedup_iou", 0.85))
                )
                provisional.sort(
                    key=lambda item: (
                        float(item["bbox_xyxy"][1]),
                        float(item["bbox_xyxy"][0]),
                        float(item["bbox_xyxy"][3]),
                        float(item["bbox_xyxy"][2]),
                    )
                )
                if not provisional:
                    rejections.append(
                        {
                            "image_id": image_id,
                            "entity_id": entity_id,
                            "entity_rank": entity["rank"],
                            "category": entity["category"],
                            "stage": "entity_consensus",
                            "reject_reason": "no_multi_grounder_entity_instance",
                            "grounder_detection_counts": dict(
                                Counter(item["grounder"] for item in raw_records)
                            ),
                            "unsupported_components": unsupported,
                        }
                    )
                    continue

                safe_entity = _safe_id(entity_id)
                for instance_index, item in enumerate(provisional, 1):
                    progress(
                        f"image={image_id} entity={entity_number}/{len(entities)} "
                        f"artifact={instance_index}/{len(provisional)}"
                    )
                    item.pop("_sort_key", None)
                    instance_id = f"{safe_entity}_i{instance_index:02d}"
                    region_id = f"{image_id}_{instance_id}"
                    candidate = {
                        "region_id": region_id,
                        "image_id": image_id,
                        "entity_id": entity_id,
                        "entity_rank": int(entity["rank"]),
                        "instance_id": instance_id,
                        "caption_span": entity["caption_span"],
                        "category": entity["category"],
                        "scope": entity["scope"],
                        "attributes": entity["attributes"],
                        "action": entity.get("action"),
                        "relations": entity["relations"],
                        "caption_visible_text": entity["visible_text"],
                        "category_query": entity["category_query"],
                        "discovery_category_only": entity.get("discovery_category_only", False),
                        "locator_query": entity["locator_query"],
                        "image_width": int(source["width"]),
                        "image_height": int(source["height"]),
                        "source_image": source["image_path"],
                        **item,
                    }
                    candidate.update(
                        {
                            "verifier_overlay_path": None,
                            "tight_crop_path": None,
                            "context_crop_path": None,
                        }
                    )
                    # Keep all peer geometry, but materialize target images only on demand.
                    plan_artifacts(candidate, artifact_root, float(rules["context_padding"]))
                    if not candidate["size_pass"]:
                        rejections.append(
                            {
                                "image_id": image_id,
                                "entity_id": entity_id,
                                "instance_id": instance_id,
                                "region_id": region_id,
                                "stage": "entity_size_filter",
                                "reject_reason": candidate["size_reject_reason"],
                                "bbox_xyxy": candidate["bbox_xyxy"],
                                "bbox_area_ratio": candidate["bbox_area_ratio"],
                            }
                        )
                    candidates.append(candidate)
            completed_images += 1
            progress(f"finished image={image_id}", force=True)
    finally:
        for image in image_cache.values():
            image.close()

    progress("Writing candidate/rejection JSONL...", force=True)
    rewrite_jsonl_atomic(candidates_path, candidates)
    rewrite_jsonl_atomic(rejections_path, rejections)
    print(
        f"Route B entity aggregation: {len(candidates)} consensus instances; "
        f"{sum(item['size_pass'] for item in candidates)} passed size filters; "
        f"{len(rejections)} rejected records; elapsed={time.monotonic() - started:.0f}s",
        flush=True,
    )


def aggregate_route_b_entities(**kwargs) -> None:
    from src.regions.route_b_aggregate_cache import aggregate_cached
    aggregate_cached(_aggregate_route_b_entities_serial, **kwargs)


def _consensus_cluster_records(
    records: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    records = deduplicate_grounder_records(
        records, float(rules.get("within_grounder_dedup_iou", 0.95))
    )
    clusters = []
    for cluster in consensus_components(records, rules):
        stats = cluster_stats(cluster, rules)
        if stats["grounder_support"] < int(rules["min_grounder_support"]):
            continue
        _, box, method = representative_box(cluster, width, height, rules)
        clusters.append(
            {
                "bbox_xyxy": box,
                "representative_method": method,
                **stats,
                "_sort_key": cluster_sort_key(cluster, rules),
            }
        )
    clusters = _deduplicate_instances(
        clusters, float(rules.get("instance_dedup_iou", 0.85))
    )
    for cluster in clusters:
        cluster.pop("_sort_key", None)
    return clusters


def evaluate_expression_reground(
    draft: dict[str, Any],
    records: list[dict[str, Any]],
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Require exactly one consensus result and require it to recover the target box."""
    clusters = _consensus_cluster_records(
        records,
        width=int(draft["image_width"]),
        height=int(draft["image_height"]),
        rules=rules,
    )
    target_box = tuple(draft["bbox_xyxy"])
    matching = [
        cluster
        for cluster in clusters
        if iou(tuple(cluster["bbox_xyxy"]), target_box) >= float(rules["iou_threshold"])
        and max(area(tuple(cluster["bbox_xyxy"])), area(target_box))
        / max(min(area(tuple(cluster["bbox_xyxy"])), area(target_box)), 1e-9)
        <= float(rules.get("reground_max_area_ratio", 1.25))
    ]
    best_match = max(
        matching,
        key=lambda item: (
            item["grounder_support"],
            iou(tuple(item["bbox_xyxy"]), target_box),
            item["median_pairwise_iou"],
        ),
        default=None,
    )
    if not clusters:
        reason = "no_multi_grounder_expression_result"
    elif not matching:
        reason = "expression_did_not_recover_target"
    elif len(clusters) > 1:
        reason = "expression_has_multiple_consensus_targets"
    else:
        reason = None
    passed = len(clusters) == 1 and best_match is not None
    confusable = [
        cluster for cluster in clusters if best_match is None or cluster is not best_match
    ]
    return {
        "passed": passed,
        "geometry_contract": "tight_target_iou_area_v1",
        "reject_reason": reason,
        "consensus_cluster_count": len(clusters),
        "target_cluster_count": len(matching),
        "expression_grounder_support": (
            int(best_match["grounder_support"])
            if best_match is not None
            else max((int(item["grounder_support"]) for item in clusters), default=0)
        ),
        "reground_iou": (
            iou(tuple(best_match["bbox_xyxy"]), target_box)
            if best_match is not None
            else 0.0
        ),
        "target_cluster": best_match,
        "confusable_clusters": confusable,
        "consensus_clusters": clusters,
    }


def validate_referring_expression(
    draft: dict[str, Any],
    *,
    category: str,
    distractor_count: int,
    min_words: int,
    max_words: int,
    min_discriminative_cues: int,
) -> list[str]:
    """Apply deterministic language checks before spending grounder compute."""
    expression = " ".join(draft["expression"].strip().split())
    lowered = expression.casefold()
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", expression)
    errors = []
    if not expression.startswith(("the ", "The ")):
        errors.append("expression_must_start_with_the")
    if len(words) < min_words or len(words) > max_words:
        errors.append("expression_word_count_out_of_range")
    if any(
        phrase in lowered
        for phrase in (
            "bounding box",
            "boxed ",
            "highlighted ",
            "region id",
            "candidate id",
            "in this image",
        )
    ):
        errors.append("expression_mentions_annotation_or_image")
    # A visually supported subtype is often more natural and more discriminative than
    # the extraction category (person -> runner, car -> SUV, boat -> riverboat). Head
    # compatibility is therefore judged visually instead of by literal token equality.
    cue_count = len(draft.get("used_attributes", []))
    cue_count += int(bool(draft.get("used_action")))
    cue_count += int(bool(draft.get("used_relation")))
    cue_count += len(draft.get("used_visible_text", []))
    if cue_count < min_discriminative_cues:
        errors.append("insufficient_visible_discriminative_cues")
    if distractor_count > 0 and not draft.get("comparison_basis"):
        errors.append("missing_same_category_comparison")
    for visible_text in draft.get("used_visible_text", []):
        if visible_text.casefold() not in lowered:
            errors.append("declared_visible_text_missing_from_expression")
            break
    return errors
