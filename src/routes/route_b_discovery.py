"""Bounded second-layer discovery from existing category consensus only."""

from __future__ import annotations

import json
from collections import defaultdict

from src.models.qwen38_client import ModelContractError, Qwen38Client
from src.regions.route_b_artifacts import ensure_artifacts
from src.routes.route_b_checkpoint import checkpointed, fingerprint
from src.routes.target_dedup import TargetDeduplicator
from src.schema import CandidatePromotion
from src.utils.geometry import iou
from src.utils.images import create_labeled_crop_montage, open_rgb, save_numbered_bbox_overlay
from src.utils.io import read_jsonl


def peer_candidates(image_id, query, candidates, preferred=()):
    """All same-category consensus peers, including size-ineligible distractors."""
    peers = [
        c
        for c in candidates
        if c["image_id"] == image_id and c["category_query"].casefold() == query.casefold()
    ]
    result = []
    for candidate in [*preferred, *sorted(peers, key=lambda c: c["instance_id"])]:
        if not any(
            iou(tuple(candidate["bbox_xyxy"]), tuple(other["bbox_xyxy"])) >= 0.85
            for other in result
        ):
            result.append(candidate)
    return result


def unique_alignments(records, threshold=0.98, *, deduplicator=None):
    """Keep one target before bbox/OCR/writer; audit duplicate alignments as rejected."""
    deduplicator = deduplicator or TargetDeduplicator(config={"final_target_dedup_iou": threshold})
    kept = []
    result = []

    def quality(record):
        candidate = record.get("selected_candidate") or {}
        return (
            -candidate.get("grounder_support", 0),
            -candidate.get("median_pairwise_iou", 0),
            record["entity_id"],
        )

    for record in sorted(records, key=quality):
        duplicate = (
            next(
                (
                    other
                    for other in kept
                    if deduplicator.compare(other, record)["same_object"]
                ),
                None,
            )
            if record["accepted"]
            else None
        )
        if duplicate:
            record = {
                **record,
                "accepted": False,
                "duplicate_of_region_id": duplicate["selected_candidate"]["region_id"],
                "target_dedup_audit": deduplicator.compare(duplicate, record),
                "alignment": {
                    **record["alignment"],
                    "reject_reason": f"duplicate target of {duplicate['entity_id']}",
                },
            }
        elif record["accepted"]:
            kept.append(record)
        result.append(record)
    return result


def promotion_tasks(candidates, alignments, config, *, deduplicator=None):
    """Never use locator-only consensus or add a new category universe."""
    deduplicator = deduplicator or TargetDeduplicator(config=config)
    matched = [a for a in alignments if a["accepted"]]
    by_image = defaultdict(list)
    for c in candidates:
        if not c.get("discovery_category_only") or not c["size_pass"]:
            continue
        if c["grounder_support"] < config["aggregation"]["min_grounder_support"]:
            continue
        if any(
            deduplicator.compare(a, c)["same_object"]
            for a in matched
        ):
            continue
        by_image[c["image_id"]].append(c)
    tasks = []
    for image_id, options in sorted(by_image.items()):
        selected = []
        for c in sorted(
            options,
            key=lambda c: (-c["grounder_support"], -c["median_pairwise_iou"], c["region_id"]),
        ):
            if any(deduplicator.compare(c, x)["same_object"] for x in selected):
                continue
            selected.append(c)
        for rank, c in enumerate(selected[: int(config.get("max_promotions_per_image", 16))], 1):
            entity_id = "d_" + fingerprint([image_id, c["category_query"], c["bbox_xyxy"]])[:20]
            tasks.append(
                {
                    "image_id": image_id,
                    "entity_id": entity_id,
                    "rank": int(config.get("max_entities_per_image", 30)) + rank,
                    "candidate": c,
                    "peers": peer_candidates(image_id, c["category_query"], candidates, [c]),
                }
            )
    return tasks


@checkpointed("tasks")
def _promote_tasks(
    *,
    tasks,
    output_path,
    failures_path,
    prompt_path,
    output_root,
    manifest,
    qwen_config,
    config,
    workers,
    resume,
    overwrite,
):
    from src.routes.route_b_stages import (
        _close_images,
        _compact_candidate,
        _open_qwen_images,
        _parallel_write,
    )
    from src.utils.io import prepare_output

    prepare_output(output_path, overwrite, resume)
    done = (
        {(r["image_id"], r["entity_id"]) for r in read_jsonl(output_path)}
        if output_path.exists()
        else set()
    )
    pending = [t for t in tasks if (t["image_id"], t["entity_id"]) not in done]
    client = Qwen38Client(qwen_config)
    prompt = prompt_path.read_text()

    def worker(task):
        candidate = task["candidate"]
        ensure_artifacts(candidate)
        source = manifest[task["image_id"]]
        cards = [_compact_candidate(c) for c in task["peers"]]
        root = output_root / "entity_alignment" / "artifacts" / "route_b"
        stem = f"{task['image_id']}_{task['entity_id']}"
        numbered = root / "target_numbered" / f"{stem}.jpg"
        montage = root / "target_montages" / f"{stem}.jpg"
        image = open_rgb(source["image_path"])
        try:
            create_labeled_crop_montage(
                image, cards, montage, highlight_id=candidate["instance_id"]
            )
        finally:
            image.close()
        images = _open_qwen_images(
            [
                source["image_path"],
                candidate["verifier_overlay_path"],
                candidate["tight_crop_path"],
                candidate["context_crop_path"],
                montage,
            ],
        )
        card = {
            "entity_id": task["entity_id"],
            "rank": task["rank"],
            "category_query": candidate["category_query"],
            "target": _compact_candidate(candidate),
            "caption_reference": source["caption"],
            "candidate_instances": cards,
        }
        def validate_promotion(decision):
            if decision.proposal and (
                decision.proposal.entity_id != task["entity_id"]
                or decision.proposal.category_query.casefold() != candidate["category_query"].casefold()
            ):
                raise ModelContractError(
                    f"proposal.entity_id must be {task['entity_id']}; "
                    f"proposal.category_query must be {candidate['category_query']}"
                )
        try:
            decision, raw = client.generate_json(
                prompt, images, CandidatePromotion, extra_text=json.dumps(card),
                validate_result=validate_promotion,
            )
        finally:
            _close_images(images)
        entity = decision.proposal.model_dump(mode="json") if decision.proposal else None
        if entity:
            if (
                entity["entity_id"] != task["entity_id"]
                or entity["category_query"].casefold() != candidate["category_query"].casefold()
            ):
                raise ValueError("promotion changed fixed target/category")
            entity.update(source="grounder_discovered", rank=task["rank"])
            span = entity["caption_span"]
            entity["caption_supported"] = bool(
                entity["caption_supported"] and span and span in source["caption"]
            )
            if not entity["caption_supported"]:
                entity["caption_span"] = ""
        # This view is not a promotion input. Publish it before checkpointing
        # accepted targets so describe/verify/resume still see the same pixels.
        if decision.accepted:
            image = open_rgb(source["image_path"])
            try:
                save_numbered_bbox_overlay(
                    image, cards, numbered, highlight_id=candidate["instance_id"]
                )
            finally:
                image.close()
        return {
            "image_id": task["image_id"],
            "entity_id": task["entity_id"],
            "entity_rank": task["rank"],
            "entity": entity,
            "candidate_instances": cards,
            "selected_candidate": candidate if entity else None,
            "distractor_instance_ids": [
                c["instance_id"] for c in cards if c["instance_id"] != candidate["instance_id"]
            ],
            "accepted": decision.accepted,
            "alignment": {
                "entity_id": task["entity_id"],
                "target_instance_id": candidate["instance_id"] if entity else None,
                "target_matches_caption": decision.accepted,
                "target_is_unique_match": decision.accepted,
                "reject_reason": decision.reject_reason,
            },
            "promotion": decision.model_dump(mode="json"),
            "target_numbered_overlay_path": str(numbered),
            "target_montage_path": str(montage),
            "metadata": {"raw_response": raw},
        }

    _parallel_write(
        items=pending,
        worker=worker,
        output_path=output_path,
        workers=workers,
        failures_path=failures_path,
        route=config["route"],
        stage="candidate_promotion",
        request_key=lambda t: t["entity_id"],
    )


def promote_candidates(*, candidates, alignments, deduplicator=None, **kwargs):
    _promote_tasks(tasks=promotion_tasks(candidates, alignments, kwargs["config"],
                                         deduplicator=deduplicator), **kwargs)
