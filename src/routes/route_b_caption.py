"""Route B: visual proposals + bounded consensus discovery -> unique expressions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.regions.route_b import (
    aggregate_route_b_entities,
    entity_grounding_requests,
    expression_grounding_requests,
)
from src.routes.common import (
    invoke_grounders,
    manifest_index,
    sampled_manifest_rows,
    selected_output,
)
from src.routes.route_b_checkpoint import (
    CHECK_ONLY,
    StagePending,
    archive_and_replace,
    fingerprint,
    semantic,
)
from src.routes.route_b_discovery import promote_candidates, unique_alignments
from src.routes.route_b_stages import (
    align_entities_to_instances,
    describe_grounded_entities,
    extract_caption_entities,
    extract_region_text,
    materialize_final_route_b_outputs,
    refine_expression_batch,
    refinement_eligible,
    verify_entity_bboxes,
    verify_generated_expressions,
)
from src.routes.target_dedup import TargetDeduplicator
from src.utils.config import load_yaml, resolve_path
from src.utils.io import add_job_arguments, prepare_output, read_jsonl, rewrite_jsonl_atomic
from src.utils.provenance import record_run
from src.visualization.route_b_review import export_route_b_review

STAGES = (
    "entities",
    "ground",
    "aggregate",
    "align",
    "promote",
    "bbox_verify",
    "ocr",
    "describe",
    "reground",
    "expression_verify",
    "refine_generate",
    "refine_reground",
    "refine_verify",
    "refine",
    "finalize",
    "review",
    "all",
)


def _records(path: Path, selected_ids: set[str], *, include_invalid=False) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required Route B input is missing: {path}")
    return [item for item in read_jsonl(path) if item["image_id"] in selected_ids
            and (include_invalid or not item.get("model_output_invalid"))]


def _invoke_or_write_empty(
    *,
    requests: list[dict[str, Any]],
    config: dict[str, Any],
    models_config: dict[str, Any],
    output_path: Path,
    output_root: Path,
    failures: Path,
    resume: bool,
    overwrite: bool,
    save_masks: bool,
) -> None:
    if not requests:
        rewrite_jsonl_atomic(output_path, [])
        rewrite_jsonl_atomic(
            output_path.with_name(f"{output_path.stem}_requests.jsonl"), []
        )
        return
    invoke_grounders(
        requests,
        config["grounders"],
        config["models_config"],
        models_config,
        output_path,
        output_root,
        failures,
        resume,
        overwrite,
        save_crops=False,
        save_overlays=False,
        save_masks=save_masks,
        worker_environment=config.get("grounder_environment"),
    )


def _latest_verifications(
    initial: list[dict[str, Any]], refined: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in [*initial, *refined]:
        key = record["base_expression_id"]
        prior = latest.get(key)
        if prior is None or int(record["revision"]) > int(prior["revision"]):
            latest[key] = record
    return latest


def _optional_records(path: Path, selected_ids: set[str]) -> list[dict[str, Any]]:
    return _records(path, selected_ids) if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_job_arguments(parser, "configs/route_b.yaml")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--check-only", action="store_true",
                        help="Read-only Qwen-stage completion probe; exit 3 means pending")
    args = parser.parse_args()
    if args.check_only and args.stage not in {
        "entities", "align", "promote", "bbox_verify", "ocr", "describe",
        "expression_verify", "refine_generate", "refine_verify", "finalize",
    }:
        parser.error("--check-only is supported only for Qwen stages")
    CHECK_ONLY.set(args.check_only)
    config = load_yaml(args.config)
    models_config = load_yaml(config["models_config"])
    qwen_config = models_config["qwen"]
    experiment = load_yaml("configs/experiment.yaml")
    output_root = resolve_path(args.output_dir or "outputs")
    if args.end_index is None:
        args.end_index = args.start_index + int(config.get("sample_size", 100))
    manifest = manifest_index(config["manifest"])
    manifest_rows = sampled_manifest_rows(
        config["manifest"], args.start_index, args.end_index,
        seed=int(config.get("sample_seed", 42)),
    )
    selected_ids = {record["image_id"] for record in manifest_rows}

    def output(key: str) -> Path:
        return selected_output(config[key], args.output_dir)

    entity_output = output("entity_output")
    entity_grounding_output = output("entity_grounding_output")
    grounded_entities_output = output("grounded_entities_output")
    grounding_rejections_output = output("grounding_rejections_output")
    alignment_output = output("alignment_output")
    promotion_output = output("promotion_output")
    bbox_verification_output = output("bbox_verification_output")
    ocr_output = output("ocr_output")
    expression_output = output("expression_output")
    expression_grounding_output = output("expression_grounding_output")
    expression_verification_output = output("expression_verification_output")
    refinement_drafts_output = output("refinement_drafts_output")
    refinement_grounding_output = output("refinement_grounding_output")
    refinement_verification_output = output("refinement_verification_output")
    verified_output = output("verified_output")
    rejected_output = output("rejected_output")
    human_review_dir = output("human_review_dir")
    failures = (
        resolve_path(args.output_dir) / "failures.jsonl"
        if args.output_dir
        else resolve_path(experiment["failures_file"])
    )

    if not args.check_only:
        rewrite_jsonl_atomic(output_root / "route_b" / "selected_manifest.jsonl", manifest_rows)
        record_run(experiment["provenance_file"], config["route"],
                   {**config, "models": models_config}, stage=args.stage)

    deduplicator = TargetDeduplicator(config=config, qwen_config=qwen_config,
                                       output_root=output_root)

    def target_alignments():
        return unique_alignments([*_records(alignment_output, selected_ids),
                                  *_records(promotion_output, selected_ids)],
                                 float(config.get("final_target_dedup_iou", 0.98)),
                                 deduplicator=deduplicator)

    # Old refinement rows are never attached to a newly discovered target by ID alone.
    if args.stage in {"refine_generate", "refine_reground", "refine_verify", "refine", "finalize"}:
        initial = _records(expression_verification_output, selected_ids)
        bases = {r["base_expression_id"]: fingerprint(semantic(r)) for r in initial}
        for path in (refinement_drafts_output, refinement_verification_output):
            if path.exists():
                kept = [r for r in read_jsonl(path) if r.get("refinement_root")
                        and r.get("refinement_root") == bases.get(r["base_expression_id"])]
                if not args.check_only:
                    archive_and_replace(path, kept)

    if args.stage in {"entities", "all"}:
        extract_caption_entities(
            manifest_rows=manifest_rows,
            output_path=entity_output,
            failures_path=failures,
            prompt_path=resolve_path(config["entity_prompt"]),
            qwen_config=qwen_config,
            config=config,
            workers=int(config.get("entity_workers", 4)),
            resume=args.resume,
            overwrite=args.overwrite,
        )

    if args.stage in {"ground", "all"}:
        entity_rows = _records(entity_output, selected_ids)
        requests = entity_grounding_requests(entity_rows, manifest, config["route"])
        _invoke_or_write_empty(
            requests=requests,
            config=config,
            models_config=models_config,
            output_path=entity_grounding_output,
            output_root=output_root,
            failures=failures,
            resume=args.resume,
            overwrite=args.overwrite,
            save_masks=bool(config.get("save_masks", True)),
        )

    if args.stage in {"aggregate", "all"}:
        if not entity_grounding_output.exists():
            raise FileNotFoundError(
                f"required Route B input is missing: {entity_grounding_output}"
            )
        aggregate_route_b_entities(
            entities_path=entity_output,
            raw_grounding_path=entity_grounding_output,
            candidates_path=grounded_entities_output,
            rejections_path=grounding_rejections_output,
            output_root=output_root,
            manifest=manifest,
            selected_ids=selected_ids,
            config=config,
            overwrite=args.overwrite,
        )

    if args.stage in {"align", "all"}:
        align_entities_to_instances(
            entity_rows=_records(entity_output, selected_ids),
            candidates=_records(grounded_entities_output, selected_ids),
            manifest=manifest,
            output_path=alignment_output,
            failures_path=failures,
            prompt_path=resolve_path(config["alignment_prompt"]),
            output_root=output_root,
            qwen_config=qwen_config,
            config=config,
            workers=int(config.get("alignment_workers", 4)),
            resume=args.resume,
            overwrite=args.overwrite,
        )

    if args.stage in {"promote", "all"}:
        promote_candidates(
            deduplicator=deduplicator,
            candidates=_records(grounded_entities_output, selected_ids),
            alignments=_records(alignment_output, selected_ids), manifest=manifest,
            output_path=promotion_output, failures_path=failures,
            prompt_path=resolve_path(config["promotion_prompt"]), output_root=output_root,
            qwen_config=qwen_config, config=config,
            workers=int(config.get("promotion_workers", 4)), resume=args.resume,
            overwrite=args.overwrite)

    if args.stage in {"bbox_verify", "all"}:
        verify_entity_bboxes(
            alignments=target_alignments(),
            output_path=bbox_verification_output,
            failures_path=failures,
            prompt_path=resolve_path(config["bbox_verifier_prompt"]),
            qwen_config=qwen_config,
            config=config,
            workers=int(config.get("bbox_verification_workers", 4)),
            resume=args.resume,
            overwrite=args.overwrite,
        )

    if args.stage in {"ocr", "all"}:
        extract_region_text(
            bbox_records=_records(bbox_verification_output, selected_ids),
            output_path=ocr_output,
            failures_path=failures,
            prompt_path=resolve_path(config["ocr_prompt"]),
            qwen_config=qwen_config,
            config=config,
            workers=int(config.get("ocr_workers", 4)),
            resume=args.resume,
            overwrite=args.overwrite,
        )

    if args.stage in {"describe", "all"}:
        describe_grounded_entities(
            bbox_records=_records(bbox_verification_output, selected_ids),
            ocr_records=_records(ocr_output, selected_ids, include_invalid=True),
            manifest=manifest,
            output_path=expression_output,
            failures_path=failures,
            prompt_path=resolve_path(config["expression_prompt"]),
            qwen_config=qwen_config,
            config=config,
            workers=int(config.get("expression_workers", 4)),
            resume=args.resume,
            overwrite=args.overwrite,
        )

    if args.stage in {"reground", "all"}:
        drafts = _records(expression_output, selected_ids)
        requests = expression_grounding_requests(
            drafts, manifest, config["route"], "generated_expression_reground"
        )
        _invoke_or_write_empty(
            requests=requests,
            config=config,
            models_config=models_config,
            output_path=expression_grounding_output,
            output_root=output_root,
            failures=failures,
            resume=args.resume,
            overwrite=args.overwrite,
            save_masks=bool(config.get("reground_save_masks", False)),
        )

    if args.stage in {"expression_verify", "all"}:
        verify_generated_expressions(
            drafts=_records(expression_output, selected_ids),
            raw_records=_records(expression_grounding_output, selected_ids),
            output_path=expression_verification_output,
            failures_path=failures,
            prompt_path=resolve_path(config["expression_verifier_prompt"]),
            output_root=output_root,
            qwen_config=qwen_config,
            config=config,
            workers=int(config.get("expression_verification_workers", 4)),
            resume=args.resume,
            overwrite=args.overwrite,
        )

    if args.stage == "refine_generate":
        initial_verifications = _records(expression_verification_output, selected_ids)
        if not args.check_only:
            prepare_output(refinement_drafts_output, args.overwrite, args.resume)
        refinement_verifications = _optional_records(
            refinement_verification_output, selected_ids
        )
        refinement_drafts = _optional_records(refinement_drafts_output, selected_ids)
        existing_ids = {record["expression_id"] for record in refinement_drafts}
        if refinement_drafts_output.exists():
            existing_ids.update(r["expression_id"] for r in
                                _records(refinement_drafts_output, selected_ids, include_invalid=True))
        max_rounds = int(config.get("max_refinement_rounds", 2))
        eligible = [
            record
            for record in _latest_verifications(
                initial_verifications, refinement_verifications
            ).values()
            if refinement_eligible(record, max_rounds)
            and f"{record['base_expression_id']}_r{int(record['revision']) + 1}"
            not in existing_ids
        ]
        if args.check_only:
            if eligible or not refinement_drafts_output.exists():
                raise StagePending("refine_generate")
            return
        if eligible:
            refine_expression_batch(
                previous_records=eligible,
                output_path=refinement_drafts_output,
                failures_path=failures,
                prompt_path=resolve_path(config["refinement_prompt"]),
                qwen_config=qwen_config,
                config=config,
                workers=int(config.get("expression_workers", 4)),
            )
        elif not refinement_drafts_output.exists():
            rewrite_jsonl_atomic(refinement_drafts_output, [])

    if args.stage == "refine_reground":
        refinement_drafts = _records(refinement_drafts_output, selected_ids)
        requests = expression_grounding_requests(
            refinement_drafts,
            manifest,
            config["route"],
            "refined_expression_reground",
        )
        _invoke_or_write_empty(
            requests=requests,
            config=config,
            models_config=models_config,
            output_path=refinement_grounding_output,
            output_root=output_root,
            failures=failures,
            resume=args.resume,
            overwrite=args.overwrite,
            save_masks=bool(config.get("reground_save_masks", False)),
        )

    if args.stage == "refine_verify":
        verify_generated_expressions(
            drafts=_records(refinement_drafts_output, selected_ids),
            raw_records=_records(refinement_grounding_output, selected_ids),
            output_path=refinement_verification_output,
            failures_path=failures,
            prompt_path=resolve_path(config["expression_verifier_prompt"]),
            output_root=output_root,
            qwen_config=qwen_config,
            config=config,
            workers=int(config.get("expression_verification_workers", 4)),
            resume=args.resume,
            overwrite=args.overwrite,
        )

    if args.stage in {"refine", "all"}:
        initial_verifications = _records(expression_verification_output, selected_ids)
        prepare_output(refinement_drafts_output, args.overwrite, args.resume)
        prepare_output(refinement_verification_output, args.overwrite, args.resume)
        grounder_overwrite = args.overwrite
        verifier_overwrite = args.overwrite
        max_rounds = int(config.get("max_refinement_rounds", 2))

        # Resume can leave generated revisions awaiting their re-ground/verification.
        for _ in range(max_rounds + 1):
            refinement_drafts = (
                _records(refinement_drafts_output, selected_ids)
                if refinement_drafts_output.exists()
                else []
            )
            refinement_verifications = (
                _records(refinement_verification_output, selected_ids)
                if refinement_verification_output.exists()
                else []
            )
            verified_ids = {
                record["expression_id"] for record in refinement_verifications
            }
            pending_drafts = [
                record
                for record in refinement_drafts
                if record["expression_id"] not in verified_ids
            ]
            if pending_drafts:
                all_requests = expression_grounding_requests(
                    refinement_drafts,
                    manifest,
                    config["route"],
                    "refined_expression_reground",
                )
                _invoke_or_write_empty(
                    requests=all_requests,
                    config=config,
                    models_config=models_config,
                    output_path=refinement_grounding_output,
                    output_root=output_root,
                    failures=failures,
                    resume=args.resume,
                    overwrite=grounder_overwrite,
                    save_masks=bool(config.get("reground_save_masks", False)),
                )
                grounder_overwrite = False
                verify_generated_expressions(
                    drafts=refinement_drafts,
                    raw_records=_records(refinement_grounding_output, selected_ids),
                    output_path=refinement_verification_output,
                    failures_path=failures,
                    prompt_path=resolve_path(config["expression_verifier_prompt"]),
                    output_root=output_root,
                    qwen_config=qwen_config,
                    config=config,
                    workers=int(config.get("expression_verification_workers", 4)),
                    resume=True,
                    overwrite=verifier_overwrite,
                )
                verifier_overwrite = False
                refinement_verifications = _records(
                    refinement_verification_output, selected_ids
                )

            latest = _latest_verifications(
                initial_verifications, refinement_verifications
            )
            existing_draft_ids = {
                record["expression_id"] for record in refinement_drafts
            }
            if refinement_drafts_output.exists():
                existing_draft_ids.update(r["expression_id"] for r in
                    _records(refinement_drafts_output, selected_ids, include_invalid=True))
            eligible = [
                record
                for record in latest.values()
                if refinement_eligible(record, max_rounds)
                and f"{record['base_expression_id']}_r{int(record['revision']) + 1}"
                not in existing_draft_ids
            ]
            if not eligible:
                break
            refine_expression_batch(
                previous_records=eligible,
                output_path=refinement_drafts_output,
                failures_path=failures,
                prompt_path=resolve_path(config["refinement_prompt"]),
                qwen_config=qwen_config,
                config=config,
                workers=int(config.get("expression_workers", 4)),
            )

        if not refinement_grounding_output.exists():
            rewrite_jsonl_atomic(refinement_grounding_output, [])
        if not refinement_verification_output.exists():
            rewrite_jsonl_atomic(refinement_verification_output, [])
        refinement_verifications = _records(
            refinement_verification_output, selected_ids
        )
        materialize_final_route_b_outputs(
            deduplicator=deduplicator,
            initial_verifications=initial_verifications,
            refinement_verifications=refinement_verifications,
            grounding_rejections=_records(grounding_rejections_output, selected_ids),
            alignments=target_alignments(),
            bbox_records=_records(bbox_verification_output, selected_ids),
            verified_path=verified_output,
            rejected_path=rejected_output,
            dedup_iou_threshold=float(config.get("final_target_dedup_iou", 0.98)),
            max_per_source_image=int(config.get("final_max_per_source_image", 30)),
        )

    if args.stage in {"finalize", "all"}:
        materialize_final_route_b_outputs(
            deduplicator=deduplicator,
            initial_verifications=_records(
                expression_verification_output, selected_ids
            ),
            refinement_verifications=_optional_records(
                refinement_verification_output, selected_ids
            ),
            grounding_rejections=_records(grounding_rejections_output, selected_ids),
            alignments=target_alignments(),
            bbox_records=_records(bbox_verification_output, selected_ids),
            verified_path=verified_output,
            rejected_path=rejected_output,
            dedup_iou_threshold=float(config.get("final_target_dedup_iou", 0.98)),
            max_per_source_image=int(config.get("final_max_per_source_image", 30)),
        )

    if args.check_only:
        return

    if args.stage in {"finalize", "review", "all"}:
        counts = Counter(r["image_id"] for r in _records(verified_output, selected_ids))
        summary = [{"image_id": image_id, "final_bbox_count": counts[image_id]}
                   for image_id in sorted(selected_ids)]
        summary_path = verified_output.with_name("route_b_counts_by_image.jsonl")
        rewrite_jsonl_atomic(summary_path, summary)
        print(f"[final counts] source_images={len(summary)} bboxes={sum(counts.values())} "
              f"distribution(count:images)={dict(sorted(Counter(counts[i] for i in selected_ids).items()))} "
              f"file={summary_path}", flush=True)

    if args.stage in {"review", "all"}:
        export_route_b_review(
            verified_path=verified_output,
            review_root=human_review_dir,
            selected_ids=selected_ids,
            max_per_source_image=int(
                config.get("review_max_per_source_image", 30)
            ),
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    try:
        main()
    except StagePending:
        raise SystemExit(3)
