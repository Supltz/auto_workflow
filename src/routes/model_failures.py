"""Auditable fail-closed records for exhausted model-answer validation."""

from src.routes.route_b_checkpoint import fingerprint, semantic
from src.utils.io import atomic_write_json, read_jsonl


def skipped_record(item, stage, error):
    record = {k: item[k] for k in ("image_id", "entity_id", "expression_id",
                                 "base_expression_id", "refinement_root", "revision") if k in item}
    if stage == "expression_refinement":
        record.update(
            expression_id=f"{item['base_expression_id']}_r{int(item['revision']) + 1}",
            revision=int(item["revision"]) + 1,
            refinement_root=item.get("refinement_root") or fingerprint(semantic(item)),
        )
    return {**record, "accepted": False, "entities": [], "model_output_invalid": True,
            "stage": stage, "reject_reason": getattr(
                error, "reject_reason", "model_output_invalid_after_retries"),
            "error_type": type(error).__name__, "error_message": str(error)}


def report_model_failures(stage, invalid, total, output_path=None):
    if output_path is not None:
        records = list(read_jsonl(output_path))
        invalid = sum(bool(r.get("model_output_invalid")) for r in records)
        total = len(records)
    warning = bool(invalid and (invalid == total or (invalid >= 5 and invalid / max(total, 1) >= 0.1)))
    print(f"[{stage}] model_output_invalid={invalid}/{total} (not visual rejection)", flush=True)
    if output_path is not None:
        atomic_write_json(output_path.with_suffix(".health.json"), {
            "stage": stage, "completed_records": total, "model_output_invalid": invalid,
            "warning": warning, "policy": "warn_if_all_invalid_or_at_least_5_and_10_percent",
        })
    if warning:
        print(f"WARNING [{stage}]: excessive invalid model answers; "
              "inspect failures.jsonl and skipped records before trusting coverage", flush=True)
