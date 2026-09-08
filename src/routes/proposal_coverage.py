"""Image-level proposal coverage, separate from model-answer validity."""

from collections import Counter

from src.utils.io import atomic_write_json


def report_proposal_coverage(records, output_path):
    """Report empty answers without treating failed answers as genuine negatives."""
    invalid = [r for r in records if r.get("model_output_invalid")]
    valid = [r for r in records if not r.get("model_output_invalid")]
    empty = sum(not r.get("entities") for r in valid)
    nonempty = len(valid) - empty
    rechecked = [r for r in valid if r.get("metadata", {}).get("empty_recheck")]
    recovered = sum(bool(r.get("entities")) for r in rechecked)
    empty_rate = empty / len(valid) if valid else None
    # A diagnostic threshold only; it never changes eligibility or forces proposals.
    warning = bool(valid and (empty == len(valid) or (len(valid) >= 10 and empty_rate >= 0.5)))
    report = {
        "stage": "visual_entity_proposal",
        "total_images": len(records),
        "valid_images": len(valid),
        "invalid_images": len(invalid),
        "nonempty_images": nonempty,
        "empty_images": empty,
        "empty_fraction_of_valid_images": empty_rate,
        "proposal_count": sum(len(r.get("entities", [])) for r in valid),
        "proposal_count_distribution": dict(sorted(Counter(
            len(r.get("entities", [])) for r in valid).items())),
        "empty_rechecks_completed": len(rechecked),
        "empty_rechecks_recovered": recovered,
        "warning": warning,
        "policy": "warn_if_all_valid_empty_or_at_least_10_valid_and_50_percent_empty",
    }
    atomic_write_json(output_path.with_suffix(".coverage.json"), report)
    print(
        f"[proposal coverage] images={len(records)} nonempty={nonempty} "
        f"empty={empty} invalid={len(invalid)} proposals={report['proposal_count']} "
        f"recheck_recovered={recovered}/{len(rechecked)}",
        flush=True,
    )
    if warning:
        print("WARNING [proposal coverage]: high empty-answer rate; inspect source images "
              "and proposal recall before trusting dataset coverage", flush=True)
    return report
