"""Caption-first ordering for proposals returned by a single model request."""

from typing import Any

from src.routes.route_b_checkpoint import fingerprint


def prioritize_proposals(
    entities: list[dict[str, Any]], *, caption: str, image_id: str, proposal_cap: int
) -> list[dict[str, Any]]:
    """Validate provenance, deduplicate and reserve capacity for supported proposals.

    A matching span is only a provenance check, not proof of visual eligibility or
    complete caption coverage. Those judgements remain in the model request.
    """
    if not 0 <= proposal_cap <= 30:
        raise ValueError("proposal_cap must be between 0 and 30")
    unique = {}
    for proposal in entities:
        entity = dict(proposal)
        entity["source"] = "visual_proposal"
        span = entity["caption_span"]
        entity["caption_supported"] = bool(
            entity["caption_supported"] and span and span in caption
        )
        if not entity["caption_supported"]:
            entity["caption_span"] = ""
        identity = {k: v for k, v in entity.items() if k not in {"entity_id", "rank"}}
        entity["entity_id"] = "v_" + fingerprint([image_id, identity])[:20]
        unique.setdefault(entity["entity_id"], entity)
    # Stable within each phase: retain the model's quality order. Apply the cap
    # after provenance validation and deduplication, not before caption priority.
    ordered = sorted(unique.values(), key=lambda entity: not entity["caption_supported"])
    selected = ordered[:proposal_cap]
    for rank, entity in enumerate(selected, 1):
        entity["rank"] = rank
    return selected
