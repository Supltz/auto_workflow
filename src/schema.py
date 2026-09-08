"""Versioned JSON schemas for Route B and its model workers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaptionEntity(StrictModel):
    """One caption-grounded, independently localizable whole entity."""

    entity_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    rank: int = Field(ge=1)
    caption_span: str = Field(min_length=1, max_length=500)
    category: str = Field(min_length=1, max_length=80)
    scope: Literal["whole_object", "small_group"]
    attributes: list[str] = Field(default_factory=list, max_length=10)
    action: str | None = Field(default=None, max_length=160)
    relations: list[str] = Field(default_factory=list, max_length=6)
    visible_text: list[str] = Field(default_factory=list, max_length=8)
    category_query: str = Field(min_length=1, max_length=120)
    locator_query: str = Field(min_length=1, max_length=300)

    @field_validator("caption_span", "category", "category_query", "locator_query")
    @classmethod
    def normalize_entity_text(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("entity text must not be empty")
        return normalized

    @field_validator("action")
    @classmethod
    def normalize_optional_entity_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        return normalized or None

    @field_validator("attributes", "relations", "visible_text")
    @classmethod
    def normalize_entity_lists(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.strip().split()) for value in values]
        normalized = [value for value in normalized if value]
        return list(dict.fromkeys(normalized))


class CaptionEntitySet(StrictModel):
    entities: list[CaptionEntity] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def unique_and_ranked(self) -> CaptionEntitySet:
        ids = [entity.entity_id.casefold() for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity IDs must be unique")
        ranks = [entity.rank for entity in self.entities]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("entity ranks must be contiguous and start at 1")
        return self


class VisualEntity(CaptionEntity):
    """Visual hypothesis; caption membership is provenance, not eligibility."""

    caption_span: str = Field(default="", max_length=500)
    short_name: str = Field(min_length=1, max_length=160)
    proposal_reason: str = Field(min_length=1, max_length=300)
    source: Literal["visual_proposal", "grounder_discovered"] = "visual_proposal"
    caption_supported: bool = False
    scope: Literal["whole_object"] = "whole_object"

    @field_validator("caption_span", "category", "category_query", "locator_query")
    @classmethod
    def normalize_entity_text(cls, value: str) -> str:
        return " ".join(value.strip().split())


class VisualEntitySet(StrictModel):
    entities: list[VisualEntity] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def unique_and_ranked(self) -> VisualEntitySet:
        ids = [entity.entity_id.casefold() for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity IDs must be unique")
        if [e.rank for e in self.entities] != list(range(1, len(ids) + 1)):
            raise ValueError("entity ranks must be contiguous and start at 1")
        return self


class CandidatePromotion(StrictModel):
    accepted: bool
    proposal: VisualEntity | None = None
    reject_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def consistent_decision(self) -> CandidatePromotion:
        if self.accepted and (self.proposal is None or self.reject_reason is not None):
            raise ValueError("accepted promotion requires a proposal and no rejection")
        if not self.accepted and (self.proposal is not None or not self.reject_reason):
            raise ValueError("rejected promotion requires a reason and no proposal")
        return self


class EntityAlignmentDecision(StrictModel):
    entity_id: str = Field(min_length=1, max_length=80)
    target_instance_id: str | None = Field(default=None, max_length=120)
    target_matches_caption: bool
    target_is_unique_match: bool
    reject_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def consistent_decision(self) -> EntityAlignmentDecision:
        accepted = (
            self.target_matches_caption
            and self.target_is_unique_match
            and self.target_instance_id is not None
        )
        if accepted and self.reject_reason is not None:
            raise ValueError("accepted alignment must have null reject_reason")
        if not accepted and not self.reject_reason:
            raise ValueError("rejected alignment must explain reject_reason")
        return self


class EntityBBoxVerification(StrictModel):
    entity_id: str = Field(min_length=1, max_length=80)
    instance_id: str = Field(min_length=1, max_length=120)
    target_matches_entity: bool
    bbox_covers_whole_entity: bool
    crop_is_clear: bool
    detail_is_sufficient: bool
    caption_attributes_visible: bool
    bbox_is_tight: bool
    bbox_excludes_unnecessary_neighbors: bool
    target_is_single_entity: bool
    preserves_target_identity: bool
    target_attributes: list[str] = Field(max_length=24)
    target_actions: list[str] = Field(max_length=8)
    reference_objects: list[str] = Field(max_length=16)
    supported_relations: list[str] = Field(max_length=16)
    uncertain_attributes: list[str] = Field(max_length=16)
    reject_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def consistent_decision(self) -> EntityBBoxVerification:
        uncertain = {value.strip().casefold() for value in self.uncertain_attributes}
        if any(value.strip().casefold() in uncertain for value in self.target_attributes):
            raise ValueError("uncertain attributes cannot also be approved target attributes")
        accepted = all(
            (
                self.target_matches_entity,
                self.bbox_covers_whole_entity,
                self.crop_is_clear,
                self.detail_is_sufficient,
                self.caption_attributes_visible,
                self.bbox_is_tight,
                self.bbox_excludes_unnecessary_neighbors,
                self.target_is_single_entity,
                self.preserves_target_identity,
            )
        )
        if accepted and self.reject_reason is not None:
            raise ValueError("accepted bbox verification must have null reject_reason")
        if not accepted and not self.reject_reason:
            raise ValueError("rejected bbox verification must explain reject_reason")
        return self


class OCRTextObservation(StrictModel):
    text: str = Field(min_length=1, max_length=160)
    normalized_text: str = Field(min_length=1, max_length=160)
    location: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("text", "normalized_text", "location")
    @classmethod
    def normalize_ocr_text(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("OCR text fields must not be empty")
        return normalized


class RegionOCRResult(StrictModel):
    entity_id: str = Field(min_length=1, max_length=80)
    instance_id: str = Field(min_length=1, max_length=120)
    observations: list[OCRTextObservation] = Field(default_factory=list, max_length=12)
    no_legible_text: bool

    @model_validator(mode="after")
    def consistent_observations(self) -> RegionOCRResult:
        if self.no_legible_text == bool(self.observations):
            raise ValueError(
                "no_legible_text must be true exactly when observations is empty"
            )
        return self


class ReferringExpressionDraft(StrictModel):
    entity_id: str = Field(min_length=1, max_length=80)
    instance_id: str = Field(min_length=1, max_length=120)
    expression: str = Field(min_length=1, max_length=500)
    head_category: str = Field(min_length=1, max_length=80)
    used_attributes: list[str] = Field(default_factory=list, max_length=10)
    used_action: str | None = Field(default=None, max_length=160)
    used_relation: str | None = Field(default=None, max_length=200)
    used_visible_text: list[str] = Field(default_factory=list, max_length=8)
    comparison_basis: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("expression", "head_category")
    @classmethod
    def normalize_expression_text(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("expression text must not be empty")
        return normalized

    @field_validator("used_action", "used_relation")
    @classmethod
    def normalize_optional_expression_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        return normalized or None

    @field_validator("used_attributes", "used_visible_text", "comparison_basis")
    @classmethod
    def normalize_expression_lists(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.strip().split()) for value in values]
        normalized = [value for value in normalized if value]
        return list(dict.fromkeys(normalized))


class ExpressionVerification(StrictModel):
    entity_id: str = Field(min_length=1, max_length=80)
    instance_id: str = Field(min_length=1, max_length=120)
    expression: str = Field(min_length=1, max_length=500)
    target_matches: bool
    describes_whole_entity: bool
    attributes_visible: bool
    grammatically_valid: bool
    target_is_unique: bool
    reground_matches_target: bool
    single_target_scope: bool
    attributes_belong_to_target: bool
    references_only_relational: bool
    bbox_tight_and_complete: bool
    confusable_instance_ids: list[str] = Field(default_factory=list, max_length=12)
    suggested_discriminator: str | None = Field(default=None, max_length=300)
    reject_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def consistent_decision(self) -> ExpressionVerification:
        accepted = all(
            (
                self.target_matches,
                self.describes_whole_entity,
                self.attributes_visible,
                self.grammatically_valid,
                self.target_is_unique,
                self.reground_matches_target,
                self.single_target_scope,
                self.attributes_belong_to_target,
                self.references_only_relational,
                self.bbox_tight_and_complete,
            )
        )
        if accepted and self.reject_reason is not None:
            raise ValueError("accepted expression verification must have null reject_reason")
        if not accepted and not self.reject_reason:
            raise ValueError("rejected expression verification must explain reject_reason")
        return self


class DatasetRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    image_id: str
    image_path: str
    caption: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_index: int = Field(ge=0)
    dataset: str
    source_shard: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_field_names(cls, value: Any) -> Any:
        """Read early manifests while always serializing the documented schema."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        aliases = {
            "image_width": "width",
            "image_height": "height",
            "source_row": "source_index",
            "source_dataset": "dataset",
        }
        for legacy, documented in aliases.items():
            if documented not in normalized and legacy in normalized:
                normalized[documented] = normalized[legacy]
            normalized.pop(legacy, None)
        return normalized


class GroundingRequest(StrictModel):
    request_id: str
    image_id: str
    image_path: str
    phrase: str
    rank: int = Field(ge=1)
    route: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegionRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    image_id: str
    route: str
    rank: int = Field(ge=1)
    phrase: str = Field(min_length=1)
    bbox_xyxy: tuple[float, float, float, float]
    mask_path: str | None = None
    crop_path: str | None = None
    overlay_path: str | None = None
    source_method: str
    grounder: str
    grounding_score: float | None = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    runtime_ms: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_bbox(self) -> RegionRecord:
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 <= self.image_width and 0 <= y1 < y2 <= self.image_height):
            raise ValueError(
                f"bbox {self.bbox_xyxy} outside {self.image_width}x{self.image_height}"
            )
        return self


class FailureRecord(StrictModel):
    image_id: str
    route: str
    stage: str
    error_type: str
    message: str
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelProvenance(StrictModel):
    model_name: str
    checkpoint: str
    git_commit: str | None
    dtype: str
    gpu: str | None
    prompt_version: str
    config_hash: str
    runtime_ms: float
