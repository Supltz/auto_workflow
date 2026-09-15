"""Typed model decisions and pure policy for role-specialized grounding."""
from __future__ import annotations
import hashlib
import json
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from src.utils.geometry import iou, area, valid_box

from src.routes.role_schedule import CONTRACT, stage_plan
STAGES = tuple(row[1] for row in stage_plan())


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()[:24]


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Hint(Strict):
    region: list[float] = Field(min_length=4, max_length=4)
    evidence: str = Field(min_length=1)
    kind: Literal['visible_region', 'spatial_gap', 'distant_repetition']


class Category(Strict):
    query: str = Field(min_length=1)
    visible_evidence: str = Field(min_length=1)
    locators: list[str]
    search_hints: list[Hint]


class Scene(Strict):
    categories: list[Category]
    more_categories: bool


class BoundaryEvidence(Strict):
    top: str = Field(min_length=1)
    bottom: str = Field(min_length=1)
    left: str = Field(min_length=1)
    right: str = Field(min_length=1)


class ObjectDecision(Strict):
    boundary_evidence: BoundaryEvidence
    box_issues: list[str]
    target_identifiable: bool
    category_correct: bool
    whole_object: bool
    severe_occlusion_or_truncation: bool
    tight_and_complete: bool
    independent_object: bool
    target_attributes: list[str]
    target_actions: list[str]
    supported_relations: list[str]
    uncertain_attributes: list[str]
    reason: str = Field(min_length=1)

    def accepted(self):
        return (self.target_identifiable and self.category_correct and self.whole_object
                and not self.severe_occlusion_or_truncation and self.tight_and_complete
                and self.independent_object and not self.box_issues)


class IdentityDecision(Strict):
    same_object: bool
    certain: bool
    preferred: Literal['A','B','neither']
    reason: str = Field(min_length=1)


class OCR(Strict):
    verified_target_text: list[str]


class Draft(Strict):
    expression: str
    visible_evidence: list[str]
    reason: str


class PhraseDecision(Strict):
    outcome: Literal['supports_A','ambiguous','not_A','uncertain']
    describes_A: bool
    facts_visible: bool
    single_whole_target: bool
    grammatical: bool
    unique_in_full_image: bool
    also_matches_B: bool
    competing_object_ids: list[str]
    discriminator: str | None
    evidence: list[str]
    reason: str = Field(min_length=1)


def phrase_verdict(decision):
    """A contradictory yes cannot silently become a verified expression."""
    supports = (decision['outcome'] == 'supports_A' and decision['describes_A']
                and decision['facts_visible'] and decision['single_whole_target']
                and decision['grammatical'] and decision['unique_in_full_image']
                and not decision['also_matches_B'] and not decision['competing_object_ids']
                and bool(decision['evidence']))
    if supports: return 'verified'
    if decision['outcome'] in ('ambiguous','not_A'): return 'needs_rewrite'
    return 'unresolved'


def normalized_region(region, width, height, padding=0.):
    if len(region)!=4 or any(not math.isfinite(x) for x in region): return None
    x1,y1,x2,y2=region
    if not 0<=x1<x2<=1000 or not 0<=y1<y2<=1000: return None
    dx=(x2-x1)*padding; dy=(y2-y1)*padding
    box=[math.floor(max(0,x1-dx)*width/1000),math.floor(max(0,y1-dy)*height/1000),
         math.ceil(min(1000,x2+dx)*width/1000),math.ceil(min(1000,y2+dy)*height/1000)]
    return box if valid_box(box,width,height) else None


def crop_to_original(box, region, crop_size):
    x1,y1,x2,y2=region; cw,ch=crop_size
    return [x1+box[0]*(x2-x1)/cw,y1+box[1]*(y2-y1)/ch,
            x1+box[2]*(x2-x1)/cw,y1+box[3]*(y2-y1)/ch]


def geometry_audit(target, predictions, config):
    boxes=[p['bbox_xyxy'] for p in predictions if p.get('bbox_xyxy')]
    same=[]
    for box in boxes:
        overlap=iou(tuple(target),tuple(box))
        ratio=max(area(tuple(target)),area(tuple(box)))/max(min(area(tuple(target)),area(tuple(box))),1e-9)
        same.append((overlap,ratio))
    passed=(len(boxes)==1 and same[0][0]>=config.get('iou_threshold',.5)
            and same[0][1]<=config.get('reground_max_area_ratio',1.25))
    return dict(passed=passed, predictions=predictions, reground_iou=max((x[0] for x in same),default=0),
                reason=None if passed else ('no_valid_locator_box' if not boxes else 'locator_target_disagreement'))


def add_hint(category, region, source, reason):
    if not reason or region is None: return
    if any(iou(tuple(region),tuple(h['region']))>=.8 for h in category['hints']): return
    category['hints'].append(dict(id=key([category['id'],region]),region=region,
                                  source=source,reason=reason,status='pending'))


def reserve_search(category, limit):
    """Retries use the original slot. Yield never expands a category's budget."""
    active=next((h for h in category['hints'] if h['status']=='running'),None)
    if active: return active
    if category['local_searches']>=limit: return None
    hint=next((h for h in category['hints'] if h['status']=='pending'),None)
    if hint:
        category['local_searches']+=1
        hint.update(status='running',slot=category['local_searches'])
    return hint
