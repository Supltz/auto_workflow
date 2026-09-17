# Route B pipeline contract

> 当前默认 Route B 使用 SAM3.1 + EGM、对象持久保留及 phrase review v2（29 stages），详见 [当前流程说明](PHRASE_CONTEXT_V2.md)。下文三模型 consensus、18 stages、8–30 词和 Top-30 的流程属于旧版。

All stages exchange JSONL. `--start-index` is inclusive, `--end-index` is exclusive, and
expensive inference stages use semantic-input checkpoints and completed request records.
The range indexes a seeded random permutation of the complete manifest, not its physical
row order. The default is 100 images without replacement with sample_seed=42. Every stage
uses this same selection; the manifest and seed must remain fixed during resume.

1. `entities`: Qwen receives the source image and English caption reference and emits 0-30
   whole-object proposals: caption-mentioned objects confirmed in the image first, then
   additional visual targets using remaining capacity. Uncertain sizes are left for
   downstream detection; numeric size limits remain enforced in aggregation. An empty
   response gets one focused recheck, which may still return empty. Each proposal includes a broad
   `category_query` and a discriminative `locator_query`.
2. `ground`: each distinct category query and each non-identical locator query is sent to
   Rex, SAM3.1, and GroundingDINO.
3. `aggregate`: detections are deduplicated within each grounder, clustered across
   grounders by IoU/containment, and retained only with support from at least two models.
   A supported SAM mask-tight box is preferred; otherwise a detection medoid is used.
   The hard bbox limits are a 32 px minimum short side and 0.001%-10% image area.
4. `align`: Qwen receives the original image, a numbered candidate overlay, a crop
   montage, the caption, and the entity card. It must identify exactly one compatible
   instance or reject the entity as missing/ambiguous.
5. `promote`: Qwen inspects up to 16 extra size-valid category-only consensus instances
   not matched to proposals. Accepted instances join the same QA path; no new categories
   or grounding requests are generated. Duplicate aligned targets are removed before QA.
6. `bbox_verify`: Qwen checks the selected overlay, tight crop, and 20%-padded context
   crop for correct entity identity, whole-object coverage, clarity, sufficient detail,
   and visible proposal attributes, regardless of caption support.
7. `ocr`: Qwen reads only text physically visible on the accepted target. Proposal text is
   supplied for verification only and cannot be copied without visual evidence.
8. `describe`: Qwen compares the target against all same-category candidates and writes
   one standalone English expression using at least two declared visual cues and 8-30
   words.
9. `reground`: the complete expression is sent to all three grounders. It passes only if
   there is exactly one multi-grounder consensus result and it matches the original bbox.
10. `expression_verify`: Qwen independently checks target identity, whole-entity wording,
   visible factuality, grammar, uniqueness against distractors, and re-ground agreement.
11. `refine_generate` / `refine_reground` / `refine_verify`: a factually sound but weak or
    ambiguous expression may be minimally revised and round-trip tested up to two times.
12. `finalize`: retain the latest accepted revision, deduplicate identical same-image
    expressions (trimmed and case-folded) directly, then quality-rank targets using the
    shared identity rule below and keep at most 30 per source. Zero accepted targets is valid.
13. `review`: export every final verified result. Each JPG contains
    exactly one bbox and its expression in the top-left panel.

The managed runner alternates Qwen-server phases and grounding phases to avoid GPU-memory
contention:

```bash
bash scripts/run_pipeline.sh
```

All Qwen requests contain image inputs followed by one `user` text message consisting of
the relevant `prompts/route_b_*.txt` instruction and a stage-specific JSON card. The code
does not send a custom system message. Responses use strict JSON Schema, temperature 0,
and disabled thinking.

## Shared target deduplication

Discovery, pre-QA alignment selection, and finalization use `src/routes/target_dedup.py`.
Identical nonempty expressions within one image are always deduplicated, regardless of
bbox overlap. For different expressions, same-image, same-category, same-scope boxes at
IoU >= 0.98 are merged directly. Suspected duplicates at IoU >= 0.75, smaller/larger area
ratio >= 0.70, and per-axis center offset <= 0.20 of the smaller box dimension receive
one joint Qwen identity check. These thresholds are configurable in `configs/route_b.yaml`.
The model sees the original, a numbered two-box overlay, and a shared context crop.
Only `same_object` merges; `different_objects` and `uncertain` retain both candidates.
Invalid responses or service errors stop processing instead of silently dropping targets.
Final selection retains the strongest verified record and records its direct duplicate
links and decision evidence in rejected records; overlapping chains are not merged transitively.

Identity decisions and their input pairs are cached under `outputs/route_b/target_dedup/`.
Cache keys include boxes, image identity/stat, model configuration, prompt, and contract.
`--stage finalize --check-only` performs a read-only probe (exit 3 for missing decisions).
The managed runner starts Qwen only when needed and still materializes final outputs when
all decisions are cached. To reprocess existing verified attempts without generation or
re-grounding, run `--stage finalize --resume` with Qwen available for uncached pairs, then
`--stage review --overwrite`, using the original selection range and output directory.
