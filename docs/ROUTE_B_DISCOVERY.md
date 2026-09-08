# Visual target discovery

## Caption-first proposal policy

Entity extraction starts from the original visual-proposal prompt with a short
caption-first instruction: prioritize caption-mentioned objects confirmed in the image,
then use remaining capacity for visual-only targets. Within each group the model ranks
by clarity and distinctive evidence. Proposal generation does not estimate exact pixel
dimensions or enforce numeric area limits; uncertain sizes are passed to downstream
detection. Final size, whole-object, grounding and expression acceptance rules are unchanged.
The output remains one entities list, without a caption inventory or per-mention decisions.

A nonempty response uses one proposal request. An empty response with a positive cap
gets one focused recheck using the same image and caption. The second response may
also be empty; there is no retry loop to force candidates. Existing answer-validation
retries still apply separately to each request. Invalid answers remain failures, not
genuine empty results. A completed empty checkpoint includes this recheck and is reused
on resume.

The proposal stage writes entities.coverage.json alongside entities.jsonl and prints
image coverage, empty/invalid counts, proposal-count distribution in the report, and
completed/recovered rechecks. Empty-rate warnings fire when all valid results are empty,
or when at least ten valid results are at least 50% empty. This is diagnostic only;
it does not stop the run, relax acceptance or force nonempty answers.

Postprocessing validates caption spans, deduplicates exact proposals, puts supported
proposals first while preserving within-group order, and then applies the cap.
This prioritizes returned proposals; it cannot guarantee the model discovered every
eligible caption target. A supported span is provenance, not visual verification.
The changed prompt invalidates old entity checkpoints on the next stage execution;
the existing checkpoint mechanism archives old records and evaluates downstream
semantic inputs normally. No GPU coverage improvement has yet been measured.

## Earlier discovery implementation

Audit before this change: the worktree was clean at `0094a44`; no interrupted visual
proposal implementation was present. Existing capabilities were category query sharing,
multi-grounder consensus, bbox QA, OCR, expression writing/re-grounding/verification,
two refinement rounds and final duplicate removal. Missing were visual-only eligibility,
category-only promotion, provenance, a final quality-ranked cap and semantic checkpoints.
The Python refinement driver drained pending drafts, but the managed driver could stop
when no *new* draft remained, overlooking unfinished grounding/verification.

The existing pilot contains 501 entities across 100 images and 214 verified outputs.
That pilot used the first 100 physical rows. Future default runs instead sample 100
images without replacement from all 1962 using sample_seed=42. Thus the old 214-output
pilot is not a same-image comparison baseline for the new default selection.
Proposal cap is now 30 (zero is valid), with up to 16 additional category consensus
promotion checks. For K unique categories and P proposals, each grounder gets at most
K+P initial requests. Shared categories still run once; promotion creates no new query.
The 12288-token proposal budget supports the larger structured output; other Qwen stages
keep their existing budget. Finalization retains up to 30 after duplicate removal;
human review exports every retained target. No stage requires at least one target.

Qwen inputs now retain decoded image dimensions and are transported as lossless PNG.
The client performs no max-side downsampling. Existing crop/overlay files retain their
stored format, and candidate montages still use thumbnail cells. The model processor
still applies its own pixel budget and patch alignment; native-size multi-image requests
have not been validated against the current 32768-token server context on a GPU.

`VisualEntity` reuses existing entity fields, adding short_name, proposal_reason, source,
and caption_supported. A missing exact caption span clears caption provenance but never
rejects the target. Stable content-derived IDs replace model-generated IDs in stored
visual proposals. Legacy schema classes remain readable. For compatibility,
target_matches_caption and caption_attributes_visible now mean matching the proposal
and its visible attributes; the prompts explicitly define this meaning.

`promote` considers only extra instances obtained from category-query detections alone,
with >=2 different grounders and the unchanged size rules. Already matched targets are
excluded. Qwen sees original, overlay, tight/context crops and peers. Accepted promotions
join ordinary alignments and pass the same bbox/OCR/expression/re-ground gates. Same-class
size-invalid candidates remain distractors. No source affects quality scoring. Origin and
caption_supported propagate to final JSONL and review CSV, never to review image labels.

Checkpoint migration occurs only when a stage is explicitly run. Old unstamped Qwen
records are archived under checkpoint_archive, because caption-gated proposal/alignment
semantics are incompatible. New Qwen records have per-input signatures covering prompts,
configuration and semantic evidence. Changes to peers invalidate dependent expressions.
Refinements also bind to their initial verification, so changed r0 cannot inherit old r1.

Grounding caches reuse exact image/query/model/environment/mask settings; completed zero
detection queries count as completed. Partial writes without a valid success marker are
recomputed. Current request metadata is rebound onto reused boxes, so category results
refer to the new proposal IDs. Legacy import checks the local legacy metadata's model settings
and manifest. Missing or unverifiable legacy provenance is conservatively recomputed.
Per-worker request manifests prevent one worker's migration from losing the others'
request identities. Combined detections include only current requests.

The managed runner probes Qwen stages using --check-only (0=complete, 3=pending), checks
grounder progress through the worker protocol, and drains refinement drafts even when no
new draft is generated. Models remain sequential on one H200. No model/GPU run was
performed during implementation; therefore no new real-model recall, promotion, final
count or comparison to 214 is claimed. Existing outputs remain the old baseline.
