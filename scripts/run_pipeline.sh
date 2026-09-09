#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"
RUN_ID="${WORKFLOW_RUN_ID:-$(date +%Y%m%dT%H%M%S)-$}"
RUN_LOG_DIR="${WORKFLOW_LOG_DIR:-${PROJECT_ROOT}/run_logs}"
START_INDEX="${ROUTE_B_START_INDEX:-0}"
END_INDEX="${ROUTE_B_END_INDEX:-100}"
QWEN_PID=""
ACTIVE_STEP_PID=""
FINISHED=0

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_LOG_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}"
export QWEN_IMAGE_LIMIT=6

stop_qwen() {
  if [[ -n "${QWEN_PID}" ]]; then
    kill -TERM "${QWEN_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${QWEN_PID}" 2>/dev/null || break
      sleep 2
    done
    kill -TERM -- "-${QWEN_PID}" 2>/dev/null || true
    kill -KILL -- "-${QWEN_PID}" 2>/dev/null || true
    wait "${QWEN_PID}" 2>/dev/null || true
    QWEN_PID=""
  fi
}

stop_active_step() {
  if [[ -n "${ACTIVE_STEP_PID}" ]]; then
    kill -TERM -- "-${ACTIVE_STEP_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${ACTIVE_STEP_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${ACTIVE_STEP_PID}" 2>/dev/null || true
    wait "${ACTIVE_STEP_PID}" 2>/dev/null || true
    ACTIVE_STEP_PID=""
  fi
}

run_step() {
  local status=0
  setsid "$@" &
  ACTIVE_STEP_PID=$!
  wait "${ACTIVE_STEP_PID}" || status=$?
  ACTIVE_STEP_PID=""
  return "${status}"
}

handle_signal() {
  trap '' USR1 TERM
  stop_active_step
  stop_qwen
  # A distinct exit status allows the caller to decide how to resume.
  exit 75
}

start_qwen() {
  local label="$1"
  local server_log="${RUN_LOG_DIR}/qwen-${RUN_ID}-${label}.log"
  setsid bash scripts/serve_qwen38.sh >"${server_log}" 2>&1 &
  QWEN_PID=$!
  for _ in $(seq 1 180); do
    if curl --silent --fail http://127.0.0.1:8000/v1/models >/dev/null; then
      echo "[$(date --iso-8601=seconds)] Qwen ready for ${label}"
      return 0
    fi
    if ! kill -0 "${QWEN_PID}" 2>/dev/null; then
      echo "Qwen exited before becoming ready for ${label}"
      tail -n 200 "${server_log}"
      return 1
    fi
    sleep 5
  done
  echo "Qwen did not become ready within 15 minutes for ${label}"
  tail -n 200 "${server_log}"
  return 1
}

# The stage's semantic input signatures and completed records determine readiness.
# Probe exit 3 means pending; any other failure aborts rather than guessing completion.
qwen_stage() {
  echo "[$(date --iso-8601=seconds)] stage=$1 checking checkpoints"
  local stage="$1"
  local status=0
  "${WORKFLOW_PYTHON}" -m src.routes.route_b_caption --stage "${stage}" --check-only \
    --start-index "${START_INDEX}" --end-index "${END_INDEX}" || status=$?
  if (( status == 0 )); then
    echo "Completed: ${stage}"
    if [[ "${stage}" == finalize ]]; then
      cpu_or_ground_stage finalize
    fi
    return 0
  fi
  if (( status != 3 )); then
    return "${status}"
  fi
  if [[ -z "${QWEN_PID}" ]]; then
    start_qwen "${stage}"
  fi
  run_step bash scripts/run_route_b.sh --stage "${stage}" --resume \
    --start-index "${START_INDEX}" --end-index "${END_INDEX}"
  echo "[$(date --iso-8601=seconds)] stage=${stage} DONE"
}

cpu_or_ground_stage() {
  echo "[$(date --iso-8601=seconds)] stage=$1 START"
  run_step bash scripts/run_route_b.sh --stage "$1" --resume \
    --start-index "${START_INDEX}" --end-index "${END_INDEX}"
  echo "[$(date --iso-8601=seconds)] stage=$1 DONE"
}

trap handle_signal USR1 TERM
trap stop_qwen EXIT

echo "[$(date --iso-8601=seconds)] Route B seeded random sample range [${START_INDEX},${END_INDEX}), run=${RUN_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

qwen_stage entities
stop_qwen
cpu_or_ground_stage ground
cpu_or_ground_stage aggregate

for stage in align promote bbox_verify ocr describe; do
  qwen_stage "${stage}"
done
stop_qwen

cpu_or_ground_stage reground
qwen_stage expression_verify
stop_qwen

# Always drain existing drafts before treating a round as done. Generating no new
# drafts does NOT imply that Rex/SAM/DINO or the verifier finished the existing ones.
for round in 1 2; do
  qwen_stage refine_generate
  stop_qwen
  cpu_or_ground_stage refine_reground
  qwen_stage refine_verify
  stop_qwen
done

qwen_stage finalize
stop_qwen
run_step bash scripts/run_route_b.sh --stage review --overwrite \
  --start-index "${START_INDEX}" --end-index "${END_INDEX}"

"${WORKFLOW_PYTHON}" - <<'PY'
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.routes.route_b_checkpoint import CHECK_ONLY
from src.routes.target_dedup import TargetDeduplicator
from src.utils.config import load_yaml

CHECK_ONLY.set(True)
dedup_config = load_yaml("configs/route_b.yaml")
deduplicator = TargetDeduplicator(
    config=dedup_config, qwen_config=load_yaml(dedup_config["models_config"])["qwen"],
    output_root=Path("outputs"))

verified_path = Path("outputs/verified_regions/route_b.jsonl")
review_root = Path("outputs/human_review/route_b")
verified = [json.loads(line) for line in verified_path.open() if line.strip()]
with (review_root / "index.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
images = list(review_root.glob("*.jpg"))
per_source = Counter(row["image_id"] for row in rows)
verified_by_source = defaultdict(list)
for record in verified:
    verified_by_source[record["image_id"]].append(record)
assert len(rows) == len(verified), (len(rows), len(verified))
assert len(images) == len(verified), (len(images), len(verified))
assert max(per_source.values(), default=0) <= int(
    load_yaml("configs/route_b.yaml")["final_max_per_source_image"]
), per_source
assert {row["region_id"] for row in rows} == {record["region_id"] for record in verified}
assert all(row["final_referring_expression"].lower().startswith("the ") for row in rows)
assert all(row["bbox_x1"] and row["bbox_y1"] and row["bbox_x2"] and row["bbox_y2"] for row in rows)
assert all(record["bbox_area_ratio"] <= 0.10 for record in verified)
assert all(min(r["bbox_xyxy"][2] - r["bbox_xyxy"][0],
               r["bbox_xyxy"][3] - r["bbox_xyxy"][1]) >= 32 for r in verified)
assert all(record["bbox_grounder_support"] >= 2 for record in verified)
assert all(
    record["reground_audit"]["passed"]
    and record["reground_audit"]["expression_grounder_support"] >= 2
    and record["expression_verification"]["target_is_unique"]
    for record in verified
)
assert len(
    {
        (record["image_id"], record["final_referring_expression"].strip().casefold())
        for record in verified
    }
) == len(verified)
for source_records in verified_by_source.values():
    for first_index, first in enumerate(source_records):
        for second in source_records[first_index + 1 :]:
            assert not deduplicator.compare(first, second)["same_object"]
print(
    json.dumps(
        {
            "verified_unique_regions": len(verified),
            "human_review_files": len(images),
            "unique_source_images_in_review": len(per_source),
            "max_review_files_per_source": max(per_source.values(), default=0),
            "max_bbox_area_ratio": max(
                (record["bbox_area_ratio"] for record in verified), default=0.0
            ),
        },
        indent=2,
    )
)
PY

FINISHED=1
echo "[$(date --iso-8601=seconds)] Route B valid100 complete"
