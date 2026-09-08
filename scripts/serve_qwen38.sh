#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"
MODEL_PATH="${QWEN_MODEL_PATH:-${PROJECT_ROOT}/models/qwen38_27b}"
QWEN_ENV="${QWEN_ENV:-${PROJECT_ROOT}/envs/qwen}"
QWEN_IMAGE_LIMIT="${QWEN_IMAGE_LIMIT:-3}"
export PATH="${QWEN_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${QWEN_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${QWEN_ENV}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name Qwen/Qwen3.8-27B \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --safetensors-load-strategy prefetch \
  --gdn-prefill-backend triton \
  --limit-mm-per-prompt "{\"image\":${QWEN_IMAGE_LIMIT}}" \
  "$@"
