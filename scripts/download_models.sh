#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"
STORE="${REGION_BENCHMARK_STORE:-${PROJECT_ROOT}}"
HF_CACHE="${STORE}/cache/huggingface"
HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
HF_CLI="${HF_CLI:-hf}"
export HF_TOKEN_PATH

mkdir -p "${STORE}/models/qwen38_27b" "${STORE}/models/egm_8b" \
  "${STORE}/models/sam3_1" \
  "${STORE}/third_party"

clone_if_missing() {
  local url="$1"
  local destination="$2"
  if [[ ! -d "${destination}/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "${url}" "${destination}"
  fi
}

clone_if_missing https://github.com/facebookresearch/sam3.git "${STORE}/third_party/sam3"

HF_HOME="${HF_CACHE}" "${HF_CLI}" download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir "${STORE}/models/qwen38_27b"
HF_HOME="${HF_CACHE}" "${HF_CLI}" download nvidia/EGM-8B \
  --revision f5bc2e9d12386875c827b10ec7fe396e8215d6d6 \
  --local-dir "${STORE}/models/egm_8b"

if [[ -n "${HF_TOKEN:-}" || -s "${HF_TOKEN_PATH}" ]]; then
  if ! HF_HOME="${HF_CACHE}" "${HF_CLI}" download facebook/sam3.1 sam3.1_multiplex.pt \
    --revision daa63191845a41281374e725f4c9e51c7a824460 \
    --local-dir "${STORE}/models/sam3_1"; then
    echo "SAM3.1 remains unavailable: the authenticated account still requires approval." >&2
  fi
else
  echo "SAM3.1 not downloaded: accept the Meta license, authenticate with hf auth login, and rerun." >&2
fi

"${WORKFLOW_PYTHON}" \
  "${PROJECT_ROOT}/scripts/record_resources.py" --output-dir "${PROJECT_ROOT}/artifacts"
