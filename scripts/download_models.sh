#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"
STORE="${REGION_BENCHMARK_STORE:-${PROJECT_ROOT}}"
HF_CACHE="${STORE}/cache/huggingface"
HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
HF_CLI="${HF_CLI:-hf}"
export HF_TOKEN_PATH

mkdir -p "${STORE}/models/qwen38_27b" "${STORE}/models/rex_omni" \
  "${STORE}/models/sam3_1" "${STORE}/models/grounding_dino" \
  "${STORE}/third_party"

clone_if_missing() {
  local url="$1"
  local destination="$2"
  if [[ ! -d "${destination}/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "${url}" "${destination}"
  fi
}

download_verified() {
  local url="$1"
  local destination="$2"
  local expected_sha256="$3"
  if [[ -f "${destination}" ]]; then
    local current_sha256
    current_sha256="$(sha256sum "${destination}" | cut -d' ' -f1)"
    if [[ "${current_sha256}" == "${expected_sha256}" ]]; then
      echo "Verified existing checkpoint: ${destination}"
      return
    fi
  fi
  curl --fail --location --retry 5 --continue-at - --output "${destination}" "${url}"
  local downloaded_sha256
  downloaded_sha256="$(sha256sum "${destination}" | cut -d' ' -f1)"
  if [[ "${downloaded_sha256}" != "${expected_sha256}" ]]; then
    echo "SHA-256 mismatch for ${destination}" >&2
    return 1
  fi
}

clone_if_missing https://github.com/IDEA-Research/Rex-Omni.git "${STORE}/third_party/Rex-Omni"
clone_if_missing https://github.com/facebookresearch/sam3.git "${STORE}/third_party/sam3"
clone_if_missing https://github.com/IDEA-Research/GroundingDINO.git "${STORE}/third_party/GroundingDINO"

HF_HOME="${HF_CACHE}" "${HF_CLI}" download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir "${STORE}/models/qwen38_27b"
HF_HOME="${HF_CACHE}" "${HF_CLI}" download IDEA-Research/Rex-Omni \
  --revision 0e5693d24657f6c0e091008dd6809bb1bd28988c \
  --local-dir "${STORE}/models/rex_omni"

download_verified \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth \
  "${STORE}/models/grounding_dino/groundingdino_swinb_cogcoor.pth" \
  46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7

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
