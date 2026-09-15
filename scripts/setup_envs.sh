#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"
MAMBA="${MAMBA:-mamba}"
ENV_ROOT="${ENV_ROOT:-${PROJECT_ROOT}/envs}"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
QWEN_VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
STORE="${REGION_BENCHMARK_STORE:-${PROJECT_ROOT}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${STORE}/cache/pip}"
export PIP_USER=0
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${STORE}/cache/conda-pkgs}"
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"

create_base() {
  local name="$1"
  local python_version="$2"
  if [[ ! -x "${ENV_ROOT}/${name}/bin/python" ]]; then
    "${MAMBA}" create -y -p "${ENV_ROOT}/${name}" "python=${python_version}" pip
  fi
}

create_base qwen 3.11
create_base sam3 3.12

"${ENV_ROOT}/qwen/bin/python" -m pip install ninja "${QWEN_VLLM_WHEEL}" \
  --extra-index-url https://download.pytorch.org/whl/cu129

# EGM uses the same Qwen3VL-capable Transformers environment; no duplicate env.
"${ENV_ROOT}/qwen/bin/python" -c 'from transformers import Qwen3VLForConditionalGeneration, AutoProcessor'

"${ENV_ROOT}/sam3/bin/python" -m pip install --index-url "${PYTORCH_INDEX}" \
  "torch==2.7.1" "torchvision==0.22.1"
"${ENV_ROOT}/sam3/bin/python" -m pip install "setuptools<81" einops pycocotools psutil
"${ENV_ROOT}/sam3/bin/python" -m pip install "pydantic>=2.10" PyYAML Pillow \
  -e "${PROJECT_ROOT}/third_party/sam3"
