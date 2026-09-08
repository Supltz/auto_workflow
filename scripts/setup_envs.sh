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
create_base rex 3.11
create_base sam3 3.12
create_base groundingdino 3.11

"${ENV_ROOT}/qwen/bin/python" -m pip install ninja "${QWEN_VLLM_WHEEL}" \
  --extra-index-url https://download.pytorch.org/whl/cu129

"${ENV_ROOT}/rex/bin/python" -m pip install --index-url "${PYTORCH_INDEX}" \
  "torch==2.7.1" "torchvision==0.22.1"
"${ENV_ROOT}/rex/bin/python" -m pip install \
  "numpy==1.26.4" "Pillow==10.4.0" "qwen_vl_utils==0.0.14" \
  "transformers==4.51.3" "accelerate==1.10.1" "pydantic>=2.10" PyYAML
"${ENV_ROOT}/rex/bin/python" -m pip install --no-deps -e "${PROJECT_ROOT}/third_party/Rex-Omni"

"${ENV_ROOT}/sam3/bin/python" -m pip install --index-url "${PYTORCH_INDEX}" \
  "torch==2.7.1" "torchvision==0.22.1"
"${ENV_ROOT}/sam3/bin/python" -m pip install "setuptools<81" einops pycocotools psutil
"${ENV_ROOT}/sam3/bin/python" -m pip install "pydantic>=2.10" PyYAML Pillow \
  -e "${PROJECT_ROOT}/third_party/sam3"

"${MAMBA}" install -y -p "${ENV_ROOT}/groundingdino" -c nvidia cuda-nvcc=12.8
"${ENV_ROOT}/groundingdino/bin/python" -m pip install --index-url "${PYTORCH_INDEX}" \
  "torch==2.7.1" "torchvision==0.22.1"
"${ENV_ROOT}/groundingdino/bin/python" -m pip install \
  "pydantic>=2.10" PyYAML "transformers==4.51.3" addict yapf timm opencv-python-headless \
  "supervision>=0.22.0" pycocotools ninja
CUDA_TARGET="${ENV_ROOT}/groundingdino/targets/x86_64-linux"
NVIDIA_PACKAGES="${ENV_ROOT}/groundingdino/lib/python3.11/site-packages/nvidia"
CUDA_HEADERS="${CUDA_TARGET}/include:${NVIDIA_PACKAGES}/cusparse/include:${NVIDIA_PACKAGES}/cublas/include:${NVIDIA_PACKAGES}/cusolver/include"
GDINO_BUILD_DIR="$(mktemp -d /tmp/groundingdino-build.XXXXXXXX)"
cp -a "${PROJECT_ROOT}/third_party/GroundingDINO/." "${GDINO_BUILD_DIR}/"
git -C "${GDINO_BUILD_DIR}" apply "${PROJECT_ROOT}/requirements/groundingdino-pytorch-2.7.patch"
PATH="${ENV_ROOT}/groundingdino/bin:${PATH}" \
CUDA_HOME="${ENV_ROOT}/groundingdino" \
CPATH="${CUDA_HEADERS}${CPATH:+:${CPATH}}" \
LIBRARY_PATH="${CUDA_TARGET}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
LD_LIBRARY_PATH="${CUDA_TARGET}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 \
  "${ENV_ROOT}/groundingdino/bin/python" -m pip install \
  --no-build-isolation --no-deps "${GDINO_BUILD_DIR}"
