#!/usr/bin/env bash
# Shared runtime configuration for command-line entry points.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.local/runtime.sh" ]]; then
  source "${PROJECT_ROOT}/.local/runtime.sh"
fi
export WORKFLOW_PYTHON="${WORKFLOW_PYTHON:-python}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
