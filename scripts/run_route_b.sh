#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"
cd "${PROJECT_ROOT}"
exec "${WORKFLOW_PYTHON}" -m src.routes.route_b_caption "$@"
