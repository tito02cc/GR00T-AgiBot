#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_1xa10080.env"
exec bash "${GROOT_REPO_ROOT}/agibot/training/launch_1xa10080.sh" "${1:-audit}"
