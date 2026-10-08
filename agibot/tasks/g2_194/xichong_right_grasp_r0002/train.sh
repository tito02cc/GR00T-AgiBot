#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_1xa10080.env"
MODE="${1:-audit}"
# A full checkpoint is about 24 GiB; saving can briefly retain three.
# Smoke can also export about 12 GiB of final root weights. Never auto-delete.
case "${MODE}" in
  baseline) REQUIRED_FREE_GIB=80 ;;
  smoke) REQUIRED_FREE_GIB=40 ;;
  resume) REQUIRED_FREE_GIB=32 ;;
  *) REQUIRED_FREE_GIB=0 ;;
esac
if (( REQUIRED_FREE_GIB > 0 )); then
  AVAILABLE_BYTES="$(df -B1 --output=avail "${CT_ROOT}" | tail -n 1)"
  if (( AVAILABLE_BYTES < REQUIRED_FREE_GIB * 1024 * 1024 * 1024 )); then
    echo "Insufficient data-disk space for ${MODE}: need ${REQUIRED_FREE_GIB} GiB free." >&2
    echo "Review smoke/old artifacts first; this script will not delete them automatically." >&2
    exit 7
  fi
fi
exec bash "${GROOT_REPO_ROOT}/agibot/training/launch_1xa10080.sh" "${MODE}"
