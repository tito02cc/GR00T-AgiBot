#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "preflight" && "${MODE}" != "audit" && \
      "${MODE}" != "smoke" && "${MODE}" != "baseline" && \
      "${MODE}" != "resume" ]]; then
  echo "Usage: $0 {preflight|audit|smoke|baseline|resume}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=xichong_right_place_r0002_1xa10080.env
source "${SCRIPT_DIR}/xichong_right_place_r0002_1xa10080.env"

if [[ "${MODE}" == "preflight" ]]; then
  exec bash "${SCRIPT_DIR}/preflight_1xa10080.sh"
fi
exec bash "${SCRIPT_DIR}/launch_1xa10080.sh" "${MODE}"
