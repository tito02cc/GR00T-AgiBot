#!/usr/bin/env bash
set -euo pipefail

GROOT_GDK_ENV="${GROOT_GDK_ENV:-/home/agi/app/env.sh}"
GROOT_BRIDGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -f "$GROOT_GDK_ENV" ]]; then
  echo "missing GDK environment: $GROOT_GDK_ENV" >&2
  exit 2
fi

set +u
source "$GROOT_GDK_ENV"
set -u
exec python3 -u "$GROOT_BRIDGE_DIR/g2_groot_right_observation_bridge.py" \
  --bind-host 127.0.0.1 \
  --port 9100 \
  --camera-timeout-ms 500 \
  --max-camera-skew-ms 100 \
  --max-state-camera-skew-ms 50
