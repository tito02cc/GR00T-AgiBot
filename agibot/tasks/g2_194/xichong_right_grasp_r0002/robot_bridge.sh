#!/usr/bin/env bash
set -euo pipefail
groot_task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
groot_task_args=("$@")
# Vendor env consumes positional arguments; do not expose ours to it.
set --
set +u
source /home/agi/app/env.sh
set -u
exec python3 -u "$groot_task_dir/robot_bridge.py" "${groot_task_args[@]}"
