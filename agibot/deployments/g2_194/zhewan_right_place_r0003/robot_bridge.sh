#!/usr/bin/env bash
# Pinned h16_collision_latched_20260911; default standby never activates motion.
set -euo pipefail
if (( $# > 1 )); then
  echo 'Usage: bash robot_bridge.sh [standby|control|observation]' >&2
  exit 2
fi
groot_task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
groot_run_mode="${1:-standby}"
case "$groot_run_mode" in
  standby) control_args=() ;;
  control) control_args=(--enable-control) ;;
  observation) control_args=() ;;
  *) echo 'Expected standby, control or observation' >&2; exit 2 ;;
esac
# The vendor env script consumes its caller's $1.
set --
set +u
source /home/agi/app/env.sh
set -u
export PYTHONPATH="$groot_task_dir/robot:${PYTHONPATH:-}"
cd "$groot_task_dir/robot"
if [[ "$groot_run_mode" == observation ]]; then
  exec python3 -u g2_groot_right_observation_bridge.py \
    --bind-host 127.0.0.1 --port 9100 \
    --gripper-joint-name idx71_gripper_r_inner_joint1 \
    --gripper-feedback-encoding native_radians --model-input-jpeg-quality 92 \
    --camera-timeout-ms 500 --max-camera-skew-ms 100 --max-state-camera-skew-ms 50
fi
exec python3 -u g2_groot_continuous_action_bridge.py \
  --initial-gripper-command 0 --workspace-min 0.3858503 -0.3010741 0.803159 --workspace-max 0.7694756 -0.1595245 1.07841 --required-motion-mode 1 --calibration-duration-s 2.0 --gripper-joint-name idx71_gripper_r_inner_joint1 --bind-host 127.0.0.1 --port 9200 --session-limit-s 1800 --required-control-mode 3 --freeze-compensation-after-calibration \
  "${control_args[@]}"
