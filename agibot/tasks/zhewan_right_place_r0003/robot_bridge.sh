#!/usr/bin/env bash
# Run on 194. Default is read-only standby; control still requires runner activation.
set -euo pipefail
if (( $# > 1 )); then
  echo 'Usage: bash robot_bridge.sh [standby|control]' >&2
  exit 2
fi
case "${1:-standby}" in
  standby) control_args=() ;;
  control) control_args=(--enable-control) ;;
  *) echo 'Expected standby or control' >&2; exit 2 ;;
esac
# env.sh interprets its caller's $1 as the application root. Do not leak
# "standby"/"control" into that vendor script.
set --
set +u
source /home/agi/app/env.sh
set -u
groot_bridge_dir=/home/agi/vla_ct/bridges/10.20.15.194/candidates/h16_collision_latched_20260911
export PYTHONPATH="$groot_bridge_dir:/home/agi/vla_ct/bridges/10.20.15.194:${PYTHONPATH:-}"
cd "$groot_bridge_dir"
# Existing task envelope, NOT a claim of a collision-free contact workspace.
exec python3 -u g2_groot_continuous_action_bridge.py \
  --initial-gripper-command 0 \
  --workspace-min 0.3858503 -0.3010741 0.8031590 \
  --workspace-max 0.7694756 -0.1595245 1.0784100 \
  --required-motion-mode 1 --required-control-mode 3 \
  --freeze-compensation-after-calibration \
  --gripper-joint-name idx71_gripper_r_inner_joint1 \
  --bind-host 127.0.0.1 --port 9200 --session-limit-s 1800 \
  "${control_args[@]}"
