#!/usr/bin/env bash
# Start the read-only right-arm observation bridge on the robot.
#
# The bridge exposes no control API and sends no motor commands: it publishes
# synchronized camera frames plus the right EEF pose and gripper joint state.
# Nothing here actuates the robot.
#
# Intended to be launched detached; the caller tunnels 19100 -> 9100.

# ``env.sh`` dereferences $1, so no nounset here.
source /home/agi/app/env.sh
cd /home/agi/vla_ct/bridges/10.20.15.194

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=/tmp/observation_bridge_$STAMP.log

setsid python3 -u g2_groot_right_observation_bridge.py \
  --bind-host 127.0.0.1 --port 9100 \
  --gripper-joint-name idx71_gripper_r_inner_joint1 \
  --gripper-feedback-encoding native_radians \
  --model-input-jpeg-quality 92 \
  --camera-timeout-ms 500 \
  --max-camera-skew-ms 100 \
  --max-state-camera-skew-ms 50 \
  > "$LOG" 2>&1 < /dev/null &

BRIDGE_PID=$!
echo "BRIDGE_PID=$BRIDGE_PID LOG=$LOG"

for _ in $(seq 1 40); do
  if (ss -ltn 2>/dev/null || netstat -ltn) | grep -q ":9100"; then
    echo "OBSERVATION_BRIDGE_LISTENING"
    break
  fi
  sleep 1
done
tail -6 "$LOG"
