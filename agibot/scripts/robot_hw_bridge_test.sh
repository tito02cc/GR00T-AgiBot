#!/usr/bin/env bash
# Real-hardware placement release handoff test.
#
# Starts the real mux in the production deploy directory, so it launches the
# real Cartesian arm child and the real omnipicker daemon.  Nothing simulated.
#
# The workspace box is centred on the arm's measured pose with a 0.02 m
# half-width.  The controller adds its own 0.01 m margin, so the arm is
# confined to 0.03 m per axis around where it already is, and every commanded
# waypoint is the arm's own desired pose.  The box is a test envelope for a
# stationary arm; it is not a widened box used to obtain a task pass.

# ``env.sh`` dereferences $1, so no nounset here.
source /home/agi/app/env.sh
DEPLOY=/home/agi/vla_ct/bridges/10.20.15.194
cd "$DEPLOY"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== residual process check ==="
pgrep -af g2_groot && { echo "ABORT: control processes already running"; exit 1; }
echo "  clean"

echo "=== pre-closing the empty jaw so release_phase starts closed ==="
python3 -u g2_groot_right_gripper_command_daemon.py \
  --bind-host 127.0.0.1 --port 9300 --session-limit-s 40 \
  --confirm ENABLE_G2_GROOT_RIGHT_GRIPPER_COMMAND_DAEMON \
  > "/tmp/hw_preclose_$STAMP.log" 2>&1 &
PRECLOSE_PID=$!
python3 - <<'PY'
import json
import socket
import time

ADDRESS = ("127.0.0.1", 9300)


def call(payload, attempts=40):
    for _ in range(attempts):
        try:
            with socket.create_connection(ADDRESS, timeout=3) as connection:
                connection.sendall((json.dumps(payload) + "\n").encode())
                return json.loads(connection.makefile("rb").readline(8192))
        except Exception:
            time.sleep(0.5)
    return {"ok": False, "message": "daemon unreachable"}


print("  close cmd :", json.dumps(call({"op": "command", "target": 0.0})))
state = {}
for _ in range(40):
    time.sleep(0.2)
    state = call({"op": "status"})
    if state.get("position", -1) >= -0.05:
        break
print("  closed    :", json.dumps(state))
call({"op": "shutdown"})
PY
wait "$PRECLOSE_PID" 2>/dev/null

echo "=== measuring the arm pose to build a tight workspace box ==="
BOX=$(python3 - <<'PY'
import sys
import time

sys.path.insert(0, "/home/agi/vla_ct/bridges/10.20.15.194")
import agibot_gdk
from g2_groot_trajectory_tracking_probe import RIGHT_FRAME, pose_values

HALF_WIDTH = 0.02
if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
    raise SystemExit("gdk_init failed")
try:
    tf = agibot_gdk.TF()
    time.sleep(2.0)
    pose = pose_values(tf, RIGHT_FRAME)
    low = [pose[i] - HALF_WIDTH for i in range(3)]
    high = [pose[i] + HALF_WIDTH for i in range(3)]
    print(" ".join(f"{value:.4f}" for value in low + high))
finally:
    agibot_gdk.gdk_release()
PY
)
BOX=$(echo "$BOX" | tail -1)
echo "  box: $BOX"
read -r WMINX WMINY WMINZ WMAXX WMAXY WMAXZ <<< "$BOX"
if [ -z "$WMAXZ" ]; then echo "ABORT: could not measure the arm pose"; exit 1; fi

echo
echo "=== starting the real mux (real arm child + real gripper daemon) ==="
python3 -u g2_groot_place_action_mux.py \
  --bind-host 127.0.0.1 --port 9200 \
  --backend-port 9201 --gripper-port 9300 \
  --required-motion-mode 1 \
  --workspace-min "$WMINX" "$WMINY" "$WMINZ" \
  --workspace-max "$WMAXX" "$WMAXY" "$WMAXZ" \
  --session-limit-s 120 \
  --confirm ENABLE_G2_GROOT_PLACE_ACTION_MUX \
  > "/tmp/hw_mux_$STAMP.log" 2>&1 &
MUX_PID=$!

for _ in $(seq 1 90); do
  grep -q "mux_ready" "/tmp/hw_mux_$STAMP.log" 2>/dev/null && break
  kill -0 "$MUX_PID" 2>/dev/null || { echo "MUX EXITED EARLY"; break; }
  sleep 1
done

echo "=== bridge test driver ==="
python3 -u /tmp/hw_place_release_bridge_test.py \
  --mux-port 9200 \
  --report "/tmp/hw_bridge_report_$STAMP.json" \
  --confirm EXECUTE_G2_GROOT_HW_PLACE_RELEASE_BRIDGE_TEST
DRIVER_RC=$?

wait "$MUX_PID" 2>/dev/null

echo
echo "=== mux + child log ==="
cat "/tmp/hw_mux_$STAMP.log"
echo
echo "=== residual processes ==="
pgrep -af g2_groot || echo "  none"
echo "REPORT=/tmp/hw_bridge_report_$STAMP.json RC=$DRIVER_RC"
