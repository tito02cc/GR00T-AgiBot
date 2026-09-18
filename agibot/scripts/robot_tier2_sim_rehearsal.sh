#!/usr/bin/env bash
# Tier 2 (simulated arm): rehearse the placement release handoff.
#
# The mux resolves its children relative to its own file, so the harness
# directory holds a copy of the real mux, the real gripper daemon, and the
# simulated arm child under the child filename the mux launches.  The
# production deploy directory is not modified.
#
# Real: mux, gripper daemon, omnipicker hardware.
# Simulated: arm child (publishes nothing), policy chunks.

# ``env.sh`` dereferences $1, so no nounset here.
source /home/agi/app/env.sh
DEPLOY=/home/agi/vla_ct/bridges/10.20.15.194
HARNESS=$DEPLOY/sim_rehearsal_harness
STAMP=$(date +%Y%m%d_%H%M%S)
rm -f /tmp/sim_arm_child_pose.json

# The mux latches release_phase to "open" when the jaw is already open at
# startup, which skips the handoff entirely.  Close the empty jaw first so the
# rehearsal starts from the same tool state as a workpiece-holding run.
echo "=== pre-closing the empty jaw so the handoff is actually exercised ==="
python3 -u "$DEPLOY/g2_groot_right_gripper_command_daemon.py" \
  --bind-host 127.0.0.1 --port 9300 --session-limit-s 40 \
  --confirm ENABLE_G2_GROOT_RIGHT_GRIPPER_COMMAND_DAEMON \
  > /tmp/rehearsal_preclose.log 2>&1 &
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
for _ in range(40):
    time.sleep(0.2)
    state = call({"op": "status"})
    if state.get("position", -1) >= -0.05:
        break
print("  closed    :", json.dumps(state))
call({"op": "shutdown"})
PY
wait "$PRECLOSE_PID" 2>/dev/null

mkdir -p "$HARNESS"
cp -f "$DEPLOY/g2_groot_place_action_mux.py" "$HARNESS/"
cp -f "$DEPLOY/g2_groot_right_gripper_command_daemon.py" "$HARNESS/"
cp -f /tmp/sim_g2_groot_arm_child_stub.py \
      "$HARNESS/g2_groot_persistent_h1_action_bridge.py"
cp -f /tmp/sim_place_release_rehearsal.py "$HARNESS/"

echo "=== harness contents ==="
ls -la "$HARNESS"
echo "=== mux copy is byte-identical to the deployed mux ==="
sha256sum "$DEPLOY/g2_groot_place_action_mux.py" "$HARNESS/g2_groot_place_action_mux.py"
echo "=== the simulated child owns no GDK objects ==="
grep -c "agibot_gdk" "$HARNESS/g2_groot_persistent_h1_action_bridge.py" \
  || echo "  0 references to agibot_gdk"

cd "$HARNESS"
echo
echo "=== starting the real mux (it will launch the real daemon + simulated arm) ==="
python3 -u g2_groot_place_action_mux.py \
  --bind-host 127.0.0.1 --port 9200 \
  --backend-port 9201 --gripper-port 9300 \
  --required-motion-mode 1 \
  --workspace-min 0.4172 -0.2392 0.9750 \
  --workspace-max 0.8403 -0.1031 1.2674 \
  --session-limit-s 90 \
  --confirm ENABLE_G2_GROOT_PLACE_ACTION_MUX \
  > "/tmp/rehearsal_mux_$STAMP.log" 2>&1 &
MUX_PID=$!

for _ in $(seq 1 60); do
  if grep -q "mux_ready" "/tmp/rehearsal_mux_$STAMP.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$MUX_PID" 2>/dev/null; then
    echo "MUX EXITED EARLY"
    break
  fi
  sleep 1
done

echo
echo "=== rehearsal driver ==="
python3 -u sim_place_release_rehearsal.py \
  --mux-port 9200 \
  --report "/tmp/rehearsal_report_$STAMP.json" \
  --confirm REHEARSE_G2_GROOT_PLACE_RELEASE_HANDOFF
DRIVER_RC=$?

wait "$MUX_PID" 2>/dev/null

echo
echo "=== mux log ==="
cat "/tmp/rehearsal_mux_$STAMP.log"
echo
echo "=== report path: /tmp/rehearsal_report_$STAMP.json (rc=$DRIVER_RC) ==="
echo "=== residual processes ==="
pgrep -af "g2_groot|sim_place_release" || echo "  none"
echo "REPORT_STAMP=$STAMP"
