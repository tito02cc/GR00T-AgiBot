#!/usr/bin/env bash
# Tier 1: right-omnipicker boundary and travel validation on real hardware.
#
# Phase A drives the pre-fix daemon with the float32 representation of the
# training open bound and expects a rejection.  That rejection happens before
# any GDK call, so phase A requests no motion at all.
#
# Phase B drives the fixed daemon with the same value.  The jaw is already at
# -0.785, so this also requests no meaningful motion.
#
# Phase C closes the empty gripper and reopens it through the mux's real
# ``open_gripper`` routine to measure actual travel, timing and refresh count.
# ``ProcessMux.open_gripper`` touches only the tool daemon, never the arm.

# ``env.sh`` dereferences $1, so no nounset here.
source /home/agi/app/env.sh
DEPLOY=/home/agi/vla_ct/bridges/10.20.15.194
cd "$DEPLOY"

FLOAT32_OPEN=-0.7850000262260437

start_daemon() {
  python3 -u "$1" \
    --bind-host 127.0.0.1 --port 9300 --session-limit-s 60 \
    --confirm ENABLE_G2_GROOT_RIGHT_GRIPPER_COMMAND_DAEMON \
    > "$2" 2>&1 &
  echo $!
}

echo "############ PHASE A: pre-fix daemon, boundary value, expect rejection"
PID_A=$(start_daemon "$DEPLOY/prefix_backup_20260907/g2_groot_right_gripper_command_daemon.py" /tmp/tier1_a.log)
FLOAT32_OPEN=$FLOAT32_OPEN python3 - <<'PY'
import json
import os
import socket
import time

ADDRESS = ("127.0.0.1", 9300)
TARGET = float(os.environ["FLOAT32_OPEN"])


def call(payload, attempts=40):
    for _ in range(attempts):
        try:
            with socket.create_connection(ADDRESS, timeout=3) as connection:
                connection.sendall((json.dumps(payload) + "\n").encode())
                return json.loads(connection.makefile("rb").readline(8192))
        except Exception:
            time.sleep(0.5)
    return {"ok": False, "message": "daemon unreachable"}


print("  before   :", json.dumps(call({"op": "status"})))
print(f"  command  : target={TARGET!r}")
print("  response :", json.dumps(call({"op": "command", "target": TARGET})))
print("  after    :", json.dumps(call({"op": "status"})))
call({"op": "shutdown"})
PY
wait "$PID_A" 2>/dev/null

echo
echo "############ PHASE B+C: fixed daemon"
PID_B=$(start_daemon "$DEPLOY/g2_groot_right_gripper_command_daemon.py" /tmp/tier1_b.log)
FLOAT32_OPEN=$FLOAT32_OPEN python3 - <<'PY'
import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, "/home/agi/vla_ct/bridges/10.20.15.194")
import g2_groot_place_action_mux as mux

ADDRESS = ("127.0.0.1", 9300)
TARGET = float(os.environ["FLOAT32_OPEN"])


def call(payload, attempts=40):
    for _ in range(attempts):
        try:
            with socket.create_connection(ADDRESS, timeout=3) as connection:
                connection.sendall((json.dumps(payload) + "\n").encode())
                return json.loads(connection.makefile("rb").readline(8192))
        except Exception:
            time.sleep(0.5)
    return {"ok": False, "message": "daemon unreachable"}


call({"op": "ping"})

print("PHASE B: same boundary value against the fixed daemon")
print("  before   :", json.dumps(call({"op": "status"})))
print(f"  command  : target={TARGET!r}")
print("  response :", json.dumps(call({"op": "command", "target": TARGET})))
time.sleep(0.5)
print("  after    :", json.dumps(call({"op": "status"})))

print()
print("PHASE C: close the empty jaw, then reopen via the real open_gripper()")
print("  closing to 0.0 :", json.dumps(call({"op": "command", "target": 0.0})))
for _ in range(30):
    time.sleep(0.2)
    closed = call({"op": "status"})
    if closed.get("position", -1) >= -0.05:
        break
print("  closed state   :", json.dumps(closed))

args = argparse.Namespace(gripper_port=9300)
supervisor = mux.ProcessMux(args)
started = time.monotonic()
try:
    outcome = supervisor.open_gripper(TARGET)
    print("  OPEN OK        : threshold", mux.OPEN_THRESHOLD)
    for key in ("target", "start_position", "opened_position", "travel",
                "elapsed_s", "commands_sent"):
        print(f"    {key:16s}= {outcome[key]!r}")
    print(f"    samples         = {len(outcome['samples'])}")
    for sample in outcome["samples"]:
        print("      ", json.dumps(sample))
except Exception as error:
    print(f"  OPEN FAILED    : {type(error).__name__}: {error}")

print()
print("  final state    :", json.dumps(call({"op": "status"})))
call({"op": "shutdown"})
PY
wait "$PID_B" 2>/dev/null

echo
echo "=== daemon logs ==="
echo "--- phase A ---"; tail -4 /tmp/tier1_a.log
echo "--- phase B/C ---"; tail -4 /tmp/tier1_b.log
echo "=== residual processes ==="
pgrep -af g2_groot || echo "  none"
