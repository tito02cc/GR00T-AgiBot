#!/usr/bin/env bash
# Tier 0: read-only right-omnipicker probe.
#
# Starts the gripper command daemon and issues only ``ping`` and ``status``.
# No ``command`` operation is sent, so no actuator motion is requested.
# Purpose: capture real jaw feedback to validate the mux's fault classifier.
# ``env.sh`` dereferences $1, so no nounset here.
source /home/agi/app/env.sh
cd /home/agi/vla_ct/bridges/10.20.15.194

python3 -u g2_groot_right_gripper_command_daemon.py \
  --bind-host 127.0.0.1 --port 9300 --session-limit-s 45 \
  --confirm ENABLE_G2_GROOT_RIGHT_GRIPPER_COMMAND_DAEMON \
  > /tmp/tier0_daemon.log 2>&1 &
DAEMON_PID=$!

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


print("PING     :", json.dumps(call({"op": "ping"})))
for index in range(5):
    print(f"STATUS[{index}]:", json.dumps(call({"op": "status"})))
    time.sleep(0.4)
print("SHUTDOWN :", json.dumps(call({"op": "shutdown"})))
PY

wait "$DAEMON_PID" 2>/dev/null
echo "=== daemon stdout/stderr ==="
cat /tmp/tier0_daemon.log
echo "=== residual processes ==="
pgrep -af g2_groot || echo "  none"
