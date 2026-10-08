#!/usr/bin/env bash
# Start the SHADOW inference stack on the inference workstation (10.20.15.170).
#
#   bash start_shadow_stack.sh          start what is missing, then report
#   bash start_shadow_stack.sh stop     stop tunnel + model server
#   bash start_shadow_stack.sh status   report only
#
# SHADOW means: observation bridge (read-only) + model server only.
# The robot ACTION bridge (port 9200) is deliberately NOT started and NOT
# tunnelled, so decoded actions can never reach the robot.
#
# Presence is detected by LISTENING PORT, not by process name, because
# pgrep -f patterns also match the shell running this script.

REPO=/home/admin1/ct/Isaac-GR00T
PY="$REPO/.venv/bin/python3.12"
MODEL="$REPO/agibot/models/xichong_rgrasp_n1d7_checkpoint-30000/model"
ROBOT=agi@10.20.15.60
OBS_LOCAL_PORT=19100
OBS_REMOTE_PORT=9100
MODEL_PORT=5564
TUNNEL_LOG=/tmp/tunnel_${OBS_LOCAL_PORT}.log
SERVER_LOG=/tmp/groot_server.log

port_up() { ss -ltn 2>/dev/null | grep -q ":$1 "; }

pids_on_port() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u; }

start_tunnel() {
  if port_up "$OBS_LOCAL_PORT"; then echo "tunnel $OBS_LOCAL_PORT already up"; return; fi
  setsid nohup ssh -N \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o BatchMode=yes \
    -L "${OBS_LOCAL_PORT}:127.0.0.1:${OBS_REMOTE_PORT}" "$ROBOT" \
    > "$TUNNEL_LOG" 2>&1 < /dev/null &
  sleep 5
  port_up "$OBS_LOCAL_PORT" && echo "tunnel $OBS_LOCAL_PORT started" \
    || { echo "tunnel FAILED, log:"; tail -5 "$TUNNEL_LOG"; }
}

start_server() {
  if port_up "$MODEL_PORT"; then echo "model server $MODEL_PORT already up"; return; fi
  [ -d "$MODEL" ] || { echo "missing model $MODEL"; return 2; }
  cd "$REPO" || return 2
  setsid nohup "$PY" -u gr00t/eval/run_gr00t_server.py \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path "$MODEL" \
    --device cuda:0 --host 127.0.0.1 --port "$MODEL_PORT" \
    > "$SERVER_LOG" 2>&1 < /dev/null &
  echo "model server launching (loads ~15-60 s, ~7.5 GB VRAM)"
}

wait_server() {
  for i in $(seq 1 40); do
    port_up "$MODEL_PORT" && { echo "model server LISTENING after ${i}0s"; return 0; }
    sleep 10
  done
  echo "model server did NOT come up; log:"; tail -20 "$SERVER_LOG"
  return 1
}

stop_all() {
  for p in "$MODEL_PORT" "$OBS_LOCAL_PORT"; do
    local pids; pids=$(pids_on_port "$p")
    if [ -n "$pids" ]; then echo "stopping port $p pids $pids"; kill $pids 2>/dev/null; fi
  done
  sleep 3
  for p in "$MODEL_PORT" "$OBS_LOCAL_PORT"; do
    local pids; pids=$(pids_on_port "$p")
    [ -n "$pids" ] && { echo "force killing port $p pids $pids"; kill -9 $pids 2>/dev/null; }
  done
  echo "stopped"
}

status_all() {
  echo "tunnel  $OBS_LOCAL_PORT : $(port_up $OBS_LOCAL_PORT && echo LISTENING || echo down)"
  echo "model   $MODEL_PORT : $(port_up $MODEL_PORT && echo LISTENING || echo down)"
  echo "action bridge tunnel: deliberately absent (shadow mode)"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
}

case "${1:-start}" in
  stop)   stop_all ;;
  status) status_all ;;
  *)      start_tunnel; start_server; wait_server; echo; status_all ;;
esac
