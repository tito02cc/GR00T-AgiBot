#!/usr/bin/env bash
# Upload only converted train/heldout data; no raw JPEGs, weights or credentials.
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TASK_DIR}/../../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOCAL_DATA="${REPO_ROOT}/agibot/gr00t_data/g2_194/xichong_right_grasp_r0002_400"
LOCAL_REPORT="${REPO_ROOT}/agibot/local_reports/g2_194/xichong_right_grasp_r0002_prepare_20260920"
CLOUD_ROOT=/root/gpufree-data/GR00T
CLOUD_DATA=${CLOUD_ROOT}/datasets/xichong_right_grasp_r0002_400
SSH_TARGET="${CLOUD_SSH_TARGET:?Set CLOUD_SSH_TARGET, e.g. root@your-host}"
SSH_PORT="${CLOUD_SSH_PORT:?Set CLOUD_SSH_PORT}"
SSH_ARGS=(-p "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
if [[ -n "${CLOUD_SSH_CONTROL_PATH:-}" ]]; then
  SSH_ARGS+=(-o "ControlPath=${CLOUD_SSH_CONTROL_PATH}")
fi
printf -v RSYNC_RSH '%q ' ssh "${SSH_ARGS[@]}"
trap 'result=$?; echo "UPLOAD_EXIT_CODE=${result}"; date --iso-8601=seconds' EXIT
"${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "AUTOMATED_DATA_CHECKS_PASS"' \
  "${LOCAL_REPORT}/preparation_status.json"
mkdir -p "${LOCAL_REPORT}/cloud"
for split in train heldout; do
  "${PYTHON}" "${REPO_ROOT}/agibot/training/check_cloud_inputs.py" inventory \
    --root "${LOCAL_DATA}/${split}" \
    --inventory "${LOCAL_REPORT}/cloud/xichong_right_grasp_r0002_${split}.inventory.json"
done
ssh "${SSH_ARGS[@]}" "${SSH_TARGET}" \
  "mkdir -p '${CLOUD_DATA}' '${CLOUD_ROOT}/manifests'"
echo "UPLOAD_STARTED $(date --iso-8601=seconds)"
if [[ -n "${CLOUD_SSH_CONTROL_PATH_2:-}" ]]; then
  "${PYTHON}" "${TASK_DIR}/partition_upload.py" --report-dir "${LOCAL_REPORT}/cloud"
  SSH_ARGS_2=(-p "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 \
    -o "ControlPath=${CLOUD_SSH_CONTROL_PATH_2}")
  printf -v RSYNC_RSH_2 '%q ' ssh "${SSH_ARGS_2[@]}"
  task_pids=()
  for part in 0 1; do
    task_transport="${RSYNC_RSH}"
    if [[ "${part}" == 1 ]]; then task_transport="${RSYNC_RSH_2}"; fi
    rsync -rt --partial --partial-dir=.rsync-partial --info=progress2 --stats \
      --files-from="${LOCAL_REPORT}/cloud/upload_part_${part}.txt" \
      -e "${task_transport}" "${LOCAL_DATA}/" "${SSH_TARGET}:${CLOUD_DATA}/" \
      > "${LOCAL_REPORT}/cloud/upload_part_${part}.log" 2>&1 &
    task_pids+=("$!")
  done
  task_result=0
  for task_pid in "${task_pids[@]}"; do
    if ! wait "${task_pid}"; then task_result=1; fi
  done
  if [[ "${task_result}" != 0 ]]; then exit "${task_result}"; fi
else
  rsync -rt --partial --partial-dir=.rsync-partial --info=progress2 --stats \
    -e "${RSYNC_RSH}" "${LOCAL_DATA}/" "${SSH_TARGET}:${CLOUD_DATA}/"
fi
for split in train heldout; do
  rsync -rt -e "${RSYNC_RSH}" \
    "${LOCAL_REPORT}/cloud/xichong_right_grasp_r0002_${split}.inventory.json" \
    "${SSH_TARGET}:${CLOUD_ROOT}/manifests/"
  ssh "${SSH_ARGS[@]}" "${SSH_TARGET}" \
    "'${CLOUD_ROOT}/Isaac-GR00T/.venv/bin/python' \
    '${CLOUD_ROOT}/Isaac-GR00T/agibot/training/check_cloud_inputs.py' check \
    --root '${CLOUD_DATA}/${split}' \
    --inventory '${CLOUD_ROOT}/manifests/xichong_right_grasp_r0002_${split}.inventory.json'"
done
echo "UPLOAD_COMPLETE_INVENTORY_PASS $(date --iso-8601=seconds)"
