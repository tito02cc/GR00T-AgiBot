#!/usr/bin/env bash
set -euo pipefail

# Audited single-GPU GR00T N1.7 launcher for the Agibot G2 right-arm dataset.
# Usage:
#   bash launch_xichong_1xa10080.sh audit
#   bash launch_xichong_1xa10080.sh smoke
#   bash launch_xichong_1xa10080.sh baseline
#   bash launch_xichong_1xa10080.sh resume

MODE="${1:-}"
if [[ "${MODE}" != "audit" && "${MODE}" != "smoke" && "${MODE}" != "baseline" && "${MODE}" != "resume" ]]; then
  echo "Usage: $0 {audit|smoke|baseline|resume}" >&2
  exit 2
fi

CT_ROOT="${CT_ROOT:?CT_ROOT is required}"
REPO_ROOT="${GROOT_REPO_ROOT:-${CT_ROOT}/Isaac-GR00T}"
PYTHON_ENV="${REPO_ROOT}/.venv"
CUDNN_LIB="${PYTHON_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib"
BASE_MODEL="${GROOT_BASE_MODEL:-${CT_ROOT}/models/GR00T-N1.7-3B}"
BACKBONE_MODEL="${GROOT_BACKBONE_MODEL:-${CT_ROOT}/models/Cosmos-Reason2-2B}"
DATASET="${GROOT_DATASET:-${CT_ROOT}/datasets/xichong_right_single_grasp_300}"
DATASET_MANIFEST="${GROOT_DATASET_MANIFEST:-${DATASET}/SHA256SUMS}"
MODALITY_CONFIG="${GROOT_MODALITY_CONFIG:-${REPO_ROOT}/agibot/configs/xichong_right_single_grasp_config.py}"

case "${MODE}" in
  audit)
    RUN_ID="${RUN_ID:-xichong_rgrasp_n1d7_e300_f10_h16_1xa10080_s30k_ckpt5k_v1}"
    MAX_STEPS="${GROOT_MAX_STEPS:-30000}"
    SAVE_STEPS="${GROOT_SAVE_STEPS:-5000}"
    SAVE_TOTAL_LIMIT="${GROOT_SAVE_TOTAL_LIMIT:-2}"
    ;;
  smoke)
    RUN_ID="${RUN_ID:-xichong_rgrasp_n1d7_e300_f10_h16_1xa10080_smoke_v1}"
    MAX_STEPS=100
    SAVE_STEPS=100
    SAVE_TOTAL_LIMIT=1
    ;;
  baseline)
    RUN_ID="${RUN_ID:-xichong_rgrasp_n1d7_e300_f10_h16_1xa10080_s30k_ckpt5k_v1}"
    MAX_STEPS="${GROOT_MAX_STEPS:-30000}"
    SAVE_STEPS="${GROOT_SAVE_STEPS:-5000}"
    SAVE_TOTAL_LIMIT="${GROOT_SAVE_TOTAL_LIMIT:-2}"
    ;;
  resume)
    RUN_ID="${RUN_ID:-xichong_rgrasp_n1d7_e300_f10_h16_1xa10080_s30k_ckpt5k_v1}"
    MAX_STEPS="${GROOT_MAX_STEPS:-30000}"
    SAVE_STEPS="${GROOT_SAVE_STEPS:-5000}"
    SAVE_TOTAL_LIMIT="${GROOT_SAVE_TOTAL_LIMIT:-2}"
    ;;
esac

OUTPUT_ROOT="${CT_ROOT}/outputs"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_FILE="${CT_ROOT}/logs/${RUN_ID}.log"
if [[ "${MODE}" == "audit" ]]; then
  LOG_FILE="${CT_ROOT}/logs/${RUN_ID}.config_audit.log"
fi

for required in \
  "${PYTHON_ENV}/bin/python" \
  "${CUDNN_LIB}/libcudnn.so.9" \
  "${BASE_MODEL}/config.json" \
  "${BACKBONE_MODEL}/config.json" \
  "${DATASET}/meta/info.json" \
  "${DATASET_MANIFEST}" \
  "${MODALITY_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required training input: ${required}" >&2
    exit 3
  fi
done

(
  cd "${DATASET}"
  sha256sum -c --quiet "${DATASET_MANIFEST}"
)

if [[ "${MODE}" == "resume" ]]; then
  if [[ ! -d "${RUN_OUTPUT_DIR}" ]] || ! compgen -G "${RUN_OUTPUT_DIR}/checkpoint-*" >/dev/null; then
    echo "No resumable checkpoint found under: ${RUN_OUTPUT_DIR}" >&2
    exit 5
  fi
elif [[ "${MODE}" != "audit" && -e "${RUN_OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse existing output directory: ${RUN_OUTPUT_DIR}" >&2
  echo "Set a new RUN_ID, or use the explicit resume workflow." >&2
  exit 4
fi

mkdir -p "${OUTPUT_ROOT}" "${CT_ROOT}/logs" \
  "${CT_ROOT}/cache/huggingface" "${CT_ROOT}/cache/torch"

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="${CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${CT_ROOT}/cache/huggingface"
export TORCH_HOME="${CT_ROOT}/cache/torch"
export UV_CACHE_DIR="${CT_ROOT}/cache/uv"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL}"
  --backbone-model-path "${BACKBONE_MODEL}"
  --transformers-local-files-only
  --dataset-path "${DATASET}"
  --embodiment-tag NEW_EMBODIMENT
  --modality-config-path "${MODALITY_CONFIG}"
  --num-gpus 1
  --output-dir "${OUTPUT_ROOT}"
  --experiment-name "${RUN_ID}"
  --global-batch-size 16
  --gradient-accumulation-steps 2
  --dataloader-num-workers 4
  --learning-rate 1e-4
  --weight-decay 1e-5
  --warmup-ratio 0.05
  --state-dropout-prob 0.2
  --episode-sampling-rate 0.1
  --shard-size 1024
  --tune-projector
  --tune-diffusion-model
  --no-tune-llm
  --no-tune-visual
  --use-percentiles
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
  --max-steps "${MAX_STEPS}"
  --logging-steps 10
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
)

if [[ "${USE_WANDB:-0}" == "1" ]]; then
  COMMON_ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT:-agibot-gr00t}")
fi

if [[ "${MODE}" == "audit" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

if [[ "${MODE}" == "resume" ]]; then
  COMMON_ARGS+=(--resume-from-checkpoint)
fi

cd "${REPO_ROOT}"
echo "mode=${MODE} run_id=${RUN_ID} output=${RUN_OUTPUT_DIR}"
echo "1 GPU, micro batch 16, accumulation 2, effective optimizer batch 32"
echo "loss logging every 10 optimizer steps; checkpoint every ${SAVE_STEPS}; keep latest ${SAVE_TOTAL_LIMIT}"

TEE_ARGS=()
if [[ "${MODE}" == "resume" ]]; then
  TEE_ARGS=(-a)
fi

"${PYTHON_ENV}/bin/python" \
  "${REPO_ROOT}/gr00t/experiment/launch_finetune.py" \
  "${COMMON_ARGS[@]}" 2>&1 | tee "${TEE_ARGS[@]}" "${LOG_FILE}"
