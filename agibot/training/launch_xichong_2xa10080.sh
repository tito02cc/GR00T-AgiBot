#!/usr/bin/env bash
set -euo pipefail

# Audited GR00T N1.7 launcher for the Agibot G2 right-arm grasp dataset.
# Usage:
#   bash launch_xichong_2xa10080.sh smoke
#   bash launch_xichong_2xa10080.sh baseline

MODE="${1:-}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "baseline" ]]; then
  echo "Usage: $0 {smoke|baseline}" >&2
  exit 2
fi

CT_ROOT="${CT_ROOT:-/root/gpufree-data/ct}"
REPO_ROOT="${CT_ROOT}/Isaac-GR00T"
PYTHON_ENV="${REPO_ROOT}/.venv"
BASE_MODEL="${CT_ROOT}/models/GR00T-N1.7-3B"
BACKBONE_MODEL="${CT_ROOT}/models/Cosmos-Reason2-2B"
DATASET="${CT_ROOT}/datasets/xichong_right_single_grasp_300"
MODALITY_CONFIG="${CT_ROOT}/scripts/xichong_right_single_grasp_config.py"

case "${MODE}" in
  smoke)
    RUN_ID="${RUN_ID:-xichong_rgrasp_n1d7_e300_f10_h16_2xa10080_smoke_v1}"
    MAX_STEPS=100
    SAVE_STEPS=100
    SAVE_TOTAL_LIMIT=1
    ;;
  baseline)
    RUN_ID="${RUN_ID:-xichong_rgrasp_n1d7_e300_f10_h16_2xa10080_baseline_v1}"
    MAX_STEPS=2000
    SAVE_STEPS=500
    SAVE_TOTAL_LIMIT=4
    ;;
esac

OUTPUT_DIR="${CT_ROOT}/outputs/${RUN_ID}"
LOG_FILE="${CT_ROOT}/logs/${RUN_ID}.log"

for required in \
  "${PYTHON_ENV}/bin/python" \
  "${PYTHON_ENV}/bin/torchrun" \
  "${BASE_MODEL}/config.json" \
  "${BACKBONE_MODEL}/config.json" \
  "${DATASET}/meta/info.json" \
  "${DATASET}/SHA256SUMS" \
  "${MODALITY_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required training input: ${required}" >&2
    exit 3
  fi
done

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse existing output directory: ${OUTPUT_DIR}" >&2
  echo "Set a new RUN_ID, or use the explicit resume workflow." >&2
  exit 4
fi

mkdir -p "${CT_ROOT}/outputs" "${CT_ROOT}/logs" \
  "${CT_ROOT}/cache/huggingface" "${CT_ROOT}/cache/torch"

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="${CT_ROOT}/cache/huggingface"
export TORCH_HOME="${CT_ROOT}/cache/torch"
export UV_CACHE_DIR="${CT_ROOT}/cache/uv"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL}"
  --backbone-model-path "${BACKBONE_MODEL}"
  --dataset-path "${DATASET}"
  --embodiment-tag NEW_EMBODIMENT
  --modality-config-path "${MODALITY_CONFIG}"
  --num-gpus 2
  --output-dir "${OUTPUT_DIR}"
  --experiment-name "${RUN_ID}"
  --global-batch-size 32
  --gradient-accumulation-steps 1
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
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
)

if [[ "${USE_WANDB:-0}" == "1" ]]; then
  COMMON_ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT:-agibot-gr00t}")
fi

cd "${REPO_ROOT}"
echo "mode=${MODE} run_id=${RUN_ID} output=${OUTPUT_DIR}"
echo "2 GPUs, global batch 32, per-GPU batch 16, accumulation 1, effective batch 32"

"${PYTHON_ENV}/bin/torchrun" \
  --standalone \
  --nproc_per_node=2 \
  "${REPO_ROOT}/gr00t/experiment/launch_finetune.py" \
  "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
