#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "audit" && "${MODE}" != "smoke" && \
      "${MODE}" != "baseline" && "${MODE}" != "resume" ]]; then
  echo "Usage: $0 {audit|smoke|baseline|resume}" >&2
  exit 2
fi

CT_ROOT="${CT_ROOT:?CT_ROOT is required}"
REPO_ROOT="${GROOT_REPO_ROOT:?GROOT_REPO_ROOT is required}"
PYTHON_ENV="${REPO_ROOT}/.venv"
CUDNN_LIB="${PYTHON_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib"
BASE_MODEL="${GROOT_BASE_MODEL:-${CT_ROOT}/models/GR00T-N1.7-3B}"
BACKBONE_MODEL="${GROOT_BACKBONE_MODEL:-${CT_ROOT}/models/Cosmos-Reason2-2B}"
DATASET="${GROOT_DATASET:?GROOT_DATASET is required}"
DATASET_VALIDATION="${GROOT_DATASET_VALIDATION:-sha256}"
DATASET_MANIFEST="${GROOT_DATASET_MANIFEST:-}"
MODALITY_CONFIG="${GROOT_MODALITY_CONFIG:?GROOT_MODALITY_CONFIG is required}"
EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-NEW_EMBODIMENT}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
MAX_STEPS="${GROOT_MAX_STEPS:?GROOT_MAX_STEPS is required}"
SAVE_STEPS="${GROOT_SAVE_STEPS:?GROOT_SAVE_STEPS is required}"
SAVE_TOTAL_LIMIT="${GROOT_SAVE_TOTAL_LIMIT:?GROOT_SAVE_TOTAL_LIMIT is required}"

if [[ "${MODE}" == "smoke" ]]; then
  RUN_ID="${GROOT_SMOKE_RUN_ID:-${RUN_ID}_smoke}"
  MAX_STEPS="${GROOT_SMOKE_STEPS:-100}"
  SAVE_STEPS="${MAX_STEPS}"
  SAVE_TOTAL_LIMIT=1
fi

OUTPUT_ROOT="${CT_ROOT}/outputs"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_FILE="${CT_ROOT}/logs/${RUN_ID}.${MODE}.log"

for required in \
  "${PYTHON_ENV}/bin/python" \
  "${CUDNN_LIB}/libcudnn.so.9" \
  "${BASE_MODEL}/config.json" \
  "${BACKBONE_MODEL}/config.json" \
  "${DATASET}/meta/info.json" \
  "${MODALITY_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required training input: ${required}" >&2
    exit 3
  fi
done

case "${DATASET_VALIDATION}" in
  sha256)
    : "${DATASET_MANIFEST:?GROOT_DATASET_MANIFEST is required for sha256 mode}"
    (cd "${DATASET}"; sha256sum -c --quiet "${DATASET_MANIFEST}")
    ;;
  inventory)
    "${PYTHON_ENV}/bin/python" "${REPO_ROOT}/agibot/training/check_cloud_inputs.py" \
      check --root "${DATASET}" \
      --inventory "${GROOT_DATASET_INVENTORY:?GROOT_DATASET_INVENTORY is required}"
    ;;
  *) echo "Unknown GROOT_DATASET_VALIDATION: ${DATASET_VALIDATION}" >&2; exit 3 ;;
esac

if [[ "${MODE}" == "resume" ]]; then
  if [[ ! -d "${RUN_OUTPUT_DIR}" ]] || \
     ! compgen -G "${RUN_OUTPUT_DIR}/checkpoint-*" >/dev/null; then
    echo "No resumable checkpoint found under: ${RUN_OUTPUT_DIR}" >&2
    exit 5
  fi
elif [[ "${MODE}" != "audit" && -e "${RUN_OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse existing output directory: ${RUN_OUTPUT_DIR}" >&2
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

# New task preparation records the normalization choice beside the dataset.
# Existing datasets without this metadata retain the historical percentile default.
DATASET_PERCENTILES="true"
if [[ -f "${DATASET}/meta/training_preprocessing.json" ]]; then
  DATASET_PERCENTILES="$("${PYTHON_ENV}/bin/python" -c \
    'import json,sys; v=json.load(open(sys.argv[1]))["use_percentiles"]; assert isinstance(v,bool); print(str(v).lower())' \
    "${DATASET}/meta/training_preprocessing.json")"
fi
USE_PERCENTILES="${GROOT_USE_PERCENTILES:-${DATASET_PERCENTILES}}"
if [[ -f "${DATASET}/meta/training_preprocessing.json" && "${USE_PERCENTILES}" != "${DATASET_PERCENTILES}" ]]; then
  echo "GROOT_USE_PERCENTILES conflicts with dataset training_preprocessing.json" >&2
  exit 6
fi
case "${USE_PERCENTILES}" in
  true) NORMALIZATION_FLAG=--use-percentiles ;;
  false) NORMALIZATION_FLAG=--no-use-percentiles ;;
  *) echo "GROOT_USE_PERCENTILES must be true or false" >&2; exit 6 ;;
esac

COMMON_ARGS=(
  --base-model-path "${BASE_MODEL}"
  --backbone-model-path "${BACKBONE_MODEL}"
  --transformers-local-files-only
  --dataset-path "${DATASET}"
  --embodiment-tag "${EMBODIMENT_TAG}"
  --modality-config-path "${MODALITY_CONFIG}"
  --num-gpus 1
  --output-dir "${OUTPUT_ROOT}"
  --experiment-name "${RUN_ID}"
  --global-batch-size "${GROOT_GLOBAL_BATCH_SIZE:-16}"
  --gradient-accumulation-steps "${GROOT_GRADIENT_ACCUMULATION_STEPS:-2}"
  --dataloader-num-workers "${GROOT_DATALOADER_WORKERS:-4}"
  --learning-rate "${GROOT_LEARNING_RATE:-1e-4}"
  --weight-decay "${GROOT_WEIGHT_DECAY:-1e-5}"
  --warmup-ratio "${GROOT_WARMUP_RATIO:-0.05}"
  --state-dropout-prob "${GROOT_STATE_DROPOUT_PROB:-0.2}"
  --episode-sampling-rate "${GROOT_EPISODE_SAMPLING_RATE:-0.1}"
  --shard-size "${GROOT_SHARD_SIZE:-1024}"
  --tune-projector
  --tune-diffusion-model
  --no-tune-llm
  --no-tune-visual
  "${NORMALIZATION_FLAG}"
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
  --max-steps "${MAX_STEPS}"
  --logging-steps "${GROOT_LOGGING_STEPS:-10}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
)

if [[ "${MODE}" == "audit" ]]; then
  COMMON_ARGS+=(--dry-run)
fi
if [[ "${MODE}" == "resume" ]]; then
  COMMON_ARGS+=(--resume-from-checkpoint)
fi
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  COMMON_ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT:-agibot-gr00t}")
fi

cd "${REPO_ROOT}"
"${PYTHON_ENV}/bin/python" "${REPO_ROOT}/gr00t/experiment/launch_finetune.py" \
  "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
