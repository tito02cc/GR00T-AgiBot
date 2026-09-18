#!/usr/bin/env bash
set -euo pipefail

# Generic single-A100 preflight. All task-specific values arrive through the
# task profile via agibot/bin/groot-g2 train.
CT_ROOT="${CT_ROOT:?CT_ROOT is required}"
REPO_ROOT="${GROOT_REPO_ROOT:?GROOT_REPO_ROOT is required}"
PYTHON_ENV="${REPO_ROOT}/.venv"
CUDNN_LIB="${PYTHON_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib"
BASE_MODEL="${GROOT_BASE_MODEL:-${CT_ROOT}/models/GR00T-N1.7-3B}"
BACKBONE_MODEL="${GROOT_BACKBONE_MODEL:-${CT_ROOT}/models/Cosmos-Reason2-2B}"
DATASET="${GROOT_DATASET:?GROOT_DATASET is required}"
DATASET_MANIFEST="${GROOT_DATASET_MANIFEST:?GROOT_DATASET_MANIFEST is required}"
MODALITY_CONFIG="${GROOT_MODALITY_CONFIG:?GROOT_MODALITY_CONFIG is required}"
EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-NEW_EMBODIMENT}"
EXPECTED_EPISODES="${GROOT_EXPECTED_EPISODES:?GROOT_EXPECTED_EPISODES is required}"
EXPECTED_VIDEOS="${GROOT_EXPECTED_VIDEOS:?GROOT_EXPECTED_VIDEOS is required}"

mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
)
if [[ "${#GPU_ROWS[@]}" -ne 1 || "${GPU_ROWS[0]}" != *"A100"* ]]; then
  echo "Expected exactly one A100 GPU, detected: ${GPU_ROWS[*]:-none}" >&2
  exit 10
fi
memory_mib="${GPU_ROWS[0]##*, }"
if (( memory_mib < 79000 )); then
  echo "Expected an 80 GB A100, detected ${memory_mib} MiB." >&2
  exit 11
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
    echo "Missing required input: ${required}" >&2
    exit 12
  fi
done

parquet_count="$(find "${DATASET}/data" -type f -name '*.parquet' | wc -l)"
video_count="$(find "${DATASET}/videos" -type f -name '*.mp4' | wc -l)"
if [[ "${parquet_count}" -ne "${EXPECTED_EPISODES}" || \
      "${video_count}" -ne "${EXPECTED_VIDEOS}" ]]; then
  echo "Dataset count mismatch: parquet=${parquet_count}, video=${video_count}" >&2
  exit 13
fi

(
  cd "${DATASET}"
  sha256sum -c --quiet "${DATASET_MANIFEST}"
)

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="${CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${PYTHON_ENV}/bin/python" -c \
  "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert torch.cuda.get_device_properties(0).total_memory >= 79 * 2**30; x=torch.randn(2,3,32,32,device='cuda',dtype=torch.bfloat16); y=torch.nn.Conv2d(3,8,3).cuda().bfloat16()(x); assert tuple(y.shape)==(2,8,30,30); print(torch.cuda.get_device_name(0), 'BF16 PASS')"

"${PYTHON_ENV}/bin/python" "${REPO_ROOT}/agibot/training/check_dataset_loader.py" \
  --dataset "${DATASET}" \
  --modality-config "${MODALITY_CONFIG}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --expected-episodes "${EXPECTED_EPISODES}"

echo "PREFLIGHT PASS: generic task profile, 1x A100 80GB, SHA and official loader."
