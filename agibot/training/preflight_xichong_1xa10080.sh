#!/usr/bin/env bash
set -euo pipefail

CT_ROOT="${CT_ROOT:?CT_ROOT is required}"
REPO_ROOT="${GROOT_REPO_ROOT:-${CT_ROOT}/Isaac-GR00T}"
PYTHON_ENV="${REPO_ROOT}/.venv"
CUDNN_LIB="${PYTHON_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib"
BASE_MODEL="${GROOT_BASE_MODEL:-${CT_ROOT}/models/GR00T-N1.7-3B}"
BACKBONE_MODEL="${GROOT_BACKBONE_MODEL:-${CT_ROOT}/models/Cosmos-Reason2-2B}"
DATASET="${GROOT_DATASET:-${CT_ROOT}/datasets/xichong_right_single_grasp_300}"
DATASET_MANIFEST="${GROOT_DATASET_MANIFEST:-${DATASET}/SHA256SUMS}"
MODALITY_CONFIG="${GROOT_MODALITY_CONFIG:-${REPO_ROOT}/agibot/configs/xichong_right_single_grasp_config.py}"
EXPECTED_EPISODES="${GROOT_EXPECTED_EPISODES:-300}"
EXPECTED_VIDEOS="${GROOT_EXPECTED_VIDEOS:-600}"

mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
)
if [[ "${#GPU_ROWS[@]}" -ne 1 ]]; then
  echo "Expected exactly 1 GPU, detected ${#GPU_ROWS[@]}." >&2
  exit 10
fi

if [[ "${GPU_ROWS[0]}" != *"A100"* ]]; then
  echo "Expected A100 GPU, detected: ${GPU_ROWS[0]}" >&2
  exit 11
fi
memory_mib="${GPU_ROWS[0]##*, }"
if (( memory_mib < 79000 )); then
  echo "Expected an 80 GB A100, detected ${memory_mib} MiB: ${GPU_ROWS[0]}" >&2
  exit 12
fi

read -r cpu_quota cpu_period < /sys/fs/cgroup/cpu.max
if [[ "${cpu_quota}" != "max" ]]; then
  cpu_limit=$((cpu_quota / cpu_period))
  if (( cpu_limit < 4 )); then
    echo "At least 4 cgroup CPUs are required for 4 dataloader workers; detected ${cpu_limit}." >&2
    exit 15
  fi
fi

memory_limit="$(cat /sys/fs/cgroup/memory.max)"
if [[ "${memory_limit}" != "max" ]] && (( memory_limit < 68719476736 )); then
  echo "At least 64 GiB cgroup RAM is required; detected ${memory_limit} bytes." >&2
  exit 16
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
    exit 13
  fi
done

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="${CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${PYTHON_ENV}/bin/python" -c \
  "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert torch.cuda.get_device_properties(0).total_memory >= 79 * 2**30; cudnn=torch.backends.cudnn.version(); x=torch.randn(2,3,32,32,device='cuda',dtype=torch.bfloat16); y=torch.nn.Conv2d(3,8,3).cuda().bfloat16()(x); assert tuple(y.shape)==(2,8,30,30); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', cudnn, torch.cuda.get_device_name(0), 'BF16 Conv PASS')"

NO_ALBUMENTATIONS_UPDATE=1 "${PYTHON_ENV}/bin/python" -c \
  "from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config; from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor; c=Gr00tN1d7Config.from_pretrained('${BASE_MODEL}', local_files_only=True); p=Gr00tN1d7Processor.from_pretrained('${BASE_MODEL}', model_name='${BACKBONE_MODEL}', model_type=c.backbone_model_type, transformers_loading_kwargs={'local_files_only': True, 'trust_remote_code': True}); assert p.max_action_horizon == 40 and p.max_state_dim == 132 and p.max_action_dim == 132; print('local N1.7 config/processor PASS')"

parquet_count="$(find "${DATASET}/data" -type f -name '*.parquet' | wc -l)"
video_count="$(find "${DATASET}/videos" -type f -name '*.mp4' | wc -l)"
if [[ "${parquet_count}" -ne "${EXPECTED_EPISODES}" || \
      "${video_count}" -ne "${EXPECTED_VIDEOS}" ]]; then
  echo "Dataset count mismatch: parquet=${parquet_count}, video=${video_count}" >&2
  exit 14
fi

(
  cd "${DATASET}"
  sha256sum -c --quiet "${DATASET_MANIFEST}"
)

PYTHONPATH="$(dirname "${MODALITY_CONFIG}")${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_ENV}/bin/python" -c \
  "from pathlib import Path; from xichong_right_single_grasp_config import XICHONG_RIGHT_SINGLE_GRASP_CONFIG as c; from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader; loader=LeRobotEpisodeLoader(Path('${DATASET}'), c); assert len(loader) == ${EXPECTED_EPISODES}; assert set(loader.get_dataset_statistics()) == {'action', 'relative_action', 'state'}; print('loader episodes', len(loader), 'PASS')"

echo "PREFLIGHT PASS: 1x A100 80GB, environment, 907-file hash gate, modality config, and official loader."
