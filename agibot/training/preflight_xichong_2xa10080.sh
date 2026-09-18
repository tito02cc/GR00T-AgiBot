#!/usr/bin/env bash
set -euo pipefail

CT_ROOT="${CT_ROOT:-/root/gpufree-data/ct}"
REPO_ROOT="${CT_ROOT}/Isaac-GR00T"
PYTHON_ENV="${REPO_ROOT}/.venv"
BASE_MODEL="${CT_ROOT}/models/GR00T-N1.7-3B"
BACKBONE_MODEL="${CT_ROOT}/models/Cosmos-Reason2-2B"
DATASET="${CT_ROOT}/datasets/xichong_right_single_grasp_300"
MODALITY_CONFIG="${CT_ROOT}/scripts/xichong_right_single_grasp_config.py"

mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
)
if [[ "${#GPU_ROWS[@]}" -ne 2 ]]; then
  echo "Expected exactly 2 GPUs, detected ${#GPU_ROWS[@]}." >&2
  exit 10
fi

for row in "${GPU_ROWS[@]}"; do
  if [[ "${row}" != *"A100"* ]]; then
    echo "Expected A100 GPU, detected: ${row}" >&2
    exit 11
  fi
  memory_mib="${row##*, }"
  if (( memory_mib < 79000 )); then
    echo "Expected an 80 GB A100, detected ${memory_mib} MiB: ${row}" >&2
    exit 12
  fi
done

read -r cpu_quota cpu_period < /sys/fs/cgroup/cpu.max
if [[ "${cpu_quota}" != "max" ]]; then
  cpu_limit=$((cpu_quota / cpu_period))
  if (( cpu_limit < 8 )); then
    echo "At least 8 cgroup CPUs are required for 8 total dataloader workers; detected ${cpu_limit}." >&2
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
  "${PYTHON_ENV}/bin/torchrun" \
  "${BASE_MODEL}/config.json" \
  "${BACKBONE_MODEL}/config.json" \
  "${DATASET}/meta/info.json" \
  "${DATASET}/SHA256SUMS" \
  "${MODALITY_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 13
  fi
done

CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_ENV}/bin/python" -c \
  "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 2; assert all(torch.cuda.get_device_properties(i).total_memory >= 79 * 2**30 for i in range(2)); print('torch', torch.__version__, 'cuda', torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(2)], 'peer_access', torch.cuda.can_device_access_peer(0, 1))"

NO_ALBUMENTATIONS_UPDATE=1 "${PYTHON_ENV}/bin/python" -c \
  "from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config; from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor; c=Gr00tN1d7Config.from_pretrained('${BASE_MODEL}', local_files_only=True); p=Gr00tN1d7Processor.from_pretrained('${BASE_MODEL}', model_name='${BACKBONE_MODEL}', model_type=c.backbone_model_type, transformers_loading_kwargs={'local_files_only': True, 'trust_remote_code': True}); assert p.max_action_horizon == 40 and p.max_state_dim == 132 and p.max_action_dim == 132; print('local N1.7 config/processor PASS')"

parquet_count="$(find "${DATASET}/data" -type f -name '*.parquet' | wc -l)"
video_count="$(find "${DATASET}/videos" -type f -name '*.mp4' | wc -l)"
if [[ "${parquet_count}" -ne 300 || "${video_count}" -ne 600 ]]; then
  echo "Dataset count mismatch: parquet=${parquet_count}, video=${video_count}" >&2
  exit 14
fi

(
  cd "${DATASET}"
  sha256sum -c --quiet SHA256SUMS
)

PYTHONPATH="${CT_ROOT}/scripts" "${PYTHON_ENV}/bin/python" -c \
  "from pathlib import Path; from xichong_right_single_grasp_config import XICHONG_RIGHT_SINGLE_GRASP_CONFIG as c; from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader; loader=LeRobotEpisodeLoader(Path('${DATASET}'), c); assert len(loader) == 300; assert set(loader.get_dataset_statistics()) == {'action', 'relative_action', 'state'}; print('loader episodes', len(loader), 'PASS')"

echo "PREFLIGHT PASS: 2x A100 80GB, environment, 907-file hash gate, modality config, and official loader."
