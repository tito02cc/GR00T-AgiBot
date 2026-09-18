#!/usr/bin/env bash
set -euo pipefail

# Download the two official GR00T N1.7 model repositories to the data disk.
# The token is read only from HF_TOKEN and is never written by this script.

CT_ROOT="${CT_ROOT:?Set CT_ROOT to the external artifact storage directory}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${GROOT_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
export CT_ROOT

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing GR00T Python environment: ${PYTHON}" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required and must have access to nvidia/Cosmos-Reason2-2B." >&2
  exit 3
fi

export HF_HOME="${CT_ROOT}/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "${CT_ROOT}/models" "${HF_HOME}"

"${PYTHON}" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ct_root = Path(os.environ["CT_ROOT"])
token = os.environ["HF_TOKEN"]

models = (
    (
        "nvidia/GR00T-N1.7-3B",
        "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
        ct_root / "models" / "GR00T-N1.7-3B",
    ),
    (
        "nvidia/Cosmos-Reason2-2B",
        "9ce19a195e423419c349abfc86fd07178b230561",
        ct_root / "models" / "Cosmos-Reason2-2B",
    ),
)

for repo_id, revision, local_dir in models:
    print(f"Downloading {repo_id}@{revision} -> {local_dir}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        token=token,
    )
PY

check_hash() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | cut -d ' ' -f1)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA256 mismatch: ${path}" >&2
    echo "expected=${expected}" >&2
    echo "actual=${actual}" >&2
    exit 4
  fi
  echo "SHA256 PASS ${path} ${actual}"
}

check_hash \
  8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d \
  "${CT_ROOT}/models/GR00T-N1.7-3B/model-00001-of-00002.safetensors"
check_hash \
  c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661 \
  "${CT_ROOT}/models/GR00T-N1.7-3B/model-00002-of-00002.safetensors"
check_hash \
  fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede \
  "${CT_ROOT}/models/Cosmos-Reason2-2B/model.safetensors"

echo "MODEL DOWNLOAD PASS: pinned N1.7 and Cosmos snapshots are complete."
