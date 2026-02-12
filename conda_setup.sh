#!/usr/bin/env bash
set -euo pipefail

# ===== User config =====
ENV_NAME="openpcdet"
PY_VER="3.9"
TORCH_VER="2.5.1"
TORCHVISION_VER="0.20.1"
TORCHAUDIO_VER="2.5.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || echo "${SCRIPT_DIR}")"
# =======================

if [[ ! -f "${REPO_DIR}/requirements.txt" ]]; then
  echo "requirements.txt not found in REPO_DIR: ${REPO_DIR}"
  echo "Run this script from the repo root, or place the script in that repo."
  exit 1
fi

echo "[1/7] Sanity checks"
command -v conda >/dev/null 2>&1 || { echo "conda not found. Install Anaconda/Miniconda first."; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found. NVIDIA driver not installed?"; exit 1; }

echo "GPU:"
nvidia-smi -L || true
echo "nvcc:"
nvcc --version || true

echo "[2/7] Create conda env: ${ENV_NAME} (python=${PY_VER})"
# env already exists? then skip creation
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "  - Env '${ENV_NAME}' already exists. Skipping create."
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[3/7] Upgrade pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[4/7] Install project python deps"
REQ_FILE="$(mktemp)"
trap 'rm -f "${REQ_FILE}"' EXIT
awk '
  BEGIN { IGNORECASE=1 }
  /^[[:space:]]*#/ || /^[[:space:]]*$/ { print; next }
  /^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]]*[<>=!~].*)?[[:space:]]*$/ { next }
  { print }
' "${REPO_DIR}/requirements.txt" > "${REQ_FILE}"
python -m pip install --upgrade -r "${REQ_FILE}"

echo "[5/7] Install PyTorch stack (pinned, CUDA 12.1 wheels)"
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install --upgrade \
  "torch==${TORCH_VER}+cu121" \
  "torchvision==${TORCHVISION_VER}+cu121" \
  "torchaudio==${TORCHAUDIO_VER}+cu121" \
  --index-url https://download.pytorch.org/whl/cu121

echo "[6/7] Install spconv + extras"
python -m pip install --upgrade spconv-cu121
python -m pip install --upgrade open3d pyquaternion

echo "[7/7] Editable install (if Python package project)"
if [[ -f "${REPO_DIR}/setup.py" || -f "${REPO_DIR}/pyproject.toml" ]]; then
  cd "${REPO_DIR}"
  python -m pip install -e . --no-build-isolation
else
  echo "  - setup.py/pyproject.toml not found. Skipping editable install."
fi

echo ""
echo "=== Verify installs ==="
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available(), "torch cuda:", torch.version.cuda)
import spconv
print("spconv:", spconv.__version__)
PY

echo ""
echo "Done. Activate with: conda activate ${ENV_NAME}"
