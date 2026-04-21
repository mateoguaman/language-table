#!/usr/bin/env bash
# One-time setup: create the 'lerobot' conda env on Tillicum.
# Run this interactively on a cluster node (login or compute).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env.tillicum"
[ -f "${SCRIPT_DIR}/.env.user" ] && source "${SCRIPT_DIR}/.env.user"

ENV_NAME="${CONDA_ENV:-lerobot}"

echo "=== Language Table VLA: Cluster Environment Setup ==="
echo "Creating conda env: ${ENV_NAME}"

# Load modules
module load "${MODULE_CONDA}"
module load "${MODULE_GCC}"
module load "${MODULE_CMAKE}"

source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env if it doesn't exist
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Activating..."
else
    echo "Creating conda env '${ENV_NAME}' with Python 3.12..."
    conda create -n "${ENV_NAME}" python=3.12 -y
fi

set +u
conda activate "${ENV_NAME}"
set -u

# Redirect pip cache + build tmpdir off the home quota (evdev builds from source on py3.12).
export PIP_CACHE_DIR
export TMPDIR
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

# Install LeRobot with all policy extras.
# Note: we use the `pyav` video backend in recipes (via --dataset.video_backend=pyav).
# pyav ships a self-contained FFmpeg wheel, so we avoid conda-installing system ffmpeg
# (which is flaky on GPFS scrubbed due to mid-extract file cleanup).
echo "Installing LeRobot with policy extras..."
echo "  PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "  TMPDIR=${TMPDIR}"
# In lerobot 0.5.1, ACT and Diffusion ship with the base install (no extra needed).
# Pi0 is now exposed as the `pi` extra (not `pi0`). SmolVLA has its own extra.
pip install 'lerobot[pi,smolvla]>=0.5.1'
pip install av

# Install accelerate for multi-GPU
pip install accelerate

# Create GPFS directories
echo "Creating GPFS directories..."
mkdir -p "${CHECKPOINT_ROOT}" "${DATASET_ROOT}" "${OUTPUT_ROOT}" "${SLURM_LOG_DIR}"
mkdir -p "${HF_HOME}" "${HF_LEROBOT_HOME}"

# Verify
echo ""
echo "=== Verification ==="
python -c "import lerobot; print(f'LeRobot {lerobot.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from accelerate import Accelerator; print('accelerate OK')"
echo ""
echo "Environment: $(which python)"
echo "Setup complete. Activate with: conda activate ${ENV_NAME}"
