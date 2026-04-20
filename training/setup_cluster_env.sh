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
    echo "Creating conda env '${ENV_NAME}' with Python 3.10..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

set +u
conda activate "${ENV_NAME}"
set -u

# Install LeRobot with all policy extras
echo "Installing LeRobot with policy extras..."
pip install 'lerobot[pi0,smolvla,diffusion,act]>=0.4.4'

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
