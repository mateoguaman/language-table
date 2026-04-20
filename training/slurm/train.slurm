#!/bin/bash
#SBATCH --job-name=lt-vla-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=lt-vla-train-%j.out
#SBATCH --error=lt-vla-train-%j.err
#
# Override defaults at submission time:
#   sbatch --gres=gpu:1 --cpus-per-task=8 --mem=50G training/slurm/train.slurm
#   sbatch --gres=gpu:8 --cpus-per-task=64 --mem=400G training/slurm/train.slurm

set -euo pipefail

# --- Resolve paths ---
# SLURM copies the script to /var/spool, so BASH_SOURCE won't work.
# Use SLURM_SUBMIT_DIR (the directory sbatch was called from) instead.
TRAINING_DIR="${TRAINING_DIR:-${SLURM_SUBMIT_DIR}/training}"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

# --- Source environment ---
source "${TRAINING_DIR}/.env.tillicum"
[ -f "${TRAINING_DIR}/.env.user" ] && source "${TRAINING_DIR}/.env.user"

# --- SLURM log directory ---
mkdir -p "${SLURM_LOG_DIR}"

# --- Module and conda setup ---
module load "${MODULE_CONDA}"
module load "${MODULE_GCC}"
module load "${MODULE_CMAKE}"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

# --- Export environment ---
export HF_HOME
export HF_LEROBOT_HOME
export HF_HUB_ETAG_TIMEOUT
export HF_HUB_ENABLE_HF_TRANSFER
export NCCL_ASYNC_ERROR_HANDLING
export NCCL_SOCKET_NTHREADS
export NCCL_NSOCKS_PERTHREAD
export NCCL_TIMEOUT
export NCCL_DEBUG
export WANDB_PROJECT

[ -n "${WANDB_API_KEY:-}" ] && export WANDB_API_KEY
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

# --- Create cache/output directories ---
mkdir -p "${HF_HOME}" "${HF_LEROBOT_HOME}" "${OUTPUT_ROOT}" "${CHECKPOINT_ROOT}"

# --- Optional: copy dataset to node-local storage for fast I/O ---
if [ -n "${TMPDIR:-}" ] && [ -n "${DATASET_ROOT:-}" ] && [ -n "${RECIPE_DATASET_NAME:-}" ]; then
    LOCAL_DATA="${TMPDIR}/datasets/${RECIPE_DATASET_NAME}"
    if [ -d "${DATASET_ROOT}/${RECIPE_DATASET_NAME}" ] && [ ! -d "${LOCAL_DATA}" ]; then
        echo "Copying dataset to node-local storage..."
        echo "  From: ${DATASET_ROOT}/${RECIPE_DATASET_NAME}"
        echo "  To:   ${LOCAL_DATA}"
        mkdir -p "${LOCAL_DATA}"
        cp -r "${DATASET_ROOT}/${RECIPE_DATASET_NAME}/"* "${LOCAL_DATA}/"
        echo "  Done. ($(du -sh "${LOCAL_DATA}" | cut -f1))"
    fi
    if [ -d "${LOCAL_DATA}" ]; then
        export DATASET_ROOT="${TMPDIR}/datasets"
    fi
fi

# --- Job info ---
echo "============================================"
echo " Language Table VLA Training"
echo "============================================"
echo "START:       $(date)"
echo "HOSTNAME:    $(hostname)"
echo "JOB_ID:      ${SLURM_JOB_ID:-local}"
echo "GPUS:        ${SLURM_GPUS_ON_NODE:-${NUM_GPUS:-1}}"
echo "CONDA_ENV:   ${CONDA_ENV}"
echo "PYTHON:      $(which python)"
echo "RECIPE:      ${RECIPE:-unset}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# --- Validate recipe ---
if [ -z "${RECIPE:-}" ]; then
    echo "ERROR: RECIPE env var not set."
    echo ""
    echo "Usage:"
    echo "  RECIPE=recipes/smolvla/expert_only_small.sh sbatch training/slurm/train.slurm"
    echo ""
    echo "Available recipes:"
    find "${TRAINING_DIR}/recipes" -name "*.sh" | sort | sed "s|${TRAINING_DIR}/||"
    exit 1
fi

RECIPE_PATH="${TRAINING_DIR}/${RECIPE}"
if [ ! -f "${RECIPE_PATH}" ]; then
    echo "ERROR: Recipe not found: ${RECIPE_PATH}"
    exit 1
fi

# --- Set GPU count from SLURM allocation ---
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-${NUM_GPUS:-1}}"

# --- Run the recipe ---
echo "Running: bash ${RECIPE_PATH}"
echo "--------------------------------------------"
bash "${RECIPE_PATH}"

echo ""
echo "============================================"
echo "END: $(date)"
