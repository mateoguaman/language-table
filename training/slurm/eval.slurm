#!/bin/bash
#SBATCH --job-name=lt-vla-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH --output=lt-vla-eval-%j.out
#SBATCH --error=lt-vla-eval-%j.err
#
# Evaluates a trained policy in the Language Table sim.
# Requires both ltvenv (for sim) and lerobot conda env (for policy serving).
#
# Environment variables:
#   POLICY_TYPE     - "lava" or "lerobot" (required)
#   CHECKPOINT_PATH - path to the checkpoint (required)
#   EVAL_OUTPUT_DIR - where to write results (default: derived from checkpoint)
#   LAVA_CONFIG     - config file for LAVA (required if POLICY_TYPE=lava)
#   NUM_EPISODES    - episodes per reward type (default: 50)

set -euo pipefail

# --- Resolve paths ---
# SLURM copies the script to /var/spool, so BASH_SOURCE won't work.
TRAINING_DIR="${TRAINING_DIR:-${SLURM_SUBMIT_DIR}/training}"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

# --- Source environment ---
source "${TRAINING_DIR}/.env.tillicum"
[ -f "${TRAINING_DIR}/.env.user" ] && source "${TRAINING_DIR}/.env.user"

# --- Modules ---
module load "${MODULE_CONDA}"
module load "${MODULE_GCC}"

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Validate inputs ---
POLICY_TYPE="${POLICY_TYPE:?ERROR: POLICY_TYPE not set (lava or lerobot)}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?ERROR: CHECKPOINT_PATH not set}"
NUM_EPISODES="${NUM_EPISODES:-50}"

# Derive output dir from checkpoint if not set
if [ -z "${EVAL_OUTPUT_DIR:-}" ]; then
    CKPT_BASENAME="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
    EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/eval/${CKPT_BASENAME}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${EVAL_OUTPUT_DIR}"

# --- Job info ---
echo "============================================"
echo " Language Table Policy Evaluation"
echo "============================================"
echo "START:       $(date)"
echo "POLICY_TYPE: ${POLICY_TYPE}"
echo "CHECKPOINT:  ${CHECKPOINT_PATH}"
echo "OUTPUT:      ${EVAL_OUTPUT_DIR}"
echo "EPISODES:    ${NUM_EPISODES} per reward type"
echo ""

# --- Run evaluation ---
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [ "${POLICY_TYPE}" = "lava" ]; then
    LAVA_CONFIG="${LAVA_CONFIG:?ERROR: LAVA_CONFIG not set for lava policy}"
    # LAVA runs entirely in ltvenv
    set +u
    conda activate ltvenv
    set -u
    python "${TRAINING_DIR}/eval/run_eval.py" \
        --policy_type=lava \
        --checkpoint_path="${CHECKPOINT_PATH}" \
        --config="${LAVA_CONFIG}" \
        --output_dir="${EVAL_OUTPUT_DIR}" \
        --num_episodes="${NUM_EPISODES}"

elif [ "${POLICY_TYPE}" = "lerobot" ]; then
    # LeRobot eval uses two processes:
    # 1. Policy server in the lerobot conda env
    # 2. Eval loop in ltvenv
    LEROBOT_PYTHON="$(conda run -n "${CONDA_ENV}" which python)"

    set +u
    conda activate ltvenv
    set -u
    python "${TRAINING_DIR}/eval/run_eval.py" \
        --policy_type=lerobot \
        --checkpoint_path="${CHECKPOINT_PATH}" \
        --server_python="${LEROBOT_PYTHON}" \
        --output_dir="${EVAL_OUTPUT_DIR}" \
        --num_episodes="${NUM_EPISODES}"
fi

echo ""
echo "============================================"
echo "Evaluation complete. Results: ${EVAL_OUTPUT_DIR}/eval_results.json"
echo "END: $(date)"
