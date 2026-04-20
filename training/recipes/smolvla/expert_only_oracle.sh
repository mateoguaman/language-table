#!/usr/bin/env bash
# SmolVLA expert-only finetune on blocktoblock_oracle_sim (200K episodes).
# Full training run with high-quality oracle demonstrations.
# Target: 1-4 GPUs, ~4GB VRAM per GPU (SmolVLA expert-only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source environment
[ -f "${TRAINING_DIR}/.env.tillicum" ] && source "${TRAINING_DIR}/.env.tillicum"
[ -f "${TRAINING_DIR}/.env.user" ] && source "${TRAINING_DIR}/.env.user"

# --- Training parameters ---
# LeRobot 0.5.1: use --policy.path for pretrained (type is inferred from checkpoint)
POLICY_PATH="lerobot/smolvla_base"
DATASET_REPO="mateoguaman/language_table_blocktoblock_oracle_sim"
DATASET_NAME="language_table_blocktoblock_oracle_sim"

BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1000}"

# --- Output ---
JOB_NAME="smolvla_expert_oracle"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT:-outputs}/${JOB_NAME}_${RUN_ID}"

# --- Dataset location ---
DATASET_ARGS="--dataset.repo_id=${DATASET_REPO}"
if [ -n "${DATASET_ROOT:-}" ] && [ -d "${DATASET_ROOT}/${DATASET_NAME}" ]; then
    DATASET_ARGS="${DATASET_ARGS} --dataset.root=${DATASET_ROOT}/${DATASET_NAME}"
fi

# --- GPU count ---
NUM_GPUS="${NUM_GPUS:-1}"

# --- Build training command ---
TRAIN_CMD=(
    --policy.path="${POLICY_PATH}"
    ${DATASET_ARGS}
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --num_workers="${NUM_WORKERS}"
    --seed="${SEED}"
    --output_dir="${OUTPUT_DIR}"
    --eval_freq=0
    --policy.push_to_hub=false
    --wandb.enable=true
    --wandb.project="${WANDB_PROJECT:-language-table-vla}"
)

# --- Launch ---
echo "=== SmolVLA Expert-Only (oracle dataset, full run) ==="
echo "Dataset: ${DATASET_REPO}"
echo "Steps: ${STEPS}, Batch: ${BATCH_SIZE}, GPUs: ${NUM_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo ""

if [ "${NUM_GPUS}" -gt 1 ]; then
    ACCELERATE_CONFIG="${TRAINING_DIR}/accelerate/ddp_${NUM_GPUS}gpu.yaml"
    if [ ! -f "${ACCELERATE_CONFIG}" ]; then
        echo "ERROR: accelerate config not found: ${ACCELERATE_CONFIG}"
        exit 1
    fi
    accelerate launch --config_file="${ACCELERATE_CONFIG}" \
        -m lerobot.scripts.lerobot_train "${TRAIN_CMD[@]}"
else
    python -m lerobot.scripts.lerobot_train "${TRAIN_CMD[@]}"
fi
