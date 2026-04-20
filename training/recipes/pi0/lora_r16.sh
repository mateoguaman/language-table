#!/usr/bin/env bash
# Pi0 LoRA (rank 16) finetune on blocktoblock_oracle_sim (200K episodes).
# Trains low-rank adapters on the VLM + action expert.
# Target: 2-4 GPUs, ~16GB VRAM per GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source environment
[ -f "${TRAINING_DIR}/.env.tillicum" ] && source "${TRAINING_DIR}/.env.tillicum"
[ -f "${TRAINING_DIR}/.env.user" ] && source "${TRAINING_DIR}/.env.user"

# --- Training parameters ---
POLICY_TYPE="pi0"
POLICY_PATH="lerobot/pi0_base"
DATASET_REPO="mateoguaman/language_table_blocktoblock_oracle_sim"
DATASET_NAME="language_table_blocktoblock_oracle_sim"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
LORA_RANK="${LORA_RANK:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1000}"

# --- Output ---
JOB_NAME="pi0_lora_r${LORA_RANK}_oracle"
OUTPUT_DIR="${OUTPUT_ROOT:-outputs}/${JOB_NAME}"

# --- Dataset location ---
DATASET_ARGS="--dataset.repo_id=${DATASET_REPO}"
if [ -n "${DATASET_ROOT:-}" ] && [ -d "${DATASET_ROOT}/${DATASET_NAME}" ]; then
    DATASET_ARGS="${DATASET_ARGS} --dataset.root=${DATASET_ROOT}/${DATASET_NAME}"
fi

# --- GPU count ---
NUM_GPUS="${NUM_GPUS:-2}"

# --- Build training command ---
TRAIN_CMD=(
    --policy.type="${POLICY_TYPE}"
    --policy.path="${POLICY_PATH}"
    ${DATASET_ARGS}
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --peft.method_type=LORA
    --peft.r="${LORA_RANK}"
    --num_workers="${NUM_WORKERS}"
    --seed="${SEED}"
    --output_dir="${OUTPUT_DIR}"
    --eval_freq=0
    --wandb.enable=true
    --wandb.project="${WANDB_PROJECT:-language-table-vla}"
)

# --- Launch ---
echo "=== Pi0 LoRA r=${LORA_RANK} (oracle dataset) ==="
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
