#!/usr/bin/env bash
# Pi0.5 full finetune on language_table_sim_combined (~1.2M eps).
# Every param trainable: SigLIP vision encoder, PaliGemma backbone,
# gemma_expert, projections.
# Target: 4 GPUs. Probed ceiling on H200: bs=4 per GPU at 68.3% VRAM (bs=8
# OOMs) — see docs/batch-size-probes.md. Effective batch at 4 GPUs is only
# 16; plan gradient accumulation if you need larger effective batch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source environment
[ -f "${TRAINING_DIR}/.env.tillicum" ] && source "${TRAINING_DIR}/.env.tillicum"
[ -f "${TRAINING_DIR}/.env.user" ] && source "${TRAINING_DIR}/.env.user"

# --- Training parameters ---
POLICY_PATH="lerobot/pi05_base"
DATASET_REPO="mateoguaman/language_table_sim_combined"
DATASET_NAME="language_table_sim_combined"

BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1000}"

# --- Output ---
JOB_NAME="pi05_full_combined_sim"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT:-outputs}/${JOB_NAME}_${RUN_ID}"

# --- Dataset location ---
DATASET_ARGS="--dataset.repo_id=${DATASET_REPO}"
if [ -n "${DATASET_ROOT:-}" ] && [ -d "${DATASET_ROOT}/${DATASET_NAME}" ]; then
    DATASET_ARGS="${DATASET_ARGS} --dataset.root=${DATASET_ROOT}/${DATASET_NAME}"
fi

# --- GPU count ---
NUM_GPUS="${NUM_GPUS:-4}"

# --- Build training command ---
# pi0.5 image-key note: see recipes/pi05/expert_only.sh for rename_map rationale.
TRAIN_CMD=(
    --policy.path="${POLICY_PATH}"
    ${DATASET_ARGS}
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.train_expert_only=false
    --policy.freeze_vision_encoder=false
    --policy.empty_cameras=2
    --rename_map='{"observation.images.rgb": "observation.images.base_0_rgb"}'
    --dataset.video_backend=pyav
    --num_workers="${NUM_WORKERS}"
    --seed="${SEED}"
    --output_dir="${OUTPUT_DIR}"
    --eval_freq=0
    --policy.push_to_hub=false
    --wandb.enable=true
    --wandb.disable_artifact=true
    --wandb.project="${WANDB_PROJECT:-language-table-vla}"
)

# --- Launch ---
echo "=== Pi0.5 Full Finetune (combined sim) ==="
echo "Dataset: ${DATASET_REPO}  (~1.2M episodes, ~58M frames)"
echo "Steps: ${STEPS}, Batch: ${BATCH_SIZE}, GPUs: ${NUM_GPUS}"
echo "Effective batch: $((BATCH_SIZE * NUM_GPUS))"
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
