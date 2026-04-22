#!/usr/bin/env bash
# Pi0.5 expert-only finetune on language_table_sim_combined (~1.2M eps).
# Freezes SigLIP + PaliGemma backbone, trains only the gemma_expert action
# head + state/action projections.
# Target: 4 GPUs. Probed ceiling on H200: bs=16 per GPU at 99.2% VRAM (see
# docs/batch-size-probes.md) — default bs=12 leaves headroom for NCCL.
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

BATCH_SIZE="${BATCH_SIZE:-12}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1000}"

# --- Output ---
JOB_NAME="pi05_expert_combined_sim"
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
# pi0.5 declares image inputs as base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb
# (plus empty_camera_{i} from empty_cameras=N). Language Table has one camera —
# rename rgb → base_0_rgb; wrist cameras stay absent and get zero-masked by
# _preprocess_images (which only needs ≥1 declared image key in the batch).
TRAIN_CMD=(
    --policy.path="${POLICY_PATH}"
    ${DATASET_ARGS}
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.train_expert_only=true
    --policy.freeze_vision_encoder=true
    --policy.empty_cameras=2
    --rename_map='{"observation.images.rgb": "observation.images.base_0_rgb"}'
    --dataset.video_backend=pyav
    --num_workers="${NUM_WORKERS}"
    --seed="${SEED}"
    --output_dir="${OUTPUT_DIR}"
    --eval_freq=0
    --policy.push_to_hub=false
    --wandb.enable=true
    --wandb.project="${WANDB_PROJECT:-language-table-vla}"
)

# --- Launch ---
echo "=== Pi0.5 Expert-Only (combined sim) ==="
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
