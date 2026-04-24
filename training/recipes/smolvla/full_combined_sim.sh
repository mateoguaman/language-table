#!/usr/bin/env bash
# SmolVLA full finetune on language_table_sim_combined (~1.2M eps, ~58M frames).
# Every param trainable: SigLIP vision, SmolLM backbone, action expert, projections.
# Target: 4 GPUs. Probed ceiling on 1×H200: bs=96 at 82.3% VRAM (~115 GiB); bs=128
# extrapolates to ~154 GiB (OOM). See docs/batch-size-probes.md.
# DDP overhead: each rank's NCCL peer-to-peer buffers take ~1 GiB on rank 0. At
# bs=96 with 8 ranks that's ~7-8 GiB extra on GPU 0 → OOM. Rule of thumb:
#   1-2 GPUs: bs=96   4 GPUs: bs=80   8 GPUs: bs=64
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source environment
[ -f "${TRAINING_DIR}/.env.tillicum" ] && source "${TRAINING_DIR}/.env.tillicum"
[ -f "${TRAINING_DIR}/.env.user" ] && source "${TRAINING_DIR}/.env.user"

# --- Training parameters ---
POLICY_PATH="lerobot/smolvla_base"
DATASET_REPO="mateoguaman/language_table_sim_combined"
DATASET_NAME="language_table_sim_combined"

BATCH_SIZE="${BATCH_SIZE:-96}"
# STEPS is derived from EPOCHS when set. LeRobot only accepts --steps, so we
# convert: steps = ceil(EPOCHS * total_frames / (BATCH_SIZE * NUM_GPUS)).
# total_frames comes from ${DATASET_ROOT}/${DATASET_NAME}/meta/info.json.
EPOCHS="${EPOCHS:-}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1000}"
RESUME="${RESUME:-false}"

# --- Output ---
# Override OUTPUT_DIR directly when resuming (RESUME=true needs the old dir).
JOB_NAME="smolvla_full_combined_sim"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-outputs}/${JOB_NAME}_${RUN_ID}}"

# --- Dataset location ---
DATASET_ARGS="--dataset.repo_id=${DATASET_REPO}"
if [ -n "${DATASET_ROOT:-}" ] && [ -d "${DATASET_ROOT}/${DATASET_NAME}" ]; then
    DATASET_ARGS="${DATASET_ARGS} --dataset.root=${DATASET_ROOT}/${DATASET_NAME}"
fi

# --- GPU count ---
NUM_GPUS="${NUM_GPUS:-4}"

# --- Convert EPOCHS -> STEPS if requested ---
if [ -n "${EPOCHS}" ]; then
    INFO_JSON=""
    if [ -n "${DATASET_ROOT:-}" ] && [ -f "${DATASET_ROOT}/${DATASET_NAME}/meta/info.json" ]; then
        INFO_JSON="${DATASET_ROOT}/${DATASET_NAME}/meta/info.json"
    fi
    if [ -z "${INFO_JSON}" ]; then
        echo "ERROR: EPOCHS=${EPOCHS} requires a local dataset so total_frames can be read."
        echo "       Expected ${DATASET_ROOT}/${DATASET_NAME}/meta/info.json"
        exit 1
    fi
    TOTAL_FRAMES=$(python -c "import json; print(json.load(open('${INFO_JSON}'))['total_frames'])")
    EFF_BS=$((BATCH_SIZE * NUM_GPUS))
    STEPS=$(python -c "import math; print(math.ceil(${EPOCHS} * ${TOTAL_FRAMES} / ${EFF_BS}))")
    echo "EPOCHS=${EPOCHS} × total_frames=${TOTAL_FRAMES} / eff_bs=${EFF_BS} -> STEPS=${STEPS}"
fi

# --- Build training command ---
# Resume path: LeRobot picks the policy branch iff --policy.path is set; otherwise
# (--resume=true with --config_path) it rehydrates policy/processors from the saved
# train_config.json. We pick one or the other based on RESUME.
if [ "${RESUME}" = "true" ]; then
    CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "ERROR: RESUME=true but ${CONFIG_PATH} not found."
        echo "       Set OUTPUT_DIR to the run directory that contains checkpoints/last/."
        exit 1
    fi
    POLICY_ARGS=(--config_path="${CONFIG_PATH}")
else
    POLICY_ARGS=(--policy.path="${POLICY_PATH}")
fi

TRAIN_CMD=(
    "${POLICY_ARGS[@]}"
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
    --rename_map='{"observation.images.rgb": "observation.images.camera1"}'
    --dataset.video_backend=pyav
    --num_workers="${NUM_WORKERS}"
    --seed="${SEED}"
    --output_dir="${OUTPUT_DIR}"
    --resume="${RESUME}"
    --eval_freq=0
    --policy.push_to_hub=false
    --wandb.enable=true
    --wandb.disable_artifact=true
    --wandb.project="${WANDB_PROJECT:-language-table-vla}"
)

# --- Launch ---
echo "=== SmolVLA Full Finetune (combined sim) ==="
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
