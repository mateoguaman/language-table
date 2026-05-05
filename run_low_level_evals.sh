#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SEED="${SEED:-0}"
NUM_TRIALS="${NUM_TRIALS:-20}"
MAX_STEPS="${MAX_STEPS:-200}"
BLOCK_MODE="${BLOCK_MODE:-BLOCK_4}"
REWARD_TYPE="${REWARD_TYPE:-blocktoabsolutelocation}"
PORT_START="${PORT_START:-50100}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/eval_low_level_results}"

LAVA_CHECKPOINT="${LAVA_CHECKPOINT:-/home/sidhraja/projects/LaMer/checkpoints}"
LAVA_CHECKPOINT_PREFIX="${LAVA_CHECKPOINT_PREFIX:-bc_resnet_sim_checkpoint_}"

mkdir -p "${OUTPUT_DIR}"

run_smolvla() {
    local label="$1"
    local checkpoint="$2"
    local port="$3"

    echo "=== Evaluating ${label} ==="
    "${PYTHON_BIN}" "${PROJECT_DIR}/eval_low_level.py" \
        --policy smolvla \
        --checkpoint "${checkpoint}" \
        --seed "${SEED}" \
        --num_trials "${NUM_TRIALS}" \
        --max_steps "${MAX_STEPS}" \
        --block_mode "${BLOCK_MODE}" \
        --reward_type "${REWARD_TYPE}" \
        --port "${port}" \
        --server_log "/tmp/${label}_low_level_eval.log" \
        --output_json "${OUTPUT_DIR}/${label}.json"
}

run_lava() {
    echo "=== Evaluating base_lava ==="
    "${PYTHON_BIN}" "${PROJECT_DIR}/eval_low_level.py" \
        --policy lava \
        --checkpoint "${LAVA_CHECKPOINT}" \
        --lava_checkpoint_prefix "${LAVA_CHECKPOINT_PREFIX}" \
        --seed "${SEED}" \
        --num_trials "${NUM_TRIALS}" \
        --max_steps "${MAX_STEPS}" \
        --block_mode "${BLOCK_MODE}" \
        --reward_type "${REWARD_TYPE}" \
        --output_json "${OUTPUT_DIR}/base_lava.json"
}

run_smolvla "smolvla_lt_combined_sim_93185" \
    "mateoguaman/smolvla_lt_combined_sim_93185" \
    "${PORT_START}"

run_smolvla "langtable_smolvla_finetuned" \
    "Sidharth-R/langtable-smolvla-finetuned" \
    "$((PORT_START + 1))"

run_smolvla "langtable_smolvla_padded" \
    "Sidharth-R/langtable-smolvla-padded" \
    "$((PORT_START + 2))"

run_lava

echo "=== Done. Results written to ${OUTPUT_DIR} ==="
