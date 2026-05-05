#!/usr/bin/env bash
# Smoke-test SmolVLA training on the local blocktoabsolute oracle dataset.
# 1 GPU, 10 steps — verifies the pipeline runs end-to-end.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_NAME="language_table_blocktoabsolute_oracle_sim_padded"
OUTPUT_DIR="${PROJECT_DIR}/outputs/smolvla_local_test_$(date +%Y%m%d_%H%M%S)"

python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla \
    --policy.pretrained_path="mateoguaman/smolvla_lt_combined_sim_93185" \
    --dataset.repo_id="sidhraja/${DATASET_NAME}" \
    --dataset.root="${PROJECT_DIR}/${DATASET_NAME}" \
    --batch_size=32 \
    --steps=8000 \
    --save_freq=8000 \
    --log_freq=1 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --dataset.video_backend=pyav \
    --num_workers=16 \
    --seed=42 \
    --output_dir="${OUTPUT_DIR}" \
    --eval_freq=0 \
    --policy.push_to_hub=false \
    --wandb.enable=false

echo ""
echo "=== Smoke test complete. Output: ${OUTPUT_DIR} ==="
