#!/bin/bash
# Convert every Language Table RLDS dataset to LeRobot v2.0.
#
# Order is smallest -> largest so any pipeline bug surfaces on the tiny
# blocktoblock_sim (8k episodes) before we burn bandwidth on the 442k-episode
# real-robot set.
#
# Usage:
#     bash convert_all_datasets.sh                       # all 9 datasets
#     OUTPUT=/data/lerobot_datasets bash convert_all_datasets.sh
#     DATASETS="language_table_sim language_table" bash convert_all_datasets.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
: "${OUTPUT:=${ROOT}/lerobot_datasets}"
: "${PYTHON:=${ROOT}/ltvenv/bin/python}"
: "${DATASETS:=language_table_blocktoblock_sim \
                language_table_blocktoblock_4block_sim \
                language_table_sim \
                language_table_blocktoblock_oracle_sim \
                language_table_blocktoblockrelative_oracle_sim \
                language_table_blocktoabsolute_oracle_sim \
                language_table_blocktorelative_oracle_sim \
                language_table_separate_oracle_sim \
                language_table}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT}"

for name in ${DATASETS}; do
    echo
    echo "==========================================================="
    echo "  Converting ${name}"
    echo "  Output: ${OUTPUT}/${name}"
    echo "==========================================================="
    "${PYTHON}" "${ROOT}/convert_to_lerobot.py" \
        --dataset_name "${name}" \
        --output_dir "${OUTPUT}"
done

echo
echo "All conversions complete. Datasets in ${OUTPUT}"
