#!/bin/bash
# Convert all 9 Language Table datasets from LeRobot v2.0 → v3.0.
# Runs smallest → largest for early bug detection.
set -e

INPUT="${INPUT:-/media/mateo/Storage/lerobot_datasets}"
OUTPUT="${OUTPUT:-/media/mateo/Storage/lerobot_datasets_v3}"
PYTHON="${PYTHON:-./ltvenv/bin/python}"

for name in \
    language_table_blocktoblock_sim \
    language_table_blocktoblock_4block_sim \
    language_table_sim \
    language_table_blocktoblock_oracle_sim \
    language_table_blocktoblockrelative_oracle_sim \
    language_table_blocktoabsolute_oracle_sim \
    language_table_blocktorelative_oracle_sim \
    language_table_separate_oracle_sim \
    language_table ; do
    echo ""
    echo "============================================================"
    echo "  Converting $name"
    echo "============================================================"
    $PYTHON convert_v2_to_v3.py \
        --input_dir "$INPUT" \
        --output_dir "$OUTPUT" \
        --dataset_name "$name"
done

echo ""
echo "============================================================"
echo "  All datasets converted!"
echo "============================================================"
