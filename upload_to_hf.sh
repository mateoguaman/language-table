#!/bin/bash
# Upload all 9 Language Table v3.0 datasets to HuggingFace Hub.
# Smallest → largest for quick feedback.
set -e

HF_USER="${HF_USER:-mateoguaman}"
INPUT="${INPUT:-/media/mateo/Storage/lerobot_datasets_v3}"

for name in \
    language_table_blocktoblock_4block_sim \
    language_table_blocktoblock_sim \
    language_table_separate_oracle_sim \
    language_table_sim \
    language_table_blocktorelative_oracle_sim \
    language_table_blocktoblock_oracle_sim \
    language_table_blocktoblockrelative_oracle_sim \
    language_table_blocktoabsolute_oracle_sim \
    language_table ; do
    echo ""
    echo "============================================================"
    echo "  Uploading $name"
    echo "============================================================"
    hf upload-large-folder \
        "${HF_USER}/${name}" \
        "${INPUT}/${name}" \
        --repo-type dataset \
        --no-private
    echo "[DONE] $name uploaded"
done

echo ""
echo "============================================================"
echo "  All datasets uploaded!"
echo "============================================================"
