#!/usr/bin/env bash
# Submit one probe job per preset. Each job runs on a single GPU for up to 1h.
# Outputs land in training/outputs/probe_${PRESET}_${JOB_ID}.json.
#
# Usage:
#   bash training/slurm/submit_all_probes.sh
#   PRESETS="smolvla_full pi05_full" bash training/slurm/submit_all_probes.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROBE_SLURM="${SCRIPT_DIR}/probe.slurm"

PRESETS="${PRESETS:-smolvla_full pi05_expert pi05_full}"

# Pull the updated meta/stats.json for each dataset (pi0.5 needs QUANTILES stats).
# Skip by exporting SKIP_STATS_PULL=1.
if [ "${SKIP_STATS_PULL:-0}" != "1" ]; then
    set -a
    source "${TRAINING_DIR}/.env.tillicum"
    [ -f "${TRAINING_DIR}/.env.user" ] && source "${TRAINING_DIR}/.env.user"
    set +a

    DATASETS="${DATASETS:-language_table language_table_sim language_table_blocktoblock_sim \
        language_table_blocktoblock_4block_sim language_table_blocktoblock_oracle_sim \
        language_table_blocktoblockrelative_oracle_sim language_table_blocktoabsolute_oracle_sim \
        language_table_blocktorelative_oracle_sim language_table_separate_oracle_sim \
        language_table_sim_combined}"
    NAMESPACE="${HF_NAMESPACE:-mateoguaman}"

    echo "Pulling meta/stats.json for each dataset from HF Hub (revision=v3.0)"
    for name in ${DATASETS}; do
        hf download \
            --repo-type dataset \
            --revision v3.0 \
            --include "meta/stats.json" \
            --local-dir "${DATASET_ROOT}/${name}" \
            "${NAMESPACE}/${name}"
    done
    echo ""
fi

for preset in ${PRESETS}; do
    echo "Submitting probe: ${preset}"
    PRESET="${preset}" sbatch "${PROBE_SLURM}"
done

echo ""
echo "Submitted. Track with:  squeue -u \$USER"
echo "Tail logs:              tail -f lt-vla-probe-<jobid>.out"
