#!/usr/bin/env bash
# Submit one probe job per preset. Each job runs on a single GPU for up to 1h.
# Outputs land in training/outputs/probe_${PRESET}_${JOB_ID}.json.
#
# Usage:
#   bash training/slurm/submit_all_probes.sh
#   PRESETS="smolvla_full pi05_full" bash training/slurm/submit_all_probes.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_SLURM="${SCRIPT_DIR}/probe.slurm"

PRESETS="${PRESETS:-smolvla_full pi05_expert pi05_full}"

# Each probe.slurm job refreshes meta/stats.json for the dataset it uses
# (see probe.slurm). Skip the refresh by exporting SKIP_STATS_PULL=1 below.
for preset in ${PRESETS}; do
    echo "Submitting probe: ${preset}"
    PRESET="${preset}" sbatch "${PROBE_SLURM}"
done

echo ""
echo "Submitted. Track with:  squeue -u \$USER"
echo "Tail logs:              tail -f lt-vla-probe-<jobid>.out"
