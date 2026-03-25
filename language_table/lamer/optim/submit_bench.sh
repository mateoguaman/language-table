#!/usr/bin/env bash
# Submit the optimization benchmark to SLURM on Tillicum.
#
# Usage:
#   cd /path/to/language-table
#   bash language_table/lamer/optim/submit_bench.sh
#
# Override settings:
#   BATCH_SIZES=1,4,16,64,256,512,1024,2048 \
#   NUM_ITERS=20 \
#     bash language_table/lamer/optim/submit_bench.sh
#
# Pass extra sbatch flags after --:
#   bash language_table/lamer/optim/submit_bench.sh -- --time=2:00:00 --qos=high

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGTABLE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LAMER_DIR="${LAMER_DIR:-$(cd "${LANGTABLE_DIR}/../LaMer" 2>/dev/null && pwd || echo "")}"

# Source cluster config
if [ -n "${LAMER_DIR}" ] && [ -f "${LAMER_DIR}/.env.language_table" ]; then
    source "${LAMER_DIR}/.env.language_table"
fi

export LAMER_DIR
export LANGTABLE_DIR
export LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-ltvenv}"
export BATCH_SIZES="${BATCH_SIZES:-1,4,16,64,256,512,1024,2048}"
export NUM_WARMUP="${NUM_WARMUP:-3}"
export NUM_ITERS="${NUM_ITERS:-10}"
export IMG_H="${IMG_H:-480}"
export IMG_W="${IMG_W:-640}"

# Export cache dirs so SLURM job inherits them
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-}"
export TMPDIR="${TMPDIR:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-}"

# Log directory
SLURM_LOG_DIR="${SLURM_LOG_DIR:-}"
SBATCH_LOG_ARGS=()
if [ -n "${SLURM_LOG_DIR}" ]; then
    mkdir -p "${SLURM_LOG_DIR}"
    SBATCH_LOG_ARGS+=(--output "${SLURM_LOG_DIR}/%x-%j.out")
    SBATCH_LOG_ARGS+=(--error "${SLURM_LOG_DIR}/%x-%j.err")
fi

SLURM_SCRIPT="${SCRIPT_DIR}/run_bench.slurm"

echo "Submitting optimization benchmark"
echo "  LANGTABLE_DIR:   ${LANGTABLE_DIR}"
echo "  LAMER_DIR:       ${LAMER_DIR}"
echo "  CONDA_ENV:       ${LANGTABLE_CONDA_ENV}"
echo "  BATCH_SIZES:     ${BATCH_SIZES}"
echo "  NUM_ITERS:       ${NUM_ITERS}"
echo ""

# Separate our args from extra sbatch args (after --)
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "${arg}" = "--" ]; then
        shift
        EXTRA_ARGS=("$@")
        break
    fi
    shift
done

exec sbatch "${SBATCH_LOG_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" "${SLURM_SCRIPT}"
