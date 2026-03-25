#!/bin/bash
# Run the _build_batch optimization benchmark.
#
# Usage (local):
#   bash language_table/lamer/optim/run_bench.sh
#
# Usage (Tillicum via SLURM):
#   sbatch language_table/lamer/optim/run_bench.slurm
#
# Optional env vars:
#   BATCH_SIZES   — comma-separated (default: 1,4,16,64,256,512,1024)
#   NUM_WARMUP    — warmup iterations (default: 2)
#   NUM_ITERS     — timed iterations (default: 10)
#   IMG_H / IMG_W — input image size (default: 180×320, matches real env)
#   ENV_GPU       — GPU id for JAX variants (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGTABLE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LAMER_DIR="${LAMER_DIR:-$(cd "${LANGTABLE_DIR}/../LaMer" 2>/dev/null && pwd || echo "")}"

# Source cluster config if available
if [ -n "${LAMER_DIR}" ] && [ -f "${LAMER_DIR}/.env.language_table" ]; then
    source "${LAMER_DIR}/.env.language_table"
fi

# Find the right Python — prefer ltvenv conda env
if [ -n "${LANGTABLE_PYTHON:-}" ]; then
    PYTHON="${LANGTABLE_PYTHON}"
elif command -v conda &>/dev/null; then
    LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-ltvenv}"
    PYTHON="$(conda run -n "${LANGTABLE_CONDA_ENV}" which python 2>/dev/null || echo "")"
    if [ -z "${PYTHON}" ]; then
        echo "WARNING: conda env '${LANGTABLE_CONDA_ENV}' not found, falling back to system python"
        PYTHON="$(which python3 || which python)"
    fi
else
    PYTHON="$(which python3 || which python)"
fi

# Defaults
ENV_GPU="${ENV_GPU:-0}"
BATCH_SIZES="${BATCH_SIZES:-1,4,16,64,256,512,1024}"
NUM_WARMUP="${NUM_WARMUP:-2}"
NUM_ITERS="${NUM_ITERS:-10}"
IMG_H="${IMG_H:-180}"
IMG_W="${IMG_W:-320}"
DISCREPANCY_BS="${DISCREPANCY_BS:-64}"
DISCREPANCY_STEPS="${DISCREPANCY_STEPS:-3}"

export PYTHONPATH="${LANGTABLE_DIR}:${LAMER_DIR:+${LAMER_DIR}:}${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "============================================"
echo "_build_batch Optimization Benchmark"
echo "============================================"
echo "PYTHON:      ${PYTHON}"
echo "LANGTABLE:   ${LANGTABLE_DIR}"
echo "GPU:         ${ENV_GPU}"
echo "BATCH_SIZES: ${BATCH_SIZES}"
echo "IMAGE:       ${IMG_H}×${IMG_W}"
echo "WARMUP/ITER: ${NUM_WARMUP}/${NUM_ITERS}"
echo ""

echo "Part 1/2: Performance benchmark"
echo ""

CUDA_VISIBLE_DEVICES=${ENV_GPU} \
${PYTHON} -m language_table.lamer.optim.bench \
    --batch_sizes "${BATCH_SIZES}" \
    --num_warmup "${NUM_WARMUP}" \
    --num_iters "${NUM_ITERS}" \
    --img_h "${IMG_H}" \
    --img_w "${IMG_W}" \
    2>&1 | tee "optim_bench_$(date +%Y%m%d-%H%M%S).log"

echo ""
echo "Part 2/2: Numerical discrepancy analysis"
echo ""

CUDA_VISIBLE_DEVICES=${ENV_GPU} \
${PYTHON} -m language_table.lamer.optim.measure_discrepancy \
    --batch_size "${DISCREPANCY_BS}" \
    --img_h "${IMG_H}" \
    --img_w "${IMG_W}" \
    --steps "${DISCREPANCY_STEPS}" \
    2>&1 | tee "discrepancy_$(date +%Y%m%d-%H%M%S).log"
