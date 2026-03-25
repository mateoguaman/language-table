#!/bin/bash
#SBATCH -D .
#SBATCH --job-name=optim-bench
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=2:00:00

# Submit from the language-table repo root:
#   sbatch language_table/lamer/optim/run_bench.slurm
#
# Or use the submit helper:
#   bash language_table/lamer/optim/submit_bench.sh
#
# Override batch sizes:
#   BATCH_SIZES=1,4,16,64,256,512,1024,2048 \
#     sbatch language_table/lamer/optim/run_bench.slurm

set -euo pipefail

# --- Cluster config ---
LAMER_DIR="${LAMER_DIR:-}"
if [ -z "${LAMER_DIR}" ]; then
    # Try to find it relative to language-table
    CANDIDATE="$(cd "$(pwd)/../LaMer" 2>/dev/null && pwd || echo "")"
    if [ -n "${CANDIDATE}" ] && [ -f "${CANDIDATE}/.env.language_table" ]; then
        LAMER_DIR="${CANDIDATE}"
    fi
fi

if [ -n "${LAMER_DIR}" ] && [ -f "${LAMER_DIR}/.env.language_table" ]; then
    source "${LAMER_DIR}/.env.language_table"
fi

LANGTABLE_DIR="${LANGTABLE_DIR:-$(pwd)}"
LANGTABLE_CONDA_ENV="${LANGTABLE_CONDA_ENV:-ltvenv}"

# --- Conda setup ---
if [ -z "${CONDA_PREFIX:-}" ]; then
    module load conda/Miniforge3-25.3.1-3 2>/dev/null || true
fi
module load gcc/13.4.0 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${LANGTABLE_CONDA_ENV}"
set -u

# --- Cache dirs for Tillicum ---
[ -n "${CONDA_PKGS_DIRS:-}" ] && mkdir -p "${CONDA_PKGS_DIRS}"
[ -n "${PIP_CACHE_DIR:-}" ] && mkdir -p "${PIP_CACHE_DIR}"
[ -n "${TMPDIR:-}" ] && mkdir -p "${TMPDIR}"
[ -n "${XDG_CACHE_HOME:-}" ] && mkdir -p "${XDG_CACHE_HOME}"

# --- Environment ---
export PYTHONPATH="${LANGTABLE_DIR}:${LAMER_DIR:+${LAMER_DIR}:}${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# --- Defaults ---
BATCH_SIZES="${BATCH_SIZES:-1,4,16,64,256,512,1024,2048}"
NUM_WARMUP="${NUM_WARMUP:-3}"
NUM_ITERS="${NUM_ITERS:-10}"
IMG_H="${IMG_H:-180}"
IMG_W="${IMG_W:-320}"
DISCREPANCY_BS="${DISCREPANCY_BS:-64}"
DISCREPANCY_STEPS="${DISCREPANCY_STEPS:-3}"

# --- Output ---
OUTPUT_DIR="${LAMER_DIR:+${LAMER_DIR}/benchmarks/results/optim-$(date +%Y%m%d-%H%M%S)}"
if [ -n "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi

echo "============================================"
echo "_build_batch Optimization Benchmark (SLURM)"
echo "============================================"
echo "START:       $(date)"
echo "HOSTNAME:    $(hostname)"
echo "JOB_ID:      ${SLURM_JOB_ID:-n/a}"
echo "CONDA ENV:   ${LANGTABLE_CONDA_ENV}"
echo "PYTHON:      $(which python)"
echo "LANGTABLE:   ${LANGTABLE_DIR}"
echo "BATCH_SIZES: ${BATCH_SIZES}"
echo "IMAGE:       ${IMG_H}×${IMG_W}"
echo "WARMUP/ITER: ${NUM_WARMUP}/${NUM_ITERS}"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

BENCH_LOG="${OUTPUT_DIR:+${OUTPUT_DIR}/optim_bench.log}"
BENCH_LOG="${BENCH_LOG:-optim_bench_$(date +%Y%m%d-%H%M%S).log}"
DISCREPANCY_LOG="${OUTPUT_DIR:+${OUTPUT_DIR}/discrepancy.log}"
DISCREPANCY_LOG="${DISCREPANCY_LOG:-discrepancy_$(date +%Y%m%d-%H%M%S).log}"

# ---- Part 1: Performance benchmark ----
echo "============================================"
echo "Part 1/2: Performance benchmark"
echo "============================================"

python -m language_table.lamer.optim.bench \
    --batch_sizes "${BATCH_SIZES}" \
    --num_warmup "${NUM_WARMUP}" \
    --num_iters "${NUM_ITERS}" \
    --img_h "${IMG_H}" \
    --img_w "${IMG_W}" \
    2>&1 | tee "${BENCH_LOG}"

# ---- Part 2: Numerical discrepancy analysis ----
echo ""
echo "============================================"
echo "Part 2/2: Numerical discrepancy analysis"
echo "============================================"

python -m language_table.lamer.optim.measure_discrepancy \
    --batch_size "${DISCREPANCY_BS}" \
    --img_h "${IMG_H}" \
    --img_w "${IMG_W}" \
    --steps "${DISCREPANCY_STEPS}" \
    2>&1 | tee "${DISCREPANCY_LOG}"

echo ""
echo "============================================"
echo "Done."
echo "  Benchmark log:    ${BENCH_LOG}"
echo "  Discrepancy log:  ${DISCREPANCY_LOG}"
echo "END: $(date)"
