#!/bin/bash
#SBATCH --job-name=lt-vla-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --output=lt-vla-probe-%j.out
#SBATCH --error=lt-vla-probe-%j.err
#
# Probe max per-GPU batch size for a given policy + training mode.
#
# Usage:
#   PRESET=smolvla_full sbatch training/slurm/probe.slurm
#   PRESET=pi05_expert BATCH_SIZES=8,16,32,64 sbatch training/slurm/probe.slurm
#   PRESET=pi05_full sbatch training/slurm/probe.slurm
#
# Env vars:
#   PRESET       (required) one of: smolvla_expert, smolvla_full,
#                pi0_expert, pi0_full, pi05_expert, pi05_full
#   BATCH_SIZES  (optional) comma-separated list. Defaults chosen per preset.
#   DATASET_REPO (optional) HF repo id; default mateoguaman/language_table_sim_combined
#   DATASET_NAME (optional) folder under $DATASET_ROOT; default language_table_sim_combined
#   STEPS        (optional) steps per probe; default 5
#   NUM_WORKERS  (optional) dataloader workers; default 2 (probe only — training uses more)

set -euo pipefail

TRAINING_DIR="${TRAINING_DIR:-${SLURM_SUBMIT_DIR}/training}"

source "${TRAINING_DIR}/.env.tillicum"
[ -f "${TRAINING_DIR}/.env.user" ] && source "${TRAINING_DIR}/.env.user"

mkdir -p "${SLURM_LOG_DIR}"

module load "${MODULE_CONDA}"
module load "${MODULE_GCC}"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

export HF_HOME HF_LEROBOT_HOME HF_HUB_ETAG_TIMEOUT HF_HUB_ENABLE_HF_TRANSFER
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

if [ -z "${PRESET:-}" ]; then
    echo "ERROR: PRESET env var not set."
    echo "Valid presets: smolvla_expert, smolvla_full, pi0_expert, pi0_full, pi05_expert, pi05_full"
    exit 1
fi

# Per-preset default batch-size ladders. Override with BATCH_SIZES=...
case "${PRESET}" in
    smolvla_expert) DEFAULT_BS="32,64,96,128,192,256,320,384" ;;
    smolvla_full)   DEFAULT_BS="8,16,32,64,96" ;;
    pi0_expert)     DEFAULT_BS="8,16,32,64,96" ;;
    pi0_full)       DEFAULT_BS="4,8,16,32,48" ;;
    pi05_expert)    DEFAULT_BS="8,16,32,64,96" ;;
    pi05_full)      DEFAULT_BS="4,8,16,32,48" ;;
    *) echo "ERROR: unknown PRESET '${PRESET}'"; exit 1 ;;
esac

BATCH_SIZES="${BATCH_SIZES:-${DEFAULT_BS}}"
DATASET_REPO="${DATASET_REPO:-mateoguaman/language_table_sim_combined}"
DATASET_NAME="${DATASET_NAME:-language_table_sim_combined}"
STEPS="${STEPS:-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"

OUTPUT_JSON="${TRAINING_DIR}/outputs/probe_${PRESET}_${SLURM_JOB_ID:-local}.json"
mkdir -p "$(dirname "${OUTPUT_JSON}")"

DATASET_ROOT_ARG=""
if [ -d "${DATASET_ROOT}/${DATASET_NAME}" ]; then
    DATASET_ROOT_ARG="--dataset_root ${DATASET_ROOT}/${DATASET_NAME}"
fi

echo "============================================"
echo " Batch-size probe: ${PRESET}"
echo "============================================"
echo "START:        $(date)"
echo "HOSTNAME:     $(hostname)"
echo "JOB_ID:       ${SLURM_JOB_ID:-local}"
echo "PRESET:       ${PRESET}"
echo "BATCH_SIZES:  ${BATCH_SIZES}"
echo "DATASET:      ${DATASET_REPO}  (root=${DATASET_ROOT}/${DATASET_NAME})"
echo "STEPS:        ${STEPS}"
echo "OUTPUT:       ${OUTPUT_JSON}"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

python "${TRAINING_DIR}/probe_batch_size.py" \
    --preset "${PRESET}" \
    --dataset_repo "${DATASET_REPO}" \
    ${DATASET_ROOT_ARG} \
    --batch_sizes "${BATCH_SIZES}" \
    --steps "${STEPS}" \
    --num_workers "${NUM_WORKERS}" \
    --output_json "${OUTPUT_JSON}"

echo ""
echo "============================================"
echo "END: $(date)"
