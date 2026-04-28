#!/usr/bin/env bash
# Comprehensive Language Table eval benchmark.
#
# Edit CHECKPOINTS below to change which policies to evaluate. All other
# settings are env vars with sensible defaults — override per run, e.g.:
#
#   SEEDS="0 1 2" BLOCK_MODES="BLOCK_4 BLOCK_8" NUM_EPISODES=100 \
#       bash training/eval/run_benchmark.sh
#
# Output:
#   eval_results/<BENCHMARK_NAME>/
#     <policy_id>/
#       summary.json
#       episodes.csv
#       <block_mode>/<seed>/<reward>/
#         result.json
#         videos/ep000_success.mp4   (if --video_episodes set)
#     comparison.md      (final aggregated table with 95% CIs)
#     server.log         (latest LeRobot server stdout/stderr)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# === Checkpoints to evaluate ===========================================
# Format: "id|type|checkpoint_path|extra"
#   id              short name, becomes a directory under the benchmark output
#   type            "lerobot" or "lava"
#   checkpoint_path passed to --checkpoint_path
#   extra           for lava: path to config .py; ignored for lerobot ("")
# To compare more policies (e.g. multiple checkpoints during training), add
# more lines.
CHECKPOINTS=(
  "smolvla_full_93185|lerobot|/media/mateo/Storage/lerobot_checkpoints/smolvla_full_combined_sim_93185/pretrained_model|"
  "lava_resnet|lava|/home/mateo/projects/Isaac/language-table/checkpoints/bc_resnet_sim_checkpoint_955000|language_table/train/configs/language_table_resnet_sim_local.py"
)

# === Knobs (env-var overridable) =======================================
BENCHMARK_NAME="${BENCHMARK_NAME:-comprehensive_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/${BENCHMARK_NAME}}"

NUM_EPISODES="${NUM_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-200}"
SEEDS="${SEEDS:-0 1 2}"
BLOCK_MODES="${BLOCK_MODES:-BLOCK_8}"
# Default reward_types: 8 of 10 in REWARD_REGISTRY. multistep and
# sortcolorstocorners emit a single compound instruction for a long-horizon
# task, which is OOD for policies trained on per-step LAVA data — opt in via
# REWARD_TYPES override if you want them.
REWARD_TYPES="${REWARD_TYPES:-blocktoblock blocktoabsolutelocation blocktoblockrelativelocation blocktorelativelocation separate block1tocorner point2block composite}"
DELAY_REWARD_STEPS="${DELAY_REWARD_STEPS:-0}"
# Episode indices to record per cell (e.g. "0 5 10"). Empty = no videos.
VIDEO_EPISODES="${VIDEO_EPISODES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

LTV_PYTHON="${LTV_PYTHON:-${REPO_ROOT}/ltvenv/bin/python}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-${REPO_ROOT}/lerobotenv/bin/python}"
SERVER_PORT="${SERVER_PORT:-50100}"
SERVER_DEVICE="${SERVER_DEVICE:-cuda}"

mkdir -p "${OUTPUT_ROOT}"

n_modes=$(echo ${BLOCK_MODES} | wc -w)
n_seeds=$(echo ${SEEDS} | wc -w)
n_rewards=$(echo ${REWARD_TYPES} | wc -w)
n_cells=$(( n_modes * n_seeds * n_rewards ))
n_eps=$(( n_cells * NUM_EPISODES ))

echo "================================================================"
echo "= Benchmark: ${BENCHMARK_NAME}"
echo "= Output:    ${OUTPUT_ROOT}"
echo "================================================================"
echo "Checkpoints:    ${#CHECKPOINTS[@]}"
echo "Block modes:    ${BLOCK_MODES} (${n_modes})"
echo "Seeds:          ${SEEDS} (${n_seeds})"
echo "Reward types:   ${n_rewards} (${REWARD_TYPES})"
echo "Episodes/cell:  ${NUM_EPISODES}"
echo "Cells/ckpt:     ${n_cells}"
echo "Episodes/ckpt:  ${n_eps}"
echo "Total episodes: $(( n_eps * ${#CHECKPOINTS[@]} ))"
echo "max_steps:      ${MAX_STEPS}"
echo "delay_reward:   ${DELAY_REWARD_STEPS}"
[ -n "${VIDEO_EPISODES}" ] && echo "Videos:         eps ${VIDEO_EPISODES}"
echo ""

# === LeRobot server lifecycle (one server per checkpoint) ==============
SERVER_PID=""
start_server() {
    local ckpt="$1"
    echo ">>> Starting LeRobot server: ${ckpt}"
    "${LEROBOT_PYTHON}" -u "${REPO_ROOT}/training/eval/lerobot_policy_server.py" \
        --checkpoint_path "${ckpt}" \
        --port "${SERVER_PORT}" \
        --device "${SERVER_DEVICE}" \
        > "${OUTPUT_ROOT}/server.log" 2>&1 &
    SERVER_PID=$!
    echo "    PID=${SERVER_PID}, port=${SERVER_PORT}, log=${OUTPUT_ROOT}/server.log"
    local timeout="${SERVER_TIMEOUT:-300}"
    for i in $(seq 1 "${timeout}"); do
        if grep -q "Policy server listening" "${OUTPUT_ROOT}/server.log" 2>/dev/null; then
            echo "    Server ready after ${i}s"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: server process died. Last log:"
            tail -30 "${OUTPUT_ROOT}/server.log"
            exit 1
        fi
        if [ $((i % 30)) -eq 0 ]; then
            echo "    ...still waiting (${i}s); last log line: $(tail -1 ${OUTPUT_ROOT}/server.log 2>/dev/null)"
        fi
        sleep 1
    done
    echo "ERROR: server did not become ready within ${timeout}s"
    tail -30 "${OUTPUT_ROOT}/server.log"
    exit 1
}

stop_server() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ">>> Stopping LeRobot server (PID ${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT INT TERM

# === Main loop =========================================================
for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r ID TYPE CKPT EXTRA <<< "${entry}"
    OUTDIR="${OUTPUT_ROOT}/${ID}"
    mkdir -p "${OUTDIR}"

    echo ""
    echo "================================================================"
    echo "= [${ID}] type=${TYPE}"
    echo "= ckpt=${CKPT}"
    echo "================================================================"

    BASE_ARGS=(
        --output_dir="${OUTDIR}"
        --num_episodes="${NUM_EPISODES}"
        --max_steps="${MAX_STEPS}"
        --delay_reward_steps="${DELAY_REWARD_STEPS}"
        --seeds ${SEEDS}
        --block_modes ${BLOCK_MODES}
        --reward_types ${REWARD_TYPES}
    )
    [ "${SKIP_EXISTING}" = "true" ] && BASE_ARGS+=(--skip_existing)
    [ -n "${VIDEO_EPISODES}" ] && BASE_ARGS+=(--video_episodes ${VIDEO_EPISODES})

    case "${TYPE}" in
        lerobot)
            start_server "${CKPT}"
            "${LTV_PYTHON}" -u "${REPO_ROOT}/training/eval/run_eval.py" \
                --policy_type=lerobot \
                --checkpoint_path="${CKPT}" \
                --server_port="${SERVER_PORT}" \
                "${BASE_ARGS[@]}"
            stop_server
            ;;
        lava)
            "${LTV_PYTHON}" -u "${REPO_ROOT}/training/eval/run_eval.py" \
                --policy_type=lava \
                --checkpoint_path="${CKPT}" \
                --config="${EXTRA}" \
                "${BASE_ARGS[@]}"
            ;;
        *)
            echo "ERROR: unknown TYPE=${TYPE} for ${ID}"
            exit 1
            ;;
    esac
done

# === Aggregate =========================================================
echo ""
echo "================================================================"
echo "= Aggregating results"
echo "================================================================"
"${LTV_PYTHON}" -u "${REPO_ROOT}/training/eval/aggregate_results.py" \
    "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/comparison.md"

echo ""
echo "Done. Results at: ${OUTPUT_ROOT}/"
echo "  Comparison:    ${OUTPUT_ROOT}/comparison.md"
