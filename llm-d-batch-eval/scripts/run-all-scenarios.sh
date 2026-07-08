#!/bin/bash
# Outer loop: run all scenarios across all models.
# Starts with Qwen3-8B for fast turnaround and tooling validation.
#
# Usage:
#   bash scripts/run-all-scenarios.sh 2>&1 | stdbuf -oL tee /tmp/batch-gateway-benchmark.log
#   # Monitor: tail -F /tmp/batch-gateway-benchmark.log
set -euo pipefail

# Re-exec with line-buffered stdout if piped (so tail -F works)
if [ ! -t 1 ] && [ -z "${_UNBUFFERED:-}" ]; then
    export _UNBUFFERED=1
    exec stdbuf -oL "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export KUBECONFIG="${KUBECONFIG:-${HOME}/psap/kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-batch}"
export HARDWARE="${HARDWARE:-2x8xH200}"
export SOFTWARE="${SOFTWARE:-rhoai-3.5ea1}"

# Scenarios: 0=interactive-only, 2=ungated, 3=aimd, 4=aimd-flow-control
SCENARIOS="${SCENARIOS:-0 2 3 4}"

echo "============================================================"
echo "  llm-d Batch Gateway Performance Evaluation"
echo "  $(date -u)"
echo "  Scenarios: ${SCENARIOS}"
echo "============================================================"

run_model() {
    local MODEL="$1"
    local MODEL_NAME="$2"
    local LLM_SERVICE_NAME="$3"
    local TENSOR_PARALLEL_SIZE="$4"
    local REPLICAS="$5"

    export MODEL MODEL_NAME LLM_SERVICE_NAME TENSOR_PARALLEL_SIZE REPLICAS

    for SCENARIO in ${SCENARIOS}; do
        export SCENARIO
        echo ""
        echo ">>> ${MODEL_NAME} replica${REPLICAS} scenario${SCENARIO} <<<"
        bash "${SCRIPT_DIR}/run-benchmark.sh" || \
            echo "WARNING: ${MODEL_NAME} scenario ${SCENARIO} replica${REPLICAS} failed, continuing"
    done
}

# ── Qwen3-8B (TP=1, fast turnaround) ────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Model: Qwen/Qwen3-8B (TP=1)"
echo "================================================================"
run_model "Qwen/Qwen3-8B" "Qwen3-8B" "qwen3-8b" 1 1
run_model "Qwen/Qwen3-8B" "Qwen3-8B" "qwen3-8b" 1 4
run_model "Qwen/Qwen3-8B" "Qwen3-8B" "qwen3-8b" 1 8

# ── FP8-70B (TP=2) ──────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Model: RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 (TP=2)"
echo "================================================================"
run_model "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8" "Meta-Llama-3.1-70B-Instruct-FP8" "llama-70b" 2 1
run_model "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8" "Meta-Llama-3.1-70B-Instruct-FP8" "llama-70b" 2 4
run_model "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8" "Meta-Llama-3.1-70B-Instruct-FP8" "llama-70b" 2 8

# ── gpt-oss-120b (TP=4) ─────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Model: openai/gpt-oss-120b (TP=4)"
echo "================================================================"
run_model "openai/gpt-oss-120b" "gpt-oss-120b" "gpt-oss-120b" 4 1
run_model "openai/gpt-oss-120b" "gpt-oss-120b" "gpt-oss-120b" 4 2
run_model "openai/gpt-oss-120b" "gpt-oss-120b" "gpt-oss-120b" 4 4

echo ""
echo "============================================================"
echo "  All benchmark runs complete"
echo "  $(date -u)"
echo "============================================================"
