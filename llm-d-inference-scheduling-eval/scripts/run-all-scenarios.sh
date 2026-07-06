#!/bin/bash
# Outer loop: run all models × profiles × EPP configs.
# Starts with Qwen3-30B-A3B smoke test for fast turnaround.
#
# Usage:
#   bash scripts/run-all-scenarios.sh 2>&1 | stdbuf -oL tee /tmp/epp-eval-benchmark.log
#   # Monitor: tail -F /tmp/epp-eval-benchmark.log
set -euo pipefail

# Re-exec with line-buffered stdout if piped (so tail -F works)
if [ ! -t 1 ] && [ -z "${_UNBUFFERED:-}" ]; then
    export _UNBUFFERED=1
    exec stdbuf -oL "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export KUBECONFIG="${KUBECONFIG:-${HOME}/psap/kubeconfig-psap-janus}"
export NAMESPACE="${NAMESPACE:-llm-d-nathans-epp-eval}"
export HARDWARE="${HARDWARE:-2x8xH200}"
export SOFTWARE="${SOFTWARE:-rhoai-3.5ea1}"

EPP_CONFIGS="${EPP_CONFIGS:-prior-default optimized-baseline}"

echo "============================================================"
echo "  EPP Scheduling Evaluation: Optimized Baseline vs Prior Default"
echo "  $(date -u)"
echo "  EPP configs: ${EPP_CONFIGS}"
echo "============================================================"

run_model() {
    local MODEL="$1"
    local MODEL_NAME="$2"
    local LLM_SERVICE_NAME="$3"
    local TENSOR_PARALLEL_SIZE="$4"
    local REPLICAS="$5"
    shift 5
    local PROFILES="$*"

    export MODEL MODEL_NAME LLM_SERVICE_NAME TENSOR_PARALLEL_SIZE REPLICAS

    for PROFILE in ${PROFILES}; do
        for EPP_CONFIG in ${EPP_CONFIGS}; do
            export PROFILE EPP_CONFIG
            echo ""
            echo ">>> ${MODEL_NAME} | ${PROFILE} | ${EPP_CONFIG} | replica${REPLICAS} <<<"
            bash "${SCRIPT_DIR}/run-benchmark.sh" || \
                echo "WARNING: ${MODEL_NAME} ${PROFILE} ${EPP_CONFIG} replica${REPLICAS} failed, continuing"
        done
    done
}

# ── Qwen3-30B-A3B (TP=1, smoke test) ───────────────────────────────────────
echo ""
echo "================================================================"
echo "  Smoke Test: Qwen/Qwen3-30B-A3B-Instruct-2507 (TP=1)"
echo "================================================================"
run_model "Qwen/Qwen3-30B-A3B-Instruct-2507" "Qwen3-30B-A3B" "qwen3-30b" 1 1 \
    multi-turn

# ── Llama-3.3-70B-FP8 (TP=2, primary model) ────────────────────────────────
echo ""
echo "================================================================"
echo "  Primary: RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic (TP=2)"
echo "================================================================"
run_model "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" "Llama-3.3-70B-FP8" "llama-70b" 2 1 \
    multi-turn heavy-heterogeneous prefix-cache-stress

# ── gpt-oss-120b (TP=4, MoE coverage) ──────────────────────────────────────
echo ""
echo "================================================================"
echo "  MoE: openai/gpt-oss-120b (TP=4)"
echo "================================================================"
run_model "openai/gpt-oss-120b" "gpt-oss-120b" "gpt-oss-120b" 4 1 \
    multi-turn

echo ""
echo "============================================================"
echo "  All benchmark runs complete"
echo "  $(date -u)"
echo "============================================================"
