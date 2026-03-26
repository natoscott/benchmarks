#!/bin/bash
# KV cache stress benchmark — RHOAI 3.3 on 2×8×H200.
#
# Identical to run-rhoai-3.3.sh except output_tokens=512 (vs 128 baseline).
# This increases per-sequence unique KV footprint ~4×, pushing the GPU KV cache
# into genuine saturation at rate=100–150 for Llama-3.1-70B-FP8:
#
#   Standard (output=128):  rate=100 → 26% GPU KV usage  → no eviction → offload never triggered
#   KV-stress (output=512): rate=100 → 121% GPU KV blocks → real eviction → offload has a role
#
# Rates are kept identical to the standard suite so results can be compared
# on the same x-axis. The interesting transition occurs at rate=100–150.
#
# Usage:
#   bash scripts/run-rhoai-3.3-kv-stress.sh
#   RUNS="no-offload" bash scripts/run-rhoai-3.3-kv-stress.sh
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$(dirname "$0")/../kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export HARDWARE="2x8xH200"
export SOFTWARE="rhoai-3.3-kv-stress"   # distinct tag — results land separately from standard runs
export TENSOR_PARALLEL_SIZE="2"
export MAX_SECONDS="120"
export TURNS="5"
export RATE_TYPE="concurrent"
export RANDOM_SEED="889"

# ── Workload — the key difference from run-rhoai-3.3.sh ─────────────────────
export PROMPT_TOKENS="512"
export OUTPUT_TOKENS="512"    # 4× the standard 128; drives genuine GPU KV saturation
export PREFIX_TOKENS="10000"  # keep shared prefix so EPP routing is still exercised
# ────────────────────────────────────────────────────────────────────────────

# Same rates as the standard suite for direct comparison
RATE_LIST="${RATE_LIST:-1,50,100,150,300,400,500,650}"
IFS=',' read -ra RATES <<< "${RATE_LIST}"

RUNS="${RUNS:-no-offload native-offload-20k}"
MODELS="${MODELS:-RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 openai/gpt-oss-120b}"
REPLICAS="${REPLICAS:-1 2}"

echo "=========================================="
echo "RHOAI 3.3 KV Cache Stress Benchmark"
echo "Hardware:     ${HARDWARE}"
echo "Software:     ${SOFTWARE}"
echo "Output tokens: ${OUTPUT_TOKENS} (standard=128)"
echo "Configs:      ${RUNS}"
echo "Models:       ${MODELS}"
echo "Replicas:     ${REPLICAS}"
echo "Rates:        ${RATE_LIST}"
echo "=========================================="

for replicas in ${REPLICAS}; do
    export CURRENT_REPLICAS="${replicas}"

    for model in ${MODELS}; do
        export MODEL="${model}"
        export MODEL_NAME="${model##*/}"

        case "${MODEL_NAME}" in
            "Meta-Llama-3.1-70B-Instruct-FP8")
                export LLM_SERVICE_NAME="llama-70b"
                export GPU_MEMORY_UTILIZATION="0.75"
                ;;
            "gpt-oss-120b")
                export LLM_SERVICE_NAME="gpt-oss-120b"
                export GPU_MEMORY_UTILIZATION="0.65"
                ;;
            *)
                echo "ERROR: Unknown model ${MODEL_NAME}"; exit 1 ;;
        esac

        for run in ${RUNS}; do
            export PARAMETERS="${run}"
            export NUM_CPU_BLOCKS="20000"

            for rate in "${RATES[@]}"; do
                export RATE="${rate}"
                export PREFIX_COUNT=$(( 2 * rate ))

                echo ""
                echo "▶ ${HARDWARE} ${SOFTWARE} | ${MODEL_NAME} | ${run} | replicas=${replicas} | rate=${rate}"

                bash "$(dirname "$0")/run-benchmark.sh"
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All KV stress benchmarks complete. Results in: results/"
echo "=========================================="
