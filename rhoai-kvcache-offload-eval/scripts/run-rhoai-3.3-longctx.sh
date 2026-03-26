#!/bin/bash
# Long-context KV cache offload benchmark — RHOAI 3.3 on 2×8×H200.
#
# Designed to push into the regime where CPU KV offload is genuinely beneficial:
#   - Long unique prompts (4,096 tokens/turn) maximise per-sequence KV footprint
#     and make recomputation expensive (~350 µs vs ~78 µs CPU fetch → 4.5× advantage)
#   - Reduced gpu_memory_utilization (0.50) shrinks the GPU KV pool to force eviction
#   - Rate list spans the full story across both models: pre-saturation → saturation
#     → active offload zone → collapse
#
# gpu_memory_utilization=0.50 for both models (140 × 0.50 = 70 GiB reserved):
#
#   Llama-70B-FP8  (~14,440 GPU blocks at 0.50):
#     Per sequence: 5 × (4,096 + 256) = 21,760 tokens = 1,360 blocks
#     rate=1:    9%   — comfortable
#     rate=10:  94%   — approaching saturation
#     rate=11: 104%   — saturation / active offload begins
#     rate=20: 188%   — heavy eviction
#     rate=50+:       — progressive collapse
#
#   gpt-oss-120b   (~131,000 GPU blocks at 0.50, MoE small KV footprint):
#     rate=100: 103%  — saturation
#     rate=200: 207%  — heavy eviction
#     rate=300: 311%  — collapse
#     (both models fully covered by the same rate list)
#
# Recomputation at 83,200 token context ≈ 1,300 µs >> CPU fetch ≈ 78 µs (17× advantage)
# This is the most favourable operating point for the OffloadingConnector.
#
# Usage:
#   bash scripts/run-rhoai-3.3-longctx.sh
#   MODELS="RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8" bash scripts/run-rhoai-3.3-longctx.sh
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$(dirname "$0")/../kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export HARDWARE="2x8xH200"
export SOFTWARE="rhoai-3.3-longctx"
export TENSOR_PARALLEL_SIZE="2"
export MAX_SECONDS="120"
export TURNS="5"
export RATE_TYPE="concurrent"
export RANDOM_SEED="889"

# ── Workload ──────────────────────────────────────────────────────────────────
export PROMPT_TOKENS="4096"    # 8× baseline — long enough for 4.5× recomputation advantage,
                               # short enough to show progression across both models' saturation points
export OUTPUT_TOKENS="256"     # modest outputs keep benchmark duration manageable
export PREFIX_TOKENS="10000"   # retain shared prefix for EPP routing realism
# ─────────────────────────────────────────────────────────────────────────────

# Rate list: low rates show saturation transition, higher rates show collapse
RATE_LIST="${RATE_LIST:-1,5,10,20,50,100,200,300}"
IFS=',' read -ra RATES <<< "${RATE_LIST}"

RUNS="${RUNS:-no-offload native-offload-20k}"
MODELS="${MODELS:-RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 openai/gpt-oss-120b}"
REPLICAS="${REPLICAS:-1 2}"

echo "=========================================="
echo "RHOAI 3.3 Long-Context KV Offload Benchmark"
echo "Hardware:      ${HARDWARE}"
echo "Software:      ${SOFTWARE}"
echo "Prompt tokens: ${PROMPT_TOKENS} (32× baseline)"
echo "Output tokens: ${OUTPUT_TOKENS}"
echo "Configs:       ${RUNS}"
echo "Models:        ${MODELS}"
echo "Replicas:      ${REPLICAS}"
echo "Rates:         ${RATE_LIST}"
echo "=========================================="

for replicas in ${REPLICAS}; do
    export CURRENT_REPLICAS="${replicas}"

    for model in ${MODELS}; do
        export MODEL="${model}"
        export MODEL_NAME="${model##*/}"

        case "${MODEL_NAME}" in
            "Meta-Llama-3.1-70B-Instruct-FP8")
                export LLM_SERVICE_NAME="llama-70b"
                # Reduced gpu_memory_utilization shrinks the GPU KV pool,
                # forcing eviction at low concurrency where recomputation is expensive.
                export GPU_MEMORY_UTILIZATION="0.50"
                ;;
            "gpt-oss-120b")
                export LLM_SERVICE_NAME="gpt-oss-120b"
                # 0.50: 70 GiB reserved, ~33 GiB weights → ~37 GiB KV/GPU → ~131,000 blocks.
                # MoE KV blocks are small so saturation arrives at rate=25-30 rather than
                # rate=3 (Llama). The rate list still captures it at rate=50 and beyond.
                export GPU_MEMORY_UTILIZATION="0.50"
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
echo "All long-context benchmarks complete. Results in: results/"
echo "=========================================="
