#!/bin/bash
# Main benchmark iteration script for RHOAI 3.3 KV cache offload evaluation.
#
# Configurations:
#   no-offload          — GPU-only baseline
#   native-offload-20k  — CPU offload via OffloadingConnector, 20K blocks
#                         (num_cpu_blocks API, compatible with RHOAI vLLM 0.4.x)
#
# Models:
#   meta-llama/Llama-3.1-70B-Instruct
#   openai/gpt-oss-120b   (MoE, MXFP4-quantised)
#
# Replica counts: 1 (single 2-GPU replica), 2 (two 2-GPU replicas, same node)
#
# Total runs: 2 configs × 2 models × 2 replica counts × 8 rates = 64 runs
#
# Usage:
#   bash scripts/run-rhoai-3.3.sh
#   RUNS="no-offload" MODELS="meta-llama/Llama-3.1-70B-Instruct" bash scripts/run-rhoai-3.3.sh
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export HARDWARE="${HARDWARE:-1x8xH200}"
export SOFTWARE="${SOFTWARE:-rhoai-3.3}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export MAX_SECONDS="${MAX_SECONDS:-120}"
export TURNS="${TURNS:-5}"
export RATE_TYPE="${RATE_TYPE:-concurrent}"
export RANDOM_SEED="${RANDOM_SEED:-889}"

# Benchmark rates (concurrency levels) — same as v0.4.0/v0.5.1 for comparability
RATE_LIST="${RATE_LIST:-1,50,100,150,300,400,500,650}"
IFS=',' read -ra RATES <<< "${RATE_LIST}"

# Configurations, models, and replica counts — can be overridden via environment
RUNS="${RUNS:-no-offload native-offload-20k}"
MODELS="${MODELS:-RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 openai/gpt-oss-120b meta-llama/Llama-3.1-70B-Instruct}"
REPLICAS="${REPLICAS:-1 2}"

echo "=========================================="
echo "RHOAI 3.3 KV Cache Offload Benchmark"
echo "Hardware:  ${HARDWARE}"
echo "Software:  ${SOFTWARE}"
echo "Configs:   ${RUNS}"
echo "Models:    ${MODELS}"
echo "Replicas:  ${REPLICAS}"
echo "Rates:     ${RATE_LIST}"
echo "=========================================="

for replicas in ${REPLICAS}; do
    export CURRENT_REPLICAS="${replicas}"

    for model in ${MODELS}; do
        export MODEL="${model}"
        export MODEL_NAME="${model##*/}"

        # Map model name to LLMInferenceService name and per-model settings
        case "${MODEL_NAME}" in
            "Meta-Llama-3.1-70B-Instruct-FP8")
                export LLM_SERVICE_NAME="llama-70b"
                export GPU_MEMORY_UTILIZATION="0.75"
                export PROMPT_TOKENS="512"
                export OUTPUT_TOKENS="128"
                export PREFIX_TOKENS="10000"
                ;;
            "gpt-oss-120b")
                export LLM_SERVICE_NAME="gpt-oss-120b"
                export GPU_MEMORY_UTILIZATION="0.65"
                export PROMPT_TOKENS="512"
                export OUTPUT_TOKENS="128"
                export PREFIX_TOKENS="10000"
                ;;
            "Llama-3.1-70B-Instruct")
                export LLM_SERVICE_NAME="llama-70b-bf16"
                export GPU_MEMORY_UTILIZATION="0.90"
                export PROMPT_TOKENS="512"
                export OUTPUT_TOKENS="128"
                export PREFIX_TOKENS="10000"
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
echo "All benchmarks complete. Results in: results/"
echo "=========================================="
