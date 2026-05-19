#!/bin/bash
set -e

# LMCache benchmarks for llm-d v0.7.0 at reduced gpu_memory_utilization (memory-pressure).
# Same gmu values as v0.5.1-lmcache-mempress and v0.6.0-lmcache-mempress for comparison.
# Installs lmcache==0.4.4 (released 2026-04-22) via pip into ghcr.io/llm-d/llm-d-cuda:v0.7.0.
# v0.4.4 targets vLLM 0.18.x; compatibility with vLLM 0.19.1 is assumed until proven otherwise.
# If v0.4.4 fails, fall back to nightly-2026-05-13-cu13 with USE_LMCACHE_IMAGE=true.

RUNS="${RUNS:-lmcache-local lmcache-valkey}"
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ}"
REPLICAS="${REPLICAS:-1}"

export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export RATE_LIST="${RATE:-1,50,100,150,300,400,500,650}"
export MAX_SECONDS=120
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="${SOFTWARE:-upstream-llm-d-0.7.0-mempress}"
export TURNS=5
export INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-llm-d-model-server}"
export TENSOR_PARALLEL_SIZE=2
export GPUS_PER_REPLICA=2

IFS=',' read -ra RATES <<< "$RATE_LIST"

for replicas in ${REPLICAS}; do
    export CURRENT_REPLICAS="${replicas}"

    echo "=========================================="
    echo "Setting replica count to: ${replicas}"
    echo "=========================================="
    kubectl --kubeconfig="${KUBECONFIG}" scale deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" --replicas="${replicas}"
    kubectl --kubeconfig="${KUBECONFIG}" rollout status deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" --timeout=300s
    sleep 30

    for model in ${MODELS}; do
        export MODEL="${model}"
        export MODEL_NAME="${model##*/}"

        # Per-model gpu_memory_utilization (same as v0.5.1-lmcache-mempress and v0.6.0-lmcache-mempress)
        case "${MODEL_NAME}" in
            "Qwen3-0.6B")    export GPU_MEMORY_UTILIZATION=0.55 ;;
            "Qwen3-8B")      export GPU_MEMORY_UTILIZATION=0.65 ;;
            "Qwen3-14B")     export GPU_MEMORY_UTILIZATION=0.70 ;;
            "Qwen3-32B-AWQ") export GPU_MEMORY_UTILIZATION=0.65 ;;
            *) echo "Unknown model: ${MODEL_NAME}"; exit 1 ;;
        esac

        # LMCache CPU cache size per model (same as v0.5.1/v0.6.0 for comparability)
        case "${MODEL_NAME}" in
            "Qwen3-0.6B")    LMCACHE_SIZE=4.0  ;;
            "Qwen3-8B")      LMCACHE_SIZE=9.0  ;;
            "Qwen3-14B")     LMCACHE_SIZE=29.0 ;;
            "Qwen3-32B-AWQ") LMCACHE_SIZE=10.0 ;;
        esac

        for run in ${RUNS}; do
            export PARAMETERS="${run}"

            case "${run}" in
                "lmcache-local")
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.7.0"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
                    export VLLM_ENV_VARS="HF_HOME=/data/.hf LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_SIZE} PYTHONHASHSEED=123"
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE=""
                    export VLLM_PRE_CMD="pip3.12 install --quiet lmcache==0.4.4"
                    ;;
                "lmcache-valkey")
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.7.0"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
                    export VLLM_ENV_VARS="HF_HOME=/data/.hf LMCACHE_REMOTE_URL=valkey://valkey.${NAMESPACE}.svc.cluster.local:6379 LMCACHE_USE_EXPERIMENTAL=true PYTHONHASHSEED=123"
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE=""
                    export VLLM_PRE_CMD="pip3.12 install --quiet lmcache==0.4.4"
                    ;;
                *)
                    echo "Unknown configuration: ${run}"
                    exit 1
                    ;;
            esac

            for rate in "${RATES[@]}"; do
                export RATE="${rate}"
                export PREFIX_COUNT=$((2 * rate))
                export SAMPLE_REQUESTS=$((2 * rate * TURNS))

                RUN_ID="${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PARAMETERS}_replica${replicas}_rate${rate}"
                OUTPUT_DIR="./results/${RUN_ID}"

                if [ -f "${OUTPUT_DIR}/guidellm-results.json.zst" ]; then
                    echo "SKIPPING (already complete): ${RUN_ID}"
                    continue
                fi

                echo ""
                echo "=========================================="
                echo "Starting: ${MODEL_NAME} / ${run} / rate=${rate} / gmu=${GPU_MEMORY_UTILIZATION}"
                echo "=========================================="

                bash "$(dirname "$0")/run-benchmark.sh"
                sleep 10
            done
        done
    done
done

echo ""
echo "=========================================="
echo "v0.7.0 lmcache mempress benchmarks complete!"
echo "Results in: results/${HARDWARE}_${SOFTWARE}_*lmcache*"
echo "=========================================="
