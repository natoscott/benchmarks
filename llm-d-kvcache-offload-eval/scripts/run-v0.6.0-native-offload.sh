#!/bin/bash
set -e

# Benchmarks for llm-d v0.6.0 (vLLM 0.17.1) — no-offload and native CPU offload.
# Mirrors run-v0.5.1-native-offload.sh for apples-to-apples version comparison.

RUNS="${RUNS:-no-offload native-offload-20k}"
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ}"
REPLICAS="${REPLICAS:-1}"

export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export RATE_LIST="${RATE:-1,50,100,150,300,400,500,650}"
export MAX_SECONDS=120
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="${SOFTWARE:-upstream-llm-d-0.6.0}"
export TURNS=5
export INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-llm-d-model-server}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPUS_PER_REPLICA="${TENSOR_PARALLEL_SIZE}"

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

        # cpu_bytes_to_use for native-offload-20k (same values as v0.5.1 — hardware unchanged)
        case "${MODEL_NAME}" in
            "Qwen3-0.6B")    CPU_BYTES_20K=72842645340 ;;
            "Qwen3-8B")      CPU_BYTES_20K=57616986275 ;;
            "Qwen3-14B")     CPU_BYTES_20K=44195213475 ;;
            "Qwen3-32B-AWQ") CPU_BYTES_20K=54546084659 ;;
            *) echo "Unknown model: ${MODEL_NAME}"; exit 1 ;;
        esac

        for run in ${RUNS}; do
            export PARAMETERS="${run}"

            case "${run}" in
                "no-offload")
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.6.0"
                    export VLLM_EXTRA_ARGS=""
                    export VLLM_ENV_VARS=""
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE=""
                    export VLLM_PRE_CMD=""
                    ;;
                "native-offload-20k")
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.6.0"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${CPU_BYTES_20K}}}'"
                    export VLLM_ENV_VARS=""
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE=""
                    export VLLM_PRE_CMD=""
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
                echo "Starting: ${MODEL_NAME} / ${run} / rate=${rate}"
                echo "=========================================="

                bash "$(dirname "$0")/run-benchmark.sh"
                sleep 10
            done
        done
    done
done

echo ""
echo "=========================================="
echo "v0.6.0 native offload benchmarks complete!"
echo "Results in: results/${HARDWARE}_${SOFTWARE}_*"
echo "=========================================="
