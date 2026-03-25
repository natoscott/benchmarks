#!/bin/bash
set -e

# Re-run the 35 corrupt v0.4.0 benchmark files (JSON-truncated by the old
# binary kubectl exec transfer method, now fixed with base64 transfer).
#
# 3 llm-d-redis cases are skipped (redis ≡ valkey per prior evaluation;
# redis service not deployed on this cluster).
#
# Deletes corrupt guidellm-results.json.zst files to trigger re-run.
# Uses original SOFTWARE naming so results overwrite the corrupt entries.

export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="upstream-llm-d-0.4.0"
export MAX_SECONDS=120
export TURNS=5
export RATE_TYPE=concurrent
export TENSOR_PARALLEL_SIZE=2
export GPUS_PER_REPLICA=2
export INFERENCE_DEPLOYMENT="llm-d-model-server"
export GPU_MEMORY_UTILIZATION=0.9   # original default

# Each entry: "MODEL_NAME CONFIG RATE"
CORRUPT_RUNS=(
    # Qwen3-0.6B
    "Qwen3-0.6B llm-d-valkey 500"
    "Qwen3-0.6B llm-d-valkey 650"
    "Qwen3-0.6B lmcache-local 500"
    "Qwen3-0.6B lmcache-valkey 500"
    "Qwen3-0.6B native-offload 500"
    "Qwen3-0.6B no-offload 500"
    "Qwen3-0.6B no-offload 650"
    # Qwen3-8B
    "Qwen3-8B lmcache-valkey 500"
    "Qwen3-8B no-offload 500"
    # Qwen3-14B
    "Qwen3-14B llm-d-valkey 500"
    "Qwen3-14B lmcache-local-20kcpu 300"
    "Qwen3-14B lmcache-local-20kcpu 400"
    "Qwen3-14B lmcache-local-20kcpu 500"
    "Qwen3-14B lmcache-valkey-20kcpu 300"
    "Qwen3-14B lmcache-valkey-20kcpu 400"
    "Qwen3-14B lmcache-valkey-20kcpu 650"
    "Qwen3-14B lmcache-valkey 500"
    "Qwen3-14B native-offload-20kcpu 300"
    "Qwen3-14B native-offload-20kcpu 400"
    "Qwen3-14B native-offload-20kcpu 500"
    "Qwen3-14B native-offload-20kcpu 650"
    "Qwen3-14B no-offload 500"
    # Qwen3-32B-AWQ
    "Qwen3-32B-AWQ llm-d-valkey 500"
    "Qwen3-32B-AWQ lmcache-local-20kcpu 150"
    "Qwen3-32B-AWQ lmcache-local-20kcpu 300"
    "Qwen3-32B-AWQ lmcache-local-20kcpu 400"
    "Qwen3-32B-AWQ lmcache-local 500"
    "Qwen3-32B-AWQ lmcache-valkey-20kcpu 300"
    "Qwen3-32B-AWQ lmcache-valkey 500"
    "Qwen3-32B-AWQ native-offload-20kcpu 150"
    "Qwen3-32B-AWQ native-offload-20kcpu 300"
    "Qwen3-32B-AWQ native-offload-20kcpu 400"
    "Qwen3-32B-AWQ native-offload-20kcpu 650"
    "Qwen3-32B-AWQ native-offload 500"
    "Qwen3-32B-AWQ no-offload 500"
)

echo "=========================================="
echo "v0.4.0 Corrupt File Repair (35 runs)"
echo "=========================================="
echo "Skipped: llm-d-redis cases (redis ≡ valkey; not deployed)"
echo ""

TOTAL=${#CORRUPT_RUNS[@]}

for i in "${!CORRUPT_RUNS[@]}"; do
    read -r MODEL_NAME CONFIG RATE <<< "${CORRUPT_RUNS[$i]}"
    RUN_NUM=$((i + 1))

    export MODEL_NAME
    export RATE
    export PARAMETERS="${CONFIG}"
    export CURRENT_REPLICAS=1
    export PREFIX_COUNT=$((2 * RATE))
    export SAMPLE_REQUESTS=$((2 * RATE * TURNS))

    # Set MODEL full path
    case "${MODEL_NAME}" in
        "Qwen3-0.6B")   export MODEL="Qwen/Qwen3-0.6B" ;;
        "Qwen3-8B")     export MODEL="Qwen/Qwen3-8B" ;;
        "Qwen3-14B")    export MODEL="Qwen/Qwen3-14B" ;;
        "Qwen3-32B-AWQ") export MODEL="Qwen/Qwen3-32B-AWQ" ;;
    esac

    # Set config-specific environment variables (matching original run-all.sh)
    case "${CONFIG}" in
        "no-offload")
            export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
            export VLLM_EXTRA_ARGS=""
            export VLLM_ENV_VARS=""
            export EPP_BACKEND_CONFIG="in-memory"
            export USE_LMCACHE_IMAGE=""
            export VLLM_PRE_CMD=""
            ;;
        "native-offload")
            export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
            export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
            export VLLM_ENV_VARS=""
            export EPP_BACKEND_CONFIG="in-memory"
            export USE_LMCACHE_IMAGE=""
            export VLLM_PRE_CMD=""
            ;;
        "native-offload-20kcpu")
            export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
            export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":20000}}'"
            export VLLM_ENV_VARS=""
            export EPP_BACKEND_CONFIG="in-memory"
            export USE_LMCACHE_IMAGE=""
            export VLLM_PRE_CMD=""
            ;;
        "lmcache-local")
            case "${MODEL_NAME}" in
                "Qwen3-0.6B")   LMCACHE_SIZE=4.0 ;;
                "Qwen3-8B")     LMCACHE_SIZE=9.0 ;;
                "Qwen3-14B")    LMCACHE_SIZE=29.0 ;;
                "Qwen3-32B-AWQ") LMCACHE_SIZE=10.0 ;;
            esac
            export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
            export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
            export VLLM_ENV_VARS="HOME=/tmp HF_HOME=/data/.hf LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_SIZE} PYTHONHASHSEED=123"
            export EPP_BACKEND_CONFIG="in-memory"
            export USE_LMCACHE_IMAGE="true"
            export VLLM_PRE_CMD=""
            ;;
        "lmcache-local-20kcpu")
            case "${MODEL_NAME}" in
                "Qwen3-0.6B")   LMCACHE_SIZE=8.0 ;;
                "Qwen3-8B")     LMCACHE_SIZE=18.0 ;;
                "Qwen3-14B")    LMCACHE_SIZE=41.0 ;;
                "Qwen3-32B-AWQ") LMCACHE_SIZE=20.0 ;;
            esac
            export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
            export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
            export VLLM_ENV_VARS="HOME=/tmp HF_HOME=/data/.hf LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_SIZE} PYTHONHASHSEED=123"
            export EPP_BACKEND_CONFIG="in-memory"
            export USE_LMCACHE_IMAGE="true"
            export VLLM_PRE_CMD=""
            ;;
        "lmcache-valkey")
            export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
            export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
            export VLLM_ENV_VARS="HOME=/tmp HF_HOME=/data/.hf LMCACHE_REMOTE_URL=valkey://valkey.${NAMESPACE}.svc.cluster.local:6379 LMCACHE_USE_EXPERIMENTAL=true PYTHONHASHSEED=123"
            export EPP_BACKEND_CONFIG="in-memory"
            export USE_LMCACHE_IMAGE="true"
            export VLLM_PRE_CMD=""
            ;;
        "lmcache-valkey-20kcpu")
            export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
            export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
            export VLLM_ENV_VARS="HOME=/tmp HF_HOME=/data/.hf LMCACHE_REMOTE_URL=valkey://valkey.${NAMESPACE}.svc.cluster.local:6379 LMCACHE_USE_EXPERIMENTAL=true PYTHONHASHSEED=123"
            export EPP_BACKEND_CONFIG="in-memory"
            export USE_LMCACHE_IMAGE="true"
            export VLLM_PRE_CMD=""
            ;;
        "llm-d-valkey")
            export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
            export VLLM_EXTRA_ARGS=""
            export VLLM_ENV_VARS=""
            export EPP_BACKEND_CONFIG="valkey"
            export USE_LMCACHE_IMAGE=""
            export VLLM_PRE_CMD=""
            ;;
        *)
            echo "ERROR: Unknown config ${CONFIG}"
            exit 1
            ;;
    esac

    OUTPUT_DIR="./results/${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PARAMETERS}_replica1_rate${RATE}"

    echo ""
    echo "[${RUN_NUM}/${TOTAL}] ${MODEL_NAME} / ${CONFIG} / rate=${RATE}"

    # Delete corrupt file to force re-run
    if [ -f "${OUTPUT_DIR}/guidellm-results.json.zst" ]; then
        echo "  Removing corrupt guidellm-results.json.zst..."
        rm -f "${OUTPUT_DIR}/guidellm-results.json.zst"
    fi

    bash "$(dirname "$0")/run-benchmark.sh"
    sleep 5
done

echo ""
echo "=========================================="
echo "Corrupt file repair complete!"
echo "=========================================="
