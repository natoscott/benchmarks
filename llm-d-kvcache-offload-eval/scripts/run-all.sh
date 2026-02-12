#!/bin/bash
set -e

# This wrapper script iterates through test configurations and models
# Usage:
#   ./run-all.sh                          # Run all tests
#   RUNS="no-offload native-offload" ./run-all.sh   # Run subset
#   MODELS="Qwen/Qwen3-0.6B" ./run-all.sh           # Single model

# Configuration list - can be overridden via environment
RUNS="${RUNS:-no-offload native-offload lmcache-local lmcache-redis lmcache-valkey llm-d-redis llm-d-valkey}"
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B Qwen/Qwen3-14B}"

# Export common variables that run-benchmark.sh will use
# These can be overridden from environment
export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export RATE="${RATE:-1,2,4,8}"
export MAX_SECONDS="${MAX_SECONDS:-30}"
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="${SOFTWARE:-upstream-llm-d-0.4.0}"

# Iterate through models
for model in ${MODELS}
do
    # Set model variables
    export MODEL="$model"  # Full path: Qwen/Qwen3-0.6B
    export MODEL_NAME="${model##*/}"  # Short name for directory: Qwen3-0.6B

    # Iterate through different KV cache configurations
    for run in ${RUNS}
    do
        export PARAMETERS="$run"

        # Configure vLLM arguments and environment variables for each case
        case "$run" in
            "no-offload")
                # Baseline: GPU-only, no CPU offloading
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{}'"
                export VLLM_ENV_VARS=""
                export EPP_BACKEND_CONFIG="in-memory"
                ;;
            "native-offload")
                # vLLM native CPU offloading with OffloadingConnector
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
                export VLLM_ENV_VARS=""
                export EPP_BACKEND_CONFIG="in-memory"
                ;;
            "lmcache-local")
                # LMCache local CPU offloading
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnector\",\"kv_role\":\"kv_both\"}'"
                export VLLM_ENV_VARS=""
                export EPP_BACKEND_CONFIG="in-memory"
                ;;
            "lmcache-redis")
                # LMCache with Redis remote backend
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnector\",\"kv_role\":\"kv_both\"}'"
                export VLLM_ENV_VARS="LMCACHE_REMOTE_URL=redis://redis.${NAMESPACE}.svc.cluster.local:6379 LMCACHE_USE_EXPERIMENTAL=true"
                export EPP_BACKEND_CONFIG="in-memory"
                ;;
            "lmcache-valkey")
                # LMCache with Valkey remote backend
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnector\",\"kv_role\":\"kv_both\"}'"
                export VLLM_ENV_VARS="LMCACHE_REMOTE_URL=valkey://valkey.${NAMESPACE}.svc.cluster.local:6379 LMCACHE_USE_EXPERIMENTAL=true"
                export EPP_BACKEND_CONFIG="in-memory"
                ;;
            "llm-d-redis")
                # llm-d EPP with Redis index backend for distributed KV-cache-aware routing
                export VLLM_EXTRA_ARGS=""
                export VLLM_ENV_VARS=""
                export EPP_BACKEND_CONFIG="redis"
                ;;
            "llm-d-valkey")
                # llm-d EPP with Valkey index backend for distributed KV-cache-aware routing
                export VLLM_EXTRA_ARGS=""
                export VLLM_ENV_VARS=""
                export EPP_BACKEND_CONFIG="valkey"
                ;;
        esac

        echo ""
        echo "******************************************"
        echo "Benchmark Iteration: $HARDWARE $SOFTWARE $MODEL_NAME $PARAMETERS"
        echo "Model: $MODEL"
        echo "vLLM Args: $VLLM_EXTRA_ARGS"
        if [ -n "$VLLM_ENV_VARS" ]; then
            echo "vLLM Env:  $VLLM_ENV_VARS"
        fi
        echo "EPP Backend: $EPP_BACKEND_CONFIG"
        echo "Rate: $RATE"
        echo "Max Seconds: $MAX_SECONDS"
        echo "******************************************"
        echo ""

        # Run benchmark (expects to be called from repo root)
        scripts/run-benchmark.sh
    done
done
