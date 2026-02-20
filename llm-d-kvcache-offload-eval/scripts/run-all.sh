#!/bin/bash
set -e

# This wrapper script iterates through test configurations and models
# Usage:
#   ./run-all.sh                          # Run all tests
#   RUNS="no-offload native-offload" ./run-all.sh   # Run subset
#   MODELS="Qwen/Qwen3-0.6B" ./run-all.sh           # Single model
#   REPLICAS="1 2" ./run-all.sh                     # Test specific replica counts

# Configuration list - can be overridden via environment
RUNS="${RUNS:-no-offload native-offload lmcache-local lmcache-redis lmcache-valkey llm-d-redis llm-d-valkey}"
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ}"
REPLICAS="${REPLICAS:-1}"

# Export common variables that run-benchmark.sh will use
# These can be overridden from environment
export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export RATE_LIST="${RATE:-1,50,100,150,300,400,500,650}"
export MAX_SECONDS="${MAX_SECONDS:-120}"
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="${SOFTWARE:-upstream-llm-d-0.4.0}"
export TURNS="${TURNS:-5}"
export INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-llm-d-model-server}"

# GPU configuration - can be overridden for different hardware setups
# TENSOR_PARALLEL_SIZE: Number of GPUs per replica for tensor parallelism
# GPUS_PER_REPLICA: Same as TENSOR_PARALLEL_SIZE, used for resource allocation
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPUS_PER_REPLICA="${TENSOR_PARALLEL_SIZE}"

# Convert comma-separated rates to array
IFS=',' read -ra RATES <<< "$RATE_LIST"

# Iterate through replica counts
for replicas in ${REPLICAS}
do
    # Export current replica count for run-benchmark.sh to use
    export CURRENT_REPLICAS="${replicas}"

    echo ""
    echo "=========================================="
    echo "Setting replica count to: ${replicas}"
    echo "=========================================="

    # Scale the inference deployment
    kubectl --kubeconfig="${KUBECONFIG}" scale deployment/"${INFERENCE_DEPLOYMENT}" -n "${NAMESPACE}" --replicas="${replicas}"

    # Wait for scaling to complete
    echo "Waiting for deployment to scale to ${replicas} replicas..."
    kubectl --kubeconfig="${KUBECONFIG}" rollout status deployment/"${INFERENCE_DEPLOYMENT}" -n "${NAMESPACE}" --timeout=300s

    # Wait for all replicas to be fully ready
    echo "Waiting 30 seconds for all replicas to stabilize..."
    sleep 30

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
                export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
                export VLLM_EXTRA_ARGS=""
                export VLLM_ENV_VARS=""
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="in-memory"
                export USE_LMCACHE_IMAGE=""
                ;;
            "native-offload")
                # vLLM native CPU offloading with OffloadingConnector
                export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
                export VLLM_ENV_VARS=""
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="in-memory"
                export USE_LMCACHE_IMAGE=""
                ;;
            "lmcache-local")
                # LMCache local CPU offloading (using official lmcache/vllm-openai image)
                # LMCACHE_MAX_LOCAL_CPU_SIZE controls CPU memory for KV cache (in GB)
                # Values adjusted per model to match ~10000 blocks similar to native-offload:
                # Qwen3-0.6B: 4 GB, Qwen3-8B: 9 GB, Qwen3-14B: 29 GB
                case "$MODEL_NAME" in
                    "Qwen3-0.6B")
                        LMCACHE_SIZE=4.0
                        ;;
                    "Qwen3-8B")
                        LMCACHE_SIZE=9.0
                        ;;
                    "Qwen3-14B")
                        LMCACHE_SIZE=29.0
                        ;;
                    *)
                        echo "Warning: Unknown model $MODEL_NAME, defaulting to 10 GB"
                        LMCACHE_SIZE=10.0
                        ;;
                esac
                export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
                export VLLM_ENV_VARS="HOME=/tmp LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_SIZE} PYTHONHASHSEED=123"
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="in-memory"
                export USE_LMCACHE_IMAGE="true"
                ;;
            "lmcache-redis")
                # LMCache with Redis remote backend (using official lmcache/vllm-openai image)
                export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
                export VLLM_ENV_VARS="HOME=/tmp LMCACHE_REMOTE_URL=redis://redis.${NAMESPACE}.svc.cluster.local:6379 LMCACHE_USE_EXPERIMENTAL=true PYTHONHASHSEED=123"
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="in-memory"
                export USE_LMCACHE_IMAGE="true"
                ;;
            "lmcache-valkey")
                # LMCache with Valkey remote backend (using official lmcache/vllm-openai image)
                export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
                export VLLM_ENV_VARS="HOME=/tmp LMCACHE_REMOTE_URL=valkey://valkey.${NAMESPACE}.svc.cluster.local:6379 LMCACHE_USE_EXPERIMENTAL=true PYTHONHASHSEED=123"
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="in-memory"
                export USE_LMCACHE_IMAGE="true"
                ;;
            "llm-d-redis")
                # llm-d EPP with Redis index backend for distributed KV-cache-aware routing
                export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
                export VLLM_EXTRA_ARGS=""
                export VLLM_ENV_VARS=""
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="redis"
                export USE_LMCACHE_IMAGE=""
                ;;
            "llm-d-valkey")
                # llm-d EPP with Valkey index backend for distributed KV-cache-aware routing
                export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
                export VLLM_EXTRA_ARGS=""
                export VLLM_ENV_VARS=""
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="valkey"
                export USE_LMCACHE_IMAGE=""
                ;;
        esac

            # Iterate through individual rate values
            for rate in "${RATES[@]}"
            do
                # Calculate dynamic parameters based on rate and turns
                export PREFIX_COUNT=$((2 * rate))
                export SAMPLE_REQUESTS=$((2 * rate * TURNS))

                # Generate experiment name with metadata
                export EXPERIMENT_NAME="${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PARAMETERS}_replica${replicas}_rate${rate}"

                # Set the current rate (single value, not comma-separated)
                export RATE="${rate}"

                echo ""
                echo "******************************************"
                echo "Benchmark Iteration: $HARDWARE $SOFTWARE $MODEL_NAME $PARAMETERS"
                echo "Model: $MODEL"
                echo "Replicas: $replicas"
                echo "Concurrency (Rate): $RATE"
                echo "Prefix Count: $PREFIX_COUNT"
                echo "Max Requests: $SAMPLE_REQUESTS"
                echo "Turns: $TURNS"
                echo "vLLM Args: $VLLM_EXTRA_ARGS"
                if [ -n "$VLLM_ENV_VARS" ]; then
                    echo "vLLM Env:  $VLLM_ENV_VARS"
                fi
                echo "EPP Backend: $EPP_BACKEND_CONFIG"
                echo "Max Seconds: $MAX_SECONDS"
                echo "Experiment: $EXPERIMENT_NAME"
                echo "******************************************"
                echo ""

                # Run benchmark (expects to be called from repo root)
                scripts/run-benchmark.sh
            done
        done
    done
done
