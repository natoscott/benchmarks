#!/bin/bash
set -e

# Specialized benchmark run for testing multi-replica deployments
# This tests llm-d's request routing and distributed KV-block indexing
# with 2 replicas @ TP=1 vs baseline 1 replica @ TP=2

# Model to test - 14B is the sweet spot for CPU offload (2x L40S)
export MODEL="${MODEL:-Qwen/Qwen3-14B}"
export MODEL_NAME="${MODEL##*/}"

# Test configurations
# Focus on the configurations that showed best results for 14B:
# - no-offload: Baseline
# - lmcache-local-20kcpu: Best single-replica performer
# - llm-d-redis: Distributed KV-block indexing (most relevant for multi-replica)
# - llm-d-valkey: Alternative distributed backend
RUNS="${RUNS:-no-offload lmcache-local-20kcpu llm-d-redis llm-d-valkey}"

# Test three configurations for apples-to-apples comparison:
# - 1 replica @ TP=1 (single GPU, baseline compute)
# - 1 replica @ TP=2 (existing baseline, 2 GPUs)
# - 2 replicas @ TP=1 (distributed, 2 GPUs total)
REPLICA_CONFIGS="${REPLICA_CONFIGS:-1:1 1:2 2:1}"

# Export common variables
export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export RATE_LIST="${RATE:-1,50,100,150,300,400,500,650}"
export MAX_SECONDS="${MAX_SECONDS:-120}"
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="${SOFTWARE:-upstream-llm-d-0.4.0}"
export TURNS="${TURNS:-5}"
export INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-llm-d-model-server}"

# Convert comma-separated rates to array
IFS=',' read -ra RATES <<< "$RATE_LIST"

echo "=========================================="
echo "MULTI-REPLICA BENCHMARK"
echo "=========================================="
echo "Testing scaling and request routing:"
echo "  - 1 replica @ TP=1 (single GPU baseline)"
echo "  - 1 replica @ TP=2 (existing baseline, tensor parallelism)"
echo "  - 2 replicas @ TP=1 (distributed request routing)"
echo ""
echo "Model: ${MODEL}"
echo "Configurations: ${RUNS}"
echo "Concurrency levels: ${RATE_LIST}"
echo "=========================================="
echo ""

# Iterate through replica configurations (format: replicas:tp_size)
for config in ${REPLICA_CONFIGS}
do
    # Parse replicas:tp_size
    replicas="${config%:*}"
    tp_size="${config#*:}"

    export TENSOR_PARALLEL_SIZE="${tp_size}"
    export GPUS_PER_REPLICA="${tp_size}"
    export CURRENT_REPLICAS="${replicas}"

    # Create descriptive label
    REPLICA_LABEL="${replicas}xTP${tp_size}"

    echo ""
    echo "=========================================="
    echo "Configuration: ${replicas} replica(s) @ TP=${TENSOR_PARALLEL_SIZE}"
    echo "=========================================="

    # Scale the inference deployment
    kubectl --kubeconfig="${KUBECONFIG}" scale deployment/"${INFERENCE_DEPLOYMENT}" -n "${NAMESPACE}" --replicas="${replicas}"

    # Wait for scaling to complete
    echo "Waiting for deployment to scale to ${replicas} replicas..."
    kubectl --kubeconfig="${KUBECONFIG}" rollout status deployment/"${INFERENCE_DEPLOYMENT}" -n "${NAMESPACE}" --timeout=300s

    # Wait for all replicas to be fully ready
    echo "Waiting 30 seconds for all replicas to stabilize..."
    sleep 30

    # Iterate through different KV cache configurations
    for run in ${RUNS}
    do
        # Override PARAMETERS to indicate replica configuration
        export PARAMETERS="${run}-${REPLICA_LABEL}"

        # Configure vLLM arguments and environment variables for each case
        case "$run" in
            "no-offload")
                # Baseline - no CPU offload
                export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
                export VLLM_EXTRA_ARGS=""
                export VLLM_ENV_VARS=""
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="in-memory"
                export USE_LMCACHE_IMAGE=""
                ;;
            "lmcache-local-20kcpu")
                # LMCache local CPU offloading with 2x CPU memory (58 GB for 14B)
                export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
                export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
                export VLLM_ENV_VARS="HOME=/tmp LMCACHE_MAX_LOCAL_CPU_SIZE=58.0 PYTHONHASHSEED=123"
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="in-memory"
                export USE_LMCACHE_IMAGE="true"
                ;;
            "llm-d-redis")
                # llm-d EPP with Redis-backed distributed KV-block indexing
                export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
                export VLLM_EXTRA_ARGS=""
                export VLLM_ENV_VARS=""
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="redis"
                export USE_LMCACHE_IMAGE=""
                ;;
            "llm-d-valkey")
                # llm-d EPP with Valkey-backed distributed KV-block indexing
                export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
                export VLLM_EXTRA_ARGS=""
                export VLLM_ENV_VARS=""
                export VLLM_INSTALL_LMCACHE=""
                export EPP_BACKEND_CONFIG="valkey"
                export USE_LMCACHE_IMAGE=""
                ;;
            *)
                echo "Error: Unknown configuration $run"
                exit 1
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
            echo "Benchmark Iteration: MULTI-REPLICA TEST"
            echo "******************************************"
            echo "Hardware: $HARDWARE"
            echo "Software: $SOFTWARE"
            echo "Model: $MODEL_NAME"
            echo "Configuration: $PARAMETERS"
            echo "Replicas: $replicas (TP=$TENSOR_PARALLEL_SIZE)"
            echo "Concurrency (Rate): $RATE"
            echo "Prefix Count: $PREFIX_COUNT"
            echo "Max Requests: $SAMPLE_REQUESTS"
            echo "Turns: $TURNS"
            echo ""
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

echo ""
echo "=========================================="
echo "MULTI-REPLICA BENCHMARK COMPLETE"
echo "=========================================="
echo ""
echo "Results will be available in results/ directory"
echo ""
echo "Compare:"
echo "  - 1xTP1 (1 replica, TP=1): Single GPU baseline"
echo "  - 1xTP2 (1 replica, TP=2): Tensor parallelism baseline (existing data)"
echo "  - 2xTP1 (2 replicas, TP=1): Distributed request routing"
echo ""
echo "Key metrics to analyze:"
echo "  - Scaling: Does TP=2 outperform 2×TP=1 with same total GPU count?"
echo "  - Request routing: How well does llm-d distribute requests across replicas?"
echo "  - Latency: What is the overhead of cross-pod routing vs tensor parallelism?"
echo "  - Load balancing: Are requests distributed evenly across replicas?"
echo "  - Cache effectiveness: Does distributed KV-block indexing improve multi-replica cache hits?"
echo "=========================================="
