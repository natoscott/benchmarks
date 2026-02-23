#!/bin/bash
set -e

# Specialized benchmark run for testing increased CPU memory capacity
# This tests whether larger models benefit from increased CPU KV-cache blocks
# and whether the 14B model (which already shows improvement) benefits even more

# Models to test with increased CPU memory
# - 14B: Already shows +11-13% improvement with 10K blocks, test if 20K improves further
# - 32B-AWQ: Shows degradation with 10K blocks, test if 20K enables benefits
MODELS="${MODELS:-Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ}"

# Test configurations
# Focus on the configurations that showed interesting results:
# - native-offload: Baseline CPU offload
# - lmcache-local: Best performer for 14B
# - lmcache-valkey: Best distributed option for 14B
RUNS="${RUNS:-native-offload lmcache-local lmcache-valkey}"

# Replicas - single replica for initial testing
REPLICAS="${REPLICAS:-1}"

# Export common variables that run-benchmark.sh will use
export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export RATE_LIST="${RATE:-1,50,100,150,300,400,500,650}"
export MAX_SECONDS="${MAX_SECONDS:-120}"
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="${SOFTWARE:-upstream-llm-d-0.4.0}"
export TURNS="${TURNS:-5}"
export INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-llm-d-model-server}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPUS_PER_REPLICA="${TENSOR_PARALLEL_SIZE}"

# Convert comma-separated rates to array
IFS=',' read -ra RATES <<< "$RATE_LIST"

echo "=========================================="
echo "INCREASED CPU MEMORY CAPACITY BENCHMARK"
echo "=========================================="
echo "Testing increased CPU KV-cache blocks:"
echo "  - 14B: 20K blocks (~58 GB) vs baseline 10K blocks (~29 GB)"
echo "  - 32B-AWQ: 20K blocks (~78 GB) vs baseline 10K blocks (~39 GB)"
echo ""
echo "Models: ${MODELS}"
echo "Configurations: ${RUNS}"
echo "Concurrency levels: ${RATE_LIST}"
echo "=========================================="
echo ""

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
        export MODEL="$model"  # Full path: Qwen/Qwen3-14B
        export MODEL_NAME="${model##*/}"  # Short name for directory: Qwen3-14B

        # Iterate through different KV cache configurations
        for run in ${RUNS}
        do
            # Override PARAMETERS to indicate increased CPU memory
            export PARAMETERS="${run}-20kcpu"

            # Configure vLLM arguments and environment variables for each case
            # Key change: Use 20K CPU blocks instead of 10K
            case "$run" in
                "native-offload")
                    # vLLM native CPU offloading with 20,000 blocks (2x baseline)
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.4.0"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":20000}}'"
                    export VLLM_ENV_VARS=""
                    export VLLM_INSTALL_LMCACHE=""
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE=""
                    ;;
                "lmcache-local")
                    # LMCache local CPU offloading with 2x CPU memory
                    # Memory requirements:
                    # - Qwen3-14B: 58 GB (2x baseline 29 GB)
                    # - Qwen3-32B-AWQ: 78 GB (2x baseline 39 GB)
                    case "$MODEL_NAME" in
                        "Qwen3-14B")
                            LMCACHE_SIZE=58.0
                            ;;
                        "Qwen3-32B-AWQ")
                            LMCACHE_SIZE=78.0
                            ;;
                        *)
                            echo "Error: Unexpected model $MODEL_NAME"
                            exit 1
                            ;;
                    esac
                    export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
                    export VLLM_ENV_VARS="HOME=/tmp LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_SIZE} PYTHONHASHSEED=123"
                    export VLLM_INSTALL_LMCACHE=""
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE="true"
                    ;;
                "lmcache-valkey")
                    # LMCache with Valkey remote backend
                    # Note: Valkey has its own memory limit, but this increases vLLM's local cache budget
                    export CONTAINER_IMAGE="docker.io/lmcache/vllm-openai:v0.3.7"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching"
                    export VLLM_ENV_VARS="HOME=/tmp LMCACHE_REMOTE_URL=valkey://valkey.${NAMESPACE}.svc.cluster.local:6379 LMCACHE_USE_EXPERIMENTAL=true PYTHONHASHSEED=123"
                    export VLLM_INSTALL_LMCACHE=""
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE="true"
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
                echo "Benchmark Iteration: INCREASED CPU MEMORY TEST"
                echo "******************************************"
                echo "Hardware: $HARDWARE"
                echo "Software: $SOFTWARE"
                echo "Model: $MODEL_NAME"
                echo "Configuration: $PARAMETERS"
                echo "Replicas: $replicas"
                echo "Concurrency (Rate): $RATE"
                echo "Prefix Count: $PREFIX_COUNT"
                echo "Max Requests: $SAMPLE_REQUESTS"
                echo "Turns: $TURNS"
                echo ""
                echo "CPU Memory Configuration:"
                case "$run" in
                    "native-offload")
                        echo "  num_cpu_blocks: 20000 (baseline: 10000)"
                        ;;
                    "lmcache-local")
                        echo "  LMCACHE_MAX_LOCAL_CPU_SIZE: ${LMCACHE_SIZE} GB"
                        case "$MODEL_NAME" in
                            "Qwen3-14B")
                                echo "  (baseline: 29 GB, increase: 2x)"
                                ;;
                            "Qwen3-32B-AWQ")
                                echo "  (baseline: 39 GB, increase: 2x)"
                                ;;
                        esac
                        ;;
                    "lmcache-valkey")
                        echo "  Distributed cache (Valkey backend)"
                        ;;
                esac
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
done

echo ""
echo "=========================================="
echo "INCREASED CPU MEMORY BENCHMARK COMPLETE"
echo "=========================================="
echo ""
echo "Results will be available in results/ directory with '-20kcpu' suffix"
echo ""
echo "Expected outcomes:"
echo "  - 14B model: May show further improvement beyond current +11-13%"
echo "  - 32B-AWQ model: May shift from degradation to improvement if memory was the constraint"
echo ""
echo "Compare results against baseline runs to assess impact of increased CPU memory capacity."
echo "=========================================="
