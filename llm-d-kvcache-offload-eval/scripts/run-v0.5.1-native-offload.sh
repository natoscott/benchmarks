#!/bin/bash
set -e

# This script runs focused benchmarks for llm-d v0.5.1 testing native CPU offload improvements
# Focus: Test if native offload improved from v0.4.0 to v0.5.1 (vLLM 0.11.2 -> 0.14.1)
# IMPORTANT: Uses EXACT same parameters as v0.4.0 run-all.sh for apples-to-apples comparison

# Configuration list - test native offload with 20k CPU blocks only
RUNS="${RUNS:-no-offload native-offload-20k}"
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ}"
REPLICAS="${REPLICAS:-1}"

# Export common variables that run-benchmark.sh will use - MATCH v0.4.0 run-all.sh
export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export RATE_LIST="${RATE:-1,50,100,150,300,400,500,650}"
export MAX_SECONDS=120  # v0.4.0 run-all.sh line 21
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="${SOFTWARE:-upstream-llm-d-0.5.1}"
export TURNS=5  # v0.4.0 run-all.sh line 24
export INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-llm-d-model-server}"

# GPU configuration
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
        export MODEL="$model"  # Full path: Qwen/Qwen3-14B
        export MODEL_NAME="${model##*/}"  # Short name for directory: Qwen3-14B

        # Iterate through different KV cache configurations
        for run in ${RUNS}
        do
            export PARAMETERS="$run"

            # Set CPU bytes based on model (matches v0.4.0 actual allocations)
            case "$MODEL_NAME" in
                "Qwen3-0.6B")
                    CPU_BYTES_10K=36421322670    # 33.92 GiB
                    CPU_BYTES_20K=72842645340    # 67.84 GiB (2x)
                    ;;
                "Qwen3-8B")
                    CPU_BYTES_10K=28808493137    # 26.83 GiB
                    CPU_BYTES_20K=57616986275    # 53.66 GiB (2x)
                    ;;
                "Qwen3-14B")
                    CPU_BYTES_10K=22097606737    # 20.58 GiB
                    CPU_BYTES_20K=44195213475    # 41.16 GiB (2x)
                    ;;
                "Qwen3-32B-AWQ")
                    CPU_BYTES_10K=27273042329    # 25.40 GiB
                    CPU_BYTES_20K=54546084659    # 50.80 GiB (2x)
                    ;;
                *)
                    echo "Unknown model: $MODEL_NAME"
                    exit 1
                    ;;
            esac

            # Configure vLLM arguments for each case
            case "$run" in
                "no-offload")
                    # Baseline: GPU-only, no CPU offloading
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.5.1"
                    export VLLM_EXTRA_ARGS=""
                    export VLLM_ENV_VARS=""
                    export EPP_BACKEND_CONFIG="in-memory"
                    ;;
                "native-offload-10k")
                    # vLLM native CPU offloading (matches v0.4.0 actual CPU allocation)
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.5.1"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${CPU_BYTES_10K}}}'"
                    export VLLM_ENV_VARS=""
                    export EPP_BACKEND_CONFIG="in-memory"
                    ;;
                "native-offload-20k")
                    # vLLM native CPU offloading (2x the 10K allocation)
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.5.1"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${CPU_BYTES_20K}}}'"
                    export VLLM_ENV_VARS=""
                    export EPP_BACKEND_CONFIG="in-memory"
                    ;;
                *)
                    echo "Unknown configuration: $run"
                    exit 1
                    ;;
            esac

            # Run benchmarks at all concurrency levels
            for rate in "${RATES[@]}"
            do
                export RATE="$rate"

                # Calculate dynamic parameters based on rate and turns (MATCH v0.4.0 run-all.sh lines 158-160)
                export PREFIX_COUNT=$((2 * rate))
                export SAMPLE_REQUESTS=$((2 * rate * TURNS))

                # Check if this benchmark is already complete
                RUN_ID="${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PARAMETERS}_replica${replicas}_rate${rate}"
                OUTPUT_DIR="./results/${RUN_ID}"

                if [ -f "${OUTPUT_DIR}/guidellm-results.json.zst" ]; then
                    echo ""
                    echo "=========================================="
                    echo "SKIPPING (already complete):"
                    echo "  Model: ${MODEL}"
                    echo "  Configuration: ${PARAMETERS}"
                    echo "  Replicas: ${replicas}"
                    echo "  Rate: ${rate}"
                    echo "=========================================="
                    continue
                fi

                echo ""
                echo "=========================================="
                echo "Starting benchmark run:"
                echo "  Model: ${MODEL}"
                echo "  Configuration: ${PARAMETERS}"
                echo "  Replicas: ${replicas}"
                echo "  Rate: ${rate}"
                echo "  Prefix Count: ${PREFIX_COUNT}"
                echo "  Sample Requests: ${SAMPLE_REQUESTS}"
                echo "  Turns: ${TURNS}"
                echo "  Max Seconds: ${MAX_SECONDS}"
                echo "  Software: ${SOFTWARE}"
                echo "=========================================="

                # Execute the benchmark
                bash "$(dirname "$0")/run-benchmark.sh"

                # Wait between runs to ensure clean state
                echo "Waiting 10 seconds before next run..."
                sleep 10
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All llm-d v0.5.0 benchmarks completed!"
echo "Results saved in: results/"
echo "=========================================="
