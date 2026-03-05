#!/bin/bash
set -e

# Extract KV-cache allocation information from vLLM startup logs
# For each model and configuration, deploy vLLM temporarily and capture:
# - GPU KV-cache memory available
# - GPU KV-cache size (tokens)
# - CPU KV-cache size (if offload enabled)
# - Maximum concurrency estimates

export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export INFERENCE_DEPLOYMENT="llm-d-model-server"

# Output file
OUTPUT_FILE="analysis/kvcache_allocations.csv"
mkdir -p analysis

# Models to test
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B-AWQ"
)

# Configurations to test
# Format: name|container_image|vllm_args|env_vars|description
CONFIGS=(
    "no-offload|ghcr.io/llm-d/llm-d-cuda:v0.4.0|||GPU-only (no offload)"
    "native-offload-10k|ghcr.io/llm-d/llm-d-cuda:v0.4.0|--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'||vLLM native offload (10K CPU blocks)"
    "native-offload-20k|ghcr.io/llm-d/llm-d-cuda:v0.4.0|--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":20000}}'||vLLM native offload (20K CPU blocks)"
    "lmcache-local-10k|docker.io/lmcache/vllm-openai:v0.3.7|--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching|HOME=/tmp LMCACHE_MAX_LOCAL_CPU_SIZE=29.0 PYTHONHASHSEED=123|LMCache local CPU (29GB for 0.6B/8B)"
    "lmcache-local-20k|docker.io/lmcache/vllm-openai:v0.3.7|--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching|HOME=/tmp LMCACHE_MAX_LOCAL_CPU_SIZE=58.0 PYTHONHASHSEED=123|LMCache local CPU (58GB for 14B)"
    "lmcache-local-39k|docker.io/lmcache/vllm-openai:v0.3.7|--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --enable-prefix-caching|HOME=/tmp LMCACHE_MAX_LOCAL_CPU_SIZE=78.0 PYTHONHASHSEED=123|LMCache local CPU (78GB for 32B-AWQ)"
)

# Tensor parallelism size
TP_SIZE=2

echo "=========================================="
echo "KV-CACHE ALLOCATION EXTRACTION"
echo "=========================================="
echo "This script will:"
echo "  1. Deploy each model configuration temporarily"
echo "  2. Extract KV-cache allocation info from logs"
echo "  3. Save results to: ${OUTPUT_FILE}"
echo ""
echo "Models: ${#MODELS[@]}"
echo "Configurations: ${#CONFIGS[@]}"
echo "Tensor Parallelism: ${TP_SIZE}"
echo "=========================================="
echo ""

# Create CSV header
cat > "${OUTPUT_FILE}" <<EOF
model,configuration,tp_size,gpu_kv_memory_gb,gpu_kv_tokens,cpu_kv_blocks,max_concurrency_estimate,model_memory_gb,notes
EOF

# Function to extract KV-cache info from logs
extract_kvcache_info() {
    local model="$1"
    local config_name="$2"
    local pod_name="$3"

    echo "  Waiting for model to load (60 seconds)..."
    sleep 60

    # Get pod logs
    local logs=$(kubectl --kubeconfig="${KUBECONFIG}" logs -n "${NAMESPACE}" "${pod_name}" --tail=500 2>/dev/null || echo "")

    if [ -z "$logs" ]; then
        echo "  ERROR: Failed to get logs from pod ${pod_name}"
        return 1
    fi

    # Extract GPU KV-cache memory (in GiB)
    local gpu_kv_memory=$(echo "$logs" | grep -o "Available KV cache memory: [0-9.]\+ GiB" | grep -o "[0-9.]\+" | head -1)

    # Extract GPU KV-cache size (tokens)
    local gpu_kv_tokens=$(echo "$logs" | grep -o "GPU KV cache size: [0-9,]\+ tokens" | sed 's/,//g' | grep -o "[0-9]\+" | head -1)

    # Extract CPU KV-cache blocks (if offload enabled)
    local cpu_kv_blocks=$(echo "$logs" | grep -o "CPU KV cache size: [0-9,]\+ blocks" | sed 's/,//g' | grep -o "[0-9]\+" | head -1)
    if [ -z "$cpu_kv_blocks" ]; then
        cpu_kv_blocks="0"
    fi

    # Extract max concurrency estimate (for 40K token requests)
    local max_concurrency=$(echo "$logs" | grep -o "Maximum concurrency for [0-9,]\+ tokens per request: [0-9.]\+x" | grep -o "[0-9.]\+x" | sed 's/x//' | head -1)

    # Extract model loading memory
    local model_memory=$(echo "$logs" | grep -o "Loading model weights took [0-9.]\+ GiB" | grep -o "[0-9.]\+" | head -1)

    # Check for any warnings or errors about memory
    local notes=""
    if echo "$logs" | grep -q "WARNING.*memory"; then
        notes="memory_warning"
    fi
    if echo "$logs" | grep -q "ERROR.*memory"; then
        notes="memory_error"
    fi

    # Output results
    echo "${model},${config_name},${TP_SIZE},${gpu_kv_memory},${gpu_kv_tokens},${cpu_kv_blocks},${max_concurrency},${model_memory},${notes}" >> "${OUTPUT_FILE}"

    echo "  GPU KV memory: ${gpu_kv_memory} GiB"
    echo "  GPU KV tokens: ${gpu_kv_tokens}"
    if [ "$cpu_kv_blocks" != "0" ]; then
        echo "  CPU KV blocks: ${cpu_kv_blocks}"
    fi
    echo "  Max concurrency (40K tokens): ${max_concurrency}x"
    echo "  Model memory: ${model_memory} GiB"
}

# Function to configure and deploy model
deploy_model() {
    local model="$1"
    local config_name="$2"
    local container_image="$3"
    local vllm_args="$4"
    local env_vars="$5"

    echo ""
    echo "=========================================="
    echo "Model: ${model}"
    echo "Configuration: ${config_name}"
    echo "=========================================="

    # Update deployment with model and configuration
    echo "  Configuring deployment..."

    # Set container image
    kubectl --kubeconfig="${KUBECONFIG}" set image deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" vllm="${container_image}" >/dev/null 2>&1

    # Set model environment variable
    kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" MODEL="${model}" >/dev/null 2>&1

    # Set tensor parallelism
    kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" TENSOR_PARALLEL_SIZE="${TP_SIZE}" >/dev/null 2>&1

    # Set vLLM args
    if [ -n "$vllm_args" ]; then
        kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
            -n "${NAMESPACE}" VLLM_EXTRA_ARGS="${vllm_args}" >/dev/null 2>&1
    else
        kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
            -n "${NAMESPACE}" VLLM_EXTRA_ARGS- >/dev/null 2>&1
    fi

    # Set additional environment variables
    if [ -n "$env_vars" ]; then
        # Parse space-separated env vars
        for env_var in $env_vars; do
            kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
                -n "${NAMESPACE}" "${env_var}" >/dev/null 2>&1
        done
    else
        # Clear LMCache-specific env vars if not needed
        kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
            -n "${NAMESPACE}" HOME- LMCACHE_MAX_LOCAL_CPU_SIZE- PYTHONHASHSEED- >/dev/null 2>&1 || true
    fi

    # Trigger rollout
    echo "  Triggering deployment rollout..."
    kubectl --kubeconfig="${KUBECONFIG}" rollout restart deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" >/dev/null 2>&1

    # Wait for rollout to complete
    echo "  Waiting for rollout to complete..."
    if ! kubectl --kubeconfig="${KUBECONFIG}" rollout status deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" --timeout=300s >/dev/null 2>&1; then
        echo "  ERROR: Rollout timed out"
        return 1
    fi

    # Get pod name
    local pod_name=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
        -l app=llm-d-model-server -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -z "$pod_name" ]; then
        echo "  ERROR: Failed to find pod"
        return 1
    fi

    echo "  Pod: ${pod_name}"

    # Extract KV-cache info
    extract_kvcache_info "${model}" "${config_name}" "${pod_name}"

    return 0
}

# Iterate through models and configurations
total_tests=$((${#MODELS[@]} * ${#CONFIGS[@]}))
current_test=0

for model in "${MODELS[@]}"; do
    for config_line in "${CONFIGS[@]}"; do
        current_test=$((current_test + 1))

        # Parse config line
        IFS='|' read -r config_name container_image vllm_args env_vars description <<< "$config_line"

        echo ""
        echo "=========================================="
        echo "Progress: ${current_test}/${total_tests}"
        echo "Model: ${model}"
        echo "Config: ${config_name} - ${description}"
        echo "=========================================="

        # Skip configurations that don't make sense for certain models
        # (e.g., don't use 78GB CPU memory for 0.6B model)
        if [[ "$model" == *"0.6B"* || "$model" == *"8B"* ]] && [[ "$config_name" == *"39k"* ]]; then
            echo "  SKIPPED: CPU memory config not applicable for this model"
            continue
        fi

        if [[ "$model" == *"14B"* ]] && [[ "$config_name" == *"10k"* && "$config_name" == *"lmcache"* ]]; then
            echo "  SKIPPED: Using 20k config for 14B model"
            continue
        fi

        if [[ "$model" == *"32B"* ]] && [[ "$config_name" == *"10k"* || "$config_name" == *"20k"* ]] && [[ "$config_name" == *"lmcache"* ]]; then
            echo "  SKIPPED: Using 39k config for 32B model"
            continue
        fi

        # Deploy and extract info
        if ! deploy_model "${model}" "${config_name}" "${container_image}" "${vllm_args}" "${env_vars}"; then
            echo "  WARNING: Failed to extract info for ${model} / ${config_name}"
            echo "${model},${config_name},${TP_SIZE},ERROR,ERROR,ERROR,ERROR,ERROR,deployment_failed" >> "${OUTPUT_FILE}"
        fi

        # Small delay between deployments
        sleep 5
    done
done

echo ""
echo "=========================================="
echo "KV-CACHE EXTRACTION COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: ${OUTPUT_FILE}"
echo ""
echo "Summary:"
wc -l < "${OUTPUT_FILE}"
echo "rows extracted"
echo ""
echo "Preview:"
head -10 "${OUTPUT_FILE}"
echo "=========================================="
