#!/bin/bash
set -e

# Extract KV-cache allocation information from llm-d native configurations only
# This avoids the complexity of switching between llm-d and LMCache images

export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export INFERENCE_DEPLOYMENT="llm-d-model-server"

# Output file
OUTPUT_FILE="analysis/kvcache_allocations.csv"
mkdir -p analysis

# Create CSV header
cat > "${OUTPUT_FILE}" <<EOF
model,configuration,tp_size,gpu_kv_memory_gb,gpu_kv_tokens,cpu_kv_blocks,model_memory_gb,graph_capture_memory_gb,max_concurrency_40k,notes
EOF

echo "=========================================="
echo "KV-CACHE ALLOCATION EXTRACTION"
echo "=========================================="
echo "Extracting from llm-d native configurations only"
echo "Output: ${OUTPUT_FILE}"
echo "=========================================="
echo ""

# Function to deploy and extract KV-cache info
extract_kvcache_info() {
    local model="$1"
    local config_name="$2"
    local vllm_extra_args="$3"
    local tp_size="${4:-2}"

    echo ""
    echo "=========================================="
    echo "Model: ${model}"
    echo "Config: ${config_name}"
    echo "=========================================="

    # Always use llm-d image
    kubectl --kubeconfig="${KUBECONFIG}" set image deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" vllm=ghcr.io/llm-d/llm-d-cuda:v0.4.0 >/dev/null 2>&1

    # Set model
    kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" MODEL="${model}" >/dev/null 2>&1

    # Set TP size
    kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" TENSOR_PARALLEL_SIZE="${tp_size}" >/dev/null 2>&1

    # Set vLLM args
    if [ -n "$vllm_extra_args" ]; then
        kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
            -n "${NAMESPACE}" VLLM_EXTRA_ARGS="${vllm_extra_args}" >/dev/null 2>&1
    else
        kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
            -n "${NAMESPACE}" VLLM_EXTRA_ARGS- >/dev/null 2>&1
    fi

    # Clear LMCache-specific vars
    kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" HOME- LMCACHE_MAX_LOCAL_CPU_SIZE- PYTHONHASHSEED- >/dev/null 2>&1 || true

    # Restart deployment
    echo "  Deploying..."
    kubectl --kubeconfig="${KUBECONFIG}" rollout restart deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" >/dev/null 2>&1

    # Wait for rollout
    echo "  Waiting for pod to be ready..."
    if ! kubectl --kubeconfig="${KUBECONFIG}" rollout status deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" --timeout=300s >/dev/null 2>&1; then
        echo "  ERROR: Deployment rollout failed"
        echo "${model},${config_name},${tp_size},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,deployment_failed" >> "${OUTPUT_FILE}"
        return 1
    fi

    # Get pod name
    local pod_name=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
        -l llm-d.ai/inference-serving=true -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -z "$pod_name" ]; then
        echo "  ERROR: No pod found"
        echo "${model},${config_name},${tp_size},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,no_pod" >> "${OUTPUT_FILE}"
        return 1
    fi

    echo "  Pod: ${pod_name}"
    echo "  Waiting for vLLM to initialize (90 seconds)..."
    sleep 90

    # Get logs
    echo "  Extracting KV-cache info from logs..."
    local logs=$(kubectl --kubeconfig="${KUBECONFIG}" logs "${pod_name}" -n "${NAMESPACE}" --tail=1000 2>/dev/null || echo "")

    if [ -z "$logs" ]; then
        echo "  ERROR: Failed to get logs"
        echo "${model},${config_name},${tp_size},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,no_logs" >> "${OUTPUT_FILE}"
        return 1
    fi

    # Extract GPU KV-cache memory (GiB)
    local gpu_kv_memory=$(echo "$logs" | grep -o "Available KV cache memory: [0-9.]\+ GiB" | grep -o "[0-9.]\+" | head -1)

    # Extract GPU KV-cache tokens
    local gpu_kv_tokens=$(echo "$logs" | grep -o "GPU KV cache size: [0-9,]\+ tokens" | sed 's/,//g' | grep -o "[0-9]\+" | head -1)

    # Extract CPU KV-cache blocks
    local cpu_kv_blocks=$(echo "$logs" | grep -o "CPU KV cache size: [0-9,]\+ blocks" | sed 's/,//g' | grep -o "[0-9]\+" | head -1)
    if [ -z "$cpu_kv_blocks" ]; then
        cpu_kv_blocks="0"
    fi

    # Extract model loading memory
    local model_memory=$(echo "$logs" | grep -o "Loading model weights took [0-9.]\+ GiB" | grep -o "[0-9.]\+" | head -1)

    # Extract graph capture memory
    local graph_memory=$(echo "$logs" | grep -o "Graph capturing took [0-9.]\+ GiB" | grep -o "[0-9.]\+" | head -1)
    if [ -z "$graph_memory" ]; then
        graph_memory="0"
    fi

    # Extract max concurrency for 40K tokens
    local max_concurrency=$(echo "$logs" | grep -o "Maximum concurrency for [0-9,]\+ tokens per request: [0-9.]\+x" | grep -o "[0-9.]\+x" | sed 's/x//' | head -1)

    # Check for warnings
    local notes=""
    if echo "$logs" | grep -q "WARNING.*memory"; then
        notes="memory_warning"
    fi
    if echo "$logs" | grep -q "ERROR"; then
        notes="${notes:+$notes,}errors_in_log"
    fi

    # Output to CSV
    echo "${model},${config_name},${tp_size},${gpu_kv_memory},${gpu_kv_tokens},${cpu_kv_blocks},${model_memory},${graph_memory},${max_concurrency},${notes}" >> "${OUTPUT_FILE}"

    echo "  ✓ GPU KV memory: ${gpu_kv_memory} GiB"
    echo "  ✓ GPU KV tokens: ${gpu_kv_tokens}"
    if [ "$cpu_kv_blocks" != "0" ]; then
        echo "  ✓ CPU KV blocks: ${cpu_kv_blocks}"
    fi
    echo "  ✓ Model memory: ${model_memory} GiB"
    echo "  ✓ Max concurrency (40K tokens): ${max_concurrency}x"

    return 0
}

# Configurations to test - llm-d native only (skip LMCache for now)

# Model: Qwen3-0.6B
extract_kvcache_info "Qwen/Qwen3-0.6B" "no-offload" "" 2
extract_kvcache_info "Qwen/Qwen3-0.6B" "native-offload" "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'" 2

# Model: Qwen3-8B
extract_kvcache_info "Qwen/Qwen3-8B" "no-offload" "" 2
extract_kvcache_info "Qwen/Qwen3-8B" "native-offload" "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'" 2

# Model: Qwen3-14B
extract_kvcache_info "Qwen/Qwen3-14B" "no-offload" "" 2
extract_kvcache_info "Qwen/Qwen3-14B" "native-offload" "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'" 2
extract_kvcache_info "Qwen/Qwen3-14B" "native-offload-20kcpu" "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":20000}}'" 2

# Model: Qwen3-32B-AWQ
extract_kvcache_info "Qwen/Qwen3-32B-AWQ" "no-offload" "" 2
extract_kvcache_info "Qwen/Qwen3-32B-AWQ" "native-offload" "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'" 2
extract_kvcache_info "Qwen/Qwen3-32B-AWQ" "native-offload-20kcpu" "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":20000}}'" 2

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
head -15 "${OUTPUT_FILE}"
echo "=========================================="
echo ""
echo "NOTE: LMCache configurations skipped due to deployment complexity."
echo "LMCache KV-cache data can be extracted from existing benchmark logs."
echo "=========================================="
