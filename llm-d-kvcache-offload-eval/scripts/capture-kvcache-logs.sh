#!/bin/bash
# Capture vLLM startup logs for all model/config combinations to extract KV-cache allocation data

set -e

KUBECONFIG="${KUBECONFIG:-kubeconfig}"
NAMESPACE="llm-d-pfc-cpu"
LOGDIR="vllm-startup-logs"
VERIFY_SCRIPT="/tmp/verify-log-content.py"

# Create log directory
mkdir -p "$LOGDIR"

# Define configurations to capture
# Format: "model:config_name:vllm_args"
CONFIGS=(
    # 0.6B model
    "Qwen/Qwen3-0.6B:no-offload:"
    "Qwen/Qwen3-0.6B:native-offload-10k:--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"

    # 8B model
    "Qwen/Qwen3-8B:no-offload:"
    "Qwen/Qwen3-8B:native-offload-10k:--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"

    # 14B model
    "Qwen/Qwen3-14B:no-offload:"
    "Qwen/Qwen3-14B:native-offload-10k:--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
    "Qwen/Qwen3-14B:native-offload-20k:--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":20000}}'"

    # 32B-AWQ model
    "Qwen/Qwen3-32B-AWQ:no-offload:"
    "Qwen/Qwen3-32B-AWQ:native-offload-10k:--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
    "Qwen/Qwen3-32B-AWQ:native-offload-20k:--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":20000}}'"
)

capture_log() {
    local model="$1"
    local config_name="$2"
    local vllm_args="$3"

    # Create safe filename
    local model_short=$(echo "$model" | sed 's|Qwen/Qwen3-||' | sed 's|-AWQ||')
    local logfile="${LOGDIR}/vllm-${model_short}-${config_name}.log"

    echo "=========================================="
    echo "Capturing: $model ($config_name)"
    echo "Log file: $logfile"
    echo "=========================================="

    # Build vLLM command
    local base_cmd="vllm serve $model --tensor-parallel-size 2 --port 8000 --max-num-seq 1024"
    local full_cmd="$base_cmd $vllm_args"

    echo "vLLM command: $full_cmd"
    echo ""

    # Create JSON patch for deployment
    # Note: We need to properly escape the command for JSON
    # The command contains JSON itself (--kv-transfer-config), so we need to:
    # 1. Escape backslashes
    # 2. Escape double quotes
    local patch_file="/tmp/vllm-patch-$$.json"
    local escaped_cmd=$(echo "$full_cmd" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
    cat > "$patch_file" << EOF
[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/args",
    "value": ["exec $escaped_cmd"]
  }
]
EOF

    echo "Patching deployment..."
    kubectl --kubeconfig="$KUBECONFIG" patch deployment llm-d-model-server -n "$NAMESPACE" \
        --type='json' -p="$(cat $patch_file)"

    rm -f "$patch_file"

    # Wait for rollout
    echo "Waiting for rollout to complete..."
    kubectl --kubeconfig="$KUBECONFIG" rollout status deployment/llm-d-model-server -n "$NAMESPACE" --timeout=300s

    # Get pod name
    echo "Getting pod name..."
    local pod_name=$(kubectl --kubeconfig="$KUBECONFIG" get pods -n "$NAMESPACE" \
        -l llm-d.ai/inference-serving=true -o jsonpath='{.items[0].metadata.name}')

    if [ -z "$pod_name" ]; then
        echo "ERROR: Could not find pod"
        return 1
    fi

    echo "Pod: $pod_name"

    # Wait a bit for vLLM to initialize
    echo "Waiting 10 seconds for vLLM initialization..."
    sleep 10

    # Capture logs
    echo "Capturing logs..."
    kubectl --kubeconfig="$KUBECONFIG" logs -n "$NAMESPACE" "$pod_name" > "$logfile" 2>&1

    local line_count=$(wc -l < "$logfile")
    echo "Captured $line_count lines to $logfile"
    echo ""

    # Verify log contains required information
    echo "Verifying log content..."
    if python3 "$VERIFY_SCRIPT" "$logfile"; then
        echo "✅ SUCCESS: Log verified and complete"
        echo ""
        return 0
    else
        echo "❌ FAILED: Log missing required information"
        echo "Keeping log file for debugging: $logfile"
        echo ""
        return 1
    fi
}

# Main execution
echo "==================================================="
echo "KV-Cache Log Capture Script"
echo "==================================================="
echo "Total configurations to capture: ${#CONFIGS[@]}"
echo "Log directory: $LOGDIR"
echo ""

# Check if verification script exists
if [ ! -f "$VERIFY_SCRIPT" ]; then
    echo "ERROR: Verification script not found at $VERIFY_SCRIPT"
    echo "Please create it first"
    exit 1
fi

# Process each configuration
success_count=0
fail_count=0
failed_configs=()

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r model config_name vllm_args <<< "$config"

    if capture_log "$model" "$config_name" "$vllm_args"; then
        ((success_count++))
    else
        ((fail_count++))
        failed_configs+=("$model:$config_name")
    fi

    # Small delay between captures
    sleep 5
done

# Summary
echo "==================================================="
echo "CAPTURE SUMMARY"
echo "==================================================="
echo "Total: ${#CONFIGS[@]}"
echo "Success: $success_count"
echo "Failed: $fail_count"
echo ""

if [ $fail_count -gt 0 ]; then
    echo "Failed configurations:"
    for failed in "${failed_configs[@]}"; do
        echo "  - $failed"
    done
    echo ""
    exit 1
fi

echo "✅ All logs captured successfully!"
echo "Logs saved in: $LOGDIR/"
ls -lh "$LOGDIR/"
