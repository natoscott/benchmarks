#!/bin/bash
# Capture vLLM startup log for a single configuration

set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model> <config_name> <vllm_args>"
    echo "Example: $0 'Qwen/Qwen3-0.6B' 'no-offload' ''"
    echo "Example: $0 'Qwen/Qwen3-14B' 'native-offload-10k' '--kv-transfer-config ...'"
    exit 1
fi

MODEL="$1"
CONFIG_NAME="$2"
VLLM_ARGS="$3"

NAMESPACE="llm-d-pfc-cpu"
LOGDIR="vllm-startup-logs"
VERIFY_SCRIPT="/tmp/verify-log-content.py"

mkdir -p "$LOGDIR"

# Create safe filename
MODEL_SHORT=$(echo "$MODEL" | sed 's|Qwen/Qwen3-||' | sed 's|-AWQ||')
LOGFILE="${LOGDIR}/vllm-${MODEL_SHORT}-${CONFIG_NAME}.log"

echo "=========================================="
echo "Capturing: $MODEL ($CONFIG_NAME)"
echo "Log file: $LOGFILE"
echo "=========================================="

# Build vLLM command
BASE_CMD="vllm serve $MODEL --tensor-parallel-size 2 --port 8000 --max-num-seq 1024"
FULL_CMD="$BASE_CMD $VLLM_ARGS"

echo "vLLM command: $FULL_CMD"
echo ""

# Create JSON patch for deployment
# Note: We need to properly escape the command for JSON
# The command contains JSON itself (--kv-transfer-config), so we need to:
# 1. Escape backslashes
# 2. Escape double quotes
PATCH_FILE="/tmp/vllm-patch-$$.json"
ESCAPED_CMD=$(echo "$FULL_CMD" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')
cat > "$PATCH_FILE" << EOF
[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/args",
    "value": ["exec $ESCAPED_CMD"]
  }
]
EOF

echo "Patching deployment..."
kubectl --kubeconfig=kubeconfig patch deployment llm-d-model-server -n "$NAMESPACE" \
    --type='json' -p="$(cat $PATCH_FILE)"

rm -f "$PATCH_FILE"

# Wait for rollout
echo "Waiting for rollout to complete..."
kubectl --kubeconfig=kubeconfig rollout status deployment/llm-d-model-server -n "$NAMESPACE" --timeout=300s

# Get pod name
echo "Getting pod name..."
POD_NAME=$(kubectl --kubeconfig=kubeconfig get pods -n "$NAMESPACE" \
    -l llm-d.ai/inference-serving=true -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "ERROR: Could not find pod"
    exit 1
fi

echo "Pod: $POD_NAME"

# Wait a bit for vLLM to initialize
echo "Waiting 10 seconds for vLLM initialization..."
sleep 10

# Capture logs
echo "Capturing logs..."
kubectl --kubeconfig=kubeconfig logs -n "$NAMESPACE" "$POD_NAME" > "$LOGFILE" 2>&1

LINE_COUNT=$(wc -l < "$LOGFILE")
echo "Captured $LINE_COUNT lines to $LOGFILE"
echo ""

# Verify log contains required information
echo "Verifying log content..."
if python3 "$VERIFY_SCRIPT" "$LOGFILE"; then
    echo "✅ SUCCESS: Log verified and complete"
    echo ""
    exit 0
else
    echo "❌ FAILED: Log missing required information"
    echo "Keeping log file for debugging: $LOGFILE"
    echo ""
    exit 1
fi
