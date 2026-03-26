#!/bin/bash
set -e

# Benchmark configuration parameters
KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
NAMESPACE="llm-d-pfc-cpu"

# Benchmark parameters (customizable)
TARGET="${TARGET:-http://llm-d-inference-gateway-istio:80}"
RATE_TYPE="${RATE_TYPE:-concurrent}"
RATE="${RATE:-1}"
MAX_SECONDS="${MAX_SECONDS:-30}"
RANDOM_SEED="${RANDOM_SEED:-889}"
PROMPT_TOKENS="${PROMPT_TOKENS:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
PREFIX_TOKENS="${PREFIX_TOKENS:-10000}"
TURNS="${TURNS:-5}"
PREFIX_COUNT="${PREFIX_COUNT:-2}"
SAMPLE_REQUESTS="${SAMPLE_REQUESTS:-10}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
REPLICAS="${CURRENT_REPLICAS:-1}"

# Inference server configuration (optional)
# If VLLM_EXTRA_ARGS is set, the inference server will be restarted with these additional arguments
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_ENV_VARS="${VLLM_ENV_VARS:-}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-llm-d-model-server}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPUS_PER_REPLICA="${GPUS_PER_REPLICA:-${TENSOR_PARALLEL_SIZE}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# EPP configuration (optional)
# If EPP_BACKEND_CONFIG is set, the EPP ConfigMap will be updated and EPP deployment restarted
# Valid values: "in-memory" (default), "redis", "valkey"
EPP_BACKEND_CONFIG="${EPP_BACKEND_CONFIG:-in-memory}"
EPP_CONFIGMAP="${EPP_CONFIGMAP:-llm-d-infpool-epp}"
EPP_DEPLOYMENT="${EPP_DEPLOYMENT:-llm-d-infpool-epp}"

# Hardware/software configuration for directory naming
HARDWARE="${HARDWARE:-1x2xL40S}"
SOFTWARE="${SOFTWARE:-upstream-llm-d-0.4.0}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
PARAMETERS="${PARAMETERS:-no-cpu-offload}"

# Generate run ID and output directory (no timestamp - results contain timestamps)
# Include replicas and rate in the directory structure for better organization
RUN_ID="${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PARAMETERS}_replica${REPLICAS}_rate${RATE}"
OUTPUT_DIR="./results/${RUN_ID}"

echo "=========================================="
echo "Preparing for Benchmark Run"
echo "=========================================="

# Determine which cache backend(s) to restart by checking both EPP config and LMCache env vars
CACHE_BACKENDS_TO_RESTART=""

# Check EPP backend configuration (llm-d-redis, llm-d-valkey)
if [ "${EPP_BACKEND_CONFIG}" = "redis" ] || [ "${EPP_BACKEND_CONFIG}" = "valkey" ]; then
    CACHE_BACKENDS_TO_RESTART="${EPP_BACKEND_CONFIG}"
fi

# Check LMCache remote URL configuration (lmcache-redis, lmcache-valkey)
if echo "${VLLM_ENV_VARS}" | grep -q "LMCACHE_REMOTE_URL=redis://"; then
    if [ -z "${CACHE_BACKENDS_TO_RESTART}" ]; then
        CACHE_BACKENDS_TO_RESTART="redis"
    elif ! echo "${CACHE_BACKENDS_TO_RESTART}" | grep -q "redis"; then
        CACHE_BACKENDS_TO_RESTART="${CACHE_BACKENDS_TO_RESTART} redis"
    fi
fi

if echo "${VLLM_ENV_VARS}" | grep -q "LMCACHE_REMOTE_URL=valkey://"; then
    if [ -z "${CACHE_BACKENDS_TO_RESTART}" ]; then
        CACHE_BACKENDS_TO_RESTART="valkey"
    elif ! echo "${CACHE_BACKENDS_TO_RESTART}" | grep -q "valkey"; then
        CACHE_BACKENDS_TO_RESTART="${CACHE_BACKENDS_TO_RESTART} valkey"
    fi
fi

# Restart identified cache backends to clear cache state before benchmarks
if [ -n "${CACHE_BACKENDS_TO_RESTART}" ]; then
    for CACHE_BACKEND in ${CACHE_BACKENDS_TO_RESTART}; do
        echo "Restarting ${CACHE_BACKEND} pod to clear cache state..."

        # Determine which deployment to restart
        CACHE_DEPLOYMENT="${CACHE_BACKEND}"

        OLD_CACHE_PODS=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" -l app="${CACHE_DEPLOYMENT}" -o jsonpath='{.items[*].metadata.name}')
        if [ -n "${OLD_CACHE_PODS}" ]; then
            for pod in ${OLD_CACHE_PODS}; do
                echo "  Deleting ${CACHE_BACKEND} pod: ${pod}"
                kubectl --kubeconfig="${KUBECONFIG}" delete pod -n "${NAMESPACE}" "${pod}" --wait=false 2>/dev/null
            done

            echo "  Waiting for old ${CACHE_BACKEND} pod(s) to terminate..."
            for pod in ${OLD_CACHE_PODS}; do
                kubectl --kubeconfig="${KUBECONFIG}" wait --for=delete pod/"${pod}" -n "${NAMESPACE}" --timeout=60s 2>/dev/null || true
            done

            echo "  Waiting for new ${CACHE_BACKEND} pod(s) to be ready..."
            kubectl --kubeconfig="${KUBECONFIG}" wait --for=condition=ready pod -l app="${CACHE_DEPLOYMENT}" -n "${NAMESPACE}" --timeout=120s
            echo "  ${CACHE_BACKEND} pod(s) restarted successfully"
        else
            echo "  Warning: No ${CACHE_BACKEND} pods found"
        fi
        echo ""
    done
fi

# Restart PCP pod to get fresh pod/directory for this benchmark run
echo "Restarting PCP pod to create fresh archive directory..."
OLD_PCP_PODS=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=pcp -o jsonpath='{.items[*].metadata.name}')
if [ -n "${OLD_PCP_PODS}" ]; then
    for pod in ${OLD_PCP_PODS}; do
        echo "  Deleting PCP pod: ${pod}"
        kubectl --kubeconfig="${KUBECONFIG}" delete pod -n "${NAMESPACE}" "${pod}" --wait=false 2>/dev/null
    done

    # Wait for old pods to be fully deleted before waiting for new ones
    echo "  Waiting for old PCP pod(s) to terminate..."
    for pod in ${OLD_PCP_PODS}; do
        kubectl --kubeconfig="${KUBECONFIG}" wait --for=delete pod/"${pod}" -n "${NAMESPACE}" --timeout=60s 2>/dev/null || true
    done

    echo "  Waiting for new PCP pod(s) to be ready..."
    kubectl --kubeconfig="${KUBECONFIG}" wait --for=condition=ready pod -l app.kubernetes.io/name=pcp -n "${NAMESPACE}" --timeout=600s
    echo "  PCP pod(s) restarted successfully"

    # Capture the PCP pod hostname now — archives live in /var/log/pcp/pmlogger/<hostname>/
    # on the hostPath. If the pod is replaced later (liveness probe failure), the new pod
    # can still read the old archives by hostname, so we need to preserve this.
    NEW_PCP_PODS=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/name=pcp -o jsonpath='{.items[*].metadata.name}')
    PCP_ARCHIVE_HOSTNAMES=""
    for pod in ${NEW_PCP_PODS}; do
        h=$(kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${pod}" -- hostname 2>/dev/null || true)
        PCP_ARCHIVE_HOSTNAMES="${PCP_ARCHIVE_HOSTNAMES} ${h}"
    done
    PCP_ARCHIVE_HOSTNAMES="${PCP_ARCHIVE_HOSTNAMES# }"
    echo "  PCP archive hostname(s): ${PCP_ARCHIVE_HOSTNAMES}"

    # Wait for pmlogger to be fully operational by checking pmcd.pmlogger.host contains expected hostname
    echo "  Waiting for pmlogger to initialize..."
    NEW_PCP_PODS=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=pcp -o jsonpath='{.items[*].metadata.name}')
    for pod in ${NEW_PCP_PODS}; do
        echo "  Checking pmlogger in pod: ${pod}"
        for i in {1..30}; do
            # Check if pminfo output contains the pod hostname
            if kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${pod}" -- sh -c "pminfo -f pmcd.pmlogger.host 2>/dev/null | grep -q \"\$(hostname)\"" 2>/dev/null; then
                echo "    pmlogger operational and logging to correct hostname"
                break
            fi
            if [ $i -eq 30 ]; then
                echo "    Warning: pmlogger not ready in ${pod} after 30 seconds"
            fi
            sleep 1
        done
    done
else
    echo "  No existing PCP pods found"
fi

# Clean up any existing EPP ConfigMap from previous runs
echo ""
echo "Cleaning up previous EPP configuration..."
kubectl --kubeconfig="${KUBECONFIG}" delete configmap "${EPP_CONFIGMAP}" -n "${NAMESPACE}" --ignore-not-found=true

# Configure EPP based on backend type
echo ""
echo "Configuring EPP backend..."
echo "  Backend: ${EPP_BACKEND_CONFIG}"

case "${EPP_BACKEND_CONFIG}" in
    "in-memory")
        MANIFEST_FILE="manifests/epp-configmap-in-memory.yaml"
        ;;
    "redis")
        MANIFEST_FILE="manifests/epp-configmap-redis.yaml"
        ;;
    "valkey")
        MANIFEST_FILE="manifests/epp-configmap-valkey.yaml"
        ;;
    *)
        echo "ERROR: Invalid EPP_BACKEND_CONFIG value: ${EPP_BACKEND_CONFIG}"
        echo "Valid values: in-memory, redis, valkey"
        exit 1
        ;;
esac

# Apply the ConfigMap from manifest
echo "  Applying ConfigMap from ${MANIFEST_FILE}..."
kubectl --kubeconfig="${KUBECONFIG}" apply -f "${MANIFEST_FILE}"

# Restart EPP deployment
echo "  Restarting EPP deployment..."
kubectl --kubeconfig="${KUBECONFIG}" rollout restart deployment/"${EPP_DEPLOYMENT}" -n "${NAMESPACE}"

# Poll for a Ready EPP pod (avoids depending on kube-controller-manager Deployment
# rollout bookkeeping, which is intermittently unavailable on this cluster).
# EPP pods carry label inferencepool=<deployment-name>.
# Polling is used instead of kubectl wait to handle the race where both the old
# terminating pod and the new pod briefly share the same label.
echo "  Waiting for EPP pod to be Ready..."
EPP_READY=""
for i in $(seq 1 60); do
    # Find a non-terminating EPP pod that has Ready=True.
    # Terminating pods retain status.phase=Running so field-selector can't exclude them;
    # python parses the full JSON and skips pods with metadata.deletionTimestamp set.
    EPP_READY=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
        -l inferencepool="${EPP_DEPLOYMENT}" -o json 2>/dev/null | \
        python3 -c "
import json, sys
d = json.load(sys.stdin)
for item in d.get('items', []):
    if item.get('metadata', {}).get('deletionTimestamp'):
        continue
    for cond in item.get('status', {}).get('conditions', []):
        if cond.get('type') == 'Ready' and cond.get('status') == 'True':
            print(item['metadata']['name'])
            sys.exit(0)
" 2>/dev/null || true)
    if [ -n "${EPP_READY}" ]; then
        echo "  EPP pod ready: ${EPP_READY}"
        break
    fi
    sleep 5
done
if [ -z "${EPP_READY}" ]; then
    echo "ERROR: EPP not ready after 5 minutes"
    exit 1
fi

# Wait for EPP to fully initialize and connect to model server
echo "  Waiting 90 seconds for EPP initialization..."
sleep 90

echo "  EPP configured successfully"

# Always restart inference server to ensure correct model and configuration
echo ""
echo "Configuring inference server..."
echo "  Model: ${MODEL}"
if [ -n "${CONTAINER_IMAGE}" ]; then
    echo "  Container image: ${CONTAINER_IMAGE}"
fi
if [ -n "${VLLM_EXTRA_ARGS}" ]; then
    echo "  Extra args: ${VLLM_EXTRA_ARGS}"
fi
if [ -n "${VLLM_ENV_VARS}" ]; then
    echo "  Extra env:  ${VLLM_ENV_VARS}"
fi

# Build vLLM command/args based on image type
if [ "${USE_LMCACHE_IMAGE}" = "true" ]; then
    # LMCache images use /opt/venv/bin/vllm command with array args
    echo "  Using LMCache image command structure"

    # Build args array with all lmcache arguments
    # All lmcache configs use --kv-transfer-config and --enable-prefix-caching
    VLLM_ARGS_ARRAY='["serve","'${MODEL}'","--tensor-parallel-size","'${TENSOR_PARALLEL_SIZE}'","--port","8000","--max-num-seq","1024","--kv-transfer-config","{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}","--enable-prefix-caching"]'
else
    # llm-d images use bash -c with exec vllm serve
    BASE_VLLM_ARGS="exec vllm serve ${MODEL} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --port 8000 --max-num-seq 1024 --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"

    # Append extra args if provided
    if [ -n "${VLLM_EXTRA_ARGS}" ]; then
        NEW_ARGS="${BASE_VLLM_ARGS} ${VLLM_EXTRA_ARGS}"
    else
        NEW_ARGS="${BASE_VLLM_ARGS}"
    fi

    # Optionally prepend a setup command (e.g. pip install for external connectors)
    # VLLM_PRE_CMD is joined with && so failures abort before vllm starts
    if [ -n "${VLLM_PRE_CMD}" ]; then
        NEW_ARGS="${VLLM_PRE_CMD} && ${NEW_ARGS}"
    fi
fi

# Build JSON patch operations
PATCH_OPS=""

# Add container image patch if CONTAINER_IMAGE is set
if [ -n "${CONTAINER_IMAGE}" ]; then
    ESCAPED_IMAGE=$(echo -n "${CONTAINER_IMAGE}" | jq -R -s '.')
    IMAGE_PATCH=$(cat <<EOF
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/image",
    "value": ${ESCAPED_IMAGE}
  }
EOF
)
    PATCH_OPS="${IMAGE_PATCH}"
fi

# Build command and args patches based on image type
if [ "${USE_LMCACHE_IMAGE}" = "true" ]; then
    # For LMCache images, replace command and args
    COMMAND_PATCH=$(cat <<EOF
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/command",
    "value": ["/opt/venv/bin/vllm"]
  }
EOF
)
    ARGS_PATCH=$(cat <<EOF
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/args",
    "value": ${VLLM_ARGS_ARRAY}
  }
EOF
)

    if [ -n "${PATCH_OPS}" ]; then
        PATCH_OPS="${PATCH_OPS},${COMMAND_PATCH},${ARGS_PATCH}"
    else
        PATCH_OPS="${COMMAND_PATCH},${ARGS_PATCH}"
    fi
else
    # For llm-d images, reset command to bash and set args to single string
    COMMAND_PATCH=$(cat <<EOF
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/command",
    "value": ["bash", "-c"]
  }
EOF
)
    ESCAPED_ARGS=$(echo -n "${NEW_ARGS}" | jq -R -s '.')
    ARGS_PATCH=$(cat <<EOF
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/args",
    "value": [${ESCAPED_ARGS}]
  }
EOF
)

    if [ -n "${PATCH_OPS}" ]; then
        PATCH_OPS="${PATCH_OPS},${COMMAND_PATCH},${ARGS_PATCH}"
    else
        PATCH_OPS="${COMMAND_PATCH},${ARGS_PATCH}"
    fi
fi

# Set or clear environment variables using kubectl set env
if [ -n "${VLLM_ENV_VARS}" ]; then
    echo "  Setting environment variables..."
    # Build env var arguments for kubectl set env
    ENV_ARGS=""
    for env_pair in ${VLLM_ENV_VARS}; do
        ENV_ARGS="${ENV_ARGS} ${env_pair}"
    done

    # Use kubectl set env to replace environment variables
    kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" -n "${NAMESPACE}" --containers=vllm ${ENV_ARGS}
else
    # Clear LMCache-specific environment variables when not using LMCache
    echo "  Clearing LMCache environment variables..."
    kubectl --kubeconfig="${KUBECONFIG}" set env deployment/"${INFERENCE_DEPLOYMENT}" -n "${NAMESPACE}" --containers=vllm \
        HOME- LMCACHE_MAX_LOCAL_CPU_SIZE- LMCACHE_REMOTE_URL- LMCACHE_USE_EXPERIMENTAL- PYTHONHASHSEED- 2>/dev/null || true
fi

# Configure health probes for lmcache images
# LMCache images don't expose /health endpoint, so we use /v1/models instead
if [ "${USE_LMCACHE_IMAGE}" = "true" ]; then
    # Replace readiness probe with HTTP GET to /v1/models
    # This endpoint only returns 200 when vLLM has fully loaded the model
    SET_READINESS_PROBE=$(jq -c '.[0]' manifests/lmcache-readiness-probe-patch.json)
    if [ -n "${PATCH_OPS}" ]; then
        PATCH_OPS="${PATCH_OPS},${SET_READINESS_PROBE}"
    else
        PATCH_OPS="${SET_READINESS_PROBE}"
    fi

    # Remove liveness and startup probes (not needed for lmcache)
    # Check if startupProbe exists
    if kubectl --kubeconfig="${KUBECONFIG}" get deployment "${INFERENCE_DEPLOYMENT}" -n "${NAMESPACE}" -o jsonpath='{.spec.template.spec.containers[0].startupProbe}' 2>/dev/null | grep -q .; then
        REMOVE_STARTUP_PROBE=$(jq -c '.[0]' manifests/lmcache-remove-startup-probe-patch.json)
        if [ -n "${PATCH_OPS}" ]; then
            PATCH_OPS="${PATCH_OPS},${REMOVE_STARTUP_PROBE}"
        else
            PATCH_OPS="${REMOVE_STARTUP_PROBE}"
        fi
    fi

    # Check if livenessProbe exists
    if kubectl --kubeconfig="${KUBECONFIG}" get deployment "${INFERENCE_DEPLOYMENT}" -n "${NAMESPACE}" -o jsonpath='{.spec.template.spec.containers[0].livenessProbe}' 2>/dev/null | grep -q .; then
        REMOVE_LIVENESS_PROBE=$(jq -c '.[0]' manifests/lmcache-remove-liveness-probe-patch.json)
        if [ -n "${PATCH_OPS}" ]; then
            PATCH_OPS="${PATCH_OPS},${REMOVE_LIVENESS_PROBE}"
        else
            PATCH_OPS="${REMOVE_LIVENESS_PROBE}"
        fi
    fi
fi

# Create final JSON patch
PATCH_JSON="[${PATCH_OPS}]"

# Apply patch
echo "${PATCH_JSON}" | kubectl --kubeconfig="${KUBECONFIG}" patch deployment "${INFERENCE_DEPLOYMENT}" -n "${NAMESPACE}" --type='json' -p "$(cat)"

# Wait for rollout (allow time for model downloads - up to 15 minutes)
echo "  Waiting for inference server rollout to complete..."
INFER_READY=""
for i in $(seq 1 180); do
    INFER_READY=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
        -l llm-d.ai/inference-serving=true -o json 2>/dev/null | \
        python3 -c "
import json, sys
d = json.load(sys.stdin)
for item in d.get('items', []):
    if item.get('metadata', {}).get('deletionTimestamp'):
        continue
    for cond in item.get('status', {}).get('conditions', []):
        if cond.get('type') == 'Ready' and cond.get('status') == 'True':
            print(item['metadata']['name'])
            sys.exit(0)
" 2>/dev/null || true)
    if [ -n "${INFER_READY}" ]; then
        echo "  Inference server ready: ${INFER_READY}"
        break
    fi
    sleep 5
done
if [ -z "${INFER_READY}" ]; then
    echo "ERROR: Inference server not ready after 15 minutes"
    exit 1
fi

# Wait for pods to be ready using Kubernetes readiness probes
echo "  Waiting for inference server pods to be ready..."
kubectl --kubeconfig="${KUBECONFIG}" wait --for=condition=Ready pod -l llm-d.ai/inference-serving=true -n "${NAMESPACE}" --timeout=600s

# Additional validation: ensure the gateway can route to the backend
# guidellm validates using the /health endpoint, but lmcache doesn't expose this
# So we need to ensure the gateway can route successfully before guidellm runs
echo "  Validating gateway routing..."
INTERACTIVE_POD=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" -l app=guidellm --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
if [ -z "${INTERACTIVE_POD}" ]; then
    echo "  Warning: No interactive pod found for validation, skipping gateway check"
else
    # Try to reach the target through the gateway (retry for up to 60 seconds)
    for i in {1..12}; do
        if kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${INTERACTIVE_POD}" -- curl -sf "${TARGET}/v1/models" >/dev/null 2>&1; then
            echo "  Gateway routing validated successfully"
            break
        fi
        if [ $i -eq 12 ]; then
            echo "  Warning: Gateway routing validation failed after 60 seconds, proceeding anyway"
        else
            sleep 5
        fi
    done
fi

echo "  Inference server configured successfully"

echo ""
echo "=========================================="
echo "Capturing vLLM Startup Logs"
echo "=========================================="

# Get inference server pod name
INFERENCE_POD=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
    -l llm-d.ai/inference-serving=true -o jsonpath='{.items[0].metadata.name}')

if [ -z "${INFERENCE_POD}" ]; then
    echo "  Warning: Could not find inference server pod, skipping log capture"
else
    echo "  Inference pod: ${INFERENCE_POD}"

    # Create output directory if it doesn't exist yet
    mkdir -p "${OUTPUT_DIR}"

    # Capture vLLM startup logs
    VLLM_LOG="${OUTPUT_DIR}/vllm-startup.log"
    echo "  Capturing logs to: ${VLLM_LOG}"
    kubectl --kubeconfig="${KUBECONFIG}" logs -n "${NAMESPACE}" "${INFERENCE_POD}" > "${VLLM_LOG}" 2>&1

    # Show log size
    LOG_LINES=$(wc -l < "${VLLM_LOG}")
    echo "  Captured ${LOG_LINES} lines"

    # Compress with zstd
    echo "  Compressing with zstd..."
    zstd -q -f "${VLLM_LOG}"
    rm -f "${VLLM_LOG}"
    echo "  Saved to: ${VLLM_LOG}.zst"
fi

echo ""
echo "=========================================="
echo "Detecting Cluster Resources"
echo "=========================================="

# Detect current pods
echo "Detecting PCP pods..."
PCP_PODS=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=pcp -o jsonpath='{.items[*].metadata.name}')
if [ -z "${PCP_PODS}" ]; then
    echo "ERROR: No PCP pods found in namespace ${NAMESPACE}"
    exit 1
fi
PCP_POD_COUNT=$(echo "${PCP_PODS}" | wc -w)
echo "Found ${PCP_POD_COUNT} PCP pod(s): ${PCP_PODS}"

echo "Detecting current interactive pod..."
INTERACTIVE_POD=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" -l app=guidellm --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
if [ -z "${INTERACTIVE_POD}" ]; then
    echo "ERROR: No running interactive pod found in namespace ${NAMESPACE}"
    exit 1
fi
echo "Found interactive pod: ${INTERACTIVE_POD}"

echo ""
echo "=========================================="
echo "Benchmark Run Configuration"
echo "=========================================="
echo "Run ID: ${RUN_ID}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Namespace: ${NAMESPACE}"
echo "Replicas: ${REPLICAS}"
echo "Interactive Pod: ${INTERACTIVE_POD}"
echo "PCP Pods (${PCP_POD_COUNT}): ${PCP_PODS}"
echo "Target: ${TARGET}"
echo "Rate (Concurrency): ${RATE}"
echo "Max Seconds: ${MAX_SECONDS}"
echo "Data: prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS},prefix_tokens=${PREFIX_TOKENS},turns=${TURNS},prefix_count=${PREFIX_COUNT}"
echo "Max Requests: ${SAMPLE_REQUESTS}"
if [ -n "${EXPERIMENT_NAME}" ]; then
    echo "Experiment: ${EXPERIMENT_NAME}"
fi
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Record start time for PCP archive filtering
START_TIME=$(date +%s)
echo "Benchmark start time: $(date -d @${START_TIME})"

# Save configuration to file
cat > "${OUTPUT_DIR}/benchmark-config.txt" <<EOF
Run ID: ${RUN_ID}
Date: $(date)
Target: ${TARGET}
Rate Type: ${RATE_TYPE}
Rate (Concurrency): ${RATE}
Max Seconds: ${MAX_SECONDS}
Random Seed: ${RANDOM_SEED}
Prompt Tokens: ${PROMPT_TOKENS}
Output Tokens: ${OUTPUT_TOKENS}
Prefix Tokens: ${PREFIX_TOKENS}
Prefix Count: ${PREFIX_COUNT}
Turns: ${TURNS}
Max Requests (Sample Requests): ${SAMPLE_REQUESTS}
Hardware: ${HARDWARE}
Software: ${SOFTWARE}
Model: ${MODEL}
Model Name: ${MODEL_NAME}
Parameters: ${PARAMETERS}
GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}
Replicas: ${REPLICAS}
Experiment Name: ${EXPERIMENT_NAME}
Namespace: ${NAMESPACE}
Interactive Pod: ${INTERACTIVE_POD}
PCP Pods: ${PCP_PODS}
PCP Pod Count: ${PCP_POD_COUNT}
Inference Deployment: ${INFERENCE_DEPLOYMENT}
vLLM Extra Args: ${VLLM_EXTRA_ARGS}
vLLM Env Vars: ${VLLM_ENV_VARS}
EPP Backend: ${EPP_BACKEND_CONFIG}
EPP ConfigMap: ${EPP_CONFIGMAP}
EPP Deployment: ${EPP_DEPLOYMENT}
EOF

echo ""
echo "Running guidellm benchmark..."

# Build guidellm command
# --sample-requests=0: suppress per-request sample data from JSON output, keeping
# files small. Confirmed with guidellm maintainer (see PR #591 for background).
GUIDELLM_CMD="guidellm benchmark run \
    --target=\"${TARGET}\" \
    --rate-type=\"${RATE_TYPE}\" \
    --rate=\"${RATE}\" \
    --max-seconds=\"${MAX_SECONDS}\" \
    --random-seed=\"${RANDOM_SEED}\" \
    --data='{\"prompt_tokens\":${PROMPT_TOKENS},\"output_tokens\":${OUTPUT_TOKENS},\"prefix_tokens\":${PREFIX_TOKENS},\"turns\":${TURNS},\"prefix_count\":${PREFIX_COUNT}}' \
    --sample-requests=0 \
    --outputs=/models/benchmark.json"

# Execute guidellm command
kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${INTERACTIVE_POD}" -- sh -c "${GUIDELLM_CMD}"

END_TIME=$(date +%s)
echo "Benchmark end time: $(date -d @${END_TIME})"
DURATION=$((END_TIME - START_TIME))
echo "Benchmark duration: ${DURATION} seconds"

echo ""
echo "Collecting guidellm results..."
RESULT_FILE="/models/benchmark.json"

# Note: Official guidellm container doesn't have zstd, but JSON files are small now
# thanks to omitting --sample-requests (see https://github.com/vllm-project/guidellm/pull/591)
# Transfer uncompressed JSON, then compress locally after download

# Use chunked transfer for reliability with large files
echo "Downloading guidellm results (chunked transfer)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/../../scripts/transfer-large-file-chunked.sh" "${KUBECONFIG}" "${NAMESPACE}" "${INTERACTIVE_POD}" "${RESULT_FILE}" "${OUTPUT_DIR}/guidellm-results.json"
COPY_RESULT=$?

# Check if file was actually copied (non-empty)
if [ -s "${OUTPUT_DIR}/guidellm-results.json" ] && [ ${COPY_RESULT} -eq 0 ]; then
    # Compress locally after download to save disk space
    echo "Compressing guidellm results locally..."
    zstd -q -f --rm "${OUTPUT_DIR}/guidellm-results.json"

    # Strip request/response content (prompt text, model output) from the result.
    # These fields are not needed for analysis and account for ~86% of file size.
    echo "Stripping request content from guidellm results..."
    python3 "${SCRIPT_DIR}/../../scripts/strip-guidellm-request-content.py" \
        "${OUTPUT_DIR}/guidellm-results.json.zst" || \
        echo "  Warning: strip failed (file may be corrupt), continuing"

    # Clean up JSON file from pod to prevent reuse in next benchmark
    kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${INTERACTIVE_POD}" -- rm -f "${RESULT_FILE}" 2>/dev/null || true
else
    echo "ERROR: Failed to copy guidellm results (exit code: ${COPY_RESULT})"
    exit 1
fi

echo ""
echo "Collecting PCP archives from ${PCP_POD_COUNT} pod(s)..."
# Create PCP archives directory
mkdir -p "${OUTPUT_DIR}/pcp-archives"

# Re-query current PCP pods at collection time — the pod may have been replaced
# by a liveness probe restart during the benchmark. We exec into whatever pod is
# currently running but read archives from the ORIGINAL hostname directory, which
# persists on the hostPath even after the pod is replaced.
CURRENT_PCP_PODS=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
    -l app.kubernetes.io/name=pcp -o jsonpath='{.items[*].metadata.name}')
EXEC_PCP_POD=$(echo "${CURRENT_PCP_PODS}" | awk '{print $1}')

echo "Finding PCP archives for time window: $(date -d @${START_TIME}) to $(date -d @${END_TIME})"
for PCP_ARCHIVE_HOST in ${PCP_ARCHIVE_HOSTNAMES}; do
    echo ""
    echo "Collecting archives for hostname: ${PCP_ARCHIVE_HOST} (via pod: ${EXEC_PCP_POD})"

    # Get the node name from the current pod (for output directory naming)
    POD_NODE=$(kubectl --kubeconfig="${KUBECONFIG}" get pod -n "${NAMESPACE}" \
        "${EXEC_PCP_POD}" -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "${PCP_ARCHIVE_HOST}")
    echo "  Node: ${POD_NODE}"

    # Create subdirectory for this node's archives
    mkdir -p "${OUTPUT_DIR}/pcp-archives/${POD_NODE}"

    # Copy archives from the original hostname's directory (not $(hostname) which would
    # give the new pod's name if the pod was replaced during the benchmark run)
    kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${EXEC_PCP_POD}" -- \
        sh -c "ls -1 /var/log/pcp/pmlogger/${PCP_ARCHIVE_HOST}/[0-9]* 2>/dev/null | grep -E \"\.[0-9]+$\" | head -20" | while read -r archive_path; do
        archive_file=$(basename "${archive_path}")

        echo "  Copying: ${archive_file}"
        timeout --signal=KILL 60 sh -c "kubectl --kubeconfig='${KUBECONFIG}' exec -n '${NAMESPACE}' '${EXEC_PCP_POD}' -- cat '${archive_path}' > '${OUTPUT_DIR}/pcp-archives/${POD_NODE}/${archive_file}'" 2>&1 || echo "    Warning: Failed to copy ${archive_file}"

        archive_base="${archive_path%.[0-9]*}"
        archive_base_name=$(basename "${archive_base}")
        for ext in index meta; do
            ext_file="${archive_base}.${ext}"
            timeout --signal=KILL 60 sh -c "kubectl --kubeconfig='${KUBECONFIG}' exec -n '${NAMESPACE}' '${EXEC_PCP_POD}' -- cat '${ext_file}' > '${OUTPUT_DIR}/pcp-archives/${POD_NODE}/${archive_base_name}.${ext}'" 2>&1 || true
        done
    done
done

echo ""
echo "Compressing PCP archives with zstd..."
# Compress PCP archive files (.meta, .index, and .[0-9]+ data files) with zstd
find "${OUTPUT_DIR}/pcp-archives" -type f \( -name "*.meta" -o -name "*.index" -o -regex ".*\.[0-9][0-9]*$" \) -print0 | while IFS= read -r -d '' file; do
    echo "  Compressing: $(basename "$file")"
    zstd -q --rm "$file" || echo "  Warning: Failed to compress $(basename "$file")"
done
echo "Compression complete"

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Contents:"
ls -lh "${OUTPUT_DIR}"
echo ""
echo "PCP Archives:"
ls -lh "${OUTPUT_DIR}/pcp-archives/" 2>/dev/null || echo "No PCP archives collected"
echo "=========================================="
