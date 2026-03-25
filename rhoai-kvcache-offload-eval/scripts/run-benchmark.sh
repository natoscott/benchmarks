#!/bin/bash
# Single benchmark run: configure LLMInferenceService, wait for ready, run guidellm,
# collect results + PCP archives.
#
# Called by run-rhoai-3.3.sh with all variables exported.
set -euo pipefail

KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"

# Benchmark parameters
RATE_TYPE="${RATE_TYPE:-concurrent}"
RATE="${RATE:-1}"
MAX_SECONDS="${MAX_SECONDS:-120}"
RANDOM_SEED="${RANDOM_SEED:-889}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
PREFIX_TOKENS="${PREFIX_TOKENS:-10000}"
TURNS="${TURNS:-5}"
PREFIX_COUNT="${PREFIX_COUNT:-2}"

# Inference server configuration
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-70B-Instruct}"
LLM_SERVICE_NAME="${LLM_SERVICE_NAME:-llama-70b}"
PARAMETERS="${PARAMETERS:-no-offload}"
REPLICAS="${CURRENT_REPLICAS:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
NUM_CPU_BLOCKS="${NUM_CPU_BLOCKS:-20000}"   # used for native-offload-20k

# Identifiers for output directory
HARDWARE="${HARDWARE:-2x8xH200}"
SOFTWARE="${SOFTWARE:-rhoai-3.3}"

# Gateway access — HTTPS via the openshift-ai-inference gateway.
# Path-prefixed per namespace/service-name. SA token used as Bearer via OPENAI_API_KEY.
# Derive the apps domain from the cluster's ingress config at runtime.
APPS_DOMAIN=$(kubectl --kubeconfig="${KUBECONFIG}" get ingresses.config.openshift.io cluster \
    -o jsonpath='{.spec.domain}' 2>/dev/null || echo "apps.example.com")
GATEWAY_HOST="inference-gateway.${APPS_DOMAIN}"
TARGET="${TARGET:-https://${GATEWAY_HOST}/${NAMESPACE}/${LLM_SERVICE_NAME}}"

RUN_ID="${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PARAMETERS}_replica${REPLICAS}_rate${RATE}"
OUTPUT_DIR="./results/${RUN_ID}"

echo "=========================================="
echo "Benchmark Run: ${RUN_ID}"
echo "=========================================="

# Skip if already complete
if [ -f "${OUTPUT_DIR}/guidellm-results.json.zst" ]; then
    echo "SKIPPING (already complete): ${OUTPUT_DIR}"
    exit 0
fi

# ── Configure VLLM_ADDITIONAL_ARGS for this run ──────────────────────────────
BASE_VLLM_ARGS="--tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --max-num-seq 1024"

case "${PARAMETERS}" in
    "no-offload")
        VLLM_EXTRA=""
        ;;
    "native-offload-20k")
        KV_CFG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":${NUM_CPU_BLOCKS}}}"
        VLLM_EXTRA="--kv-transfer-config '${KV_CFG}'"
        ;;
    *)
        echo "ERROR: Unknown PARAMETERS=${PARAMETERS}"; exit 1 ;;
esac

VLLM_ADDITIONAL_ARGS="${BASE_VLLM_ARGS}${VLLM_EXTRA:+ ${VLLM_EXTRA}}"

# ── Patch LLMInferenceService ─────────────────────────────────────────────────
echo "Configuring LLMInferenceService ${LLM_SERVICE_NAME}..."
echo "  Replicas: ${REPLICAS}"
echo "  VLLM_ADDITIONAL_ARGS: ${VLLM_ADDITIONAL_ARGS}"

# Escape args for JSON patch
ARGS_JSON=$(printf '%s' "${VLLM_ADDITIONAL_ARGS}" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")

kubectl --kubeconfig="${KUBECONFIG}" patch llminferenceservice "${LLM_SERVICE_NAME}" \
    -n "${NAMESPACE}" --type=json -p "[
  {\"op\":\"replace\",\"path\":\"/spec/replicas\",\"value\":${REPLICAS}},
  {\"op\":\"replace\",\"path\":\"/spec/template/containers/0/env/1/value\",\"value\":${ARGS_JSON}}
]"

# ── Wait for LLMInferenceService to be Ready ──────────────────────────────────
echo "Waiting for LLMInferenceService to be Ready (model may need to be downloaded)..."
TIMEOUT=1800   # 30 min: allows for model download on first run
INTERVAL=15
ELAPSED=0
while true; do
    READY=$(kubectl --kubeconfig="${KUBECONFIG}" get llminferenceservice "${LLM_SERVICE_NAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
    if [[ "${READY}" == "True" ]]; then
        echo "  LLMInferenceService is Ready"
        break
    fi
    if (( ELAPSED >= TIMEOUT )); then
        echo "ERROR: Timed out waiting for LLMInferenceService after ${TIMEOUT}s"
        kubectl --kubeconfig="${KUBECONFIG}" describe llminferenceservice "${LLM_SERVICE_NAME}" -n "${NAMESPACE}" || true
        exit 1
    fi
    echo "  Ready=${READY:-unknown} — waiting ${INTERVAL}s (${ELAPSED}/${TIMEOUT}s)"
    sleep $INTERVAL
    ELAPSED=$(( ELAPSED + INTERVAL ))
done

# Allow a settling period after ready
sleep 20

# ── Update openmetrics PMDA ConfigMap for this model's services ──────────────
# Both epp.url and vllm.url follow predictable naming from the LLMInferenceService name.
VLLM_SVC="${LLM_SERVICE_NAME}-kserve-workload-svc.${NAMESPACE}.svc.cluster.local"
EPP_SVC="${LLM_SERVICE_NAME}-epp-service.${NAMESPACE}.svc.cluster.local"
echo "Updating openmetrics URLs: vllm=${VLLM_SVC}:8000 epp=${EPP_SVC}:9090"
kubectl --kubeconfig="${KUBECONFIG}" patch configmap openmetrics-pmda-configmap \
    -n "${NAMESPACE}" --type=merge \
    -p "{\"data\":{
        \"vllm.url\":\"https://${VLLM_SVC}:8000/metrics\",
        \"epp.url\":\"http://${EPP_SVC}:9090/metrics\"
    }}"

# ── Restart PCP pod for a fresh archive ──────────────────────────────────────
echo "Restarting PCP pods for fresh archive..."
OLD_PCP_PODS=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
    -l app.kubernetes.io/name=pcp -o jsonpath='{.items[*].metadata.name}')
for pod in ${OLD_PCP_PODS}; do
    kubectl --kubeconfig="${KUBECONFIG}" delete pod -n "${NAMESPACE}" "${pod}" --wait=false 2>/dev/null
done
for pod in ${OLD_PCP_PODS}; do
    kubectl --kubeconfig="${KUBECONFIG}" wait --for=delete pod/"${pod}" -n "${NAMESPACE}" --timeout=60s 2>/dev/null || true
done
kubectl --kubeconfig="${KUBECONFIG}" wait --for=condition=ready pod \
    -l app.kubernetes.io/name=pcp -n "${NAMESPACE}" --timeout=600s

# Wait for pmlogger to be operational
NEW_PCP_PODS=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
    -l app.kubernetes.io/name=pcp -o jsonpath='{.items[*].metadata.name}')
for pod in ${NEW_PCP_PODS}; do
    for i in {1..30}; do
        if kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${pod}" -- \
            sh -c "pminfo -f pmcd.pmlogger.host 2>/dev/null | grep -q \"\$(hostname)\"" 2>/dev/null; then
            echo "  pmlogger operational in ${pod}"; break
        fi
        sleep 1
    done
done

# ── Detect guidellm pod ───────────────────────────────────────────────────────
GUIDELLM_POD=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
    -l app=guidellm --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "${GUIDELLM_POD}" ]; then
    echo "ERROR: No running guidellm pod found in namespace ${NAMESPACE}"
    exit 1
fi
echo "guidellm pod: ${GUIDELLM_POD}"

# ── Capture vLLM startup logs ─────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# Find the leader vLLM pod (LLMInferenceService creates LeaderWorkerSet pods)
VLLM_POD=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
    -l serving.kserve.io/inferenceservice="${LLM_SERVICE_NAME}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || \
    kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
    --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | \
    grep "${LLM_SERVICE_NAME}" | head -1 || echo "")

if [ -n "${VLLM_POD}" ]; then
    VLLM_LOG="${OUTPUT_DIR}/vllm-startup.log"
    kubectl --kubeconfig="${KUBECONFIG}" logs -n "${NAMESPACE}" "${VLLM_POD}" \
        --container=main > "${VLLM_LOG}" 2>&1 || true
    zstd -q -f --rm "${VLLM_LOG}" 2>/dev/null || true
fi

# ── Record run config ─────────────────────────────────────────────────────────
cat > "${OUTPUT_DIR}/benchmark-config.txt" <<EOF
Run ID: ${RUN_ID}
Date: $(date -u)
Target: ${TARGET}
Rate Type: ${RATE_TYPE}
Rate (Concurrency): ${RATE}
Max Seconds: ${MAX_SECONDS}
Prompt Tokens: ${PROMPT_TOKENS}
Output Tokens: ${OUTPUT_TOKENS}
Prefix Tokens: ${PREFIX_TOKENS}
Prefix Count: ${PREFIX_COUNT}
Turns: ${TURNS}
Hardware: ${HARDWARE}
Software: ${SOFTWARE}
Model: ${MODEL}
Model Name: ${MODEL_NAME}
LLMInferenceService: ${LLM_SERVICE_NAME}
Parameters: ${PARAMETERS}
Replicas: ${REPLICAS}
Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}
GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}
Num CPU Blocks: ${NUM_CPU_BLOCKS}
VLLM Additional Args: ${VLLM_ADDITIONAL_ARGS}
Namespace: ${NAMESPACE}
guidellm Pod: ${GUIDELLM_POD}
PCP Pods: ${NEW_PCP_PODS}
EOF

# ── Run guidellm benchmark ────────────────────────────────────────────────────
echo "Running guidellm benchmark..."
START_TIME=$(date +%s)

GUIDELLM_CMD="guidellm benchmark run \
    --target=\"${TARGET}\" \
    --rate-type=\"${RATE_TYPE}\" \
    --rate=\"${RATE}\" \
    --max-seconds=\"${MAX_SECONDS}\" \
    --random-seed=\"${RANDOM_SEED}\" \
    --data='{\"prompt_tokens\":${PROMPT_TOKENS},\"output_tokens\":${OUTPUT_TOKENS},\"prefix_tokens\":${PREFIX_TOKENS},\"turns\":${TURNS},\"prefix_count\":${PREFIX_COUNT}}' \
    --sample-requests=0 \
    --outputs=/models/benchmark.json"

# OPENAI_API_KEY is set to the pod's SA token — this satisfies the Kuadrant AuthPolicy
# which does a Kubernetes SubjectAccessReview using the bearer token.
# OPENAI_VERIFY_SSL=0 disables cert verification for the self-signed gateway cert.
kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
    sh -c 'SA_TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
    OPENAI_API_KEY="$SA_TOKEN" OPENAI_VERIFY_SSL=0 PYTHONHTTPSVERIFY=0 '"${GUIDELLM_CMD}"

END_TIME=$(date +%s)
echo "Benchmark duration: $(( END_TIME - START_TIME ))s"

# ── Collect guidellm results ──────────────────────────────────────────────────
echo "Downloading guidellm results..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/../../scripts/transfer-large-file-chunked.sh" \
    "${KUBECONFIG}" "${NAMESPACE}" "${GUIDELLM_POD}" \
    "/models/benchmark.json" "${OUTPUT_DIR}/guidellm-results.json"

if [ -s "${OUTPUT_DIR}/guidellm-results.json" ]; then
    zstd -q -f --rm "${OUTPUT_DIR}/guidellm-results.json"

    echo "Stripping request content from guidellm results..."
    python3 "${SCRIPT_DIR}/../../scripts/strip-guidellm-request-content.py" \
        "${OUTPUT_DIR}/guidellm-results.json.zst" || \
        echo "  Warning: strip failed (file may be corrupt), continuing"

    kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
        rm -f /models/benchmark.json 2>/dev/null || true
else
    echo "ERROR: Failed to retrieve guidellm results"; exit 1
fi

# ── Collect PCP archives ──────────────────────────────────────────────────────
echo "Collecting PCP archives..."
mkdir -p "${OUTPUT_DIR}/pcp-archives"

PCP_PODS=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
    -l app.kubernetes.io/name=pcp -o jsonpath='{.items[*].metadata.name}')

for PCP_POD in ${PCP_PODS}; do
    POD_NODE=$(kubectl --kubeconfig="${KUBECONFIG}" get pod -n "${NAMESPACE}" \
        "${PCP_POD}" -o jsonpath='{.spec.nodeName}')
    mkdir -p "${OUTPUT_DIR}/pcp-archives/${POD_NODE}"

    kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${PCP_POD}" -- \
        sh -c 'ls -1 /var/log/pcp/pmlogger/$(hostname)/[0-9]* 2>/dev/null | grep -E "\.[0-9]+$" | head -20' \
    | while read -r archive_path; do
        archive_file=$(basename "${archive_path}")
        timeout --signal=KILL 60 sh -c \
            "kubectl --kubeconfig='${KUBECONFIG}' exec -n '${NAMESPACE}' '${PCP_POD}' \
             -- cat '${archive_path}' > '${OUTPUT_DIR}/pcp-archives/${POD_NODE}/${archive_file}'" || \
            echo "  Warning: failed to copy ${archive_file}"
        archive_base="${archive_path%.[0-9]*}"
        archive_base_name=$(basename "${archive_base}")
        for ext in index meta; do
            timeout --signal=KILL 60 sh -c \
                "kubectl --kubeconfig='${KUBECONFIG}' exec -n '${NAMESPACE}' '${PCP_POD}' \
                 -- cat '${archive_base}.${ext}' > '${OUTPUT_DIR}/pcp-archives/${POD_NODE}/${archive_base_name}.${ext}'" 2>/dev/null || true
        done
    done
done

find "${OUTPUT_DIR}/pcp-archives" -type f \
    \( -name "*.meta" -o -name "*.index" -o -regex ".*\.[0-9][0-9]*$" \) \
    -print0 | while IFS= read -r -d '' f; do
    zstd -q --rm "$f" || true
done

echo "=========================================="
echo "Benchmark complete: ${RUN_ID}"
echo "Results: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"
echo "=========================================="
