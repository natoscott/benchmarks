#!/bin/bash
# Single benchmark run: one model × one profile × one EPP config.
# Handles EPP config swapping, model scaling, guidellm execution, and artifact collection.
#
# Called by run-all-scenarios.sh with all variables exported.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHARED_SCRIPTS="${REPO_ROOT}/../scripts"
TRANSFER_SCRIPT="${SHARED_SCRIPTS}/transfer-large-file-chunked.sh"

export KUBECONFIG="${KUBECONFIG:-${HOME}/psap/kubeconfig-psap-janus}"
NAMESPACE="${NAMESPACE:-llm-d-nathans-epp-eval}"

# ── Model configuration ─────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B}"
LLM_SERVICE_NAME="${LLM_SERVICE_NAME:-qwen3-30b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
REPLICAS="${REPLICAS:-1}"

# ── EPP configuration ───────────────────────────────────────────────────────
EPP_CONFIG="${EPP_CONFIG:-prior-default}"
EPP_SCHEDULER_NAME="epp-scheduler-${EPP_CONFIG}"

# ── Profile configuration ──────────────────────────────────────────────────
PROFILE="${PROFILE:-multi-turn}"
PROFILE_FILE="${REPO_ROOT}/profiles/${PROFILE}.yaml"

# ── Cluster metadata ────────────────────────────────────────────────────────
HARDWARE="${HARDWARE:-2x8xH200}"
SOFTWARE="${SOFTWARE:-rhoai-3.5ea1}"

# ── Derived ──────────────────────────────────────────────────────────────────
RUN_ID="${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PROFILE}_${EPP_CONFIG}_replica${REPLICAS}"
OUTPUT_DIR="${REPO_ROOT}/results/${RUN_ID}"

echo "=========================================="
echo "Benchmark Run: ${RUN_ID}"
echo "  Model:       ${MODEL} (TP=${TENSOR_PARALLEL_SIZE}, replicas=${REPLICAS})"
echo "  Profile:     ${PROFILE}"
echo "  EPP Config:  ${EPP_CONFIG} (${EPP_SCHEDULER_NAME})"
echo "=========================================="

if [ -f "${OUTPUT_DIR}/benchmark-config.txt" ]; then
    echo "SKIPPING (already complete): ${OUTPUT_DIR}"
    exit 0
fi
mkdir -p "${OUTPUT_DIR}"

[ -f "${PROFILE_FILE}" ] || { echo "ERROR: Profile not found: ${PROFILE_FILE}"; exit 1; }

# ── Derive gateway target URL ────────────────────────────────────────────────
# Use the internal cluster service for the inference gateway.
# The external LB address in status.url is unreachable from inside the cluster.
GATEWAY_SVC="${GATEWAY_SVC:-openshift-ai-inference-data-science-gateway-class.openshift-ingress.svc.cluster.local}"
TARGET="${TARGET:-https://${GATEWAY_SVC}/${NAMESPACE}/${LLM_SERVICE_NAME}}"
echo "  Target:      ${TARGET}"

# ── 1. Swap EPP config on LLMInferenceService ───────────────────────────────
echo ""
echo "[1] Swapping EPP config to ${EPP_SCHEDULER_NAME}..."

# Preserve the RHOAI base template and router-route refs; only swap the EPP scheduler ref.
RHOAI_TEMPLATE="${RHOAI_TEMPLATE:-v3-5-0-ea-1-kserve-config-llm-template}"
RHOAI_ROUTE="${RHOAI_ROUTE:-v3-5-0-ea-1-kserve-config-llm-router-route}"

kubectl patch llminferenceservice "${LLM_SERVICE_NAME}" -n "${NAMESPACE}" \
    --type=json -p "[
  {\"op\":\"replace\",\"path\":\"/spec/baseRefs\",\"value\":[{\"name\":\"${RHOAI_TEMPLATE}\"},{\"name\":\"${EPP_SCHEDULER_NAME}\"},{\"name\":\"${RHOAI_ROUTE}\"}]}
]"
echo "  baseRefs set to [${RHOAI_TEMPLATE}, ${EPP_SCHEDULER_NAME}, ${RHOAI_ROUTE}]"

# ── 2. Scale model server ───────────────────────────────────────────────────
echo ""
echo "[2] Scaling LLMInferenceServices..."

ALL_SERVICES=$(kubectl get llminferenceservices -n "${NAMESPACE}" \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
for svc in ${ALL_SERVICES}; do
    if [ "${svc}" = "${LLM_SERVICE_NAME}" ]; then
        continue
    fi
    CURRENT=$(kubectl get llminferenceservice "${svc}" -n "${NAMESPACE}" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")
    if [ "${CURRENT}" != "0" ]; then
        echo "  Scaling down: ${svc} -> 0"
        kubectl patch llminferenceservice "${svc}" -n "${NAMESPACE}" \
            --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/replicas\",\"value\":0}]" 2>/dev/null || true
    fi
    kubectl scale deployment "${svc}-kserve-router-scheduler" \
        -n "${NAMESPACE}" --replicas=0 2>/dev/null || true
done

echo "  Scaling up: ${LLM_SERVICE_NAME} -> ${REPLICAS}"
kubectl patch llminferenceservice "${LLM_SERVICE_NAME}" -n "${NAMESPACE}" \
    --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/replicas\",\"value\":${REPLICAS}}]"

kubectl scale deployment "${LLM_SERVICE_NAME}-kserve-router-scheduler" \
    -n "${NAMESPACE}" --replicas=1 2>/dev/null || true

echo "  Waiting for LLMInferenceService to be Ready..."
TIMEOUT=1800; INTERVAL=15; ELAPSED=0
while true; do
    READY=$(kubectl get llminferenceservice "${LLM_SERVICE_NAME}" -n "${NAMESPACE}" \
        -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
    [[ "${READY}" == "True" ]] && { echo "  LLMInferenceService Ready"; break; }
    if [[ "${ELAPSED}" -ge "${TIMEOUT}" ]]; then
        echo "ERROR: Timeout waiting for LLMInferenceService after ${TIMEOUT}s"
        kubectl describe llminferenceservice "${LLM_SERVICE_NAME}" -n "${NAMESPACE}" || true
        exit 1
    fi
    echo "  Ready=${READY:-unknown} (${ELAPSED}/${TIMEOUT}s)"
    sleep "${INTERVAL}"; ELAPSED=$(( ELAPSED + INTERVAL ))
done

# ── 3. Wait for all workload pods, then verify gateway routing ──────────────
echo ""
echo "[3] Waiting for workload pods and gateway routing..."
GUIDELLM_POD=$(kubectl get pods -n "${NAMESPACE}" -l app=guidellm \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
[ -n "${GUIDELLM_POD}" ] || { echo "ERROR: No running guidellm pod"; exit 1; }

echo "  Waiting for ${REPLICAS} workload pod(s) to be Ready..."
for i in $(seq 1 360); do
    READY_COUNT=$(kubectl get pods -n "${NAMESPACE}" \
        -l "app.kubernetes.io/name=${LLM_SERVICE_NAME},kserve.io/component=workload" \
        -o jsonpath='{range .items[*]}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' 2>/dev/null \
        | grep -c True 2>/dev/null || true)
    READY_COUNT="${READY_COUNT:-0}"
    if [[ "${READY_COUNT}" -ge "${REPLICAS}" ]]; then
        echo "  ${READY_COUNT}/${REPLICAS} workload pods Ready"
        break
    fi
    echo "  ${READY_COUNT}/${REPLICAS} pods Ready (${i}/360, waiting 5s)"
    sleep 5
done

GATEWAY_OK=false
for i in $(seq 1 60); do
    HTTP=$(kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
        bash -c "SA_TOKEN=\$(cat /var/run/secrets/kubernetes.io/serviceaccount/token); \
        curl -sk -o /dev/null -w '%{http_code}' --max-time 5 \
        -H \"Authorization: Bearer \$SA_TOKEN\" \
        \"${TARGET}/health\"" 2>/dev/null || echo "000")
    if [ "${HTTP}" = "200" ]; then
        echo "  Gateway live (attempt ${i})"
        GATEWAY_OK=true; break
    fi
    echo "  HTTP ${HTTP}, waiting 5s... (${i}/60)"
    sleep 5
done
[ "${GATEWAY_OK}" = "true" ] || { echo "ERROR: Gateway not live after 5 minutes"; exit 1; }

# ── 4. Update PCP openmetrics URLs and restart ──────────────────────────────
echo ""
echo "[4] Restarting PCP for fresh archive..."

VLLM_SVC="${LLM_SERVICE_NAME}-kserve-workload-svc.${NAMESPACE}.svc.cluster.local"
EPP_SVC="${LLM_SERVICE_NAME}-epp-service.${NAMESPACE}.svc.cluster.local"

kubectl patch configmap openmetrics-pmda-configmap -n "${NAMESPACE}" \
    --type=merge -p "{\"data\":{\"vllm.url\":\"https://${VLLM_SVC}:8000/metrics\",\"epp.url\":\"http://${EPP_SVC}:9090/metrics\"}}"

OLD_PCP_PODS=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=pcp \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
for pod in ${OLD_PCP_PODS}; do
    kubectl exec -n "${NAMESPACE}" "${pod}" -- \
        sh -c 'rm -rf /var/log/pcp/pmlogger/$(hostname)/*' 2>/dev/null || true
    kubectl delete pod -n "${NAMESPACE}" "${pod}" --wait=false 2>/dev/null || true
done
for pod in ${OLD_PCP_PODS}; do
    kubectl wait --for=delete pod/"${pod}" -n "${NAMESPACE}" --timeout=60s 2>/dev/null || true
done

PCP_READY=""
for i in $(seq 1 120); do
    PCP_READY=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=pcp -o json 2>/dev/null | \
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
    if [ -n "${PCP_READY}" ]; then break; fi
    sleep 5
done
[ -n "${PCP_READY}" ] || { echo "ERROR: PCP pod not ready after 10 minutes"; exit 1; }
echo "  PCP pod ready: ${PCP_READY}"

for i in {1..30}; do
    if kubectl exec -n "${NAMESPACE}" "${PCP_READY}" -- \
        sh -c "pminfo -f pmcd.pmlogger.host 2>/dev/null | grep -q \"\$(hostname)\"" 2>/dev/null; then
        echo "  pmlogger operational"; break
    fi
    sleep 1
done

PCP_PROBES="openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL"
PCP_PROBES="${PCP_PROBES} openmetrics.vllm.vllm.num_requests_running"
PCP_PROBES="${PCP_PROBES} openmetrics.epp.workqueue_depth"
echo "  Waiting for metric sources..."
kubectl exec -n "${NAMESPACE}" "${PCP_READY}" -- \
    /opt/pcp-scripts/pcp-wait-and-restart-pmlogger.sh ${PCP_PROBES} || \
    echo "  Warning: some metric sources not available (check PCP logs)"

# ── 5. Run guidellm with profile ────────────────────────────────────────────
echo ""
echo "[5] Running guidellm (profile=${PROFILE})..."

GUIDELLM_BASE="guidellm run \
    --backend kind=openai_http,target=\"${TARGET}\" \
    --seed kind=static,value=42 \
    --disable-console-interactive"

RUN_SCRIPT='SA_TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
export OPENAI_API_KEY="$SA_TOKEN"
export OPENAI_VERIFY_SSL=0
export PYTHONHTTPSVERIFY=0
unset GUIDELLM_OUTPUT_DIR
rm -rf /models/benchmark-output
mkdir -p /models/benchmark-output
'

# Parse profile YAML and build guidellm CLI invocation.
# Each profile defines sweep parameters (streams or rates) — guidellm runs
# each level sequentially, producing one JSON output per level.
PROFILE_KIND=$(python3 -c "
import yaml, sys
with open('${PROFILE_FILE}') as f:
    p = yaml.safe_load(f)
print(p['profile']['kind'])
")

# Build guidellm commands from profile YAML. Outputs one shell command per
# sweep level (concurrency or rate). Uses JSON format for --data to support
# prefix_buckets and other nested fields. Resolves expressions like
# "2*concurrency" in prefix_count per sweep level.
GUIDELLM_COMMANDS=$(python3 -c "
import yaml, json, copy

with open('${PROFILE_FILE}') as f:
    p = yaml.safe_load(f)

profile = p['profile']
data = p['data']
constraint = p['constraint']
warmup = p.get('pre_warmup')
kind = profile['kind']

def build_data_json(data_cfg, concurrency=None):
    d = copy.deepcopy(data_cfg)
    d['kind'] = 'synthetic_text'
    if 'prefix_buckets' in d and concurrency is not None:
        for bucket in d['prefix_buckets']:
            for k, v in bucket.items():
                if isinstance(v, str) and 'concurrency' in v:
                    bucket[k] = int(eval(v.replace('concurrency', str(concurrency))))
    return json.dumps(d, separators=(',', ':'))

def build_constraint(constraint_cfg, concurrency=None):
    ckind = constraint_cfg['kind']
    if ckind == 'max_requests':
        expr = str(constraint_cfg['count'])
        if 'concurrency' in expr and concurrency is not None:
            count = int(eval(expr.replace('concurrency', str(concurrency))))
        else:
            count = int(expr)
        return f'kind=max_requests,count={count}'
    else:
        return f\"kind=max_duration,seconds={constraint_cfg['seconds']}\"

if kind == 'concurrent':
    for streams in profile['streams']:
        data_json = build_data_json(data, concurrency=streams)
        cargs = build_constraint(constraint, concurrency=streams)
        print(f'echo \"=== Concurrent streams={streams} ===\"')
        print(f'guidellm_data={repr(data_json)}')
        print(f'guidellm_run --data \"\$guidellm_data\" --profile kind=concurrent,streams={streams} --constraint {cargs} --output kind=json,path=/models/benchmark-output/concurrent-{streams}.json')

elif kind == 'poisson':
    data_json = build_data_json(data)
    if warmup:
        wrate = warmup['rate']
        wsecs = warmup['constraint']['seconds']
        print(f'echo \"=== Warmup: poisson rate={wrate} for {wsecs}s ===\"')
        print(f'guidellm_data={repr(data_json)}')
        print(f'guidellm_run --data \"\$guidellm_data\" --profile kind=poisson,rate={wrate} --constraint kind=max_duration,seconds={wsecs} --output kind=json,path=/models/benchmark-output/warmup.json')
    for rate in profile['rates']:
        cargs = build_constraint(constraint)
        print(f'echo \"=== Poisson rate={rate} ===\"')
        print(f'guidellm_data={repr(data_json)}')
        print(f'guidellm_run --data \"\$guidellm_data\" --profile kind=poisson,rate={rate} --constraint {cargs} --output kind=json,path=/models/benchmark-output/poisson-rate-{rate}.json')
else:
    import sys
    print(f'echo \"ERROR: Unknown profile kind {kind}\"', file=sys.stderr)
    sys.exit(1)
")

# Insert a shell function wrapper so the generated commands can call guidellm
# with the base args. The data arg is passed via a variable to avoid quoting issues.
RUN_SCRIPT+="
guidellm_run() { ${GUIDELLM_BASE} \"\$@\"; }
"

# Append each generated command
while IFS= read -r line; do
    RUN_SCRIPT+="${line}
"
done <<< "${GUIDELLM_COMMANDS}"

RUN_SCRIPT+='echo "=== Benchmark traffic complete ==="'

kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- bash -c "${RUN_SCRIPT}"

# ── 6. Collect guidellm results ─────────────────────────────────────────────
echo ""
echo "[6] Collecting guidellm results..."

# Strip request content in-pod before transfer to reduce 200MB+ to ~1MB.
# The guidellm container has Python + json but no zstd, so strip raw JSON
# then gzip via tar czf.
echo "  Stripping request content and packaging in pod..."
kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
    python3 -c '
import json, glob, os
os.chdir("/models/benchmark-output")
for f in sorted(glob.glob("*.json")):
    if "warmup" in f:
        os.remove(f)
        continue
    with open(f) as fh:
        data = json.load(fh)
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k not in ("request_args", "output")}
        if isinstance(obj, list):
            return [strip(i) for i in obj]
        return obj
    with open(f, "w") as fh:
        json.dump(strip(data), fh, separators=(",", ":"))
    print(f"  stripped {f}", flush=True)
'
kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
    bash -c 'cd /models/benchmark-output && tar czf /tmp/guidellm-results.tar.gz *.json'
echo "  Downloading guidellm-results.tar.gz..."
"${TRANSFER_SCRIPT}" \
    "${KUBECONFIG}" "${NAMESPACE}" "${GUIDELLM_POD}" \
    "/tmp/guidellm-results.tar.gz" "${OUTPUT_DIR}/guidellm-results.tar.gz" "$((256 * 1024))"
tar xzf "${OUTPUT_DIR}/guidellm-results.tar.gz" -C "${OUTPUT_DIR}" && rm -f "${OUTPUT_DIR}/guidellm-results.tar.gz"

# Compress locally with zstd (not available in guidellm container)
for f in "${OUTPUT_DIR}"/*.json; do
    [ -f "$f" ] || continue
    zstd -q -f --rm "$f" 2>/dev/null || true
done
echo "  Downloaded $(find "${OUTPUT_DIR}" -maxdepth 1 -name '*.json.zst' 2>/dev/null | wc -l) result files"

kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
    rm -rf /models/benchmark-output 2>/dev/null || true

# ── 7. Collect vLLM startup logs ────────────────────────────────────────────
echo "  Collecting vLLM startup logs..."
VLLM_POD=$(kubectl get pods -n "${NAMESPACE}" \
    -l "app.kubernetes.io/name=${LLM_SERVICE_NAME},kserve.io/component=workload" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -n "${VLLM_POD}" ]; then
    kubectl logs -n "${NAMESPACE}" "${VLLM_POD}" --container=main \
        > "${OUTPUT_DIR}/vllm-startup.log" 2>&1 || true
    zstd -q -f --rm "${OUTPUT_DIR}/vllm-startup.log" 2>/dev/null || true
fi

# ── 8. Collect PCP archives ─────────────────────────────────────────────────
echo "  Collecting PCP archives..."
mkdir -p "${OUTPUT_DIR}/pcp-archives"

PCP_PODS=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=pcp \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
for PCP_POD in ${PCP_PODS}; do
    kubectl exec -n "${NAMESPACE}" "${PCP_POD}" -- \
        systemctl stop pmlogger 2>/dev/null || true
done
sleep 2
for PCP_POD in ${PCP_PODS}; do
    POD_NODE=$(kubectl get pod -n "${NAMESPACE}" "${PCP_POD}" -o jsonpath='{.spec.nodeName}')
    ARCHIVE_DIR="${OUTPUT_DIR}/pcp-archives/${POD_NODE}"
    mkdir -p "${ARCHIVE_DIR}"

    echo "  Packaging PCP archives in pod..."
    kubectl exec -n "${NAMESPACE}" "${PCP_POD}" -- \
        bash -c 'cd /var/log/pcp/pmlogger/$(hostname) && \
                 for f in 2*; do [ -f "$f" ] && zstd -q --rm "$f"; done && \
                 tar cf /tmp/pcp-archives.tar *.zst'
    echo "  Downloading PCP archives..."
    "${TRANSFER_SCRIPT}" \
        "${KUBECONFIG}" "${NAMESPACE}" "${PCP_POD}" \
        "/tmp/pcp-archives.tar" "${ARCHIVE_DIR}/pcp-archives.tar" "$((256 * 1024))"
    tar xf "${ARCHIVE_DIR}/pcp-archives.tar" -C "${ARCHIVE_DIR}" && rm -f "${ARCHIVE_DIR}/pcp-archives.tar"
    echo "  Downloaded $(find "${ARCHIVE_DIR}" -maxdepth 1 -name '*.zst' 2>/dev/null | wc -l) archive files"
done

# ── 9. Capture EPP pod logs ─────────────────────────────────────────────────
echo "  Collecting EPP logs..."
EPP_POD=$(kubectl get pods -n "${NAMESPACE}" 2>/dev/null \
    | { grep "${LLM_SERVICE_NAME}.*epp" || true; } | awk '{print $1}' | head -1)
if [ -n "${EPP_POD}" ]; then
    kubectl logs -n "${NAMESPACE}" "${EPP_POD}" --container=main \
        > "${OUTPUT_DIR}/epp.log" 2>&1 || true
    zstd -q -f --rm "${OUTPUT_DIR}/epp.log" 2>/dev/null || true
fi

# ── 10. Record configuration ────────────────────────────────────────────────
cat > "${OUTPUT_DIR}/benchmark-config.txt" <<CFGEOF
Run ID: ${RUN_ID}
Date: $(date -u)
EPP Config: ${EPP_CONFIG}
EPP Scheduler: ${EPP_SCHEDULER_NAME}
Profile: ${PROFILE}
Profile File: ${PROFILE_FILE}
Target: ${TARGET}
Model: ${MODEL}
Model Name: ${MODEL_NAME}
LLMInferenceService: ${LLM_SERVICE_NAME}
Replicas: ${REPLICAS}
Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}
GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}
Hardware: ${HARDWARE}
Software: ${SOFTWARE}
guidellm Pod: ${GUIDELLM_POD}
PCP Pods: ${PCP_PODS}
Namespace: ${NAMESPACE}
CFGEOF

# ── 11. Log to MLflow (non-fatal) ────────────────────────────────────────────
MLFLOW_CONF="${MLFLOW_CONF:-${REPO_ROOT}/../mlflow.conf}"
if [ -f "${MLFLOW_CONF}" ] || [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    echo ""
    echo "  Logging results to MLflow..."
    python3 "${SHARED_SCRIPTS}/mlflow-log-run.py" "${OUTPUT_DIR}" || \
        echo "  Warning: MLflow logging failed (non-fatal, results still in ${OUTPUT_DIR})"
fi

echo ""
echo "  Waiting 10s before next run..."
sleep 10
echo "=========================================="
echo "Benchmark complete: ${RUN_ID}"
echo "Results: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"
echo "=========================================="
