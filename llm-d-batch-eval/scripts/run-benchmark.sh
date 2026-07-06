#!/bin/bash
# Single benchmark run: one scenario x one model configuration.
# Handles scenario setup, batch submission, interactive traffic, and artifact collection.
#
# Called by run-all-scenarios.sh with all variables exported.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHARED_SCRIPTS="${REPO_ROOT}/../scripts"
TRANSFER_SCRIPT="${SHARED_SCRIPTS}/transfer-large-file-chunked.sh"

export KUBECONFIG="${KUBECONFIG:-${HOME}/psap/kubeconfig-psap-fire-athena}"
NAMESPACE="${NAMESPACE:-llm-d-batch}"

# ── Model configuration ─────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
LLM_SERVICE_NAME="${LLM_SERVICE_NAME:-qwen3-8b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
REPLICAS="${REPLICAS:-1}"

# ── Scenario configuration ──────────────────────────────────────────────────
SCENARIO="${SCENARIO:-0}"
HARDWARE="${HARDWARE:-2x8xH200}"
SOFTWARE="${SOFTWARE:-rhoai-3.5ea1}"

# ── Traffic configuration ────────────────────────────────────────────────────
BURST_RATE="${BURST_RATE:-15}"
IDLE_RATE="${IDLE_RATE:-1}"
BURST_SECONDS="${BURST_SECONDS:-60}"
IDLE_SECONDS="${IDLE_SECONDS:-60}"
CYCLES="${CYCLES:-3}"
WARMUP_CYCLES="${WARMUP_CYCLES:-1}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
RANDOM_SEED="${RANDOM_SEED:-889}"

# ── Batch configuration ─────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-1000}"
NUM_JOBS="${NUM_JOBS:-3}"
NUM_SYSTEM_PROMPTS="${NUM_SYSTEM_PROMPTS:-32}"

# ── Derived ──────────────────────────────────────────────────────────────────
SCENARIO_NAMES=("interactive-only" "no-batch-gateway" "ungated" "aimd" "aimd-flow-control")
SCENARIO_NAME="${SCENARIO_NAMES[${SCENARIO}]}"
RUN_ID="${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${SCENARIO_NAME}_replica${REPLICAS}"
OUTPUT_DIR="${REPO_ROOT}/results/${RUN_ID}"

echo "=========================================="
echo "Benchmark Run: ${RUN_ID}"
echo "  Scenario: ${SCENARIO} (${SCENARIO_NAME})"
echo "  Model:    ${MODEL} (TP=${TENSOR_PARALLEL_SIZE}, replicas=${REPLICAS})"
echo "=========================================="

if [ -f "${OUTPUT_DIR}/benchmark-config.txt" ]; then
    echo "SKIPPING (already complete): ${OUTPUT_DIR}"
    exit 0
fi
mkdir -p "${OUTPUT_DIR}"

# ── Derive gateway target URL ────────────────────────────────────────────────
# RHOAI 3.5 EA: internal gateway service (openshift-ingress) exposes HTTPS/443
# only and does not terminate TLS for in-cluster callers — curl -sk still fails.
# The apps domain route works because HAProxy handles TLS termination.
APPS_DOMAIN=$(kubectl get ingresses.config.openshift.io cluster \
    -o jsonpath='{.spec.domain}' 2>/dev/null || echo "apps.example.com")
TARGET="${TARGET:-https://inference-gateway.${APPS_DOMAIN}/${NAMESPACE}/${LLM_SERVICE_NAME}}"
echo "  Target:   ${TARGET}"

# ── 1. Scale model server ────────────────────────────────────────────────────
echo ""
echo "[1] Scaling LLMInferenceServices..."

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

# ── 2. Configure and scale target model ──────────────────────────────────────
echo ""
echo "[2] Configuring ${LLM_SERVICE_NAME} (replicas=${REPLICAS})..."

VLLM_ARGS="--tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --max-num-seq 1024"
ARGS_JSON=$(printf '%s' "${VLLM_ARGS}" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")

for _retry in 1 2 3 4 5; do
    kubectl patch llminferenceservice "${LLM_SERVICE_NAME}" -n "${NAMESPACE}" \
        --type=json -p "[
      {\"op\":\"replace\",\"path\":\"/spec/replicas\",\"value\":${REPLICAS}},
      {\"op\":\"replace\",\"path\":\"/spec/template/containers/0/env/1/value\",\"value\":${ARGS_JSON}}
    ]" && break
    echo "  Patch failed (attempt ${_retry}/5), retrying in 10s..."
    sleep 10
done

# kserve doesn't always reconcile router-scheduler replicas automatically
kubectl scale deployment "${LLM_SERVICE_NAME}-kserve-router-scheduler" \
    -n "${NAMESPACE}" --replicas=1 2>/dev/null || true

echo "  Waiting for LLMInferenceService to be Ready..."
TIMEOUT=1800; INTERVAL=15; ELAPSED=0
while true; do
    READY=$(kubectl get llminferenceservice "${LLM_SERVICE_NAME}" -n "${NAMESPACE}" \
        -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
    [[ "${READY}" == "True" ]] && { echo "  LLMInferenceService Ready"; break; }
    (( ELAPSED >= TIMEOUT )) && {
        echo "ERROR: Timeout waiting for LLMInferenceService after ${TIMEOUT}s"
        kubectl describe llminferenceservice "${LLM_SERVICE_NAME}" -n "${NAMESPACE}" || true
        exit 1
    }
    echo "  Ready=${READY:-unknown} (${ELAPSED}/${TIMEOUT}s)"
    sleep $INTERVAL; ELAPSED=$(( ELAPSED + INTERVAL ))
done

# ── 3. Verify gateway routing ────────────────────────────────────────────────
echo ""
echo "[3] Verifying gateway routing (/health)..."
GUIDELLM_POD=$(kubectl get pods -n "${NAMESPACE}" -l app=guidellm \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
[ -n "${GUIDELLM_POD}" ] || { echo "ERROR: No running guidellm pod"; exit 1; }

GATEWAY_OK=false
for i in $(seq 1 180); do
    HTTP=$(kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
        bash -c "SA_TOKEN=\$(cat /var/run/secrets/kubernetes.io/serviceaccount/token); \
        curl -sk -o /dev/null -w '%{http_code}' --max-time 5 \
        -H \"Authorization: Bearer \$SA_TOKEN\" \
        \"${TARGET}/health\"" 2>/dev/null || echo "000")
    if [ "${HTTP}" = "200" ]; then
        echo "  Gateway live (attempt ${i})"
        GATEWAY_OK=true; break
    fi
    echo "  HTTP ${HTTP}, waiting 5s... (${i}/180)"
    sleep 5
done
[ "${GATEWAY_OK}" = "true" ] || { echo "ERROR: Gateway not live after 15 minutes"; exit 1; }

# ── 4. Deploy/reconfigure batch gateway (scenarios 2-4) ──────────────────────
echo ""
if [ "${SCENARIO}" -ge 2 ]; then
    echo "[4] Deploying batch gateway (scenario ${SCENARIO})..."
    HELM_VALUES="${REPO_ROOT}/helm-values"
    CHART_DIR="${HOME}/git/llm-d-batch-gateway/charts/batch-gateway"

    case "${SCENARIO}" in
        2) SCENARIO_VALUES="${HELM_VALUES}/scenario-2-ungated.yaml" ;;
        3) SCENARIO_VALUES="${HELM_VALUES}/scenario-3-aimd.yaml" ;;
        4) SCENARIO_VALUES="${HELM_VALUES}/scenario-4-aimd-flow-control.yaml" ;;
    esac

    helm upgrade --install batch-gateway "${CHART_DIR}" \
        -n "${NAMESPACE}" \
        -f "${HELM_VALUES}/common.yaml" \
        -f "${SCENARIO_VALUES}" \
        --set "processor.config.globalInferenceGateway.url=${TARGET}" \
        --timeout 300s

    kubectl rollout status deployment/batch-gateway-apiserver -n "${NAMESPACE}" --timeout=120s
    kubectl rollout status deployment/batch-gateway-processor -n "${NAMESPACE}" --timeout=120s

    echo "  Cleaning batch gateway state..."
    PG_PASS=$(kubectl get secret batch-gateway-secrets -n "${NAMESPACE}" \
        -o jsonpath='{.data.postgresql-password}' | base64 -d)
    kubectl run --rm -i pg-nuke --image=registry.redhat.io/rhel9/postgresql-16:latest \
        --restart=Never -n "${NAMESPACE}" --env="PGPASSWORD=${PG_PASS}" -- \
        psql -h postgresql -U batchgw -d batchgateway \
        -c "TRUNCATE batch_items, file_items CASCADE;" 2>/dev/null || true

    kubectl exec -n "${NAMESPACE}" deployment/valkey -- \
        valkey-cli FLUSHALL 2>/dev/null || true

    # Restart processor to pick up clean state after DB/queue flush
    kubectl rollout restart deployment/batch-gateway-processor -n "${NAMESPACE}"
    kubectl rollout status deployment/batch-gateway-processor -n "${NAMESPACE}" --timeout=120s

    kubectl apply -f "${REPO_ROOT}/manifests/monitoring/batch-gateway-processor-metrics-svc.yaml" 2>/dev/null || true

    echo "  Batch gateway ready"
else
    echo "[4] Skipping batch gateway (scenario ${SCENARIO})"
fi

# ── 5. Update PCP openmetrics URLs and restart ────────────────────────────────
echo ""
echo "[5] Restarting PCP for fresh archive..."

VLLM_SVC="${LLM_SERVICE_NAME}-kserve-workload-svc.${NAMESPACE}.svc.cluster.local"
EPP_SVC="${LLM_SERVICE_NAME}-epp-service.${NAMESPACE}.svc.cluster.local"

kubectl patch configmap openmetrics-pmda-configmap -n "${NAMESPACE}" \
    --type=merge -p "{\"data\":{\"vllm.url\":\"https://${VLLM_SVC}:8000/metrics\",\"epp.url\":\"http://${EPP_SVC}:9090/metrics\"}}"

# Clean old archives before restarting so step 10 only collects this run's data
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

# Wait for all metric sources to have live data, then restart pmlogger
# so pmlogconf picks up every source. Probes are scenario-dependent.
PCP_PROBES="openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL"
PCP_PROBES="${PCP_PROBES} openmetrics.vllm.vllm.num_requests_running"
PCP_PROBES="${PCP_PROBES} openmetrics.epp.workqueue_depth"
PCP_PROBES="${PCP_PROBES} postgresql.stat.database.numbackends"
if [ "${SCENARIO}" -ge 2 ]; then
    PCP_PROBES="${PCP_PROBES} openmetrics.batch_processor.processor_inflight_requests"
fi
echo "  Waiting for metric sources..."
kubectl exec -n "${NAMESPACE}" "${PCP_READY}" -- \
    /opt/pcp-scripts/pcp-wait-and-restart-pmlogger.sh ${PCP_PROBES} || \
    echo "  Warning: some metric sources not available (check PCP logs)"

# ── 6. Submit batch jobs (scenarios 2-4) ──────────────────────────────────────
echo ""
if [ "${SCENARIO}" -ge 2 ]; then
    echo "[6] Submitting batch jobs (${NUM_JOBS} x ${BATCH_SIZE} requests)..."

    # Copy helper scripts to pod (python3 /dev/stdin doesn't work with argparse)
    kubectl cp "${HOME}/git/llm-d-batch-gateway/benchmarks/generate_prompts.py" \
        "${NAMESPACE}/${GUIDELLM_POD}:/tmp/generate_prompts.py"
    kubectl cp "${SCRIPT_DIR}/submit-batch-job.py" \
        "${NAMESPACE}/${GUIDELLM_POD}:/tmp/submit-batch-job.py"

    COMPLETION_WINDOWS=("30m" "2h" "24h")
    JOB_NAMES=("job-a" "job-b" "job-c")
    BG_URL="http://batch-gateway-apiserver.${NAMESPACE}.svc.cluster.local:8000"

    for i in $(seq 0 $(( NUM_JOBS - 1 ))); do
        NAME="${JOB_NAMES[$i]}"
        WINDOW="${COMPLETION_WINDOWS[$i]}"

        echo "  Generating and submitting ${NAME} (window=${WINDOW}, ${BATCH_SIZE} requests)..."
        kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
            python3 /tmp/generate_prompts.py \
                --num-requests "${BATCH_SIZE}" \
                --num-system-prompts "${NUM_SYSTEM_PROMPTS}" \
                --model "${MODEL}" \
                --seed $((42 + i)) \
                --output "/tmp/${NAME}.jsonl"

        kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
            python3 /tmp/submit-batch-job.py --url "${BG_URL}" --file "/tmp/${NAME}.jsonl" --window "${WINDOW}"
    done
    echo "  Batch jobs submitted"

    # Start batch progress monitor in background
    echo "  Starting batch progress monitor..."
    kubectl cp "${SCRIPT_DIR}/monitor-batch-progress.py" "${NAMESPACE}/${GUIDELLM_POD}:/tmp/monitor-batch-progress.py"
    kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
        python3 /tmp/monitor-batch-progress.py --url "${BG_URL}" --output /tmp/batch-timeline.json \
        --interval 10 &
    MONITOR_PID=$!
else
    echo "[6] Skipping batch submission (scenario ${SCENARIO})"
    MONITOR_PID=""
fi

# ── 7. Run interactive traffic ────────────────────────────────────────────────
echo ""
echo "[7] Running interactive traffic (${CYCLES} cycles, burst@${BURST_RATE}/s idle@${IDLE_RATE}/s)..."

# guidellm v0.7.0 CLI syntax
GUIDELLM_BASE="guidellm run \
    --backend kind=openai_http,target=\"${TARGET}\" \
    --data kind=synthetic_text,prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS} \
    --seed kind=static,value=${RANDOM_SEED} \
    --disable-console-interactive"

CYCLE_SCRIPT='SA_TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
export OPENAI_API_KEY="$SA_TOKEN"
export OPENAI_VERIFY_SSL=0
export PYTHONHTTPSVERIFY=0
unset GUIDELLM_OUTPUT_DIR
mkdir -p /models/benchmark-output
'

for c in $(seq 1 "${CYCLES}"); do
    if [ "${c}" -le "${WARMUP_CYCLES}" ]; then
        SUFFIX="-warmup"
        LABEL=" [WARMUP]"
    else
        SUFFIX=""
        LABEL=""
    fi
    CYCLE_SCRIPT+="
echo \"=== Cycle ${c}: IDLE (${IDLE_RATE} req/s, ${IDLE_SECONDS}s)${LABEL} ===\"
${GUIDELLM_BASE} --profile kind=concurrent,streams=${IDLE_RATE} --constraint kind=max_duration,seconds=${IDLE_SECONDS} --output kind=json,path=/models/benchmark-output/idle-${c}${SUFFIX}.json
echo \"=== Cycle ${c}: BURST (${BURST_RATE} req/s, ${BURST_SECONDS}s)${LABEL} ===\"
${GUIDELLM_BASE} --profile kind=concurrent,streams=${BURST_RATE} --constraint kind=max_duration,seconds=${BURST_SECONDS} --output kind=json,path=/models/benchmark-output/burst-${c}${SUFFIX}.json
"
done

CYCLE_SCRIPT+='echo "=== Interactive traffic complete ==="'

kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- bash -c "${CYCLE_SCRIPT}" || \
    echo "  Warning: guidellm exited non-zero (partial results may still be usable)"

# Stop batch monitor and collect timeline
if [ -n "${MONITOR_PID:-}" ]; then
    kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true
    "${TRANSFER_SCRIPT}" \
        "${KUBECONFIG}" "${NAMESPACE}" "${GUIDELLM_POD}" \
        "/tmp/batch-timeline.json" "${OUTPUT_DIR}/batch-timeline.json" "$((256 * 1024))" 2>/dev/null || \
        echo "  Warning: failed to collect batch timeline"
    kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- rm -f /tmp/batch-timeline.json 2>/dev/null || true
fi

# ── 8. Collect guidellm results ──────────────────────────────────────────────
echo ""
echo "[8] Collecting guidellm results..."

# Create tarball in-pod, kubectl cp the single file, extract locally
echo "  Packaging guidellm results in pod..."
kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
    bash -c 'cd /models/benchmark-output && rm -f *-warmup* && tar czf /tmp/guidellm-results.tar.gz *.json'
echo "  Downloading guidellm-results.tar.gz..."
kubectl cp "${NAMESPACE}/${GUIDELLM_POD}:/tmp/guidellm-results.tar.gz" "${OUTPUT_DIR}/guidellm-results.tar.gz"
tar xzf "${OUTPUT_DIR}/guidellm-results.tar.gz" -C "${OUTPUT_DIR}" && rm -f "${OUTPUT_DIR}/guidellm-results.tar.gz"

for f in "${OUTPUT_DIR}"/*.json; do
    [ -f "$f" ] || continue
    zstd -q -f --rm "$f" 2>/dev/null || true
done
for f in "${OUTPUT_DIR}"/*.json.zst; do
    [ -f "$f" ] || continue
    python3 "${SHARED_SCRIPTS}/strip-guidellm-request-content.py" "$f" 2>/dev/null || true
done
echo "  Downloaded $(ls "${OUTPUT_DIR}"/*.json.zst 2>/dev/null | wc -l) result files"

kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
    rm -rf /models/benchmark-output 2>/dev/null || true

# Capture batch job final status (scenarios 2-4)
if [ "${SCENARIO}" -ge 2 ]; then
    echo "  Capturing batch job final status..."
    BG_URL="http://batch-gateway-apiserver.${NAMESPACE}.svc.cluster.local:8000"
    kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
        bash -c "curl -s -H 'Authorization: Bearer benchmark' '${BG_URL}/v1/batches?limit=10'" \
        > "${OUTPUT_DIR}/batch-final-status.json" 2>/dev/null || true
fi

# ── 9. Collect vLLM startup logs ─────────────────────────────────────────────
echo "  Collecting vLLM startup logs..."
VLLM_POD=$(kubectl get pods -n "${NAMESPACE}" \
    -l "app.kubernetes.io/name=${LLM_SERVICE_NAME},kserve.io/component=workload" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -n "${VLLM_POD}" ]; then
    kubectl logs -n "${NAMESPACE}" "${VLLM_POD}" --container=main \
        > "${OUTPUT_DIR}/vllm-startup.log" 2>&1 || true
    zstd -q -f --rm "${OUTPUT_DIR}/vllm-startup.log" 2>/dev/null || true
fi

# ── 10. Collect PCP archives ─────────────────────────────────────────────────
echo "  Collecting PCP archives..."
mkdir -p "${OUTPUT_DIR}/pcp-archives"

# Stop pmlogger, compress and tar archives in-pod, stream out.
PCP_PODS=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=pcp \
    -o jsonpath='{.items[*].metadata.name}')
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
                 for f in [0-9]*; do [ -f "$f" ] && zstd -q --rm "$f"; done && \
                 tar cf /tmp/pcp-archives.tar *.zst'
    echo "  Downloading PCP archives..."
    kubectl cp "${NAMESPACE}/${PCP_POD}:/tmp/pcp-archives.tar" "${ARCHIVE_DIR}/pcp-archives.tar"
    tar xf "${ARCHIVE_DIR}/pcp-archives.tar" -C "${ARCHIVE_DIR}" && rm -f "${ARCHIVE_DIR}/pcp-archives.tar"
    echo "  Downloaded $(ls "${ARCHIVE_DIR}"/*.zst 2>/dev/null | wc -l) archive files"
done

# ── 11. Record configuration ─────────────────────────────────────────────────
cat > "${OUTPUT_DIR}/benchmark-config.txt" <<CFGEOF
Run ID: ${RUN_ID}
Date: $(date -u)
Scenario: ${SCENARIO} (${SCENARIO_NAME})
Target: ${TARGET}
Model: ${MODEL}
Model Name: ${MODEL_NAME}
LLMInferenceService: ${LLM_SERVICE_NAME}
Replicas: ${REPLICAS}
Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}
GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}
Hardware: ${HARDWARE}
Software: ${SOFTWARE}
Burst Rate: ${BURST_RATE}
Idle Rate: ${IDLE_RATE}
Burst Seconds: ${BURST_SECONDS}
Idle Seconds: ${IDLE_SECONDS}
Cycles: ${CYCLES}
Warmup Cycles: ${WARMUP_CYCLES}
Prompt Tokens: ${PROMPT_TOKENS}
Output Tokens: ${OUTPUT_TOKENS}
Batch Size: ${BATCH_SIZE}
Num Jobs: ${NUM_JOBS}
guidellm Pod: ${GUIDELLM_POD}
PCP Pods: ${PCP_PODS}
Namespace: ${NAMESPACE}
CFGEOF

# ── 12. Log to MLflow (non-fatal) ────────────────────────────────────────────
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
