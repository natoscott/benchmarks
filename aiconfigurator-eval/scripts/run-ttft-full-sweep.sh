#!/usr/bin/env bash
# Full TTFT concurrency sweep across three models.
# Re-runs the original b=1..64 data plus new b=2..256 points
# using a consistent methodology, replacing the old overhead-sweep-isl9000 data.
#
# Usage:
#   nohup bash scripts/run-ttft-full-sweep.sh >> /tmp/ttft-full-sweep.log 2>&1 &
#   tail -F /tmp/ttft-full-sweep.log

set -euo pipefail

KUBECONFIG="${KUBECONFIG:-$HOME/psap/kubeconfig-psap-fire-athena}"
export KUBECONFIG

NS="aiconfigurator"
ISL=9000
OSL=30
MAX_SECONDS=300    # 5 min per point for stable statistics
SAMPLE_REQUESTS=5
RANDOM_SEED=889
HEALTH_TIMEOUT=900
OUTPUT_DIR="results/ttft-full-sweep-$(date +%Y%m%d)"

mkdir -p "$OUTPUT_DIR"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

GUIDELLM_POD=$(kubectl get pod -n "$NS" -l app=guidellm \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
log "guidellm pod: $GUIDELLM_POD"

wait_for_model() {
    local target="$1"
    local elapsed=0
    log "Waiting for $target..."
    until kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=20s -- \
        bash -c "curl -sk '$target/v1/models' 2>/dev/null | grep -q '\"data\"'" 2>/dev/null; do
        echo "  ${elapsed}s: not ready..."
        sleep 15; elapsed=$((elapsed+15))
        [ $elapsed -ge $HEALTH_TIMEOUT ] && { log "Timeout waiting for model"; exit 1; }
    done
    log "Ready. Waiting 30s to stabilise..."
    sleep 30
}

run_point() {
    local tag="$1" target="$2" cc="$3"
    local outfile="${tag}-ttft-isl${ISL}-cc${cc}.json"

    if [ -f "${OUTPUT_DIR}/${outfile}.zst" ]; then
        log "  cc=${cc}: already done, skipping."
        return
    fi

    log "  cc=${cc} -> ${outfile}"
    pid=$(kubectl exec -n "$NS" "$GUIDELLM_POD" -- bash -c "
nohup guidellm benchmark run \
    --data '{\"prompt_tokens\":${ISL},\"output_tokens\":${OSL}}' \
    --profile throughput \
    --rate '${cc}' \
    --backend openai_http \
    --target '${target}' \
    --random-seed '${RANDOM_SEED}' \
    --max-seconds '${MAX_SECONDS}' \
    --sample-requests ${SAMPLE_REQUESTS} \
    --output-dir /models \
    --outputs '${outfile}' \
    > /models/${tag}-cc${cc}.log 2>&1 & echo \$!" 2>/dev/null)

    elapsed=0; timeout=$((MAX_SECONDS + 120))
    while ! kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
            bash -c "test -f '/models/${outfile}'" 2>/dev/null; do
        sleep 15; elapsed=$((elapsed+15))
        [ $elapsed -ge $timeout ] && { log "  WARNING: timed out at cc=${cc}"; return; }
    done

    bash /home/nathans/git/benchmarks/scripts/transfer-large-file-chunked.sh \
        "$KUBECONFIG" "$NS" "$GUIDELLM_POD" \
        "/models/${outfile}" "${OUTPUT_DIR}/${outfile}" "$((200 * 1024))"
    zstd --rm -f -q "${OUTPUT_DIR}/${outfile}"
    log "  Saved: ${OUTPUT_DIR}/${outfile}.zst"
}

run_model() {
    local tag="$1" manifest="$2" name="$3" target="$4" expected_pods="$5"
    shift 5; local concurrencies="$*"

    log ""
    log "======== ${tag} ========"

    # Skip entire model if all result files already exist
    local all_done=true
    for cc in $concurrencies; do
        local outfile="${tag}-ttft-isl${ISL}-cc${cc}.json"
        [ -f "${OUTPUT_DIR}/${outfile}.zst" ] || { all_done=false; break; }
    done
    if $all_done; then
        log "  All ${tag} results already collected, skipping model deployment."
        return
    fi

    kubectl apply -f "$manifest" 2>/dev/null
    local elapsed=0
    while true; do
        local running=$(kubectl get pods -n "$NS" \
            --field-selector=status.phase=Running -o name 2>/dev/null \
            | grep -c "/$name-" || true)
        [ "$running" -ge "$expected_pods" ] && break
        echo "  ${elapsed}s: ${running}/${expected_pods} pods running..."
        sleep 15; elapsed=$((elapsed+15))
        [ $elapsed -ge $HEALTH_TIMEOUT ] && { log "Timeout waiting for pods"; exit 1; }
    done
    wait_for_model "$target"

    for cc in $concurrencies; do
        run_point "$tag" "$target" "$cc"
    done

    log "Deleting ${name}..."
    kubectl delete -f "$manifest" 2>/dev/null
    sleep 10
}

# All concurrency points: original (1,4,8,16,32,64) + new (2,12,20,24,28,48,72,96,128,256)
ALL_CC="1 2 4 8 12 16 20 24 28 32 48 64 72 96 128 256"
# 32B-FP8 saturates much earlier — cap at 64
FP8_CC="1 2 4 8 12 16 20 24 28 32 48 64"

run_model "qwen3-8b" \
    "manifests/llm-inference-service-qwen3-8b.yaml" \
    "qwen3-8b-kserve" \
    "https://qwen3-8b-kserve-workload-svc.${NS}.svc.cluster.local:8000" \
    8 $ALL_CC

run_model "qwen3-14b" \
    "manifests/llm-inference-service-qwen3-14b.yaml" \
    "qwen3-14b-kserve" \
    "https://qwen3-14b-kserve-workload-svc.${NS}.svc.cluster.local:8000" \
    8 $ALL_CC

run_model "qwen3-32b-fp8-tp4" \
    "manifests/llm-inference-service-qwen3-32b-fp8-tp4.yaml" \
    "qwen3-32b-fp8-tp4-kserve" \
    "https://qwen3-32b-fp8-tp4-kserve-workload-svc.${NS}.svc.cluster.local:8000" \
    2 $FP8_CC

kubectl scale deployment/guidellm -n "$NS" --replicas=0 2>/dev/null
log ""
log "All sweeps complete. Results in ${OUTPUT_DIR}/"
log "Analyse: python3 scripts/analyse-ttft-sweep.py --results-dir ${OUTPUT_DIR}"
