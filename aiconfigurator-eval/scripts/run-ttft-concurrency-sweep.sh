#!/usr/bin/env bash
# Additional TTFT concurrency sweep to fill out the b=2..256 table.
# Runs guidellm at fixed concurrency levels at ISL=9000, OSL=30 on Qwen3-8B.
# Used to determine the right cap for the _ttft_queuing_factor model.
#
# Usage:
#   bash scripts/run-ttft-concurrency-sweep.sh 2>&1 | tee /tmp/ttft-conc-sweep.log
#   tail -F /tmp/ttft-conc-sweep.log

set -euo pipefail

KUBECONFIG="${KUBECONFIG:-$HOME/psap/kubeconfig-psap-fire-athena}"
export KUBECONFIG

NS="aiconfigurator"
TARGET="https://qwen3-8b-kserve-workload-svc.${NS}.svc.cluster.local:8000"
ISL=9000
OSL=30
MAX_SECONDS=180        # 3 min per point — enough for stable mean at each concurrency
SAMPLE_REQUESTS=5
RANDOM_SEED=889

# New concurrency values to fill out the table
CONCURRENCIES="${CONCURRENCIES:-2 12 20 24 28 48 64 72 96 128 256}"

# Model to benchmark (override via env)
MODEL_TAG="${MODEL_TAG:-qwen3-8b}"

OUTPUT_DIR="results/ttft-concurrency-sweep-$(date +%Y%m%d)"
mkdir -p "$OUTPUT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

GUIDELLM_POD=$(kubectl get pod -n "$NS" -l app=guidellm -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
log "guidellm pod: $GUIDELLM_POD"

# Wait for model server
log "Waiting for $TARGET..."
elapsed=0
until kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=20s -- \
    bash -c "curl -sk '$TARGET/v1/models' 2>/dev/null | grep -q '\"data\"'" 2>/dev/null; do
    echo "  ${elapsed}s: not ready..."
    sleep 15; elapsed=$((elapsed+15))
    [ $elapsed -ge 900 ] && { log "Timeout"; exit 1; }
done
log "Model server ready. Waiting 30s to stabilise..."
sleep 30

total=$(echo "$CONCURRENCIES" | wc -w)
run=0
for cc in $CONCURRENCIES; do
    run=$((run+1))
    outfile="${MODEL_TAG}-ttft-isl${ISL}-cc${cc}.json"
    log "Run ${run}/${total}: concurrency=${cc} -> ${outfile}"

    if [ -f "${OUTPUT_DIR}/${outfile}.zst" ]; then
        log "  Already done, skipping."
        continue
    fi

    pid=$(kubectl exec -n "$NS" "$GUIDELLM_POD" -- bash -c "
nohup guidellm benchmark run \
    --data '{\"prompt_tokens\":${ISL},\"output_tokens\":${OSL}}' \
    --profile throughput \
    --rate '${cc}' \
    --backend openai_http \
    --target '${TARGET}' \
    --random-seed '${RANDOM_SEED}' \
    --max-seconds '${MAX_SECONDS}' \
    --sample-requests ${SAMPLE_REQUESTS} \
    --output-dir /models \
    --outputs '${outfile}' \
    > /models/ttft-${MODEL_TAG}-cc${cc}.log 2>&1 & echo \$!" 2>/dev/null)
    log "  pid: $pid"

    elapsed=0
    timeout=$((MAX_SECONDS + 120))
    while ! kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
            bash -c "test -f '/models/${outfile}'" 2>/dev/null; do
        sleep 15; elapsed=$((elapsed+15))
        [ $elapsed -ge $timeout ] && { log "  WARNING: timed out"; break; }
    done

    if kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
            bash -c "test -f '/models/${outfile}'" 2>/dev/null; then
        log "  Transferring..."
        bash /home/nathans/git/benchmarks/scripts/transfer-large-file-chunked.sh \
            "$KUBECONFIG" "$NS" "$GUIDELLM_POD" \
            "/models/${outfile}" "${OUTPUT_DIR}/${outfile}"
        zstd --rm -f -q "${OUTPUT_DIR}/${outfile}"
        log "  Saved: ${OUTPUT_DIR}/${outfile}.zst"
    fi
done

log "Sweep complete. Results in ${OUTPUT_DIR}/"
log "Delete model server: kubectl delete llminferenceservice qwen3-8b -n $NS"
