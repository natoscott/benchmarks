#!/usr/bin/env bash
# Mix-step latency triangulation sweep.
#
# Measures ITL at two OSL values (128 and 30) for overlapping (b, ISL) points.
# Combined with existing OSL=128 overhead study data, this provides two equations
# per (b, ISL) point to solve for actual mix_step_latency and genonly_step_latency:
#
#   ITL(OSL=128) = (mix_lat * n_mix + gen_lat * (128 - n_mix)) / 128
#   ITL(OSL=30)  = (mix_lat * n_mix + gen_lat * (30  - n_mix)) / 30
#
# This lets us determine whether AIC's mix step latency is overestimated and
# by how much — giving a principled replacement for the empirical -3 correction
# in base_backend.py.
#
# Phase 1: OSL=30 at ISL=512/1024/2048/4096, b=4/8/16 (12 runs, ~60 min)
#   Overlaps with existing OSL=128 data from results/overhead-sweep-20260526/
# Phase 2 (optional): OSL sweep at fixed b=8, ISL=1024 (5 runs, ~25 min)
#   Over-determined: OSL=10/20/30/60/128 — validates the triangulation result
#
# Usage:
#   PHASE=1 bash scripts/run-mix-step-sweep.sh
#   PHASE=2 bash scripts/run-mix-step-sweep.sh

set -euo pipefail

KUBECONFIG="${KUBECONFIG:-$HOME/psap/kubeconfig-psap-fire-athena}"
export KUBECONFIG

NS="${NS:-aiconfigurator}"
LLMISVC_MANIFEST="${LLMISVC_MANIFEST:-manifests/llm-inference-service-qwen3-8b-1rep.yaml}"
LLMISVC_NAME="${LLMISVC_NAME:-qwen3-8b}"
TARGET="${TARGET:-https://qwen3-8b-kserve-workload-svc.aiconfigurator.svc.cluster.local:8000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
RANDOM_SEED="${RANDOM_SEED:-889}"
MAX_SECONDS="${MAX_SECONDS:-300}"
PHASE="${PHASE:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-results/mix-step-sweep-$(date +%Y%m%d)}"
MODEL_TAG="${MODEL_TAG:-qwen3-8b}"
LOG_FILE="${LOG_FILE:-/tmp/mix-step-sweep.log}"

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE  (tail -F $LOG_FILE to monitor)"

if [ "$PHASE" = "1" ]; then
    # ISL values already measured at OSL=128 in overhead-sweep-20260526
    ISL_VALUES="512 1024 2048 4096"
    RATES="4 8 16"
    OSL_VALUES="30"
    echo "Phase 1: OSL=30 triangulation sweep"
    echo "  ISL: $ISL_VALUES"
    echo "  b:   $RATES"
    echo "  OSL: $OSL_VALUES (pairs with existing OSL=128 data)"
elif [ "$PHASE" = "2" ]; then
    # Fixed (b=8, ISL=1024), vary OSL — over-determined validation
    ISL_VALUES="1024"
    RATES="8"
    OSL_VALUES="10 20 30 60 128"
    echo "Phase 2: OSL sweep at (b=8, ISL=1024) for validation"
    echo "  ISL: $ISL_VALUES"
    echo "  b:   $RATES"
    echo "  OSL: $OSL_VALUES"
else
    echo "ERROR: PHASE must be 1 or 2"; exit 1
fi

total=$(echo "$ISL_VALUES" | wc -w)
total=$((total * $(echo "$RATES" | wc -w) * $(echo "$OSL_VALUES" | wc -w)))
echo "Total runs: $total"
echo "Output:     $OUTPUT_DIR/"
echo "Target:     $TARGET"
echo "=========================================================="

# --- ensure guidellm is running ---
kubectl scale deployment/guidellm -n "$NS" --replicas=1
kubectl rollout status deployment/guidellm -n "$NS" --timeout=120s
GUIDELLM_POD=$(kubectl get pod -n "$NS" -l app=guidellm \
    -o jsonpath='{.items[0].metadata.name}')
echo "    guidellm pod: $GUIDELLM_POD"

# --- apply model service ---
kubectl apply -f "$LLMISVC_MANIFEST"
echo "Waiting for pods..."
elapsed=0
while [ "$(kubectl get pods -n "$NS" --field-selector=status.phase=Running -o name 2>/dev/null \
           | grep -c "/$LLMISVC_NAME-" || true)" -lt 1 ]; do
    sleep 15; elapsed=$((elapsed + 15))
    [ "$elapsed" -ge "$HEALTH_TIMEOUT" ] && echo "TIMEOUT" && exit 1
done
until kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
    bash -c "curl -sk '$TARGET/v1/models' 2>/dev/null | grep -q '\"data\"'"; do
    sleep 15
done
echo "    Model ready. Waiting 30s to stabilise..."
sleep 30

# --- sweep ---
run_num=0
for isl in $ISL_VALUES; do
    for rate in $RATES; do
        for osl in $OSL_VALUES; do
            run_num=$((run_num + 1))
            output_file="${MODEL_TAG}-mixstep-isl${isl}-rate${rate}-osl${osl}.json"
            echo ""
            echo "==> Run ${run_num}/${total}: ISL=${isl} b=${rate} OSL=${osl} -> ${output_file}"

            if [ -f "${OUTPUT_DIR}/${output_file}.zst" ]; then
                echo "    Already done, skipping."
                continue
            fi

            pid=$(kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
                bash -c "nohup guidellm benchmark run \
                    --data '{\"prompt_tokens\":${isl},\"output_tokens\":${osl}}' \
                    --profile throughput \
                    --rate ${rate} \
                    --backend openai_http \
                    --target '${TARGET}' \
                    --random-seed '${RANDOM_SEED}' \
                    --max-seconds '${MAX_SECONDS}' \
                    --sample-requests 5 \
                    --output-dir /models \
                    --outputs '${output_file}' \
                    > '/tmp/guidellm-mixstep-${isl}-${rate}-${osl}.log' 2>&1 & echo \$!")
            echo "    pid: $pid"

            elapsed=0
            timeout=$((MAX_SECONDS + 300))
            while ! kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
                    bash -c "test -f '/models/${output_file}'" 2>/dev/null; do
                sleep 15; elapsed=$((elapsed + 15))
                if [ "$elapsed" -ge "$timeout" ]; then
                    echo "    WARNING: timed out, skipping"
                    break
                fi
            done

            if ! kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
                    bash -c "test -f '/models/${output_file}'" 2>/dev/null; then
                echo "    WARNING: output file not found, skipping transfer"
                continue
            fi

            bash "$(dirname "$0")/../../scripts/transfer-large-file-chunked.sh" \
                "$KUBECONFIG" "$NS" "$GUIDELLM_POD" \
                "/models/${output_file}" "${OUTPUT_DIR}/${output_file}"
            zstd --rm -f -q "${OUTPUT_DIR}/${output_file}"
            echo "    Saved: ${OUTPUT_DIR}/${output_file}.zst"
        done
    done
done

# --- cleanup ---
kubectl delete -f "$LLMISVC_MANIFEST"
kubectl scale deployment/guidellm -n "$NS" --replicas=0

echo ""
echo "=========================================================="
echo "Sweep complete. ${run_num} runs."
echo "Results in: $OUTPUT_DIR/"
ls -lh "${OUTPUT_DIR}/"*.json.zst 2>/dev/null || true
echo "=========================================================="
