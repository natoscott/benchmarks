#!/usr/bin/env bash
# Poisson-arrival TTFT sweep for aiconfigurator Kingman queuing model validation.
#
# Runs guidellm with Poisson inter-arrival times (--rate-type poisson) at a
# range of arrival rates λ (req/s) to measure steady-state TTFT vs load.
# This gives TTFT(λ) data for validating the Kingman G/G/1 model:
#
#   TTFT ≈ T_prefill × (1 + ρ/(1-ρ) × (ca²+cs²)/2)
#   ρ = λ × T_prefill
#
# Unlike fixed-concurrency (throughput) sweeps, Poisson arrivals model
# production traffic and allow direct calibration of the queuing parameters.
#
# Usage:
#   # 8B model:
#   LLMISVC_MANIFEST=manifests/llm-inference-service-qwen3-8b.yaml \
#   LLMISVC_NAME=qwen3-8b \
#   TARGET=https://qwen3-8b-kserve-workload-svc.aiconfigurator.svc.cluster.local:8000 \
#   EXPECTED_PODS=8 \
#   bash scripts/run-poisson-ttft-sweep.sh 2>&1 | tee /tmp/poisson-8b.log
#
#   # 32B-FP8 model (run after 8B to reuse same GPU allocation):
#   LLMISVC_MANIFEST=manifests/llm-inference-service-qwen3-32b-fp8-tp4.yaml \
#   LLMISVC_NAME=qwen3-32b-fp8-tp4 \
#   TARGET=https://qwen3-32b-fp8-tp4-kserve-workload-svc.aiconfigurator.svc.cluster.local:8000 \
#   EXPECTED_PODS=2 \
#   bash scripts/run-poisson-ttft-sweep.sh 2>&1 | tee /tmp/poisson-32b.log

set -euo pipefail

KUBECONFIG="${KUBECONFIG:-$HOME/psap/kubeconfig-psap-fire-athena}"
export KUBECONFIG

NS="${NS:-aiconfigurator}"
LLMISVC_MANIFEST="${LLMISVC_MANIFEST:?set LLMISVC_MANIFEST to the path of the LLMInferenceService manifest}"
LLMISVC_NAME="${LLMISVC_NAME:?set LLMISVC_NAME to the metadata.name in the manifest}"
TARGET="${TARGET:?set TARGET to the kserve workload service URL}"
EXPECTED_PODS="${EXPECTED_PODS:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
RANDOM_SEED="${RANDOM_SEED:-889}"
SAMPLE_REQUESTS="${SAMPLE_REQUESTS:-5}"

# ISL/OSL fixed to match existing overhead-sweep-isl9000 data for direct comparison.
ISL="${ISL:-9000}"
OSL="${OSL:-30}"

# Arrival rates (req/s) to sweep under Poisson arrivals.
# For 8B at ISL=9000: saturation ≈ 4 req/s → sweep ρ from ~0.1 to ~0.9.
# For 32B-FP8 at ISL=9000: saturation ≈ 2-3 req/s → adjust RATES accordingly.
RATES="${RATES:-0.5 1.0 1.5 2.0 2.5 3.0 3.5}"

# Longer runs needed at low rates (λ=0.5 → 1 req every 2s → 300 reqs in 600s).
MAX_SECONDS="${MAX_SECONDS:-600}"

OUTPUT_DIR="${OUTPUT_DIR:-results/poisson-ttft-sweep-$(date +%Y%m%d)}"
MODEL_TAG="${MODEL_TAG:-$(basename "$LLMISVC_NAME")}"

mkdir -p "$OUTPUT_DIR"

LOG_FILE="${LOG_FILE:-/tmp/poisson-ttft-sweep.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE  (tail -F $LOG_FILE to monitor)"

total=$(echo "$RATES" | wc -w)

echo "=========================================================="
echo "Poisson TTFT sweep: $LLMISVC_NAME"
echo "ISL=${ISL}  OSL=${OSL}"
echo "Arrival rates (req/s): $RATES"
echo "Max seconds per run:   $MAX_SECONDS"
echo "Total runs: $total"
echo "Output:     $OUTPUT_DIR/"
echo "Target:     $TARGET"
echo ""
echo "Kingman model being validated:"
echo "  TTFT ≈ T_prefill × (1 + ρ/(1-ρ) × 0.5)"
echo "  ρ = λ × T_prefill  (λ = arrival rate, T_prefill from silicon model)"
echo "=========================================================="

# --- ensure guidellm pod is running ---
echo "==> Scaling guidellm to 1..."
kubectl scale deployment/guidellm -n "$NS" --replicas=1
kubectl rollout status deployment/guidellm -n "$NS" --timeout=120s

GUIDELLM_POD=$(kubectl get pod -n "$NS" -l app=guidellm \
    -o jsonpath='{.items[0].metadata.name}')
echo "    guidellm pod: $GUIDELLM_POD"

# --- apply LLMInferenceService ---
echo "==> Applying $LLMISVC_MANIFEST..."
kubectl apply -f "$LLMISVC_MANIFEST"

# --- wait for pods to be Running ---
echo "==> Waiting for $EXPECTED_PODS pod(s) to be Running..."
elapsed=0
while true; do
    running=$(kubectl get pods -n "$NS" \
        --field-selector=status.phase=Running \
        -o name 2>/dev/null \
        | grep -c "/$LLMISVC_NAME-" || true)
    echo "    ${elapsed}s: $running/$EXPECTED_PODS pods running"
    [ "$running" -ge "$EXPECTED_PODS" ] && break
    sleep 15
    elapsed=$((elapsed + 15))
    if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
        echo "ERROR: timed out waiting for pods"
        exit 1
    fi
done

# --- wait for vLLM health endpoint ---
echo "==> Waiting for $TARGET to respond..."
elapsed=0
until kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=20s -- \
    bash -c "curl -sk '$TARGET/v1/models' 2>/dev/null | grep -q '\"data\"'" ; do
    echo "    ${elapsed}s: not ready..."
    sleep 15
    elapsed=$((elapsed + 15))
    if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
        echo "ERROR: timed out waiting for $TARGET"
        exit 1
    fi
done
echo "    Model server ready. Waiting 30s for all replicas to stabilise..."
sleep 30

# --- sweep ---
run_num=0
for rate in $RATES; do
    run_num=$((run_num + 1))
    # Encode rate as integer-friendly tag (e.g. 1.5 -> rate1p5)
    rate_tag=$(echo "$rate" | sed 's/\./p/')
    output_file="${MODEL_TAG}-poisson-isl${ISL}-rate${rate_tag}.json"
    log_file="/tmp/guidellm-poisson-isl${ISL}-rate${rate_tag}.log"

    echo ""
    echo "==> Run ${run_num}/${total}: λ=${rate} req/s (Poisson) -> ${output_file}"

    if [ -f "${OUTPUT_DIR}/${output_file}.zst" ]; then
        echo "    Already done, skipping."
        continue
    fi

    pid=$(kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
        bash -c "nohup guidellm benchmark run \
            --data '{\"prompt_tokens\":${ISL},\"output_tokens\":${OSL}}' \
            --rate-type poisson \
            --rate '${rate}' \
            --backend openai_http \
            --target '${TARGET}' \
            --random-seed '${RANDOM_SEED}' \
            --max-seconds '${MAX_SECONDS}' \
            --sample-requests ${SAMPLE_REQUESTS} \
            --output-dir /models \
            --outputs '${output_file}' \
            > '${log_file}' 2>&1 & echo \$!")
    echo "    pid: $pid"

    elapsed=0
    timeout=$((MAX_SECONDS + 300))
    while ! kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
            bash -c "test -f '/models/${output_file}'" 2>/dev/null; do
        sleep 15
        elapsed=$((elapsed + 15))
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "    WARNING: run timed out after ${elapsed}s, skipping"
            break
        fi
    done

    if ! kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
            bash -c "test -f '/models/${output_file}'" 2>/dev/null; then
        echo "    WARNING: output file not found, skipping transfer"
        kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
            tail -5 "${log_file}" 2>/dev/null || true
        continue
    fi

    echo "    Transferring result (chunked base64)..."
    bash /home/nathans/git/benchmarks/scripts/transfer-large-file-chunked.sh \
        "$KUBECONFIG" "$NS" "$GUIDELLM_POD" \
        "/models/${output_file}" "${OUTPUT_DIR}/${output_file}"

    echo "    Compressing..."
    zstd --rm -f -q "${OUTPUT_DIR}/${output_file}"
    echo "    Saved: ${OUTPUT_DIR}/${output_file}.zst"
done

# --- delete LLMInferenceService ---
echo ""
echo "==> Deleting $LLMISVC_NAME..."
kubectl delete -f "$LLMISVC_MANIFEST"

echo ""
echo "=========================================================="
echo "Sweep complete. ${run_num} runs."
echo "Results in: $OUTPUT_DIR/"
ls -lh "${OUTPUT_DIR}/"*.json.zst 2>/dev/null || true
echo ""
echo "Next: python3 scripts/analyse-poisson-ttft.py --results-dir ${OUTPUT_DIR}"
echo "=========================================================="
