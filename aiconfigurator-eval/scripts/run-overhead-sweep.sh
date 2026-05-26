#!/usr/bin/env bash
# Decode-step overhead characterisation sweep for aiconfigurator.
#
# Runs guidellm at a grid of (ISL, concurrency) combinations with fixed OSL
# to collect ITL data for fitting the decode-step overhead model:
#
#   overhead(b, ISL) = alpha * b + beta * (b * ISL) + gamma
#
# where overhead = ITL_measured - TPOT_predicted_by_AIC.
#
# Each run holds concurrency fixed (throughput mode) so the actual batch size
# seen by the model server equals the requested rate. Results include
# inter_token_latency_ms and request_concurrency, sufficient for model fitting.
#
# Usage:
#   LLMISVC_MANIFEST=manifests/llm-inference-service-qwen3-8b.yaml \
#   LLMISVC_NAME=qwen3-8b \
#   TARGET=https://qwen3-8b-kserve-workload-svc.aiconfigurator.svc.cluster.local:8000 \
#   EXPECTED_PODS=8 \
#   bash scripts/run-overhead-sweep.sh 2>&1 | tee /tmp/overhead-sweep.log

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
# Save only 20 sample requests per status — sufficient for sanity checks while
# keeping result files small. Aggregated metrics (ITL, concurrency etc.) are
# computed across all requests regardless of this setting.
SAMPLE_REQUESTS="${SAMPLE_REQUESTS:-5}"

# Sweep dimensions
ISL_VALUES="${ISL_VALUES:-64 128 256 512 1024 2048 4096 8192}"
RATES="${RATES:-4 8 16 32 64}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"

# Per-run duration. Longer for high ISL to ensure steady-state ITL.
# At ISL=8192 with rate=64, TTFT can be several seconds so we allow more
# warmup time. The guidellm --max-seconds countdown begins after warmup.
MAX_SECONDS="${MAX_SECONDS:-300}"

OUTPUT_DIR="${OUTPUT_DIR:-results/overhead-sweep-$(date +%Y%m%d)}"
MODEL_TAG="${MODEL_TAG:-qwen3-8b}"

mkdir -p "$OUTPUT_DIR"

# All output goes to both terminal and log file. Tail the log to monitor:
#   tail -F /tmp/overhead-sweep.log
LOG_FILE="${LOG_FILE:-/tmp/overhead-sweep.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE  (tail -F $LOG_FILE to monitor)"

total_runs=$(echo "$ISL_VALUES" | wc -w)
total_rates=$(echo "$RATES" | wc -w)
total=$((total_runs * total_rates))

echo "=========================================================="
echo "Decode-step overhead sweep: $LLMISVC_NAME"
echo "ISL values:  $ISL_VALUES"
echo "Rates (b):   $RATES"
echo "OSL (fixed): $OUTPUT_TOKENS"
echo "Max seconds: $MAX_SECONDS per run"
echo "Total runs:  $total"
echo "Output:      $OUTPUT_DIR/"
echo "Target:      $TARGET"
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
until kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
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
for isl in $ISL_VALUES; do
    for rate in $RATES; do
        run_num=$((run_num + 1))
        output_file="${MODEL_TAG}-overhead-isl${isl}-rate${rate}.json"
        log_file="/tmp/guidellm-overhead-isl${isl}-rate${rate}.log"

        echo ""
        echo "==> Run ${run_num}/${total}: ISL=${isl} rate=${rate} -> ${output_file}"

        # Skip if already downloaded locally
        if [ -f "${OUTPUT_DIR}/${output_file}.zst" ]; then
            echo "    Already done, skipping."
            continue
        fi

        # Launch with nohup so the process survives API server connectivity drops.
        pid=$(kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
            bash -c "nohup guidellm benchmark run \
                --data '{\"prompt_tokens\":${isl},\"output_tokens\":${OUTPUT_TOKENS}}' \
                --profile throughput \
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

        # Poll until the output file appears — avoids dependency on pgrep.
        elapsed=0
        timeout=$((MAX_SECONDS + 300))   # allow headroom beyond max-seconds
        while ! kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
                bash -c "test -f '/models/${output_file}'" 2>/dev/null; do
            sleep 15
            elapsed=$((elapsed + 15))
            if [ "$elapsed" -ge "$timeout" ]; then
                echo "    WARNING: run timed out after ${elapsed}s, skipping"
                break
            fi
        done

        # Check output file exists before transferring.
        if ! kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
                bash -c "test -f '/models/${output_file}'" 2>/dev/null; then
            echo "    WARNING: output file not found, skipping transfer"
            echo "    Last log lines:"
            kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
                tail -5 "${log_file}" 2>/dev/null || true
            continue
        fi

        echo "    Transferring result (chunked base64)..."
        bash "$(dirname "$0")/../../scripts/transfer-large-file-chunked.sh" \
            "$KUBECONFIG" "$NS" "$GUIDELLM_POD" \
            "/models/${output_file}" "${OUTPUT_DIR}/${output_file}"

        echo "    Compressing..."
        zstd --rm -f -q "${OUTPUT_DIR}/${output_file}"

        echo "    Saved: ${OUTPUT_DIR}/${output_file}.zst"
    done
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
echo "=========================================================="
