#!/usr/bin/env bash
# Rate sweep benchmark runner for aiconfigurator-eval.
#
# Applies an LLMInferenceService, waits for it to be healthy, runs guidellm
# at each target rate, copies results locally, then deletes the service.
#
# Usage (set env vars before calling):
#
#   Qwen3-8B extended sweep (add rates 24, 32, 40 to existing results):
#     LLMISVC_MANIFEST=manifests/llm-inference-service-qwen3-8b.yaml \
#     LLMISVC_NAME=qwen3-8b \
#     TARGET=https://qwen3-8b-kserve-workload-svc.aiconfigurator.svc.cluster.local:8000 \
#     RATES="24 32 40" \
#     EXPECTED_PODS=8 \
#     OUTPUT_DIR=results/Qwen3-8B-9k-30-rate-sweep \
#     OUTPUT_PREFIX=guidellm-qwen3-8b-9k30 \
#     bash scripts/run-rate-sweep.sh 2>&1 | tee /tmp/sweep-qwen3-8b.log
#
#   Qwen3-32B-FP8 TP=4 sweep:
#     LLMISVC_MANIFEST=manifests/llm-inference-service-qwen3-32b-fp8-tp4.yaml \
#     LLMISVC_NAME=qwen3-32b-fp8-tp4 \
#     TARGET=https://qwen3-32b-fp8-tp4-kserve-workload-svc.aiconfigurator.svc.cluster.local:8000 \
#     RATES="1 2 4 6 8 12 16" \
#     EXPECTED_PODS=2 \
#     OUTPUT_DIR=results/Qwen3-32B-FP8-9k-30-tp4-rate-sweep \
#     OUTPUT_PREFIX=guidellm-qwen3-32b-fp8-tp4-9k30 \
#     bash scripts/run-rate-sweep.sh 2>&1 | tee /tmp/sweep-qwen3-32b-tp4.log

set -euo pipefail

KUBECONFIG="${KUBECONFIG:-$HOME/psap/kubeconfig-psap-fire-athena}"
export KUBECONFIG

NS="${NS:-aiconfigurator}"
LLMISVC_MANIFEST="${LLMISVC_MANIFEST:?set LLMISVC_MANIFEST to the path of the LLMInferenceService manifest}"
LLMISVC_NAME="${LLMISVC_NAME:?set LLMISVC_NAME to the metadata.name in the manifest}"
TARGET="${TARGET:?set TARGET to the kserve workload service URL}"
RATES="${RATES:-1 2 4 8 16}"
EXPECTED_PODS="${EXPECTED_PODS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for result files}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:?set OUTPUT_PREFIX for result filenames}"
MAX_SECONDS="${MAX_SECONDS:-120}"
RANDOM_SEED="${RANDOM_SEED:-889}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Rate sweep: $LLMISVC_NAME"
echo "Rates:      $RATES"
echo "Output:     $OUTPUT_DIR/${OUTPUT_PREFIX}-rate{N}.json.zst"
echo "Target:     $TARGET"
echo "=========================================="

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

# --- wait for all replicas to be Running ---
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

# --- rate sweep ---
for rate in $RATES; do
    output_file="${OUTPUT_PREFIX}-rate${rate}.json"
    echo ""
    echo "==> Rate ${rate} req/s -> ${output_file}"

    kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
        guidellm benchmark run \
            --data "{\"prompt_tokens\":9000,\"output_tokens\":30}" \
            --profile throughput \
            --rate "$rate" \
            --backend openai_http \
            --target "$TARGET" \
            --random-seed "$RANDOM_SEED" \
            --max-seconds "$MAX_SECONDS" \
            --output-dir /models \
            --outputs "${output_file}"

    echo "    Transferring from pod (chunked base64)..."
    bash "$(dirname "$0")/../../scripts/transfer-large-file-chunked.sh" \
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
echo "=========================================="
echo "Sweep complete."
echo "Results in: $OUTPUT_DIR/"
ls -lh "${OUTPUT_DIR}/${OUTPUT_PREFIX}"-*.json.zst 2>/dev/null || true
echo "=========================================="
