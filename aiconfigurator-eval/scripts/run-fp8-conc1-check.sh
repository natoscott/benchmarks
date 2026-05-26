#!/usr/bin/env bash
# Quick concurrency=1 accuracy check for Qwen3-32B-FP8 configs.
# Validates whether 0.18.0 FP8 silicon data improved the conc=1 gaps
# (TTFT 1.49×, TPOT 1.7× vs AIC prediction previously observed).
# Each run is 120 seconds — total ~10 minutes for 5 configs.

set -euo pipefail

KUBECONFIG="${KUBECONFIG:-$HOME/psap/kubeconfig-psap-fire-athena}"
export KUBECONFIG

NS="${NS:-aiconfigurator}"
GUIDELLM_POD=$(kubectl get pod -n "$NS" -l app=guidellm -o jsonpath='{.items[0].metadata.name}')
OUTPUT_DIR="${OUTPUT_DIR:-results/fp8-conc1-check}"
LOG_FILE="${LOG_FILE:-/tmp/fp8-conc1-check.log}"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

mkdir -p "$OUTPUT_DIR"

run_benchmark() {
    local name="$1" target="$2" output_file="$3"
    echo ""
    echo "==> $name -> $output_file"

    if [ -f "${OUTPUT_DIR}/${output_file}.zst" ]; then
        echo "    Already done, skipping."
        return
    fi

    pid=$(kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
        bash -c "nohup guidellm benchmark run \
            --data '{\"prompt_tokens\":9000,\"output_tokens\":30}' \
            --profile throughput \
            --rate 1 \
            --backend openai_http \
            --target '${target}' \
            --random-seed 889 \
            --max-seconds 120 \
            --sample-requests 5 \
            --output-dir /models \
            --outputs '${output_file}' \
            > '/tmp/guidellm-${name}.log' 2>&1 & echo \$!")
    echo "    pid: $pid"

    elapsed=0
    while ! kubectl exec -n "$NS" "$GUIDELLM_POD" --request-timeout=15s -- \
            bash -c "test -f '/models/${output_file}'" 2>/dev/null; do
        sleep 15; elapsed=$((elapsed + 15))
        [ "$elapsed" -ge 240 ] && echo "    TIMEOUT" && break
    done

    bash "$(dirname "$0")/../../scripts/transfer-large-file-chunked.sh" \
        "$KUBECONFIG" "$NS" "$GUIDELLM_POD" \
        "/models/${output_file}" "${OUTPUT_DIR}/${output_file}"
    zstd --rm -f -q "${OUTPUT_DIR}/${output_file}"
    echo "    Saved: ${OUTPUT_DIR}/${output_file}.zst"
}

# Scale guidellm up
kubectl scale deployment/guidellm -n "$NS" --replicas=1
kubectl rollout status deployment/guidellm -n "$NS" --timeout=120s

TP4_TARGET="https://qwen3-32b-fp8-tp4-kserve-workload-svc.aiconfigurator.svc.cluster.local:8000"
TP1_TARGET="https://qwen3-32b-fp8-kserve-workload-svc.aiconfigurator.svc.cluster.local:8000"
DISAGG_TARGET="https://qwen3-32b-fp8-disagg-kserve-workload-svc.aiconfigurator.svc.cluster.local:8000"

echo "Apply TP=4 agg service first (AIC top-1)..."
kubectl apply -f manifests/llm-inference-service-qwen3-32b-fp8-tp4.yaml
echo "Waiting for TP=4 pods..."
elapsed=0
while [ "$(kubectl get pods -n "$NS" --field-selector=status.phase=Running -o name 2>/dev/null | grep -c '/qwen3-32b-fp8-tp4-')" -lt 1 ]; do
    sleep 15; elapsed=$((elapsed + 15)); [ "$elapsed" -ge 600 ] && echo "TIMEOUT" && exit 1
done
sleep 30

run_benchmark "32b-fp8-tp4-conc1" "$TP4_TARGET" "fp8-tp4-conc1.json"
kubectl delete -f manifests/llm-inference-service-qwen3-32b-fp8-tp4.yaml

echo ""
echo "Done. Results in $OUTPUT_DIR/"
