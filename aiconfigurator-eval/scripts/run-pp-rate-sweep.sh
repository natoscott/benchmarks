#!/usr/bin/env bash
# PP validation rate sweep — deploy one LLMInferenceService, run guidellm
# at multiple concurrency levels, collect results, tear down.
#
# Uses a ConfigMap-mounted script (/scripts/run-bench.sh) in the guidellm pod
# to avoid heredoc quoting issues with kubectl exec.
#
# Required env vars:
#   LLMISVC_MANIFEST  — path to LLMInferenceService manifest
#   LLMISVC_NAME      — metadata.name in the manifest
#   MODEL_NAME        — HuggingFace model name for the API (e.g. "Qwen/Qwen3-0.6B")
#   CONCURRENCIES     — space-separated concurrency levels (e.g. "1 2 4 8 16")
#   OUTPUT_DIR        — directory for result files
#   OUTPUT_PREFIX     — filename prefix for results

set -euo pipefail

KUBECONFIG="${KUBECONFIG:?set KUBECONFIG to the cluster kubeconfig path}"
export KUBECONFIG

NS="${NS:-aiconfigurator}"
LLMISVC_MANIFEST="${LLMISVC_MANIFEST:?set LLMISVC_MANIFEST}"
LLMISVC_NAME="${LLMISVC_NAME:?set LLMISVC_NAME}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to the HuggingFace model name}"
CONCURRENCIES="${CONCURRENCIES:?set CONCURRENCIES}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:?set OUTPUT_PREFIX}"
MAX_SECONDS="${MAX_SECONDS:-120}"
RANDOM_SEED="${RANDOM_SEED:-889}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "PP validation sweep: $LLMISVC_NAME"
echo "Model:          $MODEL_NAME"
echo "Concurrencies:  $CONCURRENCIES"
echo "ISL=$PROMPT_TOKENS OSL=$OUTPUT_TOKENS"
echo "Output:         $OUTPUT_DIR/"
echo "=========================================="

# --- ensure guidellm pod is running ---
echo "==> Ensuring guidellm is running..."
kubectl scale deployment/guidellm -n "$NS" --replicas=1 2>/dev/null || true
kubectl rollout status deployment/guidellm -n "$NS" --timeout=120s

GUIDELLM_POD=$(kubectl get pod -n "$NS" -l app=guidellm \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}')
echo "    guidellm pod: $GUIDELLM_POD"

# --- apply LLMInferenceService ---
echo "==> Applying $LLMISVC_MANIFEST..."
kubectl apply -f "$LLMISVC_MANIFEST"

# --- wait for LLMInferenceService Ready ---
echo "==> Waiting for LLMInferenceService to be Ready..."
elapsed=0
while true; do
    ready=$(kubectl get llminferenceservice "$LLMISVC_NAME" -n "$NS" \
        -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
    echo "    ${elapsed}s: Ready=${ready:-unknown}"
    [ "$ready" = "True" ] && break
    sleep 15
    elapsed=$((elapsed + 15))
    if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
        echo "ERROR: timed out waiting for LLMInferenceService"
        kubectl delete -f "$LLMISVC_MANIFEST" 2>/dev/null || true
        exit 1
    fi
done

# --- derive target URL ---
# Target the workload service directly (vLLM endpoint) to avoid gateway
# routing issues. The workload service name follows the kserve convention.
TARGET="https://${LLMISVC_NAME}-kserve-workload-svc.${NS}.svc.cluster.local:8000"
echo "    Gateway target: $TARGET"

# --- wait for model health ---
echo "==> Waiting for model server..."
elapsed=0
until kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
    bash -c "curl -sk -o /dev/null -w '%{http_code}' --max-time 5 \
    '${TARGET}/health'" 2>/dev/null | grep -q 200; do
    echo "    ${elapsed}s: not ready..."
    sleep 15
    elapsed=$((elapsed + 15))
    if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
        echo "ERROR: model server not live after ${HEALTH_TIMEOUT}s"
        kubectl delete -f "$LLMISVC_MANIFEST" 2>/dev/null || true
        exit 1
    fi
done
echo "    Model server ready. Waiting 30s to stabilise..."
sleep 30

# --- ensure benchmark output dir exists in pod ---
kubectl exec -n "$NS" "$GUIDELLM_POD" -- mkdir -p /models/benchmark-output

# --- concurrency sweep using ConfigMap-mounted script ---
for conc in $CONCURRENCIES; do
    output_file="${OUTPUT_PREFIX}-conc${conc}.json"
    echo ""
    echo "==> Concurrency ${conc} -> ${output_file}"

    kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
        bash -c "SA_TOKEN=\$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
export OPENAI_API_KEY=\"\$SA_TOKEN\"
export OPENAI_VERIFY_SSL=0
export PYTHONHTTPSVERIFY=0
mkdir -p /models/benchmark-output
guidellm run \
    --backend kind=openai_http,target=\"${TARGET}\" \
    --data kind=synthetic_text,prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS} \
    --seed kind=static,value=${RANDOM_SEED} \
    --profile kind=concurrent,streams=${conc} \
    --constraint kind=max_duration,seconds=${MAX_SECONDS} \
    --output kind=json,path=/models/benchmark-output/${output_file} \
    --disable-console-interactive"
done

# --- collect: strip in-pod (97% size reduction), tar, transfer, extract ---
echo ""
echo "==> Collecting results..."

# Strip request content in-pod before transferring (reduces ~40MB to ~1MB)
kubectl exec -n "$NS" "$GUIDELLM_POD" -- python3 -c "
import json, glob, os
for f in sorted(glob.glob('/models/benchmark-output/*.json')):
    with open(f) as fh:
        d = json.load(fh)
    for b in d.get('benchmarks', []):
        for k in ['successful', 'errored', 'incomplete']:
            for r in b.get('requests', {}).get(k, []):
                for field in list(r.keys()):
                    if field in ('prompt', 'output', 'text'):
                        r[field] = ''
    with open(f, 'w') as fh:
        json.dump(d, fh)
    print(f'  stripped {os.path.basename(f)}: {os.path.getsize(f) // 1024}KB')
"

kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
    bash -c 'cd /models/benchmark-output && tar czf /tmp/pp-results.tar.gz *.json'

TRANSFER_SCRIPT="${SCRIPT_DIR}/../../scripts/transfer-large-file-chunked.sh"
bash "$TRANSFER_SCRIPT" \
    "$KUBECONFIG" "$NS" "$GUIDELLM_POD" \
    "/tmp/pp-results.tar.gz" "${OUTPUT_DIR}/pp-results.tar.gz" "$((512 * 1024))"
tar xzf "${OUTPUT_DIR}/pp-results.tar.gz" -C "${OUTPUT_DIR}" && rm -f "${OUTPUT_DIR}/pp-results.tar.gz"

for f in "${OUTPUT_DIR}"/*.json; do
    [ -f "$f" ] || continue
    zstd -q -f --rm "$f" 2>/dev/null || true
done
echo "    Collected $(ls "${OUTPUT_DIR}"/*.json.zst 2>/dev/null | wc -l | tr -d ' ') result files"

# --- clean up benchmark output in pod ---
kubectl exec -n "$NS" "$GUIDELLM_POD" -- \
    rm -rf /models/benchmark-output /tmp/pp-results.tar.gz 2>/dev/null || true

# --- delete LLMInferenceService ---
echo ""
echo "==> Deleting $LLMISVC_NAME..."
kubectl delete -f "$LLMISVC_MANIFEST"

echo "    Waiting for pods to terminate..."
kubectl wait --for=delete pod -l "app.kubernetes.io/name=${LLMISVC_NAME}" \
    -n "$NS" --timeout=120s 2>/dev/null || true

echo ""
echo "=========================================="
echo "Sweep complete: $LLMISVC_NAME"
ls -lh "${OUTPUT_DIR}/${OUTPUT_PREFIX}"-*.json.zst 2>/dev/null || true
echo "=========================================="
