#!/bin/bash
# Pre-download models to the model-storage-pvc via the guidellm pod.
# Must be run before applying LLMInferenceService manifests.
# The LLMInferenceService uses pvc:// URIs so no HF credentials are needed
# at inference time (avoids kserve storage-initializer auth complexity).
#
# Usage:
#   bash scripts/download-models.sh             # download both models
#   MODELS="meta-llama/Llama-3.1-70B-Instruct" bash scripts/download-models.sh
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$(dirname "$0")/../kubeconfig}"
NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
MODELS="${MODELS:-RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 openai/gpt-oss-120b meta-llama/Llama-3.1-70B-Instruct}"

GUIDELLM_POD=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
    -l app=guidellm --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
[ -n "${GUIDELLM_POD}" ] || { echo "ERROR: No running guidellm pod"; exit 1; }
echo "Using guidellm pod: ${GUIDELLM_POD}"

# Verify PVC mount and space
echo "PVC disk space:"
kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- df -h /models

for MODEL in ${MODELS}; do
    MODEL_DIR="/models/models/${MODEL}"
    LOG_FILE="/models/download-$(echo "${MODEL}" | tr '/' '-').log"

    echo ""
    echo "=== Downloading ${MODEL} ==="

    # Check if already present
    if kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
        test -f "${MODEL_DIR}/config.json" 2>/dev/null; then
        echo "  Already downloaded — skipping"
        continue
    fi

    echo "  Target: ${MODEL_DIR}"
    echo "  Log:    ${LOG_FILE}"
    echo "  Starting background download..."

    kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
        bash -c "
mkdir -p '${MODEL_DIR}'
nohup python3 -c \"
import os
from huggingface_hub import snapshot_download
print('Downloading ${MODEL}...', flush=True)
path = snapshot_download(
    repo_id='${MODEL}',
    local_dir='${MODEL_DIR}',
    token=os.environ['HF_TOKEN'],
)
print(f'Done: {path}', flush=True)
\" > '${LOG_FILE}' 2>&1 &
echo \"PID=\$!\"
"
    echo "  Download running. Monitor with:"
    echo "    kubectl exec -n ${NAMESPACE} ${GUIDELLM_POD} -- tail -f ${LOG_FILE}"
done

echo ""
echo "Check all download progress:"
echo "  kubectl exec -n ${NAMESPACE} ${GUIDELLM_POD} -- ls -lh /models/models/"
echo "  kubectl exec -n ${NAMESPACE} ${GUIDELLM_POD} -- du -sh /models/models/*"
