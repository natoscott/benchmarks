#!/bin/bash
# One-time infrastructure setup for RHOAI 3.3 KV cache offload benchmarks.
# Run once after install-rhoai.sh completes, before starting benchmark runs.
#
# What this does:
#   1. Installs the leader-worker-set operator (required for LLMInferenceService TP)
#   2. Patches the data-science-gateway to allow routes from llm-d-pfc-cpu namespace
#   3. Creates namespace, RBAC, PVC, HF token secret
#   4. Deploys PCP DaemonSet and guidellm pod
#   5. Applies LLMInferenceService manifests (pods start downloading models)
#
# Usage:
#   bash scripts/setup.sh
#   HF_TOKEN=hf_xxx bash scripts/setup.sh
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$(dirname "$0")/../kubeconfig}"
MANIFESTS="$(dirname "$0")/../manifests/benchmark"
NAMESPACE="llm-d-pfc-cpu"

echo "=== RHOAI 3.3 Benchmark Setup ==="
echo "Cluster: $(kubectl cluster-info 2>/dev/null | grep 'control plane' | awk '{print $NF}')"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "[pre-flight] Checking GPU nodes..."
GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu.present=true \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
[ -n "${GPU_NODES}" ] || { echo "ERROR: No GPU nodes found"; exit 1; }
echo "  GPU nodes: ${GPU_NODES}"

echo "[pre-flight] Checking /var free space on GPU nodes..."
for node in ${GPU_NODES}; do
    FREE=$(kubectl debug node/"${node}" -it --image=busybox --quiet -- \
        df -h /host/var 2>/dev/null | awk 'NR==2{print $4}' || echo "unknown")
    echo "  ${node}: /var free = ${FREE}"
done

echo "[pre-flight] Checking RHOAI is installed..."
kubectl get datascienceclusters default-dsc -o jsonpath='{.status.phase}' 2>/dev/null | \
    grep -q Ready || { echo "ERROR: RHOAI DSC not Ready. Run install-rhoai.sh first."; exit 1; }
echo "  RHOAI DSC: Ready"

# ── Step 1: leader-worker-set operator ───────────────────────────────────────
echo ""
echo "[1/7] Installing leader-worker-set operator..."
kubectl apply -f "${MANIFESTS}/leader-worker-set-subscription.yaml"

echo "  Waiting for leader-worker-set CSV to reach Succeeded..."
TIMEOUT=300; ELAPSED=0; INTERVAL=10
while true; do
    PHASE=$(kubectl get csv -n openshift-lws \
        -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
    [[ "${PHASE}" == "Succeeded" ]] && { echo "  leader-worker-set: Succeeded"; break; }
    (( ELAPSED >= TIMEOUT )) && { echo "ERROR: leader-worker-set timed out"; kubectl get csv -n openshift-lws; exit 1; }
    echo "  Phase: ${PHASE:-pending} (${ELAPSED}/${TIMEOUT}s)"
    sleep $INTERVAL; ELAPSED=$(( ELAPSED + INTERVAL ))
done

# ── Step 2: patch data-science-gateway to allow llm-d-pfc-cpu ─────────────────
echo ""
echo "[2/7] Patching data-science-gateway to allow llm-d-pfc-cpu namespace..."
# The gateway's listener currently only allows openshift-ingress and redhat-ods-applications.
# We add llm-d-pfc-cpu so LLMInferenceService HTTPRoutes in that namespace can attach.
CURRENT_NS=$(kubectl get gateway data-science-gateway -n openshift-ingress \
    -o jsonpath='{.spec.listeners[0].allowedRoutes.namespaces.selector.matchExpressions[0].values}' \
    2>/dev/null || echo "")
if echo "${CURRENT_NS}" | grep -q "llm-d-pfc-cpu"; then
    echo "  llm-d-pfc-cpu already in gateway allowlist — skipping"
else
    kubectl patch gateway data-science-gateway -n openshift-ingress --type=json -p '[
      {
        "op": "add",
        "path": "/spec/listeners/0/allowedRoutes/namespaces/selector/matchExpressions/0/values/-",
        "value": "llm-d-pfc-cpu"
      }
    ]'
    echo "  Added llm-d-pfc-cpu to gateway namespace allowlist"
fi

# ── Step 3: namespace, RBAC, HF token ─────────────────────────────────────────
echo ""
echo "[3/7] Creating namespace, RBAC, and HF token secret..."
kubectl apply -f "${MANIFESTS}/namespace.yaml"
kubectl apply -f "${MANIFESTS}/pcp-serviceaccount.yaml"

if [ -n "${HF_TOKEN:-}" ]; then
    TOKEN="${HF_TOKEN}"
elif [ -f "$(dirname "$0")/../hftoken" ]; then
    TOKEN=$(cat "$(dirname "$0")/../hftoken")
else
    echo ""
    echo "  WARNING: No HF_TOKEN env var or hftoken file found."
    echo "  Create the secret before running benchmarks:"
    echo "    kubectl create secret generic llm-d-hf-token \\"
    echo "      --from-literal=HF_TOKEN=hf_xxx -n ${NAMESPACE}"
    TOKEN=""
fi

if [ -n "${TOKEN}" ]; then
    kubectl create secret generic llm-d-hf-token \
        --from-literal=HF_TOKEN="${TOKEN}" \
        -n "${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    echo "  HF token secret applied"
fi

# ── Step 4: model storage PVC ─────────────────────────────────────────────────
echo ""
echo "[4/7] Creating model storage PVC (1 Ti, local NVMe)..."
kubectl apply -f "${MANIFESTS}/model-storage-pvc.yaml"
echo "  PVC will bind when the first consumer pod is scheduled."
echo "  NOTE: All pods sharing this PVC (vLLM replicas, guidellm) will be"
echo "  pinned to whichever GPU node the PVC binds to."

# ── Step 5: PCP ConfigMaps and DaemonSet ──────────────────────────────────────
echo ""
echo "[5/7] Deploying PCP DaemonSet (openmetrics PMDA: vLLM + DCGM)..."
kubectl apply -f "${MANIFESTS}/openmetrics-pmda-configmap.yaml"
kubectl apply -f "${MANIFESTS}/openmetrics-pmlogconf-configmap.yaml"
kubectl apply -f "${MANIFESTS}/pcp-deployment.yaml"

echo "  Waiting for PCP pods on GPU nodes..."
kubectl rollout status daemonset/pcp -n "${NAMESPACE}" --timeout=300s

# ── Step 6: guidellm pod ──────────────────────────────────────────────────────
echo ""
echo "[6/7] Deploying guidellm pod..."
kubectl apply -f "${MANIFESTS}/guidellm-deployment.yaml"
kubectl wait --for=condition=ready pod -l app=guidellm \
    -n "${NAMESPACE}" --timeout=300s
echo "  guidellm pod ready"

# ── Step 7: LLMInferenceServices (model download begins) ─────────────────────
echo ""
echo "[7/7] Applying LLMInferenceService manifests..."
echo "  Models will be downloaded from HuggingFace to the PVC."
echo "  This may take 30-60+ minutes per model depending on network."
kubectl apply -f "${MANIFESTS}/llm-inference-service-llama-70b.yaml"
kubectl apply -f "${MANIFESTS}/llm-inference-service-gpt-oss-120b.yaml"

echo ""
echo "  NOTE: After first model download completes, check vLLM startup logs to"
echo "  determine actual GPU block size and tune gpu-memory-utilization and"
echo "  num_cpu_blocks per model before starting benchmarks."
echo ""
echo "  Check download / startup progress:"
echo "    kubectl get llminferenceservices -n ${NAMESPACE}"
echo "    kubectl get pods -n ${NAMESPACE}"
echo ""
echo "  After the LLMInferenceService is Ready, update the openmetrics PMDA"
echo "  ConfigMap with the EPP service URL (auto-named <service>-epp-service):"
echo "    kubectl get svc -n ${NAMESPACE} | grep epp"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
kubectl get pods,pvc,llminferenceservices -n "${NAMESPACE}" 2>/dev/null || true
