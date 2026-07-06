#!/bin/bash
# One-time infrastructure setup for EPP scheduling evaluation.
# Run once before benchmarks. Idempotent — safe to re-run.
#
# What this does:
#   1. Pre-flight: GPU nodes, RHOAI DSC Ready, inference gateway
#   2. Namespace, RBAC, HF token secret
#   3. Inference gateway (openshift-ai-inference, with namespace allowlist)
#   4. Model storage PVC (2 Ti NFS-RWX)
#   5. PCP Deployment (openmetrics PMDA)
#   6. guidellm pod (v0.7.1)
#   7. EPP config ConfigMaps (prior-default, optimized-baseline, pd-optimized)
#   8. LLMInferenceServiceConfig resources
#   9. LLMInferenceService manifests (replicas=0)
#  10. Model presence check
#
# Usage:
#   KUBECONFIG=~/psap/kubeconfig-psap-janus bash scripts/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS="${SCRIPT_DIR}/../manifests"
SHARED_SCRIPTS="${SCRIPT_DIR}/../../scripts"
NAMESPACE="llm-d-nathans-epp-eval"

export KUBECONFIG="${KUBECONFIG:-${HOME}/psap/kubeconfig-psap-janus}"

echo "=== EPP Scheduling Evaluation Setup ==="
echo "Cluster: $(kubectl cluster-info 2>/dev/null | grep 'control plane' | awk '{print $NF}')"
echo ""

# ── 1. Pre-flight checks ────────────────────────────────────────────────────
echo "[1/10] Pre-flight checks..."

echo "  Checking GPU nodes..."
GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu.present=true \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
[ -n "${GPU_NODES}" ] || { echo "ERROR: No GPU nodes found"; exit 1; }
GPU_COUNT=$(kubectl get nodes -l nvidia.com/gpu.present=true \
    -o jsonpath='{range .items[*]}{.status.capacity.nvidia\.com/gpu}{" "}{end}' 2>/dev/null \
    | tr ' ' '\n' | { paste -sd+ || echo "0"; } | bc)
echo "  GPU nodes: ${GPU_NODES}"
echo "  Total GPUs: ${GPU_COUNT}"

echo "  Checking RHOAI DSC..."
PHASE=$(kubectl get datascienceclusters default-dsc \
    -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
if [ "${PHASE}" = "Ready" ]; then
    echo "  RHOAI DSC: Ready"
else
    echo "  WARNING: RHOAI DSC not Ready (phase=${PHASE})"
    echo "  DSC may not exist yet (ODH 3.4.0 uses DSCI instead)"
    echo "  Checking DSCI..."
    DSCI_PHASE=$(kubectl get dsci default-dsci \
        -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    echo "  DSCI: ${DSCI_PHASE:-not found}"
fi

echo "  Checking inference gateway..."
for GW_NAME in openshift-ai-inference data-science-gateway; do
    GW_PROGRAMMED=$(kubectl get gateway "${GW_NAME}" -n openshift-ingress \
        -o jsonpath='{.status.conditions[?(@.type=="Programmed")].status}' 2>/dev/null || echo "")
    if [ "${GW_PROGRAMMED}" = "True" ]; then
        echo "  Gateway '${GW_NAME}': Programmed"
        echo "  Gateway address: $(kubectl get gateway "${GW_NAME}" -n openshift-ingress \
            -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || echo 'unknown')"
        break
    fi
done

echo "  Checking RoCE/RDMA..."
ROCE_COUNT=$(kubectl get nodes -l nvidia.com/gpu.present=true \
    -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/roce}{" "}{end}' 2>/dev/null \
    | tr ' ' '\n' | paste -sd+ | bc 2>/dev/null || echo "0")
echo "  RoCE devices: ${ROCE_COUNT}"

# ── 2. Namespace, RBAC, HF token ────────────────────────────────────────────
echo ""
echo "[2/10] Creating namespace, RBAC, and HF token secret..."
kubectl apply -f "${MANIFESTS}/setup/namespace.yaml"
kubectl apply -f "${MANIFESTS}/setup/serviceaccounts.yaml"
kubectl apply -f "${MANIFESTS}/setup/guidellm-rbac.yaml" 2>/dev/null || true

HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/psap/hf_token}"
if [ -n "${HF_TOKEN:-}" ]; then
    TOKEN="${HF_TOKEN}"
elif [ -f "${HF_TOKEN_FILE}" ]; then
    TOKEN=$(cat "${HF_TOKEN_FILE}")
else
    echo "  WARNING: No HF_TOKEN env var or ${HF_TOKEN_FILE} found."
    echo "  Create the secret manually:"
    echo "    kubectl create secret generic llm-d-hf-token --from-literal=HF_TOKEN=hf_xxx -n ${NAMESPACE}"
    TOKEN=""
fi

if [ -n "${TOKEN}" ]; then
    kubectl create secret generic llm-d-hf-token \
        --from-literal=HF_TOKEN="${TOKEN}" \
        -n "${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    echo "  HF token secret applied"
fi

# ── 3. Inference gateway ────────────────────────────────────────────────────
echo ""
echo "[3/10] Creating openshift-ai-inference gateway..."
if kubectl get gateway openshift-ai-inference -n openshift-ingress &>/dev/null; then
    echo "  Gateway already exists — checking namespace allowlist"
    ALLOWED=$(kubectl get gateway openshift-ai-inference -n openshift-ingress \
        -o jsonpath='{.spec.listeners[0].allowedRoutes.namespaces.selector.matchExpressions[0].values}' 2>/dev/null || echo "")
    if echo "${ALLOWED}" | grep -q "${NAMESPACE}"; then
        echo "  Namespace ${NAMESPACE} already in allowlist"
    else
        echo "  Adding ${NAMESPACE} to gateway allowlist..."
        kubectl get gateway openshift-ai-inference -n openshift-ingress -o json | \
            python3 -c "
import json, sys
gw = json.load(sys.stdin)
vals = gw['spec']['listeners'][0]['allowedRoutes']['namespaces']['selector']['matchExpressions'][0]['values']
if '${NAMESPACE}' not in vals:
    vals.append('${NAMESPACE}')
json.dump(gw, sys.stdout)
" | kubectl apply -f -
    fi
else
    kubectl apply -f - <<GWEOF
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: openshift-ai-inference
  namespace: openshift-ingress
spec:
  gatewayClassName: data-science-gateway-class
  listeners:
  - allowedRoutes:
      namespaces:
        from: Selector
        selector:
          matchExpressions:
          - key: kubernetes.io/metadata.name
            operator: In
            values:
            - openshift-ingress
            - redhat-ods-applications
            - ${NAMESPACE}
    name: https
    port: 443
    protocol: HTTPS
GWEOF
    echo "  Gateway created"
fi

echo "  Waiting for gateway to be Programmed..."
for i in $(seq 1 60); do
    GW_STATUS=$(kubectl get gateway openshift-ai-inference -n openshift-ingress \
        -o jsonpath='{.status.conditions[?(@.type=="Programmed")].status}' 2>/dev/null || echo "")
    if [ "${GW_STATUS}" = "True" ]; then
        echo "  Gateway Programmed"
        break
    fi
    sleep 5
done

# ── 4. Model storage PVC ────────────────────────────────────────────────────
echo ""
echo "[4/10] Creating model storage PVC (2 Ti NFS-RWX)..."
if kubectl get pvc model-storage-nfs -n "${NAMESPACE}" &>/dev/null; then
    echo "  PVC already exists — skipping"
else
    kubectl apply -f "${MANIFESTS}/setup/model-storage-pvc.yaml"
    echo "  PVC created"
fi

# ── 4. PCP Deployment ───────────────────────────────────────────────────────
echo ""
echo "[5/10] Deploying PCP (openmetrics PMDA)..."

# Create the service CA ConfigMap with injection annotation so OpenShift
# populates it with the cluster's service signing CA certificate.
kubectl create configmap openshift-service-ca.crt \
    -n "${NAMESPACE}" --dry-run=client -o yaml | \
    kubectl annotate --local -f - \
    service.beta.openshift.io/inject-cabundle=true -o yaml --dry-run=client | \
    kubectl apply -f -
echo "  Service CA ConfigMap created (OpenShift will inject CA bundle)"

kubectl apply -f "${MANIFESTS}/monitoring/openmetrics-pmda-configmap.yaml"
kubectl apply -f "${MANIFESTS}/monitoring/openmetrics-pmlogconf-configmap.yaml"
kubectl create configmap pcp-scripts \
    --from-file=pcp-wait-and-restart-pmlogger.sh="${SHARED_SCRIPTS}/pcp-wait-and-restart-pmlogger.sh" \
    -n "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "${MANIFESTS}/monitoring/pcp-deployment.yaml"
kubectl rollout status deployment/pcp -n "${NAMESPACE}" --timeout=300s
echo "  PCP ready"

# ── 5. guidellm pod ────────────────────────────────────────────────────────
echo ""
echo "[6/10] Deploying guidellm pod (v0.7.1)..."
kubectl apply -f "${MANIFESTS}/benchmark/guidellm-deployment.yaml"
kubectl wait --for=condition=ready pod -l app=guidellm \
    -n "${NAMESPACE}" --timeout=300s
echo "  guidellm pod ready"

# ── 6. EPP config ConfigMaps ──────────────────────────────────────────────
echo ""
echo "[7/10] Applying EPP config ConfigMaps..."
kubectl apply -f "${MANIFESTS}/epp-configs/epp-config-prior-default.yaml"
kubectl apply -f "${MANIFESTS}/epp-configs/epp-config-optimized-baseline.yaml"
kubectl apply -f "${MANIFESTS}/epp-configs/epp-config-pd-optimized.yaml"
echo "  EPP configs applied (prior-default, optimized-baseline, pd-optimized)"

# ── 7. LLMInferenceServiceConfig resources ────────────────────────────────
echo ""
echo "[8/10] Applying LLMInferenceServiceConfig resources..."
kubectl apply -f "${MANIFESTS}/epp-configs/epp-scheduler-prior-default.yaml"
kubectl apply -f "${MANIFESTS}/epp-configs/epp-scheduler-optimized-baseline.yaml"
echo "  LLMInferenceServiceConfigs applied"

# ── 8. LLMInferenceService manifests ───────────────────────────────────────
echo ""
echo "[9/10] Applying LLMInferenceService manifests (replicas=0)..."
kubectl apply -f "${MANIFESTS}/models/llm-inference-service-qwen3-30b.yaml"
kubectl apply -f "${MANIFESTS}/models/llm-inference-service-llama-70b.yaml"
kubectl apply -f "${MANIFESTS}/models/llm-inference-service-gpt-oss-120b.yaml"
echo "  LLMInferenceServices applied"

# ── 9. Model presence check ───────────────────────────────────────────────
echo ""
echo "[10/10] Checking model presence on PVC via guidellm pod..."
GUIDELLM_POD=$(kubectl get pods -n "${NAMESPACE}" \
    -l app=guidellm --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [ -z "${GUIDELLM_POD}" ]; then
    echo "  WARNING: No running guidellm pod found, skipping model check."
else
    echo "  PVC disk usage:"
    kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- df -h /models 2>/dev/null || true
    echo ""
    echo "  Model directories on PVC:"

    MODELS_EXPECTED=(
        "hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507"
        "hub/models--RedHatAI--Llama-3.3-70B-Instruct-FP8-dynamic"
        "hub/models--openai--gpt-oss-120b"
    )
    ALL_PRESENT=true

    for model_path in "${MODELS_EXPECTED[@]}"; do
        MODEL_DIR="/models/${model_path}"
        if kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
            test -d "${MODEL_DIR}/snapshots" 2>/dev/null; then
            SIZE=$(kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
                du -sh "${MODEL_DIR}" 2>/dev/null | awk '{print $1}' || echo "?")
            echo "  + ${model_path} (${SIZE})"
        else
            echo "  - ${model_path} — NOT FOUND"
            ALL_PRESENT=false
        fi
    done

    echo ""
    if [ "${ALL_PRESENT}" = "true" ]; then
        echo "  All models present on PVC."
    else
        echo "  Some models missing. Download them before running benchmarks:"
        echo "    kubectl exec -n ${NAMESPACE} ${GUIDELLM_POD} -- \\"
        echo "      huggingface-cli download <model-name> --local-dir /models/hub/models--<org>--<model>"
    fi
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
kubectl get pods,pvc,svc -n "${NAMESPACE}" 2>/dev/null || true
echo ""
echo "LLMInferenceServices:"
kubectl get llminferenceservices -n "${NAMESPACE}" 2>/dev/null || true
echo ""
echo "EPP Configs:"
kubectl get llminferenceserviceconfigs -n "${NAMESPACE}" 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. If models are missing, download them via the guidellm pod"
echo "  2. Run benchmarks: bash scripts/run-all-scenarios.sh 2>&1 | tee /tmp/epp-eval-benchmark.log"
echo "  3. Monitor:        tail -F /tmp/epp-eval-benchmark.log"
