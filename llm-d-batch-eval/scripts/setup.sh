#!/bin/bash
# One-time infrastructure setup for llm-d Batch Gateway benchmarks on psap-fire-athena.
# Run once before benchmarks. Idempotent — safe to re-run.
#
# What this does:
#   1. Pre-flight: GPU nodes, RHOAI DSC Ready, openshift-ai-inference gateway
#   2. Namespace, RBAC, HF token secret
#   3. Generate secrets (random PostgreSQL password, batch-gateway-secrets)
#   4. Model storage PVC (600 Gi NFS-RWX)
#   5. Batch file storage PVC (50 Gi NFS-RWX)
#   6. PostgreSQL (Red Hat container, NVMe storage)
#   7. Valkey (Red Hat container)
#   8. PCP Deployment (openmetrics + native valkey/postgresql PMDAs)
#   9. guidellm pod (v0.7.0)
#  10. EPP scheduler config (Valkey-backed prefix-cache)
#  11. LLMInferenceService manifests
#  12. Model presence check
#
# Usage:
#   KUBECONFIG=~/psap/kubeconfig-psap-fire-athena bash scripts/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS="${SCRIPT_DIR}/../manifests"
NAMESPACE="llm-d-batch"

export KUBECONFIG="${KUBECONFIG:-${HOME}/psap/kubeconfig-psap-fire-athena}"

echo "=== llm-d Batch Gateway Benchmark Setup ==="
echo "Cluster: $(kubectl cluster-info 2>/dev/null | grep 'control plane' | awk '{print $NF}')"
echo ""

# ── 1. Pre-flight checks ────────────────────────────────────────────────────
echo "[1/12] Pre-flight checks..."

echo "  Checking GPU nodes..."
GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu.present=true \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
[ -n "${GPU_NODES}" ] || { echo "ERROR: No GPU nodes found"; exit 1; }
GPU_COUNT=$(kubectl get nodes -l nvidia.com/gpu.present=true \
    -o jsonpath='{range .items[*]}{.status.capacity.nvidia\.com/gpu}{" "}{end}' 2>/dev/null \
    | tr ' ' '\n' | paste -sd+ | bc)
echo "  GPU nodes: ${GPU_NODES}"
echo "  Total GPUs: ${GPU_COUNT}"

echo "  Checking RHOAI DSC..."
PHASE=$(kubectl get datascienceclusters default-dsc \
    -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
[ "${PHASE}" = "Ready" ] || { echo "ERROR: RHOAI DSC not Ready (phase=${PHASE})"; exit 1; }
echo "  RHOAI DSC: Ready"

echo "  Checking openshift-ai-inference gateway..."
GW_PROGRAMMED=$(kubectl get gateway openshift-ai-inference -n openshift-ingress \
    -o jsonpath='{.status.conditions[?(@.type=="Programmed")].status}' 2>/dev/null || echo "")
[ "${GW_PROGRAMMED}" = "True" ] || \
    echo "  WARNING: openshift-ai-inference gateway not Programmed (status=${GW_PROGRAMMED})"
echo "  Gateway: $(kubectl get gateway openshift-ai-inference -n openshift-ingress \
    -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || echo 'unknown')"

# ── 2. Namespace, RBAC, HF token ────────────────────────────────────────────
echo ""
echo "[2/12] Creating namespace, RBAC, and HF token secret..."
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

# ── 3. Generate secrets ─────────────────────────────────────────────────────
echo ""
echo "[3/12] Creating secrets..."

if kubectl get secret batch-gateway-secrets -n "${NAMESPACE}" &>/dev/null; then
    echo "  batch-gateway-secrets already exists — skipping"
    PG_PASS=$(kubectl get secret batch-gateway-secrets -n "${NAMESPACE}" \
        -o jsonpath='{.data.postgresql-password}' | base64 -d)
else
    PG_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 20)
    kubectl create secret generic batch-gateway-secrets \
        -n "${NAMESPACE}" \
        --from-literal=postgresql-password="${PG_PASS}" \
        --from-literal=redis-url="redis://valkey.${NAMESPACE}.svc.cluster.local:6379" \
        --from-literal=postgresql-url="postgresql://batchgw:${PG_PASS}@postgresql.${NAMESPACE}.svc.cluster.local:5432/batchgateway?sslmode=disable" \
        --from-literal=inference-api-key="" \
        --from-literal=s3-secret-access-key=""
    echo "  batch-gateway-secrets created (random PostgreSQL password)"
fi

# ── 4. Model storage PVC ────────────────────────────────────────────────────
echo ""
echo "[4/12] Creating model storage PVC (600 Gi NFS-RWX)..."
if kubectl get pvc model-storage-nfs -n "${NAMESPACE}" &>/dev/null; then
    echo "  PVC already exists — skipping"
else
    kubectl apply -f "${MANIFESTS}/setup/model-storage-pvc.yaml"
    echo "  PVC created (Immediate binding via nfs-rwx)"
fi

# ── 5. Batch file storage PVC ───────────────────────────────────────────────
echo ""
echo "[5/12] Creating batch file storage PVC (50 Gi NFS-RWX)..."
if kubectl get pvc batch-gateway-files -n "${NAMESPACE}" &>/dev/null; then
    echo "  PVC already exists — skipping"
else
    kubectl apply -f "${MANIFESTS}/setup/batch-files-pvc.yaml"
    echo "  PVC created"
fi

# ── 6. PostgreSQL ───────────────────────────────────────────────────────────
echo ""
echo "[6/12] Deploying PostgreSQL (Red Hat container, NVMe storage)..."
kubectl apply -f "${MANIFESTS}/infra/postgresql-deployment.yaml"
echo "  Waiting for PostgreSQL to be ready..."
kubectl rollout status deployment/postgresql -n "${NAMESPACE}" --timeout=300s
echo "  PostgreSQL ready"

# ── 7. Valkey ───────────────────────────────────────────────────────────────
echo ""
echo "[7/12] Deploying Valkey (Red Hat container)..."
kubectl apply -f "${MANIFESTS}/infra/valkey-deployment.yaml"
echo "  Waiting for Valkey to be ready..."
kubectl wait --for=condition=ready pod -l app=valkey \
    -n "${NAMESPACE}" --timeout=120s
echo "  Valkey ready"

# ── 8. PCP Deployment ───────────────────────────────────────────────────────
echo ""
echo "[8/12] Deploying PCP (openmetrics + native valkey/postgresql PMDAs)..."
kubectl apply -f "${MANIFESTS}/monitoring/openmetrics-pmda-configmap.yaml"
kubectl apply -f "${MANIFESTS}/monitoring/openmetrics-pmlogconf-configmap.yaml"
kubectl apply -f "${MANIFESTS}/monitoring/valkey-pmda-configmap.yaml"
kubectl apply -f "${MANIFESTS}/monitoring/postgresql-pmda-configmap.yaml"
kubectl apply -f "${MANIFESTS}/monitoring/pcp-deployment.yaml"
kubectl rollout status deployment/pcp -n "${NAMESPACE}" --timeout=300s
echo "  PCP ready"

# ── 9. guidellm pod ────────────────────────────────────────────────────────
echo ""
echo "[9/12] Deploying guidellm pod (v0.7.0)..."
kubectl apply -f "${MANIFESTS}/benchmark/guidellm-deployment.yaml"
kubectl wait --for=condition=ready pod -l app=guidellm \
    -n "${NAMESPACE}" --timeout=300s
echo "  guidellm pod ready"

# ── 10. EPP scheduler config ───────────────────────────────────────────────
echo ""
echo "[10/12] Applying EPP scheduler config (Valkey-backed prefix-cache)..."
kubectl apply -f "${MANIFESTS}/models/epp-config-valkey.yaml"
kubectl apply -f "${MANIFESTS}/models/epp-scheduler-config-valkey.yaml"
echo "  epp-scheduler-valkey applied"

# ── 11. LLMInferenceService manifests ───────────────────────────────────────
echo ""
echo "[11/12] Applying LLMInferenceService manifests..."
kubectl apply -f "${MANIFESTS}/models/llm-inference-service-qwen3-8b.yaml"
kubectl apply -f "${MANIFESTS}/models/llm-inference-service-llama-70b.yaml"
kubectl apply -f "${MANIFESTS}/models/llm-inference-service-gpt-oss-120b.yaml"
echo "  LLMInferenceServices applied"

# ── 12. Model presence check ───────────────────────────────────────────────
echo ""
echo "[12/12] Checking model presence on PVC via guidellm pod..."
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

    MODELS_EXPECTED="Qwen/Qwen3-8B RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 openai/gpt-oss-120b"
    ALL_PRESENT=true

    for model in ${MODELS_EXPECTED}; do
        MODEL_DIR="/models/models/${model}"
        if kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
            test -f "${MODEL_DIR}/config.json" 2>/dev/null; then
            SIZE=$(kubectl exec -n "${NAMESPACE}" "${GUIDELLM_POD}" -- \
                du -sh "${MODEL_DIR}" 2>/dev/null | awk '{print $1}' || echo "?")
            echo "  + ${model} (${SIZE})"
        else
            echo "  - ${model} — NOT FOUND"
            ALL_PRESENT=false
        fi
    done

    echo ""
    if [ "${ALL_PRESENT}" = "true" ]; then
        echo "  All models present on PVC."
    else
        echo "  Some models missing. Download them before running benchmarks:"
        echo "    kubectl exec -n ${NAMESPACE} ${GUIDELLM_POD} -- \\"
        echo "      huggingface-cli download <model-name> --local-dir /models/models/<model-name>"
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
echo "Next steps:"
echo "  1. If models are missing, download them via the guidellm pod"
echo "  2. Run benchmarks: bash scripts/run-all-scenarios.sh 2>&1 | tee /tmp/batch-gateway-benchmark.log"
echo "  3. Monitor:        tail -F /tmp/batch-gateway-benchmark.log"
