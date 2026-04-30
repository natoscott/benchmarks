#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG="${KUBECONFIG:-$HOME/psap/kubeconfig-psap-fire-athena}"
export KUBECONFIG

MANIFESTS="$(cd "$(dirname "$0")/../manifests" && pwd)"
NS=aiconfigurator

# --- pre-flight ---
echo "==> Checking cluster access..."
kubectl cluster-info --request-timeout=10s

echo "==> Checking GPU node..."
GPU_NODE=$(kubectl get nodes -l nvidia.com/gpu.present=true -o name | head -1)
[[ -z "$GPU_NODE" ]] && { echo "ERROR: no GPU node found"; exit 1; }
echo "    GPU node: $GPU_NODE"

# --- namespace + RBAC ---
echo "==> Creating namespace and service accounts..."
kubectl apply -f "$MANIFESTS/namespace.yaml"
kubectl apply -f "$MANIFESTS/serviceaccounts.yaml"
kubectl apply -f "$MANIFESTS/guidellm-rbac.yaml"

# --- HF token secret ---
if ! kubectl get secret llm-d-hf-token -n "$NS" &>/dev/null; then
    HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/psap/hf_token")}"
    kubectl create secret generic llm-d-hf-token \
        --from-literal=HF_TOKEN="$HF_TOKEN" \
        -n "$NS"
fi

# --- configmaps + workloads ---
echo "==> Deploying configmaps..."
kubectl apply -f "$MANIFESTS/openmetrics-pmda-configmap.yaml"
kubectl apply -f "$MANIFESTS/openmetrics-pmlogconf-configmap.yaml"

echo "==> Deploying valkey..."
kubectl apply -f "$MANIFESTS/valkey-deployment.yaml"

echo "==> Deploying PCP..."
kubectl apply -f "$MANIFESTS/pcp-deployment.yaml"

echo "==> Deploying guidellm..."
kubectl apply -f "$MANIFESTS/guidellm-deployment.yaml"

echo "==> Deploying aiconfigurator..."
kubectl apply -f "$MANIFESTS/aiconfigurator-deployment.yaml"

# --- wait ---
echo "==> Waiting for pods to be ready..."
kubectl rollout status deployment/valkey         -n "$NS" --timeout=120s
kubectl rollout status deployment/guidellm       -n "$NS" --timeout=120s
kubectl rollout status deployment/aiconfigurator -n "$NS" --timeout=120s
kubectl rollout status deployment/pcp            -n "$NS" --timeout=300s

echo ""
echo "==> Setup complete. Namespace: $NS"
echo ""
echo "    Run aiconfigurator:"
echo "      kubectl exec -it -n $NS deploy/aiconfigurator -- aiconfigurator cli default --model <model> --total-gpus 8 --system h200_sxm --backend vllm --deployment-target llm-d"
echo ""
echo "    Run guidellm:"
echo "      kubectl exec -it -n $NS deploy/guidellm -- bash"
