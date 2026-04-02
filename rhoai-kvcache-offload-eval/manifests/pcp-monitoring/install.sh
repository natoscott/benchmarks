#!/bin/bash
# Deploy permanent PCP monitoring DaemonSet to the cluster.
# Run from the repo root or the manifests/pcp-monitoring/ directory.
set -euo pipefail

KUBECONFIG="${KUBECONFIG:-./kubeconfig-psap-fire-athena}"
MANIFESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Deploying PCP monitoring DaemonSet..."
kubectl --kubeconfig="${KUBECONFIG}" apply -f "${MANIFESTS}/00-namespace.yaml"
kubectl --kubeconfig="${KUBECONFIG}" apply -f "${MANIFESTS}/01-rbac.yaml"
kubectl --kubeconfig="${KUBECONFIG}" apply -f "${MANIFESTS}/02-configmap.yaml"
kubectl --kubeconfig="${KUBECONFIG}" apply -f "${MANIFESTS}/03-daemonset.yaml"
kubectl --kubeconfig="${KUBECONFIG}" apply -f "${MANIFESTS}/04-service.yaml"

echo "Waiting for DaemonSet pods to be ready..."
kubectl --kubeconfig="${KUBECONFIG}" rollout status daemonset/pcp-monitoring \
    -n pcp-monitoring --timeout=120s

echo ""
echo "Deployed. Node IPs for direct pmcd/pmproxy access:"
kubectl --kubeconfig="${KUBECONFIG}" get nodes \
    -o custom-columns='NODE:.metadata.name,IP:.status.addresses[?(@.type=="InternalIP")].address'

echo ""
echo "Usage:"
echo "  # Direct (from inside cluster, or if node port 44321 is reachable):"
echo "  pmrep --host <node-ip> kernel.all.load"
echo ""
echo "  # Via port-forward (works from any laptop):"
echo "  kubectl --kubeconfig=${KUBECONFIG} port-forward -n pcp-monitoring \\"
echo "      daemonset/pcp-monitoring 44321 44322"
echo "  pmrep --host localhost kernel.all.load"
echo ""
echo "  # Check nvidia PMDA loaded on GPU worker:"
echo "  pminfo --host <gpu-node-ip> nvidia"
