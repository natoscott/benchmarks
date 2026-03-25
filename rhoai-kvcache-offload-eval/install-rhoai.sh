#!/bin/bash
# Install Red Hat OpenShift AI (RHOAI) 3.3 on an OpenShift cluster.
# Prereqs expected: NVIDIA GPU Operator, NFD.
#
# Usage: bash install-rhoai.sh [--gateway]
#   --gateway   Also apply the Gateway/GatewayClass resources (requires
#               Gateway API CRDs to be present; only needed if using
#               RHOAI inference gateway for model routing)
set -euo pipefail

MANIFESTS_DIR="$(dirname "$0")/manifests/rhoai"
export KUBECONFIG="${KUBECONFIG:-$(dirname "$0")/kubeconfig}"

APPLY_GATEWAY=false
for arg in "$@"; do
  [[ "$arg" == "--gateway" ]] && APPLY_GATEWAY=true
done

echo "=== RHOAI 3.3 Installation ==="
echo "Cluster:    $(kubectl cluster-info 2>/dev/null | grep 'control plane' | awk '{print $NF}')"
echo "Manifests:  ${MANIFESTS_DIR}"
echo "Gateway:    ${APPLY_GATEWAY}"
echo ""

# ── Step 1: Namespace ────────────────────────────────────────────────────────
echo "[1/5] Creating redhat-ods-operator namespace..."
kubectl apply -f "${MANIFESTS_DIR}/namespace.yaml"

# ── Step 2: OperatorGroup + Subscription ────────────────────────────────────
echo "[2/5] Creating OperatorGroup and Subscription (channel: fast-3.x)..."
kubectl apply -f "${MANIFESTS_DIR}/subscription.yaml"

# ── Step 3: Wait for operator to be installed ────────────────────────────────
echo "[3/5] Waiting for rhods-operator CSV to reach Succeeded phase..."
echo "      (this typically takes 3-5 minutes)"

TIMEOUT=600
INTERVAL=15
ELAPSED=0
while true; do
  PHASE=$(kubectl get csv -n redhat-ods-operator \
    -o jsonpath='{.items[?(@.spec.displayName=="Red Hat OpenShift AI")].status.phase}' 2>/dev/null || true)
  if [[ "$PHASE" == "Succeeded" ]]; then
    echo "      ✓ rhods-operator CSV is Succeeded"
    # Wait for the webhook service to be ready before applying DSCI/DSC,
    # otherwise the conversion webhook call will fail.
    echo "      Waiting for rhods-operator webhook service..."
    kubectl wait --for=condition=Available --timeout=120s \
      deployment/rhods-operator -n redhat-ods-operator 2>/dev/null || true
    sleep 10
    break
  fi
  if (( ELAPSED >= TIMEOUT )); then
    echo "ERROR: Timed out waiting for rhods-operator CSV after ${TIMEOUT}s"
    kubectl get csv -n redhat-ods-operator 2>/dev/null || true
    exit 1
  fi
  echo "      Phase: '${PHASE:-pending}' — waiting ${INTERVAL}s (${ELAPSED}/${TIMEOUT}s elapsed)"
  sleep $INTERVAL
  ELAPSED=$(( ELAPSED + INTERVAL ))
done

# ── Step 4: DSCI + DSC ───────────────────────────────────────────────────────
echo "[4/5] Creating DataScienceClusterInitialization (DSCI)..."
kubectl apply -f "${MANIFESTS_DIR}/dsci.yaml"

echo "      Waiting for DSCI to be Ready..."
ELAPSED=0
while true; do
  PHASE=$(kubectl get dscinitializations.dscinitialization.opendatahub.io default-dsci \
    -o jsonpath='{.status.phase}' 2>/dev/null || true)
  if [[ "$PHASE" == "Ready" ]]; then
    echo "      ✓ DSCI is Ready"
    break
  fi
  if (( ELAPSED >= TIMEOUT )); then
    echo "ERROR: Timed out waiting for DSCI after ${TIMEOUT}s"
    kubectl describe dscinitializations.dscinitialization.opendatahub.io default-dsci 2>/dev/null || true
    exit 1
  fi
  echo "      Phase: '${PHASE:-pending}' — waiting ${INTERVAL}s (${ELAPSED}/${TIMEOUT}s elapsed)"
  sleep $INTERVAL
  ELAPSED=$(( ELAPSED + INTERVAL ))
done

echo "      Creating DataScienceCluster (DSC)..."
kubectl apply -f "${MANIFESTS_DIR}/dsc.yaml"

echo "      Waiting for DSC to be Ready (kserve, dashboard, modelcontroller)..."
ELAPSED=0
while true; do
  PHASE=$(kubectl get datascienceclusters.datasciencecluster.opendatahub.io default-dsc \
    -o jsonpath='{.status.phase}' 2>/dev/null || true)
  if [[ "$PHASE" == "Ready" ]]; then
    echo "      ✓ DSC is Ready"
    break
  fi
  if (( ELAPSED >= TIMEOUT )); then
    echo "ERROR: Timed out waiting for DSC after ${TIMEOUT}s"
    kubectl describe datascienceclusters.datasciencecluster.opendatahub.io default-dsc 2>/dev/null || true
    exit 1
  fi
  echo "      Phase: '${PHASE:-pending}' — waiting ${INTERVAL}s (${ELAPSED}/${TIMEOUT}s elapsed)"
  sleep $INTERVAL
  ELAPSED=$(( ELAPSED + INTERVAL ))
done

# ── Step 5: Gateway (optional) ───────────────────────────────────────────────
if [[ "$APPLY_GATEWAY" == "true" ]]; then
  echo "[5/5] Applying GatewayClass and Gateway resources..."
  kubectl apply -f "${MANIFESTS_DIR}/gatewayclass.yaml"
  kubectl apply -f "${MANIFESTS_DIR}/gateway.yaml"
  APPS_DOMAIN=$(kubectl get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}' 2>/dev/null || echo "<apps-domain>")
  echo "      Gateway hostname: inference-gateway.${APPS_DOMAIN}"
else
  echo "[5/5] Skipping Gateway resources (pass --gateway to apply)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== RHOAI 3.3 Installation Complete ==="
echo ""
echo "Operator namespace:     redhat-ods-operator"
echo "Applications namespace: redhat-ods-applications"
echo "Monitoring namespace:   redhat-ods-monitoring"
echo ""
echo "Verify installation:"
echo "  kubectl get csv -n redhat-ods-operator"
echo "  kubectl get dscinitializations,datascienceclusters -A"
echo "  kubectl get pods -n redhat-ods-applications"
echo ""
echo "Dashboard URL:"
echo "  kubectl get route -n redhat-ods-applications rhods-dashboard -o jsonpath='{.spec.host}'"
echo ""
echo "KServe InferenceService CRD:"
echo "  kubectl get crd inferenceservices.serving.kserve.io"
