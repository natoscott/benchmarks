#!/bin/bash
# Check Ceph OSD commit/apply latency via the Rook toolbox pod.
#
# Use after the full Ceph stack is deployed. Compare commit_latency_ms to the
# probe-vpc-block.json write p50 latency:
#   commit_latency_ms ≈ probe write p50  -> OSD and fio-vpc-block on equivalent hardware
#   commit_latency_ms >> probe write p50 -> OSD volumes on slower storage class
#
# Enables the ODF Ceph toolbox if not already running.
#
# USAGE:
#   KUBECONFIG=/path/to/kubeconfig ./scripts/check-ceph-osd-perf.sh
#
# ENVIRONMENT:
#   KUBECONFIG   path to kubeconfig (required)
#   NS           ODF namespace (default: openshift-storage)
set -euo pipefail

KC="kubectl ${KUBECONFIG:+--kubeconfig ${KUBECONFIG}}"
NS="${NS:-openshift-storage}"

# Enable Ceph toolbox via OCSInitialization if not already running.
TOOLBOX=$($KC -n "$NS" get pods -l app=rook-ceph-tools \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

if [ -z "$TOOLBOX" ]; then
  echo "==> Enabling Ceph toolbox via OCSInitialization..."
  $KC -n "$NS" patch ocsinit ocsinit \
    --type=json \
    -p '[{"op":"replace","path":"/spec/enableCephTools","value":true}]' \
    2>/dev/null || \
  $KC -n "$NS" patch ocsinit ocsinit \
    --type=merge \
    -p '{"spec":{"enableCephTools":true}}' || {
    echo "ERROR: Could not patch OCSInitialization."
    echo "  Enable manually: oc -n ${NS} patch ocsinit ocsinit --type=merge -p '{\"spec\":{\"enableCephTools\":true}}'"
    exit 1
  }
  echo "    Waiting for toolbox pod..."
  until TOOLBOX=$($KC -n "$NS" get pods -l app=rook-ceph-tools \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) && [ -n "$TOOLBOX" ]; do
    sleep 5
  done
  echo "    Toolbox ready: ${TOOLBOX}"
fi

echo ""
echo "==> Ceph OSD performance (via ${TOOLBOX})..."
echo ""
echo "--- ceph osd perf ---"
echo "(commit_latency_ms is primarily OSD block device write latency)"
$KC -n "$NS" exec "$TOOLBOX" -- ceph osd perf
echo ""

echo "--- ceph osd status ---"
$KC -n "$NS" exec "$TOOLBOX" -- ceph osd status
echo ""

echo "--- ceph health detail ---"
$KC -n "$NS" exec "$TOOLBOX" -- ceph health detail
echo ""

echo "==> Interpretation:"
echo "  commit_latency_ms ~ probe-vpc-block.json write p50_ms  -> equivalent hardware class"
echo "  commit_latency_ms >> probe-vpc-block.json write p50_ms -> OSD volumes on slower media"
echo ""
echo "  Reference: results/probe-vpc-block.json"
if [ -f "$(dirname "$0")/../results/probe-vpc-block.json" ]; then
  jq '[.jobs[] | select(.jobname=="probe-write-j1") |
    {probe_write_p50_ms: (.write.lat_ns.percentile["50.000000"] / 1e6),
     probe_write_p99_ms: (.write.lat_ns.percentile["99.000000"] / 1e6)}]' \
    "$(dirname "$0")/../results/probe-vpc-block.json" 2>/dev/null || true
fi
