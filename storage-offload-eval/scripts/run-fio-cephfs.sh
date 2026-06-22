#!/bin/bash
# Run FIO filesystem sections against the CephFS PVC mounted in fio-pod.
# PCP archives are collected alongside FIO results for system-level analysis.
#
# USAGE:
#   KUBECONFIG=/path/to/kubeconfig ./scripts/run-fio-cephfs.sh
#
# ENVIRONMENT:
#   KUBECONFIG   path to kubeconfig (required; passed to transfer-large-file-chunked.sh)
#   NAMESPACE    pod namespace (default: storage-offload-eval)
#   POD          fio pod name (default: fio-pod)
#   RESULTS      FIO output filename under results/ (default: results-cephfs.json)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIO_CONFIG="${SCRIPT_DIR}/../fio/fio-kv.fio"
RESULTS_DIR="${SCRIPT_DIR}/../results"

KC="kubectl ${KUBECONFIG:+--kubeconfig ${KUBECONFIG}}"
NS="${NAMESPACE:-storage-offload-eval}"
POD="${POD:-fio-pod}"
RESULTS="${RESULTS:-results-cephfs.json}"

# Locate the PCP pod (must be co-deployed for system-level metrics).
PCP_POD=$($KC -n "$NS" get pods -l app=pcp --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -z "$PCP_POD" ]; then
  echo "WARNING: No running PCP pod found in ${NS}."
  echo "  Deploy manifests/pcp-serviceaccount.yaml + manifests/pcp-deployment.yaml before benchmarking."
else
  echo "==> PCP pod: ${PCP_POD} (collecting system metrics)"
fi

echo "==> Copying FIO config to ${POD}..."
$KC -n "$NS" cp "$FIO_CONFIG" "${POD}:/tmp/fio-kv.fio"

echo "==> Running FIO against CephFS (/mnt/cephfs)..."
$KC -n "$NS" exec "$POD" -- \
  fio /tmp/fio-kv.fio \
    --directory=/mnt/cephfs \
    --output-format=json+ \
    --output=/tmp/${RESULTS}

echo "==> Transferring FIO results..."
mkdir -p "$RESULTS_DIR"
"${SCRIPT_DIR}/transfer-large-file-chunked.sh" \
  "${KUBECONFIG:?KUBECONFIG must be set}" \
  "$NS" "$POD" \
  "/tmp/${RESULTS}" \
  "${RESULTS_DIR}/${RESULTS}"

# Collect PCP archives for correlation with FIO results.
if [ -n "$PCP_POD" ]; then
  echo "==> Collecting PCP archives from ${PCP_POD}..."
  ARCHIVE_NAME="pcp-fio-cephfs-$(date +%Y%m%d-%H%M%S).tar.gz"
  $KC -n "$NS" exec "$PCP_POD" -- \
    bash -c "tar czf /tmp/${ARCHIVE_NAME} -C /var/log/pcp/pmlogger ."
  "${SCRIPT_DIR}/transfer-large-file-chunked.sh" \
    "${KUBECONFIG}" "$NS" "$PCP_POD" \
    "/tmp/${ARCHIVE_NAME}" \
    "${RESULTS_DIR}/${ARCHIVE_NAME}"
  echo "    PCP archive: results/${ARCHIVE_NAME}"
fi

echo ""
echo "Results: results/${RESULTS}"
echo ""
echo "p99 read latencies (j=16):"
echo "  jq '[.jobs[] | select(.jobname | test(\"read-j16\")) | {job: .jobname, p99_ms: (.read.lat_ns.percentile[\"99.000000\"] / 1e6)}]' results/${RESULTS}"
