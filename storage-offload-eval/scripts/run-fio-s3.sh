#!/bin/bash
# Run FIO S3/RGW sections against Ceph RGW from inside fio-pod.
# PCP archives are collected alongside FIO results for system-level analysis.
#
# Fetches RGW credentials from the Rook secret in openshift-storage, generates
# a runnable config (S3 sections only, credentials substituted), copies it into
# fio-pod, runs fio, and transfers results back via transfer-large-file-chunked.sh.
#
# USAGE:
#   KUBECONFIG=/path/to/kubeconfig ./scripts/run-fio-s3.sh
#
# ENVIRONMENT:
#   KUBECONFIG   path to kubeconfig (required)
#   NAMESPACE    ODF namespace containing the Rook secret (default: openshift-storage)
#   POD_NS       fio-pod namespace (default: storage-offload-eval)
#   BUCKET       S3 bucket name (default: kvcache)
#   RESULTS      output filename under results/ (default: results-s3.json)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER="${SCRIPT_DIR}/../../scripts/transfer-large-file-chunked.sh"
FIO_CONFIG="${SCRIPT_DIR}/../fio/fio-kv.fio"
RESULTS_DIR="${SCRIPT_DIR}/../results"

KC="kubectl ${KUBECONFIG:+--kubeconfig ${KUBECONFIG}}"
NS="${NAMESPACE:-openshift-storage}"
POD_NS="${POD_NS:-storage-offload-eval}"
POD="${POD:-fio-pod}"
BUCKET="${BUCKET:-kvcache}"
STORE="kvcache-store"
USER_NAME="kvcache-user"
SECRET="rook-ceph-object-user-${STORE}-${USER_NAME}"
RESULTS="${RESULTS:-results-s3.json}"

echo "==> Fetching RGW credentials from secret ${SECRET} in ${NS}..."
ACCESS_KEY=$($KC -n "$NS" get secret "$SECRET" -o jsonpath='{.data.AccessKey}' | base64 -d)
SECRET_KEY=$($KC -n "$NS" get secret "$SECRET" -o jsonpath='{.data.SecretKey}' | base64 -d)
# Use stable Service DNS name rather than ClusterIP (survives ODF reconcile cycles).
RGW_HOST="rook-ceph-rgw-${STORE}.${NS}.svc.cluster.local"

# Locate the PCP pod (must be co-deployed for system-level metrics).
PCP_POD=$($KC -n "$POD_NS" get pods -l app=pcp --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -z "$PCP_POD" ]; then
  echo "WARNING: No running PCP pod found in ${POD_NS}."
  echo "  Deploy manifests/pcp-serviceaccount.yaml + manifests/pcp-deployment.yaml before benchmarking."
else
  echo "==> PCP pod: ${PCP_POD} (collecting system metrics)"
fi

TMPFILE=$(mktemp /tmp/fio-kv-s3-XXXXXX.fio)
trap 'rm -f "$TMPFILE"' EXIT

# Extract and uncomment only section headers ([name]) and key=value lines.
# Lines where the semicolon is immediately followed by a non-space character
# are config lines; comment prose always has a space after the semicolon.
# The === anchor is ASCII-only to avoid LANG=C awk failures with Unicode.
{
  cat <<'GLOBAL'
[global]
runtime=60
time_based=1
group_reporting=1
lat_percentiles=1

GLOBAL

  awk '/^; === S3\/RGW sections/,0 {
    if (/^;[^ ]/) { sub(/^;/, ""); print }
  }' "$FIO_CONFIG" |
    sed "s/PLACEHOLDER_RGW_HOST/${RGW_HOST}/g" |
    sed "s/PLACEHOLDER_S3_ACCESS_KEY/${ACCESS_KEY}/g" |
    sed "s/PLACEHOLDER_S3_SECRET_KEY/${SECRET_KEY}/g"
} > "$TMPFILE"

echo "==> Copying FIO config to ${POD}..."
$KC -n "$POD_NS" cp "$TMPFILE" "${POD}:/tmp/fio-kv-s3.fio"

echo "==> Running FIO against RGW (${RGW_HOST})..."
$KC -n "$POD_NS" exec "$POD" -- \
  fio /tmp/fio-kv-s3.fio \
    --output-format=json+ \
    --output=/tmp/${RESULTS}

echo "==> Transferring FIO results..."
mkdir -p "$RESULTS_DIR"
"${TRANSFER}" \
  "${KUBECONFIG:?KUBECONFIG must be set}" \
  "$POD_NS" "$POD" \
  "/tmp/${RESULTS}" \
  "${RESULTS_DIR}/${RESULTS}"

# Collect PCP archives for correlation with FIO results.
if [ -n "$PCP_POD" ]; then
  echo "==> Collecting PCP archives from ${PCP_POD}..."
  ARCHIVE_NAME="pcp-fio-s3-$(date +%Y%m%d-%H%M%S).tar.gz"
  $KC -n "$POD_NS" exec "$PCP_POD" -- \
    bash -c "tar czf /tmp/${ARCHIVE_NAME} --ignore-failed-read -C /var/log/pcp/pmlogger ."
  "${TRANSFER}" \
    "${KUBECONFIG}" "$POD_NS" "$PCP_POD" \
    "/tmp/${ARCHIVE_NAME}" \
    "${RESULTS_DIR}/${ARCHIVE_NAME}"
  echo "    PCP archive: results/${ARCHIVE_NAME}"
fi

echo ""
echo "Results: results/${RESULTS}"
echo ""
echo "p99 read latencies (j=16):"
echo "  jq '[.jobs[] | select(.jobname | test(\"s3.*read-j16\")) | {job: .jobname, p99_ms: (.read.lat_ns.percentile[\"99.000000\"] / 1e6)}]' results/${RESULTS}"
