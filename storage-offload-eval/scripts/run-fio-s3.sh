#!/bin/bash
# Run FIO S3/RGW sections against Ceph RGW from inside fio-pod.
# PCP is restarted fresh before the run so the archive covers exactly this run.
#
# Two-phase execution to ensure reads measure cold-storage latency:
#   Phase 1: write sections (PUT objects into RGW, stored on OSD)
#   Cache drop: ceph tell osd.0/1 cache drop (clears OSD BlueStore cache)
#   Phase 2: read sections (GET objects — cold reads from VPC block via RGW)
#
# USAGE:
#   KUBECONFIG=/path/to/kubeconfig ./scripts/run-fio-s3.sh
#
# ENVIRONMENT:
#   KUBECONFIG   path to kubeconfig (required)
#   NAMESPACE    ODF namespace for Rook secret (default: openshift-storage)
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
STORE="kvcache-store"
USER_NAME="kvcache-user"
SECRET="rook-ceph-object-user-${STORE}-${USER_NAME}"
RESULTS="${RESULTS:-results-s3.json}"

echo "==> Fetching RGW credentials..."
ACCESS_KEY=$($KC -n "$NS" get secret "$SECRET" -o jsonpath='{.data.AccessKey}' | base64 -d)
SECRET_KEY=$($KC -n "$NS" get secret "$SECRET" -o jsonpath='{.data.SecretKey}' | base64 -d)
RGW_HOST="rook-ceph-rgw-${STORE}.${NS}.svc.cluster.local"

MGR_POD=$($KC -n "$NS" get pods -l app=rook-ceph-mgr \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
[ -z "$MGR_POD" ] && { echo "ERROR: no running Ceph MGR pod in ${NS}"; exit 1; }
echo "==> Ceph MGR pod: ${MGR_POD}"

# ── Restart PCP DaemonSet for clean archives ─────────────────────────────────
echo "==> Restarting PCP DaemonSet for clean archives..."
$KC -n "$POD_NS" rollout restart daemonset/pcp 2>/dev/null
$KC -n "$POD_NS" rollout status daemonset/pcp --timeout=120s 2>/dev/null
PCP_PODS=$($KC -n "$POD_NS" get pods -l app=pcp --field-selector=status.phase=Running \
  -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
[ -z "$PCP_PODS" ] && echo "WARNING: No PCP pods ready — continuing without metrics collection."
[ -n "$PCP_PODS" ] && echo "==> PCP pods: ${PCP_PODS}"

# Generate S3 fio config.
TMPFILE=$(mktemp /tmp/fio-kv-s3-XXXXXX.fio)
trap 'rm -f "$TMPFILE"' EXIT
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

WRITE_SECTIONS=(
  --section=qwen8b-s3-write-j1  --section=qwen8b-s3-write-j16
  --section=gpt120b-s3-write-j1 --section=gpt120b-s3-write-j16
  --section=fp8-70b-s3-write-j1 --section=fp8-70b-s3-write-j16
)
READ_SECTIONS=(
  --section=qwen8b-s3-read-j1   --section=qwen8b-s3-read-j16
  --section=gpt120b-s3-read-j1  --section=gpt120b-s3-read-j16
  --section=fp8-70b-s3-read-j1  --section=fp8-70b-s3-read-j16
)

echo "==> Copying FIO config to ${POD}..."
$KC -n "$POD_NS" cp "$TMPFILE" "${POD}:/tmp/fio-kv-s3.fio"
mkdir -p "$RESULTS_DIR"

FIO_START=$(date -u "+%Y-%m-%d %H:%M:%S")

# ── Phase 1: Writes ───────────────────────────────────────────────────────────
echo ""
echo "==> Phase 1: Write sections (PUT objects into RGW)..."
$KC -n "$POD_NS" exec "$POD" -- \
  fio /tmp/fio-kv-s3.fio \
    "${WRITE_SECTIONS[@]}" \
    --output-format=json+ \
    --output=/tmp/writes-s3.json

# ── Cache drop ────────────────────────────────────────────────────────────────
echo ""
echo "==> Dropping OSD BlueStore cache..."
$KC -n "$NS" exec "$MGR_POD" -c watch-active -- \
  bash -c "ceph tell osd.0 cache drop && ceph tell osd.1 cache drop" 2>/dev/null
echo "    Waiting 10s for OSD to stabilise..."
sleep 10

# ── Phase 2: Reads ────────────────────────────────────────────────────────────
echo ""
echo "==> Phase 2: Read sections (cold reads from VPC block via RGW)..."
$KC -n "$POD_NS" exec "$POD" -- \
  fio /tmp/fio-kv-s3.fio \
    "${READ_SECTIONS[@]}" \
    --output-format=json+ \
    --output=/tmp/reads-s3.json

# ── Merge and transfer ────────────────────────────────────────────────────────
echo ""
echo "==> Merging results..."
$KC -n "$POD_NS" exec "$POD" -- \
  bash -c "python3 -c \"
import json, sys
with open('/tmp/writes-s3.json') as f: w = json.load(f)
with open('/tmp/reads-s3.json') as f:  r = json.load(f)
w['jobs'] = w['jobs'] + r['jobs']
json.dump(w, sys.stdout)
\" > /tmp/${RESULTS}"

"${TRANSFER}" \
  "${KUBECONFIG:?KUBECONFIG must be set}" \
  "$POD_NS" "$POD" "/tmp/${RESULTS}" "${RESULTS_DIR}/${RESULTS}"

FIO_END=$(date -u "+%Y-%m-%d %H:%M:%S")

# ── PCP extracts — focused metrics, FIO window only, one per node ─────────────
if [ -n "$PCP_PODS" ]; then
  ARCHIVE_TS=$(date +%Y%m%d-%H%M%S)
  WANTED_CONF=$(mktemp /tmp/pcp-wanted-XXXXXX.conf)
  cat > "$WANTED_CONF" << 'METRICS'
disk.dev.read_bytes
disk.dev.write_bytes
disk.dev.util
disk.dev.r_await
disk.dev.w_await
network.all.in.bytes
network.all.out.bytes
kernel.all.cpu.user
kernel.all.cpu.sys
openmetrics.ceph_mgr.ceph_health_status
openmetrics.ceph_mgr.ceph_osd_op_r
openmetrics.ceph_mgr.ceph_osd_op_w
openmetrics.ceph_mgr.ceph_pool_rd_bytes
openmetrics.ceph_mgr.ceph_pool_wr_bytes
METRICS
  for PCP_POD in $PCP_PODS; do
    NODE=$($KC -n "$POD_NS" get pod "$PCP_POD" -o jsonpath='{.spec.nodeName}' 2>/dev/null | \
      awk -F- '{print $(NF-1)"-"$NF}')
    EXTRACT="pcp-fio-s3-${NODE}-${ARCHIVE_TS}"
    echo "==> Extracting PCP metrics from ${PCP_POD} (${NODE}) [${FIO_START} → ${FIO_END}]..."
    $KC -n "$POD_NS" cp "$WANTED_CONF" "${PCP_POD}:/tmp/wanted.conf" 2>/dev/null
    $KC -n "$POD_NS" exec "$PCP_POD" -- bash -c "
      ARCH=\$(ls /var/log/pcp/pmlogger/\${HOSTNAME}/*.meta 2>/dev/null | head -1 | sed 's/\.meta//')
      pmlogextract -S '${FIO_START}' -T '${FIO_END}' \
        -c /tmp/wanted.conf \"\${ARCH}\" /tmp/${EXTRACT} 2>/dev/null
      tar --zstd -cf /tmp/${EXTRACT}.tar.zst \
        /tmp/${EXTRACT}.meta /tmp/${EXTRACT}.index /tmp/${EXTRACT}.0 2>/dev/null; true
    " 2>/dev/null
    "${TRANSFER}" "${KUBECONFIG}" "$POD_NS" "$PCP_POD" \
      "/tmp/${EXTRACT}.tar.zst" "${RESULTS_DIR}/${EXTRACT}.tar.zst" $((256 * 1024))
    echo "    PCP extract: results/${EXTRACT}.tar.zst"
  done
  rm -f "$WANTED_CONF"
fi

echo ""
echo "Results: results/${RESULTS}"
echo "p99 read latencies (j=16):"
echo "  jq '[.jobs[] | select(.jobname | test(\"s3.*read-j16\")) | {job: .jobname, p99_ms: (.read.lat_ns.percentile[\"99.000000\"] / 1e6)}]' results/${RESULTS}"
