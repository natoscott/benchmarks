#!/bin/bash
# Run FIO filesystem sections against the CephFS PVC mounted in fio-pod.
# PCP is restarted fresh before the run so the archive covers exactly this run.
#
# Two-phase execution to ensure reads measure cold-storage latency:
#   Phase 1: write sections (populates data on OSD)
#   Cache drop: ceph tell osd.0/1 cache drop (clears OSD BlueStore cache)
#   Phase 2: read + mixed sections (cold reads from VPC block via CephFS)
#
# O_DIRECT bypasses the client page cache but not OSD BlueStore cache.
# Without the cache drop, reads after writes are served from OSD memory,
# not from the VPC block device.
#
# USAGE:
#   KUBECONFIG=/path/to/kubeconfig ./scripts/run-fio-cephfs.sh
#
# ENVIRONMENT:
#   KUBECONFIG   path to kubeconfig (required)
#   NAMESPACE    benchmark namespace (default: storage-offload-eval)
#   ODF_NS       ODF namespace for ceph commands (default: openshift-storage)
#   POD          fio pod name (default: fio-pod)
#   RESULTS      output filename under results/ (default: results-cephfs.json)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER="${SCRIPT_DIR}/../../scripts/transfer-large-file-chunked.sh"
FIO_CONFIG="${SCRIPT_DIR}/../fio/fio-kv.fio"
RESULTS_DIR="${SCRIPT_DIR}/../results"

KC="kubectl ${KUBECONFIG:+--kubeconfig ${KUBECONFIG}}"
NS="${NAMESPACE:-storage-offload-eval}"
ODF_NS="${ODF_NS:-openshift-storage}"
POD="${POD:-fio-pod}"
RESULTS="${RESULTS:-results-cephfs.json}"

WRITE_SECTIONS=(
  --section=qwen8b-write-j1  --section=qwen8b-write-j16
  --section=gpt120b-write-j1 --section=gpt120b-write-j16
  --section=fp8-70b-write-j1 --section=fp8-70b-write-j16
)
READ_SECTIONS=(
  --section=qwen8b-read-j1   --section=qwen8b-read-j16   --section=qwen8b-mixed-j16
  --section=gpt120b-read-j1  --section=gpt120b-read-j16  --section=gpt120b-mixed-j16
  --section=fp8-70b-read-j1  --section=fp8-70b-read-j16  --section=fp8-70b-mixed-j16
)

# Find the Ceph MGR pod for cache drop.
MGR_POD=$($KC -n "$ODF_NS" get pods -l app=rook-ceph-mgr \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
[ -z "$MGR_POD" ] && { echo "ERROR: no running Ceph MGR pod in ${ODF_NS}"; exit 1; }
echo "==> Ceph MGR pod: ${MGR_POD}"

# ── Restart PCP DaemonSet for clean archives ─────────────────────────────────
echo "==> Restarting PCP DaemonSet for clean archives..."
$KC -n "$NS" rollout restart daemonset/pcp 2>/dev/null
$KC -n "$NS" rollout status daemonset/pcp --timeout=120s 2>/dev/null
PCP_PODS=$($KC -n "$NS" get pods -l app=pcp --field-selector=status.phase=Running \
  -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
[ -z "$PCP_PODS" ] && echo "WARNING: No PCP pods ready — continuing without metrics collection."
[ -n "$PCP_PODS" ] && echo "==> PCP pods: ${PCP_PODS}"

echo "==> Copying FIO config to ${POD}..."
$KC -n "$NS" cp "$FIO_CONFIG" "${POD}:/tmp/fio-kv.fio"
mkdir -p "$RESULTS_DIR"

FIO_START=$(date -u "+%Y-%m-%d %H:%M:%S")

# ── Phase 1: Writes ───────────────────────────────────────────────────────────
echo ""
echo "==> Phase 1: Write sections..."
$KC -n "$NS" exec "$POD" -- \
  fio /tmp/fio-kv.fio \
    --directory=/mnt/cephfs \
    "${WRITE_SECTIONS[@]}" \
    --output-format=json+ \
    --output=/tmp/writes-cephfs.json

# ── Cache drop ────────────────────────────────────────────────────────────────
echo ""
echo "==> Dropping OSD BlueStore cache..."
$KC -n "$ODF_NS" exec "$MGR_POD" -c watch-active -- \
  bash -c "ceph tell osd.0 cache drop && ceph tell osd.1 cache drop" 2>/dev/null
echo "    Waiting 10s for OSD to stabilise..."
sleep 10

# ── Phase 2: Reads + mixed ────────────────────────────────────────────────────
echo ""
echo "==> Phase 2: Read + mixed sections (cold reads from VPC block)..."
for attempt in 1 2 3; do
  $KC -n "$NS" exec "$POD" -- \
    fio /tmp/fio-kv.fio \
      --directory=/mnt/cephfs \
      "${READ_SECTIONS[@]}" \
      --output-format=json+ \
      --output=/tmp/reads-cephfs.json || true
  JSON_SIZE=$($KC -n "$NS" exec "$POD" -- stat -c '%s' /tmp/reads-cephfs.json 2>/dev/null || echo 0)
  [ "$JSON_SIZE" -gt 0 ] && break
  echo "  WARNING: empty output on attempt ${attempt}/3, retrying..."
  sleep 5
done
[ "$($KC -n "$NS" exec "$POD" -- stat -c '%s' /tmp/reads-cephfs.json 2>/dev/null || echo 0)" -gt 0 ] || \
  { echo "ERROR: fio Phase 2 produced no output after 3 attempts"; exit 1; }

# ── Merge and transfer ────────────────────────────────────────────────────────
echo ""
echo "==> Merging results..."
$KC -n "$NS" exec "$POD" -- \
  bash -c "python3 -c \"
import json, sys
with open('/tmp/writes-cephfs.json') as f: w = json.load(f)
with open('/tmp/reads-cephfs.json') as f:  r = json.load(f)
w['jobs'] = w['jobs'] + r['jobs']
json.dump(w, sys.stdout)
\" > /tmp/${RESULTS}"

"${TRANSFER}" \
  "${KUBECONFIG:?KUBECONFIG must be set}" \
  "$NS" "$POD" "/tmp/${RESULTS}" "${RESULTS_DIR}/${RESULTS}"

FIO_END=$(date -u "+%Y-%m-%d %H:%M:%S")

# ── PCP extracts — focused metrics, FIO window only, one per node ─────────────
# Full archives remain on node hostPath (/var/log/pcp/pmlogger) for later use.
# pmlogextract pulls only the metrics and time window needed for analysis.
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
    NODE=$($KC -n "$NS" get pod "$PCP_POD" -o jsonpath='{.spec.nodeName}' 2>/dev/null | \
      awk -F- '{print $(NF-1)"-"$NF}')
    EXTRACT="pcp-fio-cephfs-${NODE}-${ARCHIVE_TS}"
    echo "==> Extracting PCP metrics from ${PCP_POD} (${NODE}) [${FIO_START} → ${FIO_END}]..."
    $KC -n "$NS" cp "$WANTED_CONF" "${PCP_POD}:/tmp/wanted.conf" 2>/dev/null
    $KC -n "$NS" exec "$PCP_POD" -- bash -c "
      ARCH=\$(ls /var/log/pcp/pmlogger/\${HOSTNAME}/*.meta 2>/dev/null | head -1 | sed 's/\.meta//')
      pmlogextract -S '${FIO_START}' -T '${FIO_END}' \
        -c /tmp/wanted.conf \"\${ARCH}\" /tmp/${EXTRACT} 2>/dev/null
      tar --zstd -cf /tmp/${EXTRACT}.tar.zst /tmp/${EXTRACT}.meta \
        /tmp/${EXTRACT}.index /tmp/${EXTRACT}.0 2>/dev/null; true
    " 2>/dev/null
    "${TRANSFER}" "${KUBECONFIG}" "$NS" "$PCP_POD" \
      "/tmp/${EXTRACT}.tar.zst" "${RESULTS_DIR}/${EXTRACT}.tar.zst" $((256 * 1024))
    echo "    PCP extract: results/${EXTRACT}.tar.zst"
  done
  rm -f "$WANTED_CONF"
fi

echo ""
echo "Results: results/${RESULTS}"
echo "p99 read latencies (j=16):"
echo "  jq '[.jobs[] | select(.jobname | test(\"read-j16\")) | {job: .jobname, p99_ms: (.read.lat_ns.percentile[\"99.000000\"] / 1e6)}]' results/${RESULTS}"
