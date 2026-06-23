#!/bin/bash
# Run FIO against the raw IBM VPC block volume in fio-pod.
# Tests block device performance without filesystem or Ceph overhead —
# the baseline against which CephFS and RGW overhead is measured.
#
# Two phases:
#   1. Probe (30s, j=1, fp8-70b block size): device class fingerprint.
#      Prints p50/p99 latency — review before committing to the 15-section full run.
#      SSD/NVMe: write p50 < 1ms. SATA SSD: 1-5ms. Spinning disk: > 5ms.
#   2. Full suite: 15 sections (3 models × write/read/mixed × j1/j16), 60s each.
#
# NOTE: Phase 1 prompts for confirmation before Phase 2. Set SKIP_CONFIRM=1 to bypass.
#
# USAGE:
#   KUBECONFIG=/path/to/kubeconfig ./scripts/run-fio-vpc-block.sh
#
# ENVIRONMENT:
#   KUBECONFIG    path to kubeconfig (required)
#   NAMESPACE     pod namespace (default: storage-offload-eval)
#   POD           fio pod name (default: fio-pod)
#   RESULTS       output filename under results/ (default: results-vpc-block.json)
#   SKIP_CONFIRM  set to 1 to skip the probe confirmation prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER="${SCRIPT_DIR}/../../scripts/transfer-large-file-chunked.sh"
FIO_BLOCK_CONFIG="${SCRIPT_DIR}/../fio/fio-kv-block.fio"
RESULTS_DIR="${SCRIPT_DIR}/../results"

KC="kubectl ${KUBECONFIG:+--kubeconfig ${KUBECONFIG}}"
NS="${NAMESPACE:-storage-offload-eval}"
POD="${POD:-fio-pod}"
RESULTS="${RESULTS:-results-vpc-block.json}"
DEVICE="/dev/vpc-block"
SKIP_CONFIRM="${SKIP_CONFIRM:-}"

# ── Restart PCP DaemonSet for clean archives ─────────────────────────────────
echo "==> Restarting PCP DaemonSet for clean archives..."
$KC -n "$NS" rollout restart daemonset/pcp 2>/dev/null
$KC -n "$NS" rollout status daemonset/pcp --timeout=120s 2>/dev/null
PCP_PODS=$($KC -n "$NS" get pods -l app=pcp --field-selector=status.phase=Running \
  -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
[ -z "$PCP_PODS" ] && echo "WARNING: No PCP pods ready — continuing without metrics collection."
[ -n "$PCP_PODS" ] && echo "==> PCP pods: ${PCP_PODS}"

mkdir -p "$RESULTS_DIR"

# ── Phase 1: Device probe ─────────────────────────────────────────────────────
echo ""
echo "==> Phase 1: Device probe (j=1, 30s, bs=5m) — device class fingerprint..."

PROBE_CONFIG=$(mktemp /tmp/fio-probe-XXXXXX.fio)
trap 'rm -f "$PROBE_CONFIG"' EXIT

cat > "$PROBE_CONFIG" <<EOF
[global]
direct=1
ioengine=libaio
iodepth=1
size=225m
runtime=30
time_based=1
group_reporting=1
lat_percentiles=1
filename=${DEVICE}

[probe-write-j1]
rw=write
bs=5m
numjobs=1
stonewall

[probe-read-j1]
rw=read
bs=5m
numjobs=1
stonewall
EOF

$KC -n "$NS" cp "$PROBE_CONFIG" "${POD}:/tmp/fio-probe.fio"
$KC -n "$NS" exec "$POD" -- \
  fio /tmp/fio-probe.fio \
    --output-format=json+ \
    --output=/tmp/probe-vpc-block.json

"${TRANSFER}" \
  "${KUBECONFIG:?KUBECONFIG must be set}" \
  "$NS" "$POD" \
  "/tmp/probe-vpc-block.json" \
  "${RESULTS_DIR}/probe-vpc-block.json"

echo ""
echo "Device fingerprint (p50 / p99 latency at j=1, bs=5m):"
jq -r '
  .jobs[] |
  if .jobname == "probe-write-j1" then
    "  write  p50=\(.write.lat_ns.percentile["50.000000"] / 1e6 | . * 100 | round / 100)ms  p99=\(.write.lat_ns.percentile["99.000000"] / 1e6 | . * 100 | round / 100)ms  bw=\(.write.bw_bytes / 1048576 | . * 10 | round / 10)MB/s"
  elif .jobname == "probe-read-j1" then
    "  read   p50=\(.read.lat_ns.percentile["50.000000"] / 1e6 | . * 100 | round / 100)ms  p99=\(.read.lat_ns.percentile["99.000000"] / 1e6 | . * 100 | round / 100)ms  bw=\(.read.bw_bytes / 1048576 | . * 10 | round / 10)MB/s"
  else empty end
' "${RESULTS_DIR}/probe-vpc-block.json"

echo ""
echo "  Storage class indicators (read bandwidth is the reliable discriminator):"
echo "    Local NVMe:           read bw ~3-7 GB/s,   write/read p50 < 1ms"
echo "    Network-attached SSD: read bw 300-800 MB/s, write/read p50 5-20ms (network RTT)"
echo "    HDD:                  read bw < 200 MB/s,   p50 > 5ms"
echo ""
echo "  NOTE: latency alone does not determine usefulness for KV cache offload."
echo "  The relevant comparison is cache-hit latency vs prefix recomputation cost,"
echo "  which scales with context length and model size. Even high-latency tiers"
echo "  can be beneficial for long-context workloads."

if [ -z "$SKIP_CONFIRM" ]; then
  echo ""
  read -r -p "Proceed with full benchmark suite (~15 min)? [y/N] " CONFIRM
  [[ "${CONFIRM}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

FIO_START=$(date -u "+%Y-%m-%d %H:%M:%S")

# ── Phase 2: Full benchmark suite ─────────────────────────────────────────────
echo ""
echo "==> Phase 2: Full suite (15 sections × 60s)..."

TMPFILE=$(mktemp /tmp/fio-vpc-block-XXXXXX.fio)
trap 'rm -f "$PROBE_CONFIG" "$TMPFILE"' EXIT

sed "s|PLACEHOLDER_DEVICE|${DEVICE}|g" "$FIO_BLOCK_CONFIG" > "$TMPFILE"

$KC -n "$NS" cp "$TMPFILE" "${POD}:/tmp/fio-kv-block.fio"
$KC -n "$NS" exec "$POD" -- \
  fio /tmp/fio-kv-block.fio \
    --output-format=json+ \
    --output=/tmp/${RESULTS}

echo "==> Transferring results..."
"${TRANSFER}" \
  "${KUBECONFIG}" "$NS" "$POD" \
  "/tmp/${RESULTS}" \
  "${RESULTS_DIR}/${RESULTS}"

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
    NODE=$($KC -n "$NS" get pod "$PCP_POD" -o jsonpath='{.spec.nodeName}' 2>/dev/null | \
      awk -F- '{print $(NF-1)"-"$NF}')
    EXTRACT="pcp-fio-vpc-block-${NODE}-${ARCHIVE_TS}"
    echo "==> Extracting PCP metrics from ${PCP_POD} (${NODE}) [${FIO_START} → ${FIO_END}]..."
    $KC -n "$NS" cp "$WANTED_CONF" "${PCP_POD}:/tmp/wanted.conf" 2>/dev/null
    $KC -n "$NS" exec "$PCP_POD" -- bash -c "
      ARCH=\$(ls /var/log/pcp/pmlogger/\${HOSTNAME}/*.meta 2>/dev/null | head -1 | sed 's/\.meta//')
      pmlogextract -S '${FIO_START}' -T '${FIO_END}' \
        -c /tmp/wanted.conf \"\${ARCH}\" /tmp/${EXTRACT} 2>/dev/null
      tar --zstd -cf /tmp/${EXTRACT}.tar.zst \
        /tmp/${EXTRACT}.meta /tmp/${EXTRACT}.index /tmp/${EXTRACT}.0 2>/dev/null; true
    " 2>/dev/null
    "${TRANSFER}" "${KUBECONFIG}" "$NS" "$PCP_POD" \
      "/tmp/${EXTRACT}.tar.zst" "${RESULTS_DIR}/${EXTRACT}.tar.zst" $((256 * 1024))
    echo "    PCP extract: results/${EXTRACT}.tar.zst"
  done
  rm -f "$WANTED_CONF"
fi

echo ""
echo "Results: results/${RESULTS}  (probe: results/probe-vpc-block.json)"
echo ""
echo "p99 read latencies (j=16):"
echo "  jq '[.jobs[] | select(.jobname | test(\"blk.*read-j16\")) | {job: .jobname, p99_ms: (.read.lat_ns.percentile[\"99.000000\"] / 1e6)}]' results/${RESULTS}"
