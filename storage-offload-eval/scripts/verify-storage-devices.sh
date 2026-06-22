#!/bin/bash
# Verify storage device types across the fio test volume and Ceph OSD volumes.
#
# Checks the ROTA (rotational) flag for every relevant block device:
#   ROTA=0 -> SSD / NVMe  (expected for ibmc-vpc-block-10iops-tier)
#   ROTA=1 -> spinning disk (results are not comparable across different ROTA values)
#
# Run this before benchmarking and after Ceph is deployed. If any device shows
# ROTA=1, investigate before proceeding — benchmark comparisons will be invalid.
#
# USAGE:
#   KUBECONFIG=/path/to/kubeconfig ./scripts/verify-storage-devices.sh
#
# ENVIRONMENT:
#   KUBECONFIG   path to kubeconfig (required)
#   NS_BENCH     fio-pod namespace (default: storage-offload-eval)
#   NS_ODF       ODF namespace (default: openshift-storage)
#   POD          fio pod name (default: fio-pod)
set -euo pipefail

KC="kubectl ${KUBECONFIG:+--kubeconfig ${KUBECONFIG}}"
NS_BENCH="${NS_BENCH:-storage-offload-eval}"
NS_ODF="${NS_ODF:-openshift-storage}"
POD="${POD:-fio-pod}"

ISSUES=0

echo "========================================================"
echo " Storage device verification — ROTA flag check"
echo "========================================================"
echo ""

# ── fio-pod: fio-vpc-block raw block device ───────────────────────────────────
echo "── fio-pod (${NS_BENCH}/${POD}) ──────────────────────────────────────────"
echo ""
echo "All block devices visible in fio-pod:"
$KC -n "$NS_BENCH" exec "$POD" -- \
  lsblk -d -o NAME,ROTA,SIZE,TYPE,MODEL 2>/dev/null || \
  echo "  (lsblk failed — is fio-pod running?)"
echo ""

# Check specifically for ROTA=1 (spinning disk) among disk-type devices
ROTATIONAL_DEVS=$($KC -n "$NS_BENCH" exec "$POD" -- \
  lsblk -d -o NAME,ROTA,TYPE -n 2>/dev/null | \
  awk '$2=="1" && $3=="disk" {print $1}' || true)
if [ -n "$ROTATIONAL_DEVS" ]; then
  echo "  WARNING: spinning disk detected: ${ROTATIONAL_DEVS}"
  ISSUES=$((ISSUES + 1))
else
  echo "  All disk devices: ROTA=0 (SSD/NVMe) [OK]"
fi
echo ""

# ── Ceph OSD pods ─────────────────────────────────────────────────────────────
echo "── Ceph OSD pods (${NS_ODF}) ─────────────────────────────────────────────"
echo ""
OSD_PODS=$($KC -n "$NS_ODF" get pods -l app=rook-ceph-osd \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)

if [ -z "$OSD_PODS" ]; then
  echo "  No running OSD pods found."
  echo "  Re-run after Ceph is deployed to verify OSD device types."
else
  for OSD_POD in $OSD_PODS; do
    echo "  ${OSD_POD}:"
    $KC -n "$NS_ODF" exec "$OSD_POD" -- \
      lsblk -d -o NAME,ROTA,SIZE,TYPE 2>/dev/null | \
      grep -v "^NAME" | awk '{printf "    %-12s ROTA=%-2s SIZE=%-10s TYPE=%s\n", $1, $2, $3, $4}' \
      || echo "    (lsblk not available in this pod)"

    # Flag any spinning disks
    OSD_ROTA=$($KC -n "$NS_ODF" exec "$OSD_POD" -- \
      lsblk -d -o ROTA,TYPE -n 2>/dev/null | \
      awk '$1=="1" && $2=="disk" {print "ROTATIONAL"}' || true)
    if [ -n "$OSD_ROTA" ]; then
      echo "    WARNING: spinning disk detected in ${OSD_POD}"
      ISSUES=$((ISSUES + 1))
    fi
  done
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================================"
if [ "$ISSUES" -eq 0 ]; then
  echo " PASS — all detected devices are SSD/NVMe (ROTA=0)"
  echo " VPC block, CephFS, and RGW results are comparable."
else
  echo " FAIL — ${ISSUES} issue(s) detected (see above)"
  echo " Resolve before running benchmarks — results will not"
  echo " be comparable across different storage media types."
fi
echo "========================================================"
