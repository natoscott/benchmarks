# Setting up kvcache-storage-pvc with correct IOPS

## Background

The kvcache PVC uses `ibmc-vpc-block-custom` storage class, which allows
custom IOPS to be set at provision time via the `ibm.io/iops` annotation.

## Critical gotcha: silent IOPS fallback

**IBM Cloud silently ignores out-of-range IOPS values and falls back to the
minimum (400 IOPS) without any error or warning.**

The maximum allowed IOPS depends on the volume capacity:

| Capacity | Max IOPS |
|----------|----------|
| 256 GiB  | 6,000    |
| 1 TiB    | ~48,000  |

The first provision of this PVC used `ibm.io/iops: "64000"`, which is above
the 6,000 limit for a 256 GiB volume. IBM Cloud silently provisioned it at
400 IOPS instead. This went unnoticed until benchmark disk I/O rates were
examined — the storage appeared functional, just operating at minimum throughput.

See `manifests/kvcache-storage-pvc.yaml` for the corrected annotation (6000).

## Procedure for a new cluster / re-provisioned PVC

### Step 1: Apply the PVC manifest

```bash
kubectl apply -f manifests/kvcache-storage-pvc.yaml
```

The manifest has `ibm.io/iops: "6000"` — the verified maximum for 256 GiB
in IBM Cloud VPC (au-syd region). Do not increase this value without
checking the IBM Cloud docs for your region and capacity.

### Step 2: Verify provisioned IOPS via ibmcloud CLI

The Kubernetes annotation reflects what was *requested*, not what was
*provisioned*. Always verify via the ibmcloud CLI:

```bash
# Find the volume ID from the PV
VOLUME_ID=$(kubectl get pv $(kubectl get pvc kvcache-storage-pvc -n llm-d-pfc-cpu \
  -o jsonpath='{.spec.volumeName}') \
  -o jsonpath='{.spec.csi.volumeHandle}')

echo "Volume ID: $VOLUME_ID"

# Query the actual provisioned IOPS
ibmcloud is volume $VOLUME_ID --output json | jq '{iops, capacity, status}'
```

Expected output for a correctly provisioned 256 GiB volume:
```json
{
  "iops": 6000,
  "capacity": 256,
  "status": "available"
}
```

If `iops` shows 400, the requested value was out of range and fell back
to the minimum. Delete the PVC/PV, correct the annotation, and re-provision.

### Step 3: Update IOPS on an existing volume (if already provisioned wrong)

If the PVC is already provisioned at the wrong IOPS (e.g. 400 instead of
6000), re-provisioning requires deleting the PVC. As an alternative, the
underlying block volume can be updated in-place via:

```bash
ibmcloud is volume-update $VOLUME_ID --iops 6000
```

**Notes on in-place update:**
- The volume must be attached (not detached) for the update to apply cleanly
- Try candidate IOPS values in decreasing order — if a value is rejected,
  try the next lower one. There is no API to query the exact maximum for a
  given capacity/region combination ahead of time.
- After updating, re-run Step 2 to confirm the new IOPS took effect.
- The Kubernetes PVC annotation will still show the old value; it is metadata
  only and does not reflect the backend volume state.

### Step 4: Run a quick fio verification

Before running benchmarks, confirm the actual storage throughput matches
expectations. A fio job manifest is available at `manifests/fio.yaml`.

```bash
kubectl apply -f manifests/fio.yaml
kubectl logs -f -l job-name=kvcache-storage-fio-6kiops-q256 -n llm-d-pfc-cpu
```

At 6,000 IOPS with 256-byte blocks (matching the fs-offload block size),
expect ~150 MB/s sequential write throughput.

## Summary checklist

- [ ] PVC annotation is `ibm.io/iops: "6000"` (not 64000)
- [ ] `ibmcloud is volume $VOLUME_ID` shows `iops: 6000`
- [ ] fio confirms ~150 MB/s write at 256-byte blocks
- [ ] All fs-offload benchmarks use `threads_per_gpu=128`
