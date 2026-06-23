# Storage Characterisation for vLLM KV Cache Offload

## TL;DR

Three Ceph cold-tier backends on IBM VPC block (network-attached SSD) were
characterised using fio I/O patterns matching the vLLM KV cache connector.

KV cache block restoration — read p50 / p99 at j=16 (ms):

| Model (block size) | NVMe ref¹ | VPC block | CephFS | RGW/S3 |
|---|---|---|---|---|
| Qwen3-8B (2304k) | 1.4 / 2.3 | 48 / 49 | 173 / 259 | 98 / 451 |
| gpt-120b (1152k) | 2.7 / 4.0 | 24 / 25 | 88 / 108 | 57 / 144 |
| FP8-70B (5m) | 12.1 / 14.1 | 106 / 109 | 392 / 451 | 209 / 1,133 |

KV cache block eviction — write p50 / p99 at j=16 (ms):

| Model (block size) | NVMe ref¹ | VPC block | CephFS | RGW/S3 |
|---|---|---|---|---|
| Qwen3-8B (2304k) | 6.6 / 11.6 | 146 / 244 | 171 / 271 | 202 / 438 |
| gpt-120b (1152k) | 3.3 / 4.8 | 35 / 135 | 87 / 163 | 96 / 196 |
| FP8-70B (5m) | 14.5 / 21.9 | 346 / 484 | 342 / 633 | 165 / 1,250 |

¹ Separate cluster, single locally-attached PCIe Gen 4 NVMe.

VPC block reads saturate at 762 MB/s regardless of block size — the device bandwidth
ceiling. CephFS and RGW both cap at ~200–270 MB/s across all block sizes. RGW p50
latency is competitive with VPC block; the p99 tail under 16-job concurrent load is
3–10× higher. PCP disk I/O metrics confirmed cold-cache reads reached physical storage
on OSD 1 across both Ceph backends.

---

This report documents the derivation of the FIO benchmark configurations from first
principles and presents measured results across local NVMe and Ceph cold-tier storage.

The analysis draws on three sources: vLLM startup logs (GPU KV cache block structure),
PCP archives (disk-level I/O during benchmark runs), and connector source code (I/O
call patterns).

## Background

vLLM supports offloading GPU KV cache blocks to secondary storage when the GPU KV
pool is under pressure. The native filesystem connector
(`vllm.v1.kv_offload.tiering.fs`) writes evicted GPU KV cache blocks to a mounted
PVC and reads them back on cache hit. This connector landed in **vLLM 0.22.0**; the
FIO characterisation in this report targets that connector ahead of its adoption in
RHOAI.

Characterising the storage requirements requires understanding:
1. How large each GPU KV cache block is in bytes (model-dependent)
2. How the connector translates block evictions into filesystem I/O calls
3. What block size, I/O depth, and access pattern to use in FIO

A naive approach — using sequential 1 MB reads/writes — tests the *raw bandwidth
ceiling* of a storage backend but does not represent what the connector actually does.
This report derives the correct parameters from first principles.

---

## KV Cache Block Sizes from vLLM Startup Logs

vLLM uses a fixed **16-token GPU KV cache block** (the default `block_size` in
vLLM's cache configuration). The size of each block *in bytes* depends on the model's
architecture and the KV cache dtype.

The KV cache tensor shape reported in vLLM startup logs has the form:

```
Allocating a cross layer KV cache of shape
(num_blocks, num_layers, 2, kv_heads_per_rank, tokens_per_block, head_dim)
```

From this, the block size in bytes per rank is:

```
bytes_per_block_per_rank =
    num_layers × kv_heads_per_rank × head_dim × 2 (K+V) × dtype_bytes × tokens_per_block
```

For a full server (TP=2), multiply by world_size=2.

### Observed values across tested models

| Model | KV blocks | KV shape | KV dtype | Bytes/block/rank | Bytes/block (server) |
|---|---|---|---|---|---|
| Llama-3.1-70B-FP8, TP=2, gmu=0.75 | 27,421 | (27421, 80, 2, **16**, 4, 128) | BF16¹ | 2,621,440 B | **~5 MB** |
| Llama-3.1-70B-BF16, TP=2, gmu=0.90 | 22,955 | (22955, 80, 2, **16**, 4, 128) | BF16 | 2,621,440 B | **~5 MB** |
| gpt-oss-120b (MoE), TP=2, gmu=0.35 | 18,569 | (18569, 36, 2, **16**, 4, 64) | BF16 | 589,824 B | **~1.2 MB** |
| Qwen3-0.6B, TP=2² | 43,517 | (43517, 28, 2, **4**, 16, 128) | BF16 | 917,504 B | **~1.75 MB** |
| Qwen3-8B, TP=2² | 27,377 | (27377, 36, 2, **4**, 16, 128) | BF16 | 1,179,648 B | **~2.25 MB** |
| Qwen3-32B-AWQ, TP=2² | 14,645 | (14645, 64, 2, **4**, 16, 128) | BF16 | 2,097,152 B | **~4 MB** |

¹ kv_cache_dtype=auto resolves to BF16 even for FP8 weight models.
² KV shape derived from HuggingFace config.json (num_hidden_layers, num_key_value_heads, head_dim) + block count from vLLM startup logs at gmu=0.9 on 2× L40S (48 GiB each). vLLM 0.19 does not log the full tensor shape directly.

**Key observation:** Block size spans nearly 3× across model families —
from ~1.75 MB for Qwen3-0.6B to ~5 MB for Llama-70B. Any FIO configuration must
account for this range.

### GPU KV block tokens (16) confirmed by token count

GPU block count × 16 tokens/block = total KV tokens reported in startup log:

- FP8-70B: 27,421 × 16 = **438,736** tokens ✓  
- BF16-70B: 22,955 × 16 = **367,280** tokens ✓  
- gpt-oss-120b: 18,569 × 16 = **297,104** tokens ✓

---

## Connector Architecture

Source: `vllm/vllm/v1/kv_offload/tiering/fs/` (vLLM 0.22.0+)

One file per GPU KV block (default `block_size_factor=1`). O_DIRECT is used when
available on Linux:

```python
# io.py
O_DIRECT = getattr(os, "O_DIRECT", 0)
fd = os.open(tmp_path,
    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | O_DIRECT, 0o644)
written = os.write(fd, view_slice)  # one syscall per block
```

Each eviction = one `os.write()` call = the full GPU KV block size, bypassing the
page cache. An atomic `rename()` commits the file after write.

**FIO implications:**
- Use `direct=1` (O_DIRECT) — no page cache involvement
- Block size = GPU KV block size for the target model (see table above)
- One I/O per thread per operation
- Concurrency = `threads_per_gpu` (default: 16 per GPU)

---

## FIO Configuration

Block sizes are the actual GPU KV block sizes per server, per model family:

| Test name | Block size | Target model |
|---|---|---|
| `kv-fp8-70b` | 5m | FP8-70B, BF16-70B (TP=2) |
| `kv-gpt120b` | 1152k | gpt-oss-120b (TP=2) |
| `kv-qwen8b` | 2304k | Qwen3-8B (TP=2) |
| `kv-qwen32b` | 4m | Qwen3-32B (TP=2) |

All block sizes are server-wide (both TP=2 ranks combined).

All tests use `direct=1` (O_DIRECT), `ioengine=libaio`, `iodepth=1`, `numjobs` ∈ {1, 16},
`runtime=60s`. Each model also has a `mixed-j16` section (`rw=randrw`, `rwmixread=75`)
representing steady-state prefix-cache serving: 75% reads (block restorations) and
25% writes (evictions). The primary metric is **per-operation latency** (p50 and p99)
— block load latency is on the critical path to first token.

Run scripts are in `scripts/`; see section headers below for usage.

### Running a subset

By default the config runs all 15 filesystem sections (~15 minutes). To test only your
model's block size, use `--section`:

```bash
# Example: just the Qwen3-8B read sections
fio fio-kv.fio --section=qwen8b-read-j1 --section=qwen8b-read-j16 \
  --directory=/your/storage --output-format=json+ --output=results.json
```

Section names follow the pattern `<model>-<rw>-j<concurrency>`:
`fp8-70b`, `gpt120b`, `qwen8b`, `qwen32b` × `write`/`read`/`mixed` × `j1`/`j16`.

### Reading the output

Write results to a file to avoid mixing with FIO progress text:

```bash
fio fio-kv.fio --directory=/path/to/storage --output-format=json+ --output=results-fs.json
```

Extract p99 read latency (in ms) for each `*-read-j16` section:

```bash
jq '[.jobs[] | select(.jobname | test("read-j16"))
     | {job: .jobname,
        p99_ms: (.read.lat_ns.percentile["99.000000"] / 1e6)}]' results.json
```

`lat_ns` values are in **nanoseconds** — divide by 1,000,000 for milliseconds.

**Which section to focus on:**
- `*-read-j16`: primary signal — block restoration latency at full thread concurrency
- `*-write-j16`: eviction throughput
- `*-mixed-j16`: 75/25 restoration/eviction mix representing simultaneous block restoration and eviction

### Latency reference points

The p99 read latency at j=16 is a cold-cache-hit latency. Whether that latency is
acceptable depends entirely on the serving workload — specifically, how it compares
to the cost of recomputing the same prefix without the cached KV blocks.

Recomputation cost scales with context length and model size. A tier that is
"too slow" for short-context serving may be highly effective for long-context
workloads where recomputation is measured in seconds.

| p99 read latency (j=16) | Context: when offload adds value |
|---|---|
| < 5 ms | Any workload; overhead negligible relative to TTFT |
| 5–20 ms | Long-context (≥1K tokens); compare against prefill time |
| 20–100 ms | Beneficial where prefill cost exceeds this latency — typical for 8K+ contexts |
| 100–500 ms | Worthwhile for very long contexts (32K+) where recompute takes seconds |
| > 500 ms | Only beneficial for very large models / extremely long contexts |

These are reference points, not pass/fail criteria.

---

## S3 / RGW Backend

The `fio-kv.fio` file also characterises the **object store secondary tier**
(`vllm/v1/kv_offload/tiering/obj/`, NIXL OBJ backend). This tier stores KV cache blocks
as S3 objects in Ceph RGW, bypassing the CephFS metadata server entirely. Each GPU KV
block maps to one S3 object; block hashes become object keys.

### I/O model differences

| | Filesystem | S3 / RGW |
|---|---|---|
| Engine | `libaio`, O_DIRECT | `http`, S3 PUT/GET |
| Eviction path | POSIX `write()` → file | S3 PUT → RADOS object |
| Restoration path | POSIX `read()` → memory | S3 GET → memory |
| Metadata overhead | One MDS op per file open/close | None — key lookup goes directly to RADOS |
| Eviction policy | Space-based LRU via pvc_evictor | S3 lifecycle rules (TTL-based) |

### Running the S3 benchmark

The S3 sections in `fio-kv.fio` are commented out. `scripts/run-fio-s3.sh` fetches
credentials from the Rook secret (`rook-ceph-object-user-kvcache-store-kvcache-user`
in `openshift-storage`), generates a runnable config with credentials substituted, and
prints the fio command to run from inside your FIO pod.

```bash
# On the host (outside the cluster pod):
scripts/run-fio-s3.sh

# Copy the generated config to your FIO pod, then:
fio /tmp/fio-kv-s3-XXXXXX.fio --output-format=json+ --output=results-s3.json

# Extract p99 read latencies:
jq '[.jobs[] | select(.jobname | test("s3.*read-j16")) | {job: .jobname, p99_ms: (.read.lat_ns.percentile["99.000000"] / 1e6)}]' results-s3.json
```

### Prerequisites

- Ceph RGW deployed via ODF and `kvcache-store` CephObjectStore created
- `kvcache` bucket created (see deployment manifests)
- fio built with HTTP engine (`fio --enghelp=http` should list options)

### Interpreting results

Use the [latency reference points](#latency-reference-points) table. The comparison
between filesystem and S3 results quantifies the protocol cost at each block size
and concurrency level.

---

## Results: Single PCIe Gen 4 NVMe / XFS

Hardware: single 7.68 TB Micron 7450 NVMe (PCIe Gen 4, 4Kn native sector size),
provisioned via LVM thin pool (linear, not striped), XFS (bsize=4096, sectsz=4096,
sunit=1 MiB, swidth=32 MiB). O_DIRECT, `size=225m` per file, 60 s runtime.
The LVM thin pool is linear; a single PVC lands on one drive. Results represent
single-drive performance.

| Test | RW | BW (MB/s) | IOPS | lat p50 | lat p99 |
|---|---|---|---|---|---|
| Qwen3-8B bs=2304k j=1 | write | 4,946 | 2,203 | 0.4 ms | 0.6 ms |
| Qwen3-8B bs=2304k j=1 | read | 3,000 | 1,334 | 0.7 ms | 0.9 ms |
| Qwen3-8B bs=2304k j=16 | write | 4,830 | 2,147 | 6.6 ms | 11.6 ms |
| Qwen3-8B bs=2304k j=16 | read | 6,227 | 2,768 | 1.4 ms | 2.3 ms |
| Qwen3-8B mixed j=16 | read | 4,616 | 2,051 | 7.2 ms | 12.3 ms |
| gpt-oss-120b bs=1152k j=1 | write | 4,794 | 4,261 | 0.2 ms | 0.4 ms |
| gpt-oss-120b bs=1152k j=1 | read | 1,989 | 1,768 | 0.6 ms | 0.6 ms |
| gpt-oss-120b bs=1152k j=16 | write | 5,278 | 4,691 | 3.3 ms | 4.8 ms |
| gpt-oss-120b bs=1152k j=16 | read | 6,603 | 5,870 | 2.7 ms | 4.0 ms |
| gpt-oss-120b mixed j=16 | read | 4,674 | 4,155 | 3.3 ms | 8.0 ms |
| FP8-70B bs=5m j=1 | write | 5,322 | 1,015 | 0.9 ms | 1.4 ms |
| FP8-70B bs=5m j=1 | read | 4,576 | 873 | 1.1 ms | 1.3 ms |
| FP8-70B bs=5m j=16 | write | 5,274 | 1,006 | 14.5 ms | 21.9 ms |
| FP8-70B bs=5m j=16 | read | 6,919 | 1,320 | 12.1 ms | 14.1 ms |
| FP8-70B mixed j=16 | write | 2,863 | 546 | 2.9 ms | 8.0 ms |
| FP8-70B mixed j=16 | read | 2,809 | 536 | 26.1 ms | 41.7 ms |

**Single-thread latency is sub-millisecond for all block sizes** — a single GPU KV
block load completes in 0.1–1.1 ms. The drive sustains 4.8–5.3 GB/s per file write.

**At j=16 read p99:** Qwen3-8B 2.3 ms, gpt-oss-120b 4.0 ms, FP8-70B 14.1 ms.
Smaller block sizes benefit from higher IOPS at a given bandwidth.

**Mixed RW at j=16 (75% read / 25% write) p99 read:** Qwen3-8B 12.3 ms,
gpt-oss-120b 8.0 ms, FP8-70B 41.7 ms. Simultaneous eviction and restoration at
full concurrency elevates read latency ~3–5× vs pure read for the two larger models.


---

## Results: Ceph Cold Tier — IBM VPC Block Storage

### Test environment

- **Cluster:** OpenShift 4.20 on IBM VPC
- **Storage:** 2× IBM VPC block 10iops-tier, 4Ti each, as Ceph OSDs (size:1 pools, OSD 0 on fio-pod node, OSD 1 on second GPU worker)
- **fio-pod:** GPU worker node (H200), co-located with OSD 0 on same IBM VPC storage fabric
- **PCP DaemonSet:** one pod per GPU worker, collecting kernel PMDAs and Ceph openmetrics (14 focused metrics, pmlogextract window covering FIO run duration)
- **fio version:** 3.40
- **Backends:** IBM VPC block (raw, no filesystem), CephFS (kvcache-fs, size:1), Ceph RGW/S3 (kvcache-store, size:1)
- **Cold-cache methodology:** `ceph tell osd.0 cache drop && ceph tell osd.1 cache drop` between write and read phases; reads measured after BlueStore cache cleared

All Ceph pools use `size:1` — single replica, no redundancy. Data loss equals a cache miss.

### VPC block device probe (j=1, bs=5m)

A 30-second j=1 probe characterises the raw device with and without concurrent
Ceph OSD I/O on the same IBM VPC storage fabric.

| Condition | write p50 | write p99 | write BW | read p50 | read p99 | read BW |
|---|---|---|---|---|---|---|
| With OSD I/O | 27.7 ms | 89.7 ms | 168 MB/s | 6.7 ms | 7.9 ms | 775 MB/s |
| Without OSD I/O | 10.4 ms | 24.3 ms | 440 MB/s | 6.7 ms | 7.8 ms | 775 MB/s |

Read latency and bandwidth are stable regardless of OSD I/O load. Write bandwidth
is 2.6× lower and write p99 3.7× higher when OSD I/O is active on the shared fabric.

### j=16 results — cold-cache reads

No mixed section for RGW — randrw has no defined semantics for S3 objects.
NVMe reference is from a single Micron 7450 PCIe Gen 4 NVMe on a separate cluster
(see [Results: Single PCIe Gen 4 NVMe / XFS](#results-single-pcie-gen-4-nvme--xfs)).

| Model | Backend | write p99 | write BW | read p50 | read p99 | read BW | mixed rd p99 |
|---|---|---|---|---|---|---|---|
| **Qwen3-8B** | NVMe (ref) | 11.6 ms | 4,830 MB/s | 1.4 ms | 2.3 ms | 6,227 MB/s | 12.3 ms |
| bs=2304k | VPC block | 244 ms | 244 MB/s | 48 ms | 49 ms | 762 MB/s | 112 ms |
| | CephFS | 271 ms | 209 MB/s | 173 ms | 259 ms | 205 MB/s | 225 ms |
| | RGW | 438 ms | 159 MB/s | 98 ms | 451 ms | 252 MB/s | — |
| **gpt-oss-120b** | NVMe (ref) | 4.8 ms | 5,278 MB/s | 2.7 ms | 4.0 ms | 6,603 MB/s | 8.0 ms |
| bs=1152k | VPC block | 135 ms | 407 MB/s | 24 ms | 25 ms | 762 MB/s | 62 ms |
| | CephFS | 163 ms | 191 MB/s | 88 ms | 108 ms | 203 MB/s | 109 ms |
| | RGW | 196 ms | 172 MB/s | 57 ms | 144 ms | 260 MB/s | — |
| **FP8-70B** | NVMe (ref) | 21.9 ms | 5,274 MB/s | 12.1 ms | 14.1 ms | 6,919 MB/s | 41.7 ms |
| bs=5m | VPC block | 484 ms | 232 MB/s | 106 ms | 109 ms | 762 MB/s | 209 ms |
| | CephFS | 633 ms | 228 MB/s | 392 ms | 451 ms | 203 MB/s | 489 ms |
| | RGW | 1,250 ms | 190 MB/s | 209 ms | 1,133 ms | 269 MB/s | — |

![Read p99 latency comparison](analysis/fig-read-p99-comparison.png)

![Read bandwidth and CPU overhead](analysis/fig-bandwidth-cpu.png)

### Observations from FIO results

**VPC block reads are bandwidth-limited at 762 MB/s regardless of block size.**
At j=16, IBM VPC block 10iops-tier reaches a consistent read bandwidth ceiling.
Read p99 scales proportionally with block size: gpt-120b (1152k) at 25 ms,
Qwen3-8B (2304k) at 49 ms, FP8-70B (5m) at 109 ms — matching the block size
ratio 1:2:4.3 to within measurement precision.

**CephFS read bandwidth is ~203–205 MB/s across all block sizes — 3.7× below VPC block.**
The ceiling is consistent across all three block sizes, indicating the constraint
is in the CephFS protocol stack rather than the underlying VPC block device.

**RGW read p50 is 98–209 ms; p99 diverges to 144–1,133 ms under 16-job concurrent load.**
The p99 tail grows disproportionately with block size: gpt-120b p99 is 5.8× its
p50 (144/25 ms); FP8-70B p99 is 5.4× its p50 (1133/209 ms). RGW write p99
ranges from 196 ms (gpt-120b) to 1,250 ms (FP8-70B).

**Mixed (75% read / 25% write) read p99 exceeds pure read p99 for all backends.**
VPC block mixed rd p99 rises to 112 ms (Qwen3-8B) and 209 ms (FP8-70B), versus
49 ms and 109 ms pure. CephFS mixed rd p99 reaches 225–489 ms.

### Observations from PCP system metrics

**OSD 0 showed zero disk reads during CephFS and RGW read phases.**
PCP `disk.dev.read_bytes[vde]` (OSD 0's VPC block device, fio-pod node) remained
at 0–45 KB/s throughout both backends. All read-phase disk I/O was served by
OSD 1 (h200-k6qsd), which showed peak reads of 215 MB/s (CephFS) and 284 MB/s (RGW).
This distribution is consistent with CRUSH placement directing the FIO working set
to OSD 1. VPC block shows zero activity on vde throughout its run, confirming no
Ceph cluster involvement.

**VPC block vdf: read bandwidth peaked at 750 MB/s in pure read phases, 238 MB/s mean write in write phases.**
The full archive shows clearly alternating write and read cycles with no idle periods.

**RGW OSD write activity follows a per-model burst pattern.**
PCP shows vde on h200-k6qsd writing in three distinct bursts (one per model),
reaching 199–240 MB/s during each burst. OSD reads during the RGW read phases
peak at 245–284 MB/s per burst.

**Ceph pool throughput:** `openmetrics.ceph_mgr.ceph_pool_rd_bytes` reports
99 MB/s mean (CephFS, pool_id:4) and 98 MB/s mean (RGW, pool_id:14) during
respective read phases. VPC block shows zero activity on all Ceph pools.

**Kernel sys CPU by backend (mean, ms/s):**

| Backend | fio-pod node (h200-66lrw) | OSD-serving node (h200-k6qsd) |
|---|---|---|
| VPC block | 301 | 303 |
| CephFS | 360 | 460 |
| RGW | 499 | 633 |

VPC block is the lowest on both nodes. The OSD-serving node (k6qsd) consistently
shows higher sys CPU than the fio-pod node for the Ceph backends, tracking
OSD disk read activity on that node during read phases.

### Cold vs warm cache (CephFS)

Earlier exploratory runs measured CephFS reads without the cache drop. Comparing
p99 read latency at j=16:

| Model | Warm (no cache drop) | Cold (cache dropped) | Δ |
|---|---|---|---|
| Qwen3-8B | 190 ms | 259 ms | +37% |
| gpt-120b | 91 ms | 108 ms | +19% |
| FP8-70B | 354 ms | 451 ms | +27% |

The 19–37% increase after cache drop is smaller than expected for a full transition
from in-memory to disk-backed serving. The 10-second stabilisation period between
cache drop and the read phase may not fully drain all caching layers on the OSD node.

---

## Appendix: XFS Configuration (NVMe reference — LVM thin pool PVC)

Obtained via `xfs_growfs -n /data` on a PVC provisioned from an LVM thin pool:

```
meta-data  isize=512    agcount=17   agsize=1638144 blks
           sectsz=4096  attr=2       projid32bit=1
           crc=1        finobt=1     sparse=1       rmapbt=0
           reflink=1    bigtime=1    inobtcount=1   nrext64=0
data       bsize=4096   blocks=26214400             imaxpct=25
           sunit=256    swidth=8192  blks
naming     version 2    bsize=4096   ascii-ci=0     ftype=1
log        internal     bsize=4096   blocks=16384   version=2
           sectsz=4096  sunit=1 blks lazy-count=1
realtime   none         extsz=4096   blocks=0       rtextents=0
```

Mount options: `rw,seclabel,relatime,nouuid,attr2,inode64,logbufs=8,logbsize=32k,sunit=2048,swidth=65536,noquota`

| Parameter | Value | Meaning |
|---|---|---|
| `bsize` | 4096 B | XFS filesystem block size |
| `sectsz` | 4096 B | **4Kn NVMe** — O_DIRECT writes must be ≥ 4096 B aligned |
| `sunit` | 256 blocks = **1 MiB** | LVM thin pool chunk size; XFS aligns allocations to this |
| `swidth` | 8192 blocks = **32 MiB** | Optimal sequential I/O width for this configuration |
| `agcount` | 17 | 17 allocation groups; parallel allocation without contention |

**O_DIRECT alignment:** All block sizes must be multiples of 4096 B (4Kn sector
size). The 5 MiB Llama-70B block is perfectly aligned to sunit (1 MiB); the
~1.2 MB gpt-oss-120b block is sector-aligned but not sunit-aligned.
