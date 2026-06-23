# Storage Characterisation for vLLM KV Cache Offload

## TL;DR

**[fio/fio-kv.fio](fio/fio-kv.fio)** covers two backends in one file:
- **Filesystem** sections (active by default) — targets the native fs offload connector
- **S3/RGW** sections (commented) — targets the object store secondary tier via Ceph RGW

### Filesystem backend

```bash
fio fio-kv.fio --directory=/path/to/mounted/storage \
    --output-format=json+ --output=results-fs.json
jq '[.jobs[] | select(.jobname | test("read-j16")) | {job: .jobname, p99_ms: (.read.lat_ns.percentile["99.000000"] / 1e6)}]' results-fs.json
```

Requires ~15 GiB free space and fio ≥ 3.x.

### S3 / RGW backend

```bash
# Fetches credentials from the Rook secret and generates a runnable config:
scripts/run-fio-s3.sh
# Then copy the generated temp config to your FIO pod and run it.
```

Requires fio built with HTTP engine support (`fio --enghelp=http`) and a running Ceph RGW
instance with the `kvcache` bucket pre-created. See [S3 / RGW Backend](#s3--rgw-backend) below.

Compare the `p99_ms` at `numjobs=16` against the [latency targets](#latency-targets) table.
Targets the native vLLM filesystem offload connector (vLLM 0.22.0+) and the NIXL OBJ
object store secondary tier.

---

This report documents the derivation of the FIO benchmark configurations from first
principles, and presents measured results on NVMe-backed XFS storage.

The analysis draws on three sources: vLLM startup logs (GPU KV cache block structure),
PCP archives (observed disk-level I/O from production benchmark runs), and connector
source code (actual I/O call patterns).

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

| Test name | `bs` | Target model |
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

See [TL;DR](#tldr) above for the FIO configuration download and usage.

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
- `*-write-j16`: eviction throughput; rarely the bottleneck in practice
- `*-mixed-j16`: 75/25 read/write mix; use for workloads where eviction and restoration overlap

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

These are reference points, not pass/fail criteria. Measure recomputation cost for
your specific model and context distribution before drawing conclusions.

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

Use the same [latency targets](#latency-targets) table. RGW adds network round-trip
overhead vs local NVMe; the comparison between filesystem and S3 results quantifies
that cost at each block size and concurrency level.

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
- **Storage:** 2× IBM VPC block 10iops-tier, 4Ti each, as Ceph OSDs (size:1 pools)
- **fio-pod:** GPU worker node (H200), co-located with OSD 0 on same IBM VPC storage fabric
- **PCP:** co-located with fio-pod, collecting kernel PMDAs + Ceph openmetrics throughout each run
- **fio version:** 3.40
- **Backends:** IBM VPC block (raw, no filesystem), CephFS (kvcache-fs, size:1), Ceph RGW/S3 (kvcache-store, size:1)

All Ceph pools use `size:1` — single replica, no redundancy. Data loss equals a
cache miss, which is acceptable for a cold KV cache tier.

### VPC block device probe (j=1, bs=5m)

A 30-second j=1 probe at the FP8-70B block size characterises the raw device
before the full suite. Results shown with and without concurrent Ceph OSD I/O
on the same IBM VPC fabric.

| Condition | write p50 | write p99 | write BW | read p50 | read p99 | read BW |
|---|---|---|---|---|---|---|
| With OSD I/O | 14.9 ms | 88.6 ms | 197 MB/s | 6.6 ms | 7.9 ms | 775 MB/s |
| Without OSD I/O | 10.4 ms | 24.3 ms | 440 MB/s | 6.7 ms | 7.8 ms | 775 MB/s |

Read bandwidth (775 MB/s) is consistent across both runs. Write bandwidth varies
with concurrent OSD I/O load on the shared fabric.

### j=16 results — KV cache restoration (primary metric: read p99)

The NVMe reference row is from a single Micron 7450 PCIe Gen 4 NVMe on a
separate cluster (see [Results: Single PCIe Gen 4 NVMe / XFS](#results-single-pcie-gen-4-nvme--xfs)).
No mixed section for RGW — randrw has no defined semantics for S3 objects.

| Model | Backend | write p99 | write BW | read p50 | read p99 | read BW | mixed rd p99 |
|---|---|---|---|---|---|---|---|
| **Qwen3-8B** | NVMe (ref) | 11.6 ms | 4,830 MB/s | 1.4 ms | 2.3 ms | 6,227 MB/s | 12.3 ms |
| bs=2304k | VPC block | 244 ms | 240 MB/s | 48 ms | 49 ms | 762 MB/s | 105 ms |
| | CephFS | 271 ms | 212 MB/s | 125 ms | 190 ms | 281 MB/s | 194 ms |
| | RGW | 468 ms | 186 MB/s | 49 ms | 784 ms | 328 MB/s | — |
| **gpt-oss-120b** | NVMe (ref) | 4.8 ms | 5,278 MB/s | 2.7 ms | 4.0 ms | 6,603 MB/s | 8.0 ms |
| bs=1152k | VPC block | 140 ms | 312 MB/s | 24 ms | 25 ms | 762 MB/s | 64 ms |
| | CephFS | 124 ms | 228 MB/s | 65 ms | 91 ms | 275 MB/s | 88 ms |
| | RGW | 184 ms | 202 MB/s | 55 ms | 134 ms | 315 MB/s | — |
| **FP8-70B** | NVMe (ref) | 21.9 ms | 5,274 MB/s | 12.1 ms | 14.1 ms | 6,919 MB/s | 41.7 ms |
| bs=5m | VPC block | 468 ms | 269 MB/s | 106 ms | 109 ms | 763 MB/s | 198 ms |
| | CephFS | 522 ms | 227 MB/s | 287 ms | 354 ms | 272 MB/s | 392 ms |
| | RGW | 1,216 ms | 232 MB/s | 99 ms | 1,044 ms | 336 MB/s | — |

### Observations

**VPC block reads saturate at ~762 MB/s regardless of block size.**
At j=16, IBM VPC block 10iops-tier hits a consistent read throughput ceiling.
Read latency scales proportionally with block size at this bandwidth: gpt-120b
(1152k) at 25ms, Qwen3-8B (2304k) at 49ms, FP8-70B (5m) at 109ms — ratios
match the block size ratios 1:2:4.3.

**CephFS read bandwidth is ~64% lower than VPC block at j=16.**
VPC block reads at ~762 MB/s; CephFS reads at ~275 MB/s across all three block
sizes. Read p99 is 2.6–3.9× higher than VPC block (Qwen3-8B: 49→190ms;
FP8-70B: 109→354ms).

**RGW read p50 is close to VPC block, but p99 diverges substantially at j=16.**
S3 read p50 is within 2ms of VPC block for all three models. Under 16-job
concurrent load, p99 is 16× higher for Qwen3-8B (784ms vs 49ms) and 10×
higher for FP8-70B (1,044ms vs 109ms). gpt-120b shows moderate divergence
(134ms vs 25ms, 5.4×).

**Write latency is high and variable across all backends.**
Write p99 ranges from 124ms (CephFS gpt-120b) to 1,216ms (RGW FP8-70B).
Run-to-run variation in write bandwidth is visible in the VPC block probe
(197 vs 440 MB/s with and without OSD I/O), reflecting shared fabric usage.

**Backend ordering by read p99 (lower is better):**

| Model | VPC block | CephFS | RGW |
|---|---|---|---|
| Qwen3-8B | 49 ms | 190 ms | 784 ms |
| gpt-oss-120b | 25 ms | 91 ms | 134 ms |
| FP8-70B | 109 ms | 354 ms | 1,044 ms |

**Relevance to KV cache offload.**
The cold-cache restoration latency (read p99 above) is added to TTFT on a
cache hit. The net benefit of offload depends on the cost of the alternative
— prefix recomputation — which is workload-dependent and not measured here.

---

## Appendix: XFS Configuration (LVM thin pool PVC)

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
