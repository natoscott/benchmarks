# Storage Characterisation for vLLM KV Cache Filesystem Offload

This report documents the derivation of FIO benchmark configurations that accurately
represent the I/O access patterns of vLLM KV cache filesystem offloading connectors.
The analysis draws on three sources: vLLM startup logs (GPU KV cache block structure),
PCP archives (observed disk-level I/O from production benchmark runs), and connector
source code (actual I/O call patterns).

## Background

vLLM supports offloading GPU KV cache blocks to secondary storage (CPU RAM or
filesystem) when the GPU KV pool is under pressure. The filesystem connector variant
(`llmd_fs_connector` / `vllm.v1.kv_offload.tiering.fs`) writes evicted GPU KV cache
blocks to a mounted PVC and reads them back on cache hit.

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
| Qwen3-0.6B, TP=2 (estimated) | ~39,000 | — | BF16 | ~57,344 B | **~112 KB** |
| Qwen3-8B, TP=2 (estimated) | ~16,000 | — | BF16 | ~131,072 B | **~256 KB** |
| Qwen3-32B-AWQ, TP=2 | ~14,700 | — | BF16 | ~229,376 B | **~448 KB** |

¹ kv_cache_dtype=auto resolves to BF16 even for FP8 weight models.

**Key observation:** Block size spans two orders of magnitude across model families —
from ~112 KB for Qwen3-0.6B to ~5 MB for Llama-70B. Any FIO configuration must
account for this range.

### GPU KV block tokens (16) confirmed by token count

GPU block count × 16 tokens/block = total KV tokens reported in startup log:

- FP8-70B: 27,421 × 16 = **438,736** tokens ✓  
- BF16-70B: 22,955 × 16 = **367,280** tokens ✓  
- gpt-oss-120b: 18,569 × 16 = **297,104** tokens ✓

---

## Connector Architecture Analysis

There are two distinct connector implementations with fundamentally different I/O
architectures. The correct FIO configuration depends on which connector is in use.

### llmd_fs_backend (llm-d ≤ v0.7.0)

Source: `~/git/llm-d-kv-cache/kv_connectors/llmd_fs_backend/`

The connector groups multiple GPU KV blocks into a single file, controlled by
`block_size` (in **tokens**, not bytes):

```python
# spec.py
self.gpu_blocks_per_file = self.offloaded_block_size // hash_block_size
# With block_size=256 tokens, gpu_block_size=16 tokens:
# gpu_blocks_per_file = 256 // 16 = 16
```

File I/O uses **buffered writes** through a 1 MB C++ write buffer and the kernel
page cache — no O_DIRECT:

```cpp
// file_io.cpp
const size_t WRITE_BUFFER_SIZE = 1 * 1024 * 1024;  // 1 MB buffer
ofs.rdbuf()->pubsetbuf(thread_write_buffer.data(), WRITE_BUFFER_SIZE);
ofs.write(reinterpret_cast<const char*>(buf.ptr), buf.size);  // one call per file
```

One `write()` call per file = 16 GPU KV blocks of data. An atomic `rename()` commits
the file after write.

### vLLM v1 filesystem connector (upstream, RHOAI future)

Source: `~/git/vllm/vllm/v1/kv_offload/tiering/fs/`

One file per GPU KV block (default `block_size_factor=1`). O_DIRECT is used when
available on Linux:

```python
# io.py
O_DIRECT = getattr(os, "O_DIRECT", 0)
fd = os.open(tmp_path,
    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | O_DIRECT, 0o644)
written = os.write(fd, view_slice)  # one syscall per block
```

Each eviction = one `os.write()` call = the full GPU KV block size, bypassing page
cache.

### Side-by-side comparison

| Property | llmd_fs_backend | vLLM v1 fs connector |
|---|---|---|
| Files per GPU KV block | 1/16 (16 blocks share one file) | 1 file per block |
| I/O type | **Buffered** (1 MB C++ buffer + page cache) | **O_DIRECT** (no page cache) |
| Write call size | 16 × block_size_bytes | 1 × block_size_bytes |
| Write for FP8-70B | ~40 MB per file write | ~2.5 MB per syscall |
| Write for Qwen3-0.6B | ~896 KB per file write | ~56 KB per syscall |
| Write for gpt-oss-120b | ~9 MB per file write | ~589 KB per syscall |
| Atomic commit | `rename()` | `rename()` |
| Read (load) | One `read()` per file | One `os.read()` per block |

---

## Why PCP Showed ~1 KB Effective I/O Size

During filesystem offload benchmarks on the llmd_fs_backend, PCP disk metrics
(`disk.dev.write_bytes` / `disk.dev.write`) implied an average I/O size of ~1 KB:

| Run | Write B/s | Write ops/s | Implied I/O size |
|---|---|---|---|
| Qwen3-0.6B, rate=1 | 18.7 KB/s | 17 | 1.07 KB |
| Qwen3-0.6B, rate=100 | 94.8 KB/s | 90 | 1.05 KB |
| Qwen3-32B, rate=300 | 292.3 KB/s | 252 | 1.16 KB |

This apparently contradicts the ~896 KB per file write expected from the connector.

**The observation is consistent with buffered write fragmentation, but the precise
layer responsible cannot be determined from PCP metrics alone.** The write path for
llmd_fs_backend is: C++ `std::ofstream` with a 1 MB `pubsetbuf` buffer → `write()`
syscall to the kernel → kernel page cache → disk I/O. Note that glibc's own stdio
buffering (via `FILE*`) is not in this path; C++ stream buffering is separate.

The `disk.dev.write` metric in PCP counts disk I/O *requests issued at the block
layer*, not application `write()` syscalls. A single large `write()` of ~896 KB into
the page cache may eventually reach disk as many small requests, with the request
size determined by some combination of: the kernel's writeback page unit (typically
4 KB), the I/O scheduler's merge policy, and the underlying block device's preferred
I/O size. The observed ~1 KB is consistent with these layers fragmenting the buffered
write — but confirming exactly which layer is responsible would require block-level
tracing (e.g., `blktrace`, `bpftrace`, or `iostat -x` with extended statistics) during
an active fs-offload run.

What is certain: the ~1 KB PCP-implied size is **not** the application's write
granularity, and does **not** represent the I/O unit the connector operates in at the
application level. For FIO validation purposes, the connector's application-level I/O
granularity (file size) is the correct benchmark parameter — the block-layer
fragmentation will occur on whichever storage backend is tested, making the FIO
results directly comparable to production connector behaviour.

**Implication:** The existing sequential 1 MB FIO benchmark (`direct=1`) tests raw
block device bandwidth and is appropriate as a storage ceiling check. But it does not
reproduce the actual connector access pattern on either implementation:
- `llmd_fs_backend`: buffered writes of 896 KB–40 MB per file; FIO should use
  buffered I/O (`direct=0`) at those file sizes.
- `vLLM v1 fs`: O_DIRECT writes at the GPU KV block size (589 KB–5 MB); FIO should
  use `direct=1` at those sizes.

---

## FIO Configuration Rationale

### Common parameters (both connectors)

| Parameter | Value | Rationale |
|---|---|---|
| `rw` | `write` then `read` | Eviction (write) then cache restoration (read) are independent workloads |
| `ioengine` | `libaio` (O_DIRECT) or `sync` (buffered) | Match connector I/O model |
| `numjobs` | 1, 16, 64 | 1 = quiescent; 16 = `threads_per_gpu` default; 64 = heavy concurrency |
| `iodepth` | 1 (sync) or 16 (aio) | The connector submits one block per thread |
| `runtime` | 60 s | Standard duration for steady-state measurement |
| `group_reporting` | yes | Combined stats across jobs |

The primary metric is **per-operation latency** (mean and p99), not throughput.
Latency gates request completion time: a slow block load delays the first token of
any request that triggered a cache miss.

### Configuration A: vLLM v1 fs connector (O_DIRECT)

Block sizes are the actual GPU KV block sizes per server, per model family:

| Test name | `bs` | Target model | Rationale |
|---|---|---|---|
| `kv-fp8-70b` | 5m | FP8-70B, BF16-70B, TP=2 | 80 layers × 8 heads × 128 dim × 2 × 2 B × 16 tok × 2 ranks |
| `kv-gpt120b` | 1200k | gpt-oss-120b, TP=2 | 36 layers × 8 heads × 64 dim × 2 × 2 B × 16 tok × 2 ranks |
| `kv-qwen32b` | 450k | Qwen3-32B, TP=2 | estimated from architecture |
| `kv-qwen8b` | 256k | Qwen3-8B, TP=2 | estimated |

Use `direct=1` (O_DIRECT). The storage must sustain the target latency at each block
size and concurrency level.

### Configuration B: llmd_fs_backend (buffered, file-level)

File sizes = `gpu_blocks_per_file × block_size_bytes`. With `block_size=256 tokens`
and `gpu_block_size=16 tokens`, `gpu_blocks_per_file=16`:

| Test name | `bs` | Target model | File size = 16 × block |
|---|---|---|---|
| `file-fp8-70b` | 80m | FP8-70B | 16 × 5 MB = 80 MB |
| `file-gpt120b` | 18m | gpt-oss-120b | 16 × 1.2 MB = ~19 MB |
| `file-qwen32b` | 7m | Qwen3-32B | 16 × 448 KB ≈ 7 MB |
| `file-qwen8b` | 4m | Qwen3-8B | 16 × 256 KB ≈ 4 MB |

Use `direct=0` (buffered) to match the connector. The OS will fragment these into
the storage backend's preferred I/O block size; the benchmark reveals sustainable
throughput at the connector's actual write granularity.

---

## What Good and Bad Results Look Like

### Latency targets

The connector loads blocks on critical path (before the request can be prefilled).
For a 120-second benchmark window, a single block load adding >500 ms is material.

**Pass criteria (vLLM v1 fs, O_DIRECT):**
- p99 read latency < 100 ms at numjobs=16 for all block sizes
- p50 read latency < 20 ms

**Fail indicator:**
- p99 > 500 ms: storage is a bottleneck; CPU offload preferred over fs offload
- Throughput collapse at numjobs=64: storage IOPS-limited

### Throughput targets (llmd_fs_backend, buffered)

At steady state, the connector must sustain writes proportional to the eviction rate.
From upstream benchmarks at peak concurrency (~300 req/s), write demand is ~292 KB/s
for Qwen3-8B. Scaled to FP8-70B at the same eviction rate, demand is ~4–10 MB/s.

**Pass criteria:** Sustained write throughput ≥ 20 MB/s at numjobs=16 (2× safety margin).

---

## Appendix A: Test Hardware and XFS Configuration

### Storage hardware (athena-fire cluster, worker-2-gpu-h200-k6qsd)

| Property | Value |
|---|---|
| NVMe devices | 8× Micron 7450 MTFDKCC7T6TFR (PCIe Gen 4) |
| Capacity per drive | 7.68 TB (6.98 TB usable) |
| LVM arrangement | Linear thin pool (`vg-gpu-node-n2`); NOT striped |
| PVC allocation | One PVC lands on one NVMe drive (linear mapping) |
| I/O scheduler | `none` (NVMe pass-through — optimal for NVMe) |
| Kernel queue depth | 1023 requests per queue |
| Read-ahead | 4096 KB |

A single 100 Gi PVC used for FIO testing is backed entirely by one Micron 7450
drive. Results represent single-drive performance, not an aggregate stripe.

### XFS filesystem configuration (100 Gi PVC, `lvms-vg-gpu-node-n2`)

Obtained via `xfs_growfs -n /data` on a provisioned PVC:

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

**Key parameters and their significance for FIO:**

| Parameter | Value | Meaning |
|---|---|---|
| `bsize` | 4096 B | XFS filesystem block size |
| `sectsz` | 4096 B | **4Kn NVMe** — O_DIRECT writes must be ≥ 4096 B aligned |
| `sunit` | 256 blocks = **1 MiB** | LVMS thin pool chunk size; XFS aligns allocations to this |
| `swidth` | 8192 blocks = **32 MiB** | Optimal sequential I/O width for this configuration |
| `agcount` | 17 | 17 allocation groups; parallel allocation without contention |
| `isize` | 512 B | Inode size |
| `crc` | 1 | Metadata CRC checksums enabled (small overhead on writes) |
| `reflink` | 1 | Copy-on-write reflink supported (not used by connector) |
| `log` | internal, 64 MiB | Journal inside the data partition |

**FIO alignment implications:**
- All O_DIRECT block sizes must be multiples of 4096 B (4Kn sector size)
- Optimal block sizes for XFS are multiples of 1 MiB (`sunit`); e.g., 1 MiB, 2 MiB, 5 MiB, 8 MiB
- The Llama-70B block size of ~5 MiB = 5 × 1 MiB is perfectly sunit-aligned
- The gpt-oss-120b block size of ~576 KB = 144 × 4 KiB is sector-aligned but not sunit-aligned; XFS may split the write internally

### Reference: SNO cluster (L40S, llm-d v0.7.0 fs-offload benchmarks)

The upstream llm-d v0.7.0 benchmarks ran on a single-node OpenShift (SNO)
cluster with 2× L40S GPUs. The KV cache storage PVC during fs-offload runs
was on a local disk (`vda`), characterised from PCP archives:

| Metric | Qwen3-0.6B rate=1 | Qwen3-0.6B rate=100 | Qwen3-32B rate=100 |
|---|---|---|---|
| avg disk request size | 7.6 KB | 9.2 KB | 5.2 KB |
| I/O await (ms) | 1.6 | 1.4 | 1.2 |
| Disk utilisation | 1.3% | 1.2% | 1.1% |
| Write bandwidth | 0.9 KB/s | 0.8 KB/s | 0.4 KB/s |

The ~5–10 KB disk request size (measured via `disk.dev.avg_rqsz`) is the kernel
page cache writeback granularity — consistent with 4 KB dirty page flushes
sometimes merged to 8 KB by the I/O scheduler. The disk was essentially idle
during these runs (<2% utilisation), confirming that local NVMe is not the
bottleneck for the llmd_fs_backend at tested concurrency levels.

---

## FIO Configuration File

The standalone FIO configuration for the vLLM v1 fs connector characterisation
is available for direct download and use without Kubernetes:

**[fio/fio-v1-fs.fio](fio/fio-v1-fs.fio)**

```bash
# Run against any mounted filesystem:
fio fio/fio-v1-fs.fio --directory=/path/to/your/storage
```

Requires 32 GiB of free space (16 jobs × 2 GiB) and fio ≥ 3.x. Output is
in JSON+ format; pipe through `jq` or a script to extract the `lat_ns.percentile`
fields for p50/p99 latency per test.

---

## Results: vLLM v1 fs Connector — Single Micron 7450 NVMe / XFS

Hardware: single 7.68 TB Micron 7450 MTFDKCC7T6TFR (PCIe Gen 4, 4Kn),
provisioned via LVMS linear thin pool, XFS (bsize=4096, sectsz=4096,
sunit=1 MiB, swidth=32 MiB). O_DIRECT, 1 file per job, 2 GiB per file,
60 s runtime, `--output-format=json+`.

| Test | RW | BW (MB/s) | IOPS | lat p50 | lat p99 |
|---|---|---|---|---|---|
| Qwen3-8B bs=256k j=1 | write | 3,660 | 13,961 | 0.1 ms | 0.1 ms |
| Qwen3-8B bs=256k j=1 | read | 623 | 2,375 | 0.4 ms | 0.5 ms |
| Qwen3-8B bs=256k j=16 | write | 4,994 | 19,052 | 0.7 ms | 3.4 ms |
| Qwen3-8B bs=256k j=16 | read | 5,717 | 21,808 | 0.7 ms | 1.4 ms |
| gpt-oss-120b bs=576k j=1 | write | 4,639 | 7,866 | 0.1 ms | 0.1 ms |
| gpt-oss-120b bs=576k j=1 | read | 1,211 | 2,053 | 0.5 ms | 0.5 ms |
| gpt-oss-120b bs=576k j=16 | write | 5,286 | 8,961 | 1.6 ms | 4.1 ms |
| gpt-oss-120b bs=576k j=16 | read | 6,681 | 11,328 | 1.4 ms | 2.8 ms |
| FP8-70B bs=5m j=1 | write | 5,322 | 1,015 | 0.9 ms | 1.4 ms |
| FP8-70B bs=5m j=1 | read | 4,576 | 873 | 1.1 ms | 1.3 ms |
| FP8-70B bs=5m j=16 | write | 5,274 | 1,006 | 14.5 ms | 21.9 ms |
| FP8-70B bs=5m j=16 | read | 6,919 | 1,320 | 12.1 ms | 14.1 ms |
| FP8-70B mixed j=16 | write | 2,863 | 546 | 2.9 ms | 8.0 ms |
| FP8-70B mixed j=16 | read | 2,809 | 536 | 26.1 ms | 41.7 ms |

**Single-thread latency is sub-millisecond for all block sizes** — a single GPU KV
block load completes in 0.1–1.1 ms. The drive sustains 3.6–5.3 GB/s per file
write approaching its rated sequential bandwidth.

**At j=16 (connector default `threads_per_gpu`), FP8-70B blocks show p99 = 14–22 ms.**
Still a large gain over recomputing 16-token blocks from a 10,000-token prompt
(~100 s at Llama-70B decode rates). Smaller models remain below 5 ms p99 at j=16.

**Mixed RW at j=16 (worst case): p99 read = 41.7 ms.** Simultaneous eviction and
restoration under maximum thread pressure on a single NVMe drive. Accept for
long-context workloads; consider CPU offload for latency-critical short-context serving.

### Decision guide

Run `fio/fio-v1-fs.fio` and compare `lat_ns.percentile["99.000000"]` at `numjobs=16`
for the block size matching your model:

| p99 read latency (j=16) | Recommendation |
|---|---|
| < 5 ms | **Excellent** — fs offload recommended for all workloads |
| 5–20 ms | **Good** — beneficial for long-context; validate at target concurrency |
| 20–50 ms | **Acceptable** — worthwhile where recompute cost exceeds 50 ms |
| 50–200 ms | **Marginal** — CPU offload likely better except for very long prefills |
| > 200 ms | **Avoid** — fs offload will increase TTFT; use CPU offload only |

This cluster (single Micron 7450 / XFS) at FP8-70B: **14.1 ms p99 → Good tier**.
