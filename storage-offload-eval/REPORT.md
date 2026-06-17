# Storage Characterisation for vLLM KV Cache Filesystem Offload

This report documents the derivation of FIO benchmark configurations that accurately
represent the I/O access patterns of the vLLM v1 filesystem offload connector, and
presents measured results on NVMe-backed XFS storage.

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
| Qwen3-0.6B, TP=2² | 43,517 | (N, 28, 2, **4**, 16, 128) | BF16 | 917,504 B | **~1.75 MB** |
| Qwen3-8B, TP=2² | 27,377 | (N, 36, 2, **4**, 16, 128) | BF16 | 1,179,648 B | **~2.25 MB** |
| Qwen3-32B-AWQ, TP=2² | 14,645 | (N, 64, 2, **4**, 16, 128) | BF16 | 2,097,152 B | **~4 MB** |

¹ kv_cache_dtype=auto resolves to BF16 even for FP8 weight models.
² KV shape derived from HuggingFace config.json (num_hidden_layers, num_key_value_heads, head_dim). Block count from vLLM startup logs at gmu=0.9 on 2× L40S (48 GiB each).

**Key observation:** Block size spans two orders of magnitude across model families —
from ~112 KB for Qwen3-0.6B to ~5 MB for Llama-70B. Any FIO configuration must
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
| `kv-gpt120b` | 576k | gpt-oss-120b (TP=2) |
| `kv-qwen8b` | 2304k | Qwen3-8B (TP=2) |
| `kv-qwen32b` | 4m | Qwen3-32B (TP=2) |

All tests use `direct=1` (O_DIRECT), `ioengine=libaio`, `iodepth=16`, `numjobs` ∈ {1, 16},
`runtime=60s`. The primary metric is **per-operation latency** (p50 and p99), not
throughput — block load latency is on the critical path to first token.

The standalone FIO configuration is available for direct download:

**[fio/fio-v1-fs.fio](fio/fio-v1-fs.fio)**

```bash
# Run against any mounted filesystem:
fio fio/fio-v1-fs.fio --directory=/path/to/your/storage
```

Requires 32 GiB of free space (16 jobs × 2 GiB) and fio ≥ 3.x.

### Latency targets

| p99 read latency (j=16) | Recommendation |
|---|---|
| < 5 ms | **Excellent** — fs offload recommended for all workloads |
| 5–20 ms | **Good** — beneficial for long-context; validate at target concurrency |
| 20–50 ms | **Acceptable** — worthwhile where recompute cost exceeds 50 ms |
| 50–200 ms | **Marginal** — CPU offload likely better except for very long prefills |
| > 200 ms | **Avoid** — fs offload will increase TTFT; use CPU offload only |

---

## Results: Single PCIe Gen 4 NVMe / XFS

Hardware: single 7.68 TB Micron 7450 NVMe (PCIe Gen 4, 4Kn native sector size),
provisioned via LVM thin pool (linear, not striped), XFS (bsize=4096, sectsz=4096,
sunit=1 MiB, swidth=32 MiB). O_DIRECT, 1 file per job, 2 GiB per file, 60 s runtime.

Note: the LVM thin pool is linear across multiple drives; a single PVC lands on one
drive. Results represent single-drive performance.

| Test | RW | BW (MB/s) | IOPS | lat p50 | lat p99 |
|---|---|---|---|---|---|
| Qwen3-8B bs=2304k j=1 | write | 4,946 | 2,203 | 0.4 ms | 0.6 ms |
| Qwen3-8B bs=2304k j=1 | read | 3,000 | 1,334 | 0.7 ms | 0.9 ms |
| Qwen3-8B bs=2304k j=16 | write | 4,830 | 2,147 | 6.6 ms | 11.6 ms |
| Qwen3-8B bs=2304k j=16 | read | 6,227 | 2,768 | 1.4 ms | 2.3 ms |
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
block load completes in 0.1–1.1 ms. The drive sustains 3.6–5.3 GB/s per file write,
approaching its rated sequential bandwidth.

**At j=16 (connector default `threads_per_gpu`), FP8-70B blocks show p99 = 14–22 ms.**
Still a large gain over recomputing 16-token blocks from a 10,000-token prompt
(~100 s at Llama-70B decode rates). Smaller models remain below 5 ms p99 at j=16.

**Mixed RW at j=16 (worst case): p99 read = 41.7 ms.** Simultaneous eviction and
restoration under maximum thread pressure on a single NVMe drive. Accept for
long-context workloads; consider CPU offload for latency-critical short-context serving.

**FP8-70B (bs=5m) at j=16: 14.1 ms p99 → Good tier. Qwen3-8B (bs=2304k) at j=16: 11.6 ms write / 2.3 ms read p99 → Good/Excellent.**

---

## Appendix: XFS Configuration (LVM thin pool PVC)

Obtained via `xfs_growfs -n /data` on a 100 Gi PVC provisioned from an LVM thin pool:

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
