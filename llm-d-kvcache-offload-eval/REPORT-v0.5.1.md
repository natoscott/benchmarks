# llm-d v0.5.1 KV-Cache Offload Evaluation

This report evaluates KV-cache offload strategies in llm-d v0.5.1 (vLLM 0.15.1) and places the results in the context of prior evaluations on v0.4.0 (vLLM 0.11.2) and v0.5.0 (vLLM 0.14.1). Two new offload paths are evaluated for the first time: filesystem offload via the `llmd_fs_connector` external module, and hierarchical CPU+filesystem offload via vLLM's `MultiConnector`.

A supplementary memory-pressure suite re-ran all configurations with reduced `gpu_memory_utilization` per model (0.55–0.70 vs the default 0.9) to create KV-cache pressure across all model sizes. Results for v0.5.1 configurations under memory pressure are included in this report; v0.4.0 memory-pressure results are in REPORT-v0.4.0.md §Memory-Pressure Analysis.

**Software Versions:**
- **llm-d**: v0.5.1 (vLLM 0.15.1, GuideLLM 0.5.4)
- **Prior baselines**: llm-d v0.5.0 (vLLM 0.14.1), llm-d v0.4.0 (vLLM 0.11.2)

**Hardware:** 2× NVIDIA L40S GPUs (48 GB total VRAM), 48 vCPUs, OpenShift on IBM Cloud

**Models:** Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ

**Concurrency levels:** 1, 50, 100, 150, 300, 400, 500, 650

---

## Summary

128 benchmark runs across four KV-cache configurations (gmu=0.9), four model sizes, and eight concurrency levels. A supplementary memory-pressure suite re-ran all configurations at reduced per-model `gpu_memory_utilization` (0.55–0.70). See §Memory-Pressure Analysis.

**Configurations:**
- **no-offload**: GPU-only KV-cache (baseline)
- **native-offload-20k**: CPU offload via `OffloadingConnector` / `CPUOffloadingSpec`, 20K-block equivalent
- **fs-offload**: Filesystem offload via `SharedStorageOffloadingSpec` (`llmd_fs_connector` wheel), IBM VPC block PVC
- **cpu+fs-offload-20k**: CPU+filesystem hierarchical offload via `MultiConnector`

**Peak Throughput (gmu=0.9):**

| Model | no-offload | native-offload-20k | fs-offload | cpu+fs-offload-20k |
|-------|:----------:|:------------------:|:----------:|:-----------------:|
| Qwen3-0.6B | 636.8 tok/s | 622.9 (-2.2%) | 268.8¹ (-57.8%) | 211.2¹ (-66.8%) |
| Qwen3-8B | 114.1 tok/s | 80.0 (-29.9%) | 75.7 (-33.6%) | 75.7 (-33.6%) |
| Qwen3-14B | 58.7 tok/s | 67.2 (+14.5%) | 60.8 (+3.6%) | 62.9 (+7.3%) |
| Qwen3-32B-AWQ | 51.2 tok/s | 21.3 (-58.3%) | 22.4 (-56.2%) | 22.4 (-56.2%) |

¹ *Qwen3-0.6B fs-offload and cpu+fs-offload peaks occur at rate=1 (single-request); sustained throughput at rate=50 is 85.3 tok/s and 25.6 tok/s respectively. These configurations show zero completed requests at rate=300 and rate=500, indicating instability at sustained load.*

---

## Test Configuration

### Hardware

**System:** OpenShift cluster on IBM Cloud (Single Node OpenShift, SNO)
- **GPUs**: 2× NVIDIA L40S (24 GB VRAM each, 48 GB total)
  - Tensor Parallelism: 2 GPUs per model
- **CPU**: 48 vCPUs
- **Storage**: IBM VPC block PVC (256 GiB, `ibmc-vpc-block-custom`, 64K IOPS) mounted at `/kvcache` for filesystem offload

### Software

| Component | Version |
|-----------|---------|
| llm-d | v0.5.1 |
| vLLM | 0.15.1 (bundled in llm-d-cuda:v0.5.1) |
| GuideLLM | latest |
| llmd_fs_connector | 0.15.1 (external wheel) |
| OpenShift | 4.x (SNO) |
| PCP | 7.x |

### Workload

**Profile:** Concurrent multi-turn conversation with shared prefix
- Concurrency levels: 1, 50, 100, 150, 300, 400, 500, 650
- Duration: 120 seconds per concurrency level
- Prompt tokens: 128 per turn
- Output tokens: 128 per turn
- Prefix tokens: 10,000 (shared across requests)
- Turns: 5 per conversation
- Prefix count: rate × 2 unique prefixes per run

### Configurations

#### 1. no-offload (baseline)
```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024
```

#### 2. native-offload-20k
CPU offload via `OffloadingConnector` / `CPUOffloadingSpec`. In vLLM 0.15.1 the allocation is specified in bytes rather than block counts (API change from v0.5.0's `num_cpu_blocks`).

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"cpu_bytes_to_use":<per_model_bytes>}}'
```

Per-model CPU byte allocations (equivalent to 20K KV blocks each):

| Model | cpu_bytes_to_use | Approx. GiB |
|-------|----------------:|------------:|
| Qwen3-0.6B | 72,842,645,340 | 67.8 |
| Qwen3-8B | 57,616,986,275 | 53.7 |
| Qwen3-14B | 44,195,213,475 | 41.2 |
| Qwen3-32B-AWQ | 54,546,084,659 | 50.8 |

#### 3. fs-offload
Filesystem offload via `SharedStorageOffloadingSpec` from the `llmd_fs_connector` external wheel. Requires the wheel installed at runtime due to the llm-d-cuda:v0.5.1 image base (RHEL 9, GCC 11) not shipping a new enough libstdc++ for the pre-built CUDA extension (GLIBCXX_3.4.30+ required).

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024 \
  --distributed-executor-backend mp \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"spec_name":"SharedStorageOffloadingSpec",
      "shared_storage_path":"/kvcache/kv-cache/","block_size":256,
      "threads_per_gpu":64,"spec_module_path":"llmd_fs_backend.spec"}}'
```

Runtime workarounds required:
- Wheel pre-staged on PVC: `pip3.12 install --target /tmp/llmd_packages /data/llmd_fs_connector-0.15.1-cp312-cp312-linux_x86_64.whl`
- `PYTHONPATH=/tmp/llmd_packages`
- `LD_PRELOAD=/opt/nvidia/nsight-compute/2025.2.1/.../libstdc++.so.6` (GLIBCXX_3.4.33)

#### 4. cpu+fs-offload-20k
Hierarchical offload via `MultiConnector`, combining `CPUOffloadingSpec` and `SharedStorageOffloadingSpec`. On each request:
- **Write**: GPU saves to CPU and filesystem simultaneously (parallel)
- **Read**: CPU checked first; filesystem used if CPU misses

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024 \
  --distributed-executor-backend mp \
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"connectors":[
      {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
       "kv_connector_extra_config":{"cpu_bytes_to_use":<per_model_bytes>}},
      {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
       "kv_connector_extra_config":{"spec_name":"SharedStorageOffloadingSpec",
         "shared_storage_path":"/kvcache/kv-cache/","block_size":256,
         "threads_per_gpu":64,"spec_module_path":"llmd_fs_backend.spec"}}
    ]}}'
```

---

## Version Progression

### No-Offload Baseline Across Versions

| Model | v0.4.0 | v0.5.0 | v0.5.1 | v0.5.0→v0.5.1 |
|-------|-------:|-------:|-------:|--------------:|
| Qwen3-0.6B | 602.0 | 634.7 | 636.8 | +0.3% |
| Qwen3-8B | 113.0 | 114.1 | 114.1 | 0.0% |
| Qwen3-14B | 58.7 | 66.1 | 58.7 | **-11.2%** |
| Qwen3-32B-AWQ | 49.2 | 51.2 | 51.2 | 0.0% |

Qwen3-14B no-offload throughput decreased by -11.2% (66.1 → 58.7 tok/s) between vLLM 0.14.1 (v0.5.0) and vLLM 0.15.1 (v0.5.1). All other models are within measurement variance. The v0.5.1 14B no-offload value matches the v0.4.0 value exactly.

### Native CPU Offload Across Versions

The v0.4.0 figure uses 10K blocks (`num_cpu_blocks`); v0.5.0 and v0.5.1 use 20K blocks (`cpu_bytes_to_use`). The API changed in vLLM 0.15.x from block counts to bytes.

| Model | v0.4.0 native (10k) | v0.5.0 native (20k) | v0.5.1 native (20k) | v0.5.0→v0.5.1 |
|-------|--------------------:|--------------------:|--------------------:|--------------:|
| Qwen3-0.6B | 426.8 (-29.1%) | 632.5 (-0.3%) | 622.9 (-2.2%) | -1.5% |
| Qwen3-8B | 71.8 (-36.5%) | 84.3 (-26.1%) | 80.0 (-29.9%) | -5.1% |
| Qwen3-14B | 59.0 (+0.6%) | 65.1 (-1.6%) | 67.2 (+14.5%) | +3.2% |
| Qwen3-32B-AWQ | 48.7 (-1.0%) | 21.3 (-58.4%) | 21.3 (-58.3%) | +0.2% |

The Qwen3-14B native offload improves from -1.6% to +14.5% vs baseline between v0.5.0 and v0.5.1. Given the no-offload baseline also decreased (-11.2%), the absolute throughput with offload is similar (65.1 vs 67.2 tok/s).

![Version Comparison](analysis/v0.5.1_version_comparison.png)
*Figure: Peak throughput across v0.4.0, v0.5.0, and v0.5.1 for no-offload and native-offload-20k configurations. The Qwen3-14B no-offload regression from v0.5.0 to v0.5.1 and Qwen3-0.6B's recovery from the v0.4.0 native offload degradation are the most pronounced inter-version changes.*

---

## v0.5.1 Performance Results

### Peak Throughput

| Model | Config | Peak (tok/s) | Optimal Rate | vs Baseline |
|-------|--------|:------------:|:------------:|:-----------:|
| **Qwen3-0.6B** | no-offload | 636.8 | 50 | — |
| | native-offload-20k | 622.9 | 50 | -2.2% |
| | fs-offload | 268.8¹ | 1 | -57.8% |
| | cpu+fs-offload-20k | 211.2¹ | 1 | -66.8% |
| **Qwen3-8B** | no-offload | 114.1 | 50 | — |
| | native-offload-20k | 80.0 | 50 | -29.9% |
| | fs-offload | 75.7 | 100 | -33.6% |
| | cpu+fs-offload-20k | 75.7 | 50 | -33.6% |
| **Qwen3-14B** | no-offload | 58.7 | 50 | — |
| | native-offload-20k | 67.2 | 50 | +14.5% |
| | fs-offload | 60.8 | 50 | +3.6% |
| | cpu+fs-offload-20k | 62.9 | 50 | +7.3% |
| **Qwen3-32B-AWQ** | no-offload | 51.2 | 1 | — |
| | native-offload-20k | 21.3 | 100 | -58.3% |
| | fs-offload | 22.4 | 50 | -56.2% |
| | cpu+fs-offload-20k | 22.4 | 50 | -56.2% |

¹ *See Qwen3-0.6B section below.*

![Peak Throughput](analysis/v0.5.1_peak_throughput.png)
*Figure: Peak output token throughput by model and configuration. Qwen3-14B is the only model showing throughput above baseline for any offload configuration.*

### Throughput vs Concurrency

![Throughput Curves](analysis/v0.5.1_throughput_curves.png)
*Figure: Output token throughput vs concurrency level for all four models across four configurations. Each panel shows one model. Qwen3-0.6B fs-offload and cpu+fs-offload curves show zero throughput at rate=300 and rate=500 (connector instability). Qwen3-32B-AWQ peaks at rate=1 for no-offload, while offload configurations shift the optimum to rate=50–100.*

#### Qwen3-0.6B

native-offload-20k tracks the baseline closely at all concurrency levels (-2.2% at peak). The fs-offload and cpu+fs-offload-20k configurations show throughput at rate=1 (268.8 and 211.2 tok/s respectively) but drop to 85.3 and 25.6 tok/s at rate=50, and produce zero successful requests at rate=300 and rate=500.

**Failure mechanism at rate≥300:** vLLM does not crash. The API server process remains alive (`/health` returns 200 OK) but the EngineCore deadlocks: filesystem I/O worker threads saturate under write pressure and cannot drain the shared memory broadcast queue used by `--distributed-executor-backend mp`. The EngineCore stalls after 60 seconds waiting for broadcast acknowledgement; the gateway returns HTTP 503 for all subsequent requests. At rate=150, 39 requests completed before the deadlock; at rate=300 and rate=500, zero requests complete (53,000–54,000 errored with `503 Service Unavailable`). The deadlock does not occur for larger models, where slower token generation reduces KV block write pressure.

See [llm-d-kv-cache issue #457](https://github.com/llm-d/llm-d-kv-cache/issues/457) and `BUG-shm-broadcast-deadlock.md` for full reproduction details and root cause analysis.

#### Qwen3-8B

native-offload-20k, fs-offload, and cpu+fs-offload-20k all converge to similar throughput levels at rate=50 (80.0, 75.7, 75.7 tok/s). The overhead profile is -29.9% to -33.6% vs baseline. At rate=50, fs-offload shows elevated ITL (804.7 ms vs 260.8 ms baseline), consistent with request queue depth (34.6 mean waiting requests vs 1.4 for baseline).

#### Qwen3-14B

All offload configurations show throughput above the no-offload baseline at rate=50. native-offload-20k achieves the highest gain (+14.5%). Latency metrics at rate=50 are within 7% across all four configurations (TTFT 22–23.5 s, ITL 304–326 ms).

#### Qwen3-32B-AWQ

No-offload peaks at rate=1 (51.2 tok/s). All offload configurations degrade throughput by -56 to -58%. The optimal concurrency shifts from rate=1 to rate=50–100 under offload. At rate=1, all offload configs show approximately 2× higher TTFT and 7–8× higher ITL vs no-offload (TTFT: 0.10 s baseline vs 0.21 s offload; ITL: 18.3 ms vs 145–148 ms).

![Performance Delta Heatmap](analysis/v0.5.1_delta_heatmap.png)
*Figure: Throughput delta (%) vs no-offload baseline. Rows are offload configurations, columns are model sizes. Qwen3-14B (green column) shows positive delta across all offload types. All other models (red) show degradation under offload.*

### Latency Analysis

Latency measured at rate=50 for 0.6B, 8B, and 14B; rate=1 for 32B-AWQ (which achieves peak throughput at rate=1).

#### Time to First Token (TTFT)

![TTFT Comparison](analysis/v0.5.1_latency_ttft.png)
*Figure: Median TTFT at peak-throughput concurrency by model and configuration. Qwen3-14B shows near-identical TTFT across all four configurations (22–23.5 s). Qwen3-8B fs-offload shows elevated TTFT (9.5 s) vs baseline (10.9 s), an anomaly attributable to queue dynamics.*

**TTFT at rate=50:**

| Model | no-offload | native-offload-20k | fs-offload | cpu+fs-offload-20k |
|-------|:----------:|:-----------------:|:----------:|:-----------------:|
| Qwen3-0.6B | 0.72 s | 0.69 s | 3.64 s | 0.29 s² |
| Qwen3-8B | 10.9 s | 19.0 s | 9.5 s | 25.9 s |
| Qwen3-14B | 22.6 s | 23.4 s | 23.5 s | 22.0 s |

**TTFT at rate=1 (Qwen3-32B-AWQ):** 0.10 s (no-offload) → 0.21 s (all offload configs, +2×)

² *Qwen3-0.6B cpu+fs-offload-20k at rate=50: anomalously low TTFT (0.29 s) and ITL (11.1 ms) with only 25.6 tok/s throughput, indicating a large fraction of requests are erroring or being dropped.*

#### Inter-Token Latency (ITL)

![ITL Comparison](analysis/v0.5.1_latency_itl.png)
*Figure: Median ITL at peak-throughput concurrency. Qwen3-8B fs-offload shows 804.7 ms ITL (3× the no-offload 260.8 ms), the largest relative deviation observed. Qwen3-14B ITL is within 7% across all configurations.*

**ITL at rate=50:**

| Model | no-offload | native-offload-20k | fs-offload | cpu+fs-offload-20k |
|-------|:----------:|:-----------------:|:----------:|:-----------------:|
| Qwen3-0.6B | 67.1 ms | 69.0 ms | 65.8 ms | 11.1 ms² |
| Qwen3-8B | 260.8 ms | 343.2 ms (+32%) | 804.7 ms (+208%) | 206.7 ms (-21%) |
| Qwen3-14B | 325.7 ms | 310.0 ms (-5%) | 306.4 ms (-6%) | 304.0 ms (-7%) |

**ITL at rate=1 (Qwen3-32B-AWQ):** 18.3 ms (no-offload) → 145–148 ms (all offload configs, +7–8×)

---

## Filesystem Offload Analysis

### Storage I/O Characterisation

Disk I/O was measured from PCP `disk.dev.*` metrics during all fs-offload and cpu+fs-offload benchmark runs.

![Disk I/O](analysis/v0.5.1_pcp_disk_io.png)
*Figure: Disk read and write throughput (MB/s) during fs-offload and cpu+fs-offload benchmark runs vs no-offload baseline. Values are near-zero across all runs.*

Peak observed disk I/O across all 64 filesystem-offload runs: **0.04 MB/s write, 0.03 MB/s read**. No-offload baseline shows similar background I/O (0.04 MB/s write for Qwen3-32B-AWQ).

The negligible disk I/O indicates that the IBM VPC block PVC is operating as a page-cache-backed filesystem during the 120-second benchmark window. KV cache blocks written to `/kvcache/kv-cache/` reside in OS page cache rather than reaching physical storage. As a result, the fs-offload path behaves as an additional CPU memory tier rather than a persistent storage tier during these benchmarks.

This has two implications:
1. The performance comparison between native-offload-20k (explicit CPU allocation) and fs-offload (filesystem-backed, effectively page-cache-backed) measures implementation overhead differences, not storage tier differences.
2. For deployments targeting persistent cross-restart or cross-instance KV cache reuse (the primary use case for `SharedStorageOffloadingSpec`), longer-running workloads or explicit cache flushing would be required to bypass the page cache.

### Qwen3-0.6B EngineCore Deadlock

At rate=300 and rate=500, vLLM's multiprocess EngineCore deadlocks under the combined load of high-concurrency requests and filesystem KV connector I/O. The API server process remains alive (HTTP 503 responses are returned by the gateway rather than connection failures). The failure is a shared memory broadcast queue starvation in `--distributed-executor-backend mp`: filesystem I/O worker threads block before they can drain the broadcast queue, causing the EngineCore to wait indefinitely for worker acknowledgement. The effect is total request failure (53,000+ 503 errors) for the full 120-second benchmark window. See the Qwen3-0.6B section under Throughput vs Concurrency for the full event sequence, and the accompanying bug report (`BUG-shm-broadcast-deadlock.md`) for reproduction steps.

### MultiConnector (cpu+fs-offload-20k) Behaviour

The `MultiConnector` implementation writes to both CPU and filesystem simultaneously and reads from CPU first (priority ordering). In these single-replica benchmarks:

- **Qwen3-14B**: cpu+fs-offload-20k (+7.3%) falls between fs-offload (+3.6%) and native-offload-20k (+14.5%)
- **Qwen3-8B**: cpu+fs-offload-20k (-33.6%) and fs-offload (-33.6%) are identical at peak
- **Qwen3-32B-AWQ**: all three offload configs show identical throughput (-56 to -58%)

External prefix cache hit rates (the vLLM metric tracking KV connector hits) ranged from 0–7.5% across all configurations and models, with Qwen3-8B cpu+fs-offload-20k showing the highest at 7.5%. These low rates are expected in a single-replica deployment where the primary use of the external cache is intra-request KV reuse, not cross-instance sharing.

---

## System-Level Analysis (PCP)

Performance Co-Pilot metrics were captured throughout all 128 benchmark runs. The analysis below focuses on rate=50 for 0.6B, 8B, and 14B models; rate=1 for 32B-AWQ.

### GPU Utilization

![GPU Utilization](analysis/v0.5.1_pcp_gpu_util.png)
*Figure: GPU utilization (% of GPU compute cycles, summed across 2× L40S) by configuration and model. Values exceeding 100% indicate both GPUs are active.*

GPU utilization increases with offload overhead:
- no-offload: 71–89% (combined both GPUs)
- native-offload-20k: 80–89%
- fs-offload: 83–96% (Qwen3-8B approaches saturation)
- cpu+fs-offload-20k: 29–108% (high variance; 0.6B shows low utilization due to instability)

For Qwen3-14B, all offload configs show higher GPU utilisation than no-offload while also achieving higher throughput.

### KV Cache Usage

![KV Cache Usage](analysis/v0.5.1_pcp_kvcache.png)
*Figure: GPU KV cache usage (%) at peak-throughput concurrency by configuration and model. fs-offload shows near-zero GPU KV cache usage for Qwen3-0.6B, consistent with aggressive offloading from GPU to storage.*

GPU KV cache utilization at peak:
- Qwen3-0.6B fs-offload: near 0% GPU KV cache (blocks offloaded aggressively to filesystem)
- Larger models (8B, 14B, 32B-AWQ): 27–44% GPU KV cache across all configurations
- native-offload-20k and cpu+fs-offload-20k maintain similar GPU KV cache levels to no-offload for larger models

Qwen3-0.6B fs-offload shows near-zero GPU KV cache usage, confirming blocks are being offloaded to the filesystem path.

### External Prefix Cache Hit Rates

![External Cache Hits](analysis/v0.5.1_pcp_external_hits.png)
*Figure: External prefix cache hit rate (hits/queries) by configuration and model. All values are below 8%, consistent with single-replica deployment where cross-instance cache sharing is inactive.*

The `vllm:external_prefix_cache_hits_total` and `vllm:external_prefix_cache_queries_total` metrics capture KV connector activity. Hit rates across all configurations and models: 0–7.5%. Qwen3-8B cpu+fs-offload-20k at rate=50 shows the highest rate (7.5%).

For the cpu+fs-offload-20k configuration, the external hit metric aggregates hits from both the CPU and filesystem sub-connectors without distinguishing between them. Attributing hits to specific tiers would require additional instrumentation not currently exposed by `MultiConnector`.

---

## API and Configuration Changes from v0.4.0 / v0.5.0

### Native Offload API Change

The `OffloadingConnector` allocation API changed between vLLM 0.11.x and 0.15.x:

| Version | Parameter | Unit |
|---------|-----------|------|
| v0.4.0 (vLLM 0.11.2) | `num_cpu_blocks` | Number of GPU KV-cache blocks |
| v0.5.0 (vLLM 0.14.1) | `num_cpu_blocks` | Same |
| v0.5.1 (vLLM 0.15.1) | `cpu_bytes_to_use` | Bytes of CPU memory |

The `num_cpu_blocks` parameter is no longer accepted in vLLM 0.15.1; specifying it causes a startup error (`cpu_bytes_to_use must be specified`).

### OffloadingSpec Factory

In vLLM 0.15.1, the `OffloadingSpecFactory` uses a `spec_name` parameter in `kv_connector_extra_config` to select the implementation:
- Default (no `spec_name`): `CPUOffloadingSpec` (built-in, CPU memory)
- `spec_name: SharedStorageOffloadingSpec`: requires external module via `spec_module_path`

Third-party specs are loaded via Python import from `spec_module_path`, making the connector framework extensible without vLLM source changes.

### MultiConnector

`MultiConnector` is registered in vLLM 0.15.1 and accepts a `connectors` list in `kv_connector_extra_config`, each element being a standard `KVTransferConfig`. The load and save semantics are:
- **Load**: first connector in list that reports available tokens is used
- **Save**: all connectors receive the write simultaneously

### GuideLLM

The `--sample-requests` flag was removed from the guidellm command per PR #591. This eliminates per-request sample data from the JSON output, reducing file sizes substantially at high concurrency (rate=650 files reduced from ~69 MB to ~4 MB).

---

## Deployment Considerations

### libstdc++ Version Constraint

The `llmd_fs_connector` v0.15.1 wheel requires `GLIBCXX_3.4.30` (GCC 12+). The `llm-d-cuda:v0.5.1` image is RHEL 9-based and provides only `GLIBCXX_3.4.29` (GCC 11). Workaround used in these benchmarks: `LD_PRELOAD` of the Nsight Compute-bundled `libstdc++.so.6` (`GLIBCXX_3.4.33`).

See [llm-d-kv-cache issue #445](https://github.com/llm-d/llm-d-kv-cache/issues/445) for status and upstream fix options.

### Filesystem Offload Deployment Notes

For `SharedStorageOffloadingSpec` to serve as persistent cross-restart or cross-instance cache:
1. The storage backend must be `ReadWriteMany` (RWX) for multi-instance sharing. The IBM VPC block PVC used here is `ReadWriteOnce` (RWO), limiting the setup to single-pod use.
2. Latency-sensitive deployments should account for the difference between page-cached (hot) and cold storage reads. These benchmarks only characterise the page-cached case.
3. The connector does not manage storage eviction. An external evictor (e.g., the PVC evictor from the llm-d-kv-cache repository) is required for bounded storage growth.

---

## Memory-Pressure Analysis

All four v0.5.1 configurations were re-run with per-model reduced `gpu_memory_utilization` to create GPU KV-cache pressure across all model sizes. This is the same gmu setting applied to v0.4.0 in REPORT-v0.4.0.md §Memory-Pressure Analysis, enabling direct cross-version comparison at matched memory pressure.

### Configuration

| Model | gmu | GPU KV tokens (gmu=0.9) | GPU KV tokens (mempress) |
|-------|:---:|:-----------------------:|:------------------------:|
| Qwen3-0.6B | 0.55 | ~634K | ~335K |
| Qwen3-8B | 0.65 | ~390K | ~215K |
| Qwen3-14B | 0.70 | ~268K | ~142K |
| Qwen3-32B-AWQ | 0.65 | ~207K | ~116K |

### Peak Throughput (tok/s)

| Config | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|--------|:----------:|:--------:|:---------:|:-------------:|
| no-offload | 526.9 | 117.3 | 71.5 | 51.2 |
| native-offload-20k | 644.3 (+22.3%) | 113.1 (-3.6%) | 78.9 (+10.4%) | 34.1 (-33.3%) |
| fs-offload | 217.6 (-58.7%) | 68.3 (-41.8%) | 42.7 (-40.3%) | 51.2 (0.0%) |
| cpu+fs-offload-20k | 75.7 (-85.6%) | 68.3 (-41.8%) | 42.7 (-40.3%) | 50.1 (-2.1%) |

![v0.5.1 Memory-Pressure Peak Throughput](analysis/v0.5.1-mempress_peak_throughput.png)
*Figure: Peak throughput at reduced gpu_memory_utilization. native-offload-20k shows throughput above the mempress no-offload baseline for 0.6B and 14B.*

![v0.5.1 Memory-Pressure Delta Heatmap](analysis/v0.5.1-mempress_delta_heatmap.png)
*Figure: % throughput delta vs mempress no-offload baseline.*

### External KV Cache Hit Rates

`external_prefix_cache_hits_total` / `external_prefix_cache_queries_total` at peak concurrency:

| Config | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|--------|:----------:|:--------:|:---------:|:-------------:|
| native-offload-20k | **26.9%** | **13.7%** | **8.5%** | 2.2% |
| fs-offload | ~0% | ~0% | ~0% | ~0% |
| cpu+fs-offload-20k | ~0% | ~0% | ~0% | ~0% |

native-offload-20k shows measurable external KV cache utilisation at mempress gmu values: 0.6B 26.9%, 8B 13.7%, 14B 8.5%, 32B-AWQ 2.2%. At gmu=0.9, the same configurations showed 0–7.5%.

fs-offload and cpu+fs-offload-20k show near-zero external hit rates despite their lower throughput. GPU KV-cache utilisation for these configs is below 2% (vs 44–74% for native-offload-20k), indicating the filesystem offload path does not effectively feed the GPU KV-cache in the configuration tested.

![v0.5.1 External Cache Hit Rates](analysis/v0.5.1-mempress_external_cache_hits.png)
*Figure: External prefix KV cache hit rate at peak concurrency. native-offload-20k shows active offload utilisation under memory pressure; filesystem configs show near-zero.*

### GPU KV-Cache Utilisation

| Config | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|--------|:----------:|:--------:|:---------:|:-------------:|
| no-offload | 74% | 48% | 47% | — |
| native-offload-20k | 48% | 44% | 45% | 42% |
| fs-offload | <2% | <2% | <2% | 4% |
| cpu+fs-offload-20k | <2% | <2% | <2% | ~0% |

![v0.5.1 GPU KV-Cache Utilisation](analysis/v0.5.1-mempress_gpu_kvcache.png)
*Figure: GPU KV-cache utilisation at peak concurrency.*

### Throughput vs Concurrency

![v0.5.1 Memory-Pressure Throughput Curves](analysis/v0.5.1-mempress_throughput_curves.png)
*Figure: Throughput vs concurrency at mempress gmu. 4-panel by model.*

### Comparison: gmu=0.9 vs gmu=mempress (native-offload-20k)

![gmu=0.9 vs mempress comparison](analysis/v0.5.1-mempress_vs_original.png)
*Figure: native-offload-20k throughput at gmu=0.9 vs gmu=mempress. Memory pressure converts the 0.6B outcome from -2.2% to +22.3% and reduces the 8B overhead from -29.9% to -3.6%.*

### Observations

**Qwen3-0.6B:** native-offload-20k at mempress gmu shows +22.3% vs the mempress no-offload baseline, and a 26.9% external KV cache hit rate. At gmu=0.9, the same config showed -2.2% with near-zero external hits. The TTFT at rate=50 decreases from 3,273ms (no-offload) to 2,465ms (native-offload-20k).

**Qwen3-8B:** native-offload-20k shows -3.6% vs no-offload at mempress gmu, compared to -29.9% at gmu=0.9 — a 26.3 pp reduction in overhead. External KV cache hit rate of 13.7% confirms active offload utilisation.

**Qwen3-14B:** native-offload-20k at +10.4%, external hit rate 8.5%.

**Qwen3-32B-AWQ:** native-offload-20k at -33.3% — the largest throughput reduction observed across the mempress suite. Mean waiting requests at rate=50 is 38.6 and median TTFT is 27,982ms. At gmu=0.65, the GPU KV-cache token capacity (116K tokens) may be insufficient for 20K-block CPU offload to provide a net benefit at this model size.

**fs-offload and cpu+fs-offload-20k:** Both show -40% to -86% throughput reduction. GPU KV-cache utilisation below 2% and near-zero external hit rates indicate the filesystem offload connector does not populate the GPU KV-cache effectively at the concurrency levels tested. This is consistent with the low disk I/O (≤0.04 MB/s) observed at gmu=0.9 — the page-cached storage path does not appear to differ materially at reduced gmu.

### Cross-Version Comparison at Matched Memory Pressure

The table below compares v0.4.0 and v0.5.1 native-offload configurations at the same per-model gmu values:

| Model | v0.4.0 nat-10k | v0.4.0 nat-20k | v0.5.1 nat-20k |
|-------|:--------------:|:--------------:|:--------------:|
| Qwen3-0.6B | -8.3% | +9.5% | +22.3% |
| Qwen3-8B | -19.3% | -9.2% | -3.6% |
| Qwen3-14B | 0.0% | +22.6% | +10.4% |
| Qwen3-32B-AWQ | -2.3% | -2.3% | -33.3% |

*All deltas vs same-version no-offload baseline at matched gmu.*

![Cross-version native offload comparison](analysis/mempress_crossversion_native_offload.png)
*Figure: Native offload throughput delta across versions at matched memory pressure.*

For 0.6B and 8B, v0.5.1 shows lower offload overhead than v0.4.0 at matched memory pressure. For 14B, v0.4.0-20k shows a larger absolute gain (+22.6%) than v0.5.1-20k (+10.4%), noting that the no-offload baselines differ (66.1 vs 71.5 tok/s). The 32B-AWQ regression at v0.5.1 (-33.3%) has no equivalent in v0.4.0 (-2.3%).

---

## Observations

Results across all v0.5.1 configurations and experiments (gmu=0.9 and memory-pressure runs):

1. **Qwen3-14B** shows throughput above the no-offload baseline for all offload types at gmu=0.9: native-offload-20k +14.5%, cpu+fs-offload-20k +7.3%, fs-offload +3.6%. All other models show throughput reduction under all offload configurations at gmu=0.9.

2. **Qwen3-14B no-offload** regressed from v0.5.0 to v0.5.1: 66.1 → 58.7 tok/s (-11.2%), returning to the v0.4.0 value. All other models are stable vs v0.5.0.

3. **Qwen3-0.6B native-offload-20k**: -2.2% at gmu=0.9, recovering from v0.4.0's -29.1% with the same hardware.

4. **Filesystem offload disk I/O**: ≤0.04 MB/s across all 64 fs-offload runs. The IBM VPC block PVC operates via OS page cache during 120-second benchmark windows.

5. **Qwen3-0.6B fs-offload deadlock** at rate≥300: 53,000+ HTTP 503 errors, 0 completions. EngineCore shared-memory broadcast queue starvation under high KV write pressure. ([issue #457](https://github.com/llm-d/llm-d-kv-cache/issues/457))

6. **External KV cache hit rate** at gmu=0.9: 0–7.5% across all configurations and models. At mempress gmu, native-offload-20k shows 26.9% (0.6B), 13.7% (8B), 8.5% (14B), 2.2% (32B-AWQ). fs-offload and cpu+fs-offload-20k remain near-zero under pressure.

7. **Memory-pressure results** (matched gmu, native-offload-20k): 0.6B +22.3%, 8B -3.6%, 14B +10.4%, 32B-AWQ -33.3% vs mempress no-offload baseline. The 32B-AWQ regression has no equivalent in v0.4.0 (-2.3% at matched pressure).

8. **TTFT at rate=50 (gmu=0.9)**: 0.6B: no-offload 0.72s, native-offload-20k 0.69s, fs-offload 3.64s. 8B: native-offload-20k 19.0s (+74%), cpu+fs-offload-20k 25.9s (+138%). 14B: all configs within 7% of each other (22.0–23.5s).

9. **libstdc++ ABI incompatibility**: `llmd_fs_connector` v0.15.1 wheel requires `GLIBCXX_3.4.30+`; `llm-d-cuda:v0.5.1` (RHEL 9) provides `GLIBCXX_3.4.29`. ([issue #445](https://github.com/llm-d/llm-d-kv-cache/issues/445))

---

## Appendix: Methodology

### Benchmark Execution

All benchmarks used the `run-benchmark.sh` script with:
- GuideLLM concurrent profile, 120 seconds duration
- PCP pod restarted before each run to create a fresh archive
- EPP ConfigMap reapplied before each run
- Model server restarted and readiness probed via `/health` before guidellm fires
- Results collected via chunked base64 transfer (avoids kubectl exec binary stream truncation)
- Results compressed with zstd locally

### PCP Metrics Collection

PCP daemon collects at 10-second intervals (pcp-zeroconf package, default configuration). Metrics scraped:
- `openmetrics.*` (all vLLM/llm-d Prometheus metrics via openmetrics PMDA)
- `nvidia.*` (GPU utilization, memory, power via DCGM)
- `disk.dev.*` (disk I/O per device)
- `kernel.all.cpu.*` (CPU utilization)

### Analysis

Analysis script: `scripts/analyze-v0.5.1.py`

Output files:
- `analysis/v0.5.1_throughput_all.csv` — raw throughput per run and concurrency level
- `analysis/v0.5.1_pcp_metrics.csv` — PCP metrics extracted per run
- `analysis/v0.5.1_summary.csv` — peak throughput and PCP metrics per configuration

### Data Files

- GuideLLM results: `results/1x2xL40S_upstream-llm-d-0.5.1_*/guidellm-results.json.zst`
- PCP archives: `results/1x2xL40S_upstream-llm-d-0.5.1_*/pcp-archives/nathans-offload-nndsn-master-0/`
- vLLM startup logs: `results/1x2xL40S_upstream-llm-d-0.5.1_*/vllm-startup.log.zst`

---

*Report generated from benchmark runs completed March 2026*
*Analysis framework: Performance Co-Pilot + GuideLLM*
*System: Single Node OpenShift on IBM Cloud with 2× NVIDIA L40S GPUs*
