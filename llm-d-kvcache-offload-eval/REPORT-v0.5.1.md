# llm-d v0.5.1 KV-Cache Offload Evaluation

This report evaluates KV-cache offload strategies in llm-d v0.5.1 (vLLM 0.15.1) and places the results in the context of prior evaluations on v0.4.0 (vLLM 0.11.2) and v0.5.0 (vLLM 0.14.1). Two new offload paths are evaluated for the first time: filesystem offload via the `llmd_fs_connector` external module, and hierarchical CPU+filesystem offload via vLLM's `MultiConnector`.

This report supersedes REPORT-v0.5.0.md.

**Software Versions:**
- **llm-d**: v0.5.1 (vLLM 0.15.1, GuideLLM)
- **Prior baselines**: llm-d v0.5.0 (vLLM 0.14.1), llm-d v0.4.0 (vLLM 0.11.2)

**Hardware:** 2× NVIDIA L40S GPUs (48 GB total VRAM), 48 vCPUs, OpenShift on IBM Cloud

**Models:** Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ

**Concurrency levels:** 1, 50, 100, 150, 300, 400, 500, 650

---

## Summary

128 benchmark runs were collected across four KV-cache configurations, four model sizes, and eight concurrency levels. The configurations tested are:

- **no-offload**: GPU-only KV-cache (baseline)
- **native-offload-20k**: CPU offload via `OffloadingConnector` / `CPUOffloadingSpec`, 20K-block equivalent CPU allocation per model
- **fs-offload**: Filesystem offload via `OffloadingConnector` / `SharedStorageOffloadingSpec` (external `llmd_fs_connector` wheel), IBM VPC block storage PVC
- **cpu+fs-offload-20k**: Hierarchical CPU+filesystem offload via `MultiConnector` (CPU first on load, simultaneous write to both)

**Observations:**

1. **Qwen3-14B is the only model that benefits from all offload configurations**: native-offload-20k shows +14.5% throughput vs baseline; cpu+fs-offload-20k +7.3%; fs-offload +3.6%.

2. **Qwen3-0.6B and Qwen3-8B show throughput loss under all offload configurations**: native-offload-20k -2.2% and -29.9% respectively. Filesystem-based configs show instability for Qwen3-0.6B at sustained concurrency.

3. **Qwen3-32B-AWQ shows -56 to -58% throughput loss under all offload configs**: consistent across native CPU, filesystem, and MultiConnector configurations.

4. **Filesystem offload disk I/O was negligible** (peak 0.04 MB/s): the IBM VPC block PVC is backed by the OS page cache during the 120-second benchmark window, so the fs-offload path behaves similarly to CPU-backed memory rather than persistent storage I/O.

5. **External prefix cache hit rates are low (0–7.5%)**: expected for a single-replica deployment where cross-instance cache reuse does not apply.

6. **Qwen3-14B no-offload throughput regressed from v0.5.0 to v0.5.1**: 66.1 tok/s (v0.5.0) → 58.7 tok/s (v0.5.1), returning to the v0.4.0 level. All other models are stable vs v0.5.0.

**Peak Throughput Summary:**

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

native-offload-20k tracks the baseline closely at all concurrency levels (-2.2% at peak). The fs-offload and cpu+fs-offload-20k configurations show throughput at rate=1 (268.8 and 211.2 tok/s respectively) but drop to 85.3 and 25.6 tok/s at rate=50, and show zero completed requests at rate=300 and rate=500. This indicates the `SharedStorageOffloadingSpec` path is unstable at sustained high concurrency for this model size. The IBM VPC block PVC exhibited negligible disk I/O (≤0.04 MB/s) throughout, suggesting the instability is not storage-I/O bound.

#### Qwen3-8B

native-offload-20k, fs-offload, and cpu+fs-offload-20k all converge to similar throughput levels at rate=50 (80.0, 75.7, 75.7 tok/s). The overhead profile is -29.9% to -33.6% vs baseline. At rate=50, fs-offload shows elevated ITL (804.7 ms vs 260.8 ms baseline), consistent with request queue depth (34.6 mean waiting requests vs 1.4 for baseline).

#### Qwen3-14B

All offload configurations show throughput above the no-offload baseline at rate=50. native-offload-20k achieves the highest gain (+14.5%). Latency metrics at rate=50 are within 7% across all four configurations (TTFT 22–23.5 s, ITL 304–326 ms), indicating that the throughput differences reflect scheduler capacity rather than individual request latency.

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

### Qwen3-0.6B Instability

The Qwen3-0.6B model shows zero completed requests at rate=300 and rate=500 for both fs-offload and cpu+fs-offload-20k configurations. PCP data shows GPU utilization drops to near zero at these concurrency levels for these configurations, while the no-offload and native-offload-20k configurations continue operating. The cause is not identified from the available metrics. The vLLM startup log confirms `SharedStorageOffloadingSpec` initialised correctly (both workers). The instability may reflect a connector scheduling interaction at high concurrency that does not manifest for larger models.

### MultiConnector (cpu+fs-offload-20k) Behaviour

The `MultiConnector` implementation writes to both CPU and filesystem simultaneously and reads from CPU first (priority ordering). In these single-replica benchmarks:

- **Qwen3-14B**: cpu+fs-offload-20k (+7.3%) falls between fs-offload (+3.6%) and native-offload-20k (+14.5%). The combined overhead of managing two connectors appears to offset some of the CPU cache benefit.
- **Qwen3-8B**: cpu+fs-offload-20k (-33.6%) and fs-offload (-33.6%) are identical at peak, with the CPU connector adding write overhead without read benefit at this concurrency.
- **Qwen3-32B-AWQ**: All three offload configs show identical throughput (-56 to -58%), suggesting the bottleneck is not the specific offload mechanism.

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

The pattern is consistent with v0.4.0 observations: higher GPU utilization with offload reflects CPU-GPU transfer cycles consuming compute time rather than token generation. For Qwen3-14B, all offload configs show higher GPU utilization than no-offload while also achieving higher throughput, indicating the GPU time is productive (processing more requests from the expanded effective KV cache).

### KV Cache Usage

![KV Cache Usage](analysis/v0.5.1_pcp_kvcache.png)
*Figure: GPU KV cache usage (%) at peak-throughput concurrency by configuration and model. fs-offload shows near-zero GPU KV cache usage for Qwen3-0.6B, consistent with aggressive offloading from GPU to storage.*

GPU KV cache utilization at peak:
- Qwen3-0.6B fs-offload: near 0% GPU KV cache (blocks offloaded aggressively to filesystem)
- Larger models (8B, 14B, 32B-AWQ): 27–44% GPU KV cache across all configurations
- native-offload-20k and cpu+fs-offload-20k maintain similar GPU KV cache levels to no-offload for larger models

The Qwen3-0.6B result confirms the connector is functioning: GPU KV cache is being offloaded to the filesystem path. The instability at high concurrency is therefore not due to the offloading mechanism failing, but likely a scheduling or connector interaction at high request rates.

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

The `llmd_fs_connector` v0.15.1 wheel requires `GLIBCXX_3.4.30` or later, compiled with GCC 12+. The `llm-d-cuda:v0.5.1` image is RHEL 9-based (GCC 11, `libstdc++-11.5.0-11.el9.x86_64`, providing up to `GLIBCXX_3.4.29`).

Workaround: `LD_PRELOAD` of the Nsight Compute-bundled `libstdc++.so.6` (providing `GLIBCXX_3.4.33`). This requires the NVIDIA Nsight Compute toolkit to be present in the container image.

The upstream fix is to rebuild the wheel against GCC 11 (targeting RHEL 9 / `GLIBCXX_3.4.29`), or update the base image to RHEL 10 / include a GCC 12 libstdc++.

### Filesystem Offload Deployment Notes

For `SharedStorageOffloadingSpec` to serve as persistent cross-restart or cross-instance cache:
1. The storage backend must be `ReadWriteMany` (RWX) for multi-instance sharing. The IBM VPC block PVC used here is `ReadWriteOnce` (RWO), limiting the setup to single-pod use.
2. Latency-sensitive deployments should account for the difference between page-cached (hot) and cold storage reads. These benchmarks only characterise the page-cached case.
3. The connector does not manage storage eviction. An external evictor (e.g., the PVC evictor from the llm-d-kv-cache repository) is required for bounded storage growth.

---

## Observations Summary

| Observation | Detail |
|-------------|--------|
| Qwen3-14B benefits from all offload types | +14.5% (native-20k), +7.3% (cpu+fs), +3.6% (fs) vs baseline |
| Qwen3-0.6B fs-offload unstable at high concurrency | Zero completed requests at rate=300 and rate=500 |
| Qwen3-8B and Qwen3-32B-AWQ: degradation under all offload | -29.9% to -58.3% depending on config |
| Disk I/O negligible for fs-offload | ≤0.04 MB/s; storage operates via OS page cache |
| cpu+fs-offload MultiConnector: save parallelism, load priority | Not a true tiered cache; simultaneous GPU→CPU and GPU→FS writes |
| External cache hit rate: 0–7.5% | Expected for single-replica; no cross-instance sharing |
| Qwen3-14B no-offload regressed -11.2% vs v0.5.0 | 66.1 → 58.7 tok/s; returns to v0.4.0 level |
| Qwen3-0.6B native-offload-20k near-parity | -2.2% vs baseline; confirms v0.5.0 recovery from v0.4.0 -29.1% |
| libstdc++ ABI incompatibility | fs_connector wheel requires GLIBCXX_3.4.30+; RHEL9 image provides 3.4.29 |

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
