# llm-d v0.6.0 KV-Cache Offload Evaluation

This report evaluates KV-cache offload strategies in llm-d v0.6.0 (vLLM 0.17.1, GuideLLM 0.6.0) and places the results in the context of the prior evaluation on v0.5.1 (vLLM 0.15.1). Six configurations are evaluated: no-offload (GPU-only baseline), native CPU offload via `OffloadingConnector`, LMCache with local and Valkey remote backends, filesystem offload via `SharedStorageOffloadingSpec`, and hierarchical CPU+filesystem offload via `MultiConnector`. LMCache is evaluated for the first time with LMCache v0.4.2 (upgraded from v0.3.15 in v0.5.1). A memory-pressure suite re-ran the four non-filesystem configurations with reduced `gpu_memory_utilization` per model (0.55–0.70 vs the default 0.9).

**Software Versions:**

| Component | Version |
|-----------|---------|
| llm-d | v0.6.0 |
| vLLM | 0.17.1 (bundled in llm-d-cuda:v0.6.0) |
| LMCache | v0.4.2 (`lmcache/vllm-openai:v0.4.2`) |
| GuideLLM | 0.6.0 |
| llmd_fs_connector | 0.18.0 |
| gateway-api-inference-extension (EPP) | v0.7.1 |
| Valkey | 8-alpine |
| OpenShift | 4.22.0 (SNO) |
| PCP | 7.0.3 |

**Hardware:** 2× NVIDIA L40S GPUs (48 GB total VRAM), 48 vCPUs, OpenShift on IBM Cloud

**Models:** Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ

**Concurrency levels:** 1, 50, 100, 150, 300, 400, 500, 650

---

## Summary

297 benchmark runs across six KV-cache configurations at gmu=0.9 (four models), four configurations under memory pressure (three models), and eight concurrency levels. Compared to v0.5.1, vLLM 0.17.1 delivers large throughput gains for Qwen3-0.6B (+26.8%) and Qwen3-8B (+72.9%) at gmu=0.9 no-offload baseline. Qwen3-14B no-offload decreases by -5.5% vs v0.5.1. Qwen3-32B-AWQ no-offload is stable (-2.1%).

**Configurations:**
1. **no-offload**: GPU-only KV-cache (baseline)
2. **native-offload-20k**: CPU offload via `OffloadingConnector` / `CPUOffloadingSpec`, 20K-block equivalent
3. **lmcache-local**: LMCache v0.4.2 with local CPU backend (`lmcache/vllm-openai:v0.4.2`)
4. **lmcache-valkey**: LMCache v0.4.2 with Valkey remote backend
5. **fs-offload**: Filesystem offload via `SharedStorageOffloadingSpec` (`llmd_fs_connector` v0.18.0), IBM VPC block PVC
6. **cpu+fs-offload-20k**: Hierarchical CPU+filesystem offload via `MultiConnector`

**Note:** 32B-AWQ offload configs (mempress only) were not run due to vLLM #38515 crash (negative Prometheus counter under KV offload + high preemption); 32B-AWQ no-offload is included at both gmu levels. Filesystem offload was not run under mempress conditions.

At default gpu_memory_utilization=0.9, only Qwen3-14B has insufficient GPU KV-cache to benefit from CPU offload on this hardware (20.58 GiB vs 269K token capacity). Memory-pressure runs reduce gmu per-model to create KV-cache constraint across all sizes, providing a more complete characterisation of offload behaviour.

**Peak Throughput (mempress gmu — primary result set):**

| Model | Config | v0.6.0 (tok/s) | Rate | vs no-offload | vs v0.5.1 |
|-------|--------|:--------------:|:----:|:-------------:|:---------:|
| Qwen3-0.6B | no-offload | 524.8 | 50 | — | -0.4% |
| | native-offload-20k | 794.7 | 50 | **+51.4%** | +23.3% |
| | lmcache-local | 426.7 | 50 | -18.7% | -15.1% |
| | lmcache-valkey | 474.7 | 50 | -9.6% | -4.9% |
| Qwen3-8B | no-offload | 104.5 | 100 | — | -10.9% |
| | native-offload-20k | 87.5 | 50 | -16.3% | -22.7% |
| | lmcache-local | 104.5 | 100 | 0.0% | -12.5% |
| | lmcache-valkey | 107.7 | 50 | +3.1% | -9.0% |
| Qwen3-14B | no-offload | 69.3 | 100 | — | -3.0% |
| | native-offload-20k¹ | 61.9 | 300 | -10.8% | -21.6% |
| | lmcache-local | 68.3 | 100 | -1.5% | -1.5% |
| | lmcache-valkey | 68.3 | 100 | -1.5% | -3.0% |
| Qwen3-32B-AWQ | no-offload | 50.1 | 1 | — | -2.1% |
| | offload configs | not run | — | — | — |

¹ *Qwen3-14B native-offload-20k mempress: three rates excluded due to vLLM #38515 (>10% error rate: rate=50, rate=100, rate=500). Peak from clean rates (rate=150–650, excluding rate=500).*

Under memory pressure, Qwen3-0.6B native-offload-20k reaches +51.4% vs the mempress no-offload baseline (up from +22.3% in v0.5.1). Qwen3-8B lmcache-valkey delivers +3.1%, recovering from -28.1% at gmu=0.9. LMCache shows -1.5% to 0.0% for 8B and 14B under mempress.

**Peak Throughput at default gpu_memory_utilization=0.9 (unconstrained conditions):**

| Model | no-offload | native-offload-20k | lmcache-local | lmcache-valkey | fs-offload | cpu+fs-offload-20k |
|-------|:----------:|:------------------:|:-------------:|:--------------:|:----------:|:-----------------:|
| Qwen3-0.6B | 807.5 tok/s | 809.6 (+0.3%) | 665.6 (-17.6%) | 789.3 (-2.2%) | 320.6 (-60.3%) | 789.3 (-2.2%) |
| Qwen3-8B | 197.3 tok/s | 184.5 (-6.5%) | 194.1 (-1.6%) | 141.9 (-28.1%) | **199.5 (+1.1%)** | 197.3 (0.0%) |
| Qwen3-14B | 55.5 tok/s | 56.5 (+1.9%) | 54.4 (-1.9%) | 56.5 (+1.9%) | **163.2 (+194.2%)** | **164.3 (+196.2%)** |
| Qwen3-32B-AWQ | 50.1 tok/s | 18.1 (-63.8%) | 18.1 (-63.8%) | 19.2 (-61.7%) | 18.1 (-63.8%) | 18.1 (-63.8%) |

At gmu=0.9, native-offload-20k shows near-zero overhead for Qwen3-0.6B (+0.3%) and Qwen3-14B (+1.9%). Qwen3-8B incurs -6.5% with native-offload-20k, a +23.4 pp improvement from v0.5.1 (-29.9%). lmcache-valkey shows elevated TTFT for Qwen3-8B (6.555 s vs 0.696 s no-offload) associated with -28.1% throughput overhead at this model size. fs-offload delivers +194.2% for Qwen3-14B and +1.1% for Qwen3-8B; cpu+fs-offload-20k delivers +196.2% for 14B and 0.0% for 8B. Qwen3-0.6B fs-offload shows -60.3% while cpu+fs-offload-20k matches lmcache-valkey at -2.2%.

---

## Test Configuration

### Hardware

**System:** OpenShift cluster on IBM Cloud (Single Node OpenShift, SNO)
- **GPUs**: 2× NVIDIA L40S (24 GB VRAM each, 48 GB total)
  - Tensor Parallelism: 2 GPUs per model
- **CPU**: 48 vCPUs
- **Storage**: IBM VPC block PVC (256 GiB, `ibmc-vpc-block-custom`, 6,000 IOPS) mounted at `/kvcache` for filesystem offload

### Software

See software versions table in the preamble above.

**EPP breaking change in v0.6.0:** gateway-api-inference-extension v0.7.1 removed the `--kv-cache-usage-percentage-metric` flag (present in v0.5.x EPP). The EPP configuration in v0.6.0 no longer accepts this flag; all EPP manifests were updated to remove it.

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

CPU offload via `OffloadingConnector` / `CPUOffloadingSpec`. Allocation specified in bytes (`cpu_bytes_to_use`) — same API as v0.5.1 (unchanged from vLLM 0.15.1 to 0.17.1).

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

#### 3. lmcache-local

LMCache v0.4.2 with local CPU backend. Uses the `lmcache/vllm-openai:v0.4.2` image (patched vLLM 0.17.1 build with LMCache integrated). Upgraded from v0.3.15 used in v0.5.1. Per-model CPU cache size matched to v0.5.1 for comparability.

| Model | LMCACHE_MAX_LOCAL_CPU_SIZE |
|-------|:-------------------------:|
| Qwen3-0.6B | 4 GB |
| Qwen3-8B | 9 GB |
| Qwen3-14B | 29 GB |
| Qwen3-32B-AWQ | 10 GB |

```bash
# Environment variables on lmcache/vllm-openai:v0.4.2 image
HOME=/tmp
HF_HOME=/data/.hf
LMCACHE_MAX_LOCAL_CPU_SIZE=<per_model_GB>
PYTHONHASHSEED=123
```

#### 4. lmcache-valkey

LMCache v0.4.2 with Valkey remote backend.

```bash
# Environment variables on lmcache/vllm-openai:v0.4.2 image
HOME=/tmp
HF_HOME=/data/.hf
LMCACHE_REMOTE_URL=valkey://valkey.<namespace>.svc.cluster.local:6379
PYTHONHASHSEED=123
```

Valkey pod restarted before each lmcache-valkey run to clear cache state between benchmark rates.

#### 5. fs-offload

Filesystem offload via `SharedStorageOffloadingSpec` from the `llmd_fs_connector` v0.18.0 wheel. The v0.6.0 image (llm-d-cuda:v0.6.0) ships libstdc++ built with `-static-libstdc++` (fix from [llm-d/llm-d-kv-cache#498](https://github.com/llm-d/llm-d-kv-cache/pull/498)), eliminating the GLIBCXX version dependency that required the LD_PRELOAD workaround in v0.5.1.

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024 \
  --distributed-executor-backend mp \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"spec_name":"SharedStorageOffloadingSpec",
      "shared_storage_path":"/kvcache/kv-cache/","block_size":256,
      "threads_per_gpu":128,"spec_module_path":"llmd_fs_backend.spec"}}'
```

#### 6. cpu+fs-offload-20k

Hierarchical offload via `MultiConnector`. Writes to CPU and filesystem simultaneously; reads from CPU first.

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
         "threads_per_gpu":128,"spec_module_path":"llmd_fs_backend.spec"}}
    ]}}'
```

**Memory-pressure `gpu_memory_utilization` per model:**

| Model | gmu=0.9 | mempress gmu |
|-------|:-------:|:------------:|
| Qwen3-0.6B | 0.9 | 0.55 |
| Qwen3-8B | 0.9 | 0.65 |
| Qwen3-14B | 0.9 | 0.70 |
| Qwen3-32B-AWQ | 0.9 | 0.65 (no-offload only) |

---

## Known Issues

### vLLM #38515: Crash Under KV Offload + High Preemption

vLLM #38515 describes a negative Prometheus counter that causes vLLM to crash and restart when KV cache offload is active and GPU-side preemption is occurring. The crash is triggered when the native `OffloadingConnector` or LMCache backend attempts to offload while blocks are being preempted from GPU KV cache. PRs #37354, #38712, and #37460 address related aspects; none are merged in vLLM 0.17.1.

**Observed impact in v0.6.0:**

In the Qwen3-14B mempress native-offload-20k runs, the crash triggers at concurrency levels where preemption frequency is high. Three benchmark rates show error rates above 10% (consistent with vLLM restart under load):
- rate=50: 99.9% error rate (33,806 errors / 33,874 total)
- rate=100: 99.5% error rate (23,992 errors / 24,120 total)
- rate=500: 92.8% error rate (7,022 errors / 7,566 total)

These runs are excluded from peak throughput selection. The clean peak for Qwen3-14B mempress native-offload-20k is 61.9 tok/s at rate=300 (-10.8% vs no-offload baseline; -21.6% vs v0.5.1).

The 32B-AWQ mempress offload configurations were not run at all for native-offload-20k, lmcache-local, and lmcache-valkey due to confirmed crashes at all concurrency levels during initial testing. Qwen3-32B-AWQ mempress no-offload (50.1 tok/s) is included and unaffected.

### libstdc++ ABI: Resolved in v0.6.0 Builds

The `llmd_fs_connector` wheel (v0.18.0) requires GLIBCXX_3.4.30+. The llm-d-cuda:v0.6.0 image removed Nsight Compute (which provided the workaround libstdc++ in v0.5.1 via LD_PRELOAD). Fix PR [llm-d/llm-d-kv-cache#498](https://github.com/llm-d/llm-d-kv-cache/pull/498) added `-static-libstdc++` to `setup.py`, embedding the C++ runtime in the `.so` and eliminating the runtime GLIBCXX version dependency. This fix is present in the v0.18.0 wheel used for these benchmarks; no LD_PRELOAD workaround is required.

---

## Memory-Pressure Analysis

Memory-pressure runs use reduced `gpu_memory_utilization` per model (see §Test Configuration) to create GPU KV-cache pressure across all model sizes. 32B-AWQ offload configs were not run due to vLLM #38515.

### Peak Throughput (mempress)

| Model | Config | v0.6.0 (tok/s) | Rate | vs no-offload | vs v0.5.1 |
|-------|--------|:--------------:|:----:|:-------------:|:---------:|
| Qwen3-0.6B | no-offload | 524.8 | 50 | — | -0.4% |
| | native-offload-20k | 794.7 | 50 | **+51.4%** | +23.3% |
| | lmcache-local | 426.7 | 50 | -18.7% | -15.1% |
| | lmcache-valkey | 474.7 | 50 | -9.6% | -4.9% |
| Qwen3-8B | no-offload | 104.5 | 100 | — | -10.9% |
| | native-offload-20k | 87.5 | 50 | -16.3% | -22.7% |
| | lmcache-local | 104.5 | 100 | 0.0% | -12.5% |
| | lmcache-valkey | 107.7 | 50 | +3.1% | -9.0% |
| Qwen3-14B | no-offload | 69.3 | 100 | — | -3.0% |
| | native-offload-20k¹ | 61.9 | 300 | -10.8% | -21.6% |
| | lmcache-local | 68.3 | 100 | -1.5% | -1.5% |
| | lmcache-valkey | 68.3 | 100 | -1.5% | -3.0% |
| Qwen3-32B-AWQ | no-offload | 50.1 | 1 | — | -2.1% |
| | offload configs | not run | — | — | — |

¹ *Qwen3-14B native-offload-20k mempress: three rates excluded due to vLLM #38515 (>10% error rate: rate=50, rate=100, rate=500). Peak from clean rates (rate=150–650, excluding rate=500).*

### Version Comparison (mempress)

Qwen3-0.6B native-offload-20k mempress: +51.4% vs no-offload in v0.6.0, up from +22.3% in v0.5.1. The absolute throughput (794.7 tok/s) is +23.3% above v0.5.1 (644.3 tok/s), with no-offload baseline stable (-0.4%).

Qwen3-8B native-offload-20k mempress: -16.3% vs baseline in v0.6.0, compared to -3.6% in v0.5.1. The absolute throughput (87.5 tok/s) is -22.7% below v0.5.1 (113.1 tok/s), while the no-offload baseline also decreased (-10.9%). The net offload overhead increase is -12.7 pp.

Qwen3-14B native-offload-20k mempress: -10.8% vs baseline at rate=300 (clean runs only), compared to +10.4% in v0.5.1. The impact of vLLM #38515 makes this comparison uncertain; the true steady-state capability of the connector at mempress gmu may be higher.

![Mempress Peak Throughput](analysis/v0.6.0_mempress_peak_throughput.png)
*Figure: Memory-pressure peak throughput by configuration for 3 models. Hatched bars show v0.5.1 values.*

![Mempress Version Comparison](analysis/v0.6.0_mempress_version_comparison.png)
*Figure: v0.5.1 vs v0.6.0 mempress peak throughput — no-offload and native-offload-20k configurations.*

---

## v0.6.0 Performance Results (gmu=0.9)

At gmu=0.9, most models are not GPU KV-cache limited on this hardware. Only Qwen3-14B (20.58 GiB GPU KV-cache, 269K token capacity) shows meaningful throughput sensitivity to CPU offload under these unconstrained conditions.

### Peak Throughput

| Model | Config | Peak (tok/s) | Optimal Rate | vs Baseline |
|-------|--------|:------------:|:------------:|:-----------:|
| **Qwen3-0.6B** | no-offload | 807.5 | 50 | — |
| | native-offload-20k | 809.6 | 50 | +0.3% |
| | lmcache-local | 665.6 | 50 | -17.6% |
| | lmcache-valkey | 789.3 | 50 | -2.2% |
| | fs-offload | 320.6 | 50 | -60.3% |
| | cpu+fs-offload-20k | 789.3 | 50 | -2.2% |
| **Qwen3-8B** | no-offload | 197.3 | 50 | — |
| | native-offload-20k | 184.5 | 50 | -6.5% |
| | lmcache-local | 194.1 | 50 | -1.6% |
| | lmcache-valkey | 141.9 | 50 | -28.1% |
| | fs-offload | 199.5 | 50 | +1.1% |
| | cpu+fs-offload-20k | 197.3 | 50 | 0.0% |
| **Qwen3-14B** | no-offload | 55.5 | 100 | — |
| | native-offload-20k | 56.5 | 100 | +1.9% |
| | lmcache-local | 54.4 | 100 | -1.9% |
| | lmcache-valkey | 56.5 | 50 | +1.9% |
| | fs-offload | 163.2 | 50 | +194.2% |
| | cpu+fs-offload-20k | 164.3 | 50 | +196.2% |
| **Qwen3-32B-AWQ** | no-offload | 50.1 | 1 | — |
| | native-offload-20k | 18.1 | 100 | -63.8% |
| | lmcache-local | 18.1 | 100 | -63.8% |
| | lmcache-valkey | 19.2 | 50 | -61.7% |
| | fs-offload | 18.1 | 100 | -63.8% |
| | cpu+fs-offload-20k | 18.1 | 100 | -63.8% |

![Peak Throughput](analysis/v0.6.0_peak_throughput.png)
*Figure: 4-panel peak throughput by configuration for each model. Hatched bars show v0.5.1 values for comparison.*

### Throughput vs Concurrency

![Throughput Curves](analysis/v0.6.0_throughput_curves.png)
*Figure: Output token throughput vs concurrency level for all four models across four configurations (gmu=0.9). Each panel shows one model.*

#### Qwen3-0.6B

No-offload and native-offload-20k track closely across all concurrency levels (+0.3% at peak rate=50). lmcache-valkey tracks near no-offload (-2.2% at rate=50). lmcache-local shows -17.6% at peak, with throughput declining from rate=100 onward, consistent with CPU cache eviction pressure at higher concurrency.

#### Qwen3-8B

No-offload peaks sharply at rate=50 (197.3 tok/s) and falls to 90–100 tok/s at rate=100+, indicating GPU KV-cache saturation above rate=50. lmcache-local tracks no-offload at rate=50 (-1.6%) before diverging. native-offload-20k achieves 184.5 tok/s at rate=50 (-6.5%), a 23.4 pp overhead improvement from v0.5.1 (-29.9%). lmcache-valkey shows 141.9 tok/s at rate=50 (-28.1%) with elevated TTFT (6.555 s vs 0.696 s baseline), indicating Valkey round-trip overhead at this concurrency.

#### Qwen3-14B

Throughput curves are relatively flat across all configurations between rate=50 and rate=650 (52–56 tok/s range), consistent with Qwen3-14B being the GPU-memory-constrained model on this hardware. native-offload-20k and lmcache-valkey both achieve +1.9% vs baseline at their respective optimal rates. lmcache-local shows -1.9% (-1.1 tok/s).

#### Qwen3-32B-AWQ

No-offload peaks at rate=1 (50.1 tok/s) with throughput declining to 17–19 tok/s at rate=50+, consistent with GPU KV-cache saturation from the large model at higher concurrency. All offload configurations hold at 18–19 tok/s across rate=50–650 (-61.7% to -63.8%). Optimal rate shifts from rate=1 to rate=50–100 under offload (GPU KV-cache memory freed by offload allows more concurrent requests, but at lower total throughput). TTFT increases from 0.115 s (no-offload, rate=1) to 0.216–0.233 s (offload configs, rate=1).

![Delta Heatmap](analysis/v0.6.0_delta_heatmap.png)
*Figure: Throughput delta (%) vs no-offload baseline (magma colormap). Positive values (lighter) indicate throughput gains; negative values (darker) indicate overhead.*

### Latency at Rate=50 (gmu=0.9)

Rate=1 used for Qwen3-32B-AWQ (peak throughput rate).

| Model | Config | TTFT (s) | ITL (ms) |
|-------|--------|:--------:|:--------:|
| Qwen3-0.6B | no-offload | 0.248 | 47.4 |
| | native-offload-20k | 0.264 | 46.8 |
| | lmcache-local | 0.376 | 57.5 |
| | lmcache-valkey | 0.262 | 47.6 |
| Qwen3-8B | no-offload | 0.696 | 110.9 |
| | native-offload-20k | 0.900 | 135.5 |
| | lmcache-local | 0.707 | 110.3 |
| | lmcache-valkey | 6.555 | 216.9 |
| Qwen3-14B | no-offload | 40.009 | 357.7 |
| | native-offload-20k | 40.555 | 359.4 |
| | lmcache-local | 41.670 | 364.2 |
| | lmcache-valkey | 31.525 | 367.4 |
| | fs-offload | **7.063** | **87.8** |
| | cpu+fs-offload-20k | **6.911** | **87.3** |
| Qwen3-32B-AWQ | no-offload | 0.115 | 18.3 |
| | native-offload-20k | 0.216 | 135.1 |
| | lmcache-local | 0.224 | 147.2 |
| | lmcache-valkey | 0.233 | 148.4 |

Qwen3-14B fs-offload and cpu+fs-offload-20k show TTFT of 6.9–7.1 s and ITL of 87 ms at rate=50, compared to 40.0 s and 357.7 ms for no-offload — an 82% reduction in TTFT and 76% reduction in ITL. This reflects the high external prefix cache hit rate eliminating prefill latency. Qwen3-8B lmcache-valkey TTFT (6.555 s) is 9.4× the no-offload value (0.696 s), indicating that Valkey round-trip overhead accumulates under concurrency for this model size. Qwen3-32B-AWQ native-offload TTFT at rate=1 increases from 0.115 s to 0.216 s (+88%), with ITL increasing from 18.3 ms to 135.1 ms (+638%), consistent with KV block transfer overhead at each decode step.

---

## LMCache Results

### Peak Throughput (gmu=0.9)

| Model | no-offload | lmcache-local | vs nooff | lmcache-valkey | vs nooff |
|-------|:----------:|:-------------:|:--------:|:--------------:|:--------:|
| Qwen3-0.6B | 807.5 | 665.6 | -17.6% | 789.3 | -2.2% |
| Qwen3-8B | 197.3 | 194.1 | -1.6% | 141.9 | -28.1% |
| Qwen3-14B | 55.5 | 54.4 | -1.9% | 56.5 | +1.9% |
| Qwen3-32B-AWQ | 50.1 | 18.1 | -63.8% | 19.2 | -61.7% |

### Peak Throughput (mempress)

| Model | no-offload | lmcache-local | vs nooff | lmcache-valkey | vs nooff |
|-------|:----------:|:-------------:|:--------:|:--------------:|:--------:|
| Qwen3-0.6B | 524.8 | 426.7 | -18.7% | 474.7 | -9.6% |
| Qwen3-8B | 104.5 | 104.5 | 0.0% | 107.7 | +3.1% |
| Qwen3-14B | 69.3 | 68.3 | -1.5% | 68.3 | -1.5% |

Qwen3-8B lmcache-local reaches 0.0% overhead and lmcache-valkey reaches +3.1% under memory pressure, compared to -1.6% and -28.1% at gmu=0.9. Under memory pressure, Valkey overhead for Qwen3-8B narrows substantially as the GPU KV-cache constraint becomes the binding factor. Qwen3-0.6B lmcache-local overhead increases under memory pressure (-18.7% vs -17.6% at gmu=0.9). Qwen3-14B lmcache overhead under memory pressure is -1.5% for both backends, within measurement variance.

### Latency

Qwen3-0.6B lmcache-local TTFT at rate=50 (0.376 s) is 52% above no-offload (0.248 s), while lmcache-valkey TTFT (0.262 s) is within 6% of no-offload. This pattern — lmcache-valkey matching no-offload latency while lmcache-local shows overhead — reverses the Qwen3-8B pattern where lmcache-valkey TTFT (6.555 s) exceeds lmcache-local (0.707 s) by 9.3×. The Qwen3-8B lmcache-valkey latency anomaly at rate=50 reflects Valkey round-trip latency accumulation relative to the model's token generation rate.

Qwen3-32B-AWQ lmcache-local and lmcache-valkey show near-identical latency at rate=1 (TTFT: 0.224/0.233 s; ITL: 147.2/148.4 ms), consistent with both backends encountering KV-cache capacity limits rather than backend-specific overhead.

### Version Comparison

lmcache-local and lmcache-valkey absolute throughput increased substantially for Qwen3-8B between versions (+71.6% and +23.1% respectively), tracking the no-offload baseline gain. The overhead of lmcache-local vs no-offload is -1.6% in v0.6.0 vs -0.9% in v0.5.1; lmcache-valkey overhead is -28.1% in v0.6.0 vs +0.9% in v0.5.1. The v0.6.0 lmcache-valkey regression for Qwen3-8B is associated with the elevated TTFT noted above; the underlying cause is under investigation.

For Qwen3-14B, both lmcache-local and lmcache-valkey show reduced absolute throughput in v0.6.0 vs v0.5.1 (-13.5% and -10.1%), tracking the vLLM 0.17.1 no-offload baseline decrease for this model.

---

## Version Progression

### No-Offload Baseline Across Versions

| Model | v0.4.0 | v0.5.1 | v0.6.0 | v0.5.1→v0.6.0 |
|-------|-------:|-------:|-------:|--------------:|
| Qwen3-0.6B | 602.0 | 636.8 | 807.5 | **+26.8%** |
| Qwen3-8B | 113.0 | 114.1 | 197.3 | **+72.9%** |
| Qwen3-14B | 58.7 | 58.7 | 55.5 | -5.5% |
| Qwen3-32B-AWQ | 49.2 | 51.2 | 50.1 | -2.1% |

vLLM 0.17.1 delivers large no-offload throughput gains for Qwen3-0.6B (+26.8%) and Qwen3-8B (+72.9%) vs vLLM 0.15.1. Qwen3-14B decreases by -5.5% (58.7 → 55.5 tok/s). Qwen3-32B-AWQ is within measurement variance (-2.1%).

### Native CPU Offload Across Versions

| Model | v0.5.1 nat-20k | v0.5.1 vs nooff | v0.6.0 nat-20k | v0.6.0 vs nooff | v0.5.1→v0.6.0 |
|-------|:--------------:|:---------------:|:--------------:|:---------------:|:-------------:|
| Qwen3-0.6B | 622.9 | -2.2% | 809.6 | +0.3% | +30.0% |
| Qwen3-8B | 80.0 | -29.9% | 184.5 | -6.5% | +130.7% |
| Qwen3-14B | 67.2 | +14.5% | 56.5 | +1.9% | -15.9% |
| Qwen3-32B-AWQ | 21.3 | -58.4% | 18.1 | -63.8% | -14.9% |

The Qwen3-8B native-offload overhead narrows from -29.9% to -6.5% in v0.6.0, a +23.4 pp improvement. Qwen3-0.6B native-offload moves from -2.2% to +0.3% (overhead eliminated). Qwen3-14B native-offload reverses from +14.5% to +1.9% (absolute throughput: 67.2 → 56.5 tok/s), tracking the no-offload baseline regression. Qwen3-32B-AWQ widens from -58.4% to -63.8%.

![Version Comparison](analysis/v0.6.0_version_comparison.png)
*Figure: Peak throughput (tok/s) for v0.5.1 vs v0.6.0 at gmu=0.9 — no-offload and native-offload-20k configurations. Per-bar annotations show absolute values; italic delta percentages show v0.5.1→v0.6.0 change.*

### LMCache Across Versions (v0.5.1 → v0.6.0)

LMCache version upgraded from v0.3.15 (v0.5.1) to v0.4.2 (v0.6.0). Both use the same local CPU and Valkey remote backends.

**lmcache-local gmu=0.9:**

| Model | v0.5.1 | v0.6.0 | Delta | v0.6.0 vs no-offload |
|-------|-------:|-------:|------:|:--------------------:|
| Qwen3-0.6B | 605.9 | 665.6 | +9.9% | -17.6% |
| Qwen3-8B | 113.1 | 194.1 | +71.6% | -1.6% |
| Qwen3-14B | 62.9 | 54.4 | -13.5% | -1.9% |
| Qwen3-32B-AWQ | 22.4 | 18.1 | -19.0% | -63.8% |

**lmcache-valkey gmu=0.9:**

| Model | v0.5.1 | v0.6.0 | Delta | v0.6.0 vs no-offload |
|-------|-------:|-------:|------:|:--------------------:|
| Qwen3-0.6B | 606.9 | 789.3 | +30.1% | -2.2% |
| Qwen3-8B | 115.2 | 141.9 | +23.1% | -28.1% |
| Qwen3-14B | 62.9 | 56.5 | -10.1% | +1.9% |
| Qwen3-32B-AWQ | 21.3 | 19.2 | -9.9% | -61.7% |

Qwen3-8B lmcache-local shows a +71.6% increase in absolute throughput (113.1 → 194.1 tok/s), tracking the no-offload baseline improvement while maintaining near-zero overhead (-1.6%). Qwen3-14B lmcache-local decreases by -13.5% absolute (62.9 → 54.4 tok/s), which is consistent with the underlying vLLM 0.17.1 no-offload baseline decrease for this model. Qwen3-8B lmcache-valkey shows elevated TTFT (6.555 s at rate=50 vs 0.696 s no-offload), indicating Valkey round-trip latency at this concurrency (see §LMCache Results — Latency).

![LMCache Version Comparison](analysis/v0.6.0_lmcache_version_comparison.png)
*Figure: LMCache peak throughput (tok/s) for v0.5.1 vs v0.6.0 at gmu=0.9.*

---

## Filesystem Offload Results

### Peak Throughput (gmu=0.9)

| Model | no-offload | fs-offload | vs nooff | cpu+fs-offload-20k | vs nooff |
|-------|:----------:|:----------:|:--------:|:-----------------:|:--------:|
| Qwen3-0.6B | 807.5 | 320.6 | -60.3% | 789.3 | -2.2% |
| Qwen3-8B | 197.3 | **199.5** | **+1.1%** | 197.3 | 0.0% |
| Qwen3-14B | 55.5 | **163.2** | **+194.2%** | **164.3** | **+196.2%** |
| Qwen3-32B-AWQ | 50.1 | 18.1 | -63.8% | 18.1 | -63.8% |

Qwen3-14B shows the largest throughput gain: fs-offload +194.2% and cpu+fs-offload-20k +196.2% vs the no-offload baseline. Qwen3-8B fs-offload (+1.1%) and cpu+fs-offload-20k (0.0%) are effectively neutral vs baseline. Qwen3-0.6B fs-offload shows -60.3%; cpu+fs-offload-20k recovers to -2.2% (matching lmcache-valkey). Qwen3-32B-AWQ shows -63.8% for both fs configs, consistent with other offload configurations at this model size.

### Cross-Version Comparison (v0.5.1 → v0.6.0)

| Model | Config | v0.5.1 | v0.5.1 vs nooff | v0.6.0 | v0.6.0 vs nooff | Delta |
|-------|--------|:------:|:---------------:|:------:|:---------------:|:-----:|
| Qwen3-0.6B | fs-offload | 842.7 | +32.3% | 320.6 | -60.3% | -62.0% |
| Qwen3-8B | fs-offload | 212.3 | +86.0% | 199.5 | +1.1% | -6.0% |
| Qwen3-14B | fs-offload | 183.5 | +212.7% | 163.2 | +194.2% | -11.1% |
| Qwen3-32B-AWQ | fs-offload | 22.4 | -56.2% | 18.1 | -63.8% | -19.0% |
| Qwen3-0.6B | cpu+fs-offload-20k | 854.4 | +34.2% | 789.3 | -2.2% | -7.6% |
| Qwen3-8B | cpu+fs-offload-20k | 188.8 | +65.4% | 197.3 | 0.0% | +4.5% |
| Qwen3-14B | cpu+fs-offload-20k | 169.6 | +189.1% | 164.3 | +196.2% | -3.1% |
| Qwen3-32B-AWQ | cpu+fs-offload-20k | 21.3 | -58.3% | 18.1 | -63.8% | -14.9% |

Qwen3-0.6B fs-offload shows -62.0% absolute decline from v0.5.1 to v0.6.0 (842.7 → 320.6 tok/s) while the no-offload baseline increased +26.8%. In v0.5.1, 0.6B fs-offload exceeded the no-offload baseline (+32.3%); in v0.6.0 it falls well below (-60.3%). cpu+fs-offload-20k partially recovers for 0.6B (-7.6% absolute, -2.2% vs no-offload), suggesting the CPU tier buffers some of the regression.

Qwen3-14B maintains high absolute throughput under fs-offload across versions (183.5 → 163.2 tok/s, -11.1%), with both configurations remaining above 160 tok/s vs the 55.5 tok/s no-offload baseline.

![Filesystem Offload Version Comparison](analysis/v0.6.0_fs_version_comparison.png)
*Figure: fs-offload and cpu+fs-offload-20k peak throughput (tok/s) for v0.5.1 vs v0.6.0 at gmu=0.9.*

### Latency (gmu=0.9, rate=50)

| Model | Config | TTFT (s) | ITL (ms) |
|-------|--------|:--------:|:--------:|
| Qwen3-0.6B | no-offload | 0.248 | 47.4 |
| | fs-offload | 0.322 | 47.3 |
| | cpu+fs-offload-20k | 0.260 | 46.6 |
| Qwen3-8B | no-offload | 0.696 | 110.9 |
| | fs-offload | 0.959 | 130.9 |
| | cpu+fs-offload-20k | 0.967 | 137.2 |
| Qwen3-14B | no-offload | 40.009 | 357.7 |
| | fs-offload | **7.063** | **87.8** |
| | cpu+fs-offload-20k | **6.911** | **87.3** |

Qwen3-14B fs-offload reduces TTFT from 40.0 s to 7.1 s (-82%) and ITL from 357.7 ms to 87.8 ms (-75%) at rate=50, driven by cache hits eliminating prefix recomputation. Qwen3-8B fs-offload shows a modest TTFT increase (0.696 → 0.959 s, +38%) with near-neutral throughput (+1.1%). Qwen3-0.6B latency is broadly comparable across configurations at rate=50.

---

## Observations

1. **vLLM 0.17.1 (v0.6.0) delivers 26.8% and 72.9% no-offload throughput gains for Qwen3-0.6B and Qwen3-8B respectively vs vLLM 0.15.1 (v0.5.1).** Qwen3-14B no-offload throughput decreases by 5.5% (58.7 → 55.5 tok/s). Qwen3-32B-AWQ no-offload is within measurement variance (-2.1%).

2. **Native-offload-20k overhead at gmu=0.9 narrows for Qwen3-0.6B (from -2.2% to +0.3%) and Qwen3-8B (from -29.9% to -6.5%).** The Qwen3-14B native-offload gain reverses from +14.5% to +1.9%, tracking the underlying no-offload regression. Qwen3-32B-AWQ native-offload widens from -58.4% to -63.8%.

3. **lmcache-local maintains near-zero overhead for Qwen3-8B (-1.6% vs -0.9% in v0.5.1) and Qwen3-14B (-1.9%).** Qwen3-0.6B lmcache-local overhead increases to -17.6% (from -4.9% in v0.5.1).

4. **lmcache-valkey shows elevated TTFT for Qwen3-8B at rate=50 (6.555 s vs 0.696 s no-offload), a 9.4× increase.** This latency pattern is not present for other model sizes and is associated with the -28.1% throughput overhead for this configuration.

5. **Under memory pressure, Qwen3-0.6B native-offload-20k reaches +51.4% vs no-offload baseline (794.7 tok/s), up from +22.3% in v0.5.1.** Memory-pressure native-offload overhead for Qwen3-8B increases from -3.6% in v0.5.1 to -16.3% in v0.6.0.

6. **Qwen3-14B mempress native-offload-20k is impacted by vLLM #38515**, with three of eight concurrency rates showing >10% error rates (rates 50, 100, 500). The clean-run peak (rate=300, 61.9 tok/s, -10.8% vs baseline) is below the v0.5.1 mempress value of 78.9 tok/s. The full impact cannot be determined from v0.6.0 data alone.

7. **Qwen3-32B-AWQ offload configurations under memory pressure were not run** due to confirmed vLLM #38515 crashes across all concurrency levels during initial testing. No-offload (50.1 tok/s) is included and stable.

8. **Qwen3-14B fs-offload delivers +194.2% throughput and reduces TTFT from 40.0 s to 7.1 s (-82%) at rate=50.** Both fs-offload and cpu+fs-offload-20k exceed 160 tok/s vs the 55.5 tok/s no-offload baseline. This is the largest throughput gain observed for any configuration across all v0.6.0 results.

9. **Qwen3-0.6B fs-offload regresses from v0.5.1 (+32.3% vs no-offload) to v0.6.0 (-60.3%).** cpu+fs-offload-20k recovers to -2.2% for 0.6B, matching lmcache-valkey. The underlying cause of the 0.6B fs-offload regression between versions is not yet determined.

10. **libstdc++ ABI issue resolved in v0.6.0:** `llmd_fs_connector` v0.18.0 is built with `-static-libstdc++` ([llm-d/llm-d-kv-cache#498](https://github.com/llm-d/llm-d-kv-cache/pull/498)), eliminating the GLIBCXX runtime dependency and the LD_PRELOAD workaround required in v0.5.1.

11. **EPP v0.7.1 removes the `--kv-cache-usage-percentage-metric` flag** present in v0.5.x. All v0.6.0 EPP deployments were updated accordingly; no functional impact on benchmark workloads was observed.

---

## Appendix: Methodology

**Data collection:** GuideLLM 0.6.0 drives inference requests against the llm-d gateway (Kubernetes Service with EPP scheduler). Each benchmark run is 120 seconds at a fixed concurrency level. Throughput is computed as `output_token_count.successful.total_sum / duration`. TTFT and ITL are median values from `time_to_first_token_ms.successful.median` and `inter_token_latency_ms.successful.median`.

**Peak throughput selection:** The peak is the maximum throughput value across all eight concurrency levels, excluding runs with error rate > 10% (vLLM #38515 crash indicator). Three runs are excluded on this basis (all Qwen3-14B mempress native-offload-20k).

**Error rate flag:** `errored / total > 0.10`, where `errored` and `total` come from `request_totals` in the GuideLLM JSON output. Flagged runs are reported but not used for peak selection.

**System monitoring:** Performance Co-Pilot (PCP) archives are recorded for each run, capturing GPU utilization (DCGM), vLLM internal metrics (via OpenMetrics PMDA), disk I/O, and CPU metrics. PCP analysis is not included in this report (the primary value is in KV cache hit rates and GPU saturation metrics, which are most informative for native-offload configurations where external cache counters are active).

**Historical baselines:** v0.5.1 peak throughput values are taken from `scripts/analyze-v0.5.1.py` (gmu=0.9) and `scripts/analyze-v0.5.1-lmcache.py` (mempress) output. v0.4.0 no-offload values are from REPORT-v0.4.0.md.

**Analysis script:** `scripts/analyze-v0.6.0.py`, outputs in `analysis/v0.6.0_*.{png,csv}`.

**Run count:** 297 total benchmark runs — 6 configs × 4 models × 8 rates (gmu=0.9) plus 4 configs × 3 models × 8 rates (mempress) plus Qwen3-32B-AWQ no-offload at both gmu levels.
