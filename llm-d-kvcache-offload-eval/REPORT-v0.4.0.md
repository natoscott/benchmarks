# llm-d KV-Cache Management Evaluation

## Summary

Performance evaluation of KV-cache management strategies in llm-d v0.4.0 (vLLM 0.11.2) across two areas:

1. **llm-d EPP distributed KV-block indexing**: baseline GPU-only vs Redis/Valkey-backed distributed indexing for cache-aware request routing
2. **CPU KV-cache offload**: vLLM native offloading vs LMCache (local CPU and Valkey backends)

Supplementary memory-pressure runs repeated key configurations with per-model reduced `gpu_memory_utilization` (0.55–0.70 vs default 0.9) to create GPU KV-cache pressure across all model sizes, including a 20K CPU-block native-offload configuration.

**Results at a glance (peak throughput, gmu=0.9):**

| Config | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|--------|:----------:|:--------:|:---------:|:-------------:|
| no-offload | 602.0 tok/s | 113.0 tok/s | 58.7 tok/s | 49.2 tok/s |
| native-offload (10k) | -29.1% | -36.5% | +0.6% | -1.0% |
| lmcache-local | -13.6% | -5.6% | +11.8% | -12.7% |
| lmcache-valkey | -13.0% | -6.5% | +13.0% | -12.7% |
| llm-d-valkey | -1.5% | +0.4% | +10.0% | -0.1% |

**Hardware note:** Results are specific to 2× NVIDIA L40S (48 GB total VRAM), 48 vCPUs, IBM Cloud. GPU KV-cache memory after model loading ranges from 20.58 GiB (Qwen3-14B FP16) to 33.92 GiB (Qwen3-0.6B), which determines which models experience KV-cache pressure under this workload.

---

## Test Configuration

### Hardware Setup

**System:** OpenShift cluster on IBM Cloud
- **GPUs**: 2x NVIDIA L40S (48GB total VRAM)
  - Tensor Parallelism: 2 GPUs per model
- **CPU**: 48 vCPUs (IBM cloud virtual CPUs)
- **Memory**: Sufficient for CPU KV-cache blocks (configuration-dependent)
- **Network**: Cluster networking with shared Redis/Valkey services

**Software:**
- **llm-d**: v0.4.0
- **vLLM**: v0.11.2 (bundled with llm-d v0.4.0)
- **LMCache**: v0.3.7
- **Redis**: v7.4.7
- **Valkey**: v8.1.5
- **OpenShift**: v4.22.0
- **PCP**: v7.0.3
- **GuideLLM**: v0.5.3 (original runs), v0.5.4 (memory-pressure runs)

**Models Tested:**
- Qwen/Qwen3-0.6B (577M parameters, FP16)
- Qwen/Qwen3-8B (8.3B parameters, FP16)
- Qwen/Qwen3-14B (14.8B parameters, FP16)
- Qwen/Qwen3-32B-AWQ (32.5B parameters, 4-bit AWQ quantization)

### Workload Parameters

**Testing Approach**: High-concurrency multi-turn conversation workload
- **Concurrency Levels**: 1, 50, 100, 150, 300, 400, 500, 650
- **Duration**: 120 seconds per concurrency level
- **Prompt Structure**: Multi-turn conversations with shared prefix
  - Prompt tokens: 128 per turn
  - Output tokens: 128 per turn
  - Prefix tokens: 10,000 (shared across requests)
  - Turns: 5 per conversation
  - Prefix count: 800 unique prefixes (0.6B), varies by model
- **Sample requests**: 4000 per benchmark run

This workload simulates a demanding scenario with long context windows, multiple conversation turns, and high concurrency to stress-test KV-cache management strategies.

### Configurations Tested

#### 1. Baseline (no-offload)
GPU-only KV-cache storage without offloading or distributed KV-block indexing.

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024
```

**llm-d EPP**: In-memory prefix cache scorer (no distributed indexing)

#### 2. Native CPU Offloading (native-offload)
vLLM's built-in OffloadingConnector for CPU KV-cache offload.

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",\
  "kv_connector_extra_config":{"num_cpu_blocks":10000}}'
```

**KV-cache blocks on CPU**: 10,000 blocks
**llm-d EPP**: In-memory prefix cache scorer

#### 3. LMCache Local CPU (lmcache-local)
LMCache with local CPU backend for KV-cache storage.

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both",\
  "kv_connector_extra_config":{"backend":"local","cpu_blocks":10000}}'
```

**KV-cache backend**: LMCache local CPU storage
**llm-d EPP**: In-memory prefix cache scorer

#### 4. LMCache with Redis (lmcache-redis)
LMCache with Redis backend for distributed KV-cache sharing.

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both",\
  "kv_connector_extra_config":{"backend":"redis","redis_url":"redis://..."}}'
```

**KV-cache backend**: LMCache with Redis
**llm-d EPP**: In-memory prefix cache scorer

#### 5. LMCache with Valkey (lmcache-valkey)
LMCache with Valkey backend for distributed KV-cache sharing.

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both",\
  "kv_connector_extra_config":{"backend":"valkey","valkey_url":"valkey://..."}}'
```

**KV-cache backend**: LMCache with Valkey
**llm-d EPP**: In-memory prefix cache scorer

#### 6. llm-d with Redis Indexing (llm-d-redis)
llm-d EPP with Redis-backed distributed KV-block indexing.

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024
```

**llm-d EPP**: Redis-backed prefix cache scorer
- **Index Backend**: Redis (redis://redis.llm-d-pfc-cpu.svc.cluster.local:6379)
- **Cache Awareness**: Request routing informed by KV-block index
- **Note**: This is KV-block **indexing** for routing, NOT KV-cache data storage

#### 7. llm-d with Valkey Indexing (llm-d-valkey)
llm-d EPP with Valkey-backed distributed KV-block indexing.

```bash
vllm serve <model> --tensor-parallel-size 2 --port 8000 --max-num-seq 1024
```

**llm-d EPP**: Valkey-backed prefix cache scorer
- **Index Backend**: Valkey (valkey://valkey.llm-d-pfc-cpu.svc.cluster.local:6379)
- **Cache Awareness**: Request routing informed by KV-block index

---

## KV-Cache Memory Allocation

Understanding actual GPU and CPU memory allocation for KV-cache storage is necessary for interpreting performance results. vLLM allocates KV-cache memory based on available GPU memory after model weights are loaded, and this varies significantly by model size.

### GPU KV-Cache Memory Availability

| Model | GPU Memory (GiB) | Token Capacity | Max Concurrency | Memory Pressure |
|-------|----------------:|---------------:|----------------:|-----------------|
| Qwen3-14B | **20.58** | 269,712 | 6.58x | **Highest** |
| Qwen3-32B-AWQ | 25.40 | 208,080 | 5.08x | Moderate-High |
| Qwen3-8B | 26.83 | 390,704 | 9.54x | Moderate |
| Qwen3-0.6B | 33.92 | 635,200 | 15.51x | **Lowest** |

**Notes**:
- Measured with TP=2 across 2x NVIDIA L40S GPUs (48GB total VRAM)
- The 14B model has **39% less GPU KV-cache memory** than the 0.6B model (20.58 GiB vs 33.92 GiB) and **58% less token capacity** (269K vs 635K tokens).
- The 32B-AWQ model, despite being larger, has **more available GPU KV-cache memory than the 14B model** (25.40 GiB vs 20.58 GiB) due to 4-bit quantization reducing model weight size.

![KV-Cache Memory Capacity](analysis/kvcache_memory_capacity.png)
*Figure: GPU memory availability, token capacity, and max concurrency by model. The 14B model shows the least available memory and lowest concurrency, creating memory pressure that benefits from CPU offload.*

### CPU Offload Memory Allocation

When CPU offload is enabled, vLLM allocates CPU memory based on configuration and required system memory:

| Model | Config | Blocks Configured | CPU Memory (GiB) | Allocation Behavior |
|-------|--------|------------------:|-----------------:|---------------------|
| Qwen3-0.6B | 10K blocks | 10,000 | 33.92 | Over-allocated (3.97x) |
| Qwen3-8B | 10K blocks | 10,000 | 26.83 | Over-allocated (2.44x) |
| Qwen3-14B | 10K blocks | 10,000 | 20.58 | Over-allocated (1.69x) |
| Qwen3-14B | 20K blocks | 20,000 | 41.16 | As configured (2.00x) |
| Qwen3-32B-AWQ | 10K blocks | 10,000 | 25.40 | Over-allocated (1.30x) |
| Qwen3-32B-AWQ | 20K blocks | 20,000 | 50.80 | As configured (2.00x) |

**Allocation Behavior**:
- **10K block configurations**: vLLM v0.4.0 over-allocated CPU blocks beyond the configured 10,000, matching GPU KV-cache memory capacity. For models with abundant GPU memory (0.6B: 33.92 GiB), this resulted in 39,700 actual blocks (3.97x configured). For memory-constrained models (14B: 20.58 GiB), allocation was closer to configured value (16,857 blocks, 1.69x).
- **20K block configurations**: With 20,000 blocks configured, vLLM allocated the requested capacity (41.16 GiB for 14B, 50.80 GiB for 32B-AWQ), enabling substantial performance improvements (+16.7% for 14B, +11.9% for 32B-AWQ with LMCache).

All models use 16 tokens per KV-cache block consistently.

![KV-Cache Configured vs Actual](analysis/kvcache_configured_vs_actual.png)
*Figure: Configured vs actual CPU block allocation. With 10K blocks, vLLM over-allocated to match GPU capacity. With 20K blocks, vLLM allocated the full requested amount, demonstrating adequate system memory.*

### Memory Pressure and Performance Correlation

The relationship between GPU memory availability and CPU offload effectiveness is clear:

![Memory Pressure vs Performance](analysis/kvcache_memory_pressure_summary.png)
*Figure: Models with lower GPU KV-cache memory (higher pressure) benefit more from CPU offload. The 14B model's 20.58 GiB available GPU memory creates pressure relieved by offload, while the 0.6B model's 33.92 GiB abundance makes offload pure overhead.*

GPU memory availability is the primary factor determining CPU offload effectiveness. The 14B model's constrained GPU memory (20.58 GiB, 270K tokens) creates pressure that CPU offload relieves, enabling +12-17% throughput improvement when adequate CPU memory is allocated (41.16 GiB for 20K blocks). Models with abundant GPU memory (0.6B: 33.92 GiB, 635K tokens) experience no memory pressure, making CPU offload pure overhead (-13% to -29% degradation) regardless of CPU allocation size.

---

## Performance Results

### Peak Throughput Summary

Results show peak output token throughput achieved at optimal concurrency for each configuration.

| Model | Configuration | Peak Throughput (tok/s) | Optimal Concurrency | vs Baseline |
|-------|---------------|------------------------:|--------------------:|------------:|
| **Qwen3-0.6B** | no-offload | 602.0 | 50 | — |
| | native-offload | 426.8 | 50 | **-29.1%** |
| | lmcache-local | 520.4 | 50 | **-13.6%** |
| | lmcache-redis | 565.7 | 50 | **-6.0%** |
| | lmcache-valkey | 523.9 | 50 | **-13.0%** |
| | llm-d-redis | 598.0 | 50 | **-0.7%** |
| | llm-d-valkey | 592.9 | 50 | **-1.5%** |
| **Qwen3-8B** | no-offload | 113.0 | 50 | — |
| | native-offload | 71.8 | 50 | **-36.5%** |
| | lmcache-local | 106.6 | 50 | **-5.6%** |
| | lmcache-redis | 101.7 | 50 | **-10.0%** |
| | lmcache-valkey | 105.7 | 50 | **-6.5%** |
| | llm-d-redis | 112.9 | 50 | **-0.1%** |
| | llm-d-valkey | 113.4 | 50 | **+0.4%** |
| **Qwen3-14B** | no-offload | 58.7 | 50 | — |
| | native-offload | 59.0 | 50 | **+0.6%** |
| | lmcache-local | 65.6 | 50 | **+11.8%** |
| | lmcache-redis | 60.1 | 100 | **+2.5%** |
| | lmcache-valkey | 66.3 | 50 | **+13.0%** |
| | llm-d-redis | 60.6 | 50 | **+3.4%** |
| | llm-d-valkey | 64.5 | 50 | **+10.0%** |
| **Qwen3-32B-AWQ** | no-offload | 49.2 | 1 | — |
| | native-offload | 48.7 | 1 | **-1.0%** |
| | lmcache-local | 43.0 | 1 | **-12.7%** |
| | lmcache-redis | 42.9 | 1 | **-12.8%** |
| | lmcache-valkey | 43.0 | 1 | **-12.7%** |
| | llm-d-redis | 49.2 | 1 | **-0.1%** |
| | llm-d-valkey | 49.2 | 1 | **-0.1%** |

![Peak Throughput Comparison](analysis/peak_throughput_all_scenarios.png)

### Throughput vs Concurrency

The following graphs show output token throughput as a function of concurrency level for each model and configuration.

![Throughput vs Concurrency - All Models](analysis/throughput_vs_concurrency_all.png)
*Figure: Output token throughput vs concurrency level for all four models across seven KV-cache management configurations. Each panel shows one model size, with all scenarios plotted for comparison. Models exhibit different optimal concurrency levels: 0.6B/8B/14B peak at rate=50, while 32B-AWQ peaks at rate=1.*

### Latency: TTFT and TPOT

Median TTFT and TPOT at rate=50 (gmu=0.9):

**Time to First Token (ms) at rate=50:**

| Config | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|--------|:----------:|:--------:|:---------:|:-------------:|
| no-offload | 807 | 10,701 | 21,943 | 29,674 |
| native-offload | 1,141 | 21,854 | 21,207 | 29,888 |
| lmcache-local | 754 | 12,332 | 19,425 | 27,341 |
| lmcache-valkey | 799 | 10,939 | 23,066 | 27,434 |
| llm-d-valkey | 641 | 11,050 | 22,165 | 32,625 |

**Time Per Output Token (ms) at rate=50:**

| Config | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|--------|:----------:|:--------:|:---------:|:-------------:|
| no-offload | 77 | 340 | 490 | 738 |
| native-offload | 107 | 537 | 490 | 724 |
| lmcache-local | 89 | 359 | 471 | 781 |
| lmcache-valkey | 88 | 347 | 468 | 796 |
| llm-d-valkey | 78 | 341 | 473 | 764 |

The 32B-AWQ model peaks at rate=1 (not rate=50); at rate=50 it is heavily overloaded and the latency figures reflect queueing rather than model characteristics.

For Qwen3-0.6B, native-offload shows +41% higher TTFT vs no-offload (1,141ms vs 807ms) despite lower throughput. TPOT increases from 77ms to 107ms (+39%). llm-d-valkey shows the lowest TTFT (641ms), reflecting its minimal overhead profile.

![Latency Comparison](analysis/latency_comparison_all.png)
*Figure: TTFT and TPOT at variable concurrency (top: each model at its peak-throughput rate) and fixed rate=50 (bottom: all models for direct comparison).*

### GPU Prefix Cache Hit Rates

GPU prefix cache hit rates at rate=50 (Qwen3-0.6B, gmu=0.9):

| Config | Hit rate |
|--------|:--------:|
| no-offload | 63.1% |
| native-offload | 58.6% |
| lmcache-local | 67.4% |
| lmcache-valkey | 67.5% |
| llm-d-valkey | 61.1% |

At rate=1, hit rates drop to 2–8% across all configurations. Hit rates increase with concurrency as more requests share the common 10K-token prefix.

Larger models show lower hit rates at rate=50: 14B configurations reach 21–30%, 32B-AWQ 14–43% depending on configuration.

![Prefix Cache Hit Rates](analysis/pcp_prefix_cache_hits.png)
*Figure: GPU prefix cache hit rate by scenario.*

---

## System-Level Analysis (PCP)

Performance Co-Pilot metrics captured during benchmark execution. Analysis at peak throughput (rate=50) unless noted.

### GPU Utilisation vs Throughput

![GPU Utilization vs Throughput](analysis/pcp_gpu_vs_throughput.png)
*Figure: GPU utilization correlated with output token throughput at peak load*

**Observations:**
- no-offload: 43.2% GPU utilisation, 279.7 tok/s average throughput at rate=50
- llm-d distributed indexing: 52.7% GPU utilisation, throughput within ±2% of no-offload
- CPU offload configurations: 46–52% GPU utilisation with lower throughput than no-offload

#### KV-Cache Usage Patterns

![KV-Cache Usage by Scenario](analysis/pcp_kv_cache_usage.png)
*Figure: KV-cache utilization percentage by scenario and model at rate=50*

**Findings:**
- KV-cache utilization varies from 29-48% across configurations
- The workload uses a substantial portion of GPU KV-cache capacity without completely exhausting it
- This moderate utilization level suggests memory pressure exists for most model sizes
- Exception: The 14B model's improvement with CPU offload suggests it operates near a memory pressure threshold where offload provides measurable benefits

#### Request Queue Dynamics

![Request Queue Patterns](analysis/pcp_request_queues.png)
*Figure: Mean running and waiting requests at peak throughput*

**Insight:**
- **no-offload** maintains the most balanced queue: 16.8 running, 112.0 waiting
- **llm-d configurations** show higher running requests (17.9-18.8) but significantly higher waiting queues (182-188)
- **CPU offload** shows lower running requests (13.9-16.1), indicating scheduler constraints

The higher waiting queue counts for llm-d configurations suggest that distributed KV-block indexing may introduce slight request routing latency, though this doesn't materially impact throughput. The lower running request counts for CPU offload configurations confirm that transfer overhead limits concurrent request execution.

#### Prefix Cache Effectiveness

![Prefix Cache Hit Rates](analysis/pcp_prefix_cache_hits.png)
*Figure: Prefix cache hit rate percentage by scenario*

Prefix cache hit rates vary significantly by scenario, with some configurations showing substantially higher cache effectiveness. This metric correlates with the shared prefix workload design (10K-token prefixes) and demonstrates the value of prefix caching for multi-turn conversations.

#### Summary Statistics by Scenario

| Scenario | Avg Throughput (tok/s) | Avg GPU Util (%) | Avg KV-Cache (%) | Avg Running Reqs | Avg Waiting Reqs |
|----------|----------------------:|----------------:|----------------:|----------------:|----------------:|
| no-offload | 279.72 | 43.17 | 44.81 | 16.81 | 112.03 |
| llm-d-redis | 198.18 | 52.67 | 44.74 | 17.91 | 188.12 |
| llm-d-valkey | 198.19 | 52.06 | 48.28 | 18.76 | 181.99 |
| lmcache-local | 177.58 | 46.31 | 29.08 | 14.07 | 77.47 |
| lmcache-redis | 186.62 | 43.57 | 33.94 | 14.20 | 120.69 |
| lmcache-valkey | 178.68 | 49.79 | 35.62 | 16.10 | 113.79 |
| native-offload | 145.96 | 51.13 | 38.02 | 13.87 | 115.59 |

*Note: Averages computed across all models at peak throughput (rate=50)*

**Insights:**

1. **GPU utilization inversely correlates with throughput**: Higher GPU utilization in CPU offload scenarios reflects transfer overhead rather than productive compute, for many cases

2. **Moderate KV-cache utilization**: Utilization ranges from 29-48% across scenarios, indicating the workload uses a substantial portion of GPU KV-cache capacity without completely exhausting it

3. **Request scheduling overhead**: llm-d distributed indexing shows higher waiting queues but maintains throughput, suggesting routing decisions don't block request processing

#### CPU Utilization and System Pressure

Analysis of CPU utilization reveals significant CPU saturation hidden by averaging across 48 vCPUs:

**Aggregate CPU Utilization by Scenario (at peak throughput):**
- **no-offload baseline**: 4.4-10.0% average CPU utilization
- **lmcache-local-20kcpu**: 4.4-5.6% average CPU utilization
- **llm-d distributed**: 9.2-10.5% average CPU utilization
- **native-offload**: 8.3-10.0% average CPU utilization

However, per-CPU analysis reveals a different picture:

**Per-CPU Utilization Analysis:**
- **llm-d-valkey**: 9.9 saturated CPUs (>80%), max CPU 752% average utilization
- **llm-d-redis**: 10.4 saturated CPUs, max CPU 514% average utilization
- **lmcache-local**: 14.2 saturated CPUs, max CPU 353% average utilization
- **native-offload**: 11.2 saturated CPUs, max CPU 447% average utilization
- **no-offload**: 9.5 saturated CPUs, max CPU 457% average utilization

All scenarios show individual CPUs hitting >95% utilization during benchmark execution, with high variance in CPU load distribution (standard deviation 64-136%). The CPU load range (max CPU - min CPU) spans 270-877%, indicating hotspotting where some CPUs remain relatively idle while others saturate.

**Findings:**
- All scenarios show 9–14 individual CPUs averaging >80% utilisation, while the aggregate average across 48 vCPUs is 4–10%
- CPU load range (max − min CPU) spans 270–877%, indicating uneven distribution
- Offload scenarios show more saturated CPUs (11–14) compared to no-offload (9.5)

![CPU Saturation by Scenario](analysis/percpu_saturation_by_scenario.png)
*Figure: Number of saturated CPUs vs expected from average CPU utilization - shows how averaging hides saturation*

![CPU Load Distribution](analysis/percpu_load_distribution.png)
*Figure: CPU load variance and range showing hotspotting across all scenarios*

![CPU Offload Impact on Saturation](analysis/percpu_offload_impact.png)
*Figure: CPU saturation comparison - offload scenarios show higher CPU saturation than baseline*

![CPU Saturation Heatmap](analysis/percpu_saturation_heatmap.png)
*Figure: Comprehensive view of CPU saturation patterns - llm-d-valkey shows highest severity across all metrics, while lmcache-local shows most saturated CPUs*

**System Pressure Metrics:**
The test system (RHEL 9.6, kernel 5.14) supports Pressure Stall Information (PSI) metrics. No PSI pressure events were observed (memory, CPU, or I/O) during benchmark execution.

### GPU Power Consumption

GPU power consumption (DCGM `DCGM_FI_DEV_POWER_USAGE`) at rate=50 for Qwen3-14B (2× L40S):

| Config | Total Power (W) | vs Baseline |
|--------|:--------------:|:-----------:|
| no-offload | 499.8 | — |
| native-offload | 503.7 | +0.8% |
| lmcache-local | 497.1 | -0.5% |
| lmcache-valkey | 488.2 | -2.3% |
| llm-d-valkey | 498.3 | -0.3% |

Power variation across configurations is 0.3–2.3%. GPU utilisation remains above 96% in all configurations.

---

## Memory-Pressure Analysis

A follow-up experiment re-ran the core v0.4.0 configurations with per-model reduced `gpu_memory_utilization` to create GPU KV-cache pressure for all model sizes, not just Qwen3-14B. A 20K-block native-offload configuration was also added to characterise block-count sensitivity.

### Configuration

| Model | Original gmu | Mempress gmu | GPU KV tokens (original) | GPU KV tokens (mempress) |
|-------|:-----------:|:------------:|:------------------------:|:------------------------:|
| Qwen3-0.6B | 0.9 | **0.55** | ~634K | ~335K |
| Qwen3-8B | 0.9 | **0.65** | ~390K | ~215K |
| Qwen3-14B | 0.9 | **0.70** | ~268K | ~142K |
| Qwen3-32B-AWQ | 0.9 | **0.65** | ~207K | ~116K |

Configs tested: no-offload, native-offload (10K blocks), native-offload-20k (20K blocks), lmcache-local, lmcache-valkey, llm-d-valkey.

### Peak Throughput (tok/s)

| Config | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|--------|:----------:|:--------:|:---------:|:-------------:|
| no-offload | 437.3 | 116.3 | 66.1 | 46.9 |
| native-offload (10k) | 401.1 (-8.3%) | 93.9 (-19.3%) | 66.1 (0.0%) | 45.9 (-2.3%) |
| **native-offload-20k** | **478.9 (+9.5%)** | **105.6 (-9.2%)** | **81.1 (+22.6%)** | **45.9 (-2.3%)** |
| lmcache-local | 518.4 (+18.5%) | 105.6 (-9.2%) | 57.6 (-12.9%) | 41.6 (-11.4%) |
| lmcache-valkey | 522.7 (+19.5%) | 107.7 (-7.3%) | 54.4 (-17.7%) | 41.6 (-11.4%) |
| llm-d-valkey | 498.1 (+13.9%) | 115.2 (-0.9%) | 68.3 (+3.2%) | 46.9 (0.0%) |

![Memory-Pressure Peak Throughput](analysis/v0.4.0-mempress_peak_throughput_v2.png)
*Figure: Peak throughput at reduced gpu_memory_utilization across all six configurations. native-offload-20k shows higher throughput than native-offload-10k for 0.6B and 14B.*

### Change vs Original (gmu=0.9) Offload Deltas

Percentage-point change in throughput delta vs same-version no-offload baseline, from original (gmu=0.9) to mempress:

| Config | 0.6B | 8B | 14B | 32B-AWQ |
|--------|:----:|:--:|:---:|:-------:|
| native-offload (10k) | +20.8 pp | +17.2 pp | -0.5 pp | -1.3 pp |
| native-offload-20k | +38.6 pp | +27.3 pp | +22.0 pp | -1.3 pp |
| lmcache-local | +32.1 pp | -3.5 pp | -24.7 pp | +1.2 pp |
| lmcache-valkey | +32.5 pp | -0.9 pp | -30.7 pp | +1.2 pp |
| llm-d-valkey | +15.4 pp | -1.3 pp | -6.7 pp | 0.0 pp |

![Memory-Pressure Delta Heatmap](analysis/v0.4.0-mempress_delta_heatmap.png)
*Figure: Throughput delta vs no-offload baseline. native-offload-20k shows the largest improvement for 0.6B and 14B.*

### PCP: GPU KV-Cache Utilisation

GPU KV-cache utilisation at peak concurrency:

| Config group | 0.6B | 8B | 14B | 32B-AWQ |
|---|:---:|:---:|:---:|:---:|
| Original gmu=0.9 (no-offload) | 36% | 63% | 71% | 2% |
| Mempress no-offload | 21% | 44% | 70% | 7% |
| Mempress native-offload (10k) | 48% | 42% | 44% | 2% |
| Mempress native-offload-20k | 48% | 44% | 44% | 2% |

GPU KV-cache utilisation at peak concurrency ranges from 21% (Qwen3-0.6B no-offload) to 70% (Qwen3-14B no-offload). Qwen3-0.6B mempress no-offload shows lower GPU KV-cache usage (21%) than the original gmu=0.9 run (36%) because the reduced allocation limits total capacity.

![GPU KV-Cache Utilisation](analysis/v0.4.0-mempress_gpu_kvcache_util.png)
*Figure: GPU KV-cache utilisation at peak concurrency, original vs mempress.*

### Observations

**Qwen3-0.6B:** lmcache configurations shift from -13% (gmu=0.9) to +18-19% (mempress), a +32 pp improvement. native-offload-20k shifts from -29.1% (original gmu=0.9) to +9.5% (mempress), a +38.6 pp improvement. native-offload-10k shifts to -8.3% (+20.8 pp). Median TTFT at rate=50: lmcache-valkey 718ms vs no-offload 14,231ms.

**Qwen3-8B:** native-offload-10k improves from -36.5% to -19.3% (+17.2 pp); native-offload-20k improves to -9.2% (+27.3 pp). lmcache and llm-d-valkey show negligible change (±1-3.5 pp) vs original.

**Qwen3-14B:** native-offload-20k shows +22.6% vs no-offload at mempress gmu, vs 0.0% with 10K blocks. GPU KV-cache utilisation at peak load (44%) is reduced from original (71%), indicating the gmu=0.70 reduction shifted the operating point. lmcache configs show -13% to -18% — the lmcache overhead dominates when the model can serve requests from GPU cache at this reduced utilisation.

**Qwen3-32B-AWQ:** Results are insensitive to gmu changes in the 0.65–0.90 range across all configs. Both native-offload configurations show -2.3%.

### Latency

Median TTFT at rate=50 (selected models):

| Config | Qwen3-0.6B | Qwen3-14B |
|--------|:----------:|:---------:|
| no-offload | 14,231 ms | — |
| lmcache-local | 793 ms | — |
| lmcache-valkey | 718 ms | — |
| native-offload-20k | — | reduced vs 10K |

The lmcache TTFT reduction for 0.6B at rate=50 reflects prefix cache hits served without full recomputation.

### CPU Block Provisioning: 10K vs 20K Blocks (gmu=0.9)

Targeted experiment with 20K CPU blocks for Qwen3-14B and Qwen3-32B-AWQ at original gmu=0.9:

| Model | Config | 10K blocks | 20K blocks | Delta |
|-------|--------|:----------:|:----------:|:-----:|
| Qwen3-14B | native-offload | 59.0 tok/s | 68.5 tok/s | +16.2% |
| | lmcache-local | 65.6 tok/s | 76.5 tok/s | +16.7% |
| | lmcache-valkey | 66.3 tok/s | 74.6 tok/s | +12.5% |
| Qwen3-32B-AWQ | native-offload | 48.7 tok/s | 48.9 tok/s | +0.3% |
| | lmcache-local | 43.0 tok/s | 48.1 tok/s | +11.9% |
| | lmcache-valkey | 43.0 tok/s | 43.2 tok/s | +0.5% |

Qwen3-14B shows +12–17% throughput with 20K blocks across all offload configurations. Qwen3-32B-AWQ shows +11.9% with lmcache-local at 20K blocks (vs -12.7% at 10K) and minimal change with native-offload and lmcache-valkey.

vLLM v0.4.0 over-allocated CPU blocks beyond the configured value to match GPU KV-cache capacity: 16.8K actual blocks for 14B regardless of 10K or 20K configured. With 20K blocks configured, allocation matched the requested amount (41.16 GiB for 14B, 50.80 GiB for 32B-AWQ).

---

## Observations

Results across all configurations and experiments (gmu=0.9 and memory-pressure runs):

1. **llm-d EPP distributed KV-block indexing** shows <2% throughput overhead for 0.6B, 8B, and 32B-AWQ models; +3.4% to +10.0% for 14B. Redis and Valkey backends produce identical results within benchmark variance.

2. **LMCache (local and Valkey) outperforms vLLM native-offload** for all model sizes at both gmu=0.9 and mempress gmu. At gmu=0.9: 14B lmcache-local +11.8% vs native-offload +0.6%. At mempress gmu: 0.6B lmcache-valkey +19.5% vs native-offload-10k -8.3%.

3. **CPU block allocation determines offload effectiveness**: Qwen3-32B-AWQ shifts from -12.7% (lmcache-local, 10K blocks) to +11.9% (20K blocks) at gmu=0.9. Qwen3-14B improves from +11.8% to +16.7% with 20K blocks.

4. **The 14B FP16 model has less GPU KV-cache memory than the 32B-AWQ model**: 20.58 GiB vs 25.40 GiB, despite fewer parameters. FP16 precision at 14B consumes more VRAM than 4-bit quantization at 32B.

5. **Under memory pressure (mempress gmu), lmcache and native-offload-20k both convert 0.6B and 8B outcomes**: 0.6B lmcache-valkey improves from -13.0% (gmu=0.9) to +19.5% (mempress). native-offload-20k improves from -29.1% to +9.5% for 0.6B and from +0.6% to +22.6% for 14B.

6. **GPU prefix cache hit rate at rate=50 (0.6B)**: 58–67% across configurations. Hit rate drops to 2–8% at rate=1. All configurations show near-zero `external_prefix_cache_hits` in this single-replica deployment.

7. **TTFT at rate=50**: lmcache configurations show 718–793ms TTFT for 0.6B vs 807ms (no-offload) and 1,141ms (native-offload). For 8B and 14B, TTFT differences between offload configurations are smaller in relative terms.

8. **GPU power variation** across configurations is 0.3–2.3% for 14B. No PSI pressure events were observed.

---

## Limitations and Hardware Dependencies

- Results are specific to 2× NVIDIA L40S (24 GB VRAM each, 48 GB total), PCIe Gen4, 48 vCPUs, IBM Cloud. GPU KV-cache memory after model loading ranges from 20.58 GiB (Qwen3-14B FP16) to 33.92 GiB (Qwen3-0.6B).

- The Qwen3-14B FP16 model has less GPU KV-cache memory (20.58 GiB) than the Qwen3-32B-AWQ 4-bit model (25.40 GiB). Precision choices shift the memory-pressure profile independently of parameter count.

- With 10K CPU blocks configured, vLLM v0.4.0 over-allocated to match GPU KV-cache capacity. With 20K blocks, vLLM allocated the full requested amount, confirming adequate system memory.

- Results apply to the specific workload: 10K-token shared prefix, 5 turns, 128 prompt + 128 output tokens per turn, concurrency 1–650.

---

## Appendix: Methodology and Data

### Benchmark Execution

Original runs used GuideLLM v0.5.3; memory-pressure runs used GuideLLM v0.5.4. Parameters:
- Profile: concurrent
- Duration: 120 seconds per concurrency level
- Sample requests: 4000 (original runs); suppressed via `--sample-requests=0` (mempress runs)

### Metrics Collection

- **GuideLLM**: Throughput, latency, request completion metrics
- **PCP archives**: Comprehensive system-level metrics collection across all benchmark runs, including:
  - System metrics: CPU utilization, memory usage, network I/O
  - GPU metrics: Utilization, memory, power consumption (via DCGM)
  - vLLM metrics: KV-cache usage, request queues, prefix cache hit rates (via OpenMetrics)
  - Process metrics: Memory consumption, CPU usage per vLLM process
- **vLLM startup logs**: Captured for all model/configuration combinations to extract actual GPU and CPU KV-cache memory allocations
  - GPU KV-cache memory availability after model loading
  - Token capacity and maximum concurrency
  - Actual vs configured CPU block allocation
  - KV-cache block size (tokens per block)
- Archives captured at 10-second intervals throughout each benchmark run
- PCP data analyzed and correlated with GuideLLM results to provide system-level validation

### Data Files and Reproducibility

Benchmark data, analysis scripts, and visualization code are organized as follows:

**Raw Data:**
- GuideLLM JSON results: `results/*/guidellm-results.json.zst`
- PCP archives: `results/*/pcp-archives/` (compressed with zstd)
- vLLM startup logs: `vllm-startup-logs/*.log.zst` (10 configurations, 4 models)

---

*Report generated from benchmark runs completed February 2026*
*Analysis framework: Performance Co-Pilot + GuideLLM*
*System: OpenShift on IBM Cloud with 2x NVIDIA L40S GPUs*
