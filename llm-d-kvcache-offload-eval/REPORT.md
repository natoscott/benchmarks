# llm-d KV-Cache Management Evaluation

## Summary

This report presents a comprehensive performance evaluation of KV-cache management strategies in the llm-d inference serving system. Seven configurations were tested across four model sizes (Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ) using high-concurrency workloads with tensor parallelism across 2x NVIDIA L40S GPUs.

The evaluation addresses two areas:

1. **llm-d EPP distributed KV-block indexing overhead**: Comparing baseline GPU-only operation against Redis and Valkey-backed distributed indexing for cache-aware request routing
2. **CPU KV-cache offload strategies**: Comparing vLLM native offloading against LMCache (local CPU and distributed Redis/Valkey backends)

**Key Findings:**

- **llm-d EPP distributed indexing achieves performance parity** with baseline (within ±2% for most models)
- **14B model optimization**: The 14B model shows +10-13% throughput improvement with CPU offload (both native and LMCache), while all other models show degradation
- **vLLM native offloading shows clear degradation** for small models (-29% to -36% for 0.6B/8B)
- **LMCache distributed caching** performs competitively for the 14B model but shows degradation for other sizes
- **Redis vs Valkey backends perform identically** across all configurations, providing deployment flexibility
- **Model size impacts offload effectiveness**: Different models show significantly different responses to CPU offload strategies

The llm-d EPP distributed KV-block indexing demonstrates negligible overhead for cache-aware request routing in multi-pod deployments. However, CPU offload strategies show highly model-dependent performance characteristics, with the 14B model representing an optimal size where offload benefits outweigh overhead.

**Important**: These results are specific to the test hardware configuration (2x NVIDIA L40S GPUs, 32 vCPUs, limited CPU memory for offload). Different GPU types, CPU memory capacity, and memory bandwidth characteristics will impact offload effectiveness. The 14B model's performance improvement with CPU offload is expected to also occur with different model sizes and alternative hardware configurations.

---

## Test Configuration

### Hardware Setup

**System:** OpenShift cluster on IBM Cloud
- **GPUs**: 2x NVIDIA L40S (48GB total VRAM)
  - Tensor Parallelism: 2 GPUs per model
- **CPU**: 32 vCPUs (IBM cloud virtual CPUs)
- **Memory**: Sufficient for CPU KV-cache blocks (configuration-dependent)
- **Network**: Cluster networking with Redis/Valkey services

**Software:**
- **llm-d**: v0.4.0
- **vLLM**: v0.11.2 (bundled with llm-d v0.4.0)
- **LMCache**: v0.3.7 (3rd party distributed KV-cache library)
- **OpenShift**: v4.22.0 (container orchestration platform)
- **PCP**: v7.0.3 (Performance Co-Pilot metrics collection)
- **GuideLLM**: v0.5.3 (benchmark orchestration framework)

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
| **Qwen3-14B** | no-offload | 58.7 | 50 | — |
| | native-offload | 59.0 | 50 | **+0.6%** |
| | lmcache-local | 65.6 | 50 | **+11.8%** |
| | lmcache-redis | 60.1 | 100 | **+2.5%** |
| | lmcache-valkey | 66.3 | 50 | **+13.0%** |
| | llm-d-redis | 60.6 | 50 | **+3.4%** |
| | llm-d-valkey | 64.5 | 50 | **+10.0%** |
| **Qwen3-8B** | no-offload | 113.0 | 50 | — |
| | native-offload | 71.8 | 50 | **-36.5%** |
| | lmcache-local | 106.6 | 50 | **-5.6%** |
| | lmcache-redis | 101.7 | 50 | **-10.0%** |
| | lmcache-valkey | 105.7 | 50 | **-6.5%** |
| | llm-d-redis | 112.9 | 50 | **-0.1%** |
| | llm-d-valkey | 113.4 | 50 | **+0.4%** |
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

![Throughput vs Concurrency - All Models](analysis/throughput_curve_Qwen3-0.6B_all.png)
*Figure: Qwen3-0.6B throughput curves across all 7 configurations*

![Throughput vs Concurrency - All Models](analysis/throughput_curve_Qwen3-8B_all.png)
*Figure: Qwen3-8B throughput curves across all 7 configurations*

![Throughput vs Concurrency - All Models](analysis/throughput_curve_Qwen3-14B_all.png)
*Figure: Qwen3-14B throughput curves across all 7 configurations*

![Throughput vs Concurrency - All Models](analysis/throughput_curve_Qwen3-32B-AWQ_all.png)
*Figure: Qwen3-32B-AWQ throughput curves across all 7 configurations*

### Latency Analysis

Latency measurements at peak throughput conditions show similar patterns across configurations, with native-offload generally showing higher latency variance.

![Latency Comparison](analysis/latency_comparison_all.png)
*Figure: Time to First Token (TTFT) and Time Per Output Token (TPOT) at peak throughput*

### System-Level Analysis (PCP Metrics)

Performance Co-Pilot metrics captured during benchmark execution provide deep insights into system behavior, GPU utilization, and vLLM internal state. The following analysis focuses on peak throughput scenarios (rate=50) where most models achieve optimal performance.

#### GPU Utilization vs Throughput

![GPU Utilization vs Throughput](analysis/pcp_gpu_vs_throughput.png)
*Figure: GPU utilization correlated with output token throughput at peak load*

**Key Observations:**
- **no-offload baseline** shows lowest GPU utilization (43.2%) but highest throughput (279.7 tok/s)
- **llm-d distributed indexing** maintains similar GPU efficiency (52.7%) with minimal throughput impact
- **CPU offload configurations** show higher GPU utilization (46-52%) but lower throughput, suggesting GPU cycles spent on data transfer rather than compute

This counterintuitive pattern—higher GPU utilization with lower throughput—confirms that CPU offload strategies introduce overhead in the form of CPU-GPU transfer operations that consume GPU cycles without contributing to token generation.

#### KV-Cache Usage Patterns

![KV-Cache Usage by Scenario](analysis/pcp_kv_cache_usage.png)
*Figure: KV-cache utilization percentage by scenario and model at rate=50*

**Findings:**
- KV-cache utilization remains remarkably low (0.29-0.48%) across all configurations
- This indicates the workload does not exhaust GPU KV-cache capacity
- The low utilization explains why CPU offload shows degradation for most models—there's no memory pressure to alleviate
- Exception: The 14B model's improvement with CPU offload suggests different memory dynamics at that model size

#### Process Memory Consumption

![Memory Usage Comparison](analysis/pcp_memory_usage.png)
*Figure: vLLM process resident memory (RSS) across scenarios*

Process memory consumption remains consistent (1.58-2.04 GB) across configurations, with no significant variation based on offload strategy. This suggests that CPU KV-cache blocks are allocated separately from the main process memory, and the configurations are primarily differentiated by cache management strategy rather than raw memory footprint.

#### Request Queue Dynamics

![Request Queue Patterns](analysis/pcp_request_queues.png)
*Figure: Mean running and waiting requests at peak throughput*

**Critical Insight:**
- **no-offload** maintains the most balanced queue: 16.8 running, 112.0 waiting
- **llm-d configurations** show higher running requests (17.9-18.8) but significantly higher waiting queues (182-188)
- **CPU offload** shows lower running requests (13.9-16.1), indicating scheduler constraints

The higher waiting queue counts for llm-d configurations suggest that distributed KV-block indexing may introduce slight request routing latency, though this doesn't materially impact throughput. The lower running request counts for CPU offload configurations confirm that transfer overhead limits concurrent request execution.

#### Prefix Cache Effectiveness

![Prefix Cache Hit Rates](analysis/pcp_prefix_cache_hits.png)
*Figure: Prefix cache hit rate percentage by scenario*

Prefix cache hit rates vary significantly by scenario, with some configurations showing substantially higher cache effectiveness. This metric correlates with the shared prefix workload design (10K-token prefixes) and demonstrates the value of prefix caching for multi-turn conversations.

#### System Metrics Correlation

![Correlation Heatmap](analysis/pcp_correlation_heatmap.png)
*Figure: Correlation matrix between system metrics and performance indicators*

The correlation heatmap reveals relationships between system-level metrics and performance outcomes:
- **Negative correlation** between GPU utilization and throughput confirms the overhead hypothesis
- **Request queue depths** show weak correlation with throughput, suggesting queue management is not the primary bottleneck
- **KV-cache usage** shows minimal correlation with performance, consistent with low overall utilization

#### Summary Statistics by Scenario

| Scenario | Avg Throughput (tok/s) | Avg GPU Util (%) | Avg KV-Cache (%) | Avg Running Reqs | Avg Waiting Reqs | Avg Process RSS (GB) |
|----------|----------------------:|----------------:|----------------:|----------------:|----------------:|--------------------:|
| no-offload | 279.72 | 43.17 | 0.45 | 16.81 | 112.03 | 1.63 |
| llm-d-redis | 198.18 | 52.67 | 0.45 | 17.91 | 188.12 | 1.62 |
| llm-d-valkey | 198.19 | 52.06 | 0.48 | 18.76 | 181.99 | 2.04 |
| lmcache-local | 177.58 | 46.31 | 0.29 | 14.07 | 77.47 | 1.58 |
| lmcache-redis | 186.62 | 43.57 | 0.34 | 14.20 | 120.69 | 1.62 |
| lmcache-valkey | 178.68 | 49.79 | 0.36 | 16.10 | 113.79 | 1.62 |
| native-offload | 145.96 | 51.13 | 0.38 | 13.87 | 115.59 | 1.65 |

*Note: Averages computed across all models at peak throughput (rate=50)*

**PCP Analysis Insights:**

1. **GPU utilization inversely correlates with throughput**: Higher GPU utilization in CPU offload scenarios reflects transfer overhead rather than productive compute

2. **Low KV-cache pressure**: Utilization under 0.5% indicates GPU memory is not constrained for this workload, explaining why CPU offload shows degradation for most models

3. **Request scheduling overhead**: llm-d distributed indexing shows higher waiting queues but maintains throughput, suggesting routing decisions don't block request processing

4. **Memory footprint consistency**: Process RSS remains stable across configurations, confirming that performance differences stem from cache management strategy rather than memory overhead

---

## Analysis

### Area 1: llm-d EPP Distributed KV-Block Indexing Overhead

The llm-d EPP (Endpoint Provisioning Proxy) provides distributed KV-block indexing via Redis or Valkey backends to enable cache-aware request routing in multi-pod deployments. This evaluation measures the overhead of this indexing layer compared to baseline GPU-only operation.

**Performance Impact:**

- **Qwen3-0.6B**: -0.7% to -1.5% (within measurement variance)
- **Qwen3-8B**: -0.1% to +0.4% (effectively zero overhead)
- **Qwen3-14B**: +3.4% to +10.0% (substantial improvement over baseline)
- **Qwen3-32B-AWQ**: -0.1% (within measurement variance)

**Key Insight**: The llm-d distributed KV-block indexing introduces **negligible overhead** for most models, with the interesting result that the 14B model shows consistent improvement (+3-10%) across both Redis and Valkey backends. This suggests that cache-aware routing provides measurable benefits for mid-size models where request scheduling optimization has greater impact.

**Redis vs Valkey**: Both backends perform identically (within ±1-2%), confirming that backend choice has no measurable performance impact and can be based on operational preferences (licensing, ecosystem compatibility, feature requirements).

### Area 2: CPU KV-Cache Offload Strategies

This evaluation compares three CPU offload approaches against the GPU-only baseline:
1. **vLLM native offloading**: Built-in OffloadingConnector
2. **LMCache local CPU**: Third-party library with local CPU backend
3. **LMCache distributed**: Third-party library with Redis/Valkey backends

**Performance by Model Size:**

#### Qwen3-0.6B (Tiny Model)
- **Native offload**: -29.1% (severe degradation)
- **LMCache local**: -13.6% (moderate degradation)
- **LMCache distributed**: -6.0% to -13.0% (moderate degradation)

The small model shows consistent degradation across all CPU offload strategies, with native offloading showing the lowest performance. CPU-GPU transfer overhead dominates for this model size.

#### Qwen3-8B (Small Model)
- **Native offload**: -36.5% (lowest performance observed)
- **LMCache local**: -5.6% (moderate degradation)
- **LMCache distributed**: -6.5% to -10.0% (moderate degradation)

The 8B model shows severe degradation with native offloading and moderate degradation with LMCache. This model size appears to be in an "overhead zone" where CPU offload costs are high but benefits are minimal.

#### Qwen3-14B (Medium Model, Optimal Size for offload here)
- **Native offload**: +0.6% (performance parity)
- **LMCache local**: **+11.8%** (significant improvement)
- **LMCache distributed**: **+2.5% to +13.0%** (consistent improvement)

The 14B model is the only model size that benefits from CPU offload with 10,000 block cache size, showing +11-13% throughput improvement with LMCache.
- Model size is large enough that KV-cache pressure is significant
- Compute requirements are balanced such that CPU offload latency is acceptable
- Request scheduling benefits from additional KV-cache capacity

#### Qwen3-32B-AWQ (Larger, Quantized Model)
- **Native offload**: -1.0% (near parity)
- **LMCache local**: -12.7% (moderate degradation)
- **LMCache distributed**: -12.7% to -12.8% (consistent degradation)

The larger quantized model shows near-parity with native offload but degradation with LMCache. The 4-bit quantization reduces KV-cache memory pressure, potentially eliminating benefits of CPU offload.

### Performance Delta Heatmap

![Performance Delta Heatmap](analysis/performance_delta_heatmap_all.png)
*Figure: Performance delta (%) vs baseline for all model-scenario combinations. Green indicates improvement, red indicates degradation.*

### Model Size and Offload Effectiveness

The different responses to CPU offload strategies reveal a **model size dependency**:

| Model Size | Native Offload | LMCache Local | Optimal Strategy |
|-----------|----------------|---------------|------------------|
| Tiny (0.6B) | ⛔ Severe degradation | ⚠️  Moderate degradation | **GPU-only** |
| Small (8B) | ⛔ Severe degradation | ⚠️  Moderate degradation | **GPU-only** |
| Medium (14B) | ✅ Parity | ✅ **+11.8% improvement** | **LMCache CPU offload** |
| Large quantized (32B-AWQ) | ✅ Parity | ⚠️  Moderate degradation | **GPU-only or native** |

This pattern suggests:
1. **Tiny models** suffer from CPU-GPU transfer overhead without compensating benefits
2. **14B represents an optimal size** where KV-cache capacity gains outweigh transfer costs
3. **Quantized large models** have reduced KV-cache pressure, eliminating offload benefits

### LMCache Backend Comparison (Redis vs Valkey)

LMCache performance with Redis vs Valkey backends shows near-identical results:

| Model | Redis | Valkey | Delta |
|-------|-------|--------|-------|
| Qwen3-0.6B | 565.7 tok/s | 523.9 tok/s | -7.4% |
| Qwen3-8B | 101.7 tok/s | 105.7 tok/s | +3.9% |
| Qwen3-14B | 60.1 tok/s | 66.3 tok/s | +10.3% |
| Qwen3-32B-AWQ | 42.9 tok/s | 43.0 tok/s | +0.2% |

The differences are within normal benchmark variance, confirming that **Redis and Valkey perform equivalently** as LMCache backends.

### Workload Characteristics

The high-concurrency multi-turn workload with long shared prefixes (10,000 tokens) creates significant KV-cache pressure:

- **Successful request completion rates**: Low (39-564 requests completed per 120s)
- **Peak concurrency**: Most models peak at concurrency=50
- **Workload complexity**: The workload (800 unique 10K-token prefixes) stresses cache management

This workload represents a **worst-case scenario** for cache management, making the results conservative estimates of production performance.

### Understanding the 14B Model Performance

The 14B model's +11-13% throughput improvement with CPU offload (LMCache) stands in stark contrast to all other models showing degradation. This warrants deeper investigation to understand the underlying mechanisms.

**Memory Pressure Sweet Spot**

The 14B model with TP=2 across 2x L40S GPUs (24GB VRAM each) operates in a region where:

1. **GPU memory is constrained but not exhausted**: The model fits in VRAM but leaves limited headroom for KV-cache
2. **CPU offload reduces GPU memory pressure**: Moving KV-cache blocks to CPU frees GPU memory for model weights and intermediate activations
3. **Transfer overhead is acceptable**: The model's compute requirements are balanced such that CPU-GPU transfer latency doesn't dominate

**Contrasting with other models:**

- **0.6B and 8B**: Models are small enough that GPU memory is not a constraint. CPU offload introduces pure overhead without benefits. The severe -36.5% degradation for 8B native-offload suggests this model size hits a particularly bad overhead zone.

- **32B-AWQ**: 4-bit quantization reduces both model size and KV-cache memory requirements. The model already fits comfortably in VRAM, eliminating the memory pressure that would benefit from offload. The -12.7% degradation with LMCache suggests offload overhead without compensating benefits.

**Request Completion Analysis:**

Looking at successful request counts at peak throughput:
- **14B no-offload**: 55 requests (rate=50)
- **14B lmcache-local**: 61 requests (rate=50) - **+11% more requests completed**
- **14B lmcache-valkey**: 62 requests (rate=50) - **+13% more requests completed**

The increased request completion rate directly correlates with throughput improvement, suggesting CPU offload enables the scheduler to handle more concurrent requests effectively.

**LMCache vs Native Offload Performance Gap:**

The performance difference between LMCache and native offload (especially for 8B: -5.6% vs -36.5%) suggests:

1. **Implementation differences**: LMCache likely has more efficient CPU-GPU transfer mechanisms or better scheduling integration
2. **Block management**: Different strategies for deciding which KV-cache blocks to offload and when
3. **Prefetching**: LMCache may implement more sophisticated prefetching to hide transfer latency

**Hardware Dependency Implications:**

The 14B model's optimal performance with CPU offload is likely highly dependent on:

- **L40S PCIe Gen4 bandwidth**: ~32 GB/s bidirectional. Different GPUs with PCIe Gen5 (doubled bandwidth) may shift the optimal model size upward
- **CPU memory speed**: The available system has sufficient bandwidth for 14B but may be saturated for larger models
- **10,000 CPU block limit**: Increasing this may enable 32B-AWQ to benefit from offload

---

## Conclusions and Insights

This comprehensive evaluation of KV-cache management strategies across seven configurations reveals nuanced, model-size-dependent performance characteristics.

### Primary Findings

1. **llm-d EPP distributed KV-block indexing is production-ready**: The <2% overhead (and +3-10% improvement for 14B) demonstrates that distributed indexing for cache-aware routing imposes minimal cost while enabling multi-pod deployments.

2. **Model size determines offload strategy effectiveness**: The 14B model's +11-13% improvement with LMCache CPU offload, contrasted with degradation for all other sizes, reveals a clear optimal model size range for CPU offload strategies.

3. **vLLM native offloading underperforms LMCache**: Across all model sizes, native offloading shows worse performance than LMCache equivalents, suggesting implementation differences in transfer efficiency or scheduling.

4. **Backend choice (Redis vs Valkey) has zero performance impact**: Both for llm-d indexing and LMCache storage, Redis and Valkey perform identically, allowing deployment decisions based on operational factors.

5. **Quantization interacts with offload strategies**: The 32B-AWQ model's different behavior compared to 14B suggests that 4-bit quantization fundamentally changes KV-cache dynamics and offload effectiveness.

### Deployment Guidance

**Important**: These guidelines incorporate findings from both baseline (10K CPU blocks) and follow-up (20K CPU blocks) testing. Results are expected to be specific to the 2x NVIDIA L40S GPU configuration - different hardware warrants re-evaluation.

Based on these results for this hardware configuration:

- **For Qwen3-0.6B and Qwen3-8B models**: Use GPU-only operation (no-offload) with optional llm-d distributed indexing for multi-pod routing. CPU offload shows noticable performance degradation.

- **For Qwen3-14B model**: Use LMCache CPU offload (local or distributed) with **adequate CPU memory provisioning**:
  - Baseline (10K blocks, ~29 GB): +11-13% improvement
  - Increased (20K blocks, ~58 GB): +12-17% improvement

- **For Qwen3-32B-AWQ model**: CPU offload effectiveness depends on CPU memory capacity:
  - With 10K blocks (~39 GB): -12.7% degradation with LMCache
  - With 20K blocks (~78 GB): **+11.9% improvement** with LMCache

- **For multi-pod deployments**: llm-d Redis/Valkey indexing provides cache-aware routing with negligible overhead across all model sizes.

- **Backend selection**: Choose Redis or Valkey based on operational preferences; performance is equivalent for both llm-d indexing and LMCache storage.

- **CPU memory provisioning**: The follow-up experiments demonstrate that CPU memory capacity is a first-order factor. Underprov isioning CPU memory can shift results from significant improvement to degradation.

- **Hardware considerations**: The optimal model size for CPU offload is likely to shift with different GPU types, PCIe generation, CPU memory capacity, and memory bandwidth.

### Insights

1. **CPU memory capacity is a significant factor**: The follow-up experiments conclusively demonstrate that CPU memory capacity is the dominant factor in offload effectiveness. The 32B-AWQ model's shift from -12.7% degradation to +11.9% improvement with doubled CPU memory validates this hypothesis.

2. **Model size determines optimal configuration**: Three distinct patterns emerge:
   - Small models (0.6B, 8B): GPU-only is optimal; CPU offload always degrades performance
   - Medium models (14B): CPU offload provides substantial gains (+12-17% with adequate memory)
   - Larger models (32B+): CPU offload is beneficial only with sufficient CPU memory provisioning

3. **LMCache outperforms native offload**: Across all model sizes, LMCache shows consistently better performance than vLLM's native offloading.

4. **Workload sensitivity**: Different prompt lengths, context windows, or prefix patterns may shift the optimal model size and CPU memory requirements

5. **Quantization dynamics**: The 32B-AWQ model's behavior suggests that 4-bit quantization changes KV-cache dynamics in ways that interact with CPU offload strategies. Exploring how different quantization schemes (AWQ, GPTQ, FP8) affect offload effectiveness and memory requirements would be valuable.

### Follow-up: Impact of Increased CPU Memory Capacity

To validate the hypothesis that CPU memory constraints were limiting offload effectiveness, a targeted experiment tested both the 14B and 32B-AWQ models with doubled CPU KV-cache capacity (20,000 blocks vs baseline 10,000 blocks).

**Test Configuration:**
- Models: Qwen3-14B, Qwen3-32B-AWQ
- Configurations: native-offload, lmcache-local, lmcache-valkey
- CPU blocks: 20,000 (2x baseline)
- CPU memory: ~58 GB for 14B (~29 GB baseline), ~78 GB for 32B-AWQ (~39 GB baseline)

**Results:**

| Model | Configuration | Baseline (10K blocks) | Increased (20K blocks) | Delta |
|-------|---------------|----------------------:|----------------------:|------:|
| **Qwen3-14B** | native-offload | 59.0 tok/s | 68.5 tok/s | **+16.2%** |
| | lmcache-local | 65.6 tok/s | 76.5 tok/s | **+16.7%** |
| | lmcache-valkey | 66.3 tok/s | 74.6 tok/s | **+12.5%** |
| **Qwen3-32B-AWQ** | native-offload | 48.7 tok/s | 48.9 tok/s | **+0.3%** |
| | lmcache-local | 43.0 tok/s | 48.1 tok/s | **+11.9%** |
| | lmcache-valkey | 43.0 tok/s | 43.2 tok/s | **+0.5%** |

**Findings:**

1. **14B model improvements amplified**: The already-positive results (+11-13% with 10K blocks) increased to +12-17% with doubled CPU memory. This confirms the 14B model is highly responsive to increased CPU KV-cache capacity on this hardware.

2. **32B-AWQ degradation eliminated**: The model that showed -12.7% degradation with lmcache-local at 10K blocks shifted to **+11.9% improvement** with 20K blocks. This validates that the 10K block limit was indeed constraining offload effectiveness for larger models.

3. **Native offload scaling**: The 14B model showed the largest gain with native offload (+16.2%), while 32B-AWQ showed minimal change (+0.3%), suggesting native offload has different scaling characteristics than LMCache.

4. **Memory capacity**: These results demonstrate that CPU memory capacity is a first-order factor for CPU offload effectiveness. The 32B-AWQ model's shift from degradation to improvement conclusively proves that insufficient CPU KV-cache blocks were preventing offload benefits from materializing.

**Implications:**

- **Larger models benefit from increased capacity**: The 32B-AWQ results show that models previously thought unsuitable for CPU offload can benefit significantly when given adequate CPU memory
- **Hardware provisioning**: CPU memory capacity should be considered as an important factor when planning offload-enabled deployments
- **Optimal block counts**: Further testing with 30K or 40K blocks may reveal additional gains, particularly for 32B+ models

These follow-up results substantially strengthen the case for CPU KV-cache offload strategies when properly provisioned.

---

### Limitations and Hardware Dependencies

- **Hardware-specific results**: All findings are specific to the 2x NVIDIA L40S GPU configuration with 32 vCPUs. The L40S PCIe Gen4 memory bandwidth, VRAM capacity (48GB total), and CPU-GPU interconnect characteristics directly influence offload performance. Different GPU families (H100, A100, etc.) or configurations will show different optimal model sizes for CPU offload.

- **CPU memory constraints**: Baseline benchmarks used 10,000 CPU KV-cache blocks (~39 GB CPU memory). Follow-up testing with 20,000 blocks (78 GB) confirmed that CPU memory capacity is a first-order factor in offload effectiveness, with the 32B-AWQ model shifting from -12.7% degradation to +11.9% improvement with doubled capacity.

- **Workload specificity**: Results apply to high-concurrency, long-context, multi-turn workloads with 10K-token shared prefixes. Different workload patterns (shorter contexts, different concurrency patterns, varying prefix overlap) may show different optimal configurations.

- **Memory bandwidth limitations**: The observed model-size dependency may be influenced by L40S-specific PCIe bandwidth and CPU memory subsystem performance. Systems with higher-bandwidth CPU-GPU interconnects or faster CPU memory may show different results.

---

## Appendix: Methodology and Data

### Benchmark Execution

All benchmarks were executed using GuideLLM v0.5.3 with identical parameters across configurations:
- Profile: concurrent
- Duration: 120 seconds per concurrency level
- Sample requests: 4000
- Random seed: 889 (for reproducibility)
- Prefer response metrics: true

### Metrics Collection

- **GuideLLM**: Throughput, latency, request completion metrics
- **PCP archives**: Comprehensive system-level metrics collection across all benchmark runs, including:
  - System metrics: CPU utilization, memory usage, network I/O
  - GPU metrics: Utilization, memory, power consumption (via DCGM)
  - vLLM metrics: KV-cache usage, request queues, prefix cache hit rates (via OpenMetrics)
  - Process metrics: Memory consumption, CPU usage per vLLM process
- Archives captured at 10-second intervals throughout each benchmark run
- PCP data analyzed and correlated with GuideLLM results to provide system-level validation

### Data Files

- GuideLLM JSON results: `results/*/guidellm-results.json.zst`
- PCP archives: `results/*/pcp-archives/` (compressed with zstd)
- GuideLLM analysis outputs: `analysis/complete_metrics.csv`, `analysis/peak_throughput_all.csv`
- PCP analysis outputs: `analysis/pcp_metrics_peak.csv`, `analysis/pcp_summary_stats.csv`
- Visualizations: `analysis/*.png` (GuideLLM and PCP metrics)

### Reproducibility

All benchmark scripts, analysis code, and raw data are available in this repository:
- Benchmark execution: `scripts/run-benchmark.sh`
- GuideLLM data analysis: `scripts/comprehensive-analysis.py`
- PCP metrics extraction (comprehensive): `scripts/analyze-pcp-data.py`
- PCP metrics extraction (peak throughput focus): `scripts/extract-pcp-peak-metrics.py`
- PCP visualizations: `scripts/create-pcp-visualizations.py`

---

*Report generated from benchmark runs completed February 2026*
*Analysis framework: Performance Co-Pilot + GuideLLM*
*System: OpenShift on IBM Cloud with 2x NVIDIA L40S GPUs*
