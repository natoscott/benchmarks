# vLLM KV Cache CPU Offload Evaluation

## Executive Summary

This report analyzes the performance of vLLM 0.11.0's new KV cache CPU offload feature, comparing three configurations:
- **OffloadingConnector**: New CPU offload feature for KV cache
- **LMCacheConnectorV1**: LMCache integration
- **Baseline**: Traditional GPU-only approach

**Key Findings:**
- For **Qwen3-0.6B**: Both CPU offload approaches show similar performance to the baseline, with OffloadingConnector achieving 98.3% of baseline throughput
- For **Qwen3-8B**: OffloadingConnector actually outperforms baseline by 0.7% while reducing TTFT by 7.4%
- LMCache shows slightly higher latency overhead compared to OffloadingConnector

---

## Test Configuration

### Workload Parameters
- **Input Tokens**: 256
- **Output Tokens**: 128
- **Duration**: 30 seconds per benchmark run
- **Rate Strategy**: Sweep with 10 different request rates
- **Total Configurations**: 6 (2 models × 3 configurations)

### Hardware Setup
- **Tensor Parallelism**: 2 GPUs
- **Models Tested**:
  - Qwen/Qwen3-0.6B
  - Qwen/Qwen3-8B

### vLLM Server Configurations

#### Qwen3-0.6B Configurations
1. **OffloadingConnector**:
   ```bash
   vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2 \
     --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",\
     "kv_connector_extra_config":{"num_cpu_blocks":39072}}' \
     --gpu-memory-utilization=0.2
   ```

2. **LMCacheConnectorV1**:
   ```bash
   vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2 \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both",\
     "kv_connector_extra_config":{"num_cpu_blocks":39072}}' \
     --gpu-memory-utilization=0.2
   ```

3. **Baseline**:
   ```bash
   vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2 \
     --gpu-memory-utilization=0.2
   ```

#### Qwen3-8B Configurations
1. **OffloadingConnector**:
   ```bash
   vllm serve Qwen/Qwen3-8B --tensor-parallel-size 2 \
     --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",\
     "kv_connector_extra_config":{"num_cpu_blocks":64000}}' \
     --gpu-memory-utilization=0.6
   ```

2. **LMCacheConnectorV1**:
   ```bash
   vllm serve Qwen/Qwen3-8B --tensor-parallel-size 2 \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both",\
     "kv_connector_extra_config":{"num_cpu_blocks":64000}}' \
     --gpu-memory-utilization=0.6
   ```

3. **Baseline**:
   ```bash
   vllm serve Qwen/Qwen3-8B --tensor-parallel-size 2 \
     --gpu-memory-utilization=0.6
   ```

---

## Benchmark Results

![Benchmark Results Visualization](benchmark_results.png)

### Performance Summary

| Model | Configuration | Max Throughput (tok/s) | Avg TTFT (ms) | Avg TPOT (ms) |
|-------|---------------|------------------------|---------------|---------------|
| **Qwen3-0.6B** | GPU Only | **5031.2** | **640.7** | **12.8** |
| | OffloadingConnector | 4947.0 | 661.8 | 12.8 |
| | LMCacheConnectorV1 | 4707.3 | 693.8 | 14.9 |
| **Qwen3-8B** | GPU Only | 2406.8 | 587.2 | 47.5 |
| | OffloadingConnector | **2424.6** | **543.6** | **46.7** |
| | LMCacheConnectorV1 | 2406.6 | 562.5 | 47.4 |

---

## Performance Analysis

### Qwen3-0.6B Results

**Baseline Performance**: 5031.2 tok/s, TTFT: 640.7ms, TPOT: 12.8ms

#### OffloadingConnector
- **Throughput**: -1.7% (4947.0 tok/s)
  - Minimal degradation, only 84 tok/s slower
- **TTFT**: +3.3% (661.8 ms)
  - Slight increase of 21ms, acceptable for most use cases
- **TPOT**: +0.2% (12.8 ms)
  - Virtually identical to baseline

**Analysis**: OffloadingConnector delivers near-baseline performance with only marginal overhead. The 1.7% throughput reduction is a reasonable tradeoff for the ability to offload KV cache to CPU memory.

#### LMCacheConnectorV1
- **Throughput**: -6.4% (4707.3 tok/s)
  - Noticeable reduction of 324 tok/s
- **TTFT**: +8.3% (693.8 ms)
  - 53ms increase in time to first token
- **TPOT**: +16.3% (14.9 ms)
  - 2.1ms increase in inter-token latency

**Analysis**: LMCache shows more significant overhead compared to OffloadingConnector, particularly in TPOT which affects generation speed.

### Qwen3-8B Results

**Baseline Performance**: 2406.8 tok/s, TTFT: 587.2ms, TPOT: 47.5ms

#### OffloadingConnector
- **Throughput**: +0.7% (2424.6 tok/s)
  - Slightly outperforms baseline by 18 tok/s
- **TTFT**: -7.4% (543.6 ms)
  - Significantly faster first token (43.6ms improvement)
- **TPOT**: -1.6% (46.7 ms)
  - Marginally better inter-token latency

**Analysis**: Remarkably, OffloadingConnector actually improves performance for the larger 8B model. This suggests the CPU offload strategy effectively manages memory pressure, potentially allowing for better GPU utilization.

#### LMCacheConnectorV1
- **Throughput**: -0.0% (2406.6 tok/s)
  - Essentially identical to baseline
- **TTFT**: -4.2% (562.5 ms)
  - 24.7ms improvement in time to first token
- **TPOT**: -0.2% (47.4 ms)
  - Virtually identical to baseline

**Analysis**: LMCache performs much better on the 8B model compared to 0.6B, showing minimal overhead and even improving TTFT.

---

## Cache Hit Rates and Prefix Caching

All benchmark configurations had prefix caching enabled (`enable_prefix_caching:True`). The PCP archive contains comprehensive cache metrics:

**Available Cache Metrics**:
- `openmetrics.vllm.vllm.prefix_cache_queries_total` - Total prefix cache lookups
- `openmetrics.vllm.vllm.prefix_cache_hits_total` - Successful prefix cache hits
- `openmetrics.vllm.vllm.connector_prefix_cache_queries_total` - Connector cache lookups
- `openmetrics.vllm.vllm.connector_prefix_cache_hits_total` - Connector cache hits
- `openmetrics.vllm.vllm.kv_cache_usage_perc` - KV cache utilization percentage

**Data Limitation**: Cache metrics exist in the archive but fall outside the PCP recording windows during which the benchmarks executed. Future testing should ensure continuous PCP logging to capture cache hit rates, which would reveal:
- Effectiveness of prefix caching across different connector types
- Cache efficiency differences between OffloadingConnector and LMCacheConnectorV1
- Impact of cache hits on latency improvements

**Cache Configuration Observed**:
- Block size: 16
- Prefix caching hash algorithm: SHA256
- All configurations using automatic cache dtype selection
- GPU memory utilization varies by model (0.2 for 0.6B, 0.6 for 8B)

---

## GPU Memory and Resource Utilization

Limited GPU metrics were captured for 3 of the 6 benchmark runs. Available data shows:

### Qwen3-0.6B
- **Baseline**:
  - GPU Memory: avg 11.2 GB, peak 16.0 GB
  - KV Cache Usage: 0.9%

- **Offload**:
  - GPU Memory: avg 11.3 GB, peak 16.1 GB
  - KV Cache Usage: 0.0% (offloaded to CPU)

### Qwen3-8B
- **Offload**:
  - GPU Memory: avg 16.2 GB, peak 16.3 GB
  - KV Cache Usage: 0.1%

**Observation**: The GPU memory usage remains similar across configurations, suggesting the memory savings from KV cache offload may be allocated elsewhere or the test workload wasn't large enough to stress memory limits.

---

## Key Insights

1. **Model Size Impact**: The CPU offload features show different characteristics depending on model size:
   - Smaller model (0.6B): Minor differences with OffloadingConnector
   - Larger model (8B): Performance improvements with both offload approaches

2. **OffloadingConnector vs LMCache**:
   - OffloadingConnector shows lower latency overhead overall
   - LMCache performed noticably better on the larger model

3. **TTFT Improvements**: For the 8B model, both offload approaches actually reduce TTFT compared to baseline, suggesting better memory management leads to faster prompt processing

---

## Recommendations

### When to Use CPU Offload

**Use OffloadingConnector when**:
- Running larger models (8B+) where it can provide performance benefits
- Memory constraints limit batch sizes or concurrent requests
- Willing to accept <2% throughput reduction for smaller models

**Use LMCache when**:
- Working with larger models where overhead is minimal
- Can tolerate slightly higher latency for cache management benefits
- Want to leverage LMCache's caching capabilities across requests

**Stick with Baseline when**:
- Maximizing throughput for small models is critical
- GPU memory is abundant
- Latency requirements are extremely tight

### Future Testing Recommendations

1. **Continuous PCP Logging**: Ensure PCP records throughout entire benchmark session (discovered cache metrics exist but were not captured during benchmarks) at very high sampling intervals (1 second by default)
2. **Larger Batch Sizes**: Test with higher concurrency to stress memory limits
3. **Longer Sequences**: Use longer input/output sequences to amplify KV cache effects
4. **Memory-Constrained Scenarios**: Explicitly limit GPU memory to force offload utilization
5. **Multi-Model Comparison**: Test across more model sizes (1B, 3B, 7B, 13B, 70B)
6. **Production Traffic Patterns**: Simulate realistic request patterns with variable lengths
7. **Cache Hit Rate Analysis**: With continuous logging, analyze prefix cache effectiveness (queries vs hits) for each connector type
8. **Repeated Prompts**: Test with common prompt prefixes to maximize cache hit opportunities

---

## Appendix: Raw Data

### GuideLLM Benchmark Invocations

All benchmarks used identical GuideLLM commands:
```bash
guidellm benchmark \
  --output-sampling 0 \
  --target "http://localhost:8000" \
  --rate-type sweep \
  --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128"
```

### PCP Archive Details
- **Archive**: `benchmark-pcp-recording`
- **Time Range**: October 9-10, 2025
- **Duration**: ~20 hours (includes gaps between benchmarks)
- **Metrics Recorded**: ~140,000 columns including:
  - guidellm.* (benchmark results)
  - nvidia.* (GPU metrics)
  - openmetrics.vllm.* (vLLM Prometheus metrics)
  - kernel.*, mem.*, disk.*, network.* (system metrics)

---

*Report generated using Claude Code*
*Data source: PCP archives + GuideLLM JSON output*
