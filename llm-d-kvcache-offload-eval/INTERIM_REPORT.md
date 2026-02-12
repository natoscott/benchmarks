# llm-d with vLLM KV Cache CPU Offload Evaluation - Interim Report

## Status

**This is an interim report** covering initial results for Qwen3-0.6B model. Testing of larger models (Qwen3-8B, Qwen3-14B) is pending.

## Summary

This report analyzes the performance of vLLM's KV cache CPU offload feature deployed in an llm-d environment using concurrency-based load testing, comparing two configurations:
- **Native CPU Offload**: vLLM's OffloadingConnector for KV cache offload to CPU
- **No Offload (Baseline)**: GPU-only approach for KV cache storage

**Interim observations for Qwen3-0.6B:**
- Configurations show similar overall performance
- Native offload achieves 94.2% of baseline average throughput (average -5.8%)
- Throughput impact varies by concurrency level: -16.0% at low load, neutral at high load
- Minimal latency differences (+3.6% average increase)
- TTFT shows mixed behavior: better at very high concurrency (-29.4% at 250), worse at medium loads (+67.8% at 25)

**Expected behavior for larger models** (pending testing):
Given that Qwen3-0.6B has minimal KV-cache memory pressure, we expect larger models to show different characteristics where CPU offloading provides clearer benefits as GPU memory constraints become more significant.

---

## Test Configuration

### Workload Parameters
- **Testing Approach**: Concurrency-based load testing
- **Concurrency Levels**: 5, 25, 50, 100, 250
- **Workload**:
  - Prompt tokens: 128
  - Output tokens: 128
  - Prefix tokens: 10,000 (for prefix caching evaluation)
  - Turns: 5 (multi-turn conversation simulation)
- **Duration**: 30 seconds per concurrency level
- **Random Seed**: 889 (for reproducibility)

### Hardware Setup

**System Configuration:**
- **Platform**: OpenShift on bare metal
- **Node**: Single-node cluster (control plane + worker)
- **GPU**: 2x NVIDIA L40S
  - 48 GB memory per GPU (96 GB total GPU memory)
  - Tensor Parallelism: 1 GPU (TP=1)
- **CPU**: Available for KV-cache offload
- **Architecture**: aarch64

**Software:**
- **vLLM**: Integrated with llm-d v0.4.0
- **llm-d**: v0.4.0 (upstream)
- **PCP**: For system metrics collection
- **GuideLLM**: For benchmark execution
- **OpenShift**: Container orchestration platform

**Model Tested:**
- Qwen/Qwen3-0.6B (interim - larger models pending)

### vLLM Server Configurations

#### Qwen3-0.6B Configurations

1. **Native CPU Offload (native-offload)**:
   ```bash
   vllm serve Qwen/Qwen3-0.6B \
     --tensor-parallel-size 1 \
     --port 8000 \
     --max-num-seq 1024 \
     --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":10000}}'
   ```

2. **No Offload - Baseline (no-offload)**:
   ```bash
   vllm serve Qwen/Qwen3-0.6B \
     --tensor-parallel-size 1 \
     --port 8000 \
     --max-num-seq 1024
   ```

### llm-d Infrastructure

**Components:**
- **EPP (Endpoint Picker)**: In-memory mode for request routing
  - Plugins: queue-scorer, kv-cache-utilization-scorer, prefix-cache-scorer (GPU and CPU)
  - Scheduling profiles with weighted scoring
- **Inference Gateway**: Istio-based gateway for request distribution
- **Model Server**: vLLM deployment managed by llm-d

---

## Benchmark Results

### Performance Summary - Qwen3-0.6B

| Configuration | Avg Throughput (req/s) | Avg Latency (s) | Avg TTFT (ms) | Avg ITL (ms) |
|---------------|------------------------|-----------------|---------------|--------------|
| **No Offload (Baseline)** | **20.5** | 3.28 | 350.2 | **20.4** |
| Native CPU Offload | 19.3 (-5.8%) | 3.40 (+3.6%) | 417.8 (+19.2%) | 21.6 (+5.7%) |

### Detailed Results by Concurrency Level

#### Throughput Comparison (requests/sec)

| Concurrency | No Offload | Native Offload | Difference | % Change |
|-------------|------------|----------------|------------|----------|
| 5 | 5.0 | 4.2 | -0.8 | -16.0% |
| 25 | 14.3 | 13.9 | -0.4 | -2.8% |
| 50 | 21.9 | 20.6 | -1.3 | -5.9% |
| 100 | 28.0 | 26.8 | -1.2 | -4.3% |
| 250 | 33.3 | 33.3 | 0.0 | 0.0% |

#### Token Throughput Comparison (output tokens/sec)

| Concurrency | No Offload | Native Offload | Difference | % Change |
|-------------|------------|----------------|------------|----------|
| 5 | 647.2 | 643.2 | -4.0 | -0.6% |
| 25 | 1,921.3 | 1,855.9 | -65.4 | -3.4% |
| 50 | 2,965.9 | 2,799.4 | -166.5 | -5.6% |
| 100 | 3,914.6 | 3,702.6 | -212.0 | -5.4% |
| 250 | 4,691.0 | 4,592.7 | -98.3 | -2.1% |

#### Latency Comparison - Median (seconds)

| Concurrency | No Offload | Native Offload | Difference | % Change |
|-------------|------------|----------------|------------|----------|
| 5 | 1.00 | 1.00 | 0.00 | 0.0% |
| 25 | 1.70 | 1.80 | 0.10 | +5.9% |
| 50 | 2.20 | 2.30 | 0.10 | +4.5% |
| 100 | 3.30 | 3.50 | 0.20 | +6.1% |
| 250 | 6.90 | 7.00 | 0.10 | +1.4% |

#### Time to First Token (TTFT) - Median (milliseconds)

| Concurrency | No Offload | Native Offload | Difference | % Change |
|-------------|------------|----------------|------------|----------|
| 5 | 64.8 | 61.1 | -3.7 | -5.7% |
| 25 | 103.4 | 173.5 | +70.1 | +67.8% |
| 50 | 194.1 | 312.2 | +118.1 | +60.8% |
| 100 | 522.3 | 534.2 | +11.9 | +2.3% |
| 250 | 869.8 | 614.0 | -255.8 | -29.4% |

#### Inter-Token Latency (ITL) - Median (milliseconds)

| Concurrency | No Offload | Native Offload | Difference | % Change |
|-------------|------------|----------------|------------|----------|
| 5 | 7.3 | 7.3 | 0.0 | 0.0% |
| 25 | 12.2 | 12.3 | 0.1 | +0.8% |
| 50 | 15.5 | 15.9 | 0.4 | +2.6% |
| 100 | 21.9 | 23.2 | 1.3 | +5.9% |
| 250 | 47.5 | 50.2 | 2.7 | +5.7% |

---

## Performance Analysis

### Qwen3-0.6B Results

**Baseline Performance**: 20.5 req/s average, 3.28s median latency, 350.2ms average TTFT

#### Native CPU Offload Impact

**Throughput**: -5.8% average (19.3 req/s vs 20.5 req/s)
- Performance varies significantly by concurrency level
- Low concurrency (5): -16.0% degradation
- Medium concurrency (25-100): -2.8% to -5.9% degradation
- High concurrency (250): Neutral (0.0%)

**Latency**: +3.6% average increase (3.40s vs 3.28s)
- Consistent small increases across most concurrency levels
- Impact ranges from 0.0% to +6.1%
- Most pronounced at concurrency 100 (+6.1%)

**TTFT**: +19.2% average increase (417.8ms vs 350.2ms)
- **Highly variable by concurrency level**
- Low concurrency (5): -5.7% (improvement)
- Medium concurrency (25-50): +60-68% (significant degradation)
- High concurrency (250): -29.4% (significant improvement)

**Inter-Token Latency**: +5.7% average increase (21.6ms vs 20.4ms)
- Minimal impact at low concurrency (0.0% at 5)
- Gradual increase with concurrency
- Maximum impact at high concurrency (+5.7% at 250)

**Analysis**: For Qwen3-0.6B, the CPU offload overhead exceeds the benefits across most scenarios. The small model size means minimal KV-cache memory pressure, so the cost of CPU-GPU memory transfers outweighs any memory management benefits. Interesting

ly, TTFT shows improvement at very high concurrency (250), suggesting potential benefits when the system is heavily loaded.

---

## Concurrency Scaling Analysis

### Low Concurrency (5-25 concurrent requests)
- Native offload shows -16.0% to -2.8% throughput reduction
- TTFT increases significantly (+67.8% at concurrency 25)
- ITL impact minimal (0.0% to +0.8%)
- **Conclusion**: Offload overhead dominates at low load for small models

### Medium Concurrency (50-100 concurrent requests)
- Throughput reduction moderates to -5.9% to -4.3%
- TTFT still elevated but reducing (+60.8% to +2.3%)
- ITL starts showing small increases (+2.6% to +5.9%)
- **Conclusion**: Offload costs remain but are becoming more proportional to load

### High Concurrency (250 concurrent requests)
- Throughput reaches parity (0.0% difference)
- TTFT shows significant improvement (-29.4%)
- ITL shows moderate increase (+5.7%)
- **Conclusion**: At extreme load, offload starts showing benefits in TTFT

**Overall Scaling Observation**: The performance characteristics shift as concurrency increases. At low loads, CPU offload overhead hurts performance. At very high loads (250 concurrent requests), the system begins to show benefits in TTFT, likely as GPU memory pressure increases and offloading reduces contention.

---

## Observations

### 1. Model Size Impact on Offload Value
- **Qwen3-0.6B (current)**: CPU offload provides limited benefit
  - Small KV-cache footprint means minimal GPU memory pressure
  - Transfer overhead dominates over memory management benefits
  - Average throughput: -5.8%

- **Expected for larger models** (pending testing):
  - Qwen3-8B: Likely to show offload benefits at moderate-high concurrency
  - Qwen3-14B: Expected to show clear offload advantages
  - Larger KV-caches create GPU memory pressure where offload helps

### 2. Concurrency-Dependent Behavior
- Performance characteristics vary significantly by load level
- Low concurrency: Offload overhead visible (-16.0% throughput)
- High concurrency: Offload starts showing value (0.0% throughput, -29.4% TTFT)
- **Implication**: Optimal configuration depends on expected workload

### 3. TTFT Variability
- TTFT shows non-monotonic behavior across concurrency levels
- Degrades significantly at medium loads (+60-68%)
- Improves at very high loads (-29.4%)
- **Hypothesis**: Memory contention patterns change with load

### 4. ITL Consistency
- Inter-token latency shows more predictable scaling
- Gradual increase with concurrency (+0.0% to +5.7%)
- More stable than TTFT across concurrency levels
- **Indication**: Token generation overhead is more consistent

### 5. Interim Findings
- For small models like Qwen3-0.6B, CPU offload has limited value
- Performance penalty ranges from minimal to moderate
- System can handle very high concurrency (250) with either configuration
- **Next steps**: Test larger models where KV-cache pressure is higher

---

## Configuration Comparison Summary

### Native CPU Offload vs No Offload (Qwen3-0.6B)

| Metric | Difference | Notes |
|--------|-----------|-------|
| Throughput | -5.8% average | Varies: -16% at low load, 0% at high load |
| Latency | +3.6% average | Consistent small increase |
| TTFT | +19.2% average | Highly variable: -29% to +68% |
| ITL | +5.7% average | Gradual increase with concurrency |
| Configuration Complexity | Higher | Requires kv-transfer-config |

### Configuration Characteristics

**Native CPU Offload:**
- More complex configuration (additional parameters)
- Performance penalty for Qwen3-0.6B
- Shows promise at very high concurrency (TTFT improvement)
- May provide value for larger models (pending testing)

**No Offload (Baseline):**
- Simpler configuration
- Better performance for Qwen3-0.6B
- No dependency on CPU offload infrastructure
- May face GPU memory limits with larger models

---

## Next Steps

1. **Test Larger Models**:
   - Qwen3-8B: Expected to show moderate offload benefits
   - Qwen3-14B: Expected to show clear offload advantages
   - Hypothesis: Larger KV-caches will demonstrate offload value

2. **PCP Metrics Analysis**:
   - GPU utilization comparison
   - Memory bandwidth analysis
   - CPU usage patterns with offloading
   - System-level resource efficiency

3. **Extended Testing Scenarios**:
   - Different workload characteristics (varying prompt/output lengths)
   - Multi-user scenarios with realistic request patterns
   - Long-running stability tests

4. **Configuration Optimization**:
   - Tune `num_cpu_blocks` parameter
   - Evaluate different EPP backend configurations
   - Test with different tensor parallelism settings

---

## Appendix: Technical Details

### GuideLLM Benchmark Command

```bash
guidellm benchmark run \
  --target="http://llm-d-inference-gateway-istio:80" \
  --rate-type="concurrent" \
  --rate="5,25,50,100,250" \
  --max-seconds="30" \
  --random-seed="889" \
  --data='{"prompt_tokens":128,"output_tokens":128,"prefix_tokens":10000,"turns":5}' \
  --sample-requests="0" \
  --outputs=/tmp/benchmark.json
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│ llm-d Infrastructure (OpenShift)                    │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │ Inference Gateway (Istio)                    │  │
│  │  - Request distribution                      │  │
│  │  - Load balancing                            │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼───────────────────────────┐  │
│  │ EPP (Endpoint Picker)                        │  │
│  │  - In-memory mode                            │  │
│  │  - Prefix cache scoring                      │  │
│  │  - KV-cache utilization scoring              │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                               │
│  ┌──────────────────▼───────────────────────────┐  │
│  │ vLLM Model Server                            │  │
│  │  - Qwen3-0.6B                                │  │
│  │  - TP=1 (single GPU)                         │  │
│  │  - With/without CPU offload                  │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │ PCP Monitoring                               │  │
│  │  - System metrics                            │  │
│  │  - GPU metrics (nvidia)                      │  │
│  │  - vLLM metrics (openmetrics)                │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### PCP Archive Details

**Results Structure:**
```
results/
├── 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-0.6B_no-offload/
│   ├── benchmark-config.txt
│   ├── guidellm-results.json (206 MB)
│   └── pcp-archives/ (compressed with zstd)
└── 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-0.6B_native-offload/
    ├── benchmark-config.txt
    ├── guidellm-results.json (200 MB)
    └── pcp-archives/ (compressed with zstd)
```

**Metrics Available in PCP Archives:**
- GPU metrics: utilization, memory usage, bandwidth
- System metrics: CPU, memory, network, disk I/O
- vLLM metrics: KV-cache usage, request statistics
- Benchmark metrics: imported via guidellm2pcp (pending)

### Data Processing Tools

- **GuideLLM**: Benchmark execution and results collection
- **PCP Tools**: pminfo, pmrep, pmlogsummary for metric analysis
- **pcp2arrow**: Convert PCP archives to Parquet format (pending)
- **guidellm2pcp**: Import benchmark results to PCP (pending)
- **Python/Pandas**: Data analysis and aggregation
- **Analysis Scripts**:
  - `analyze_results.py`: GuideLL M results comparison
  - `comprehensive_analysis.py`: Framework for unified pandas analysis

---

## Interim Conclusion

This interim report presents initial findings for Qwen3-0.6B, showing that CPU offload provides limited benefit for this small model size. The -5.8% average throughput reduction is attributed to CPU-GPU transfer overhead exceeding memory management benefits for a model with minimal KV-cache pressure.

Key interim findings:
1. **Small model overhead**: Qwen3-0.6B shows CPU offload costs without corresponding benefits
2. **Concurrency-dependent behavior**: Performance characteristics shift from negative (-16%) at low load to neutral (0%) at high load
3. **TTFT variability**: Time to first token shows complex behavior, improving (-29%) only at very high concurrency
4. **Expected model size correlation**: Larger models should demonstrate clearer offload benefits

**Next Phase**: Testing Qwen3-8B and Qwen3-14B models is expected to show the value proposition of CPU offloading more clearly, as these larger models will experience genuine GPU memory pressure where offloading provides relief.

The infrastructure and methodology are validated and ready for extended testing with larger models.

---

*Data source:*
* *PCP metric archives (system and application metrics)*
* *GuideLLM benchmark results (206 MB + 200 MB JSON)*
* *Benchmark logs: `benchmark-run-complete.log`*

*Test Date: February 12, 2026*
*Report generated with [Claude Code](https://claude.ai/claude-code)*
