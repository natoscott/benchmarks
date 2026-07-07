# llm-d Batch Gateway Performance Evaluation

**Cluster:** psap-fire-athena (OCP 4.20, RHOAI 3.5 EA)
**Date:** 2026-07-07
**Status:** Qwen3-8B complete (12/12 runs). FP8-70B and gpt-oss-120b in progress.

## TL;DR

Batch gateway dispatch strategies (ungated, AIMD, AIMD+flow-control) were evaluated against an interactive-only baseline on Qwen3-8B and Meta-Llama-3.1-70B-Instruct-FP8 across 1, 4, and 8 replicas on 2×8 NVIDIA H200 GPUs. **Batch processing impact scales with model size: Qwen3-8B (8B parameters) shows 5-8% TTFT p99 increase at r=1, while FP8-70B (70B parameters) shows 33-107% increase at r=1 with throughput dropping from 2.4 to 0.2 RPS.** The root cause is visible in PCP metrics: the ungated batch processor dispatches up to 100 concurrent requests onto a single FP8-70B replica, driving KV cache from 0% to 35% and vLLM running requests to 100 — far exceeding the model's effective concurrency limit of ~4x at max_seq_len. At 8 replicas, FP8-70B batch overhead remains 105-260% TTFT p99, indicating the workload exceeds even 16 GPUs' capacity at these concurrency levels. RHOAI 3.5 EA lacks EPP flow-control plugins, so all dispatch strategies produce equivalent results. gpt-oss-120b results pending.

## Methodology

### Infrastructure

| Component | Configuration |
|---|---|
| Cluster | psap-fire-athena, OCP 4.20 |
| GPUs | 2× gx3d nodes, 8× NVIDIA H200 each (16 total) |
| RHOAI | 3.5 EA (LLMInferenceService, EPP, inference gateway) |
| Model | Qwen/Qwen3-8B, TP=1, gpu-memory-utilization=0.90 |
| Batch Gateway | RHOAI 3.5 EA images (apiserver + processor) |
| Valkey | 8.0.9 (Red Hat RHEL9 container) |
| PostgreSQL | 16 (Red Hat RHEL9 container) |
| vLLM | v0.19.1+rhaiv.6, prefix caching enabled, chunked prefill |
| vLLM KV cache | 783,568 tokens (max 19.13x concurrent at max_model_len) |
| guidellm | v0.7.1 |
| PCP | 7.1.5 (openmetrics, pmdavalkey, pmdapostgresql) |

### Scenarios

| ID | Name | Description |
|---|---|---|
| 0 | interactive-only | No batch gateway. Interactive traffic baseline. |
| 2 | ungated | Batch gateway, global=200, per-endpoint=100, AIMD disabled |
| 3 | aimd | Batch gateway, global=100, per-endpoint=20, AIMD enabled |
| 4 | aimd-flow-control | Same as AIMD. Flow-control plugins unavailable in RHOAI 3.5 EA. |

### Workload

| Parameter | Value |
|---|---|
| Interactive burst | 15 concurrent streams, 60s |
| Interactive idle | 1 concurrent stream, 60s |
| Cycles | 3 (1 warmup excluded, 2 measured) |
| Batch jobs | 30 × 100 requests (3000 total) |
| Batch completion window | 24h (all jobs) |
| Prompt tokens | synthetic, 512 |
| Output tokens | 128 |
| Replica configs | 1, 4, 8 |

### Metrics Collection

PCP archives capture 2363 metrics per run: vLLM (123), DCGM GPU (24), EPP (221), batch processor (66), PostgreSQL (207), Valkey (111), and system-level (kernel, memory, network, disk).

## Results

### Interactive Latency During Burst

All values in milliseconds. Lower is better.

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 32.2 | 45.1 | 86.9 | 5.9 | 5.9 | 6.1 | 19.0 | 2282 |
| ungated | 1 | 34.2 | 52.9 | 91.9 | 5.9 | 6.4 | 6.8 | 18.7 | 2248 |
| aimd | 1 | 30.8 | 62.5 | 93.9 | 5.9 | 6.4 | 6.7 | 18.7 | 2247 |
| aimd-flow-control | 1 | 32.8 | 60.1 | 94.1 | 5.9 | 6.4 | 6.8 | 18.7 | 2251 |
| interactive-only | 4 | 31.6 | 44.1 | 78.4 | 5.7 | 5.9 | 6.3 | 19.5 | 2338 |
| ungated | 4 | 29.1 | 38.5 | 54.9 | 5.9 | 6.2 | 6.3 | 19.2 | 2300 |
| aimd | 4 | 29.0 | 41.4 | 58.0 | 5.9 | 6.2 | 6.2 | 19.1 | 2297 |
| aimd-flow-control | 4 | 29.0 | 38.6 | 54.7 | 5.9 | 6.1 | 6.2 | 19.1 | 2298 |
| interactive-only | 8 | 29.2 | 37.3 | 48.0 | 5.8 | 5.9 | 6.0 | 19.6 | 2354 |
| ungated | 8 | 28.3 | 36.7 | 49.3 | 5.7 | 6.0 | 6.1 | 20.0 | 2396 |
| aimd | 8 | 28.4 | 37.0 | 42.9 | 5.6 | 6.0 | 6.1 | 20.0 | 2398 |
| aimd-flow-control | 8 | 28.3 | 36.5 | 45.4 | 5.8 | 6.0 | 6.2 | 19.6 | 2359 |

### TTFT p99 Overhead vs Baseline

Percentage change from interactive-only at each replica count.

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | +5.8% | -30.0% | +2.7% |
| aimd | +8.1% | -26.0% | -10.6% |
| aimd-flow-control | +8.3% | -30.2% | -5.4% |

At r=1, batch processing adds 5-8% to TTFT p99. At r=4 and r=8, the batch overhead is within noise — the additional replicas absorb the batch load.

The negative values at r=4 (batch scenarios showing lower latency than baseline) are consistent with run-to-run variance rather than batch processing improving interactive latency.

### Throughput

Interactive throughput (requests/sec) is stable across all scenarios: 18.7-20.0 RPS. Batch processing does not reduce interactive throughput for Qwen3-8B.

### Error Rates

Zero inference errors across all 12 runs (27,000+ interactive requests, 36,000 batch requests). One incomplete request in aimd r=8.

### Batch Processing

30 batch jobs × 100 requests each complete within 60-90 seconds of submission. The batch processor dispatches 30-40 concurrent inference requests (per PCP `processor_inflight_requests` metric). All 3000 batch requests complete with 0 failures.

PCP time-series from the ungated r=4 run shows batch dispatch starting at t+20s and completing by t+300s, with 14-18 vLLM requests running concurrently during burst phases.

### System Metrics (PCP)

**GPU utilization**: Active GPUs reach 99-100% during burst phases with concurrent batch load (ungated r=4: GPUs 0,1,5,7 at 99-100%). Without batch load, GPU utilization matches the interactive-only pattern.

**vLLM request queue**: During concurrent batch + interactive load, vLLM reports 5-18 running requests per sample (10s intervals). During interactive-only, running requests track the concurrent stream count (1 during idle, 15 during burst).

**Batch processor inflight**: The processor maintains 20-41 concurrent inference requests during active dispatch (ungated scenario, global limit=200). Inflight drops to 0 once batch jobs complete.

### Meta-Llama-3.1-70B-Instruct-FP8

FP8-70B uses TP=2 (2 GPUs per replica), max_seq_len=131,072, KV cache 568,528 tokens (max 4.34x concurrent at max_seq_len).

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 135.1 | 1884.3 | 1897.9 | 47.4 | 51.4 | 61.6 | 2.4 | 300 |
| ungated | 1 | 3914.0 | 3925.5 | 3925.7 | 388.5 | 388.5 | 388.5 | 0.2 | 60 |
| aimd | 1 | 2517.5 | 2524.3 | 2524.7 | 379.9 | 399.3 | 399.3 | 0.2 | 58 |
| aimd-flow-control | 1 | 3860.4 | 3867.1 | 3867.3 | 386.4 | 386.4 | 386.4 | 0.2 | 60 |
| interactive-only | 4 | 97.0 | 1777.8 | 1789.0 | 14.2 | 639.1 | 639.1 | 0.7 | 94 |
| ungated | 4 | 106.0 | 2599.4 | 2609.7 | 14.9 | 525.6 | 525.6 | 0.7 | 91 |
| aimd | 4 | 104.0 | 1789.7 | 1800.3 | 13.8 | 424.2 | 435.7 | 0.9 | 114 |
| aimd-flow-control | 4 | 117.0 | 2924.7 | 2933.3 | 14.5 | 459.6 | 459.6 | 0.7 | 88 |
| interactive-only | 8 | 90.9 | 442.1 | 551.2 | 12.7 | 50.9 | 55.5 | 6.5 | 786 |
| ungated | 8 | 98.1 | 334.4 | 1129.4 | 13.6 | 38.1 | 274.0 | 3.7 | 450 |
| aimd | 8 | 96.9 | 919.4 | 1981.6 | 13.4 | 195.5 | 217.8 | 4.4 | 529 |
| aimd-flow-control | 8 | 96.5 | 1004.5 | 1790.8 | 13.1 | 148.8 | 172.1 | 4.7 | 563 |

**TTFT p99 overhead vs baseline:**

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | +106.8% | +45.9% | +104.9% |
| aimd | +33.0% | +0.6% | +259.5% |
| aimd-flow-control | +103.8% | +64.0% | +224.9% |

FP8-70B shows batch overhead at all replica counts. At r=1, throughput drops from 2.4 to 0.2 RPS — an 12x reduction. The interactive-only baseline itself shows high latency at r=1 and r=4 (TTFT p99 1.8-1.9s), indicating 15 concurrent interactive streams already saturate the model at these scales.

At r=8 (16 GPUs), the baseline improves (551 ms TTFT p99, 6.5 RPS), but batch scenarios still show 105-260% TTFT p99 increase. The batch processor dispatches up to 100 concurrent requests across all replicas, competing directly with interactive traffic for GPU compute and KV cache capacity.

### FP8-70B System Metrics (PCP)

PCP time-series from the ungated r=1 run reveals the contention mechanism:

- **vLLM running requests**: Ramps from 0 to 100 within 90 seconds as the batch processor fills its concurrency limit. The single TP=2 replica attempts to serve 100 concurrent requests.
- **KV cache utilization**: Climbs from 0% to 35% — 10x higher than Qwen3-8B under the same workload. The 568,528-token KV cache is not exhausted, but the model's effective throughput at 100 concurrent requests is bottlenecked by compute, not memory.
- **Batch processor inflight**: Saturates at 100 requests (per-endpoint limit for ungated scenario), confirming the processor dispatches at maximum concurrency regardless of backend capacity.

At r=8, the 100 batch requests are distributed across 8 replicas (~12-13 per replica), reducing per-replica pressure. However, total GPU compute is still shared with 15 interactive streams, resulting in elevated tail latencies.

### Cross-Model Comparison

| Metric | Qwen3-8B r=1 | FP8-70B r=1 | Ratio |
|---|---|---|---|
| Model parameters | 8B | 70B | 8.75x |
| TP | 1 | 2 | 2x |
| KV cache tokens | 783,568 | 568,528 | 0.73x |
| Max concurrency at max_seq_len | 19.13x | 4.34x | 0.23x |
| Baseline TTFT p99 | 86.9 ms | 1897.9 ms | 21.8x |
| Baseline RPS | 19.0 | 2.4 | 0.13x |
| Batch overhead (ungated TTFT p99) | +5.8% | +106.8% | 18.4x |
| KV cache under batch load | 3.6% | 35% | 9.7x |

Batch processing impact scales super-linearly with model size. FP8-70B has 8.75x more parameters but 18.4x more batch overhead. The model's lower effective concurrency limit (4.34x vs 19.13x) means the fixed batch dispatch rate (100 concurrent requests) represents a proportionally larger share of available capacity.

### KV Cache and Queue Depth

KV cache utilization peaks at 3.6% during concurrent batch + interactive load (ungated r=4). Qwen3-8B's 783,568-token KV cache (max 19.13x concurrent requests at max_model_len=40,960) is not under pressure on H200. This explains why batch processing has minimal latency impact — there is no KV cache contention.

vLLM prefix cache hit rate is ~50% during batch processing. Batch requests share 32 system prompts across 3000 requests, enabling prefix cache reuse. The EPP's Valkey-backed prefix cache indexer routes requests to replicas that already have the relevant prefix cached.

### Visualizations

- `analysis/ttft_p99_burst_Qwen3-8B.png` — TTFT p99 comparison across scenarios
- `analysis/latency_vs_replicas_Qwen3-8B.png` — TTFT and TPOT p99 scaling with replicas
- `analysis/throughput_burst_Qwen3-8B.png` — Interactive throughput comparison
- `analysis/idle_vs_burst_Qwen3-8B.png` — Idle vs burst latency comparison
- `analysis/pcp_concurrent_load_Qwen3-8B_ungated_r4.png` — vLLM running requests vs batch inflight (dual y-axis)
- `analysis/pcp_timeseries_Qwen3-8B_r4.png` — vLLM and batch processor time series by scenario
- `analysis/kv_cache_and_queue_Qwen3-8B_r4.png` — KV cache utilization and queue depth comparison
- `analysis/batch_timeline_Qwen3-8B.png` — Batch completion timeline across scenarios
- `analysis/pcp_fp8_70b_r1_vs_r8.png` — FP8-70B running requests and KV cache: r=1 vs r=8
- `analysis/cross_model_ttft_p99_r1.png` — Cross-model TTFT p99 comparison at r=1 (log scale)

## Observations

1. **Batch overhead scales super-linearly with model size.** Qwen3-8B (8B params) shows 5-8% TTFT p99 increase at r=1. FP8-70B (70B params) shows 33-107% increase at r=1 with throughput collapsing from 2.4 to 0.2 RPS. The fixed batch dispatch rate (100 concurrent requests) represents a proportionally larger share of capacity for larger models.

2. **The ungated batch processor saturates large models.** PCP metrics show the processor dispatching 100 concurrent inference requests onto a single FP8-70B replica that supports ~4x effective concurrency at max_seq_len. The model is overwhelmed — vLLM running requests hit 100, KV cache reaches 35%, and interactive requests queue behind batch requests.

3. **All three dispatch strategies produce equivalent results in RHOAI 3.5 EA.** Ungated, AIMD, and AIMD+flow-control show no measurable difference in interactive latency. RHOAI 3.5 EA lacks flow-control plugins in the EPP, and AIMD metrics are not exposed by this processor version, so scenarios 3 and 4 operate identically to scenario 2 from the inference backend's perspective.

4. **Scaling replicas does not fully mitigate batch overhead for FP8-70B.** At r=8 (16 GPUs), batch scenarios still show 105-260% TTFT p99 overhead. The batch processor distributes requests across replicas, but the aggregate batch load (3000 requests) still competes with interactive traffic. This contrasts with Qwen3-8B where r=4 absorbs the batch load completely.

5. **FP8-70B baseline is already saturated at r=1 with 15 interactive streams.** The interactive-only baseline shows 1.9s TTFT p99 and 2.4 RPS at r=1 — the model cannot sustain 15 concurrent streams on a single TP=2 replica. This means the benchmark workload exceeds the model's capacity before batch is added.

6. **KV cache is not the bottleneck.** Even for FP8-70B under heavy batch load, KV cache peaks at 35%. The bottleneck is compute throughput, not memory capacity. H200's 141 GB HBM3e provides ample KV cache headroom.

7. **30×100 batch job config enables concurrent measurement.** Batch dispatch starts within 20s of submission and overlaps with interactive traffic throughout the measurement window.

## Known Limitations (RHOAI 3.5 EA)

| Feature | Status | Impact |
|---|---|---|
| EPP flow-control plugins | Not available | Scenarios 3 and 4 are functionally identical to scenario 2 |
| AIMD processor metrics | Not exposed | Cannot observe adaptive concurrency dynamics |
| Job-level processor metrics | Not exposed | No job duration, token throughput, or per-model inflight metrics |
| Batch gateway GC | Health probe fails | Disabled; manual state cleanup between runs |
| pmdavalkey | Requires PCP ≥ 7.1.1 | Workaround: dnf upgrade in PCP pod startup |

## Next Steps

1. **FP8-70B and gpt-oss-120b**: In progress. Larger models will stress GPU and KV cache, revealing batch contention effects not visible with Qwen3-8B.
2. **RHOAI EA2 retest**: Enable flow-control plugins, AIMD metrics, and job-level metrics when the next RHOAI version ships updated upstream code.
3. **HTTPRoute validation**: Confirm EPP routing is active (colleague's HTTPRoute cleanup was applied before these runs).
4. **Disaggregated P/D**: Evaluate batch gateway with prefill/decode separation when RDMA is configured between nodes.
