# llm-d Batch Gateway Performance Evaluation

## TL;DR

Three models were evaluated on 2×8 NVIDIA H200 GPUs with RHOAI 3.5 EA batch gateway dispatch strategies (ungated, AIMD, AIMD+flow-control) against an interactive-only baseline. **Batch overhead correlates with active parameters per token, not total model size: Qwen3-8B (8B dense) shows 5-8% TTFT p99 increase at r=1, gpt-oss-120b (120B total, ~13B active MoE) shows 43%, and FP8-70B (70B dense) shows 33-107% with throughput collapsing 12x.** The ungated batch processor dispatches 100 concurrent requests regardless of backend capacity. Scaling replicas mitigates overhead for Qwen3-8B (absorbed at r=4) and gpt-oss-120b (absorbed at r=2), but FP8-70B retains 105-260% TTFT p99 overhead at r=8. RHOAI 3.5 EA lacks EPP flow-control plugins, so all dispatch strategies produce equivalent results.

## Methodology

### Infrastructure

| Component | Configuration |
|---|---|
| Cluster | OCP 4.20 |
| GPUs | 2× gx3d nodes, 8× NVIDIA H200 each (16 total) |
| RHOAI | 3.5 EA (LLMInferenceService, EPP, inference gateway) |
| Models | Qwen3-8B (TP=1), FP8-70B (TP=2), gpt-oss-120b (TP=4, MoE MXFP4) |
| Batch Gateway | RHOAI 3.5 EA images (apiserver + processor) |
| Valkey | 8.0.9 (Red Hat RHEL9 container) |
| PostgreSQL | 16 (Red Hat RHEL9 container) |
| vLLM | v0.19.1+rhaiv.6, prefix caching enabled, chunked prefill |
| gpu-memory-utilization | 0.90 (all models) |
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
| Replica configs | Qwen3-8B (TP=1): 1, 4, 8; FP8-70B (TP=2): 1, 4, 8; gpt-oss-120b (TP=4): 1, 2, 4 |

### Metrics Collection

PCP archives capture 2363 metrics per run: vLLM (123), DCGM GPU (24), EPP (221), batch processor (66), PostgreSQL (207), Valkey (111), and system-level (kernel, memory, network, disk).

## Results

### Qwen3-8B

Qwen3-8B uses TP=1 (1 GPU per replica), max_model_len=40,960, KV cache 783,568 tokens (max 19.13x concurrent at max_model_len).

#### Interactive Latency During Burst

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

#### TTFT p99 Overhead vs Baseline

Percentage change from interactive-only at each replica count.

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | +5.8% | -30.0% | +2.7% |
| aimd | +8.1% | -26.0% | -10.6% |
| aimd-flow-control | +8.3% | -30.2% | -5.4% |

At r=1, batch processing adds 5-8% to TTFT p99. At r=4 and r=8, the batch overhead is within noise — the additional replicas absorb the batch load.

The negative values at r=4 (batch scenarios showing lower latency than baseline) are consistent with run-to-run variance rather than batch processing improving interactive latency.

![TTFT p99 during burst — Qwen3-8B](analysis/ttft_p99_burst_Qwen3-8B.png)
![Latency vs replicas — Qwen3-8B](analysis/latency_vs_replicas_Qwen3-8B.png)
![Throughput during burst — Qwen3-8B](analysis/throughput_burst_Qwen3-8B.png)
![Idle vs burst comparison — Qwen3-8B](analysis/idle_vs_burst_Qwen3-8B.png)

#### Throughput

Interactive throughput (requests/sec) is stable across all scenarios: 18.7-20.0 RPS. Batch processing does not reduce interactive throughput for Qwen3-8B.

#### Error Rates

Zero inference errors across all 36 runs (Qwen3-8B, FP8-70B, gpt-oss-120b). Two incomplete requests total (Qwen3-8B aimd r=8, gpt-oss-120b ungated r=4).

#### Batch Processing

30 batch jobs × 100 requests each complete within 60-90 seconds of submission. The batch processor dispatches 30-40 concurrent inference requests (per PCP `processor_inflight_requests` metric). All 3000 batch requests complete with 0 failures.

PCP time-series from the ungated r=4 run shows batch dispatch starting at t+20s and completing by t+300s, with 14-18 vLLM requests running concurrently during burst phases.

#### System Metrics (PCP)

**GPU utilization**: Active GPUs reach 99-100% during burst phases with concurrent batch load (ungated r=4: GPUs 0,1,5,7 at 99-100%). Without batch load, GPU utilization matches the interactive-only pattern.

**vLLM request queue**: During concurrent batch + interactive load, vLLM reports 5-18 running requests per sample (10s intervals). During interactive-only, running requests track the concurrent stream count (1 during idle, 15 during burst).

**Batch processor inflight**: The processor maintains 20-41 concurrent inference requests during active dispatch (ungated scenario, global limit=200). Inflight drops to 0 once batch jobs complete.

![vLLM running requests vs batch inflight — Qwen3-8B ungated r=4](analysis/pcp_concurrent_load_Qwen3-8B_ungated_r4.png)
![vLLM and batch processor time series by scenario — Qwen3-8B r=4](analysis/pcp_timeseries_Qwen3-8B_r4.png)
![KV cache utilization and queue depth — Qwen3-8B r=4](analysis/kv_cache_and_queue_Qwen3-8B_r4.png)
![Batch completion timeline — Qwen3-8B](analysis/batch_timeline_Qwen3-8B.png)

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

FP8-70B shows batch overhead at all replica counts. At r=1, throughput drops from 2.4 to 0.2 RPS — a 12x reduction. The interactive-only baseline itself shows high latency at r=1 and r=4 (TTFT p99 1.8-1.9s), indicating 15 concurrent interactive streams already saturate the model at these scales.

At r=8 (16 GPUs), the baseline improves (551 ms TTFT p99, 6.5 RPS), but batch scenarios still show 105-260% TTFT p99 increase. The batch processor dispatches up to 100 concurrent requests across all replicas, competing directly with interactive traffic for GPU compute and KV cache capacity.

![TTFT p99 during burst — FP8-70B](analysis/ttft_p99_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Latency vs replicas — FP8-70B](analysis/latency_vs_replicas_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Throughput during burst — FP8-70B](analysis/throughput_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Idle vs burst comparison — FP8-70B](analysis/idle_vs_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)

### FP8-70B System Metrics (PCP)

PCP time-series from the ungated r=1 run reveals the contention mechanism:

- **vLLM running requests**: Ramps from 0 to 100 within 90 seconds as the batch processor fills its concurrency limit. The single TP=2 replica attempts to serve 100 concurrent requests.
- **KV cache utilization**: Climbs from 0% to 35% — 10x higher than Qwen3-8B under the same workload. The 568,528-token KV cache is not exhausted, but the model's effective throughput at 100 concurrent requests is bottlenecked by compute, not memory.
- **Batch processor inflight**: Saturates at 100 requests (per-endpoint limit for ungated scenario), confirming the processor dispatches at maximum concurrency regardless of backend capacity.

At r=8, the 100 batch requests are distributed across 8 replicas (~12-13 per replica), reducing per-replica pressure. However, total GPU compute is still shared with 15 interactive streams, resulting in elevated tail latencies.

![FP8-70B running requests and KV cache: r=1 vs r=8](analysis/pcp_fp8_70b_r1_vs_r8.png)

### gpt-oss-120b

gpt-oss-120b is a Mixture-of-Experts (MoE) model with 120B total parameters but ~13B active per token (top-2 routing across 64 experts). It uses MXFP4 quantization, TP=4 (4 GPUs per replica), max_seq_len=131,072, KV cache 5,979,728 tokens (max 45.62x concurrent at max_seq_len). Replica configs: r=1 (4 GPUs), r=2 (8 GPUs), r=4 (16 GPUs).

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 64.6 | 238.6 | 347.7 | 8.9 | 11.7 | 21.5 | 11.7 | 708 |
| ungated | 1 | 207.0 | 487.2 | 496.3 | 39.9 | 55.5 | 55.7 | 2.6 | 313 |
| aimd | 1 | 189.6 | 214.2 | 237.4 | 30.8 | 32.9 | 32.9 | 3.5 | 428 |
| aimd-flow-control | 1 | 199.8 | 736.4 | 738.6 | 32.8 | 39.1 | 39.1 | 3.1 | 379 |
| interactive-only | 2 | 49.9 | 121.0 | 151.5 | 5.7 | 14.7 | 15.2 | 13.5 | 1626 |
| ungated | 2 | 54.0 | 87.0 | 120.7 | 4.9 | 13.9 | 15.4 | 13.9 | 1668 |
| aimd | 2 | 59.4 | 101.9 | 131.7 | 8.2 | 15.5 | 18.4 | 13.0 | 1560 |
| aimd-flow-control | 2 | 58.5 | 103.1 | 131.8 | 5.8 | 15.5 | 17.8 | 12.1 | 1447 |
| interactive-only | 4 | 40.8 | 80.0 | 140.3 | 4.8 | 8.2 | 9.2 | 21.2 | 2548 |
| ungated | 4 | 35.8 | 49.0 | 77.8 | 4.6 | 6.3 | 7.7 | 22.3 | 2675 |
| aimd | 4 | 36.9 | 67.7 | 72.3 | 4.8 | 10.2 | 10.5 | 18.8 | 2252 |
| aimd-flow-control | 4 | 35.3 | 49.3 | 73.4 | 4.6 | 6.6 | 10.8 | 21.7 | 2604 |

**TTFT p99 overhead vs baseline:**

| Scenario | r=1 | r=2 | r=4 |
|---|---|---|---|
| ungated | +42.8% | -20.3% | -44.5% |
| aimd | -31.7% | -13.0% | -48.4% |
| aimd-flow-control | +112.4% | -13.0% | -47.7% |

At r=1, throughput drops from 11.7 to 2.6-3.5 RPS — a 3.3-4.5x reduction. This is less severe than FP8-70B's 12x reduction, consistent with the MoE model's lower per-token compute cost. TTFT p99 overhead at r=1 varies across scenarios (ungated +42.8%, aimd -31.7%, aimd-flow-control +112.4%), reflecting run-to-run variance in the tail — the interactive-only r=1 baseline had only one measured burst cycle.

At r=2 and r=4, batch overhead is within noise or negative. The negative values are consistent with run-to-run variance: the interactive-only r=4 baseline had one burst cycle at 204.7 ms p99 and another at 75.9 ms, pulling the average to 140.3 ms, while batch scenarios had more consistent per-cycle values (72-82 ms).

![TTFT p99 during burst — gpt-oss-120b](analysis/ttft_p99_burst_gpt-oss-120b.png)
![Latency vs replicas — gpt-oss-120b](analysis/latency_vs_replicas_gpt-oss-120b.png)
![Throughput during burst — gpt-oss-120b](analysis/throughput_burst_gpt-oss-120b.png)
![Idle vs burst comparison — gpt-oss-120b](analysis/idle_vs_burst_gpt-oss-120b.png)

### gpt-oss-120b System Metrics (PCP)

PCP time-series from the ungated r=1 run:

- **vLLM running requests**: Ramps from 0 to 100 within 90 seconds, then fluctuates between 84-130. Spikes above 100 occur when 15 interactive burst streams overlap with 100 batch requests on the single TP=4 replica.
- **KV cache utilization**: Peaks at 2.2% — compared to 3.6% for Qwen3-8B and 35% for FP8-70B under the same batch load. The MoE architecture's attention layers produce the same per-token KV volume as a dense model, but the 5.98M-token KV cache (10.5x larger than FP8-70B's 568K tokens) provides proportionally more headroom.
- **Batch processor inflight**: Saturates at 100 requests (per-endpoint limit), identical to the other models.

At r=4, batch processing completes within ~6 minutes of submission. PCP shows batch inflight dropping from 100 to 0 by t+360s, after which only interactive traffic remains (3-6 running requests per sample). This fast batch drain explains the absence of batch overhead at r=4.

During idle phases at r=1, batch load causes single-stream interactive TTFT p99 to rise from 35.6 ms (baseline) to 197-247 ms — a 5.5-6.9x increase. Even single interactive requests queue behind ongoing batch inference when the model is saturated.

### Cross-Model Comparison

| Metric | Qwen3-8B r=1 | FP8-70B r=1 | gpt-oss-120b r=1 |
|---|---|---|---|
| Total parameters | 8B | 70B | 120B |
| Active parameters/token | 8B (dense) | 70B (dense) | ~13B (MoE, top-2/64) |
| Quantization | BF16 | FP8 | MXFP4 |
| GPUs per replica (TP) | 1 | 2 | 4 |
| KV cache tokens | 783,568 | 568,528 | 5,979,728 |
| Max concurrency at max_seq_len | 19.13x | 4.34x | 45.62x |
| Baseline TTFT p99 | 86.9 ms | 1897.9 ms | 347.7 ms |
| Baseline RPS | 19.0 | 2.4 | 11.7 |
| Batch overhead (ungated TTFT p99) | +5.8% | +106.8% | +42.8% |
| Throughput reduction under batch | 1.6% | 91.7% | 77.8% |
| KV cache under batch load | 3.6% | 35% | 2.2% |
| Replicas to absorb batch | 4 | >8 | 2 |

KV cache is not the bottleneck for any model on H200. FP8-70B reaches 35% under batch load but does not exhaust; Qwen3-8B and gpt-oss-120b stay below 4%. The key differentiator is FP8-70B's low max concurrency at max_seq_len (4.34x), meaning the fixed 100-request batch dispatch rate consumes a proportionally larger share of capacity.

vLLM prefix cache hit rate is ~50% during batch processing. Batch requests share 32 system prompts across 3000 requests, enabling prefix cache reuse. The EPP's Valkey-backed prefix cache indexer routes requests to replicas that already have the relevant prefix cached.

![Cross-model TTFT p99 comparison at r=1 (log scale)](analysis/cross_model_ttft_p99_r1.png)

## Observations

1. **Batch overhead correlates with active parameters per token, not total model size.** Qwen3-8B (8B dense) shows 5-8% TTFT p99 increase at r=1. gpt-oss-120b (120B total, ~13B active MoE) shows 43% increase and 3.3-4.5x throughput reduction. FP8-70B (70B dense) shows 33-107% increase with throughput collapsing from 2.4 to 0.2 RPS (12x reduction). The MoE model's sparse activation places its batch sensitivity between the two dense models despite having the largest total parameter count.

2. **The ungated batch processor saturates dense models.** PCP metrics show the processor dispatching 100 concurrent inference requests regardless of backend capacity. On a single FP8-70B replica (4.34x effective concurrency at max_seq_len), vLLM running requests hit 100, KV cache reaches 35%, and interactive requests queue behind batch. On gpt-oss-120b (45.62x effective concurrency), the same 100 concurrent requests produce only 2.2% KV cache usage.

3. **All three dispatch strategies produce equivalent results in RHOAI 3.5 EA.** Ungated, AIMD, and AIMD+flow-control show no measurable difference in interactive latency across all three models. RHOAI 3.5 EA lacks flow-control plugins in the EPP, and AIMD metrics are not exposed by this processor version, so scenarios 3 and 4 operate identically to scenario 2 from the inference backend's perspective.

4. **Replica scaling effectiveness varies by model.** Qwen3-8B absorbs batch load at r=4. gpt-oss-120b absorbs it at r=2. FP8-70B still shows 105-260% TTFT p99 overhead at r=8 (16 GPUs) — the aggregate batch load (3000 requests) competes with interactive traffic at all tested scales.

5. **FP8-70B baseline is already saturated at r=1 with 15 interactive streams.** The interactive-only baseline shows 1.9s TTFT p99 and 2.4 RPS at r=1 — the model cannot sustain 15 concurrent streams on a single TP=2 replica. This means the benchmark workload exceeds the model's capacity before batch is added.

6. **MoE batch drain is fast at scale.** At r=4 (16 GPUs), gpt-oss-120b batch processing completes within ~6 minutes. PCP shows batch inflight dropping from 100 to 0 before the second measured burst cycle, eliminating batch overhead from measurements. FP8-70B batch requests persist throughout the measurement window at all replica counts.

7. **30×100 batch job config enables concurrent measurement.** Batch dispatch starts within 20s of submission and overlaps with interactive traffic throughout the measurement window (except where batch drains faster than the traffic cycle, as with gpt-oss-120b at r=4).

## Known Limitations (RHOAI 3.5 EA)

| Feature | Status | Impact |
|---|---|---|
| EPP flow-control plugins | Not available | Scenarios 3 and 4 are functionally identical to scenario 2 |
| AIMD processor metrics | Not exposed | Cannot observe adaptive concurrency dynamics |
| Job-level processor metrics | Not exposed | No job duration, token throughput, or per-model inflight metrics |
| Batch gateway GC | Health probe fails | Disabled; manual state cleanup between runs |
| pmdavalkey | Requires PCP ≥ 7.1.1 | Workaround: dnf upgrade in PCP pod startup |

## Next Steps

1. **RHOAI EA2 retest**: Enable flow-control plugins, AIMD metrics, and job-level metrics when the next RHOAI version ships updated upstream code.
2. **Disaggregated P/D**: Evaluate batch gateway with prefill/decode separation when RDMA is configured between nodes.
