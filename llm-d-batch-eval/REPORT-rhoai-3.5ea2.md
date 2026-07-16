# llm-d Batch Gateway Performance Evaluation — RHOAI 3.5 EA2

**Scope:** 37 runs (36 standard + 1 scenario 5) using full RHOAI 3.5 EA2 stack: EA2 EPP (llm-d-router), EA2 batch gateway, EA2 KV cache.

## TL;DR

Three models were evaluated on 2x8 NVIDIA H200 GPUs with RHOAI 3.5 EA2 full-stack dispatch strategies against an interactive-only baseline. **With the full EA2 stack, flow-control priority dispatch ordering reduces FP8-70B TTFT p99 overhead from +137% (ungated) to +102% at r=1 and from +274% to +215% at r=8, but AIMD remains non-functional — vLLM never returns 429/5xx so the concurrency limit stays flat across all scenarios including scenario 5 (perEndpoint=5).** EA2 EPP auto-registers utilization-detector (saturation-based shedding). `perEndpoint` concurrency limits are now enforced (were ignored in EA1). Qwen3-8B: ungated adds only +32% at r=1, aimd/FC add +135-176% (scheduling overhead). gpt-oss-120b: FC best at r=1 (+1.8% vs ungated -2.5%), high variability at r=2/r=4. Batch throughput tradeoff: ungated completes 30 jobs 5-6x faster than aimd/FC but at cost of interactive latency for constrained models.

## Methodology

Common infrastructure, workload, and scenario definitions are in [REPORT.md](REPORT.md).

### EA2-Specific Configuration

| Component | Configuration |
|---|---|
| EPP | odh-llm-d-router-endpoint-picker-rhel9 (renamed from inference-scheduler), apiVersion llm-d.ai/v1alpha1 |
| EPP features | utilization-detector auto-registered (queueDepthThreshold=5, kvCacheUtilThreshold=0.8), precise-prefix-cache-producer for Valkey, data layer enabled by default |
| KV cache | Updated digest |
| Batch Gateway | Digest-pinned EA2 via ImageDigestMirrorSet |
| GC | Enabled and functional |
| `perEndpoint` enforcement | Enforced |
| Scenario 5 | aimd-low-concurrency (perEndpoint=5, global=20, FP8-70B r=1 only) |

### EA2 Changes from EA1+EA2 Batch

| Change | Detail |
|---|---|
| EPP binary | EA2 EPP replaces EA1 EPP. Binary renamed from inference-scheduler to llm-d-router. apiVersion changed to llm-d.ai/v1alpha1. |
| utilization-detector | Auto-registered in EA2 EPP (queueDepthThreshold=5, kvCacheUtilThreshold=0.8). Not present in EA1 EPP. |
| prefix-cache-producer | precise-prefix-cache-producer for Valkey enabled by default. |
| Data layer | Enabled by default in EA2 EPP. |
| KV cache images | Updated digest. |
| Scenario 5 | New: aimd-low-concurrency with perEndpoint=5, global=20 to test AIMD adaptation under tighter limits. |

### Scenarios

| ID | Name | Description |
|---|---|---|
| 0 | interactive-only | No batch gateway. Interactive traffic baseline. |
| 2 | ungated | Batch gateway, global=200, per-endpoint=100, AIMD disabled |
| 3 | aimd | Batch gateway, global=100, per-endpoint=20, AIMD enabled |
| 4 | aimd-flow-control | Same as AIMD + EPP `flowControl` feature gate with priority bands. Batch processor sets `x-gateway-inference-objective: batch-sheddable` (priority -1). Interactive traffic has no objective header (default priority 0). |
| 5 | aimd-low-concurrency | AIMD enabled, perEndpoint=5, global=20. FP8-70B r=1 only. Tests whether tighter concurrency limits trigger AIMD adaptation. |

## Results

### Qwen3-8B

Qwen3-8B uses TP=1 (1 GPU per replica), max_model_len=40,960, KV cache 783,568 tokens (max 19.13x concurrent at max_model_len).

#### Interactive Latency During Burst

All overhead measurements are from burst phases (15 concurrent interactive streams), where batch contention is highest. All values in milliseconds. Lower is better.

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 32.6 | 69.1 | 93.3 | 5.9 | 6.3 | 6.8 | 18.8 | 2251 |
| ungated | 1 | 34.8 | 99.2 | 123.2 | 5.7 | 6.0 | 6.5 | 19.3 | 2321 |
| aimd | 1 | 169.5 | 204.2 | 219.2 | 8.8 | 11.6 | 12.1 | 11.6 | 1411 |
| aimd-flow-control | 1 | 160.1 | 188.2 | 257.8 | 8.2 | 11.4 | 12.2 | 12.4 | 1487 |
| interactive-only | 4 | 30.4 | 41.7 | 58.1 | 5.9 | 6.1 | 6.2 | 19.2 | 2302 |
| ungated | 4 | 31.4 | 43.7 | 53.9 | 5.9 | 6.0 | 6.1 | 19.2 | 2306 |
| aimd | 4 | 31.9 | 41.8 | 54.6 | 5.9 | 6.0 | 6.1 | 19.1 | 2294 |
| aimd-flow-control | 4 | 32.0 | 40.7 | 49.2 | 5.9 | 6.1 | 6.2 | 19.1 | 2289 |
| interactive-only | 8 | 31.3 | 38.2 | 46.2 | 5.6 | 5.8 | 5.9 | 20.0 | 2403 |
| ungated | 8 | 32.8 | 39.5 | 50.4 | 5.6 | 5.8 | 5.9 | 19.9 | 2387 |
| aimd | 8 | 32.4 | 38.7 | 46.7 | 5.6 | 5.8 | 5.9 | 19.9 | 2392 |
| aimd-flow-control | 8 | 32.7 | 39.9 | 45.7 | 5.6 | 5.8 | 5.8 | 20.0 | 2397 |

#### TTFT p99 Overhead vs Baseline

Percentage change from interactive-only at each replica count.

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | +32.0% | -7.2% | +9.1% |
| aimd | +134.9% | -6.0% | +1.1% |
| aimd-flow-control | +176.3% | -15.3% | -1.1% |

At r=1, ungated adds +32.0% overhead (123.2 ms vs 93.3 ms baseline). Aimd and flow-control show +135% and +176% respectively — 219.2 ms and 257.8 ms versus 93.3 ms baseline. ITL increases from 6.8 ms to 12.1-12.2 ms. Throughput drops from 18.8 to 11.6-12.4 RPS (34-38% reduction). The aimd/FC overhead at r=1 is driven by perEndpoint=20 limiting batch to 20 concurrent requests (vs 100 for ungated) while the EPP priority scheduling adds latency.

At r=4 and r=8, all scenarios converge within noise (-15% to +9%). Sufficient replica capacity absorbs both batch and interactive load. At r=4, flow-control shows the lowest TTFT p99 (49.2 ms vs 58.1 ms baseline, -15.3%).

Batch throughput: ungated completes 30 jobs in mean 67s at r=1. Aimd/FC: 372-386s (5.5x slower).

![TTFT p99 during burst — Qwen3-8B](analysis/rhoai-3.5ea2-full/ttft_p99_burst_Qwen3-8B.png)
![Latency vs replicas — Qwen3-8B](analysis/rhoai-3.5ea2-full/latency_vs_replicas_Qwen3-8B.png)
![Throughput during burst — Qwen3-8B](analysis/rhoai-3.5ea2-full/throughput_burst_Qwen3-8B.png)
![Idle vs burst comparison — Qwen3-8B](analysis/rhoai-3.5ea2-full/idle_vs_burst_Qwen3-8B.png)

#### Throughput

Interactive throughput at r=4 and r=8 is stable across all scenarios: 19.1-20.0 RPS. At r=1, aimd/FC throughput drops to 11.6-12.4 RPS (vs 18.8-19.3 for interactive-only/ungated) due to scheduling overhead with the enforced 20-request dispatch limit.

#### Error Rates

Zero inference errors across all Qwen3-8B runs. All interactive requests completed.

#### Batch Processing

With `perEndpoint` enforced at 20 for aimd/FC scenarios, batch dispatch pressure is reduced compared to ungated. The `model_inflight_requests` metric confirms 20 concurrent batch requests for aimd/FC (vs 100 for ungated). All 3000 batch requests complete with 0 failures. Qwen3-8B completes all 30 batch jobs in every scenario.

### Meta-Llama-3.1-70B-Instruct-FP8

FP8-70B uses TP=2 (2 GPUs per replica), max_seq_len=131,072, KV cache 568,528 tokens (max 4.34x concurrent at max_seq_len).

#### Interactive Latency During Burst

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 54.8 | 208.2 | 400.3 | 13.2 | 14.7 | 15.9 | 8.2 | 997 |
| ungated | 1 | 675.9 | 945.4 | 947.1 | 41.0 | 45.3 | 45.3 | 4.0 | 487 |
| aimd | 1 | 615.1 | 1215.4 | 1217.7 | 24.4 | 30.7 | 31.3 | 4.1 | 504 |
| aimd-flow-control | 1 | 486.2 | 807.9 | 809.7 | 24.6 | 30.2 | 32.0 | 4.1 | 505 |
| interactive-only | 4 | 71.0 | 445.1 | 612.9 | 13.0 | 48.9 | 52.8 | 5.5 | 660 |
| ungated | 4 | 102.4 | 936.8 | 947.7 | 14.7 | 567.9 | 567.9 | 0.8 | 101 |
| aimd | 4 | 98.0 | 1258.1 | 2601.1 | 13.5 | 131.3 | 153.0 | 2.7 | 332 |
| aimd-flow-control | 4 | 96.0 | 993.0 | 2959.9 | 13.6 | 159.4 | 170.1 | 2.7 | 328 |
| interactive-only | 8 | 85.6 | 181.5 | 443.3 | 12.7 | 38.1 | 48.7 | 7.2 | 864 |
| ungated | 8 | 95.2 | 1055.7 | 1658.9 | 13.3 | 158.4 | 212.2 | 4.8 | 577 |
| aimd | 8 | 97.4 | 573.8 | 1639.2 | 13.2 | 97.6 | 117.3 | 4.8 | 580 |
| aimd-flow-control | 8 | 95.2 | 505.2 | 1397.4 | 12.9 | 83.7 | 107.7 | 5.0 | 601 |

#### TTFT p99 Overhead vs Baseline

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | +136.6% | +54.6% | +274.3% |
| aimd | +204.2% | +324.4% | +269.8% |
| aimd-flow-control | +102.3% | +383.0% | +215.3% |

Flow-control provides the lowest overhead at r=1 (+102.3%, 809.7 ms vs 400.3 ms baseline) compared to ungated (+136.6%, 947.1 ms) and aimd (+204.2%, 1217.7 ms). At r=8, flow-control again shows the lowest overhead (+215.3%, 1397.4 ms vs 443.3 ms baseline) compared to ungated (+274.3%, 1658.9 ms) and aimd (+269.8%, 1639.2 ms). Priority dispatch ordering — interactive dispatched before batch at every scheduling tick — provides measurable benefit for this capacity-constrained model.

At r=4, all batch scenarios show elevated overhead (+55% to +383%). The baseline itself has elevated latency (612.9 ms p99) suggesting cluster conditions affected this replica count.

Throughput impact: all batch scenarios reduce throughput at r=1 to 4.0-4.1 RPS (vs 8.2 baseline, 50% reduction). At r=8, flow-control preserves 5.0 RPS vs 4.8 for ungated/aimd.

Batch: ungated completes 30 jobs in 365s mean. Aimd/FC complete only 4-5 jobs during the run window (constrained model, low concurrency). Scenario 5 completes 17 jobs.

![TTFT p99 during burst — FP8-70B](analysis/rhoai-3.5ea2-full/ttft_p99_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Latency vs replicas — FP8-70B](analysis/rhoai-3.5ea2-full/latency_vs_replicas_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Throughput during burst — FP8-70B](analysis/rhoai-3.5ea2-full/throughput_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Idle vs burst comparison — FP8-70B](analysis/rhoai-3.5ea2-full/idle_vs_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)

### gpt-oss-120b

gpt-oss-120b is a Mixture-of-Experts (MoE) model with 120B total parameters but ~13B active per token (top-2 routing across 64 experts). It uses MXFP4 quantization, TP=4 (4 GPUs per replica), max_seq_len=131,072, KV cache 5,979,728 tokens (max 45.62x concurrent at max_seq_len). Replica configs: r=1 (4 GPUs), r=2 (8 GPUs), r=4 (16 GPUs).

#### Interactive Latency During Burst

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 47.5 | 138.9 | 150.3 | 5.6 | 7.2 | 7.6 | 18.4 | 2212 |
| ungated | 1 | 44.6 | 97.9 | 146.5 | 5.3 | 6.5 | 7.1 | 19.9 | 2391 |
| aimd | 1 | 50.1 | 119.6 | 731.3 | 6.7 | 8.1 | 10.6 | 16.5 | 1983 |
| aimd-flow-control | 1 | 50.5 | 120.0 | 153.0 | 7.0 | 8.8 | 9.1 | 16.0 | 1918 |
| interactive-only | 2 | 44.3 | 51.1 | 101.7 | 5.3 | 5.3 | 5.5 | 20.7 | 2480 |
| ungated | 2 | 44.5 | 52.3 | 110.0 | 5.3 | 5.3 | 5.4 | 20.7 | 2489 |
| aimd | 2 | 50.1 | 121.6 | 182.7 | 7.0 | 8.6 | 14.4 | 15.9 | 1913 |
| aimd-flow-control | 2 | 45.8 | 140.0 | 591.0 | 5.6 | 17.4 | 22.8 | 15.1 | 1812 |
| interactive-only | 4 | 44.6 | 51.5 | 86.5 | 5.3 | 5.3 | 5.4 | 20.8 | 2496 |
| ungated | 4 | 45.5 | 119.0 | 893.1 | 5.3 | 5.7 | 6.8 | 19.9 | 2394 |
| aimd | 4 | 50.2 | 118.6 | 174.5 | 7.0 | 8.7 | 11.7 | 15.9 | 1913 |
| aimd-flow-control | 4 | 40.4 | 117.9 | 575.9 | 4.8 | 5.8 | 10.4 | 21.6 | 2594 |

#### TTFT p99 Overhead vs Baseline

| Scenario | r=1 | r=2 | r=4 |
|---|---|---|---|
| ungated | -2.5% | +8.2% | +932.5% |
| aimd | +386.6% | +79.6% | +101.7% |
| aimd-flow-control | +1.8% | +481.1% | +565.8% |

gpt-oss-120b shows high p99 variability across scenarios. At r=1, ungated (-2.5%, 146.5 ms vs 150.3 ms) and flow-control (+1.8%, 153.0 ms vs 150.3 ms) are both within noise of baseline. Aimd shows a 731.3 ms p99 spike (+386.6%) — a single tail event drives the percentage.

At r=2, ungated is within noise (+8.2%), aimd shows +79.6% (182.7 ms vs 101.7 ms), and flow-control shows +481.1% (591.0 ms vs 101.7 ms) with ITL p99 rising from 5.5 ms to 22.8 ms.

At r=4, ungated shows a 893.1 ms p99 spike (+932.5%), aimd +101.7% (174.5 ms vs 86.5 ms), and flow-control +565.8% (575.9 ms vs 86.5 ms). The MoE model's large KV cache (5.98M tokens) generally absorbs batch load, but tail latency is unstable across runs — individual p99 spikes inflate overhead percentages without reflecting sustained degradation.

Throughput at r=1: ungated 19.9 RPS, aimd 16.5 RPS, FC 16.0 RPS (vs 18.4 baseline). The aimd/FC throughput reduction reflects perEndpoint=20 scheduling overhead. At r=4, flow-control shows the highest throughput (21.6 RPS vs 20.8 baseline).

![TTFT p99 during burst — gpt-oss-120b](analysis/rhoai-3.5ea2-full/ttft_p99_burst_gpt-oss-120b.png)
![Latency vs replicas — gpt-oss-120b](analysis/rhoai-3.5ea2-full/latency_vs_replicas_gpt-oss-120b.png)
![Throughput during burst — gpt-oss-120b](analysis/rhoai-3.5ea2-full/throughput_burst_gpt-oss-120b.png)
![Idle vs burst comparison — gpt-oss-120b](analysis/rhoai-3.5ea2-full/idle_vs_burst_gpt-oss-120b.png)
![Error rate — gpt-oss-120b](analysis/rhoai-3.5ea2-full/error_rate_gpt-oss-120b.png)

## Scenario 5: AIMD Low Concurrency Investigation

FP8-70B r=1 with perEndpoint=5, global=20. AIMD enabled with min=1.

| Metric | burst-2 | burst-3 |
|---|---|---|
| TTFT p99 (ms) | 605.6 | 333.7 |
| RPS | 6.2 | 6.5 |
| Batch jobs completed | 17 of 30 | — |

AIMD concurrency limit: flat at 5 throughout. `model_inflight_requests`: flat at 5.

This confirms the cross-team finding from llm-d/llm-d-batch-gateway#491: vLLM continuous batching queues excess requests internally rather than returning 429/5xx. Without rejection signals, AIMD has no feedback mechanism and functions as a static rate limiter at the configured perEndpoint value. This is not specific to this setup — the same behavior is observed across all configurations tested by multiple teams.

With perEndpoint=5 (vs 20 in scenario 3), batch dispatch pressure is lower and interactive latency improves: TTFT p99 of 333.7-605.6 ms vs 809.7 ms (scenario 4, FC) and 1217.7 ms (scenario 3, aimd) at r=1. The tradeoff is batch throughput: 17 of 30 jobs complete (vs 4-5 for scenarios 3/4 with perEndpoint=20, and 30 for ungated with perEndpoint=100).

## Batch Processor Metrics

EA2 batch processor metrics provide visibility into batch job lifecycle across scenarios.

### Batch Job Completion and Latency

| Model | Scenario | R | Jobs Completed | Mean E2E (s) |
|---|---|---|---|---|
| Qwen3-8B | ungated | 1 | 30 | 67 |
| Qwen3-8B | aimd | 1 | 30 | 372 |
| Qwen3-8B | aimd-flow-control | 1 | 30 | 386 |
| FP8-70B | ungated | 1 | 30 | 365 |
| FP8-70B | aimd | 1 | 5 | — |
| FP8-70B | aimd-flow-control | 1 | 4 | — |
| FP8-70B | aimd-low-concurrency (s5) | 1 | 17 | — |
| gpt-oss-120b | ungated | 1 | 30 | 171 |
| gpt-oss-120b | aimd | 1 | 30 | 345 |
| gpt-oss-120b | aimd-flow-control | 1 | 30 | — |

Observations:

- Ungated processes batch 5-6x faster than aimd/FC (more concurrent requests: 100 vs 20).
- FP8-70B aimd/FC complete only 4-5 of 30 jobs during the run window (constrained model, low concurrency). Scenario 5 (perEndpoint=5) completes 17 — tighter limits reduce per-request interference but still limit throughput.
- Qwen3-8B completes all 30 jobs in every scenario. The model has sufficient capacity to handle batch at any concurrency level.
- gpt-oss-120b completes all 30 jobs. Ungated: 171s mean vs aimd: 345s mean (2x slower with perEndpoint=20).

## Cross-Model Comparison

| Metric | Qwen3-8B r=1 | FP8-70B r=1 | gpt-oss-120b r=1 |
|---|---|---|---|
| Total parameters | 8B | 70B | 120B |
| Active parameters/token | 8B (dense) | 70B (dense) | ~13B (MoE, top-2/64) |
| Quantization | BF16 | FP8 | MXFP4 |
| GPUs per replica (TP) | 1 | 2 | 4 |
| KV cache tokens | 783,568 | 568,528 | 5,979,728 |
| Max concurrency at max_seq_len | 19.13x | 4.34x | 45.62x |
| Baseline TTFT p99 | 93.3 ms | 400.3 ms | 150.3 ms |
| Baseline RPS | 18.8 | 8.2 | 18.4 |
| Batch overhead (ungated TTFT p99) | +32.0% | +136.6% | -2.5% |
| Batch overhead (FC TTFT p99) | +176.3% | +102.3% | +1.8% |
| Throughput reduction under batch (ungated) | +2.7% (to 19.3) | -51.2% (to 4.0) | +8.2% (to 19.9) |
| Throughput with flow-control | 12.4 RPS | 4.1 RPS | 16.0 RPS |
| Best scenario at r=1 | ungated | aimd-flow-control | ungated / FC (within noise) |
| Best scenario at max replicas | all within noise (r=8) | aimd-flow-control (r=8) | varies (high tail variability) |
| Replicas to absorb batch (within noise) | 4 | >8 | 2 (ungated) |

With `perEndpoint` enforced at 20 for aimd/FC and 100 for ungated, the batch throughput vs interactive latency tradeoff is controlled by concurrency limits:

- **Ungated (100 concurrent):** Higher batch throughput, batch completes faster, shorter contention window. Optimal for models with spare capacity (Qwen3-8B, gpt-oss-120b).
- **Aimd/FC (20 concurrent):** Lower instantaneous batch pressure with priority scheduling. FC provides measurable benefit for capacity-constrained models (FP8-70B) where priority dispatch ordering reduces interactive tail latency.
- **Scenario 5 (5 concurrent):** Lowest batch pressure, best interactive latency under constrained conditions, but batch throughput is further reduced.

## Observations

1. **Flow-control priority dispatch ordering provides measurable benefit for FP8-70B.** At r=1, FC reduces TTFT p99 overhead from +137% (ungated) and +204% (aimd) to +102%. At r=8, FC shows +215% vs ungated +274% and aimd +270%. Interactive requests are dispatched before batch requests at every scheduling tick.

2. **AIMD is non-functional across all scenarios including scenario 5.** `batch_processor_aimd_concurrency_limit` stays flat at the configured perEndpoint value (20 or 5) throughout all runs. vLLM continuous batching queues excess requests internally rather than returning 429/5xx. Without rejection signals, AIMD has no feedback mechanism. This is confirmed by multiple teams (llm-d/llm-d-batch-gateway#491).

3. **perEndpoint is enforced.** The `model_inflight_requests` metric confirms aimd/FC dispatch at the configured perEndpoint value (20 for scenarios 3/4, 5 for scenario 5). Ungated dispatches at 100. This is consistent with the EA2 batch gateway behavior observed in the previous evaluation.

4. **EA2 EPP utilization-detector is auto-registered but saturation-based shedding is not observed.** The utilization-detector plugin (queueDepthThreshold=5, kvCacheUtilThreshold=0.8) is registered in the EA2 EPP, but without a rejection path from vLLM, saturation-based shedding cannot trigger. The plugin detects saturation conditions but has no mechanism to shed load.

5. **Ungated maximizes batch throughput at cost of interactive latency.** Ungated completes 30 batch jobs 5-6x faster than aimd/FC (67s vs 372-386s for Qwen3-8B at r=1; 365s vs incomplete for FP8-70B). For models with spare capacity, batch completes before contention accumulates.

6. **Qwen3-8B: ungated is the optimal scenario.** The model has ample capacity (19.13x concurrent at max_seq_len). Ungated adds +32% overhead at r=1 and is within noise at r=4/r=8. Aimd/FC scheduling overhead (+135-176% at r=1) exceeds the batch contention it prevents.

7. **Batch throughput vs interactive latency is a tradeoff controlled by concurrency limits.** Lower perEndpoint values reduce interactive latency impact but extend batch completion time. Scenario 5 (perEndpoint=5) shows this: TTFT p99 of 334-606 ms (vs 810 ms for FC at perEndpoint=20) but only 17 of 30 jobs complete.

8. **FP8-70B baseline variability continues across test sessions.** Interactive-only TTFT p99 at r=1 is 400.3 ms in this evaluation, vs 471.2 ms in EA1+EA2 batch and 1897.9 ms in EA1. The EPP and vLLM versions differ between EA1 and EA2 but cluster thermal state and scheduling conditions also contribute. All overhead comparisons use within-session baselines.

9. **priority-holdback-policy (PR #1592, merged 2026-07-10) would enable graduated shedding.** This feature is not in the EA2 build. It would allow the EPP to hold back low-priority (batch) requests under saturation, providing a shedding mechanism that does not depend on vLLM returning 429/5xx.

## Known Limitations

| Feature | Status | Impact |
|---|---|---|
| AIMD adaptation | No adaptation observed | AIMD concurrency limit stays at the configured perEndpoint value (20 or 5). vLLM never returns 429/5xx. AIMD functions as a static rate limiter in all tested scenarios including scenario 5. |
| utilization-detector | Registered but no rejection path | Saturation-based shedding requires the EPP to reject or hold back requests. Without a rejection mechanism from vLLM, the plugin detects saturation but cannot act on it. |
| priority-holdback-policy | Not in EA2 build | PR #1592 merged 2026-07-10 but not included in the EA2 EPP binary used in this evaluation. Would provide graduated shedding without depending on vLLM 429/5xx. |
| FP8-70B baseline variability | 400-1898 ms TTFT p99 across sessions | Interactive-only TTFT p99 varies across test sessions. Overhead comparisons are valid within each evaluation but not directly comparable across evaluations. |
| Scenario 5 | FP8-70B r=1 only | Only tested for one model at one replica count. Additional model/replica combinations would provide more data on the perEndpoint=5 tradeoff. |

## Next Steps

1. **Test with priority-holdback-policy.** When a build includes PR #1592, evaluate whether graduated shedding improves interactive latency protection without depending on vLLM 429/5xx signals.
2. **Investigate vLLM `--max-waiting-queue-length` RFC for 429 generation.** A vLLM-side mechanism to return 429 when the waiting queue exceeds a threshold would provide the feedback signal AIMD requires. This would enable AIMD to function as designed rather than as a static rate limiter.
3. **Consider concurrency-detector with tuned maxConcurrency per model.** Instead of a single perEndpoint value for all models, per-model concurrency limits based on KV cache capacity and TP configuration could provide model-appropriate batch pressure control.
