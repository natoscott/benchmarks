# llm-d Batch Gateway Performance Evaluation — RHOAI 3.5 EA1

**Scope:** 45 runs (36 initial + 9 flow-control retest) using RHOAI 3.5 EA1 batch gateway images with EA1 EPP.

## TL;DR

Three models were evaluated on 2×8 NVIDIA H200 GPUs with RHOAI 3.5 EA batch gateway dispatch strategies against an interactive-only baseline. **EPP flow-control (priority scheduling) protects interactive latency for large dense models but adds harmful overhead for small models.** With flow-control enabled, FP8-70B TTFT p99 improves from +107% overhead to -49% at r=1 and -91% at r=4, with throughput recovering from 0.2 to 3.8-8.4 RPS. gpt-oss-120b (MoE) improves from +43% to -39%. However, Qwen3-8B overhead increases from 5-8% to 58-91% — the scheduling layer adds more latency than the batch contention it prevents. The batch processor ignores per-endpoint concurrency limits and AIMD configuration in this build (all scenarios dispatch 100 concurrent requests regardless of config), so the EPP is the only effective isolation mechanism.

## Methodology

Common infrastructure, workload, and scenario definitions are in [REPORT.md](REPORT.md).

### EA1-Specific Configuration

| Component | Configuration |
|---|---|
| Batch Gateway | RHOAI 3.5 EA1 images (registry.redhat.io, digest-pinned) |
| EPP | RHOAI 3.5 EA1 (`@v0.0.0-20260409231514-905fb67a04d5`) |
| GC | Disabled (health probe fails in EA1) |
| `perEndpoint` enforcement | Not enforced — all scenarios dispatch at global limit |

## Results

### Qwen3-8B

Qwen3-8B uses TP=1 (1 GPU per replica), max_model_len=40,960, KV cache 783,568 tokens (max 19.13x concurrent at max_model_len).

#### Interactive Latency During Burst

All overhead measurements are from burst phases (15 concurrent interactive streams), where batch contention is highest. Idle phase data (1 stream) is available in the idle-vs-burst charts. All values in milliseconds. Lower is better.

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 32.2 | 45.1 | 86.9 | 5.9 | 5.9 | 6.1 | 19.0 | 2282 |
| ungated | 1 | 34.2 | 52.9 | 91.9 | 5.9 | 6.4 | 6.8 | 18.7 | 2248 |
| aimd | 1 | 30.8 | 62.5 | 93.9 | 5.9 | 6.4 | 6.7 | 18.7 | 2247 |
| aimd-flow-control | 1 | 36.5 | 114.0 | 137.3 | 5.9 | 5.9 | 6.7 | 18.7 | 2254 |
| interactive-only | 4 | 31.6 | 44.1 | 78.4 | 5.7 | 5.9 | 6.3 | 19.5 | 2338 |
| ungated | 4 | 29.1 | 38.5 | 54.9 | 5.9 | 6.2 | 6.3 | 19.2 | 2300 |
| aimd | 4 | 29.0 | 41.4 | 58.0 | 5.9 | 6.2 | 6.2 | 19.1 | 2297 |
| aimd-flow-control | 4 | 33.0 | 89.4 | 123.7 | 5.8 | 14.9 | 15.4 | 14.4 | 1730 |
| interactive-only | 8 | 29.2 | 37.3 | 48.0 | 5.8 | 5.9 | 6.0 | 19.6 | 2354 |
| ungated | 8 | 28.3 | 36.7 | 49.3 | 5.7 | 6.0 | 6.1 | 20.0 | 2396 |
| aimd | 8 | 28.4 | 37.0 | 42.9 | 5.6 | 6.0 | 6.1 | 20.0 | 2398 |
| aimd-flow-control | 8 | 32.3 | 83.7 | 91.8 | 5.7 | 13.7 | 14.3 | 17.3 | 2078 |

#### TTFT p99 Overhead vs Baseline

Percentage change from interactive-only at each replica count.

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | +5.8% | -30.0% | +2.7% |
| aimd | +8.1% | -26.0% | -10.6% |
| aimd-flow-control | +57.9% | +57.7% | +91.1% |

Without flow-control (ungated, aimd), batch adds 5-8% to TTFT p99 at r=1, within noise at r=4/r=8. With flow-control enabled, TTFT p99 overhead increases to 58-91% across all replica counts, and throughput drops (14.4 RPS at r=4 vs 19.1 without). ITL p95 also increases (14.9 ms vs 5.9 ms at r=4). The flow-control scheduling layer adds latency that exceeds the batch contention it prevents — Qwen3-8B handles batch load with minimal impact, so the priority scheduling overhead is a net negative.

![TTFT p99 during burst — Qwen3-8B](analysis/rhoai-3.5ea1/ttft_p99_burst_Qwen3-8B.png)
![Latency vs replicas — Qwen3-8B](analysis/rhoai-3.5ea1/latency_vs_replicas_Qwen3-8B.png)
![Throughput during burst — Qwen3-8B](analysis/rhoai-3.5ea1/throughput_burst_Qwen3-8B.png)
![Idle vs burst comparison — Qwen3-8B](analysis/rhoai-3.5ea1/idle_vs_burst_Qwen3-8B.png)

#### Throughput

Interactive throughput (requests/sec) is stable across ungated and aimd scenarios: 18.7-20.0 RPS. With flow-control enabled, throughput drops to 14.4-18.7 RPS due to scheduling overhead.

#### Error Rates

Zero inference errors across all 45 runs (36 initial + 9 flow-control retest). Four incomplete requests total across all runs.

#### Batch Processing

30 batch jobs × 100 requests each complete within 60-90 seconds of submission. The batch processor dispatches 30-40 concurrent inference requests (per `processor_inflight_requests` metric). All 3000 batch requests complete with 0 failures.

Time-series metrics from the ungated r=4 run show batch dispatch starting at t+20s and completing by t+300s, with 14-18 vLLM requests running concurrently during burst phases.

#### System Metrics

**GPU utilization**: Active GPUs reach 99-100% during burst phases with concurrent batch load (ungated r=4: GPUs 0,1,5,7 at 99-100%). Without batch load, GPU utilization matches the interactive-only pattern.

**vLLM request queue**: During concurrent batch + interactive load, vLLM reports 5-18 running requests per sample (10s intervals). During interactive-only, running requests track the concurrent stream count (1 during idle, 15 during burst).

**Batch processor inflight**: The processor maintains 20-41 concurrent inference requests during active dispatch (ungated scenario, global limit=200). Inflight drops to 0 once batch jobs complete.

![vLLM running requests vs batch inflight — Qwen3-8B ungated r=4](analysis/rhoai-3.5ea1/pcp_concurrent_load_Qwen3-8B_ungated_r4.png)
![vLLM and batch processor time series by scenario — Qwen3-8B r=4](analysis/rhoai-3.5ea1/pcp_timeseries_Qwen3-8B_r4.png)
![KV cache utilization and queue depth — Qwen3-8B r=4](analysis/rhoai-3.5ea1/kv_cache_and_queue_Qwen3-8B_r4.png)
![Batch completion timeline — Qwen3-8B](analysis/rhoai-3.5ea1/batch_timeline_Qwen3-8B.png)

### Meta-Llama-3.1-70B-Instruct-FP8

FP8-70B uses TP=2 (2 GPUs per replica), max_seq_len=131,072, KV cache 568,528 tokens (max 4.34x concurrent at max_seq_len).

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 135.1 | 1884.3 | 1897.9 | 47.4 | 51.4 | 61.6 | 2.4 | 300 |
| ungated | 1 | 3914.0 | 3925.5 | 3925.7 | 388.5 | 388.5 | 388.5 | 0.2 | 60 |
| aimd | 1 | 2517.5 | 2524.3 | 2524.7 | 379.9 | 399.3 | 399.3 | 0.2 | 58 |
| aimd-flow-control | 1 | 671.2 | 866.8 | 961.4 | 41.1 | 48.8 | 49.4 | 3.8 | 454 |
| interactive-only | 4 | 97.0 | 1777.8 | 1789.0 | 14.2 | 639.1 | 639.1 | 0.7 | 94 |
| ungated | 4 | 106.0 | 2599.4 | 2609.7 | 14.9 | 525.6 | 525.6 | 0.7 | 91 |
| aimd | 4 | 104.0 | 1789.7 | 1800.3 | 13.8 | 424.2 | 435.7 | 0.9 | 114 |
| aimd-flow-control | 4 | 90.7 | 138.8 | 154.7 | 13.3 | 13.9 | 14.2 | 8.4 | 1008 |
| interactive-only | 8 | 90.9 | 442.1 | 551.2 | 12.7 | 50.9 | 55.5 | 6.5 | 786 |
| ungated | 8 | 98.1 | 334.4 | 1129.4 | 13.6 | 38.1 | 274.0 | 3.7 | 450 |
| aimd | 8 | 96.9 | 919.4 | 1981.6 | 13.4 | 195.5 | 217.8 | 4.4 | 529 |
| aimd-flow-control | 8 | 100.9 | 2215.1 | 3755.0 | 14.1 | 382.4 | 391.9 | 1.7 | 206 |

**TTFT p99 overhead vs baseline:**

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | +106.8% | +45.9% | +104.9% |
| aimd | +33.0% | +0.6% | +259.5% |
| aimd-flow-control | -49.3% | -91.4% | +581.3% |

Without flow-control (ungated, aimd), FP8-70B shows 33-107% TTFT p99 overhead at r=1 with throughput collapsing from 2.4 to 0.2 RPS (12x).

With flow-control enabled, r=1 and r=4 show substantial improvement: TTFT p99 drops to 961 ms (-49%) at r=1 and 155 ms (-91%) at r=4, with throughput recovering to 3.8 and 8.4 RPS respectively. The EPP deprioritizes batch requests (priority -1) relative to interactive (default priority 0), preventing the batch processor from monopolizing compute.

The r=8 result (3755 ms, +581%) is anomalous and requires investigation — it may reflect an interaction between flow-control scheduling and the larger number of replicas/EPP routing decisions.

![TTFT p99 during burst — FP8-70B](analysis/rhoai-3.5ea1/ttft_p99_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Latency vs replicas — FP8-70B](analysis/rhoai-3.5ea1/latency_vs_replicas_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Throughput during burst — FP8-70B](analysis/rhoai-3.5ea1/throughput_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Idle vs burst comparison — FP8-70B](analysis/rhoai-3.5ea1/idle_vs_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)

### FP8-70B System Metrics

Time-series metrics from the ungated r=1 run reveal the contention mechanism:

- **vLLM running requests**: Ramps from 0 to 100 within 90 seconds as the batch processor fills its concurrency limit. The single TP=2 replica attempts to serve 100 concurrent requests.
- **KV cache utilization**: Climbs from 0% to 35% — 10x higher than Qwen3-8B under the same workload. The 568,528-token KV cache is not exhausted, but the model's effective throughput at 100 concurrent requests is bottlenecked by compute, not memory.
- **Batch processor inflight**: Saturates at 100 requests (the global concurrency limit), confirming the processor dispatches at maximum concurrency regardless of backend capacity.

At r=8, the 100 batch requests are distributed across 8 replicas (~12-13 per replica), reducing per-replica pressure. However, total GPU compute is still shared with 15 interactive streams, resulting in elevated tail latencies.

![FP8-70B running requests and KV cache: r=1 vs r=8](analysis/rhoai-3.5ea1/pcp_fp8_70b_r1_vs_r8.png)

### gpt-oss-120b

gpt-oss-120b is a Mixture-of-Experts (MoE) model with 120B total parameters but ~13B active per token (top-2 routing across 64 experts). It uses MXFP4 quantization, TP=4 (4 GPUs per replica), max_seq_len=131,072, KV cache 5,979,728 tokens (max 45.62x concurrent at max_seq_len). Replica configs: r=1 (4 GPUs), r=2 (8 GPUs), r=4 (16 GPUs).

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 64.6 | 238.6 | 347.7 | 8.9 | 11.7 | 21.5 | 11.7 | 708 |
| ungated | 1 | 207.0 | 487.2 | 496.3 | 39.9 | 55.5 | 55.7 | 2.6 | 313 |
| aimd | 1 | 189.6 | 214.2 | 237.4 | 30.8 | 32.9 | 32.9 | 3.5 | 428 |
| aimd-flow-control | 1 | 76.3 | 147.3 | 211.3 | 6.1 | 7.2 | 7.5 | 17.7 | 2129 |
| interactive-only | 2 | 49.9 | 121.0 | 151.5 | 5.7 | 14.7 | 15.2 | 13.5 | 1626 |
| ungated | 2 | 54.0 | 87.0 | 120.7 | 4.9 | 13.9 | 15.4 | 13.9 | 1668 |
| aimd | 2 | 59.4 | 101.9 | 131.7 | 8.2 | 15.5 | 18.4 | 13.0 | 1560 |
| aimd-flow-control | 2 | 38.0 | 53.9 | 73.6 | 5.0 | 5.4 | 5.6 | 21.8 | 2619 |
| interactive-only | 4 | 40.8 | 80.0 | 140.3 | 4.8 | 8.2 | 9.2 | 21.2 | 2548 |
| ungated | 4 | 35.8 | 49.0 | 77.8 | 4.6 | 6.3 | 7.7 | 22.3 | 2675 |
| aimd | 4 | 36.9 | 67.7 | 72.3 | 4.8 | 10.2 | 10.5 | 18.8 | 2252 |
| aimd-flow-control | 4 | 37.7 | 86.3 | 122.9 | 4.5 | 6.2 | 6.5 | 23.2 | 2785 |

**TTFT p99 overhead vs baseline:**

| Scenario | r=1 | r=2 | r=4 |
|---|---|---|---|
| ungated | +42.8% | -20.3% | -44.5% |
| aimd | -31.7% | -13.0% | -48.4% |
| aimd-flow-control | -39.2% | -51.4% | -12.4% |

Without flow-control (ungated, aimd), throughput drops from 11.7 to 2.6-3.5 RPS at r=1 — a 3.3-4.5x reduction. TTFT p99 overhead varies across scenarios, reflecting run-to-run variance with the saturated model.

With flow-control enabled, gpt-oss-120b shows improvement at all replica counts. At r=1, TTFT p99 drops from 347.7 ms (baseline) to 211.3 ms (-39%), and throughput increases to 17.7 RPS — higher than the 11.7 baseline because the EPP deprioritizes batch requests, giving interactive traffic near-uncontested GPU access. At r=2, TTFT p99 drops to 73.6 ms (-51%) with 21.8 RPS. At r=4, overhead is within noise (-12%).

![TTFT p99 during burst — gpt-oss-120b](analysis/rhoai-3.5ea1/ttft_p99_burst_gpt-oss-120b.png)
![Latency vs replicas — gpt-oss-120b](analysis/rhoai-3.5ea1/latency_vs_replicas_gpt-oss-120b.png)
![Throughput during burst — gpt-oss-120b](analysis/rhoai-3.5ea1/throughput_burst_gpt-oss-120b.png)
![Idle vs burst comparison — gpt-oss-120b](analysis/rhoai-3.5ea1/idle_vs_burst_gpt-oss-120b.png)

### gpt-oss-120b System Metrics

Time-series metrics from the ungated r=1 run:

- **vLLM running requests**: Ramps from 0 to 100 within 90 seconds, then fluctuates between 84-130. Spikes above 100 occur when 15 interactive burst streams overlap with 100 batch requests on the single TP=4 replica.
- **KV cache utilization**: Peaks at 2.2% — compared to 3.6% for Qwen3-8B and 35% for FP8-70B under the same batch load. The MoE architecture's attention layers produce the same per-token KV volume as a dense model, but the 5.98M-token KV cache (10.5x larger than FP8-70B's 568K tokens) provides proportionally more headroom.
- **Batch processor inflight**: Saturates at 100 requests (the global concurrency limit), identical to the other models.

At r=4, batch processing completes within ~6 minutes of submission. Metrics show batch inflight dropping from 100 to 0 by t+360s, after which only interactive traffic remains (3-6 running requests per sample). This fast batch drain explains the absence of batch overhead at r=4.

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
| Batch overhead (flow-control TTFT p99) | +57.9% | -49.3% | -39.2% |
| Throughput reduction under batch | 1.6% | 91.7% | 77.8% |
| Throughput with flow-control | 18.7 RPS | 3.8 RPS | 17.7 RPS |
| KV cache under batch load | 3.6% | 35% | 2.2% |
| Replicas to absorb batch (no FC) | 4 | >8 | 2 |

Flow-control effectiveness is inversely related to how well the model handles batch load without it. FP8-70B benefits most (-49% TTFT p99 at r=1) because batch contention is severe without isolation. gpt-oss-120b benefits moderately (-39%). Qwen3-8B is harmed (+58%) because the flow-control scheduling overhead exceeds the minimal batch contention the model experiences.

KV cache is not the bottleneck for any model on H200. FP8-70B reaches 35% under batch load but does not exhaust; Qwen3-8B and gpt-oss-120b stay below 4%.

vLLM prefix cache hit rate is ~50% during batch processing. Batch requests share 32 system prompts across 3000 requests, enabling prefix cache reuse. The EPP's Valkey-backed prefix cache indexer routes requests to replicas that already have the relevant prefix cached.

![Cross-model TTFT p99 comparison at r=1 (log scale)](analysis/rhoai-3.5ea1/cross_model_ttft_p99_r1.png)

## Observations

1. **EPP flow-control protects interactive latency for large dense models.** With the `flowControl` feature gate enabled, FP8-70B TTFT p99 drops from 1898 ms (baseline) to 961 ms (-49%) at r=1 and from 1789 ms to 155 ms (-91%) at r=4. Throughput recovers from 0.2 to 3.8 RPS (r=1) and 0.7 to 8.4 RPS (r=4). Without flow-control, the same scenarios show +33-107% overhead. The EPP's priority scheduling (interactive priority 0, batch priority -1) prevents batch requests from monopolizing compute on capacity-constrained models.

2. **Flow-control adds overhead that harms small models.** Qwen3-8B TTFT p99 overhead increases from 5-8% (without flow-control) to 58-91% (with flow-control) across all replica counts. Throughput drops from 19-20 to 14-18 RPS, and ITL p95 increases from 6 ms to 15 ms at r=4/r=8. The flow-control scheduling layer adds latency that exceeds the batch contention it prevents — Qwen3-8B handles batch load with minimal impact, so priority scheduling is a net negative.

3. **Flow-control effectiveness is inversely related to batch contention severity.** FP8-70B (high contention without FC) benefits most: -49% to -91% TTFT p99. gpt-oss-120b (moderate contention) benefits: -39% to -51%. Qwen3-8B (minimal contention) is harmed: +58% to +91%. This suggests flow-control should be selectively enabled based on model capacity relative to batch load, not applied uniformly.

4. **The batch processor ignores per-endpoint concurrency limits and AIMD configuration.** Despite configuring `perEndpoint=20` for aimd/aimd-flow-control (vs `perEndpoint=100` for ungated), `model_inflight_requests` shows all scenarios saturating at 100 concurrent requests — the global limit. The upstream source code implements dual semaphore acquisition (per-endpoint before global), so this may be a RHOAI build-specific issue or a config field mapping difference.

5. **Without flow-control, all dispatch strategies produce equivalent batch load.** Scenarios 2 (ungated) and 3 (aimd) dispatch identically because per-endpoint limits are ignored (observation 4). TTFT p99 variation between them at r=1 is consistent with run-to-run variance.

6. **The batch processor saturates dense models.** On a single FP8-70B replica (4.34x effective concurrency at max_seq_len), 100 concurrent batch requests drive vLLM running requests to 100 and KV cache to 35%. On gpt-oss-120b (45.62x effective concurrency), the same 100 concurrent requests produce only 2.2% KV cache usage.

7. **FP8-70B r=8 flow-control result is anomalous.** TTFT p99 of 3755 ms (+581%) at r=8 with flow-control contradicts the r=1 and r=4 improvements. This may reflect an interaction between flow-control scheduling and EPP routing across 8 replicas, or a transient issue during the test run. Requires investigation.

8. **FP8-70B baseline is already saturated at r=1 with 15 interactive streams.** The interactive-only baseline shows 1.9s TTFT p99 and 2.4 RPS at r=1 — the model cannot sustain 15 concurrent streams on a single TP=2 replica.

9. **30×100 batch job config enables concurrent measurement.** Batch dispatch starts within 20s of submission and overlaps with interactive traffic throughout the measurement window.

## Known Limitations (RHOAI 3.5 EA)

| Feature | Status | Impact |
|---|---|---|
| EPP `utilization-detector` plugin | Not registered | Saturation-based shedding requires this plugin to detect when queue depth or KV cache exceeds thresholds. Not compiled into this EPP build (`@v0.0.0-20260409231514-905fb67a04d5`). Flow-control operates without it — priority scheduling works but active shedding under saturation does not. |
| Processor per-endpoint limits | Ignored | `perEndpoint` concurrency config has no effect. `model_inflight_requests` shows all scenarios saturating at the global limit (100), regardless of `perEndpoint=20` vs `perEndpoint=100`. Upstream source implements dual semaphore acquisition correctly — may be a RHOAI build-specific issue. |
| Processor AIMD | No observable effect | AIMD enabled in scenarios 3 and 4 but no concurrency reduction observed. Inflight stays at global limit throughout. AIMD metrics not exposed. |
| Job-level processor metrics | Not exposed | No job duration, token throughput, or per-model inflight metrics |
| Batch gateway GC | Health probe fails | Disabled; manual state cleanup between runs |

## Next Steps

1. **Investigate FP8-70B r=8 flow-control anomaly**: TTFT p99 of 3755 ms (+581%) contradicts r=1 and r=4 improvements. Check time-series metrics for EPP routing behavior and batch processor inflight at r=8.
2. **Processor per-endpoint investigation**: Confirm whether `perEndpoint` enforcement is a RHOAI build issue or config field mapping difference.
3. **RHOAI EA2**: Verify `utilization-detector` plugin availability (saturation-based shedding), AIMD metrics exposure, and job-level processor metrics.
