# llm-d Batch Gateway Performance Evaluation — RHOAI 3.5 EA1 EPP + EA2 Batch Gateway

**Scope:** 36 runs using RHOAI 3.5 EA2 batch gateway images (apiserver, processor, GC) with the EA1 EPP (unchanged).

## TL;DR

Three models were evaluated on 2x8 NVIDIA H200 GPUs with RHOAI 3.5 EA2 batch gateway dispatch strategies against an interactive-only baseline. **EA2 enforces per-endpoint concurrency limits (aimd/FC dispatch at 20, not 100), but the reduced batch pressure does not improve interactive latency — ungated (dispatching at 100) remains the best-performing scenario for most models because the EPP already handles contention.** With `perEndpoint` now enforced at 20, aimd and flow-control scenarios show lower batch throughput without a corresponding interactive latency benefit. Qwen3-8B at r=4 with aimd achieves -50.9% TTFT p99 overhead (52.0 ms vs 105.9 ms baseline) — the best interactive latency result across all models. FP8-70B baselines improved from 1898 ms to 471 ms TTFT p99 at r=1 compared to EA1, reflecting cluster conditions rather than software changes. gpt-oss-120b ungated r=4 burst data is missing and excluded from analysis.

## Methodology

Common infrastructure, workload, and scenario definitions are in [REPORT.md](REPORT.md).

### EA2-Specific Configuration

| Component | Configuration |
|---|---|
| Batch Gateway | RHOAI 3.5 EA2 images (quay.io, tag `rhoai-3.5-ea.2`) |
| EPP | RHOAI 3.5 EA1 binary (unchanged, `@v0.0.0-20260409231514-905fb67a04d5`) |
| GC | Enabled (health probe fixed in EA2) |
| `perEndpoint` enforcement | Enforced — aimd/FC dispatch at 20 concurrent (verified by `model_inflight_requests`) |

### EA2 Changes from EA1

| Change | Detail |
|---|---|
| `perEndpoint` enforcement | aimd/FC dispatch at 20 concurrent. Ungated remains at 100. EA1 ignored this setting entirely. |
| AIMD concurrency limit metric | `batch_processor_aimd_concurrency_limit` exposed and stays at 20. No adaptation observed (gateway never returns 429/5xx). |
| GC functional | EA1 GC health probe failed; EA2 GC runs 30-min reconciliation cycles. |
| New processor metrics | `batch_job_e2e_latency_seconds`, `job_processing_duration_seconds`, `job_queue_wait_duration_seconds`, `batch_request_prompt_tokens_total`, `batch_request_generation_tokens_total`, `model_request_execution_duration_seconds`, `processor_max_inflight_concurrency`, `plan_build_duration_seconds`, `file_storage_operations_total` |
| EPP | Unchanged EA1 binary. Flow-control feature gate enabled for scenario 4. |

## Results

### Qwen3-8B

Qwen3-8B uses TP=1 (1 GPU per replica), max_model_len=40,960, KV cache 783,568 tokens (max 19.13x concurrent at max_model_len).

#### Interactive Latency During Burst

All overhead measurements are from burst phases (15 concurrent interactive streams), where batch contention is highest. All values in milliseconds. Lower is better.

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 33.0 | 76.1 | 98.3 | 5.9 | 6.3 | 6.8 | 18.7 | 2248 |
| ungated | 1 | 31.8 | 68.1 | 87.2 | 5.7 | 6.1 | 6.4 | 19.4 | 2331 |
| aimd | 1 | 133.1 | 188.3 | 221.8 | 8.4 | 11.6 | 12.6 | 12.1 | 1456 |
| aimd-flow-control | 1 | 161.8 | 206.6 | 261.8 | 8.1 | 11.2 | 11.7 | 12.6 | 1516 |
| interactive-only | 4 | 31.4 | 86.1 | 105.9 | 5.9 | 14.8 | 15.5 | 14.8 | 1774 |
| ungated | 4 | 32.4 | 88.6 | 109.3 | 5.9 | 14.8 | 15.4 | 14.5 | 1746 |
| aimd | 4 | 31.7 | 46.0 | 52.0 | 5.9 | 6.0 | 6.1 | 19.1 | 2296 |
| aimd-flow-control | 4 | 32.1 | 43.6 | 54.0 | 5.9 | 6.1 | 6.2 | 19.0 | 2281 |
| interactive-only | 8 | 30.6 | 60.1 | 90.3 | 5.7 | 12.7 | 13.7 | 17.7 | 2120 |
| ungated | 8 | 31.7 | 82.6 | 92.6 | 5.6 | 13.2 | 14.1 | 17.4 | 2087 |
| aimd | 8 | 33.1 | 84.9 | 93.1 | 5.6 | 13.6 | 14.5 | 17.2 | 2066 |
| aimd-flow-control | 8 | 32.5 | 85.0 | 92.4 | 5.6 | 14.2 | 14.7 | 17.0 | 2046 |

#### TTFT p99 Overhead vs Baseline

Percentage change from interactive-only at each replica count.

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | -11.3% | +3.3% | +2.5% |
| aimd | +125.5% | -50.9% | +3.1% |
| aimd-flow-control | +166.2% | -49.0% | +2.3% |

At r=1, ungated shows -11.3% overhead (within noise), while aimd and flow-control show +126% and +166% respectively — 221.8 ms and 261.8 ms versus 98.3 ms baseline. ITL increases from 6.8 ms to 12.6 ms and 11.7 ms. Throughput drops from 18.7 to 12.1-12.6 RPS (35% reduction).

At r=4, the pattern reverses: aimd and flow-control produce the best latency results (-50.9% and -49.0%), with TTFT p99 at 52.0-54.0 ms versus 105.9 ms baseline. ITL returns to 6.0-6.2 ms and throughput recovers to 19.0-19.1 RPS. Ungated shows +3.3% (within noise).

At r=8, all scenarios converge within noise (+2-3%). Sufficient replica capacity absorbs both batch and interactive load.

**EA1 comparison:** In EA1, aimd and ungated were functionally identical (both dispatching at 100) and showed 5-8% overhead at r=1. In EA2, enforcing `perEndpoint=20` for aimd causes scheduling overhead at r=1 but enables load isolation at r=4. The flow-control overhead that harmed Qwen3-8B in EA1 (+58-91% across all replica counts) now only manifests at r=1 (+166%); at r=4 and r=8 it is within noise.

![TTFT p99 during burst — Qwen3-8B](analysis/rhoai-3.5ea2/ttft_p99_burst_Qwen3-8B.png)
![Latency vs replicas — Qwen3-8B](analysis/rhoai-3.5ea2/latency_vs_replicas_Qwen3-8B.png)
![Throughput during burst — Qwen3-8B](analysis/rhoai-3.5ea2/throughput_burst_Qwen3-8B.png)
![Idle vs burst comparison — Qwen3-8B](analysis/rhoai-3.5ea2/idle_vs_burst_Qwen3-8B.png)

#### Throughput

Interactive throughput (requests/sec) at r=4 and r=8 is stable across all scenarios: 14.5-19.1 RPS. At r=1, aimd/FC throughput drops to 12.1-12.6 RPS (vs 18.7-19.4 for interactive-only/ungated) due to scheduling overhead with the enforced 20-request dispatch limit.

#### Error Rates

Zero inference errors across all Qwen3-8B runs. All interactive requests completed.

#### Batch Processing

With `perEndpoint` enforced at 20 for aimd/FC scenarios, batch dispatch pressure is reduced compared to EA1. The `model_inflight_requests` metric confirms 20 concurrent batch requests for aimd/FC (vs 100 for ungated). All 3000 batch requests complete with 0 failures.

![Batch inflight comparison: ungated (100) vs aimd (20) — Qwen3-8B r=1](analysis/rhoai-3.5ea2/pcp_inflight_comparison_Qwen3-8B_r1.png)

### Meta-Llama-3.1-70B-Instruct-FP8

FP8-70B uses TP=2 (2 GPUs per replica), max_seq_len=131,072, KV cache 568,528 tokens (max 4.34x concurrent at max_seq_len).

**Baseline note:** The interactive-only TTFT p99 at r=1 is 471.2 ms in this EA2 evaluation versus 1897.9 ms in EA1. This 4x improvement reflects cluster conditions (GPU thermal state, scheduling, co-located workloads) — not EA2 software changes. The EPP binary is identical. All overhead comparisons use the EA2 baselines.

#### Interactive Latency During Burst

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 59.9 | 337.9 | 471.2 | 13.1 | 14.7 | 16.8 | 8.2 | 999 |
| ungated | 1 | 596.7 | 685.3 | 687.0 | 41.4 | 47.9 | 48.3 | 4.0 | 482 |
| aimd | 1 | 660.4 | 1342.7 | 1661.1 | 25.0 | 29.8 | 30.6 | 4.0 | 496 |
| aimd-flow-control | 1 | 476.3 | 659.7 | 1297.0 | 24.6 | 31.4 | 32.7 | 4.1 | 502 |
| interactive-only | 4 | 69.3 | 109.2 | 146.9 | 13.2 | 14.1 | 14.3 | 8.5 | 1019 |
| ungated | 4 | 93.3 | 134.9 | 150.2 | 13.3 | 14.1 | 14.4 | 8.3 | 998 |
| aimd | 4 | 96.1 | 151.5 | 531.2 | 14.3 | 19.4 | 20.7 | 7.1 | 856 |
| aimd-flow-control | 4 | 97.8 | 152.4 | 366.3 | 14.8 | 19.7 | 21.1 | 7.0 | 845 |
| interactive-only | 8 | 63.4 | 350.9 | 520.2 | 12.5 | 40.3 | 47.4 | 7.1 | 856 |
| ungated | 8 | 90.9 | 472.3 | 717.8 | 12.7 | 55.0 | 75.4 | 6.2 | 744 |
| aimd | 8 | 96.5 | 511.4 | 975.3 | 13.1 | 97.9 | 111.8 | 4.8 | 579 |
| aimd-flow-control | 8 | 97.6 | 622.3 | 1929.9 | 13.2 | 99.9 | 123.8 | 4.6 | 551 |

#### TTFT p99 Overhead vs Baseline

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | +45.8% | +2.3% | +38.0% |
| aimd | +252.5% | +261.6% | +87.5% |
| aimd-flow-control | +175.2% | +149.4% | +271.0% |

Ungated shows the least overhead at all replica counts: +45.8% at r=1, +2.3% at r=4, +38.0% at r=8. With 100 concurrent batch requests distributed across replicas, the per-replica load remains manageable when sufficient replicas are available.

Aimd and flow-control show consistently elevated overhead: +88% to +271% across all replica counts. With `perEndpoint` enforced at 20, fewer batch requests compete with interactive traffic per endpoint, but the scheduling overhead (routing decisions, priority evaluation) adds latency that exceeds the isolation benefit on this model.

**EA1 comparison:** In EA1, flow-control dramatically improved FP8-70B latency at r=1 (-49%) and r=4 (-91%) by deprioritizing batch traffic. In EA2, that improvement is absent — flow-control shows +175% at r=1 and +149% at r=4. The key difference is the EA2 baseline: with an interactive-only TTFT p99 of 471 ms (vs 1898 ms in EA1), the model is no longer saturated by interactive load alone, so batch deprioritization provides less benefit. The elevated aimd/FC overhead may also reflect the enforced `perEndpoint=20` adding routing complexity.

Throughput impact is consistent across scenarios: ungated preserves 4.0-8.3 RPS (vs 7.1-8.5 baseline), while aimd/FC reduce throughput to 4.0-7.1 RPS at r=1/r=4 and 4.6-4.8 RPS at r=8 (vs 7.1 baseline).

![TTFT p99 during burst — FP8-70B](analysis/rhoai-3.5ea2/ttft_p99_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Latency vs replicas — FP8-70B](analysis/rhoai-3.5ea2/latency_vs_replicas_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Throughput during burst — FP8-70B](analysis/rhoai-3.5ea2/throughput_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)
![Idle vs burst comparison — FP8-70B](analysis/rhoai-3.5ea2/idle_vs_burst_Meta-Llama-3.1-70B-Instruct-FP8.png)

### gpt-oss-120b

gpt-oss-120b is a Mixture-of-Experts (MoE) model with 120B total parameters but ~13B active per token (top-2 routing across 64 experts). It uses MXFP4 quantization, TP=4 (4 GPUs per replica), max_seq_len=131,072, KV cache 5,979,728 tokens (max 45.62x concurrent at max_seq_len). Replica configs: r=1 (4 GPUs), r=2 (8 GPUs), r=4 (16 GPUs).

**Data note:** The ungated r=4 burst phase data is missing (only idle phase was captured). This scenario is excluded from the r=4 analysis.

#### Interactive Latency During Burst

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 44.7 | 130.3 | 516.3 | 5.6 | 6.8 | 9.9 | 18.5 | 2222 |
| ungated | 1 | 43.4 | 107.9 | 169.7 | 5.3 | 5.9 | 6.8 | 20.3 | 2437 |
| aimd | 1 | 47.5 | 150.1 | 216.8 | 6.9 | 8.5 | 11.2 | 16.1 | 1937 |
| aimd-flow-control | 1 | 43.1 | 49.2 | 89.9 | 5.3 | 5.3 | 5.4 | 20.8 | 2495 |
| interactive-only | 2 | 42.9 | 49.2 | 90.9 | 5.3 | 5.3 | 5.5 | 20.9 | 2504 |
| ungated | 2 | 42.9 | 49.1 | 86.5 | 5.3 | 5.3 | 5.4 | 20.8 | 2495 |
| aimd | 2 | 43.0 | 50.7 | 129.9 | 5.3 | 5.3 | 6.0 | 20.6 | 2478 |
| aimd-flow-control | 2 | 41.5 | 143.1 | 386.7 | 5.6 | 9.9 | 10.5 | 18.9 | 2274 |
| interactive-only | 4 | 42.9 | 49.1 | 84.8 | 5.3 | 5.3 | 5.4 | 20.9 | 2515 |
| aimd | 4 | 43.4 | 49.0 | 87.1 | 5.3 | 5.3 | 5.4 | 20.8 | 2495 |
| aimd-flow-control | 4 | 39.3 | 146.5 | 211.3 | 4.9 | 9.4 | 11.6 | 19.7 | 2370 |

#### TTFT p99 Overhead vs Baseline

| Scenario | r=1 | r=2 | r=4 |
|---|---|---|---|
| ungated | -67.1% | -4.9% | (missing) |
| aimd | -58.0% | +42.9% | +2.7% |
| aimd-flow-control | -82.6% | +325.4% | +149.2% |

gpt-oss-120b shows an inverted pattern compared to FP8-70B. At r=1, all batch scenarios produce lower TTFT p99 than the interactive-only baseline: ungated -67.1% (169.7 ms vs 516.3 ms), aimd -58.0% (216.8 ms vs 516.3 ms), flow-control -82.6% (89.9 ms vs 516.3 ms). The baseline at r=1 has high variance (516.3 ms p99 vs 44.7 ms p50), so the negative overhead may reflect the batch traffic stabilizing vLLM scheduling rather than a genuine improvement.

At r=2, ungated is within noise (-4.9%), aimd shows +42.9% (129.9 ms vs 90.9 ms), and flow-control shows +325.4% (386.7 ms vs 90.9 ms). The flow-control overhead at r=2 is driven by ITL p95 increasing from 5.3 ms to 9.9 ms and TTFT p95 rising from 49.2 ms to 143.1 ms.

At r=4, aimd is within noise (+2.7%), while flow-control shows +149.2% (211.3 ms vs 84.8 ms). Ungated r=4 burst data is missing.

**EA1 comparison:** In EA1, flow-control improved gpt-oss-120b at all replica counts (-12% to -51%). In EA2, flow-control only improves r=1 (-82.6%) and degrades r=2 (+325%) and r=4 (+149%). The MoE model's large KV cache (5.98M tokens) absorbs batch load without contention at r>=2, making the flow-control scheduling overhead a net negative at higher replica counts.

![TTFT p99 during burst — gpt-oss-120b](analysis/rhoai-3.5ea2/ttft_p99_burst_gpt-oss-120b.png)
![Latency vs replicas — gpt-oss-120b](analysis/rhoai-3.5ea2/latency_vs_replicas_gpt-oss-120b.png)
![Throughput during burst — gpt-oss-120b](analysis/rhoai-3.5ea2/throughput_burst_gpt-oss-120b.png)
![Idle vs burst comparison — gpt-oss-120b](analysis/rhoai-3.5ea2/idle_vs_burst_gpt-oss-120b.png)
![Error rate — gpt-oss-120b](analysis/rhoai-3.5ea2/error_rate_gpt-oss-120b.png)

## New EA2 Processor Metrics

EA2 exposes 9 new processor metrics not available in EA1. These provide visibility into batch job lifecycle and processing characteristics.

| Metric | Type | Description |
|---|---|---|
| `batch_job_e2e_latency_seconds` | Histogram | End-to-end latency from job submission to completion |
| `job_processing_duration_seconds` | Histogram | Time spent actively processing a job (inference execution) |
| `job_queue_wait_duration_seconds` | Histogram | Time a job spends waiting in the processor queue before dispatch |
| `batch_request_prompt_tokens_total` | Counter | Total prompt tokens processed across all batch requests |
| `batch_request_generation_tokens_total` | Counter | Total generation tokens produced across all batch requests |
| `model_request_execution_duration_seconds` | Histogram | Per-request inference execution duration at the model endpoint |
| `processor_max_inflight_concurrency` | Gauge | Maximum inflight concurrency the processor will allow |
| `plan_build_duration_seconds` | Histogram | Time to build a dispatch plan for pending batch requests |
| `file_storage_operations_total` | Counter | Total file storage operations (reads/writes for batch input/output) |
| `batch_processor_aimd_concurrency_limit` | Gauge | Current AIMD concurrency limit (stays at 20 throughout — no adaptation observed) |

The `batch_processor_aimd_concurrency_limit` metric confirms that AIMD does not adapt during these runs. The limit remains at 20 (the configured `perEndpoint` value) throughout all aimd/FC scenarios. AIMD adaptation requires the gateway to return 429 or 5xx responses, which does not occur under the tested load levels.

![AIMD concurrency limit — FP8-70B aimd r=1 (flat at 20, no adaptation)](analysis/rhoai-3.5ea2/pcp_aimd_concurrency_FP8-70B_r1.png)

## Cross-Model Comparison

| Metric | Qwen3-8B r=1 | FP8-70B r=1 | gpt-oss-120b r=1 |
|---|---|---|---|
| Total parameters | 8B | 70B | 120B |
| Active parameters/token | 8B (dense) | 70B (dense) | ~13B (MoE, top-2/64) |
| Quantization | BF16 | FP8 | MXFP4 |
| GPUs per replica (TP) | 1 | 2 | 4 |
| KV cache tokens | 783,568 | 568,528 | 5,979,728 |
| Max concurrency at max_seq_len | 19.13x | 4.34x | 45.62x |
| Baseline TTFT p99 | 98.3 ms | 471.2 ms | 516.3 ms |
| Baseline RPS | 18.7 | 8.2 | 18.5 |
| Batch overhead (ungated TTFT p99) | -11.3% | +45.8% | -67.1% |
| Batch overhead (aimd TTFT p99) | +125.5% | +252.5% | -58.0% |
| Batch overhead (FC TTFT p99) | +166.2% | +175.2% | -82.6% |
| Throughput reduction under batch (ungated) | -3.7% (to 19.4) | -51.2% (to 4.0) | +9.7% (to 20.3) |
| Throughput with flow-control | 12.6 RPS | 4.1 RPS | 20.8 RPS |
| Best scenario at r=1 | ungated | ungated | aimd-flow-control |
| Best scenario at max replicas | all within noise (r=8) | ungated (r=4) | aimd (r=4) |

With `perEndpoint` enforced at 20, batch dispatch behavior now differs between scenarios: ungated dispatches 100 concurrent requests while aimd/FC dispatch 20. This creates a tradeoff:

- **Ungated (100 concurrent):** Higher batch throughput, batch completes faster, shorter contention window.
- **Aimd/FC (20 concurrent):** Lower instantaneous batch pressure, but batch takes longer to complete, extending the contention window.

For Qwen3-8B at r=1, the reduced dispatch rate in aimd/FC introduces scheduling overhead without sufficient batch pressure to justify isolation — ungated is better. At r=4, however, the 20-request limit aligns well with per-replica capacity, producing the best latency results.

For FP8-70B, ungated is the least-worst option at all replica counts. The model is sensitive to batch contention, but the aimd/FC scheduling overhead exceeds the benefit of reduced dispatch rate.

For gpt-oss-120b, flow-control at r=1 produces the best result (-82.6% TTFT p99, 20.8 RPS). The MoE model's large KV cache and low per-token compute cost allow it to absorb batch load while the EPP's priority scheduling prevents queue buildup.

## Observations

1. **EA2 enforces `perEndpoint` concurrency limits.** The `model_inflight_requests` metric confirms aimd/FC dispatch at 20 concurrent requests (vs 100 in EA1). Ungated continues to dispatch at 100. This is the primary functional change from EA1 in the batch processor.

2. **Reduced batch dispatch (20 vs 100) does not consistently improve interactive latency.** At r=1 across all models, ungated (100 concurrent) shows equal or lower overhead than aimd/FC (20 concurrent). The scheduling overhead of enforcing per-endpoint limits and priority bands exceeds the benefit of reduced batch pressure for most configurations.

3. **Qwen3-8B aimd/FC overhead at r=1 is 126-166% — but at r=4, aimd achieves -51% improvement.** The enforced 20-request dispatch limit creates scheduling overhead on a single replica but enables effective load isolation across 4 replicas, producing the lowest TTFT p99 (52.0 ms) in the entire evaluation.

4. **FP8-70B baselines differ from EA1 by 4x.** Interactive-only TTFT p99 at r=1 is 471 ms (EA2) vs 1898 ms (EA1). The EPP binary is unchanged; this reflects cluster thermal state and scheduling conditions. All EA2 overhead percentages are relative to the EA2 baselines.

5. **FP8-70B flow-control no longer provides the r=1/r=4 improvements seen in EA1.** In EA1, flow-control reduced TTFT p99 by -49% (r=1) and -91% (r=4). In EA2, it shows +175% (r=1) and +149% (r=4). The improved EA2 baseline (model no longer saturated at baseline) means batch deprioritization provides less benefit, while scheduling overhead remains.

6. **AIMD does not adapt.** `batch_processor_aimd_concurrency_limit` stays at 20 throughout all runs. AIMD requires 429/5xx responses to trigger concurrency reduction, which does not occur. The AIMD mechanism is effectively a static rate limiter at the configured `perEndpoint` value.

7. **gpt-oss-120b flow-control shows a scale inversion.** Flow-control improves r=1 (-82.6%) but degrades r=2 (+325%) and r=4 (+149%). At higher replica counts, the MoE model absorbs batch load without contention, making the EPP priority scheduling overhead a net negative.

8. **gpt-oss-120b ungated r=4 burst data is missing.** Only idle-phase data was captured. This gap prevents full cross-scenario comparison at r=4.

9. **GC is now functional.** EA1 GC health probe failed, requiring manual state cleanup between runs. EA2 GC runs 30-minute reconciliation cycles.

## Known Limitations

| Feature | Status | Impact |
|---|---|---|
| EPP `utilization-detector` plugin | Not registered | Saturation-based shedding requires this plugin. Not compiled into the EA1 EPP build. Flow-control operates without it — priority scheduling works but active shedding under saturation does not. |
| AIMD adaptation | No adaptation observed | AIMD concurrency limit stays at 20 (configured `perEndpoint` value). No 429/5xx responses trigger adaptation. AIMD functions as a static rate limiter in these tests. |
| gpt-oss-120b ungated r=4 | Burst data missing | Only idle-phase data captured; scenario excluded from r=4 analysis. |
| FP8-70B baseline variability | 4x difference vs EA1 | Interactive-only TTFT p99 varies from 471 ms (EA2) to 1898 ms (EA1) across test sessions. Overhead comparisons are valid within each evaluation but not directly comparable across evaluations. |
| EPP binary | EA1 unchanged | The EPP binary is from EA1. Flow-control behavior, plugin set, and routing logic are identical to the EA1 evaluation. |

## Next Steps

1. **Evaluate EA2 EPP.** The current evaluation uses the EA1 EPP binary. EA2 or later EPP builds may include the `utilization-detector` plugin and updated routing logic that could change flow-control effectiveness.
2. **Test with higher load to trigger AIMD.** The current workload does not generate 429/5xx responses. Increasing batch concurrency or reducing replica count may trigger AIMD adaptation, allowing evaluation of dynamic concurrency adjustment.
3. **Rerun gpt-oss-120b ungated r=4.** Capture burst-phase data to complete the comparison matrix.
4. **Investigate FP8-70B baseline variability.** The 4x difference in interactive-only TTFT p99 between EA1 and EA2 test sessions needs root cause analysis (GPU clock states, thermal throttling, co-located workloads, or vLLM scheduling state).
5. **Evaluate selective flow-control.** The data suggests flow-control should be enabled per-model based on capacity: beneficial for models under contention (gpt-oss-120b at r=1), harmful for models with spare capacity (Qwen3-8B at r=1, all models at high replica counts).
