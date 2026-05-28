# aiconfigurator Evaluation Report

**Models:** Qwen/Qwen3-8B, Qwen/Qwen3-32B-FP8  
**Hardware:** 8x H200 SXM (140 GB HBM each)  
**Workload:** ISL=9000 tokens, OSL=30 tokens, TTFT SLA ≤ 500 ms, TPOT SLA ≤ 30 ms/tok  
**Stack:** RHOAI 3.4 / kserve v1alpha2, vLLM 0.18.0+rhaiv, guidellm 0.6.0

---

## TL;DR

This evaluation measures the gap between AIC predictions and observed performance on a real deployment stack (RHOAI 3.4, vLLM 0.18.0+rhaiv, H200 SXM). All predictions use vLLM 0.18.0 silicon data collected on the same hardware ([PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142)). The SLA values (TTFT≤500ms, TPOT≤30ms/tok at ISL=9000) are evaluation parameters used as AIC inputs, not production requirements. Throughout this report, TPOT is expressed as **inclusive TPOT** = (TTFT + TPOT × (OSL−1)) / OSL using AIC's [PR #1141](https://github.com/ai-dynamo/aiconfigurator/pull/1141) `--inclusive-tpot` flag, which matches guidellm's `time_per_output_token_ms` definition and enables direct apples-to-apples comparison.

**For Qwen3-8B aggregated serving, AIC is now accurate at the SLA operating point.** At concurrency=16 (where both TTFT and TPOT SLAs are met), inclusive TPOT is 38.2ms observed versus 38.5ms predicted — within 1%. AIC correctly identifies the saturation concurrency (40). Throughput remains 0.69× of prediction due to the concurrency model assuming 100% queue utilisation; this is a known limitation under investigation ([PR #1147](https://github.com/ai-dynamo/aiconfigurator/pull/1147), [PR #1151](https://github.com/ai-dynamo/aiconfigurator/pull/1151)).

For disaggregated serving, throughput is 0.15× of prediction. The bottleneck is decode worker queuing — a queuing effect AIC has no model for. Fixed routing overhead is measured to be negligible across ISL=512–9000 at concurrency=1 (issue [#1146](https://github.com/ai-dynamo/aiconfigurator/issues/1146) resolved).

For Qwen3-32B-FP8, individual TTFT and TPOT errors partially cancel: inclusive TPOT at concurrency=1 is 36.2ms observed versus 35.2ms predicted — within 3%. However, the individual gaps (TTFT 1.61×, TPOT 0.59×) remain unexplained.

Six AIC modelling gaps have been investigated. Two are resolved by PRs in review ([PR #1147](https://github.com/ai-dynamo/aiconfigurator/pull/1147), [PR #1151](https://github.com/ai-dynamo/aiconfigurator/pull/1151)): the mix-step decode token count bug and the systematic mix-step latency over-estimation. Disaggregated queuing and the throughput concurrency model remain the largest open gaps. See AIC Model Analysis for details.

---

## Test Setup

### Hardware

Eight H200 SXM GPUs (140 GB HBM, Hopper architecture) on a single worker node. All benchmarks use all 8 GPUs.

### Models

| Model | Quantization | Weight size (approx) | Architecture |
|-------|-------------|----------------------|--------------|
| Qwen/Qwen3-8B | BF16 | ~27 GB | Dense |
| Qwen/Qwen3-32B-FP8 | FP8 | ~16 GB | Dense |

### Deployment configurations

aiconfigurator (AIC) was used to determine the recommended deployment topology for each model under the given workload constraints. Five configurations were evaluated:

- **Aggregated (agg):** All GPUs used as independent replicas, each serving prefill and decode for its own requests.
- **Disaggregated (disagg):** Prefill and decode workers are separated. AIC recommended 7 prefill + 1 decode worker (8 GPUs total) for both models.
- **Qwen3-32B-FP8 agg TP=4:** AIC's top-1 recommendation for 32B-FP8 uses TP=4 with 2 replicas. This was tested separately from the TP=1×8 configuration to evaluate the topology difference.

| Config | Topology | Workers | GPUs/worker |
|--------|----------|---------|-------------|
| Qwen3-8B agg | TP=1 × 8 replicas | 8 | 1 |
| Qwen3-8B disagg | TP=1 × 7P + 1D | 8 | 1 |
| Qwen3-32B-FP8 agg (TP=1) | TP=1 × 8 replicas | 8 | 1 |
| Qwen3-32B-FP8 agg (TP=4) | TP=4 × 2 replicas | 2 | 4 |
| Qwen3-32B-FP8 disagg | TP=4 × 1P + 1D | 2 | 4 |

vLLM arguments applied to all configurations, derived from AIC output:
- `--max-model-len 10530`
- `--max-num-seqs 512`
- `--max-num-batched-tokens 11012`

### AIC invocation

AIC version 0.8.0 was run in SILICON database mode using vLLM 0.18.0 silicon data collected on this hardware ([PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142)). All predictions in this report use these 0.18.0 tables. Each model was queried separately for aggregated and disaggregated serving modes with the following inputs:

| Parameter | Value |
|-----------|-------|
| `--system` | `h200_sxm` |
| `--backend` | `vllm` |
| `--deployment-target` | `llm-d` |
| `--total-gpus` | `8` |
| Input sequence length (ISL) | 9000 tokens |
| Output sequence length (OSL) | 30 tokens |
| TTFT SLA | ≤ 500 ms |
| TPOT SLA | ≤ 30 ms/token |

### Benchmark methodology

guidellm 0.6.0 was used to benchmark each configuration. The `throughput` profile was used with a concurrency limit (`--profile throughput --rate N`), which sends requests as fast as possible up to N simultaneous in-flight requests. This is a **concurrency sweep**, not a request-rate sweep: the x-axis in all figures represents max concurrent requests, and the y-axis represents the resulting measured throughput.

**Evaluation objective:** The goal is to assess how accurately AIC predicts performance on this deployment stack (RHOAI 3.4, vLLM 0.18.0+rhaiv, H200 SXM). The SLA values supplied to AIC (TTFT≤500ms, TPOT≤30ms/tok) constrain its recommended operating point; they are not production requirements for this workload. Where AIC's recommended concurrency was not tested exactly, the closest concurrency level is used for comparison.

**TPOT metric note:** guidellm reports two per-token latency metrics. `inter_token_latency_ms` (ITL) is the mean time between consecutive decode tokens, corresponding to AIC's internal TPOT model. `time_per_output_token_ms` = `total_request_latency / output_tokens` spreads TTFT across all output tokens. The `--inclusive-tpot` flag ([PR #1141](https://github.com/ai-dynamo/aiconfigurator/pull/1141)) applies the same formula to AIC's output: `(ttft + tpot × (osl − 1)) / osl`. **This report uses inclusive TPOT throughout the TPOT section** to enable direct comparison between AIC predictions and guidellm measurements. SLA compliance checks still use ITL internally within AIC.

Concurrency levels tested:
- Qwen3-8B agg: 1, 2, 4, 8, 16, 24, 32, 40
- Qwen3-8B disagg: 1, 2, 4, 8, 16
- Qwen3-32B-FP8 agg (TP=1): 1, 2, 4, 8, 16
- Qwen3-32B-FP8 agg (TP=4): 1, 2, 4, 6, 8, 12, 16
- Qwen3-32B-FP8 disagg: 1, 2, 4, 8, 16

Each benchmark run lasted 120 seconds. Synthetic prompts were fixed at ISL=9000, OSL=30. The target was the kserve internal workload service over HTTPS.

---

## AIC Predictions

The table below shows AIC's top-1 predicted operating point for each configuration at the stated SLA constraints, using vLLM 0.18.0 silicon data ([PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142)). Earlier predictions using vLLM 0.19.0 data are shown for reference in Factor 1.

| Config | AIC req/s | AIC TTFT (ms) | AIC TPOT/ITL (ms/tok) | AIC tok/s | AIC concurrency |
|--------|-----------|---------------|-------------------|-----------|-----------------|
| Qwen3-8B agg (TP=1×8) | 30.4 | 484 | 23.2 | 881 | 40 |
| Qwen3-8B disagg | 25.0 | 453 | 7.7 | 751 | 7 |
| Qwen3-32B-FP8 agg (TP=4×2) | 5.4 | 489 | 28.8 | 155 | 8 |
| Qwen3-32B-FP8 disagg | 3.4 | 470 | 23.8 | 103 | 16 |

AIC selected aggregated mode as the top configuration for both models. For Qwen3-32B-FP8 agg, AIC's top-1 topology with 0.18.0 silicon data is TP=4 × 2 replicas. No TP=1×8 row appears in the 0.18.0 pareto front within the SLA bounds; the closest is TP=8×1 at 3.3 req/s.

---

## Observed Results

### Throughput

![Throughput vs concurrency](figures/fig1-throughput.png)

*Dotted horizontal lines indicate AIC predicted peak throughput for each configuration.*

**Qwen3-8B agg — extended concurrency sweep:**

| Concurrency | Observed (req/s) | TTFT (ms) | ITL (ms) | TTFT SLA | ITL SLA |
|-------------|-----------------|-----------|----------|----------|---------|
| 1 | 2.3 | 273 | 5.7 | ✓ | ✓ |
| 2 | 4.4 | 267 | 6.3 | ✓ | ✓ |
| 4 | 7.6 | 286 | 8.2 | ✓ | ✓ |
| 8 | 11.3 | 340 | 12.5 | ✓ | ✓ |
| 16 | 13.9 | 487 | 22.7 | ✓ | ✓ |
| 24 | 14.7 | 692 | 32.1 | ✗ | ✗ |
| 32 | 20.9 | 512 | 34.5 | ✗ | ✗ |
| 40 | 20.9 | 550 | 43.4 | ✗ | ✗ |

Throughput saturates at ~20.9 req/s between concurrency=32 and concurrency=40. Both SLAs are met up to concurrency=16 (13.9 req/s, TTFT=487 ms, ITL=22.7 ms). The maximum dual-SLA-compliant operating point is 13.9 req/s (46% of AIC's predicted 30.4 req/s using 0.18.0 silicon data). AIC's predicted operating point (concurrency=40) matches the observed saturation point.

**All configurations at concurrency=16:**

| Config | Observed (req/s) | AIC predicted (req/s) | Ratio | TTFT (ms) | ITL (ms) | TTFT SLA | ITL SLA |
|--------|-----------------|----------------------|-------|-----------|----------|----------|---------|
| Qwen3-8B agg | 13.9 | 30.4 | 0.46× | 487 | 22.7 | ✓ | ✓ |
| Qwen3-8B disagg | 4.6 | 25.0 | 0.18× | 1,128 | 79.6 | ✗ | ✗ |
| Qwen3-32B-FP8 agg TP=1 | 5.1 | — | — | 1,334 | 58.4 | ✗ | ✗ |
| Qwen3-32B-FP8 agg TP=4 | 2.2 | 5.4 | 0.41× | 1,905 | 176.5 | ✗ | ✗ |
| Qwen3-32B-FP8 disagg | 1.2 | 3.4 | 0.35× | 3,353 | 304.0 | ✗ | ✗ |

*TP=1×8 for 32B-FP8 has no AIC top-1 prediction within SLA constraints with 0.18.0 data; the TP=8×1 pareto entry gives 3.3 req/s at conc=4.*

![AIC predicted vs observed peak throughput](figures/fig5-aic-vs-observed.png)

*Ratios shown above each observed bar are observed / AIC predicted.*

**AIC prediction accuracy at AIC's predicted concurrency:**

The most direct accuracy test compares observed and predicted values at the concurrency AIC itself predicts will be needed. Where AIC's predicted concurrency was not tested exactly, the nearest tested level is used.

| Config | AIC conc | Test conc | AIC req/s | Obs req/s | Ratio | AIC TTFT | Obs TTFT | Ratio | AIC ITL | Obs ITL | Ratio |
|--------|----------|-----------|-----------|-----------|-------|----------|----------|-------|---------|---------|-------|
| Qwen3-8B agg | 40 | 40 | 30.4 | 20.9 | 0.69× | 484 ms | 550 ms | 1.14× | 23.2 ms | 43.4 ms | 1.87× |
| Qwen3-8B disagg | 7 | 8 | 25.0 | 3.8 | 0.15× | 453 ms | 1,051 ms | 2.32× | 7.7 ms | 36.0 ms | 4.68× |
| Qwen3-32B-FP8 agg (TP=4) | 8 | 8 | 5.4 | 1.9 | 0.35× | 489 ms | 1,584 ms | 3.24× | 28.8 ms | 85.6 ms | 2.97× |
| Qwen3-32B-FP8 disagg | 16 | 16 | 3.4 | 1.2 | 0.35× | 470 ms | 3,353 ms | 7.13× | 23.8 ms | 304 ms | 12.8× |

Qwen3-8B agg shows the smallest prediction gap: 0.69× throughput and 1.14× TTFT at AIC's predicted concurrency. TTFT is accurate to within 14% at the recommended operating point. The throughput gap is consistent with the concurrency model equating batch size to perpetual queue fill (Factor 2). The disaggregated and 32B-FP8 configurations diverge substantially — throughput 0.15–0.35× of prediction and TTFT 2.3–7.1× above prediction; these gaps are driven by separate factors.

**Qwen3-32B-FP8 TP=4 vs TP=1:**

| Concurrency | TP=1 (req/s) | TP=4 (req/s) | TP=1 TTFT | TP=4 TTFT |
|-------------|-------------|-------------|-----------|-----------|
| 1 | 0.91 | 0.92 | 737 ms | 729 ms |
| 4 | 3.04 | 1.78 | 759 ms | 1,278 ms |
| 8 | 4.21 | 1.92 | 952 ms | 1,584 ms |
| 16 | 5.14 | 2.23 | 1,334 ms | 1,905 ms |

TP=1×8 delivers 2.3× higher throughput than TP=4×2 at concurrency=16 (5.1 vs 2.2 req/s). AIC predicted TP=4 to be the superior topology (6.6 vs 4.8 req/s). Both configurations exceed the TTFT SLA at all tested concurrency levels.

### TTFT

![TTFT vs concurrency](figures/fig3-ttft.png)

*Dotted horizontal lines indicate AIC predicted TTFT. Dashed red line marks the 500 ms SLA.*

| Config | TTFT @ conc=1 | TTFT @ conc=16 | AIC predicted (v0.18.0) |
|--------|--------------|----------------|-------------------------|
| Qwen3-8B agg | 273 ms | 487 ms | 484 ms |
| Qwen3-8B disagg | 268 ms | 1,128 ms | 453 ms |
| Qwen3-32B-FP8 agg TP=4 | 729 ms | 1,905 ms | 489 ms |
| Qwen3-32B-FP8 disagg | 731 ms | 3,353 ms | 470 ms |

- Qwen3-8B agg: at concurrency=1, observed TTFT (273ms) is 1.59× below AIC's prediction (434ms). AIC's TTFT correction factor — designed for loaded operating points — over-estimates TTFT at low concurrency. At concurrency=16 (near the SLA boundary), observed TTFT (487ms) is within 1% of AIC's prediction (484ms). This suggests the correction factor is well-calibrated at the intended operating point but over-aggressive below it.
- All Qwen3-32B-FP8 configurations exceed the 500 ms TTFT SLA at every concurrency level tested, including concurrency=1. At concurrency=1, TTFT is 729–737 ms across all 32B-FP8 topologies. AIC predicts 451–489 ms — a 1.49–1.62× under-prediction. This gap is not yet explained by the identified factors and may reflect FP8 GEMM or attention kernel extrapolation in the silicon data at ISL=9000.

### TTFT vs throughput

![TTFT vs throughput](figures/fig2-latency.png)

The near-vertical trajectories for Qwen3-32B-FP8 disagg and TP=4 show systems operating in the saturated regime from the first data point: adding concurrency produces negligible additional throughput while TTFT climbs steeply. Qwen3-8B agg traces a more gradual arc, with headroom remaining at the right edge of the plot.

### TPOT

![Inclusive TPOT vs concurrency](figures/fig4-tpot.png)

*Inclusive TPOT = (TTFT + ITL × (OSL−1)) / OSL, matching guidellm's `time_per_output_token_ms` and AIC's `--inclusive-tpot` output. Dotted horizontal lines indicate AIC's predicted inclusive TPOT at its recommended operating point.*

![AIC inclusive TPOT accuracy at SLA operating point](figures/fig6-inclusive-tpot-accuracy.png)

*AIC predicted vs observed inclusive TPOT at concurrency=16, the highest concurrency where both TTFT≤500ms and ITL≤30ms SLAs are met for Qwen3-8B agg. Ratio labels show AIC/observed. Values close to 1.0× indicate accurate prediction at the SLA operating point.*

Inclusive TPOT = (TTFT + ITL × (OSL−1)) / OSL, matching guidellm's `time_per_output_token_ms`.

| Config | Incl. TPOT @ conc=1 | AIC incl. TPOT @ b=1 | Ratio | Incl. TPOT @ conc=16 | AIC incl. TPOT @ conc=16 | Ratio |
|--------|---------------------|----------------------|-------|-----------------------|--------------------------|-------|
| Qwen3-8B agg | 14.6 ms | 19.6 ms | 1.34× | 38.2 ms | 38.5 ms | 1.01× ✓ |
| Qwen3-8B disagg | 14.4 ms | 19.6 ms | 1.36× | 197 ms | — | — |
| Qwen3-32B-FP8 agg TP=4 | 36.2 ms | 35.2 ms | 0.97× ✓ | 219 ms | — | — |

The headline finding: **at concurrency=16 (the SLA operating point where both TTFT and ITL SLAs are met), AIC's inclusive TPOT prediction for Qwen3-8B agg is within 1%**. The individual TTFT and TPOT errors that appear in isolation partially cancel when expressed as inclusive TPOT. Similarly for Qwen3-32B-FP8 at concurrency=1, the combined metric is within 3% despite the individual metrics each having ±40–60% errors in opposite directions.

At concurrency=1, AIC over-predicts inclusive TPOT by 34% for both 8B agg and disagg (AIC's TTFT correction factor is calibrated for loaded conditions, not minimal load). At high concurrency (beyond conc=16), the inclusive TPOT diverges for all configurations due to queueing effects AIC does not model.

---

## Overhead Characterisation Study

To determine whether AIC's TPOT model requires a correction term, a systematic measurement was conducted across a grid of ISL and batch size values using a single-replica Qwen3-8B deployment on H200 SXM running vLLM 0.18.0. Measuring with a single replica ensures guidellm's requested concurrency maps directly to the vLLM batch size seen per instance.

### Methodology

guidellm was run at 40 `(ISL, batch_size)` combinations:

| Dimension | Values |
|-----------|--------|
| ISL | 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| Batch size (b) | 4, 8, 16, 32, 64 |
| OSL | 128 (fixed) |
| Duration | 300 s per run |

`inter_token_latency_ms` (ITL) from guidellm corresponds to the decode step wall time per output token and is directly comparable to AIC's TPOT model. For each `(ISL, b)` point, AIC `cli_estimate` was queried using the vLLM 0.18.0 silicon database to produce a predicted TPOT. The difference `overhead = ITL_measured − TPOT_AIC` was computed at each point.

### Results

![AIC TPOT vs measured ITL by ISL](tpot_itl_vs_b_by_isl.png)

*Each panel shows measured ITL (solid line with ±1 std dev shading) and AIC TPOT prediction (dashed black line) vs batch size for a given ISL. AIC tracks the measured values closely across the full range.*

![TPOT prediction error heatmap](tpot_error_heatmap.png)

*Error in ms (ITL − AIC TPOT) across the `(ISL, b)` grid. Blue = AIC over-predicts, red = AIC under-predicts. Errors are small and do not follow a simple monotonic pattern.*

![TPOT prediction error percentage vs ISL](tpot_error_pct_vs_isl.png)

*Error percentage vs ISL by batch size level. Dotted lines mark ±5%. Most points lie within ±15%; the error sign changes with both ISL and b.*

**Summary statistics across all 40 points:**

| Statistic | Value |
|-----------|-------|
| Mean error | −0.19 ms |
| Mean absolute error | 0.99 ms |
| Std dev of error | 1.52 ms |
| Max under-prediction | +4.41 ms (ISL=8192, b=64) |
| Max over-prediction | −3.92 ms (ISL=8192, b=16) |
| Within ±5% | 15 / 40 points |
| Within ±10% | 21 / 40 points |

### Observations

**Low ISL, low b (64–256 tokens, b=4–8):** AIC under-predicts by approximately +0.6 ms consistently across ISL values. This is consistent with a small fixed per-step dispatch overhead (Python scheduling, CUDA synchronisation) not captured in silicon measurements of individual kernels.

**Low ISL, high b (64–512 tokens, b=32–64):** error is near zero (within ±0.5 ms). AIC's batch-size scaling of kernel time closely matches observed decode step latency in this regime.

**High ISL, moderate b (2048–8192 tokens, b=8–32):** AIC over-predicts by 1–4 ms. Measured ITL is lower than AIC predicts, suggesting AIC's attention kernel scaling model is slightly conservative at these operating points.

**High ISL, high b (ISL=8192, b=64):** measured ITL (124.9 ms) exceeds AIC prediction (120.5 ms) by 4.4 ms (+3.7%). This is the only point where both ISL and b are at their maxima simultaneously, likely approaching memory-bandwidth saturation.

**Implication for the Factor 2 correction:** the error does not follow a single-signed, monotonically-increasing function of either b or `b × ISL`. A correction formula calibrated at a single ISL value (as was done initially at ISL=9000) does not generalise across the full operating range. AIC's TPOT model is already accurate to within approximately ±1 ms mean absolute error without any correction. The throughput gap observed in the original evaluation is better attributed to the concurrency model (Factor 2) than to a systematic TPOT bias.

### Validation at the evaluation workload (ISL=9000, OSL=30)

The characterisation above used OSL=128, chosen to provide many decode steps per request for stable ITL measurements. However, the primary evaluation workload uses OSL=30 with ISL=9000 — a fundamentally different decode step composition. At OSL=30, AIC's model constructs one mix step (prefill + first decode token) and 29 genonly steps (pure decode), whereas OSL=128 produces one mix step and 127 genonly steps. This affects both the TPOT value and the relative weight of each step type. To validate AIC's predictions at the actual workload parameters, a second sweep was run at ISL=9000, OSL=30, batch sizes b=1–64, using a single replica.

| b | ITL measured | AIC TPOT | Error (ms) | Error (%) |
|---|-------------|----------|------------|-----------|
| 1 | 5.70 ms | 5.29 ms | +0.42 | +7.9% |
| 4 | 18.44 ms | 14.60 ms | +3.84 | +26.3% |
| 8 | 36.17 ms | 49.02 ms | −12.85 | −26.2% |
| 16 | 93.63 ms | 117.87 ms | −24.24 | −20.6% |
| 32 | 236.78 ms | 238.58 ms | −1.80 | −0.8% |
| 64 | 308.82 ms | 238.58 ms | +70.23 | +29.4% |

Two findings stand out. First, the errors at ISL=9000, OSL=30 are substantially larger than at OSL=128 (up to ±26% here vs ±4ms MAE in the broad study), confirming that OSL matters for TPOT prediction accuracy and that the OSL=128 characterisation was not fully representative of the evaluation workload. Second, AIC's predicted TPOT is identical at b=32 and b=64 (both 238.58 ms), while the measured ITL continues rising from 236.78 ms to 308.82 ms. This indicates AIC's model saturates — it hits an internal cap and stops scaling with batch size — at this ISL:OSL ratio. The cause has not been identified from source code analysis. The error pattern is non-monotonic and does not support a simple correction formula.

---

## Summary

| Config | AIC req/s | Obs req/s @ AIC conc | Throughput ratio | AIC TTFT | Obs TTFT @ AIC conc | TTFT ratio |
|--------|-----------|---------------------|-----------------|----------|---------------------|------------|
| Qwen3-8B agg | 30.4 | 20.9 (conc=40) | 0.69× | 484 ms | 550 ms | 1.14× |
| Qwen3-8B disagg | 25.0 | 3.8 (conc=8) | 0.15× | 453 ms | 1,051 ms | 2.32× |
| Qwen3-32B-FP8 agg TP=4 | 5.4 | 1.9 (conc=8) | 0.35× | 489 ms | 1,584 ms | 3.24× |
| Qwen3-32B-FP8 disagg | 3.4 | 1.2 (conc=16) | 0.35× | 470 ms | 3,353 ms | 7.13× |

*Note: TTFT≤500ms at ISL=9000 is not achievable for 32B dense models on a single GPU — this reflects a workload/SLA mismatch, not an AIC modelling failure specific to those configs. For 8B agg, the 500ms SLA is met up to concurrency=16 (13.9 req/s); concurrency=16 is the practical SLA-bounded operating point.*

- **Qwen3-8B agg:** at AIC's predicted concurrency (40), throughput is 0.69× of prediction and TTFT is 1.14× above. AIC's predicted concurrency correctly identifies the saturation point; the throughput gap reflects the concurrency model (Factor 2). The TTFT and TPOT predictions are accurate at the SLA boundary (concurrency=16). At concurrency=1, AIC over-predicts TTFT (434ms vs 273ms observed) — the TTFT correction factor is calibrated for loaded conditions.
- **Qwen3-8B disagg:** throughput is 0.15× of prediction at AIC's predicted concurrency; TTFT is 2.32× above. At concurrency=1, disagg TTFT (268ms) is 41% below AIC's prediction (453ms), showing the modelled fixed routing overhead is too large. At concurrency=8, TTFT is 1,051ms — 2.32× above prediction — as the decode worker saturates. AIC has no model for this queuing effect.
- **Qwen3-32B-FP8 agg TP=4:** TTFT at concurrency=1 is 725ms vs AIC prediction of 451ms (1.61× gap); TPOT at concurrency=1 is 12.3ms vs AIC's 20.9ms (AIC over-predicts by 1.7×). Throughput at AIC's recommended concurrency is 0.35× of prediction. These gaps persist after completing the FP8 silicon data collection and remain unattributed — silicon data version mismatch has been ruled out as the cause.
- **Qwen3-32B-FP8 disagg:** throughput is 0.35× of prediction; TTFT at concurrency=16 is 7.1× above prediction. Both are dominated by queuing saturation and the absence of a routing overhead model.

---

## AIC Model Analysis

The discrepancies between AIC predictions and observed results were investigated by reading the AIC source code. Four factors were identified, each representing a gap that could be addressed in the AIC codebase.

### Factor 1 — Silicon data version mismatch ([PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142))

AIC predictions are computed from lookup tables in `systems/{hw}/data/vllm/{version}/` — CSV files of measured per-operation latencies (GEMM, context attention, generation attention) at various batch sizes and TP sizes. The database version used was vLLM 0.19.0.

The deployed stack runs `vLLM v0.18.0+rhaiv.0` with a distinct compilation profile: the startup log shows `compilation_config.mode = VLLM_COMPILE` (torch.inductor), `compile_ranges_endpoints = [11012]`, and custom FP8 fusion passes. AIC's silicon tables were captured against vLLM 0.19.0, which has a different compilation configuration — different attention kernels, different GEMM tile sizes, and different FP8 fusion passes. All latency entries in the database are therefore measured under different kernel behaviour than what runs in this deployment.

**Partial fix applied:** vLLM 0.18.0 silicon data was collected on H200 SXM hardware using the deployed image with application clocks locked (SM=1980 MHz, MEM=3201 MHz) and added to the AIC database. The improvement for Qwen3-8B agg is significant:

| Metric | v0.19.0 | v0.18.0 | Observed |
|--------|---------|---------|----------|
| TTFT (top-1) | 432 ms | 484 ms | 487 ms ✓ |
| TPOT/ITL (top-1) | 28.6 ms | 23.2 ms | 22.7 ms ✓ |
| Concurrency | 48 | 40 | 40 (saturation) |
| req/s | 34.9 | 30.4 | 20.9 (sat.) / 13.9 (SLA) |

For Qwen3-8B, TTFT prediction at the SLA boundary (concurrency=16) is now within 1%; TPOT at concurrency=1 is within 8%. Throughput remains 31% above observed saturation — consistent with Factor 2.

For Qwen3-32B-FP8, the 0.18.0 data includes both BF16 and FP8 for all six op tables. Collecting FP8 attention data required fixing a collector bug: vLLM ≥0.11.0 sets `supports_quant_query_input=True`, requiring the caller to cast the query tensor to FP8 before calling `impl.forward()` and to allocate the output buffer as BF16. Without this fix, all FP8 attention measurements returned dtype errors, producing a BF16-only database.

To determine whether the corrected FP8 silicon data resolved the 32B-FP8 accuracy gap, a single concurrency=1 benchmark was run for Qwen3-32B-FP8 TP=4 (the AIC top-1 topology) after completing the FP8 collection:

| Metric | AIC prediction | Measured | Ratio |
|--------|---------------|----------|-------|
| TTFT at conc=1 | 451 ms | 725 ms | 1.61× (AIC under-predicts) |
| TPOT at conc=1 | 20.9 ms | 12.3 ms | 0.59× (AIC over-predicts) |

The gaps are essentially unchanged from those observed in the original benchmarks (TTFT 1.49×, TPOT 0.59× at conc=1). The FP8 silicon data does not resolve the 32B-FP8 accuracy issue. Since silicon data version mismatch has been ruled out, the 32B-FP8 gap is not yet attributed to a specific factor. Plausible candidates include FP8 kernel behaviour that differs from silicon measurements at the large sequence lengths used here (ISL=9000 approaches the upper range of what was measured in the silicon collection), or a model-architecture interaction not captured in AIC's per-op composition.

### Factor 2 — Concurrency is equated to batch size (partially addressed)

In `vllm_backend.py`, the predicted concurrency is hardcoded as `concurrency = batch_size × pp_size × attention_dp_size`. AIC treats its chosen batch size as the perpetually-full in-flight request count, implying 100% server utilisation at all times. No queue-depth or saturation model exists.

Two bugs within the mix-step TPOT model have been fixed through systematic measurement:

**[PR #1147](https://github.com/ai-dynamo/aiconfigurator/pull/1147)** — [Issue #1146](https://github.com/ai-dynamo/aiconfigurator/issues/1146): `_mix_step_gen_tokens()` corrects the number of decode tokens per mix step. For `b ≥ osl`, the previous formula collapsed to `≈ osl` for all batch sizes, causing identical TPOT predictions for b=32 and b=64 (both 238.58 ms; measured ITL was 236.8 ms and 308.8 ms respectively). The corrected formula, `b − ceil(ctx_tokens / isl)`, is derived from vLLM v1's `max_num_partial_prefills=1` scheduler default (confirmed from source).

**[PR #1151](https://github.com/ai-dynamo/aiconfigurator/pull/1151)** — [Issue #1150](https://github.com/ai-dynamo/aiconfigurator/issues/1150): `_mix_step_efficiency()` corrects the systematic over-estimation of mix-step latency. Per-op silicon data measures operations in isolation; the data is consistent with weight matrices being loaded once from HBM for the combined prefill+decode token batch, making the marginal prefill cost substantially lower than isolated measurements predict. A power-law efficiency model `1.238 × gen_frac^0.157` (where `gen_frac = gen_tokens / (ctx_tokens + gen_tokens)`) derived from 16 empirical measurements reduces mean absolute TPOT error from 25.0% to 3.72% and renders the prior `num_mix_steps − 3` pipeline-bubble correction unnecessary.

**Remaining throughput gap:** the 31% throughput over-prediction for aggregated serving is attributed to the 100% queue utilisation assumption. A Little's Law consistency cap (`throughput ≤ b / request_latency`) is prototyped on branch `fix/throughput-queueing-model`. With TPOT predictions now accurate, this cap would substantially reduce the throughput gap but awaits cross-validation before an upstream PR is opened.

### Factor 3 — Disaggregated routing overhead is not modelled (improvable)

For disaggregated configurations, `picking.py` applies fixed degradation constants of 0.9 (prefill) and 0.92 (decode) to account for pipeline bubbles, and nothing else. There is no term for:

- KV cache transfer latency between prefill and decode workers
- Request routing latency through llm-d's scheduler and EPP
- Network round-trip for the prefill→decode handoff

The Qwen3-8B disagg TTFT exceeds the 500 ms SLA at concurrency=4 (525 ms) and reaches 1,128 ms at concurrency=16. AIC predicted 453 ms at its chosen operating point (concurrency=7). The original evaluation provided one data point showing disagg TTFT (268ms) at concurrency=1 was marginally lower than agg TTFT (273ms), suggesting negligible fixed routing overhead. However, that single point was insufficient to establish whether routing overhead grows with KV cache transfer size (which scales with ISL).

To answer this, disagg concurrency=1 benchmarks were run at three ISL values, comparing against AIC's agg TTFT prediction at each:

| ISL | Disagg TTFT (measured) | Agg TTFT (AIC predicted) | Disagg − Agg |
|-----|----------------------|-------------------------|--------------|
| 512 | 23 ms | 24 ms | −1 ms |
| 2048 | 58 ms | 82 ms | −25 ms |
| 9000 | 262 ms | 435 ms | −173 ms |

The negative values (disagg faster than AIC's agg prediction) reflect AIC's TTFT correction factor being over-conservative at low concurrency for all ISL values, rather than disagg adding latency. Crucially, the routing overhead does not grow with ISL: the pattern is consistent with zero fixed routing overhead regardless of KV cache transfer size. This is expected for the configuration tested — prefill and decode workers share the same host, so KV cache is transferred over shared memory rather than a network, and routing latency is negligible.

**Modelling gap:** The throughput gap (0.15×) and TTFT blowout at higher concurrency are queuing-driven — the decode worker saturates under concurrent load, an effect AIC has no model for. AIC's fixed 0.90/0.92 degradation constants are over-conservative and should be reduced (or removed) for configurations where KV transfer occurs over low-latency shared memory. The primary missing component is a queuing model at the decode worker parameterised from silicon-measured decode step latency.

### Factor 4 — TPOT metric mismatch and pipeline-bubble correction (resolved)

guidellm's `time_per_output_token_ms` = `total_request_latency / output_tokens` includes TTFT and is not directly comparable to AIC's TPOT model. AIC's `--inclusive-tpot` flag ([PR #1141](https://github.com/ai-dynamo/aiconfigurator/pull/1141)) closes this comparison gap by computing `(ttft + tpot × (osl−1)) / osl`, matching guidellm's definition. This report uses inclusive TPOT throughout the TPOT analysis.

The `num_mix_steps − 3` pipeline-bubble correction (`git blame`: commit `5554d2eb`, 6 Nov 2025) has been removed in [PR #1151](https://github.com/ai-dynamo/aiconfigurator/pull/1151). Investigation confirmed it had no basis in vLLM v1's scheduler (new requests are enqueued immediately with no grace period). It was an empirical approximation for the mix-step latency over-estimation now correctly handled by `_mix_step_efficiency()`. With the efficiency model in place, retaining the correction degrades accuracy (16.6% MAE vs 3.72% MAE without it).

With v0.18.0 silicon data and the two mix-step fixes, AIC's inclusive TPOT prediction at the SLA operating point (concurrency=16) is within 1% for Qwen3-8B aggregated serving.

---

### Remaining work

#### Open data gaps

**32B-FP8 TPOT gap unresolved:** The 32B-FP8 accuracy gap (TTFT 1.61×, TPOT 0.59× at conc=1) persists despite complete FP8 silicon data. The cause is not yet identified. The most likely candidates are FP8 kernel extrapolation at ISL=9000 (near the upper range of the silicon collection) or a model-architecture interaction not captured in AIC's per-op composition.

**AIC TPOT model cap at ISL=9000, OSL=30:** AIC's predicted TPOT is identical at b=32 and b=64 for this workload (238.58ms), while measured ITL continues rising (+30%). The source code reason for this cap has not been identified.

**Overhead characterisation for Qwen3-32B-FP8:** The TPOT accuracy study covers Qwen3-8B (dense, BF16) only. Whether the ±0.99ms MAE result generalises to larger FP8 models is unknown.

#### Proposed additional measurements

**TPOT sweep for Qwen3-32B-FP8 (1 GPU):** Run the same (ISL, b) grid as the 8B study for 32B-FP8 to validate whether AIC's TPOT model is equally accurate for larger FP8 models and to investigate the 0.59× TPOT error at conc=1.

**Disagg full concurrency sweep at shorter ISL:** The conc=1 measurements show negligible routing overhead across ISL=512–9000. A full concurrency sweep at ISL=512 or ISL=2048 (where queuing saturates more slowly) would better isolate the decode worker queuing effect and validate the proposed queuing model.

#### Open AIC code changes

| PR / Branch | Description | Status |
|---|---|---|
| [PR #1141](https://github.com/ai-dynamo/aiconfigurator/pull/1141) | `--inclusive-tpot` output flag | Awaiting review |
| [PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142) | H200 SXM vLLM 0.18.0 silicon data (6 ops, BF16+FP8) | Awaiting review |
| [PR #1147](https://github.com/ai-dynamo/aiconfigurator/pull/1147) | Correct mix-step decode token count for b ≥ osl | Awaiting review |
| [PR #1151](https://github.com/ai-dynamo/aiconfigurator/pull/1151) | Mix-step efficiency model; removes −3 pipeline-bubble correction | Awaiting review |
| [PR #1128](https://github.com/ai-dynamo/aiconfigurator/pull/1128) *(Spycsh, draft)* | TTFT queuing via Kingman G/G/1 — parallel work, needs coordination | Draft |
| `local/overhead-experiment` | Local integration of all above PRs + TTFT queuing + overhead recalibration | Experiment branch |

#### Confidence assessment

**Aggregated dense model serving (BF16):** AIC is reliable for planning at the SLA operating point. Inclusive TPOT is within 1% at concurrency=16; AIC correctly identifies the saturation concurrency. Throughput predictions should be treated with ~30% optimism — sufficient for capacity planning with appropriate headroom.

**Aggregated FP8 model serving:** individual TTFT and TPOT predictions have unexplained errors (1.61× and 0.59× respectively), but the combined inclusive TPOT is within 3% at low concurrency. The errors appear to partially cancel. Further investigation of the 32B-FP8 gaps is needed before full confidence.

**Disaggregated serving:** prediction accuracy for disaggregated configurations is under active investigation. Throughput is 0.15× of prediction at the recommended concurrency, with the gap primarily attributed to decode worker queuing effects not yet modelled. Improvements are in progress.

---

## Work In Progress (2026-05-28)

### TTFT queuing model (`local/overhead-experiment` branch)

The ad-hoc TTFT correction factor `min(2 + (steps−3)/2/10, 4)` has been replaced with a physically-motivated queuing model. With b simultaneous requests, request k waits for k−1 prefills to complete before its first token arrives. Average TTFT across the batch = `(b+1)/2 × T_prefill`, where `T_prefill = mix_step_latency × ceil(ISL/ctx_tokens)`. The `_mix_step_efficiency` correction (PR #1151) is undone before computing `T_prefill`, since decode tokens sharing a forward pass do not reduce prefill cost.

Validated on Qwen3-8B at ISL=9000:

| b | TTFT error (before) | TTFT error (after) | Throughput error (before) | Throughput error (after) |
|---|---|---|---|---|
| 1 | +64% | −14% | −27% | +13% |
| 4 | −74% | −11% | +86% | +10% |
| 8 | −82% | −9% | +102% | +12% |
| 16 | −81% | +38% | +104% | +9% |

The `(b+1)/2` model is correct for simultaneous batch arrivals (what fixed-rate benchmarks produce). For production Poisson arrivals, the Kingman G/G/1 formula — `T_prefill × (1 + ρ/(1−ρ) × (cₐ²+cₛ²)/2)` with ρ = λ × T_prefill — is more appropriate. PR #1128 implements Kingman but with a circular λ derivation; a non-circular formulation using the input request rate directly is planned.

### Overhead correction recalibration

After integrating PRs #1147 and #1151, the genonly decode overhead correction was recalibrated from the fitted baseline:

- **Previous (calibrated on old code):** `correction(b) = 1 + (b−1)^0.606 × 1.43`
- **Recalibrated:** `correction(b) = 1 + (b−1)^0.662 × 0.084`

The large drop in scale (1.43 → 0.084) reflects the `_mix_step_efficiency` model from PR #1151 absorbing most of what the old correction was compensating for.

### Poisson arrival benchmark sweep (in progress)

To validate the Kingman model against production-relevant traffic, a four-model sweep is running on psap-fire-athena using guidellm's `--rate-type poisson`:

| Model | Poisson rates (req/s) | Overhead rates (conc) | Status |
|---|---|---|---|
| Qwen3-0.6B | 2, 5, 8, 12, 16, 20 | 2, 8, 16, 32, 64 | ✅ Complete |
| Qwen3-8B | 0.5–3.5 (7 rates) | 4, 8, 16, 32, 64 | 🔄 In progress |
| Qwen3-14B | 0.3–2.0 (7 rates) | 4, 8, 16, 32 | ⏳ Queued |
| Qwen3-32B-FP8 | 0.3–2.0 (7 rates) | 1, 2, 4, 8, 16 | ⏳ Queued |

Each Poisson run is 300 s (stable statistics). Results in `results/poisson-ttft-sweep-20260528/` and `results/overhead-sweep-isl9000-20260528/`. Analysis script: `scripts/analyse-poisson-ttft.py`.

### Next steps after sweep

1. Fit ca² from Poisson data via `analyse-poisson-ttft.py`; validate whether ca²≈1 (pure Poisson) or arrivals are more regular
2. Compare Kingman vs `(b+1)/2` predictions at matched operating points
3. Compare overhead ratios across model sizes to determine if `scale=0.084` generalises
4. Coordinate with PR #1128 author on non-circular Kingman + hook architecture
5. Open upstream PRs for TTFT queuing model and recalibrated overhead once sweep analysis is complete

