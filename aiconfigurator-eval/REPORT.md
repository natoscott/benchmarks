# aiconfigurator Evaluation Report

**Models:** Qwen/Qwen3-8B, Qwen/Qwen3-32B-FP8  
**Hardware:** 8x H200 SXM (140 GB HBM each)  
**Workload:** ISL=9000 tokens, OSL=30 tokens, TTFT SLA ≤ 500 ms, TPOT SLA ≤ 30 ms/tok  
**Stack:** RHOAI 3.4 / kserve v1alpha2, vLLM 0.18.0+rhaiv, guidellm 0.6.0

---

## TL;DR

This evaluation measures the gap between AIC predictions and observed performance on a real deployment stack (RHOAI 3.4, vLLM 0.18.0+rhaiv, H200 SXM). All predictions use vLLM 0.18.0 silicon data collected on the same hardware ([PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142)). The SLA values (TTFT≤500ms, TPOT≤30ms/tok at ISL=9000) are evaluation parameters used as AIC inputs, not production requirements.

For Qwen3-8B aggregated serving, AIC's predicted concurrency (40) matches the observed saturation point. Throughput is 0.69× of prediction and TTFT is 1.14× above prediction. A 40-point TPOT characterisation study (ISL 64–8192, batch size 4–64, OSL=128) found mean absolute error of 0.99 ms — AIC's per-step latency model is accurate over a broad range. The throughput gap is attributed to the concurrency model, which assumes 100% queue utilisation.

For disaggregated serving, throughput is 0.15× of prediction. At concurrency=1 — where there is no queuing — disagg TTFT (268ms) is 1.7× below AIC's prediction (453ms), indicating AIC over-estimates the fixed routing overhead. At higher concurrency, TTFT blows out to 1,051ms at concurrency=8 as the decode worker saturates, a queuing effect AIC has no model for.

For Qwen3-32B-FP8, TTFT at concurrency=1 is 729ms versus AIC's 489ms prediction — a 1.49× gap that may reflect FP8 kernel extrapolation in the silicon data. TPOT at concurrency=1 is 12.3ms versus AIC's 20.9ms — AIC over-predicts. These gaps are not yet explained.

Four AIC modelling gaps are under investigation: silicon data version (addressed by [PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142)), concurrency model (Factor 2), disaggregated queuing (Factor 3), and an undocumented TPOT pipeline-bubble correction (Factor 4). See AIC Model Analysis for details. Data gaps and proposed additional measurements are listed in the Remaining Work section.

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

**TPOT metric note:** guidellm reports two distinct per-token latency metrics. `time_per_output_token_ms` = `total_request_latency / output_tokens`, which includes TTFT and is not directly comparable to AIC's TPOT model. `inter_token_latency_ms` (ITL) = mean time between consecutive output tokens during the decode phase, which corresponds to AIC's TPOT model. All TPOT columns and SLA checks in this report use ITL. The `--inclusive-tpot` flag added to AIC ([PR #1141](https://github.com/ai-dynamo/aiconfigurator/pull/1141)) computes `(ttft + tpot × (osl − 1)) / osl` at output time, matching guidellm's `time_per_output_token_ms` definition for direct comparison when needed.

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

![ITL vs concurrency](figures/fig4-tpot.png)

*Dotted horizontal lines indicate AIC predicted TPOT (decode interval). Dashed red line marks the 30 ms/tok SLA. ITL (inter-token latency) is the decode-phase interval between consecutive output tokens and matches AIC's TPOT model. guidellm's `time_per_output_token_ms` = total_latency/output_tokens and is not shown here.*

| Config | ITL @ conc=1 | AIC TPOT @ b=1 | Ratio | ITL @ conc=16 | AIC TPOT @ rec. conc |
|--------|-------------|----------------|-------|---------------|----------------------|
| Qwen3-8B agg | 5.7 ms | 5.3 ms | 1.08× | 22.7 ms | 23.2 ms @ conc=40 |
| Qwen3-8B disagg | 5.7 ms | — | — | 79.6 ms | 7.7 ms @ conc=7 |
| Qwen3-32B-FP8 agg TP=1 | 12.3 ms | — | — | 58.4 ms | — |
| Qwen3-32B-FP8 agg TP=4 | 12.3 ms | 20.9 ms | 0.59× | 176.5 ms | 28.8 ms @ conc=8 |
| Qwen3-32B-FP8 disagg | 12.4 ms | — | — | 304.0 ms | 23.8 ms @ conc=16 |

At concurrency=1 (no queuing), Qwen3-8B agg ITL is 5.7 ms against AIC's prediction of 5.3 ms — within 8%. For Qwen3-32B-FP8 agg TP=4, AIC predicts 20.9 ms but observes 12.3 ms at concurrency=1 — AIC over-predicts by 1.7×. The direction of this error (AIC too conservative) is not explained by the known factors and warrants further investigation with the FP8 silicon data.

At concurrency=16, Qwen3-8B ITL (22.7 ms) closely matches AIC's prediction at the recommended concurrency=40 (23.2 ms), but this comparison conflates two different operating points. The model diverges at high concurrency — at conc=40, observed ITL is 43.4 ms versus AIC's 23.2 ms — consistent with the concurrency=batch_size assumption (Factor 2). The disagg and 32B-FP8 ITL values at concurrency=16 far exceed AIC predictions due to queuing saturation, not TPOT misprediction.

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
- **Qwen3-32B-FP8 agg TP=4:** TTFT at concurrency=1 is 729ms vs AIC prediction of 489ms (1.49× gap); throughput at AIC's recommended concurrency is 0.35× of prediction. TPOT at concurrency=1 is 12.3ms vs AIC's 20.9ms — AIC over-predicts by 1.7×. These gaps are not yet attributed to a specific factor.
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

For Qwen3-32B-FP8, the 0.18.0 data includes both BF16 and FP8 for all six op tables. Despite complete silicon data, TTFT at concurrency=1 is 1.49× below AIC's prediction and TPOT at concurrency=1 is 1.7× above prediction. A collector bug fix was required: vLLM ≥0.11.0 sets `supports_quant_query_input=True`, requiring the caller to pass an FP8-dtype query to `impl.forward()` and allocate the output buffer as BF16. Without this fix, all FP8 attention measurements returned dtype errors. Whether the corrected data fully resolves the 32B-FP8 accuracy gap has not yet been validated by re-running the concurrency benchmarks.

### Factor 2 — Concurrency is equated to batch size (under investigation)

In `vllm_backend.py`, the predicted concurrency is hardcoded as `concurrency = batch_size × pp_size × attention_dp_size`. AIC treats its chosen batch size as the perpetually-full in-flight request count, implying 100% server utilisation at all times. No queue-depth or saturation model exists.

An initial analysis (two data points at ISL=9000) produced an empirical decode overhead correction formula. Subsequent collection of a broader dataset (see Overhead Characterisation Study below) revealed that AIC's TPOT model is already accurate to within ±1ms mean across a 40-point (ISL, batch size) grid, and that the apparent correction at ISL=9000 was not reproducible across the full ISL range. The correction formula has been withdrawn pending a principled analysis.

**Remaining work:** the throughput gap is likely explained by the concurrency model — AIC equates its batch size to the perpetually-filled request queue, overestimating achievable throughput at high concurrency. A Little's Law consistency cap (`throughput ≤ b / request_latency`) has been prototyped on branch `fix/throughput-queueing-model` and reduces the Qwen3-8B agg throughput prediction from 34.9 req/s to 24.4 req/s against observed saturation of 20.9 req/s. Further validation is required before this is opened as an upstream PR.

### Factor 3 — Disaggregated routing overhead is not modelled (improvable)

For disaggregated configurations, `picking.py` applies fixed degradation constants of 0.9 (prefill) and 0.92 (decode) to account for pipeline bubbles, and nothing else. There is no term for:

- KV cache transfer latency between prefill and decode workers
- Request routing latency through llm-d's scheduler and EPP
- Network round-trip for the prefill→decode handoff

The Qwen3-8B disagg TTFT exceeds the 500 ms SLA at concurrency=4 (525 ms) and reaches 1,128 ms at concurrency=16. AIC predicted 453 ms at its chosen operating point (concurrency=7). At concurrency=1, disagg TTFT (268ms) is 41% below AIC's prediction (453ms) and marginally below agg TTFT (273ms). This shows the fixed routing and KV-transfer overhead is effectively zero at concurrency=1 — AIC's fixed degradation constants (0.90/0.92) are over-conservative at low load. The throughput gap (0.15×) and TTFT blowout at higher concurrency are queuing-driven: the decode worker saturates rapidly under load.

**Modelling gap:** AIC needs a queuing model at the decode worker, parameterised from silicon-measured decode step latency. The constant 0.90/0.92 degradation factors are not sufficient to capture load-dependent behaviour. A separate routing overhead term should be removed or reduced to reflect the observed negligible fixed routing cost.

### Factor 4 — TPOT metric mismatch and pipeline-bubble correction (largely explained)

guidellm's `time_per_output_token_ms` = `total_request_latency / output_tokens` includes TTFT and is not directly comparable to AIC's TPOT model. AIC's TPOT corresponds to `inter_token_latency_ms` (ITL). All TPOT comparisons in this report use ITL.

With v0.18.0 silicon data, AIC TPOT predictions at ISL=9000 for Qwen3-8B are accurate to within 5% at b=1 (AIC 5.4 ms, observed 5.7 ms). The Overhead Characterisation Study confirms this accuracy extends across ISL=64–8192 and b=4–64, with a mean absolute error of 0.99 ms. No additional correction to the TPOT model is required to explain the observed TPOT values.

The pipeline-bubble correction `num_mix_steps_for_tpot_calc = max(1, num_mix_steps − 3)` (`git blame`: commit `5554d2eb`, 6 Nov 2025, initial H100-only vLLM backend) clamps most batch sizes to a single mix step for TPOT calculation at ISL=9000. Its physical meaning for H200 SXM is not documented and has not been verified. The accuracy of AIC's TPOT predictions over the characterisation grid suggests the current implementation produces reasonable results, but the constant should be reviewed for correctness before extending the model to other hardware.

---

### Remaining work

#### Data gaps

The following comparisons cannot be made with current data:

**32B-FP8 concurrency benchmarks not re-run after FP8 silicon fix:** All 32B-FP8 benchmarks predate the FP8 collector bug fix. We do not know whether the 0.18.0 FP8 silicon data improves the 32B-FP8 TTFT and TPOT gaps. The TTFT gap (1.49× at conc=1) and TPOT gap (0.59× at conc=1) are currently unattributed.

**Overhead characterisation at ISL=9000, OSL=30:** The TPOT study used OSL=128. The actual evaluation workload (ISL=9000, OSL=30) has not been validated directly. At ISL=9000, the mix:genonly step ratio differs significantly (1 mix + 29 genonly for OSL=30 vs 1 mix + 127 genonly for OSL=128), which affects the TPOT composition.

**Overhead characterisation for Qwen3-32B-FP8:** The ±0.99ms MAE result covers Qwen3-8B (dense, BF16) only. Whether AIC's TPOT model is equally accurate for a larger FP8 model is unknown.

**Disagg routing overhead at varying ISL:** A single data point (ISL=9000, conc=1) shows the routing overhead is negligible. Measurements at shorter ISL would determine whether this holds for smaller KV transfer sizes and establish the correct value for AIC's routing constant.

#### Proposed additional measurements

**1. Single-replica sweep at ISL=9000, OSL=30 for Qwen3-8B (1 GPU):** Directly validates AIC TPOT predictions at the evaluation workload parameters. Batch sizes b=1–64. Uses the existing `run-overhead-sweep.sh` with `ISL_VALUES=9000 OUTPUT_TOKENS=30`.

**2. 32B-FP8 concurrency benchmarks (re-run):** Repeat the 5-configuration benchmark suite with the completed 0.18.0 FP8 silicon data to measure whether the TTFT and TPOT gaps for 32B-FP8 are reduced.

**3. Single-replica TPOT sweep for Qwen3-32B-FP8:** Same (ISL, b) grid as the 8B study to validate whether AIC's TPOT accuracy generalises to larger FP8 models.

**4. Disagg concurrency=1 across ISL values:** Run Qwen3-8B disagg at concurrency=1 for ISL in {512, 2048, 9000} to characterise whether the fixed routing overhead is truly ISL-independent, establishing the correct value for AIC's routing constant.

#### Open AIC code changes

- **[PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142)** (`data/h200-sxm-vllm-0.18.0`): silicon data, awaiting review
- **[PR #1141](https://github.com/ai-dynamo/aiconfigurator/pull/1141)** (`feat/inclusive-tpot-output`): `--inclusive-tpot` flag for guidellm-comparable reporting, awaiting review  
- **`fix/throughput-queueing-model`**: Little's Law throughput cap (Factor 2). Awaits cross-validation at multiple ISL values and model types before upstream PR.
- **Factor 3**: no prototype yet. Requires a queuing model parameterised from silicon-measured decode step latency.
- **Factor 4**: `num_mix_steps − 3` constant to be reviewed with original author (commit `5554d2eb`).

