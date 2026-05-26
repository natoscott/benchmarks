# aiconfigurator Evaluation Report

**Models:** Qwen/Qwen3-8B, Qwen/Qwen3-32B-FP8  
**Hardware:** 8x H200 SXM (140 GB HBM each)  
**Workload:** ISL=9000 tokens, OSL=30 tokens, TTFT SLA ≤ 500 ms, TPOT SLA ≤ 30 ms/tok  
**Stack:** RHOAI 3.4 / kserve v1alpha2, vLLM 0.18.0+rhaiv, guidellm 0.6.0

---

## TL;DR

This evaluation assesses how closely AIC predictions match observed performance on a real deployment stack. The SLA values (TTFT≤500ms, TPOT≤30ms/tok at ISL=9000) were used as AIC inputs to constrain its recommendation; they are evaluation parameters, not production requirements. Predictions were generated using AIC's vLLM 0.19.0 silicon database (the version available at evaluation time). Subsequent collection of vLLM 0.18.0 silicon data for this hardware reduced the TTFT prediction error from 1.3× to under 1% at the SLA boundary. For 32B-FP8, TTFT≤500ms at ISL=9000 is not achievable for any dense topology — a workload/SLA mismatch unrelated to AIC accuracy. Disaggregated serving is furthest from predictions (throughput 0.13× for 8B disagg), consistent with AIC having no model of decode worker queuing. A 40-point TPOT characterisation study (ISL: 64–8192, batch size: 4–64, Qwen3-8B on H200 SXM vLLM 0.18.0) found mean absolute TPOT prediction error of 0.99 ms — AIC's per-step latency model is already accurate over a broad operating range. The throughput gap is therefore better attributed to the concurrency model (Factor 2) than to TPOT misprediction. Source code analysis identified four gaps: silicon data version mismatch (now addressed by PR #1142), concurrency equated to batch size (under investigation), no disaggregated queuing model, and an undocumented TPOT pipeline-bubble correction of uncertain validity for H200. See AIC Model Analysis for details.

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

AIC version 0.8.0 was run in SILICON database mode against vLLM 0.19.0 reference data. Each model was queried separately for aggregated and disaggregated serving modes with the following inputs:

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

**TPOT metric note:** guidellm reports two distinct per-token latency metrics. `time_per_output_token_ms` = `total_request_latency / output_tokens`, which includes TTFT and is not directly comparable to AIC's TPOT model. `inter_token_latency_ms` (ITL) = mean time between consecutive output tokens during the decode phase, which corresponds to AIC's TPOT model. All TPOT columns and SLA checks in this report use ITL.

Concurrency levels tested:
- Qwen3-8B agg: 1, 2, 4, 8, 16, 24, 32, 40
- Qwen3-8B disagg: 1, 2, 4, 8, 16
- Qwen3-32B-FP8 agg (TP=1): 1, 2, 4, 8, 16
- Qwen3-32B-FP8 agg (TP=4): 1, 2, 4, 6, 8, 12, 16
- Qwen3-32B-FP8 disagg: 1, 2, 4, 8, 16

Each benchmark run lasted 120 seconds. Synthetic prompts were fixed at ISL=9000, OSL=30. The target was the kserve internal workload service over HTTPS.

---

## AIC Predictions

The table below shows AIC's top-1 predicted operating point for each configuration at the stated SLA constraints, using vLLM 0.19.0 silicon data (the version used at evaluation time). Substantially improved predictions using vLLM 0.18.0 silicon data are described in the AIC Model Analysis section.

| Config | AIC req/s | AIC TTFT (ms) | AIC TPOT/ITL (ms/tok) | AIC tok/s | AIC concurrency |
|--------|-----------|---------------|-------------------|-----------|-----------------|
| Qwen3-8B agg (TP=1×8) | **34.9** | 432 | 28.6 | 1,013 | 48 |
| Qwen3-8B disagg (7P+1D) | 28.8 | 394 | 8.9 | 863 | 10 |
| Qwen3-32B-FP8 agg (TP=4×2) | 6.6 | 493 | 18.2 | 191 | 8 |
| Qwen3-32B-FP8 agg (TP=1×8) | 4.8 | 315 | 13.7 | — | — |
| Qwen3-32B-FP8 disagg (7P+1D) | 3.4 | 473 | 10.4 | 103 | 4 |

AIC selected aggregated mode as the top configuration for both models. For Qwen3-32B-FP8 agg, AIC's top-1 topology is TP=4 × 2 replicas (6.6 req/s). TP=1 × 8 replicas appears lower in the AIC pareto front at 4.8 req/s.

The Qwen3-32B-FP8 agg (TP=1×8) row is included for comparison only — AIC did not recommend this topology and produced no tok/s figure for it. The 4.8 req/s and associated TTFT/TPOT values are read from the AIC pareto front, not from its top-1 output.

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

Throughput saturates at ~20.9 req/s between concurrency=32 and concurrency=40. Both SLAs are met up to concurrency=16 (13.9 req/s, TTFT=487 ms, ITL=22.7 ms). The maximum dual-SLA-compliant operating point is **13.9 req/s** (40% of AIC's predicted 34.9 req/s).

**All configurations at concurrency=16:**

| Config | Observed (req/s) | AIC predicted (req/s) | Ratio | TTFT (ms) | ITL (ms) | TTFT SLA | ITL SLA |
|--------|-----------------|----------------------|-------|-----------|----------|----------|---------|
| Qwen3-8B agg | 13.9 | 34.9 | 0.40× | 487 | 22.7 | ✓ | ✓ |
| Qwen3-8B disagg | 4.6 | 28.8 | 0.16× | 1,128 | 79.6 | ✗ | ✗ |
| Qwen3-32B-FP8 agg TP=1 | 5.1 | 4.8 | 1.06× | 1,334 | 58.4 | ✗ | ✗ |
| Qwen3-32B-FP8 agg TP=4 | 2.2 | 6.6 | 0.34× | 1,905 | 176.5 | ✗ | ✗ |
| Qwen3-32B-FP8 disagg | 1.2 | 3.4 | 0.35× | 3,353 | 304.0 | ✗ | ✗ |

![AIC predicted vs observed peak throughput](figures/fig5-aic-vs-observed.png)

*Ratios shown above each observed bar are observed / AIC predicted.*

**AIC prediction accuracy at AIC's predicted concurrency:**

The most direct accuracy test compares observed and predicted values at the concurrency AIC itself predicts will be needed. Where AIC's predicted concurrency was not tested exactly, the nearest tested level is used.

| Config | AIC conc | Test conc | AIC req/s | Obs req/s | Ratio | AIC TTFT | Obs TTFT | Ratio | AIC ITL | Obs ITL | Ratio |
|--------|----------|-----------|-----------|-----------|-------|----------|----------|-------|---------|---------|-------|
| Qwen3-8B agg | 48 | 40 | 34.9 | 20.9 | 0.60× | 432 ms | 550 ms | 1.27× | 28.6 ms | 43.4 ms | 1.52× |
| Qwen3-8B disagg | 10 | 8 | 28.8 | 3.8 | 0.13× | 394 ms | 1,051 ms | 2.67× | 8.9 ms | 36.0 ms | 4.04× |
| Qwen3-32B-FP8 agg (TP=4) | 8 | 8 | 6.6 | 1.9 | 0.29× | 493 ms | 1,584 ms | 3.21× | 18.2 ms | 85.6 ms | 4.70× |
| Qwen3-32B-FP8 disagg | 4 | 4 | 3.4 | 1.2 | 0.35× | 473 ms | 1,944 ms | 4.11× | 10.4 ms | 46.6 ms | 4.48× |

Qwen3-8B agg is the closest match at 0.60× throughput and 1.27× TTFT against v0.19.0 predictions. With vLLM 0.18.0 silicon data and throughput model corrections (see AIC Model Analysis), Qwen3-8B agg improves to ~0.93× throughput and ~1.19× TTFT at the same concurrency. The disaggregated configurations and all 32B-FP8 configurations diverge substantially — throughput 0.13–0.35× of prediction and TTFT 2.7–4.1× above prediction; these gaps are driven by separate factors unaffected by the silicon data update.

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

| Config | TTFT @ conc=1 | TTFT @ conc=16 | AIC predicted |
|--------|--------------|----------------|---------------|
| Qwen3-8B agg | 273 ms | 487 ms | 432 ms |
| Qwen3-8B disagg | 268 ms | 1,128 ms | 394 ms |
| Qwen3-32B-FP8 agg TP=1 | 737 ms | 1,334 ms | 315 ms |
| Qwen3-32B-FP8 agg TP=4 | 729 ms | 1,905 ms | 493 ms |
| Qwen3-32B-FP8 disagg | 731 ms | 3,353 ms | 473 ms |

- Qwen3-8B agg is the only configuration that meets the TTFT SLA at low concurrency. It exceeds 500 ms above concurrency=16.
- All Qwen3-32B-FP8 configurations exceed the 500 ms TTFT SLA at every concurrency level tested, including concurrency=1. At concurrency=1, TTFT is 729–737 ms across all 32B-FP8 topologies. AIC predicted 315–493 ms depending on topology.

### TTFT vs throughput

![TTFT vs throughput](figures/fig2-latency.png)

The near-vertical trajectories for Qwen3-32B-FP8 disagg and TP=4 show systems operating in the saturated regime from the first data point: adding concurrency produces negligible additional throughput while TTFT climbs steeply. Qwen3-8B agg traces a more gradual arc, with headroom remaining at the right edge of the plot.

### TPOT

![ITL vs concurrency](figures/fig4-tpot.png)

*Dotted horizontal lines indicate AIC predicted TPOT (decode interval). Dashed red line marks the 30 ms/tok SLA. ITL (inter-token latency) is the decode-phase interval between consecutive output tokens and matches AIC's TPOT model. guidellm's `time_per_output_token_ms` = total_latency/output_tokens and is not shown here.*

| Config | ITL @ conc=1 | ITL @ conc=16 | AIC predicted TPOT |
|--------|-------------|---------------|-------------------|
| Qwen3-8B agg | 5.7 ms | 22.7 ms | 28.6 ms |
| Qwen3-8B disagg | 5.7 ms | 79.6 ms | 8.9 ms |
| Qwen3-32B-FP8 agg TP=1 | 12.3 ms | 58.4 ms | 13.7 ms |
| Qwen3-32B-FP8 agg TP=4 | 12.3 ms | 176.5 ms | 18.2 ms |
| Qwen3-32B-FP8 disagg | 12.4 ms | 304.0 ms | 10.4 ms |

Notably, at concurrency=1 (minimal load) AIC's TPOT prediction closely matches observed ITL for Qwen3-8B agg (AIC: 5.4 ms, measured: 5.7 ms). The model diverges at higher concurrency, consistent with the concurrency=batch_size assumption (Factor 2).

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
| Qwen3-8B agg | 34.9 | 20.9 (conc=40) | 0.60× | 432 ms | 550 ms | 1.27× |
| Qwen3-8B disagg | 28.8 | 3.8 (conc=8) | 0.13× | 394 ms | 1,051 ms | 2.67× |
| Qwen3-32B-FP8 agg TP=4 | 6.6 | 1.9 (conc=8) | 0.29× | 493 ms | 1,584 ms | 3.21× |
| Qwen3-32B-FP8 agg TP=1 | 4.8 | 4.2 (conc=8) | 0.88× | 315 ms | 952 ms | 3.02× |
| Qwen3-32B-FP8 disagg | 3.4 | 1.2 (conc=4) | 0.35× | 473 ms | 1,944 ms | 4.11× |

*Note: TTFT≤500ms at ISL=9000 is not achievable for 32B dense models on a single GPU — this reflects a workload/SLA mismatch, not an AIC modelling failure specific to those configs. For 8B agg, the 500ms SLA is met up to concurrency=16 (13.9 req/s); concurrency=16 is the practical SLA-bounded operating point.*

- **Qwen3-8B agg** is the best-predicted configuration. At AIC's predicted concurrency (~40–48), throughput is 0.60× of prediction and TTFT is 1.27× above prediction — the most accurate result across all configs. The 500ms TTFT SLA is met up to concurrency=16 (13.9 req/s); the extended sweep shows saturation at ~20.9 req/s at concurrency=32–40.
- **Qwen3-8B disagg** throughput is 0.13× of prediction at AIC's predicted concurrency and TTFT is 2.67× above prediction. The decode worker saturates rapidly under concurrent load — a queuing effect AIC does not model.
- **Qwen3-32B-FP8** configurations all show TTFT well above 500ms even at concurrency=1 (730–737ms). This reflects the compute cost of prefilling 9000 tokens at 32B dense — a physics constraint unrelated to AIC accuracy. TTFT accuracy is 3–4× off prediction across 32B configurations; the discrepancy is consistent with the silicon data version gap identified in Factor 1. Throughput prediction accuracy ranges from 0.29× (TP=4) to 0.88× (TP=1).
- **Qwen3-32B-FP8 agg TP=4**, AIC's top-1 recommendation, delivers 2.3× lower throughput than TP=1×8, contrary to AIC's prediction (6.6 vs 4.8 req/s). At concurrency=1, per-instance TTFT is nearly identical (TP=4: 729ms, TP=1: 737ms). The throughput gap at higher concurrency is consistent with the replica count difference: TP=4×2 provides 2 instances against TP=1×8's 8.

---

## AIC Model Analysis

The discrepancies between AIC predictions and observed results were investigated by reading the AIC source code. Four factors were identified, each representing a gap that could be addressed in the AIC codebase.

### Factor 1 — Silicon data version mismatch (partially addressed)

AIC predictions are computed from lookup tables in `systems/{hw}/data/vllm/{version}/` — CSV files of measured per-operation latencies (GEMM, context attention, generation attention) at various batch sizes and TP sizes. The database version used was vLLM 0.19.0.

The deployed stack runs `vLLM v0.18.0+rhaiv.0` with a distinct compilation profile: the startup log shows `compilation_config.mode = VLLM_COMPILE` (torch.inductor), `compile_ranges_endpoints = [11012]`, and custom FP8 fusion passes. AIC's silicon tables were captured against vLLM 0.19.0, which has a different compilation configuration — different attention kernels, different GEMM tile sizes, and different FP8 fusion passes. All latency entries in the database are therefore measured under different kernel behaviour than what runs in this deployment.

**Partial fix applied:** vLLM 0.18.0 silicon data was collected on H200 SXM hardware using the deployed image with application clocks locked (SM=1980 MHz, MEM=3201 MHz) and added to the AIC database. The improvement for Qwen3-8B agg is significant:

| Metric | v0.19.0 | v0.18.0 | Observed |
|--------|---------|---------|----------|
| TTFT (top-1) | 432 ms | **488 ms** | 487 ms ✓ |
| TPOT/ITL (top-1) | 28.6 ms | **23.3 ms** | 22.7 ms ✓ |
| Concurrency | 48 | **40** | 40 (saturation) |
| req/s | 34.9 | **30.1** | 20.9 (sat.) / 13.9 (SLA) |

TTFT and TPOT predictions are now accurate to within 5% for Qwen3-8B. Throughput remains 44% above the observed saturation value — the remaining gap is consistent with Factor 2.

**Remaining gap:** The 0.18.0 collection only captured BF16 attention data. Qwen3-32B-FP8 requires FP8 generation attention measurements, which must be collected in a separate pass with FP8 KV cache enabled.

### Factor 2 — Concurrency is equated to batch size (under investigation)

In `vllm_backend.py`, the predicted concurrency is hardcoded as `concurrency = batch_size × pp_size × attention_dp_size`. AIC treats its chosen batch size as the perpetually-full in-flight request count, implying 100% server utilisation at all times. No queue-depth or saturation model exists.

An initial analysis (two data points at ISL=9000) produced an empirical decode overhead correction formula. Subsequent collection of a broader dataset (see Overhead Characterisation Study below) revealed that AIC's TPOT model is already accurate to within ±1ms mean across a 40-point (ISL, batch size) grid, and that the apparent correction at ISL=9000 was not reproducible across the full ISL range. The correction formula has been withdrawn pending a principled analysis.

**Remaining work:** the throughput gap is likely explained by the concurrency model — AIC equates its batch size to the perpetually-filled request queue, overestimating achievable throughput at high concurrency. A Little's Law consistency cap (`throughput ≤ b / request_latency`) has been prototyped on branch `fix/throughput-queueing-model` and reduces the Qwen3-8B agg throughput prediction from 34.9 req/s to 24.4 req/s against observed saturation of 20.9 req/s. Further validation is required before this is opened as an upstream PR.

### Factor 3 — Disaggregated routing overhead is not modelled (improvable)

For disaggregated configurations, `picking.py` applies fixed degradation constants of 0.9 (prefill) and 0.92 (decode) to account for pipeline bubbles, and nothing else. There is no term for:

- KV cache transfer latency between prefill and decode workers
- Request routing latency through llm-d's scheduler and EPP
- Network round-trip for the prefill→decode handoff

The Qwen3-8B disagg TTFT exceeds the 500 ms SLA at concurrency=4 (525 ms) and reaches 1,128 ms at concurrency=16. AIC predicted 394 ms at its chosen operating point. AIC has no model for routing or KV-transfer overhead, which are plausible contributors to the observed gap. The disagg throughput is 6–10× below prediction across both models.

**Fix:** the observed TTFT blowout is queuing-driven — at concurrency=1, disagg TTFT (268ms) is marginally lower than agg (273ms), so there is no measurable fixed per-request routing overhead. The bottleneck is the decode worker saturating under concurrent load. A queueing model at the decode worker (M/M/1 with service time derived from silicon-measured decode step latency) would capture this behaviour. Separately, a KV-transfer latency term could be added for deployments where network bandwidth is a constraint.

### Factor 4 — TPOT metric mismatch and pipeline-bubble correction (largely explained)

Investigation revealed that guidellm's `time_per_output_token_ms` = `total_request_latency / output_tokens`, which includes TTFT. AIC's TPOT model corresponds instead to `inter_token_latency_ms` (ITL). Correcting this comparison reduces the apparent TPOT gap from 5–8× to 2–5× against v0.19.0 data.

With v0.18.0 silicon data, AIC TPOT predictions at ISL=9000 for Qwen3-8B are accurate to within 5% at b=1 (AIC 5.4 ms, observed 5.7 ms). The Overhead Characterisation Study confirms this accuracy extends across ISL=64–8192 and b=4–64, with a mean absolute error of 0.99 ms. No additional correction to the TPOT model is required to explain the observed TPOT values.

The pipeline-bubble correction `num_mix_steps_for_tpot_calc = max(1, num_mix_steps − 3)` (`git blame`: commit `5554d2eb`, 6 Nov 2025, initial H100-only vLLM backend) clamps most batch sizes to a single mix step for TPOT calculation at ISL=9000. Its physical meaning for H200 SXM is not documented and has not been verified. The accuracy of AIC's TPOT predictions over the characterisation grid suggests the current implementation produces reasonable results, but the constant should be reviewed for correctness before extending the model to other hardware.

---

### Remaining work

The following items have been identified for future investigation:

**Factor 1 — FP8 context attention data (PR #1142):** The vLLM 0.18.0 silicon data PR is open. A collector bug was fixed (vLLM ≥0.11.0 requires the caller to pass an FP8-dtype query tensor when `supports_quant_query_input=True`; the output buffer must remain BF16). All six op tables are now complete and the PR awaits review.

**Factor 2 — Concurrency model:** A Little's Law throughput cap is prototyped on `fix/throughput-queueing-model`. The Overhead Characterisation Study has ruled out a TPOT correction as the primary fix. The remaining question is whether the throughput gap is entirely explained by the 100%-utilisation assumption or whether a queueing model is needed. Cross-validation benchmarks at ISL=512 and ISL=4096 for Qwen3-8B and a MoE model are needed.

**Factor 3 — Disaggregated queuing model:** AIC has no model for decode worker saturation under concurrent load. The Qwen3-8B disagg result (throughput 0.13× of prediction) is dominated by this effect. An M/M/1 or token-bucket model parameterised from silicon-measured decode step latency would improve disagg predictions.

**Factor 4 — Pipeline-bubble correction:** the `num_mix_steps − 3` constant should be reviewed with the original author (commit `5554d2eb`) to determine whether it is physically motivated or empirically fitted for H100 only.

**Additional data collection:** the Overhead Characterisation Study covered Qwen3-8B (dense, BF16). The same sweep should be run for Qwen3-32B-FP8 and at least one MoE model to verify that the ±1 ms TPOT accuracy generalises across model architectures and quantisation modes.

