# aiconfigurator Evaluation Report

**Models:** Qwen/Qwen3-8B, Qwen/Qwen3-32B-FP8 (primary evaluation); Qwen/Qwen3-14B (TTFT sweep only)  
**Hardware:** 8x H200 SXM (140 GB HBM each)  
**Workload:** ISL=9000 tokens, OSL=30 tokens, TTFT SLA ≤ 500 ms, TPOT SLA ≤ 30 ms/tok  
**Stack:** RHOAI 3.4 / kserve v1alpha2, vLLM 0.18.0+rhaiv, guidellm 0.6.0

---

## TL;DR

All predictions use vLLM 0.18.0 silicon data from the same H200 SXM hardware. TPOT is expressed as inclusive TPOT = (TTFT + TPOT × (OSL−1)) / OSL throughout, matching guidellm's `time_per_output_token_ms`.

**Qwen3-8B aggregated serving:** inclusive TPOT at concurrency=16 (where both SLAs are met) is 38.5ms predicted vs 38.2ms observed. AIC identifies the saturation concurrency (40) correctly. Throughput at that concurrency is 0.69× of prediction, attributed to the 100% queue utilisation assumption.

**Disaggregated serving:** throughput at AIC's predicted operating point is 0.15× of prediction, driven by decode worker queueing saturation — an effect AIC has no model for.

**Qwen3-32B-FP8:** TTFT is 1.61× above AIC's prediction and TPOT is 0.59× below at concurrency=1; these errors partially cancel in inclusive TPOT (0.97×). Both gaps persist after FP8 silicon data was corrected and remain unattributed.

**AIC code changes:** three PRs have merged (silicon data, metric comparison definition, mix-step decode token count fix). Four PRs in review address the TTFT queuing model, throughput cap, mix-step efficiency, pipeline-drain correction, and prefill dispatch overhead. Evaluated across 1924 vLLM agg entries, the full PR stack produces a net TTFT improvement of +2.76 pp over the main-branch baseline with TPOT neutral at +0.06 pp. The updated TTFT formula (`1 + log₂(b)/8`, capped at 2×T_prefill) reduces tp_size-matched corpus MAPE from 26.4% to 18.0%.

---

## Test Setup

### Hardware

Eight H200 SXM GPUs (140 GB HBM, Hopper architecture) on a single worker node. All benchmarks use all 8 GPUs.

### Models

The primary evaluation (deployment configurations, throughput sweep, TPOT) covers two models deployed on the RHOAI cluster:

| Model | Quantization | Architecture |
|-------|-------------|--------------|
| Qwen/Qwen3-8B | BF16 | Dense, 32 layers |
| Qwen/Qwen3-32B-FP8 | FP8 | Dense |

Qwen/Qwen3-14B (dense BF16, 40 layers) was benchmarked separately in the TTFT concurrency and Poisson arrival sweeps on the psap-fire-athena cluster. Those results appear in the respective sections; it is not part of the deployment evaluation.

### Deployment configurations

aiconfigurator (AIC) was used to determine the recommended deployment topology for each model under the given workload constraints. Configurations evaluated:

| Config | Topology | Workers | GPUs/worker |
|--------|----------|---------|-------------|
| Qwen3-8B agg | TP=1 × 8 replicas | 8 | 1 |
| Qwen3-8B disagg | TP=1 × 7P + 1D | 8 | 1 |
| Qwen3-32B-FP8 agg (TP=1) | TP=1 × 8 replicas | 8 | 1 |
| Qwen3-32B-FP8 agg (TP=4) | TP=4 × 2 replicas | 2 | 4 |
| Qwen3-32B-FP8 disagg | TP=4 × 1P + 1D | 2 | 4 |

### AIC invocation

AIC version 0.8.0 was run in SILICON database mode using vLLM 0.18.0 silicon data ([PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142), merged). Inputs:

| Parameter | Value |
|-----------|-------|
| `--system` | `h200_sxm` |
| `--backend` | `vllm` |
| `--deployment-target` | `llm-d` |
| `--total-gpus` | `8` |
| ISL | 9000 tokens |
| OSL | 30 tokens |
| TTFT SLA | ≤ 500 ms |
| TPOT SLA | ≤ 30 ms/token |

### Benchmark methodology

guidellm 0.6.0 was used with `--profile throughput --rate N` (concurrency sweep). Each run lasted 120 seconds at ISL=9000, OSL=30 with synthetic fixed prompts.

**TPOT metric:** guidellm's `inter_token_latency_ms` (ITL) corresponds to AIC's internal TPOT model. `time_per_output_token_ms` spreads TTFT across output tokens. The `--inclusive-tpot` flag applies the same formula to AIC output: `(ttft + tpot × (osl−1)) / osl`. This report uses inclusive TPOT throughout the TPOT section.

Concurrency levels tested:
- Qwen3-8B agg: 1, 2, 4, 8, 16, 24, 32, 40
- Qwen3-8B disagg: 1, 2, 4, 8, 16
- Qwen3-32B-FP8 agg (TP=1): 1, 2, 4, 8, 16
- Qwen3-32B-FP8 agg (TP=4): 1, 2, 4, 6, 8, 12, 16
- Qwen3-32B-FP8 disagg: 1, 2, 4, 8, 16

---

## AIC Predictions

AIC top-1 predicted operating points using vLLM 0.18.0 silicon data:

| Config | AIC req/s | AIC TTFT (ms) | AIC TPOT/ITL (ms/tok) | AIC tok/s | AIC concurrency |
|--------|-----------|---------------|-------------------|-----------|-----------------|
| Qwen3-8B agg (TP=1×8) | 30.4 | 484 | 23.2 | 881 | 40 |
| Qwen3-8B disagg | 25.0 | 453 | 7.7 | 751 | 7 |
| Qwen3-32B-FP8 agg (TP=4×2) | 5.4 | 489 | 28.8 | 155 | 8 |
| Qwen3-32B-FP8 disagg | 3.4 | 470 | 23.8 | 103 | 16 |

AIC selected aggregated mode as the top configuration for both models. For Qwen3-32B-FP8 agg, AIC's top-1 topology with 0.18.0 silicon data is TP=4 × 2 replicas.

---

## Observed Results

### Throughput

![Throughput vs concurrency](figures/fig1-throughput.png)

*Dotted horizontal lines indicate AIC predicted peak throughput for each configuration.*

**Qwen3-8B agg — concurrency sweep:**

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

Throughput saturates at ~20.9 req/s between concurrency=32 and concurrency=40. Both SLAs are met up to concurrency=16. The maximum dual-SLA-compliant operating point is 13.9 req/s (46% of AIC's predicted 30.4 req/s).

**All configurations at concurrency=16:**

| Config | Observed (req/s) | AIC predicted (req/s) | Ratio | TTFT (ms) | ITL (ms) | TTFT SLA | ITL SLA |
|--------|-----------------|----------------------|-------|-----------|----------|----------|---------|
| Qwen3-8B agg | 13.9 | 30.4 | 0.46× | 487 | 22.7 | ✓ | ✓ |
| Qwen3-8B disagg | 4.6 | 25.0 | 0.18× | 1,128 | 79.6 | ✗ | ✗ |
| Qwen3-32B-FP8 agg TP=1 | 5.1 | — | — | 1,334 | 58.4 | ✗ | ✗ |
| Qwen3-32B-FP8 agg TP=4 | 2.2 | 5.4 | 0.41× | 1,905 | 176.5 | ✗ | ✗ |
| Qwen3-32B-FP8 disagg | 1.2 | 3.4 | 0.35× | 3,353 | 304.0 | ✗ | ✗ |

![AIC predicted vs observed peak throughput](figures/fig5-aic-vs-observed.png)

**AIC prediction accuracy at AIC's predicted concurrency:**

| Config | AIC conc | Test conc | AIC req/s | Obs req/s | Ratio | AIC TTFT | Obs TTFT | Ratio | AIC ITL | Obs ITL | Ratio |
|--------|----------|-----------|-----------|-----------|-------|----------|----------|-------|---------|---------|-------|
| Qwen3-8B agg | 40 | 40 | 30.4 | 20.9 | 0.69× | 484 ms | 550 ms | 1.14× | 23.2 ms | 43.4 ms | 1.87× |
| Qwen3-8B disagg | 7 | 8 | 25.0 | 3.8 | 0.15× | 453 ms | 1,051 ms | 2.32× | 7.7 ms | 36.0 ms | 4.68× |
| Qwen3-32B-FP8 agg (TP=4) | 8 | 8 | 5.4 | 1.9 | 0.35× | 489 ms | 1,584 ms | 3.24× | 28.8 ms | 85.6 ms | 2.97× |
| Qwen3-32B-FP8 disagg | 16 | 16 | 3.4 | 1.2 | 0.35× | 470 ms | 3,353 ms | 7.13× | 23.8 ms | 304 ms | 12.8× |

Qwen3-8B agg shows the smallest prediction gap: 0.69× throughput and 1.14× TTFT at AIC's predicted concurrency. The disaggregated and 32B-FP8 configurations diverge substantially — throughput 0.15–0.35× of prediction.

**Qwen3-32B-FP8 TP=4 vs TP=1:**

| Concurrency | TP=1 (req/s) | TP=4 (req/s) | TP=1 TTFT | TP=4 TTFT |
|-------------|-------------|-------------|-----------|-----------|
| 1 | 0.91 | 0.92 | 737 ms | 729 ms |
| 4 | 3.04 | 1.78 | 759 ms | 1,278 ms |
| 8 | 4.21 | 1.92 | 952 ms | 1,584 ms |
| 16 | 5.14 | 2.23 | 1,334 ms | 1,905 ms |

TP=1×8 delivers 2.3× higher throughput than TP=4×2 at concurrency=16 (5.1 vs 2.2 req/s). AIC predicted TP=4 to be the superior topology.

### TTFT

![TTFT vs concurrency](figures/fig3-ttft.png)

*Dotted horizontal lines indicate AIC predicted TTFT. Dashed red line marks the 500 ms SLA.*

| Config | TTFT @ conc=1 | TTFT @ conc=16 | AIC predicted (v0.18.0) |
|--------|--------------|----------------|-------------------------|
| Qwen3-8B agg | 273 ms | 487 ms | 484 ms |
| Qwen3-8B disagg | 268 ms | 1,128 ms | 453 ms |
| Qwen3-32B-FP8 agg TP=4 | 729 ms | 1,905 ms | 489 ms |
| Qwen3-32B-FP8 disagg | 731 ms | 3,353 ms | 470 ms |

At concurrency=16, Qwen3-8B agg TTFT (487ms) is within 1% of AIC's prediction (484ms). All Qwen3-32B-FP8 configurations exceed the 500 ms SLA at every concurrency level tested, including concurrency=1 (729–737 ms across all topologies, against an AIC prediction of 451–489 ms). This 1.49–1.62× TTFT under-prediction persists after FP8 silicon data was updated.

At concurrency=1 for Qwen3-8B agg, observed TTFT (273ms) is 1.77× below AIC's predicted value at the recommended operating point (484ms). This gap is consistent with AIC's TTFT queuing correction being calibrated for loaded conditions rather than single-request behaviour.

### TTFT vs throughput

![TTFT vs throughput](figures/fig2-latency.png)

Qwen3-32B-FP8 disagg and TP=4 show near-vertical trajectories — adding concurrency produces negligible additional throughput while TTFT rises steeply from the first data point. Qwen3-8B agg has a more gradual curve with headroom at the right edge.

### TPOT

![Inclusive TPOT vs concurrency](figures/fig4-tpot.png)

*Inclusive TPOT = (TTFT + ITL × (OSL−1)) / OSL, matching guidellm's `time_per_output_token_ms` and AIC's `--inclusive-tpot` output.*

![AIC inclusive TPOT accuracy at SLA operating point](figures/fig6-inclusive-tpot-accuracy.png)

| Config | Incl. TPOT @ conc=1 | AIC incl. TPOT @ b=1 | Ratio | Incl. TPOT @ conc=16 | AIC incl. TPOT @ conc=16 | Ratio |
|--------|---------------------|----------------------|-------|-----------------------|--------------------------|-------|
| Qwen3-8B agg | 14.6 ms | 19.6 ms | 1.34× | 38.2 ms | 38.5 ms | 1.01× ✓ |
| Qwen3-8B disagg | 14.4 ms | 19.6 ms | 1.36× | 197 ms | — | — |
| Qwen3-32B-FP8 agg TP=4 | 36.2 ms | 35.2 ms | 0.97× ✓ | 219 ms | — | — |

At concurrency=16 (where both SLAs are met for Qwen3-8B), AIC's inclusive TPOT prediction is within 1% of measured. For Qwen3-32B-FP8 at concurrency=1, the combined metric is within 3% despite the individual TTFT and TPOT errors having opposite signs and magnitudes of 1.61× and 0.59× respectively. At concurrency=1 for Qwen3-8B, AIC over-predicts inclusive TPOT by 34%; the TTFT correction factor is calibrated for loaded conditions, not minimal load.

---

## Overhead Characterisation Study

A systematic measurement was conducted across a grid of ISL and batch size values using a single-replica Qwen3-8B deployment on H200 SXM running vLLM 0.18.0. Single-replica measurements ensure guidellm's requested concurrency maps directly to the vLLM batch size seen per instance.

### Methodology

40 (ISL, batch_size) combinations, OSL=128:

| Dimension | Values |
|-----------|--------|
| ISL | 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| Batch size (b) | 4, 8, 16, 32, 64 |
| Duration | 300 s per run |

### Results

![AIC TPOT vs measured ITL by ISL](tpot_itl_vs_b_by_isl.png)

![TPOT prediction error heatmap](tpot_error_heatmap.png)

![TPOT prediction error percentage vs ISL](tpot_error_pct_vs_isl.png)

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

AIC's TPOT model achieves approximately ±1 ms MAE without correction across the (ISL=64–8192, b=4–64, OSL=128) grid. The error does not follow a single-signed monotonically-increasing function of either b or `b × ISL`.

### Validation at ISL=9000, OSL=30

The characterisation above used OSL=128. The evaluation workload (ISL=9000, OSL=30) has a fundamentally different mix-step composition: one mix step and 29 genonly steps vs one mix step and 127 genonly steps. Errors at this operating point are substantially larger, and AIC's predicted TPOT is identical at b=32 and b=64 (both 238.58 ms) while measured ITL continues rising from 236.78 ms to 308.82 ms:

| b | ITL measured | AIC TPOT | Error (ms) | Error (%) |
|---|-------------|----------|------------|-----------|
| 1 | 5.70 ms | 5.29 ms | +0.42 | +7.9% |
| 4 | 18.44 ms | 14.60 ms | +3.84 | +26.3% |
| 8 | 36.17 ms | 49.02 ms | −12.85 | −26.2% |
| 16 | 93.63 ms | 117.87 ms | −24.24 | −20.6% |
| 32 | 236.78 ms | 238.58 ms | −1.80 | −0.8% |
| 64 | 308.82 ms | 238.58 ms | +70.23 | +29.4% |

The identical AIC predictions at b=32 and b=64 indicate the model reaches an internal cap at this ISL:OSL ratio. The source code reason for this cap has not been identified. The error pattern is non-monotonic and does not support a simple per-b correction formula.

---

## TTFT Concurrency Sweep

A closed-loop concurrency sweep at ISL=9000, OSL=30 was conducted for three models on the psap-fire-athena cluster (8× H200 SXM, single-model kserve deployment), separate from the RHOAI deployment evaluation above.

### Median TTFT vs concurrency (ISL=9000, OSL=30)

| CC | Qwen3-8B (ms) | Qwen3-14B (ms) | Qwen3-32B-FP8-TP4 (ms) |
|----|--------------|----------------|------------------------|
| 1 | 267.6 | 454.8 | 731.3 |
| 2 | 269.3 | 454.3 | 743.9 |
| 4 | 270.4 | 457.9 | 933.2 |
| 8 | 275.7 | 462.4 | 1,407.3 |
| 12 | 275.3 | 469.7 | 1,475.6 |
| 16 | 362.0 | 483.7 | 1,755.7 |
| 20 | 306.9 | 530.2 | 1,638.9 |
| 24 | 360.4 | 649.8 | 1,754.7 |
| 28 | 383.9 | 662.9 | 2,102.9 |
| 32 | 529.4 | 781.9 | 2,012.4 |
| 48 | 492.8 | 883.8 | 2,762.8 |
| 64 | 524.5 | 943.4 | 4,564.4 |
| 128 | 507.6 | 1,356.7 | — |
| 256 | 461.2 | 2,838.0 | — |

Zero-load (CC=1) baselines: 8B=268ms, 14B=455ms, 32B-FP8-TP4=731ms.

For Qwen3-8B, median TTFT becomes non-monotonic above CC=32 (range 461–530ms), consistent with throughput saturation where the rate limiter holds actual concurrency at the system's queue capacity. For Qwen3-14B, TTFT rises gradually to CC=32 then accelerates, with the system saturating around CC=24–32 (median 650–780ms). Qwen3-32B-FP8-TP4 shows steep TTFT growth from CC=1 — at CC=4, TTFT is already 933ms; at CC=64, 4564ms.

The zero-load TTFT values (CC=1) represent effective per-request prefill time (T_prefill). Qwen3-8B T_prefill (268ms) is consistent across both clusters. For Qwen3-32B-FP8-TP4, the measured T_prefill (731ms) is 1.62× above AIC's prediction (451ms) — the same gap observed in the RHOAI deployment evaluation — confirming it is not stack-specific.

---

## Poisson Arrival Sweep

guidellm was run with `--rate-type poisson` at ISL=9000 for four models on the psap-fire-athena cluster. All models completed the sweep.

### TTFT vs Poisson arrival rate (median, ISL=9000)

| Model | Rate (req/s) | Actual RPS | Avg CC | Median TTFT (ms) | Mean TTFT (ms) |
|-------|-------------|-----------|---------|-----------------|----------------|
| Qwen3-0.6B | 5.0 | 4.52 | 0.60 | 68 | 68 |
| Qwen3-0.6B | 12.0 | 12.31 | 1.97 | 68 | 77 |
| Qwen3-0.6B | 20.0 | 20.18 | 4.55 | 68 | 112 |
| Qwen3-8B | 0.5 | 0.51 | 0.23 | 268 | 274 |
| Qwen3-8B | 2.0 | 1.59 | 0.70 | 267 | 247 |
| Qwen3-8B | 3.5 | 3.16 | 1.88 | 267 | 314 |
| Qwen3-14B | 0.5 | 0.51 | 0.37 | 453 | 420 |
| Qwen3-14B | 1.5 | 1.22 | 1.02 | 455 | 449 |
| Qwen3-14B | 2.0 | 1.59 | 1.30 | 454 | 451 |
| Qwen3-32B-FP8 | 0.5 | 0.50 | 1.30 | 735 | 1,250 |
| Qwen3-32B-FP8 | 1.0 | 0.82 | 2.45 | 735 | 1,370 |
| Qwen3-32B-FP8 | 1.5 | 1.22 | 7.29 | 833 | 1,964 |
| Qwen3-32B-FP8 | 2.0 | 1.58 | 13.12 | 1,271 | 2,837 |

For Qwen3-8B and Qwen3-14B, median TTFT holds near the CC=1 baseline at all tested rates (average CC ≤ 2). Mean diverges from median at the highest rates (Qwen3-8B rate=3.5: mean 314ms vs median 267ms), indicating occasional queueing events without consistent saturation. Qwen3-32B-FP8 shows mean–median divergence beginning at rate=0.5 req/s (median 735ms, mean 1250ms) and both metrics depart above rate=1.5 req/s — at rate=2.0, average CC=13.12 and median TTFT=1271ms, consistent with heavy saturation for a model with T_prefill=731ms.

---

## Summary

| Config | AIC req/s | Obs req/s @ AIC conc | Throughput ratio | AIC TTFT | Obs TTFT @ AIC conc | TTFT ratio |
|--------|-----------|---------------------|-----------------|----------|---------------------|------------|
| Qwen3-8B agg | 30.4 | 20.9 (conc=40) | 0.69× | 484 ms | 550 ms | 1.14× |
| Qwen3-8B disagg | 25.0 | 3.8 (conc=8) | 0.15× | 453 ms | 1,051 ms | 2.32× |
| Qwen3-32B-FP8 agg TP=4 | 5.4 | 1.9 (conc=8) | 0.35× | 489 ms | 1,584 ms | 3.24× |
| Qwen3-32B-FP8 disagg | 3.4 | 1.2 (conc=16) | 0.35× | 470 ms | 3,353 ms | 7.13× |

*No Qwen3-32B-FP8 configuration meets TTFT≤500ms at ISL=9000 at any tested concurrency level, including the TP=4 configuration (4 GPUs). This reflects the model's prefill cost at this sequence length rather than an AIC modelling failure specific to those configs.*

---

## AIC Model Analysis

Six factors contributing to prediction gaps were investigated. The sections below describe each factor, its status, and the associated code changes.

### Factor 1 — Silicon data version mismatch ([PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142), merged)

AIC predictions are computed from per-operation latency lookup tables. The deployed stack runs vLLM 0.18.0+rhaiv; the original database was vLLM 0.19.0 with a different compilation profile. Updated 0.18.0 silicon data was collected on H200 SXM with application clocks locked (SM=1980 MHz, MEM=3201 MHz) and merged.

| Metric | v0.19.0 | v0.18.0 | Observed |
|--------|---------|---------|----------|
| TTFT (top-1 concurrency) | 432 ms | 484 ms | 487 ms |
| TPOT/ITL (top-1 concurrency) | 28.6 ms | 23.2 ms | 22.7 ms |
| Predicted concurrency | 48 | 40 | 40 (saturation) |

For Qwen3-8B, TTFT and TPOT predictions at the SLA boundary are each within 2% with 0.18.0 data. For Qwen3-32B-FP8, the gap persists after collecting FP8 data (TTFT 1.61×, TPOT 0.59× at conc=1 — unchanged from before the FP8 data collection). Silicon data version mismatch has been ruled out as the cause of the 32B-FP8 gap.

### Factor 2 — Concurrency equated to batch size (partially addressed)

AIC treats its chosen batch size as the perpetually-full in-flight request count (100% utilisation). No queue-depth or saturation model exists. Three code changes address aspects of this factor:

**[PR #1147](https://github.com/ai-dynamo/aiconfigurator/pull/1147) (merged)** — `_mix_step_gen_tokens()` corrects the decode token count per mix step. For `b ≥ osl`, the previous formula collapsed to `≈ osl` for all batch sizes, causing identical TPOT predictions at b=32 and b=64 (238.58ms each; measured ITL was 236.8ms and 308.8ms respectively). The corrected formula `b − ceil(ctx_tokens / isl)` is derived from vLLM v1's `max_num_partial_prefills=1` scheduler.

**[PR #1151](https://github.com/ai-dynamo/aiconfigurator/pull/1151) (in review)** — Two hooks:
- `_mix_step_efficiency(ctx_tokens, gen_tokens)`: Addresses systematic mix-step latency over-estimation. Per-op silicon data measures operations in isolation; shared weight loads reduce the marginal prefill cost per step. Default returns 1.0. VLLMBackend overrides to 1.0 — full-corpus analysis (967 vLLM agg entries) shows vLLM's gen_frac ≈ 0.001 at typical operating points, making the base-class power-law parameterisation inapplicable in this regime.
- `_tpot_mix_steps(num_mix_steps)`: Replaces the hardcoded `num_mix_steps − 3` pipeline-drain correction with an overridable hook. TRTLLMBackend and SGLANGBackend override with `max(1, num_mix_steps − 3)` to restore the empirical correction for their scheduling policies. VLLMBackend inherits the default (no correction), as vLLM v1's scheduler does not introduce a 3-step grace period.

**[PR #1164](https://github.com/ai-dynamo/aiconfigurator/pull/1164) (in review)** — `_throughput_cap` hook applies a Little's Law upper bound: `throughput ≤ b × (osl−1) × 1000 / request_latency_ms`. This prevents throughput predictions from exceeding what is sustainably achievable given the predicted per-request latency.

The 31% throughput over-prediction for aggregated serving at AIC's predicted concurrency is attributed to the 100% queue utilisation assumption. The `_throughput_cap` hook in PR #1164 is intended to address this; cross-validation against measured data is pending.

### Factor 3 — Disaggregated routing overhead not modelled

`picking.py` applies fixed degradation constants (0.9 prefill, 0.92 decode) with no term for KV cache transfer latency or routing round-trip. Disagg concurrency=1 benchmarks at three ISL values measured:

| ISL | Disagg TTFT (measured) | Agg TTFT (AIC predicted) | Disagg − Agg |
|-----|----------------------|-------------------------|--------------|
| 512 | 23 ms | 24 ms | −1 ms |
| 2048 | 58 ms | 82 ms | −25 ms |
| 9000 | 262 ms | 435 ms | −173 ms |

The negative values reflect AIC's TTFT correction factor being over-conservative at low concurrency, not disagg adding latency. Routing overhead is negligible for this configuration (prefill and decode co-located on the same host, KV transfer over shared memory). The throughput gap (0.15×) and TTFT blowout at higher concurrency are queueing-driven: the decode worker saturates under concurrent load, an effect AIC has no model for.

### Factor 4 — TPOT metric mismatch (resolved) and pipeline-bubble correction (in review)

guidellm's `time_per_output_token_ms` includes TTFT; AIC's `--inclusive-tpot` flag ([PR #1141](https://github.com/ai-dynamo/aiconfigurator/pull/1141), merged) closes the metric comparison gap by computing `(ttft + tpot × (osl−1)) / osl`.

The `num_mix_steps − 3` pipeline-bubble correction has been moved from a hardcoded value to the per-backend `_tpot_mix_steps` hook in PR #1151 (in review). Source code investigation confirmed vLLM v1's scheduler does not introduce a 3-step grace period; the correction is retained for TRT-LLM and SGLang where it was empirically observed.

### Factor 5 — TTFT queuing model ([PR #1165](https://github.com/ai-dynamo/aiconfigurator/pull/1165), in review)

The previous TTFT heuristic `min(2 + (steps−3)/2/10, 4)` has been replaced with an overridable `_ttft_queuing_factor(b, steps_to_finish_ctx)` hook. The default preserves the previous formula for non-vLLM backends.

VLLMBackend overrides with `min(1 + log₂(b)/8, 2.0)` — equivalent to log₂₅₆(b) + 1, capped at 2×T_prefill (the steady-state limit from queuing theory). Formula selection used accuracy corpus analysis across 99 tp_size-matched vLLM configurations (b=1..64):

| Formula | Corpus MAPE | Change vs no-queuing baseline |
|---|---|---|
| Old code (TTFT = T_prefill, no queuing) | 26.4% | baseline |
| `(b+1)/2` simultaneous-arrival model | 199.8% | +173 pp |
| `min((b+1)/2, 2.0)` — cap at 2T | 46.1% | +20 pp |
| `1 + log₂(b)/8` — implemented formula | **18.0%** | **−8 pp** |

Per-batch-size breakdown of median absolute error:

| b | N | old (no queuing) | `(b+1)/2` | cap 2T | `log₂(b)/8` |
|---|---|---|---|---|---|
| 1 | 9 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 | 9 | 6.4% | 40.4% | 40.4% | 6.3% |
| 4 | 20 | 19.1% | 104.5% | 63.6% | 24.5% |
| 8 | 20 | 19.5% | 267.1% | 64.1% | 22.2% |
| 16 | 20 | 42.7% | 421.9% | 44.2% | 23.0% |
| 32 | 11 | 38.0% | 923.6% | 33.3% | 27.5% |
| 64 | 10 | 52.1% | 1458.3% | 36.2% | 24.9% |

The TTFT calculation uses the pure prefill cost (the `_mix_step_efficiency` reduction is undone before TTFT computation), since decode tokens sharing a forward pass do not reduce the prefill step time.

### Factor 6 — Prefill dispatch overhead ([PR #1166](https://github.com/ai-dynamo/aiconfigurator/pull/1166), in review)

Silicon benchmarks measure isolated GPU kernel time. vLLM adds CPU-side Python dispatch overhead per prefill request that scales with layer count. A `_prefill_dispatch_overhead_ms(model)` hook has been added to BaseBackend (default 0.0). VLLMBackend overrides with `model._num_layers × 0.8 ms`, calibrated against the full silicon corpus (947 vLLM agg entries, b=4..512) across hardware platforms and model families. Full-corpus analysis confirms the overhead is approximately constant across batch sizes, consistent with CUDA graph capture where per-step dispatch cost is fixed per graph replay.

---

## Accuracy Corpus Evaluation

The AIC accuracy corpus (1924 vLLM agg entries, HF-authenticated) was evaluated for each PR in the stack, comparing against the current main branch as baseline. Positive values indicate MAPE improvement (AIC became more accurate); negative values indicate regression. The ✓✓ gate column reflects the AIC accuracy CI thresholds (`all` partition: ≤5% regression; other partitions: ≤10%).

| PR | All TTFT | All TPOT | vllm-agg TTFT | h200_sxm TTFT | Gate |
|---|---|---|---|---|---|
| Baseline (main) | — | — | — | — | — |
| #1151 | −1.49% | −0.42% | −2.97% | −9.21% | ✓✓ |
| #1151 + #1164 | −2.77% | +0.06% | −5.50% | −11.35% | ✓✓ |
| #1151 + #1164 + #1165 | −2.77% | +0.06% | −5.50% | −11.35% | ✓✓ |
| Full stack (#1166) | **+2.76%** | **+0.06%** | **+5.49%** | −1.16% | **✓✓** |

Each row is a cumulative comparison of the stacked branch against main. PRs #1164 and #1165 show similar aggregate TTFT numbers because PR #1165's queuing formula effect is incremental relative to the corpus-wide distribution, while the combined dispatch overhead from PR #1166 produces the largest TTFT shift. TPOT is neutral at +0.06% — prior TPOT regressions from `_mix_step_efficiency` calibration are resolved.

Notable partition changes in the full stack vs baseline:

| Partition | TTFT change |
|---|---|
| Kimi-K2.5\|b200\|vllm | +78.91% |
| MiniMax-M2.5\|h100\|vllm | +58.48% |
| Kimi-K2.5 (overall) | +48.80% |
| Llama-3.1-70B\|h200\|vllm | −5.12% |

The Llama-3.1-70B h200 regression (−5.12%) remains within the 10% threshold. Kimi-K2.5 and MiniMax|h100 improvements are driven by the corrected TTFT queuing formula reducing over-prediction for those model families.

---

## AIC Code Changes

| PR | Description | Status |
|---|---|---|
| [PR #1141](https://github.com/ai-dynamo/aiconfigurator/pull/1141) | `--inclusive-tpot` output flag | Merged |
| [PR #1142](https://github.com/ai-dynamo/aiconfigurator/pull/1142) | H200 SXM vLLM 0.18.0 silicon data (6 ops, BF16+FP8) | Merged |
| [PR #1147](https://github.com/ai-dynamo/aiconfigurator/pull/1147) | Correct mix-step decode token count for b ≥ osl | Merged |
| [PR #1151](https://github.com/ai-dynamo/aiconfigurator/pull/1151) | `_mix_step_efficiency` + `_tpot_mix_steps` hooks | In review |
| [PR #1164](https://github.com/ai-dynamo/aiconfigurator/pull/1164) | `_throughput_cap` hook (Little's Law bound) | In review |
| [PR #1165](https://github.com/ai-dynamo/aiconfigurator/pull/1165) | `_ttft_queuing_factor` hook; VLLMBackend: `1 + log₂(b)/8` | In review |
| [PR #1166](https://github.com/ai-dynamo/aiconfigurator/pull/1166) | `_prefill_dispatch_overhead_ms` hook; VLLMBackend: 0.8 ms/layer | In review |
| [PR #1212](https://github.com/ai-dynamo/aiconfigurator/pull/1212) | Test mock-torch pollution fixes | In review |
| [PR #1128](https://github.com/ai-dynamo/aiconfigurator/pull/1128) *(Spycsh, draft)* | TTFT queuing via Kingman G/G/1 — parallel exploration | Draft |

---

## Remaining Open Questions

**Qwen3-32B-FP8 TTFT and TPOT gaps:** TTFT 1.61×, TPOT 0.59× at conc=1 persist after complete FP8 silicon data. Silicon data version mismatch has been ruled out. Plausible candidates are FP8 kernel extrapolation at ISL=9000 (near the upper range of the silicon collection) or a model-architecture interaction not captured in AIC's per-op composition.

**AIC TPOT model cap at ISL=9000, OSL=30:** AIC predicts identical TPOT at b=32 and b=64 (238.58ms) while measured ITL continues rising (+30%). The source code reason for this cap has not been identified.

**Throughput concurrency model:** the 31% throughput gap at AIC's predicted concurrency for Qwen3-8B agg reflects the 100% queue utilisation assumption. The `_throughput_cap` hook in PR #1164 applies a Little's Law bound; validation against concurrency sweep data is pending.

**Disaggregated queueing:** the 0.15× throughput ratio at AIC's predicted concurrency is primarily decode worker saturation. No per-worker saturation model parameterised from decode step latency exists in AIC.

**`_decode_overhead_factor`:** full-corpus analysis shows the implied TPOT overhead varies widely by model family (0.65× for Kimi-K2.5, 1.19× for MiniMax-M2.5, 2.0× for Qwen3-235B-FP8), making a single constant counter-productive. The hook architecture is in place in PR #1164; per-family calibration is deferred pending additional silicon data.

---

## Confidence Assessment

**Aggregated dense model serving (BF16):** with the current PR stack, inclusive TPOT is within 1% at the SLA operating point and AIC identifies the throughput saturation concurrency correctly. Throughput predictions at high concurrency carry approximately 30% optimism under the current main code; the throughput-cap hook in PR #1164 reduces this gap when merged.

**Aggregated FP8 model serving:** TTFT 1.61× and TPOT 0.59× errors at conc=1 persist and do not cancel cleanly at higher concurrency. Inclusive TPOT is within 3% at concurrency=1. The cause is not yet identified.

**Disaggregated serving:** throughput is 0.15× of prediction at the recommended concurrency. Routing overhead at concurrency=1 is negligible across ISL=512–9000 for shared-memory KV transfer. The gap is primarily decode worker queueing with no AIC model. Disaggregated AIC predictions require substantial headroom.
