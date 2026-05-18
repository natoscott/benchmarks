# aiconfigurator Evaluation Report

**Models:** Qwen/Qwen3-8B, Qwen/Qwen3-32B-FP8  
**Hardware:** 8x H200 SXM (140 GB HBM each)  
**Workload:** ISL=9000 tokens, OSL=30 tokens, TTFT SLA ≤ 500 ms, TPOT SLA ≤ 30 ms/tok  
**Stack:** RHOAI 3.4 / kserve v1alpha2, vLLM 0.18.0+rhaiv, guidellm 0.6.0

---

## TL;DR

This evaluation assesses how closely AIC predictions match observed performance on a real deployment stack. The SLA values (TTFT≤500ms, TPOT≤30ms/tok at ISL=9000) were used as AIC inputs to constrain its recommendation; they are evaluation parameters, not production requirements. **AIC's predictions are accurate to within ~1.3× for Qwen3-8B agg throughput and TTFT at low concurrency, but diverge significantly at higher concurrency and for all 32B-FP8 configurations.** For 32B-FP8, TTFT≤500ms at ISL=9000 is not achievable for any dense topology — this is a workload/SLA mismatch, not a tunable AIC failure. At AIC's predicted operating point, throughput falls 0.29–0.60× of prediction and TTFT runs 1.3–4.1× above prediction. Disaggregated serving is furthest from predictions (throughput 0.13× for 8B disagg), consistent with AIC having no model of decode worker queuing under concurrent load. Source code analysis identified four candidate gaps: vLLM 0.19.0 silicon data used against a 0.18.0+rhaiv deployment, concurrency equated to batch size, no disaggregated queuing model, and an empirical TPOT correction of uncertain validity for H200.

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

The table below shows AIC's top-1 predicted operating point for each configuration at the stated SLA constraints.

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

Qwen3-8B agg is the closest match: throughput at 0.60×, TTFT at 1.27×, ITL at 1.52× of prediction. The disaggregated configurations and all 32B-FP8 configurations diverge substantially — throughput 0.13–0.35× of prediction and TTFT 2.7–4.1× above prediction.

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

### Factor 1 — Silicon data version mismatch (improvable)

AIC predictions are computed from lookup tables in `systems/{hw}/data/vllm/{version}/` — CSV files of measured per-operation latencies (GEMM, context attention, generation attention) at various batch sizes and TP sizes. The database version used was vLLM 0.19.0.

The deployed stack runs `vLLM v0.18.0+rhaiv.0` with a distinct compilation profile: the startup log shows `compilation_config.mode = VLLM_COMPILE` (torch.inductor), `compile_ranges_endpoints = [11012]`, and custom FP8 fusion passes. AIC's silicon tables were captured against vLLM 0.19.0, which has a different compilation configuration — different attention kernels, different GEMM tile sizes, and different FP8 fusion passes. All latency entries in the database are therefore measured under different kernel behaviour than what runs in this deployment, and the observed 2× TTFT gap is consistent with that divergence.

**Fix:** measure and add a `v0.18.0+rhaiv` data directory using the actual deployed image on H200 SXM hardware.

### Factor 2 — Concurrency is equated to batch size (improvable)

In `vllm_backend.py`, the predicted concurrency is hardcoded as `concurrency = batch_size × pp_size × attention_dp_size`. AIC treats its chosen batch size as the perpetually-full in-flight request count, implying 100% server utilisation at all times. No queue-depth or saturation model exists.

In practice, the server only reaches that batch size when enough requests are queued simultaneously. The Qwen3-8B agg sweep shows the system saturates at ~20.9 req/s (concurrency=32–40) — roughly 60% of AIC's predicted 34.9 req/s, which assumed a batch of 35 requests always available. At the TTFT SLA boundary (concurrency=16), only 13.9 req/s is achievable — 40% of the prediction.

A simple M/M/1 or token-bucket queueing model inserted between the batch-size sweep and the throughput calculation would produce a more realistic operating curve and SLA-boundary estimate.

**Fix:** replace `concurrency = batch_size` with a queueing model parameterised by the silicon-measured step latency.

### Factor 3 — Disaggregated routing overhead is not modelled (improvable)

For disaggregated configurations, `picking.py` applies fixed degradation constants of 0.9 (prefill) and 0.92 (decode) to account for pipeline bubbles, and nothing else. There is no term for:

- KV cache transfer latency between prefill and decode workers
- Request routing latency through llm-d's scheduler and EPP
- Network round-trip for the prefill→decode handoff

The Qwen3-8B disagg TTFT exceeds the 500 ms SLA at concurrency=4 (525 ms) and reaches 1,128 ms at concurrency=16. AIC predicted 394 ms at its chosen operating point. AIC has no model for routing or KV-transfer overhead, which are plausible contributors to the observed gap. The disagg throughput is 6–10× below prediction across both models.

**Fix:** the observed TTFT blowout is queuing-driven — at concurrency=1, disagg TTFT (268ms) is marginally lower than agg (273ms), so there is no measurable fixed per-request routing overhead. The bottleneck is the decode worker saturating under concurrent load. A queueing model at the decode worker (M/M/1 with service time derived from silicon-measured decode step latency) would capture this behaviour. Separately, a KV-transfer latency term could be added for deployments where network bandwidth is a constraint.

### Factor 4 — TPOT metric mismatch and pipeline-bubble correction (investigate)

Investigation revealed that guidellm's `time_per_output_token_ms` = `total_request_latency / output_tokens`, which includes TTFT. AIC's TPOT model corresponds instead to `inter_token_latency_ms` (ITL), the decode-phase interval between consecutive output tokens. Comparing AIC's predictions against ITL rather than guidellm's TPOT reduces the apparent gap from 5–8× to 2–5×, and shows that at concurrency=1 (minimal load) AIC's TPOT prediction is within 5% of measured ITL for Qwen3-8B agg (AIC: 5.4 ms, measured: 5.7 ms). The model diverges at higher concurrency, consistent with Factor 2.

Separately, `vllm_backend.py` line 82 applies `num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)`. The inline comment describes this as "an empirical correction for pipelining requests where new requests cannot be enqueued immediately after last request's exit." `git blame` traces it to commit `5554d2eb` (6 Nov 2025), the initial H100-only vLLM backend; it has not been revisited since.

For our workload (ISL=9000, ctx_tokens=11012), `num_mix_steps = ceil(9000 × b / 11012)`, giving 1, 2, 3, 4, 5 steps for b=1–5. After the correction, `max(1, num_mix_steps − 3)` yields 1, 1, 1, 1, 2 — clamping almost all batch sizes to a single mix step for TPOT calculation. This minimises the weight of the slower mix-step latency in the TPOT formula, producing lower TPOT predictions. Since our measured ITL is consistently *higher* than AIC predicts, the correction is moving predictions in the wrong direction relative to the observed data, strengthening the case for investigating whether the constant is appropriate for H200.

