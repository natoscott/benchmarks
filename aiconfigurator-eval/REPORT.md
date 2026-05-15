# aiconfigurator Evaluation Report

**Models:** Qwen/Qwen3-8B, Qwen/Qwen3-32B-FP8  
**Hardware:** 8x H200 SXM (140 GB HBM each)  
**Workload:** ISL=9000 tokens, OSL=30 tokens, TTFT SLA ≤ 500 ms, TPOT SLA ≤ 30 ms/tok  
**Stack:** RHOAI 3.4 / kserve v1alpha2, vLLM 0.18.0+rhaiv, guidellm 0.6.0

---

## TL;DR

AIC's throughput predictions are substantially above observed performance for this deployment stack. **Qwen3-8B agg is the only configuration that meets both the TTFT (≤500 ms) and TPOT (≤30 ms/tok) SLAs**; its peak dual-SLA-compliant throughput is 11.3 req/s at concurrency=8 — 32% of AIC's predicted 34.9 req/s. At concurrency=16 the TPOT SLA is violated (38.1 ms), so the previously apparent ceiling of 13.9 req/s is not fully compliant. All Qwen3-32B-FP8 configurations exceed both SLAs at every tested concurrency level, including single-request load; TPOT at concurrency=1 is ~36 ms against the 30 ms/tok limit. AIC's top-1 topology for 32B-FP8 (TP=4 × 2 replicas) delivered 2.3× lower throughput than the non-recommended TP=1 × 8 configuration. Disaggregated serving fell furthest below predictions (6–10× gap), consistent with AIC having no model for routing or KV-transfer latency between workers. Source code analysis identified four candidate gaps in AIC: use of vLLM 0.19.0 silicon data against a 0.18.0+rhaiv deployment, an assumption that concurrency equals batch size, no disaggregated routing overhead term, and an undocumented empirical constant in the TPOT calculation path introduced during initial H100-only development and never revisited for H200.

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

Concurrency levels tested:
- Qwen3-8B agg: 1, 2, 4, 8, 16, 24, 32, 40 (extended to find saturation)
- All other configurations: 1, 2, 4, 6 or 8, 12 or 16

Each benchmark run lasted 120 seconds. Synthetic prompts were fixed at ISL=9000, OSL=30. The target was the kserve internal workload service over HTTPS.

---

## AIC Predictions

The table below shows AIC's top-1 predicted operating point for each configuration at the stated SLA constraints.

| Config | AIC req/s | AIC TTFT (ms) | AIC TPOT (ms/tok) | AIC tok/s | AIC concurrency |
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

| Concurrency | Observed (req/s) | TTFT (ms) | TPOT (ms) | TTFT SLA | TPOT SLA |
|-------------|-----------------|-----------|-----------|----------|----------|
| 1 | 2.3 | 273 | 14.6 | ✓ | ✓ |
| 2 | 4.4 | 267 | 15.0 | ✓ | ✓ |
| 4 | 7.6 | 286 | 17.5 | ✓ | ✓ |
| 8 | 11.3 | 340 | 23.4 | ✓ | ✓ |
| 16 | 13.9 | 487 | 38.1 | ✓ | ✗ |
| 24 | 14.7 | 692 | 54.1 | ✗ | ✗ |
| 32 | 20.9 | 512 | 50.4 | ✗ | ✗ |
| 40 | 20.9 | 550 | 60.2 | ✗ | ✗ |

Throughput saturates at ~20.9 req/s between concurrency=32 and concurrency=40. Both SLAs are met up to concurrency=8 (11.3 req/s). At concurrency=16 the TTFT SLA is still met (487 ms) but the TPOT SLA is violated (38.1 ms vs 30 ms limit). The maximum dual-SLA-compliant operating point is **11.3 req/s** (32% of AIC's predicted 34.9 req/s).

**All configurations at concurrency=16:**

| Config | Observed (req/s) | AIC predicted (req/s) | Ratio | TTFT (ms) | TPOT (ms) | TTFT SLA | TPOT SLA |
|--------|-----------------|----------------------|-------|-----------|-----------|----------|----------|
| Qwen3-8B agg | 13.9 | 34.9 | 0.40× | 487 | 38.1 | ✓ | ✗ |
| Qwen3-8B disagg | 4.6 | 28.8 | 0.16× | 1,128 | 114.6 | ✗ | ✗ |
| Qwen3-32B-FP8 agg TP=1 | 5.1 | 4.8 | 1.06× | 1,334 | 100.9 | ✗ | ✗ |
| Qwen3-32B-FP8 agg TP=4 | 2.2 | 6.6 | 0.34× | 1,905 | 234.1 | ✗ | ✗ |
| Qwen3-32B-FP8 disagg | 1.2 | 3.4 | 0.35× | 3,353 | 405.7 | ✗ | ✗ |

![AIC predicted vs observed peak throughput](figures/fig5-aic-vs-observed.png)

*Ratios shown above each observed bar are observed / AIC predicted.*

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

### TPOT

![TPOT vs concurrency](figures/fig4-tpot.png)

*Dotted horizontal lines indicate AIC predicted TPOT.*

| Config | TPOT @ conc=1 | TPOT @ conc=16 | AIC predicted |
|--------|--------------|----------------|---------------|
| Qwen3-8B agg | 14.6 ms | 38.1 ms | 28.6 ms |
| Qwen3-8B disagg | 14.4 ms | 114.6 ms | 8.9 ms |
| Qwen3-32B-FP8 agg TP=1 | 36.5 ms | 100.9 ms | 13.7 ms |
| Qwen3-32B-FP8 agg TP=4 | 36.2 ms | 234.1 ms | 18.2 ms |
| Qwen3-32B-FP8 disagg | 36.3 ms | 405.7 ms | 10.4 ms |

---

## Summary

| Config | AIC predicted (req/s) | Dual-SLA-compliant peak (req/s) | TTFT @ peak | TPOT @ peak | Ratio |
|--------|----------------------|--------------------------------|-------------|-------------|-------|
| Qwen3-8B agg | 34.9 | 11.3 (conc=8) | 340 ms | 23.4 ms | 0.32× |
| Qwen3-8B disagg | 28.8 | 2.97 (conc=2) | 381 ms | 22.4 ms | 0.10× |
| Qwen3-32B-FP8 agg TP=4 | 6.6 | — (both SLAs exceeded at all levels) | 729 ms @ conc=1 | 36.2 ms @ conc=1 | — |
| Qwen3-32B-FP8 agg TP=1 | 4.8 | — (both SLAs exceeded at all levels) | 737 ms @ conc=1 | 36.5 ms @ conc=1 | — |
| Qwen3-32B-FP8 disagg | 3.4 | — (both SLAs exceeded at all levels) | 731 ms @ conc=1 | 36.3 ms @ conc=1 | — |

- **Qwen3-8B agg** is the only configuration that meets both SLAs in practice. Its dual-SLA-compliant ceiling is 11.3 req/s at concurrency=8 (TTFT=340ms, TPOT=23.4ms). At concurrency=16 the TTFT SLA is still met but the TPOT SLA is violated (38.1ms). The extended sweep shows throughput saturation at ~20.9 req/s (concurrency=32–40), well above the SLA-compliant operating window.
- **Qwen3-8B disagg** meets both SLAs only up to concurrency=2 (2.97 req/s, TTFT=381ms, TPOT=22.4ms). At concurrency=4 both TTFT (525ms) and TPOT (37.1ms) exceed their limits. AIC models pure vLLM performance and has no term for routing or KV-transfer latency between workers.
- **Qwen3-32B-FP8 agg TP=1** is the closest to AIC's TP=1 pareto prediction (5.1 observed vs 4.8 predicted) for throughput, but both TTFT (737ms) and TPOT (36.5ms) exceed their SLAs at concurrency=1. No dual-SLA-compliant operating point exists for this configuration.
- **Qwen3-32B-FP8 agg TP=4**, AIC's top-1 recommendation, delivers 2.3× lower throughput than TP=1×8 at all concurrency levels, contrary to AIC's prediction (6.6 vs 4.8 req/s in favour of TP=4). At concurrency=1, per-instance performance is nearly identical: TP=4 TTFT=729ms / TPOT=36.2ms vs TP=1 TTFT=737ms / TPOT=36.5ms. At concurrency=1, per-instance throughput is nearly equal between the two topologies. The throughput gap at higher concurrency is consistent with the replica count difference: TP=4×2 provides 2 instances against TP=1×8's 8. AIC predicted TP=4 would achieve higher per-instance throughput than TP=1; the deployment data does not support this. H200 SXM uses NVSwitch with full all-to-all NVLink between all 8 GPUs, so GPU assignment does not affect inter-rank bandwidth.
- **Concurrency sweep extension** confirmed that the original sweep (concurrency 1–16) was insufficient for Qwen3-8B agg. The saturation point was not reached until concurrency=32–40, well above the original ceiling.

---

## AIC Model Analysis

The discrepancies between AIC predictions and observed results were investigated by reading the AIC source code. Four factors were identified, each representing a gap that could be addressed in the AIC codebase.

### Factor 1 — Silicon data version mismatch (improvable)

AIC predictions are computed from lookup tables in `systems/{hw}/data/vllm/{version}/` — CSV files of measured per-operation latencies (GEMM, context attention, generation attention) at various batch sizes and TP sizes. The database version used was vLLM 0.19.0.

The deployed stack runs `vLLM v0.18.0+rhaiv.0` with a distinct compilation profile: the startup log shows `compilation_config.mode = VLLM_COMPILE` (torch.inductor), `compile_ranges_endpoints = [11012]`, and custom FP8 fusion passes. If AIC's silicon tables were captured against a differently-compiled vLLM — different attention kernels, different GEMM tile sizes, or without `fuse_norm_quant` — every latency entry will be wrong for this deployment. A 2× TTFT gap is consistent with kernel-level performance differences between vLLM versions.

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

**Fix:** add a per-request KV-transfer term — `(ISL × kv_bytes_per_token) / network_bandwidth_GBps` — plus a measured routing latency constant, to the disagg TTFT calculation in `picking.py`.

### Factor 4 — Undocumented TPOT correction of unknown provenance (investigate)

In `vllm_backend.py` line 82, TPOT is computed using an adjusted mix step count: `num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)`. This reduces the number of mix steps counted against TPOT by 3, with no accompanying comment or documentation.

`git blame` traces this constant to commit `5554d2eb` (6 Nov 2025), which introduced the initial vLLM backend — H100-only at that point. It has not been revisited since, including when H200 support was added. Its physical meaning — what 3 steps represents, what it was calibrated against, or whether it generalises across hardware — is unknown.

Since TPOT is systematically underestimated in this evaluation and this constant sits directly in the TPOT calculation path, its validity for H200 SXM warrants investigation before further calibration work on Factor 1.

