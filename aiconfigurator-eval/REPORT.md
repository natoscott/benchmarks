# aiconfigurator Evaluation Report

**Models:** Qwen/Qwen3-8B, Qwen/Qwen3-32B-FP8  
**Hardware:** 8x H200 SXM (140 GB HBM each)  
**Workload:** ISL=9000 tokens, OSL=30 tokens, TTFT SLA ≤ 500 ms  
**Stack:** RHOAI 3.4 / kserve v1alpha2, vLLM 0.18.0+rhaiv, guidellm 0.6.0

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

aiconfigurator (AIC) was used to determine the recommended deployment topology for each model under the given workload constraints. Two serving modes were evaluated:

- **Aggregated (agg):** All GPUs used as independent replicas, each serving prefill and decode for its own requests.
- **Disaggregated (disagg):** Prefill and decode workers are separated. AIC recommended 7 prefill + 1 decode worker (8 GPUs total) for both models.

| Config | Topology | Workers | GPUs/worker |
|--------|----------|---------|-------------|
| Qwen3-8B agg | TP=1 × 8 replicas | 8 | 1 |
| Qwen3-8B disagg | TP=1 × 7P + 1D | 8 | 1 |
| Qwen3-32B-FP8 agg | TP=1 × 8 replicas | 8 | 1 |
| Qwen3-32B-FP8 disagg | TP=1 × 7P + 1D | 8 | 1 |

vLLM arguments applied to all configurations, derived from AIC output:
- `--max-model-len 10530`
- `--max-num-seqs 512`
- `--max-num-batched-tokens 11012`

### Benchmark methodology

guidellm 0.6.0 was used to drive requests at five target rates: 1, 2, 4, 8, and 16 req/s. Each run lasted 120 seconds at a fixed synthetic prompt length of 9000 prompt tokens and 30 output tokens. The target service endpoint was the kserve internal ClusterIP workload service over HTTPS/HTTP2.

---

## AIC Predictions

The table below shows the AIC top-1 predicted operating point for each configuration at the stated SLA constraints. AIC version 0.8.0, database mode SILICON, modelled against vLLM 0.19.0.

| Config | AIC req/s | AIC TTFT (ms) | AIC TPOT (ms/tok) | AIC tok/s |
|--------|-----------|---------------|-------------------|-----------|
| Qwen3-8B agg | **34.9** | 432 | 28.6 | 1,013 |
| Qwen3-8B disagg | 28.8 | 394 | 8.9 | 863 |
| Qwen3-32B-FP8 agg | 6.6 | 493 | 18.2 | 191 |
| Qwen3-32B-FP8 disagg | 3.4 | 473 | 10.4 | 103 |

AIC selected aggregated mode as the top configuration for both models on this workload.

---

## Observed Results

### Throughput

![Actual vs target throughput](figures/fig1-throughput.png)

*Dotted horizontal lines indicate AIC predicted peak throughput.*

| Config | Observed @ rate=16 | AIC predicted | Ratio |
|--------|-------------------|---------------|-------|
| Qwen3-8B agg | 13.9 req/s | 34.9 req/s | 0.40× |
| Qwen3-8B disagg | 4.6 req/s | 28.8 req/s | 0.16× |
| Qwen3-32B-FP8 agg | 5.1 req/s | 6.6 req/s | 0.77× |
| Qwen3-32B-FP8 disagg | 1.2 req/s | 3.4 req/s | 0.35× |

![AIC predicted vs observed bar chart](figures/fig5-aic-vs-observed.png)

- **Qwen3-32B-FP8 agg** achieves 77% of AIC's predicted throughput.
- **Qwen3-8B agg** at rate=16 delivers 13.9 req/s against a target of 16 req/s, indicating the system has not yet saturated at the top of this sweep.
- Both disaggregated configurations deliver substantially less throughput than their aggregated counterparts at all tested rates.

### Request latency

![Request latency vs throughput](figures/fig2-latency.png)

*Shaded bands show p95 latency. Dashed grey line marks 500 ms.*

| Config | Lat median @ rate=1 | Lat median @ rate=16 |
|--------|--------------------|-----------------------|
| Qwen3-8B agg | 444 ms | 703 ms |
| Qwen3-8B disagg | 279 ms | 4,170 ms |
| Qwen3-32B-FP8 agg | 448 ms | 2,217 ms |
| Qwen3-32B-FP8 disagg | 469 ms | 12,133 ms |

### Time to First Token (TTFT)

![TTFT vs throughput](figures/fig3-ttft.png)

*Dotted lines indicate AIC predicted TTFT. Dashed grey line marks 500 ms SLA.*

| Config | TTFT @ rate=1 | TTFT @ rate=16 | AIC predicted |
|--------|--------------|----------------|---------------|
| Qwen3-8B agg | 252 ms | 283 ms | 432 ms |
| Qwen3-8B disagg | 330 ms | 1,214 ms | 394 ms |
| Qwen3-32B-FP8 agg | 446 ms | 900 ms | 493 ms |
| Qwen3-32B-FP8 disagg | 465 ms | 3,564 ms | 473 ms |

- Qwen3-8B agg TTFT remains below 300 ms across all tested rates.
- Qwen3-32B-FP8 agg TTFT at rate=1 (446 ms) is close to AIC's prediction (493 ms).
- TTFT for both disaggregated configurations degrades sharply at higher rates.

### Time Per Output Token (TPOT)

![TPOT vs throughput](figures/fig4-tpot.png)

*Dotted lines indicate AIC predicted TPOT.*

| Config | TPOT @ rate=1 | TPOT @ rate=16 | AIC predicted |
|--------|--------------|----------------|---------------|
| Qwen3-8B agg | 8.5 ms | 23.4 ms | 28.6 ms |
| Qwen3-8B disagg | 13.3 ms | 139.0 ms | 8.9 ms |
| Qwen3-32B-FP8 agg | 24.4 ms | 73.9 ms | 18.2 ms |
| Qwen3-32B-FP8 disagg | 38.7 ms | 404.4 ms | 10.4 ms |

---

## Summary

| Config | AIC req/s | Observed peak req/s | AIC TTFT | Observed TTFT @ rate=1 |
|--------|-----------|---------------------|----------|------------------------|
| Qwen3-8B agg | 34.9 | 13.9 (not saturated) | 432 ms | 252 ms |
| Qwen3-8B disagg | 28.8 | 4.6 | 394 ms | 330 ms |
| Qwen3-32B-FP8 agg | 6.6 | 5.1 | 493 ms | 446 ms |
| Qwen3-32B-FP8 disagg | 3.4 | 1.2 | 473 ms | 465 ms |

- Qwen3-32B-FP8 agg is the closest match to AIC predictions across throughput and TTFT.
- Qwen3-8B agg TTFT is consistently below AIC's prediction at all tested rates.
- Disaggregated configurations produce lower throughput than aggregated at all tested rates for both models.
- The rate sweep does not reach saturation for Qwen3-8B agg; a higher rate ceiling is needed to determine its throughput limit.

---

## Notes

- AIC predictions were generated using vLLM 0.19.0 silicon data. Deployments ran on vLLM 0.18.0+rhaiv.
- guidellm ITL metrics returned `None` for all requests across all runs; ITL is not reported.
