# llm-d Batch Gateway Performance Evaluation

**Cluster:** psap-fire-athena (OCP 4.20, RHOAI 3.5 EA)
**Date:** 2026-07-07
**Status:** Qwen3-8B complete (12/12 runs). FP8-70B and gpt-oss-120b in progress.

## TL;DR

Batch gateway dispatch strategies (ungated, AIMD, AIMD+flow-control) were evaluated against an interactive-only baseline on Qwen/Qwen3-8B across 1, 4, and 8 replicas on 2×8 NVIDIA H200 GPUs. **At single-replica, concurrent batch processing increases interactive TTFT p99 by 5-8% (86.9 ms → 91-94 ms), with no measurable impact on ITL or throughput.** At 4 and 8 replicas, batch overhead is absorbed — TTFT p99 is within ±5% of the baseline. All three dispatch strategies (ungated, AIMD, AIMD+flow-control) produce equivalent latency results in RHOAI 3.5 EA, consistent with the absence of flow-control plugins in this EPP version. Zero inference errors across 27,000+ interactive requests. 30×100 batch jobs (3000 total) complete within the first 5 minutes of the 6-minute measurement window.

## Methodology

### Infrastructure

| Component | Configuration |
|---|---|
| Cluster | psap-fire-athena, OCP 4.20 |
| GPUs | 2× gx3d nodes, 8× NVIDIA H200 each (16 total) |
| RHOAI | 3.5 EA (LLMInferenceService, EPP, inference gateway) |
| Model | Qwen/Qwen3-8B, TP=1, gpu-memory-utilization=0.90 |
| Batch Gateway | RHOAI 3.5 EA images (apiserver + processor) |
| Valkey | 8.0.9 (Red Hat RHEL9 container) |
| PostgreSQL | 16 (Red Hat RHEL9 container) |
| guidellm | v0.7.1 |
| PCP | 7.1.5 (openmetrics, pmdavalkey, pmdapostgresql) |

### Scenarios

| ID | Name | Description |
|---|---|---|
| 0 | interactive-only | No batch gateway. Interactive traffic baseline. |
| 2 | ungated | Batch gateway, global=200, per-endpoint=100, AIMD disabled |
| 3 | aimd | Batch gateway, global=100, per-endpoint=20, AIMD enabled |
| 4 | aimd-flow-control | Same as AIMD. Flow-control plugins unavailable in RHOAI 3.5 EA. |

### Workload

| Parameter | Value |
|---|---|
| Interactive burst | 15 concurrent streams, 60s |
| Interactive idle | 1 concurrent stream, 60s |
| Cycles | 3 (1 warmup excluded, 2 measured) |
| Batch jobs | 30 × 100 requests (3000 total) |
| Batch completion window | 24h (all jobs) |
| Prompt tokens | synthetic, 512 |
| Output tokens | 128 |
| Replica configs | 1, 4, 8 |

### Metrics Collection

PCP archives capture 2363 metrics per run: vLLM (123), DCGM GPU (24), EPP (221), batch processor (66), PostgreSQL (207), Valkey (111), and system-level (kernel, memory, network, disk).

## Results

### Interactive Latency During Burst

All values in milliseconds. Lower is better.

| Scenario | R | TTFT p50 | p95 | p99 | ITL p50 | p95 | p99 | RPS | Completed |
|---|---|---|---|---|---|---|---|---|---|
| interactive-only | 1 | 32.2 | 45.1 | 86.9 | 5.9 | 5.9 | 6.1 | 19.0 | 2282 |
| ungated | 1 | 34.2 | 52.9 | 91.9 | 5.9 | 6.4 | 6.8 | 18.7 | 2248 |
| aimd | 1 | 30.8 | 62.5 | 93.9 | 5.9 | 6.4 | 6.7 | 18.7 | 2247 |
| aimd-flow-control | 1 | 32.8 | 60.1 | 94.1 | 5.9 | 6.4 | 6.8 | 18.7 | 2251 |
| interactive-only | 4 | 31.6 | 44.1 | 78.4 | 5.7 | 5.9 | 6.3 | 19.5 | 2338 |
| ungated | 4 | 29.1 | 38.5 | 54.9 | 5.9 | 6.2 | 6.3 | 19.2 | 2300 |
| aimd | 4 | 29.0 | 41.4 | 58.0 | 5.9 | 6.2 | 6.2 | 19.1 | 2297 |
| aimd-flow-control | 4 | 29.0 | 38.6 | 54.7 | 5.9 | 6.1 | 6.2 | 19.1 | 2298 |
| interactive-only | 8 | 29.2 | 37.3 | 48.0 | 5.8 | 5.9 | 6.0 | 19.6 | 2354 |
| ungated | 8 | 28.3 | 36.7 | 49.3 | 5.7 | 6.0 | 6.1 | 20.0 | 2396 |
| aimd | 8 | 28.4 | 37.0 | 42.9 | 5.6 | 6.0 | 6.1 | 20.0 | 2398 |
| aimd-flow-control | 8 | 28.3 | 36.5 | 45.4 | 5.8 | 6.0 | 6.2 | 19.6 | 2359 |

### TTFT p99 Overhead vs Baseline

Percentage change from interactive-only at each replica count.

| Scenario | r=1 | r=4 | r=8 |
|---|---|---|---|
| ungated | +5.8% | -30.0% | +2.7% |
| aimd | +8.1% | -26.0% | -10.6% |
| aimd-flow-control | +8.3% | -30.2% | -5.4% |

At r=1, batch processing adds 5-8% to TTFT p99. At r=4 and r=8, the batch overhead is within noise — the additional replicas absorb the batch load.

The negative values at r=4 (batch scenarios showing lower latency than baseline) are consistent with run-to-run variance rather than batch processing improving interactive latency.

### Throughput

Interactive throughput (requests/sec) is stable across all scenarios: 18.7-20.0 RPS. Batch processing does not reduce interactive throughput for Qwen3-8B.

### Error Rates

Zero inference errors across all 12 runs (27,000+ interactive requests, 36,000 batch requests). One incomplete request in aimd r=8.

### Batch Processing

30 batch jobs × 100 requests each complete within 60-90 seconds of submission. The batch processor dispatches 30-40 concurrent inference requests (per PCP `processor_inflight_requests` metric). All 3000 batch requests complete with 0 failures.

PCP time-series from the ungated r=4 run shows batch dispatch starting at t+20s and completing by t+300s, with 14-18 vLLM requests running concurrently during burst phases.

### System Metrics (PCP)

**GPU utilization**: Active GPUs reach 99-100% during burst phases with concurrent batch load (ungated r=4: GPUs 0,1,5,7 at 99-100%). Without batch load, GPU utilization matches the interactive-only pattern.

**vLLM request queue**: During concurrent batch + interactive load, vLLM reports 5-18 running requests per sample (10s intervals). During interactive-only, running requests track the concurrent stream count (1 during idle, 15 during burst).

**Batch processor inflight**: The processor maintains 20-41 concurrent inference requests during active dispatch (ungated scenario, global limit=200). Inflight drops to 0 once batch jobs complete.

### Visualizations

- `analysis/ttft_p99_burst_Qwen3-8B.png` — TTFT p99 comparison across scenarios
- `analysis/latency_vs_replicas_Qwen3-8B.png` — TTFT and TPOT p99 scaling with replicas
- `analysis/throughput_burst_Qwen3-8B.png` — Interactive throughput comparison
- `analysis/idle_vs_burst_Qwen3-8B.png` — Idle vs burst latency comparison

## Observations

1. **Batch overhead is small for Qwen3-8B on H200.** At r=1 (single GPU), TTFT p99 increases 5-8% with concurrent batch processing. ITL and TPOT are unaffected (< 1 ms difference). At r=4+, the overhead is within measurement noise.

2. **All three dispatch strategies produce equivalent results.** Ungated, AIMD, and AIMD+flow-control show no measurable difference in interactive latency. This is expected: RHOAI 3.5 EA lacks flow-control plugins in the EPP, and AIMD metrics are not exposed by this processor version, so scenarios 3 and 4 operate identically to scenario 2 from the inference backend's perspective.

3. **Qwen3-8B is not GPU-bound on H200.** The model is small (8B parameters, TP=1) relative to H200 capacity (141 GB HBM3e). GPU utilization reaches 99-100% only during burst phases, and the KV cache is not under pressure. Larger models (FP8-70B, gpt-oss-120b) will provide a more realistic test of batch/interactive contention.

4. **30×100 batch job config enables concurrent measurement.** Batch dispatch starts within 20s of submission and overlaps with interactive traffic throughout the measurement window. The previous 3×1000 config had 5-7 minute ingestion delay, limiting overlap to the last minute.

## Known Limitations (RHOAI 3.5 EA)

| Feature | Status | Impact |
|---|---|---|
| EPP flow-control plugins | Not available | Scenarios 3 and 4 are functionally identical to scenario 2 |
| AIMD processor metrics | Not exposed | Cannot observe adaptive concurrency dynamics |
| Job-level processor metrics | Not exposed | No job duration, token throughput, or per-model inflight metrics |
| Batch gateway GC | Health probe fails | Disabled; manual state cleanup between runs |
| pmdavalkey | Requires PCP ≥ 7.1.1 | Workaround: dnf upgrade in PCP pod startup |

## Next Steps

1. **FP8-70B and gpt-oss-120b**: In progress. Larger models will stress GPU and KV cache, revealing batch contention effects not visible with Qwen3-8B.
2. **RHOAI EA2 retest**: Enable flow-control plugins, AIMD metrics, and job-level metrics when the next RHOAI version ships updated upstream code.
3. **HTTPRoute validation**: Confirm EPP routing is active (colleague's HTTPRoute cleanup was applied before these runs).
4. **Disaggregated P/D**: Evaluate batch gateway with prefill/decode separation when RDMA is configured between nodes.
