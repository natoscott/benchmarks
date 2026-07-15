# llm-d Batch Gateway Performance Evaluation

Evaluation of the llm-d batch gateway's ability to protect interactive inference latency while processing batch workloads on shared GPU infrastructure. Tests run across multiple RHOAI versions, dispatch strategies, and model architectures.

## Summary

The batch gateway dispatches batch inference requests alongside interactive traffic. Three dispatch strategies are evaluated: ungated (high concurrency, no controls), AIMD (adaptive concurrency), and AIMD+flow-control (EPP priority scheduling). Across two RHOAI versions, the key finding is that **EPP flow-control effectiveness depends on both model capacity and batch dispatch rate** — it protects large dense models under high batch pressure but adds harmful overhead when batch contention is already low (either because the model is fast, replicas absorb the load, or per-endpoint limits reduce batch pressure). The batch processor's `perEndpoint` concurrency limit was not enforced in EA1 (all scenarios dispatched at the global limit of 100), but is enforced in EA2 (aimd/FC dispatch at 20).

## Common Infrastructure

| Component | Configuration |
|---|---|
| Cluster | OCP 4.20 |
| GPUs | 2x gx3d nodes, 8x NVIDIA H200 each (16 total) |
| Models | Qwen3-8B (TP=1), FP8-70B (TP=2), gpt-oss-120b (TP=4, MoE MXFP4) |
| vLLM | v0.19.1+rhaiv.6, prefix caching enabled, chunked prefill |
| gpu-memory-utilization | 0.90 (all models) |
| Metrics | PCP 7.1.5 (openmetrics, pmdavalkey, pmdapostgresql) |
| Interactive burst | 15 concurrent streams, 60s |
| Interactive idle | 1 concurrent stream, 60s |
| Cycles | 3 (1 warmup excluded, 2 measured) |
| Batch jobs | 30 x 100 requests (3000 total) |
| Prompt / output tokens | 512 / 128 (synthetic) |
| Replica configs | Qwen3-8B: 1, 4, 8; FP8-70B: 1, 4, 8; gpt-oss-120b: 1, 2, 4 |

### Scenarios

| ID | Name | Description |
|---|---|---|
| 0 | interactive-only | No batch gateway. Interactive traffic baseline. |
| 2 | ungated | Batch gateway, global=200, per-endpoint=100, AIMD disabled |
| 3 | aimd | Batch gateway, global=100, per-endpoint=20, AIMD enabled |
| 4 | aimd-flow-control | Same as AIMD + EPP `flowControl` feature gate with priority bands. Batch processor sets `x-gateway-inference-objective: batch-sheddable` (priority -1). Interactive traffic has no objective header (default priority 0). |

All overhead measurements are from burst phases (15 concurrent interactive streams). Idle phase data is available in per-version reports.

## Reports

| Version | Components | Runs | Key Findings | Report |
|---|---|---|---|---|
| RHOAI 3.5 EA1 | EA1 batch gateway + EA1 EPP | 45 | No isolation effective initially; flow-control feature gate discovered and tested — protects large models, harms small models | [REPORT-rhoai-3.5ea1.md](REPORT-rhoai-3.5ea1.md) |
| RHOAI 3.5 EA1 EPP + EA2 Batch | EA2 batch gateway + EA1 EPP | 36 | `perEndpoint` now enforced (20 vs 100); AIMD visible but not adapting; ungated outperforms aimd/FC for most models due to lower scheduling overhead | [REPORT-rhoai-3.5ea1+ea2-batch.md](REPORT-rhoai-3.5ea1+ea2-batch.md) |

## Cross-Version Comparison

TTFT p99 overhead vs interactive-only baseline (burst phase, r=1):

| Model | Scenario | EA1 | EA2 Batch | Change |
|---|---|---|---|---|
| Qwen3-8B | ungated | +5.8% | -11.3% | Within noise |
| Qwen3-8B | aimd | +8.1% | +125.5% | perEndpoint=20 enforced; scheduling overhead at low batch pressure |
| Qwen3-8B | flow-control | +57.9% | +166.2% | Same; FC overhead dominates |
| FP8-70B | ungated | +106.8% | +45.8% | Baseline 4x faster in EA2 (cluster conditions) |
| FP8-70B | aimd | +33.0% | +252.5% | perEndpoint=20 enforced; higher tail latency |
| FP8-70B | flow-control | -49.3% | +175.2% | Baseline improvement eliminates FC benefit |
| gpt-oss-120b | ungated | +42.8% | -67.1% | Different baseline (cluster conditions) |
| gpt-oss-120b | flow-control | -39.2% | -82.6% | FC still helps at r=1 |

### Processor behavior

| Metric | EA1 | EA2 Batch |
|---|---|---|
| `perEndpoint` enforcement | Ignored (all at global limit) | Enforced (aimd/FC at 20) |
| AIMD adaptation | Not observable (no metrics) | Visible but static (limit stays at 20, no 429/5xx signals) |
| Batch processor metrics | 2 (inflight, active_workers) | 12+ (job latency, queue wait, token throughput, AIMD limit, etc.) |
| GC | Disabled (health probe fails) | Functional (30-min reconciliation) |
| EPP | EA1 binary | EA1 binary (unchanged) |

## Known Issues Across Versions

| Issue | EA1 | EA2 Batch | Status |
|---|---|---|---|
| EPP `utilization-detector` plugin | Not registered | Not registered (same EPP) | Requires updated EPP build |
| AIMD signal path | No metrics | Metrics visible, no adaptation | Gateway never returns 429/5xx; AIMD has no overload signal |
| Flow-control overhead for small models | +58-91% TTFT p99 | +125-166% TTFT p99 | By design — scheduling overhead exceeds batch contention benefit |
| FP8-70B baseline variability | 1898 ms TTFT p99 (r=1) | 471 ms TTFT p99 (r=1) | Cluster conditions; within-version comparisons valid |
| Per-model flow-control policy | Not available | Not available | Flow-control is global to EPP, not per-model |
