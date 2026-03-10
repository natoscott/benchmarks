# llm-d KV-Cache Management Evaluation

## Overview

This repository contains performance evaluations of KV-cache management strategies in the llm-d inference serving system across multiple versions. Testing covers seven configurations across four model sizes (Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ) using high-concurrency workloads with tensor parallelism across 2x NVIDIA L40S GPUs.

**Hardware:** 2x NVIDIA L40S GPUs (48GB total VRAM), 48 vCPUs, OpenShift on IBM Cloud

**Models:** Qwen/Qwen3-0.6B, Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-32B-AWQ

---

## Version Reports

Detailed analysis is organized by llm-d version, with each report comparing against the immediately prior version:

### [REPORT-v0.5.0.md](REPORT-v0.5.0.md) - llm-d v0.5.0 (vLLM 0.14.1)

**Status:** Native offload evaluation complete

**Key Findings:**
- vLLM 0.14.1 significantly improves native CPU offload for small models (0.6B: +26.1 pp improvement, -3.0% overhead vs v0.4.0's -29.1%)
- Large models show regressions (14B: -8.7 pp, 32B-AWQ: -55.2 pp)
- 20K CPU block allocation nearly eliminates overhead for 0.6B model (-0.3%)
- 32B-AWQ shows -56.2% degradation with native offload

**Comparison:** v0.5.0 vs v0.4.0 side-by-side analysis

### [REPORT-v0.4.0.md](REPORT-v0.4.0.md) - llm-d v0.4.0 (vLLM 0.11.2, LMCache v0.3.7)

**Status:** Complete

**Key Findings:**
- GPU KV-cache memory availability determines offload effectiveness: 14B model benefits (+12-17%) due to constrained memory (20.58 GiB), while 0.6B model (-13% to -29%) has abundant memory (33.92 GiB)
- llm-d EPP distributed indexing achieves performance parity (within ±2% for most models)
- vLLM native offloading underperforms LMCache across all model sizes
- CPU memory capacity is critical: 32B-AWQ shifted from -12.7% to +11.9% when CPU blocks doubled from 10K to 20K

**Configurations:**
- Baseline GPU-only (no-offload)
- vLLM native CPU offload (10K/20K blocks)
- LMCache local CPU (10K/20K blocks)
- LMCache distributed (Redis/Valkey backends)
- llm-d EPP distributed indexing (Redis/Valkey backends)

---

## Cross-Version Insights

### GPU Memory Availability is the Dominant Factor

Analysis of vLLM startup logs across both versions confirms that actual GPU memory available for KV-cache after model loading (not parameter count) predicts offload effectiveness:

| Model | GPU Memory (GiB) | Token Capacity | Offload Benefit | Optimal Strategy |
|-------|----------------:|---------------:|:---------------:|------------------|
| Qwen3-0.6B | 33.92 | 635K | ❌ | GPU-only |
| Qwen3-8B | 26.83 | 391K | ❌ | GPU-only |
| Qwen3-14B | 20.58 | 270K | ✅ | CPU offload (v0.4.0 LMCache) |
| Qwen3-32B-AWQ | 25.40 | 208K | ⚠️ | CPU offload with adequate capacity |

**Memory Pressure Threshold:** ~26 GiB GPU memory appears to be the crossover point where memory pressure justifies offload overhead on this hardware.

### vLLM Version Comparison: Native Offload Performance

| Model | v0.4.0 Native (10K) | v0.5.0 Native (10K) | Change | Winner |
|-------|--------------------:|--------------------:|-------:|--------|
| Qwen3-0.6B | -29.1% | -3.0% | **+26.1 pp** | v0.5.0 |
| Qwen3-8B | -36.5% | -25.2% | **+11.3 pp** | v0.5.0 |
| Qwen3-14B | +0.6% | -8.1% | **-8.7 pp** | v0.4.0 |
| Qwen3-32B-AWQ | -1.0% | -56.2% | **-55.2 pp** | v0.4.0 |

vLLM 0.14.1 introduces substantial KV offloading changes: physical block sizes increased from 8-32 KB to 0.5-2 MB by consolidating all layers into contiguous blocks (2×num_layers factor), asynchronous DMA transfers, and CLI interface redesign. These changes benefit small models through reduced transfer overhead but create challenges for larger models where increased block transfer granularity may dominate.

**Takeaway:** vLLM 0.14.1 (v0.5.0) improves small model offload but regresses for large/quantized models. For 14B and 32B-AWQ, v0.4.0 with LMCache provides superior performance.

### Backend Equivalence: Redis vs Valkey

Both versions confirm that Redis and Valkey perform identically (within ±1-2%) for:
- llm-d EPP distributed KV-block indexing
- LMCache distributed KV-cache storage

Backend selection can be based on operational factors (licensing, ecosystem, features) without performance concerns.

---

## Hardware Dependencies and Limitations

**Test Hardware:** 2x NVIDIA L40S GPUs (24GB VRAM each, 48GB total), 48 vCPUs, OpenShift on IBM Cloud

**Findings are specific to:**
- L40S PCIe Gen4 bandwidth (~32 GB/s bidirectional)
- GPU KV-cache memory ranging from 20.58 GiB (14B) to 33.92 GiB (0.6B)
- CPU memory capacity matching or exceeding GPU KV-cache capacity
- High-concurrency workloads with 10K-token shared prefixes

**Expected shifts with different hardware:**
- **Higher GPU memory** (H100: 80GB, A100: 40GB/80GB): Shifts optimal offload model size upward
- **PCIe Gen5** (doubled bandwidth): Reduces offload overhead, potentially benefiting larger models
- **Different CPU memory bandwidth**: May shift CPU offload effectiveness threshold
- **Shorter context workloads**: Reduces memory pressure, shifts optimal model size upward

---

## Methodology

### Benchmark Framework

- **Tool**: GuideLLM v0.5.3
- **Workload**: High-concurrency multi-turn conversations with shared 10K-token prefixes
- **Concurrency levels**: 1, 50, 100, 150, 300, 400, 500, 650
- **Duration**: 120 seconds per concurrency level
- **Sample requests**: 4000 per benchmark run

### Metrics Collection

- **GuideLLM**: Throughput, latency (TTFT, ITL, TPOT), request completion
- **Performance Co-Pilot**: System-level metrics (CPU, GPU, memory, vLLM internals)
- **vLLM startup logs**: Actual GPU/CPU KV-cache memory allocations
- **Sampling**: 10-second intervals throughout benchmark execution

### Data Organization

**Raw Data:**
- GuideLLM results: `results/*/guidellm-results.json.zst`
- PCP archives: `results/*/pcp-archives/*.zst`
- vLLM logs: `results/*/vllm-startup.log.zst` or `vllm-startup-logs/*.log.zst`

**Analysis:**
- CSV extracts: `analysis/*.csv`
- Visualizations: `analysis/*.png`
- Scripts: `scripts/analyze-*.py`, `scripts/extract-*.py`

See [README.md](README.md) for complete documentation.

---

*Reports generated from benchmark runs completed February-March 2026*

*Analysis framework: Performance Co-Pilot + GuideLLM*

*System: OpenShift on IBM Cloud with 2x NVIDIA L40S GPUs*
