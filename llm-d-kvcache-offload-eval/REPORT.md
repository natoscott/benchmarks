# llm-d KV-Cache Management Evaluation

## Overview

This repository contains performance evaluations of KV-cache management strategies in the llm-d inference serving system across multiple versions. Testing covers configurations across four model sizes (Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ) using high-concurrency workloads with tensor parallelism across 2x NVIDIA L40S GPUs.

**Hardware:** 2x NVIDIA L40S GPUs (48GB total VRAM), 48 vCPUs, OpenShift on IBM Cloud (Single Node OpenShift)

**Models:** Qwen/Qwen3-0.6B, Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-32B-AWQ

---

## Version Reports

### [REPORT-v0.5.1.md](REPORT-v0.5.1.md) — llm-d v0.5.1 (vLLM 0.15.1) ← current

**Status:** Complete — 128 runs across 4 configurations, 4 models, 8 concurrency levels

**Configurations:**
- `no-offload` — GPU-only KV-cache (baseline)
- `native-offload-20k` — CPU offload via `OffloadingConnector` / `CPUOffloadingSpec`
- `fs-offload` — Filesystem offload via `SharedStorageOffloadingSpec` (llmd_fs_connector 0.15.1)
- `cpu+fs-offload-20k` — Hierarchical CPU+filesystem via `MultiConnector`

**Observations:**
- Qwen3-14B benefits from all offload configurations: +14.5% (native-20k), +7.3% (cpu+fs), +3.6% (fs)
- Qwen3-0.6B fs-offload unstable at sustained concurrency (zero completions at rate≥300)
- Disk I/O negligible for filesystem offload runs (≤0.04 MB/s); IBM VPC block PVC operates via OS page cache
- MultiConnector writes to CPU and filesystem simultaneously; reads prioritise CPU over filesystem
- Qwen3-14B no-offload regressed -11.2% from v0.5.0 to v0.5.1 (66.1→58.7 tok/s)
- libstdc++ ABI incompatibility: llmd_fs_connector requires GLIBCXX_3.4.30+; RHEL9 image provides 3.4.29

**Supersedes:** REPORT-v0.5.0.md

---

### [REPORT-v0.4.0.md](REPORT-v0.4.0.md) — llm-d v0.4.0 (vLLM 0.11.2, LMCache v0.3.7)

**Status:** Complete — 272 runs across 7 configurations + 160 memory-pressure runs (5 configs × 4 models × 8 rates, gmu=0.55–0.70)

**Configurations:**
- `no-offload`, `native-offload` (10K/20K blocks)
- `lmcache-local`, `lmcache-redis`, `lmcache-valkey`
- `llm-d-redis`, `llm-d-valkey` (EPP distributed KV-block indexing)

**Observations:**
- GPU KV-cache memory availability after model loading determines offload effectiveness
- Qwen3-14B (20.58 GiB GPU KV-cache) benefits from CPU offload (+12–17% with LMCache)
- Qwen3-0.6B (33.92 GiB GPU KV-cache) shows degradation under all offload strategies (-13% to -29%)
- llm-d EPP distributed KV-block indexing within ±2% of baseline for all models
- Redis and Valkey perform identically for both EPP indexing and LMCache storage
- vLLM native offload underperforms LMCache at all model sizes in v0.4.0

---

## Cross-Version Summary

### Native CPU Offload: v0.4.0 → v0.5.0 → v0.5.1

| Model | v0.4.0 (10k) | v0.5.0 (20k) | v0.5.1 (20k) |
|-------|:------------:|:------------:|:------------:|
| Qwen3-0.6B | -29.1% | -0.3% | -2.2% |
| Qwen3-8B | -36.5% | -26.1% | -29.9% |
| Qwen3-14B | +0.6% | -1.6% | **+14.5%** |
| Qwen3-32B-AWQ | -1.0% | -58.4% | -58.3% |

Values are % throughput delta vs same-version no-offload baseline. Block counts are not directly comparable: v0.4.0 used `num_cpu_blocks`; v0.5.0/v0.5.1 use `cpu_bytes_to_use` (API changed in vLLM 0.15.x).

### No-Offload Baseline Throughput (tok/s)

| Model | v0.4.0 | v0.5.0 | v0.5.1 |
|-------|-------:|-------:|-------:|
| Qwen3-0.6B | 602.0 | 634.7 | 636.8 |
| Qwen3-8B | 113.0 | 114.1 | 114.1 |
| Qwen3-14B | 58.7 | 66.1 | 58.7 |
| Qwen3-32B-AWQ | 49.2 | 51.2 | 51.2 |

### Filesystem Offload (v0.5.1 only)

| Model | fs-offload | cpu+fs-offload-20k |
|-------|:----------:|:-----------------:|
| Qwen3-0.6B | unstable¹ | unstable¹ |
| Qwen3-8B | -33.6% | -33.6% |
| Qwen3-14B | +3.6% | +7.3% |
| Qwen3-32B-AWQ | -56.2% | -56.2% |

¹ Zero completed requests at rate≥300.

---

## Hardware and Scope

All results are specific to: 2× NVIDIA L40S (24 GB VRAM each), PCIe Gen4, 48 vCPUs, IBM VPC block storage. GPU KV-cache memory available after model loading ranges from 20.58 GiB (Qwen3-14B FP16) to 33.92 GiB (Qwen3-0.6B). This range determines which models benefit from offload on this hardware.

---

## Methodology

- **Tool**: GuideLLM
- **Workload**: Multi-turn concurrent conversations, 10K-token shared prefix, 128 prompt + 128 output tokens per turn, 5 turns
- **Concurrency**: 1, 50, 100, 150, 300, 400, 500, 650 (120 seconds each)
- **Metrics**: GuideLLM (throughput, TTFT, ITL, TPOT) + PCP (GPU, CPU, disk, vLLM internals via OpenMetrics)
- **Data**: `results/*/guidellm-results.json.zst`, `results/*/pcp-archives/`, `results/*/vllm-startup.log.zst`
- **Analysis**: `scripts/analyze-*.py`, outputs in `analysis/`

---

*Analysis framework: Performance Co-Pilot + GuideLLM*
*System: Single Node OpenShift on IBM Cloud with 2× NVIDIA L40S GPUs*
