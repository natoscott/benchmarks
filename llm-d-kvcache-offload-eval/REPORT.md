# llm-d KV-Cache Management Evaluation

## Overview

This repository contains performance evaluations of KV-cache management strategies in the llm-d inference serving system across multiple versions. Testing covers configurations across four model sizes (Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ) using high-concurrency workloads with tensor parallelism across 2x NVIDIA L40S GPUs.

**Hardware:** 2x NVIDIA L40S GPUs (48GB total VRAM), 48 vCPUs, OpenShift on IBM Cloud (Single Node OpenShift)

**Models:** Qwen/Qwen3-0.6B, Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-32B-AWQ

---

## Version Reports

### [REPORT-v0.7.0.md](REPORT-v0.7.0.md) — llm-d v0.7.0 (vLLM 0.19.1) ← current

**Status:** Complete — 384 runs (6 configs × 4 models × 8 rates at gmu=0.9; 6 configs × 4 models × 8 rates mempress)

**Configurations:** `no-offload`, `native-offload-20k`, `lmcache-local`, `lmcache-valkey`, `fs-offload`, `cpu+fs-offload-20k`

- Qwen3-8B no-offload: +34.1% vs v0.6.0 (197→264 tok/s); largest cross-version gain in this evaluation
- Qwen3-32B-AWQ native-offload overhead eliminated (−63.9% in v0.6.0 → 0.0%); vLLM #38515 resolved
- Qwen3-8B native-offload regression: −6.5% (v0.6.0) → −39.9% at gmu=0.9
- First valid filesystem offload results (v0.5.1/v0.6.0 excluded due to system misconfiguration): Qwen3-14B −3.5%, Qwen3-0.6B −9.2%, Qwen3-8B −41.9% vs no-offload at gmu=0.9
- Qwen3-0.6B mempress cpu+fs-offload: +34.6% vs mempress no-offload baseline
- LMCache v0.4.4; EPP v1.5.0; CUDA 13.0.2 (requires driver ≥ 580)

**Supersedes:** REPORT-v0.6.0.md

---

### [REPORT-v0.6.0.md](REPORT-v0.6.0.md) — llm-d v0.6.0 (vLLM 0.17.1)

**Status:** Complete — 297 runs. **Note:** `fs-offload` and `cpu+fs-offload-20k` results invalid (system misconfiguration, PVC); see report for details.

- Qwen3-0.6B no-offload: +26.8% vs v0.5.1 (636→807 tok/s); native-offload overhead eliminated (+0.3%)
- Qwen3-8B no-offload: +72.9% vs v0.5.1 (114→197 tok/s); native-offload overhead narrows to -6.5% (from -29.9%)
- Qwen3-0.6B mempress native-offload: +51.4% vs no-offload baseline (up from +22.3% in v0.5.1)
- LMCache upgraded to v0.4.2 (from v0.3.15); EPP v0.7.1

**Supersedes:** REPORT-v0.5.1.md

---

### [REPORT-v0.5.1.md](REPORT-v0.5.1.md) — llm-d v0.5.1 (vLLM 0.15.1)

**Status:** Complete — 256 runs (6 configs × 4 models × 8 rates gmu=0.9) + 128 memory-pressure runs (gmu=0.55–0.70)

**Configurations:** `no-offload`, `native-offload-20k`, `fs-offload`, `cpu+fs-offload-20k`, `lmcache-local`, `lmcache-valkey`

- Qwen3-14B: fs-offload +212.7%, cpu+fs +189.1%, native-20k +14.5% vs no-offload at gmu=0.9 (threads_per_gpu=128)
- Qwen3-8B: fs-offload +86.0%, cpu+fs +65.4% vs no-offload at gmu=0.9; native-offload-20k -29.9%
- Qwen3-0.6B: fs-offload +32.3%, cpu+fs +34.2%; fs-offload deadlock at rate≥300 with threads_per_gpu=64 ([issue #457](https://github.com/llm-d/llm-d-kv-cache/issues/457)) resolved at 128
- Qwen3-32B-AWQ: -56 to -58% under all offload configs
- Qwen3-14B no-offload: 66.1→58.7 tok/s regression from v0.5.0 (-11.2%)

**Supersedes:** REPORT-v0.5.0.md

---

### [REPORT-v0.4.0.md](REPORT-v0.4.0.md) — llm-d v0.4.0 (vLLM 0.11.2, LMCache v0.3.7)

**Status:** Complete — 272 runs (7 configs × 4 models × 8 rates) + 192 memory-pressure runs (6 configs × 4 models × 8 rates, gmu=0.55–0.70, including native-offload-20k)

**Configurations:** `no-offload`, `native-offload` (10K/20K blocks), `lmcache-local`, `lmcache-valkey`, `llm-d-valkey`

- Qwen3-14B (20.58 GiB GPU KV-cache): +0.6% to +13% with offload at gmu=0.9; +22.6% with native-offload-20k under memory pressure
- Qwen3-0.6B (33.92 GiB GPU KV-cache): -13% to -29% at gmu=0.9; lmcache flips to +18-19% under memory pressure
- llm-d EPP distributed KV-block indexing within ±2% of baseline for all models
- Valkey and Redis perform identically; Redis dropped from ongoing benchmark plans
- vLLM native offload underperforms LMCache for all model sizes

---

## Cross-Version Summary

CPU KV-cache offload provides throughput gains when GPU KV-cache is constrained. On 2× NVIDIA L40S, only Qwen3-14B FP16 is naturally constrained at default gmu=0.9 (20.58 GiB GPU KV-cache). Reducing `gpu_memory_utilization` creates pressure for all model sizes and provides a more complete characterisation of offload behaviour. The memory-pressure results are presented first as the primary result set; gmu=0.9 results follow as the unconstrained reference condition.

### No-Offload Baseline Throughput (tok/s)

| Model | v0.4.0 | v0.5.0 | v0.5.1 | v0.6.0 | v0.7.0 |
|-------|-------:|-------:|-------:|-------:|-------:|
| Qwen3-0.6B | 602.0 | 634.7 | 636.8 | 807.5 | **810.7** |
| Qwen3-8B | 113.0 | 114.1 | 114.1 | 197.3 | **264.5** |
| Qwen3-14B | 58.7 | 66.1 | 58.7 | 55.5 | **57.6** |
| Qwen3-32B-AWQ | 49.2 | 51.2 | 51.2 | 50.1 | **50.1** |

### Native CPU Offload at Matched Memory Pressure (gmu=0.55–0.70) — primary result set

Per-model reduced `gpu_memory_utilization` creates comparable KV-cache pressure across all model sizes and versions:

| Model | v0.4.0 nat-10k | v0.4.0 nat-20k | v0.5.1 nat-20k | v0.6.0 nat-20k | v0.7.0 nat-20k |
|-------|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Qwen3-0.6B | -8.3% | +9.5% | +22.3% | +51.4% | **+53.1%** |
| Qwen3-8B | -19.3% | -9.2% | -3.6% | -16.3% | −18.0% |
| Qwen3-14B | 0.0% | +22.6% | +10.4% | -10.8%† | **+4.6%** |
| Qwen3-32B-AWQ | -2.3% | -2.3% | -33.3% | not run | 0.0% |

*All deltas vs same-version no-offload baseline at matched gmu. See individual version reports for full data.*
†Qwen3-14B v0.6.0 mempress native-offload: 3/8 rates excluded due to vLLM #38515 crash; value from clean rates only.

External KV cache hit rates (v0.5.1 native-offload-20k at mempress gmu): 0.6B 26.9%, 8B 13.7%, 14B 8.5%, 32B-AWQ 2.2%. Equivalent v0.4.0 configurations show near-zero external cache activity.

### LMCache CPU Offload at Matched Memory Pressure (gmu=0.55–0.70) — primary result set

| Model | v0.4.0 local | v0.4.0 valkey | v0.5.1 local | v0.5.1 valkey | v0.6.0 local | v0.6.0 valkey |
|-------|:------------:|:-------------:|:------------:|:-------------:|:------------:|:-------------:|
| Qwen3-0.6B | +18.5% | +19.5% | -4.7% | -5.3% | -18.7% | -9.6% |
| Qwen3-8B | -9.2% | -7.3% | +1.8% | +0.9% | 0.0% | +3.1% |
| Qwen3-14B | -12.9% | -17.7% | -3.0% | -1.5% | -1.5% | -1.5% |
| Qwen3-32B-AWQ | -11.4% | -11.4% | -2.1% | -2.1% | not run | not run |

### Native CPU Offload: v0.4.0 → v0.5.0 → v0.5.1 → v0.6.0 → v0.7.0 (default gmu=0.9)

At gmu=0.9 only Qwen3-14B is GPU KV-cache constrained on this hardware; other models show overhead from offload rather than benefit.

| Model | v0.4.0 (10k) | v0.5.0 (20k) | v0.5.1 (20k) | v0.6.0 (20k) | v0.7.0 (20k) |
|-------|:------------:|:------------:|:------------:|:------------:|:------------:|
| Qwen3-0.6B | -29.1% | -0.3% | -2.2% | +0.3% | −0.1% |
| Qwen3-8B | -36.5% | -26.1% | -29.9% | -6.5% | −39.9% |
| Qwen3-14B | +0.6% | -1.6% | **+14.5%** | +1.9% | −1.9% |
| Qwen3-32B-AWQ | -1.0% | -58.4% | -58.3% | -63.8% | **0.0%** |

Values are % throughput delta vs same-version no-offload baseline. Block counts are not directly comparable: v0.4.0 used `num_cpu_blocks`; v0.5.0+ use `cpu_bytes_to_use` (API changed in vLLM 0.15.x).

### LMCache CPU Offload: v0.4.0 → v0.5.1 → v0.6.0 (default gmu=0.9)

Values are % throughput delta vs same-version no-offload baseline. LMCache versions: v0.3.7 (v0.4.0), v0.3.15 (v0.5.1), v0.4.2 (v0.6.0).

| Model | v0.4.0 local | v0.4.0 valkey | v0.5.1 local | v0.5.1 valkey | v0.6.0 local | v0.6.0 valkey |
|-------|:------------:|:-------------:|:------------:|:-------------:|:------------:|:-------------:|
| Qwen3-0.6B | -13.6% | -13.0% | -4.9% | -4.7% | -17.6% | -2.2% |
| Qwen3-8B | -5.6% | -6.5% | -0.9% | +0.9% | -1.6% | -28.1%† |
| Qwen3-14B | +11.8% | +13.0% | +7.3% | +7.3% | -1.9% | +1.9% |
| Qwen3-32B-AWQ | -12.7% | -12.7% | -56.2% | -58.3% | -63.8% | -61.7% |

†Qwen3-8B lmcache-valkey v0.6.0: -28.1% throughput with 9.4× TTFT increase at rate=50 (0.696 s → 6.555 s); Valkey round-trip latency accumulation under concurrency.

### Filesystem Offload (v0.7.0, gmu=0.9) — first valid baseline

v0.5.1 and v0.6.0 filesystem offload results are excluded due to system misconfiguration (PVC) in those evaluations. v0.7.0 provides the first valid baseline, with `llmd_fs_connector` v0.19 baked into the image.

**gmu=0.9** — % throughput delta vs same-version no-offload baseline, `threads_per_gpu=128`:

| Config | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|-------|:----------:|:--------:|:---------:|:------------:|
| v0.7.0 fs-offload | −9.2% | −41.9% | −3.5% | −2.0% |
| v0.7.0 cpu+fs-offload-20k | −10.1% | −43.1% | −5.6% | −2.0% |

At gmu=0.9, Qwen3-0.6B and Qwen3-8B GPU KV caches are not saturated on this hardware, so filesystem offload adds write overhead without cache-extension benefit. Qwen3-14B and Qwen3-32B-AWQ approach KV cache saturation, resulting in smaller overhead (−2.0% to −5.6%).

**mempress gmu (0.55–0.70)** — % throughput delta vs same-version mempress no-offload baseline:

| Config | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|-------|:----------:|:--------:|:---------:|:------------:|
| v0.7.0 fs-offload | −6.7% | −24.0% | −1.5% | −2.0% |
| v0.7.0 cpu+fs-offload-20k | **+34.6%** | −21.0% | +3.2% | −2.0% |

Qwen3-0.6B cpu+fs-offload-20k reaches +34.6% vs mempress no-offload under memory pressure; the CPU tier provides primary cache extension for the smallest model. Qwen3-8B incurs overhead for both fs configs under mempress, consistent with PVC I/O throughput being a binding constraint at this model size and token rate.

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
