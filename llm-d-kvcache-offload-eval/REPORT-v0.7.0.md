# llm-d v0.7.0 KV-Cache Offload Evaluation

This report evaluates KV-cache offload strategies in llm-d v0.7.0 (vLLM 0.19.1, CUDA 13.0.2) across six configurations, four model sizes, and eight concurrency levels. Results are compared against v0.6.0 (vLLM 0.17.1) where valid baselines exist. This is the first evaluation with a valid filesystem offload configuration; fs-offload and cpu+fs-offload-20k results from v0.6.0 are excluded from comparisons due to a system misconfiguration (PVC) in that version (see REPORT-v0.6.0.md).

Memory-pressure results (reduced `gpu_memory_utilization` per model) are presented first as the primary result set, since KV-cache offload delivers measurable throughput benefit only when GPU KV-cache is constrained. Results at the default gmu=0.9 follow as an unconstrained reference condition; on this hardware only Qwen3-14B is GPU KV-cache limited at gmu=0.9.

**TL;DR:** **Under memory pressure, Qwen3-0.6B native-offload-20k reaches +53.1% vs the mempress no-offload baseline (802 vs 524 tok/s), and cpu+fs-offload-20k reaches +34.6%.** Qwen3-14B native-offload delivers +4.6% under mempress; cpu+fs-offload +3.2%. Qwen3-8B native-offload shows −18.0% under mempress — the PVC write rate at this model's token throughput exceeds the storage IOPS budget. This is the first evaluation with a working filesystem offload connector (`llmd_fs_connector` v0.19, baked into the image). At gmu=0.9 (unconstrained), Qwen3-8B no-offload gains +34.1% vs v0.6.0 with vLLM 0.19.1; Qwen3-8B native-offload regresses to −39.9% overhead at this setting.

**Software Versions:**

| Component | Version |
|-----------|---------|
| llm-d | v0.7.0 |
| vLLM | 0.19.1 (bundled in llm-d-cuda:v0.7.0) |
| CUDA | 13.0.2 (requires NVIDIA driver ≥ 580) |
| LMCache | v0.4.4 (installed via pip at pod startup) |
| GuideLLM | 0.6.0 |
| llmd_fs_connector | 0.19 (baked into llm-d-cuda:v0.7.0) |
| gateway-api-inference-extension (EPP) | v1.5.0 |
| Valkey | 8-alpine |
| OpenShift | 4.22.0 (SNO) |
| PCP | 7.0.3 |

**Hardware:** 2× NVIDIA L40S GPUs (48 GB total VRAM), 48 vCPUs, OpenShift on IBM Cloud (Single Node OpenShift)

**Models:** Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ

**Concurrency levels:** 1, 50, 100, 150, 300, 400, 500, 650

---

## Summary

384 benchmark runs: 192 under memory pressure (6 configs × 4 models × 8 rates, primary result set) and 192 at default gmu=0.9 (6 configs × 4 models × 8 rates, unconstrained reference). All four models completed all six configurations with no exclusions due to crashes (vLLM #38515 resolved).

**Configurations:**
1. **no-offload**: GPU-only KV-cache (baseline)
2. **native-offload-20k**: CPU offload via `OffloadingConnector` / `CPUOffloadingSpec`, 20K-block equivalent
3. **lmcache-local**: LMCache v0.4.4 with local CPU backend
4. **lmcache-valkey**: LMCache v0.4.4 with Valkey remote backend
5. **fs-offload**: Filesystem offload via `SharedStorageOffloadingSpec` (`llmd_fs_connector` v0.19, first valid baseline)
6. **cpu+fs-offload-20k**: Hierarchical CPU+filesystem offload via `MultiConnector` (first valid baseline)

**Peak Throughput — Memory Pressure (primary result set):**

| Model | Config | v0.7.0 (tok/s) | Rate | vs mempress no-offload | vs v0.6.0 |
|-------|--------|:--------------:|:----:|:----------------------:|:---------:|
| Qwen3-0.6B | no-offload | 523.7 | 50 | — | −0.2% |
| | native-offload-20k | 802.1 | 50 | **+53.1%** | +0.9% |
| | lmcache-local | 494.9 | 50 | −5.5% | +16.0% |
| | lmcache-valkey | 501.3 | 50 | −4.3% | +5.6% |
| | fs-offload | 488.5 | 50 | −6.7% | (new) |
| | cpu+fs-offload-20k | 705.1 | 50 | **+34.6%** | (new) |
| Qwen3-8B | no-offload | 106.7 | 100 | — | +2.1% |
| | native-offload-20k | 87.5 | 50 | −18.0% | 0.0% |
| | lmcache-local | 104.5 | 100 | −2.1% | 0.0% |
| | lmcache-valkey | 108.8 | 100 | +1.9% | +1.0% |
| | fs-offload | 81.1 | 100 | −24.0% | (new) |
| | cpu+fs-offload-20k | 84.3 | 100 | −21.0% | (new) |
| Qwen3-14B | no-offload | 69.3 | 100 | — | 0.0% |
| | native-offload-20k | 72.5 | 300 | **+4.6%** | — |
| | lmcache-local | 68.3 | 100 | −1.5% | 0.0% |
| | lmcache-valkey | 68.3 | 100 | −1.5% | 0.0% |
| | fs-offload | 68.3 | 100 | −1.5% | (new) |
| | cpu+fs-offload-20k | 71.5 | 100 | **+3.2%** | (new) |
| Qwen3-32B-AWQ | no-offload | 50.1 | 1 | — | +0.1% |
| | native-offload-20k | 50.1 | 1 | 0.0% | +0.1% |
| | lmcache-local | 50.1 | 1 | 0.0% | 0.0% |
| | lmcache-valkey | 50.1 | 1 | 0.0% | 0.0% |
| | fs-offload | 49.1 | 1 | −2.0% | (new) |
| | cpu+fs-offload-20k | 49.1 | 1 | −2.0% | (new) |

(new) = first valid baseline; v0.6.0 fs-offload excluded due to system misconfiguration (PVC).

**Peak Throughput — gmu=0.9 (unconstrained reference):**

| Model | Config | v0.7.0 (tok/s) | vs no-offload | vs v0.6.0 |
|-------|--------|:--------------:|:-------------:|:---------:|
| Qwen3-0.6B | no-offload | 810.7 | — | +0.4% |
| | native-offload-20k | 809.6 | −0.1% | 0.0% |
| | lmcache-local | 647.5 | −20.1% | −2.7% |
| | lmcache-valkey | 791.5 | −2.4% | +0.3% |
| | fs-offload | 736.0 | −9.2% | (new) |
| | cpu+fs-offload-20k | 728.5 | −10.1% | (new) |
| Qwen3-8B | no-offload | 264.5 | — | **+34.1%** |
| | native-offload-20k | 158.9 | −39.9% | — |
| | lmcache-valkey | 204.8 | −22.6% | +44.3% |
| | fs-offload | 153.6 | −41.9% | (new) |
| Qwen3-14B | no-offload | 57.6 | — | +3.8% |
| | native-offload-20k | 56.5 | −1.9% | +0.1% |
| | fs-offload | 55.5 | −3.5% | (new) |
| Qwen3-32B-AWQ | no-offload | 50.1 | — | +0.1% |
| | native-offload-20k | 50.1 | 0.0% | +177%† |

†Large v0.6.0→v0.7.0 change reflects a known v0.6.0 misconfiguration (vLLM #38515), not a v0.7.0 improvement.

---

## Test Configuration

### Hardware

**System:** OpenShift cluster on IBM Cloud (Single Node OpenShift, SNO)
- **GPUs**: 2× NVIDIA L40S (24 GB VRAM each, 48 GB total); Tensor Parallelism: 2
- **CPU**: 48 vCPUs
- **Storage**: IBM VPC block PVC (256 GiB, `ibmc-vpc-block-custom`, 6,000 IOPS) at `/kvcache` for filesystem offload

### Software

See software versions table in the preamble.

**EPP v1.5.0:** `kv-cache-utilization-scorer` replaces deprecated `--kv-cache-usage-percentage-metric` flag.

**CUDA 13.0.2:** Requires NVIDIA driver ≥ 580.

**llmd_fs_connector deployment:** `llmd_fs_connector` v0.19 is baked into `ghcr.io/llm-d/llm-d-cuda:v0.7.0` at image build time via `docker/scripts/cuda/runtime/install-offloading-connector.sh`. No runtime pip install required.

### Workload

- Concurrency levels: 1, 50, 100, 150, 300, 400, 500, 650 (120 s each)
- Prompt: 128 tokens/turn, 128 output tokens/turn, 10,000-token shared prefix, 5 turns, seed=889

### Memory-Pressure gpu_memory_utilization

| Model | default gmu | mempress gmu |
|-------|:-----------:|:------------:|
| Qwen3-0.6B | 0.9 | 0.55 |
| Qwen3-8B | 0.9 | 0.65 |
| Qwen3-14B | 0.9 | 0.70 |
| Qwen3-32B-AWQ | 0.9 | 0.65 |

---

## Memory-Pressure Results (Primary)

Reducing `gpu_memory_utilization` below the default constrains GPU KV-cache capacity across all model sizes, creating conditions where offload strategies can extend effective capacity. These are the primary results for assessing KV-cache offload behaviour.

![Mempress Throughput Curves](analysis/v0.7.0_mempress_throughput_curves.png)
*Throughput vs concurrency under memory pressure for all six configurations. Each panel shows one model at its mempress gmu.*

### Qwen3-0.6B (gmu=0.55)

At gmu=0.55, GPU KV-cache is constrained to approximately 30% of model capacity, creating strong offload incentive. native-offload-20k reaches 802.1 tok/s (+53.1% vs mempress no-offload 523.7 tok/s), consistent with the trend from v0.5.1 (+22.3%) and v0.6.0 (+51.4%). cpu+fs-offload-20k reaches 705.1 tok/s (+34.6%): the CPU tier provides primary cache extension with the filesystem tier adding additional capacity. fs-offload alone shows −6.7%, indicating filesystem write overhead without sufficient CPU-tier benefit at this model size.

LMCache shows −5.5% (local) and −4.3% (valkey) vs mempress no-offload, performing below no-offload under memory pressure for this model.

### Qwen3-8B (gmu=0.65)

native-offload-20k shows −18.0% vs mempress no-offload (87.5 vs 106.7 tok/s). fs-offload (−24.0%) and cpu+fs-offload-20k (−21.0%) perform below native-offload, consistent with PVC IOPS being a binding constraint at Qwen3-8B's generation rate even under reduced gmu. lmcache-valkey reaches +1.9% (108.8 tok/s) — the only offload config that matches or exceeds no-offload for this model under mempress.

### Qwen3-14B (gmu=0.70)

native-offload-20k delivers +4.6% (72.5 vs 69.3 tok/s). cpu+fs-offload-20k shows +3.2% (71.5 tok/s). Both CPU-offload strategies provide modest but consistent benefit. fs-offload, lmcache-local, and lmcache-valkey are each −1.5%.

### Qwen3-32B-AWQ (gmu=0.65)

All configurations match within ±2% of mempress no-offload (49.1–50.1 tok/s). The model saturates at rate=1 regardless of offload config; KV-cache capacity is not the binding constraint at this utilization level.

### Latency under Memory Pressure

When offload strategies extend KV-cache capacity, the throughput gain is accompanied by a reduction in TTFT — cache hits eliminate prefill computation for cached prefix tokens. The converse also holds: offload configurations that incur throughput overhead show increased latency.

**TTFT and ITL at rate=50 — Memory Pressure:**

| Model | Config | TTFT (s) | vs no-offload | ITL (ms) | vs no-offload |
|-------|--------|:--------:|:-------------:|:--------:|:-------------:|
| Qwen3-0.6B | no-offload | 3.293 | — | 65.8 | — |
| | native-offload-20k | **1.754** | **−46.7%** | **46.4** | **−29.3%** |
| | lmcache-local | 3.577 | +8.6% | 68.4 | +3.9% |
| | lmcache-valkey | 3.168 | −3.8% | 70.3 | +6.8% |
| | fs-offload | 3.507 | +6.5% | 70.4 | +7.0% |
| | cpu+fs-offload-20k | **2.033** | **−38.3%** | **52.9** | **−19.6%** |
| Qwen3-8B | no-offload | 24.238 | — | 205.2 | — |
| | native-offload-20k | 27.534 | +13.6% | 234.6 | +14.3% |
| | lmcache-local | 25.058 | +3.4% | 208.1 | +1.4% |
| | lmcache-valkey | **21.802** | **−10.1%** | 205.6 | +0.2% |
| | fs-offload | 29.640 | +22.3% | 259.3 | +26.4% |
| | cpu+fs-offload-20k | 28.378 | +17.1% | 239.5 | +16.7% |
| Qwen3-14B | no-offload | 41.047 | — | 219.4 | — |
| | native-offload-20k | 40.768 | −0.7% | **206.2** | **−6.0%** |
| | lmcache-valkey | **37.415** | **−8.8%** | 217.6 | −0.8% |
| | cpu+fs-offload-20k | 41.731 | +1.7% | **210.8** | **−3.9%** |
| Qwen3-32B-AWQ | no-offload | 42.819 | — | 317.0 | — |
| | lmcache-valkey | **40.797** | **−4.7%** | **307.1** | **−3.1%** |
| | native-offload-20k | 45.692 | +6.7% | 305.7 | −3.6% |

![Mempress TTFT Curves](analysis/v0.7.0_mempress_ttft_curves.png)
*Time to first token vs concurrency under memory pressure.*

![Mempress ITL Curves](analysis/v0.7.0_mempress_itl_curves.png)
*Inter-token latency vs concurrency under memory pressure.*

**Qwen3-0.6B:** native-offload-20k reduces TTFT from 3.293 s to 1.754 s (−46.7%) and ITL from 65.8 to 46.4 ms (−29.3%) at rate=50. This is the strongest latency improvement in the evaluation: when GPU KV-cache is constrained, offload cache hits eliminate prefill computation for the 10,000-token shared prefix, directly reducing time to first token. cpu+fs-offload-20k shows a similar pattern (TTFT −38.3%, ITL −19.6%). fs-offload, lmcache-local, and lmcache-valkey add small latency overhead (+3.8% to +8.6% TTFT) without the prefill reduction benefit of the native connector.

**Qwen3-8B:** lmcache-valkey is the only offload config to reduce TTFT under mempress (−10.1%, 21.8 vs 24.2 s). native-offload-20k, fs-offload, and cpu+fs-offload-20k all increase TTFT (+13.6% to +22.3%), consistent with write overhead exceeding any cache benefit at this model's throughput. fs-offload shows the largest ITL increase (+26.4%, 259.3 vs 205.2 ms).

**Qwen3-14B:** lmcache-valkey reduces TTFT by 8.8% (37.4 vs 41.0 s), consistent with remote prefix cache hits. native-offload-20k shows near-zero TTFT change (−0.7%) but reduces ITL by 6.0% (206.2 vs 219.4 ms), suggesting cache hits reduce the per-token generation cost. cpu+fs-offload-20k reduces ITL by 3.9% with minimal TTFT impact.

**Qwen3-32B-AWQ:** lmcache-valkey reduces TTFT by 4.7% (40.8 vs 42.8 s). All other configs show TTFT within ±7% of no-offload.

![Mempress Comparison](analysis/v0.7.0_mempress_comparison.png)
*Memory-pressure peak throughput: v0.6.0 vs v0.7.0, for configurations with valid v0.6.0 baselines.*

### Version Comparison (Memory Pressure)

| Model | Config | v0.5.1 | v0.6.0 | v0.7.0 | vs mempress no-offload |
|-------|--------|:------:|:------:|:------:|:----------------------:|
| Qwen3-0.6B | native-offload-20k | +22.3% | +51.4% | **+53.1%** | |
| | cpu+fs-offload-20k | — | — | **+34.6%** | (first valid) |
| Qwen3-8B | native-offload-20k | −3.6% | −16.3% | −18.0% | |
| | lmcache-valkey | +0.9% | +3.1% | **+1.9%** | |
| Qwen3-14B | native-offload-20k | +10.4% | −10.8%† | **+4.6%** | |
| | cpu+fs-offload-20k | — | — | **+3.2%** | (first valid) |

†v0.6.0 Qwen3-14B native-offload mempress: 3/8 rates excluded due to vLLM #38515.

---

## Filesystem Offload — First Valid Baseline

`fs-offload` and `cpu+fs-offload-20k` results in this evaluation are the first valid measurements with `llmd_fs_connector` correctly loaded. Prior evaluations produced invalid results due to system misconfiguration (PVC).

### Under Memory Pressure

![Filesystem Offload Mempress Baseline](analysis/v0.7.0_fs_offload_mempress_baseline.png)
*v0.7.0 filesystem offload under memory pressure — first valid baseline.*

Under memory pressure (the relevant operating condition for fs-offload):

| Model | fs-offload | vs mempress no-offload | cpu+fs-offload-20k | vs mempress no-offload |
|-------|:----------:|:----------------------:|:------------------:|:----------------------:|
| Qwen3-0.6B | 488.5 tok/s | −6.7% | 705.1 tok/s | **+34.6%** |
| Qwen3-8B | 81.1 tok/s | −24.0% | 84.3 tok/s | −21.0% |
| Qwen3-14B | 68.3 tok/s | −1.5% | 71.5 tok/s | **+3.2%** |
| Qwen3-32B-AWQ | 49.1 tok/s | −2.0% | 49.1 tok/s | −2.0% |

Qwen3-0.6B cpu+fs-offload-20k delivers the most significant result: +34.6% vs mempress no-offload. The CPU tier provides primary cache extension; the filesystem tier extends capacity further. Qwen3-8B incurs large overhead for both fs configs under mempress (−21% to −24%), consistent with the IBM VPC block PVC's 6,000 IOPS cap being saturated by this model's write rate even at reduced gmu. Qwen3-14B shows small but positive gains with cpu+fs (+3.2%), and near-zero overhead with fs-offload (−1.5%).

### At gmu=0.9 (unconstrained reference)

![Filesystem Offload Baseline](analysis/v0.7.0_fs_offload_baseline.png)
*v0.7.0 fs-offload at gmu=0.9. At this setting, Qwen3-0.6B and Qwen3-8B GPU KV caches are not saturated, so offload adds write overhead without cache-extension benefit.*

At gmu=0.9, Qwen3-0.6B and Qwen3-8B GPU KV caches are unsaturated on this hardware, so filesystem offload adds write overhead without benefit. Qwen3-14B and Qwen3-32B-AWQ, being nearer to KV saturation, show smaller overhead (−2.0% to −5.6% vs no-offload). The storage backend (6,000 IOPS PVC) is a binding constraint for Qwen3-8B at its 264.5 tok/s baseline throughput, producing −41.9% overhead.

---

## gmu=0.9 Results (Unconstrained Reference)

At the default gpu_memory_utilization=0.9, only Qwen3-14B is GPU KV-cache constrained on this hardware. Offload strategies show overhead rather than benefit for the other three models at this setting. These results document the unconstrained cost of each offload mechanism.

### Throughput vs Concurrency

![Throughput Curves](analysis/v0.7.0_throughput_curves.png)
*Output token throughput vs concurrency for all six configurations (gmu=0.9).*

#### Qwen3-0.6B

No-offload and native-offload-20k track within 0.1% across all concurrency levels (peak: 810.7 and 809.6 tok/s). lmcache-valkey follows at −2.4%. lmcache-local shows −20.1% at peak with throughput declining above rate=100, consistent with CPU cache eviction pressure. fs-offload peaks at 736.0 tok/s (rate=400) and declines at higher concurrency due to PVC write throughput limits.

#### Qwen3-8B

No-offload peaks at 264.5 tok/s (+34.1% vs v0.6.0). native-offload-20k incurs −39.9% overhead (158.9 tok/s), regressing from −6.5% in v0.6.0 and accompanied by 1.9× TTFT increase at rate=50 (5.95→11.46 s). lmcache-valkey recovers from the v0.6.0 anomaly (141.9 tok/s) to 204.8 tok/s. fs-offload and cpu+fs-offload-20k fall below lmcache configs (150–153 tok/s), consistent with PVC write overhead.

#### Qwen3-14B

Throughput is flat across all configs and concurrency levels (54–57 tok/s), confirming KV-cache saturation. All offload configs are within ±5.6% of no-offload. lmcache-valkey TTFT (34.9 s at rate=50) is −13.7% vs no-offload (40.4 s), indicating prefix cache hits reducing prefill time.

#### Qwen3-32B-AWQ

No-offload peaks at rate=1 (50.1 tok/s). All offload configs match within ±2% at rate=1 — a substantial improvement vs v0.6.0 (18.1 tok/s) where vLLM #38515 was responsible.

### Throughput Delta vs No-Offload

![Overhead Heatmap](analysis/v0.7.0_overhead_heatmap.png)
*Throughput % delta vs no-offload baseline (diverging colormap: red = overhead, green = gain, gmu=0.9).*

### Version Comparison (gmu=0.9)

![Version Comparison](analysis/v0.7.0_version_comparison.png)
*Peak throughput v0.5.1 / v0.6.0 / v0.7.0 at gmu=0.9 for configurations with valid baselines.*

![Peak Throughput](analysis/v0.7.0_peak_throughput.png)
*Peak throughput by configuration (gmu=0.9). Dashed outlines show v0.6.0 values where valid.*

---

## Latency Results (gmu=0.9)

### TTFT and ITL at rate=50

| Model | Config | TTFT (s) | ITL (ms) |
|-------|--------|:--------:|:--------:|
| Qwen3-0.6B | no-offload | 1.183 | 50.6 |
| | native-offload-20k | 1.147 | 50.8 |
| | lmcache-local | 1.761 | 61.6 |
| | lmcache-valkey | 1.197 | 51.4 |
| | fs-offload | 1.325 | 55.5 |
| | cpu+fs-offload-20k | 1.355 | 56.2 |
| Qwen3-8B | no-offload | 5.954 | 111.9 |
| | native-offload-20k | **11.463** | **170.7** |
| | lmcache-local | 8.880 | 145.1 |
| | lmcache-valkey | 7.046 | 146.9 |
| | fs-offload | 12.024 | 174.5 |
| | cpu+fs-offload-20k | 12.411 | 175.2 |
| Qwen3-14B | no-offload | 40.431 | 349.7 |
| | native-offload-20k | 38.116 | 349.7 |
| | lmcache-valkey | **34.895** | 349.7 |
| | fs-offload | 40.757 | 354.5 |
| Qwen3-32B-AWQ | no-offload | 31.050 | 537.1 |
| | lmcache-valkey | **28.443** | 528.6 |

![TTFT Curves](analysis/v0.7.0_ttft_curves.png)
*Time to first token vs concurrency (gmu=0.9).*

![ITL Curves](analysis/v0.7.0_itl_curves.png)
*Inter-token latency vs concurrency (gmu=0.9).*

### Latency Observations

**Qwen3-0.6B:** All configs show low TTFT at rate=50 (1.1–1.8 s). lmcache-local shows the largest TTFT (+48.9% vs no-offload, 1.761 s). native-offload-20k and lmcache-valkey are within 2% of no-offload on both TTFT and ITL.

**Qwen3-8B:** native-offload-20k TTFT is 11.463 s at rate=50, 1.92× the no-offload baseline (5.954 s). ITL increases from 111.9 to 170.7 ms (+52.5%). fs-offload and cpu+fs-offload-20k show similar profiles (12.0–12.4 s TTFT). The TTFT elevation is consistent with KV block transfer latency adding to prefill time for a model whose GPU KV cache is not constrained — offload writes occur on every step without cache-hit benefit.

**Qwen3-14B:** lmcache-valkey TTFT (34.9 s, −13.7% vs no-offload 40.4 s) is the lowest among offload configs, consistent with remote prefix cache hits reducing prefill computation. ITL is stable across all configs (349–360 ms).

**Qwen3-32B-AWQ:** lmcache-valkey TTFT (28.4 s, −8.4% vs no-offload 31.1 s) shows a modest improvement. ITL is within ±2% across all configs.

---

## Notable Changes vs v0.6.0

- **vLLM 0.19.1 / CUDA 13.0.2**: Requires NVIDIA driver ≥ 580.
- **llmd_fs_connector baked in**: Deployed via image build; no runtime pip install required. First valid fs-offload baseline.
- **LMCache v0.4.4**: LMCacheConnectorV1 for vLLM 0.19.x compatibility.
- **EPP v1.5.0**: `kv-cache-utilization-scorer` replaces deprecated flag.
- **vLLM #38515 resolved**: All four models complete all configurations without crashes.
- **Qwen3-8B no-offload +34.1%** at gmu=0.9 — specific vLLM 0.19.1 change not isolated.
- **Qwen3-8B native-offload regression** at gmu=0.9: −6.5% (v0.6.0) → −39.9% — cause not isolated.

**Supersedes:** REPORT-v0.6.0.md
