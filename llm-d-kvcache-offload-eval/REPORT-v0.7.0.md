# llm-d v0.7.0 KV-Cache Offload Evaluation

This report evaluates KV-cache offload strategies in llm-d v0.7.0 (vLLM 0.19.1, CUDA 13.0.2) across six configurations, four model sizes, and eight concurrency levels. Results are compared against v0.6.0 (vLLM 0.17.1) where valid baselines exist. This is the first evaluation with a valid filesystem offload configuration; fs-offload and cpu+fs-offload-20k results from v0.6.0 are excluded from comparisons due to a system misconfiguration (PVC) in that version (see REPORT-v0.6.0.md).

**TL;DR:** **Qwen3-8B no-offload throughput increases from 197 to 264 tok/s (+34.1%) with vLLM 0.19.1, the largest cross-version gain in this evaluation.** Qwen3-32B-AWQ native-offload overhead is eliminated (−63.9% in v0.6.0 to 0.0% in v0.7.0). Qwen3-8B native-offload regresses from −6.5% to −39.9% overhead vs no-offload at gmu=0.9, accompanied by 1.9× TTFT increase at rate=50 (5.95 s → 11.46 s). The first valid filesystem offload results show −3.5% overhead for Qwen3-14B and −9.2% for Qwen3-0.6B at gmu=0.9; Qwen3-8B incurs −41.9%. Under memory pressure, Qwen3-0.6B cpu+fs-offload reaches +34.6% vs the mempress no-offload baseline.

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

384 benchmark runs: 192 at default gpu_memory_utilization=0.9 (6 configs × 4 models × 8 rates) and 192 under memory pressure (6 configs × 4 models × 8 rates). All four models completed all six configurations with no exclusions due to crashes (vLLM #38515 resolved in this version).

**Configurations:**
1. **no-offload**: GPU-only KV-cache (baseline)
2. **native-offload-20k**: CPU offload via `OffloadingConnector` / `CPUOffloadingSpec`, 20K-block equivalent
3. **lmcache-local**: LMCache v0.4.4 with local CPU backend
4. **lmcache-valkey**: LMCache v0.4.4 with Valkey remote backend
5. **fs-offload**: Filesystem offload via `SharedStorageOffloadingSpec` (`llmd_fs_connector` v0.19, first valid baseline)
6. **cpu+fs-offload-20k**: Hierarchical CPU+filesystem offload via `MultiConnector` (first valid baseline)

**Peak Throughput at gmu=0.9:**

| Model | Config | v0.7.0 (tok/s) | Rate | vs no-offload | vs v0.6.0 |
|-------|--------|:--------------:|:----:|:-------------:|:---------:|
| Qwen3-0.6B | no-offload | 810.7 | 650 | — | +0.4% |
| | native-offload-20k | 809.6 | 650 | −0.1% | 0.0% |
| | lmcache-local | 647.5 | 650 | −20.1% | −2.7% |
| | lmcache-valkey | 791.5 | 650 | −2.4% | +0.3% |
| | fs-offload | 736.0 | 400 | −9.2% | (new) |
| | cpu+fs-offload-20k | 728.5 | 400 | −10.1% | (new) |
| Qwen3-8B | no-offload | 264.5 | 650 | — | +34.1% |
| | native-offload-20k | 158.9 | 650 | −39.9% | −13.9% |
| | lmcache-local | 196.3 | 650 | −25.8% | +1.1% |
| | lmcache-valkey | 204.8 | 650 | −22.6% | +44.3% |
| | fs-offload | 153.6 | 400 | −41.9% | (new) |
| | cpu+fs-offload-20k | 150.4 | 400 | −43.1% | (new) |
| Qwen3-14B | no-offload | 57.6 | 650 | — | +3.8% |
| | native-offload-20k | 56.5 | 500 | −1.9% | +0.1% |
| | lmcache-local | 55.5 | 650 | −3.6% | +2.0% |
| | lmcache-valkey | 56.5 | 650 | −1.9% | +0.1% |
| | fs-offload | 55.5 | 650 | −3.5% | (new) |
| | cpu+fs-offload-20k | 54.4 | 650 | −5.6% | (new) |
| Qwen3-32B-AWQ | no-offload | 50.1 | 1 | — | +0.1% |
| | native-offload-20k | 50.1 | 1 | 0.0% | +177% |
| | lmcache-local | 50.1 | 1 | 0.0% | +177% |
| | lmcache-valkey | 50.1 | 1 | 0.0% | +161% |
| | fs-offload | 49.1 | 1 | −2.0% | (new) |
| | cpu+fs-offload-20k | 49.1 | 1 | −2.0% | (new) |

(new) = first valid baseline; v0.6.0 fs-offload excluded due to system misconfiguration (PVC).

The large v0.6.0→v0.7.0 percentage changes for Qwen3-32B-AWQ offload configs reflect a known misconfiguration in v0.6.0 (vLLM #38515), not a genuine v0.7.0 improvement.

![Peak Throughput](analysis/v0.7.0_peak_throughput.png)
*Peak throughput by configuration for each model (gmu=0.9). Dashed outlines show v0.6.0 values where valid; fs-offload has no v0.6.0 baseline.*

**Native Offload Overhead vs No-Offload (gmu=0.9):**

| Version | Qwen3-0.6B | Qwen3-8B | Qwen3-14B | Qwen3-32B-AWQ |
|---------|:----------:|:--------:|:---------:|:-------------:|
| v0.5.1 | −2.2% | −29.9% | +14.5% | −58.4% |
| v0.6.0 | +0.3% | −6.5% | +1.8% | −63.9% |
| v0.7.0 | −0.1% | −39.9% | −1.9% | 0.0% |

**Peak Throughput under Memory Pressure:**

| Model | Config | v0.7.0 (tok/s) | Rate | vs mempress no-offload |
|-------|--------|:--------------:|:----:|:----------------------:|
| Qwen3-0.6B | no-offload | 523.7 | 50 | — |
| | native-offload-20k | 802.1 | 50 | +53.1% |
| | lmcache-local | 494.9 | 50 | −5.5% |
| | lmcache-valkey | 501.3 | 50 | −4.3% |
| | fs-offload | 488.5 | 50 | −6.7% |
| | cpu+fs-offload-20k | 705.1 | 50 | +34.6% |
| Qwen3-8B | no-offload | 106.7 | 100 | — |
| | native-offload-20k | 87.5 | 50 | −18.0% |
| | lmcache-local | 104.5 | 100 | −2.1% |
| | lmcache-valkey | 108.8 | 100 | +1.9% |
| | fs-offload | 81.1 | 100 | −24.0% |
| | cpu+fs-offload-20k | 84.3 | 100 | −21.0% |
| Qwen3-14B | no-offload | 69.3 | 100 | — |
| | native-offload-20k | 72.5 | 300 | +4.6% |
| | lmcache-local | 68.3 | 100 | −1.5% |
| | lmcache-valkey | 68.3 | 100 | −1.5% |
| | fs-offload | 68.3 | 100 | −1.5% |
| | cpu+fs-offload-20k | 71.5 | 100 | +3.2% |
| Qwen3-32B-AWQ | no-offload | 50.1 | 1 | — |
| | native-offload-20k | 50.1 | 1 | 0.0% |
| | lmcache-local | 50.1 | 1 | 0.0% |
| | lmcache-valkey | 50.1 | 1 | 0.0% |
| | fs-offload | 49.1 | 1 | −2.0% |
| | cpu+fs-offload-20k | 49.1 | 1 | −2.0% |

---

## Test Configuration

### Hardware

**System:** OpenShift cluster on IBM Cloud (Single Node OpenShift, SNO)
- **GPUs**: 2× NVIDIA L40S (24 GB VRAM each, 48 GB total)
  - Tensor Parallelism: 2 GPUs per model
- **CPU**: 48 vCPUs
- **Storage**: IBM VPC block PVC (256 GiB, `ibmc-vpc-block-custom`, 6,000 IOPS) at `/kvcache` for filesystem offload

### Software

See software versions table in the preamble.

**EPP change in v0.7.0:** gateway-api-inference-extension v1.5.0 removes `--kv-cache-usage-percentage-metric` (deprecated in v1.x). The `kv-cache-utilization-scorer` is now configured via `EndpointPickerConfig` in the EPP ConfigMap.

**CUDA 13.0.2:** All llm-d v0.7.0 images use CUDA 13.0.2, requiring NVIDIA driver ≥ 580 on host nodes.

**llmd_fs_connector deployment:** `llmd_fs_connector` v0.19 is baked into `ghcr.io/llm-d/llm-d-cuda:v0.7.0` at image build time via `docker/scripts/cuda/runtime/install-offloading-connector.sh`, using the `linux_x86_64`-tagged wheels from the `kv_connectors/llmd_fs_backend/wheels/` directory. No runtime pip install is required.

### Workload

**Profile:** Concurrent multi-turn conversation with shared prefix
- Concurrency levels: 1, 50, 100, 150, 300, 400, 500, 650
- Duration: 120 seconds per concurrency level
- Prompt tokens: 128 per turn; Output tokens: 128 per turn
- Prefix tokens: 10,000 (shared across requests); Turns: 5; Random seed: 889

### Memory-Pressure gpu_memory_utilization Settings

| Model | gmu=0.9 (default) | mempress gmu |
|-------|:-----------------:|:------------:|
| Qwen3-0.6B | 0.9 | 0.55 |
| Qwen3-8B | 0.9 | 0.65 |
| Qwen3-14B | 0.9 | 0.70 |
| Qwen3-32B-AWQ | 0.9 | 0.65 |

---

## v0.7.0 Performance Results (gmu=0.9)

### Throughput vs Concurrency

![Throughput Curves](analysis/v0.7.0_throughput_curves.png)
*Output token throughput vs concurrency for all six configurations (gmu=0.9). Each panel shows one model.*

#### Qwen3-0.6B

No-offload and native-offload-20k track within 0.1% across all concurrency levels, peaking at 810.7 and 809.6 tok/s at rate=650. lmcache-valkey follows closely (−2.4% at peak). lmcache-local shows −20.1% at peak, with throughput declining from rate=100 onward, consistent with CPU cache eviction pressure at higher concurrency. fs-offload peaks at 736.0 tok/s (rate=400) and declines at higher concurrency, consistent with PVC I/O throughput limiting write throughput at high token rates. cpu+fs-offload-20k follows a similar profile (728.5 tok/s).

#### Qwen3-8B

No-offload peaks at 264.5 tok/s (rate=650), a 34.1% improvement vs v0.6.0. native-offload-20k incurs −39.9% overhead (158.9 tok/s), regressing from −6.5% in v0.6.0. The regression is accompanied by 1.9× TTFT increase at rate=50 (5.95 s → 11.46 s) — see latency section. lmcache-valkey reaches 204.8 tok/s (−22.6%), recovering from the v0.6.0 anomaly where valkey underperformed local by 27%. lmcache-local reaches 196.3 tok/s (−25.8%). fs-offload and cpu+fs-offload-20k both fall below lmcache configs (153–150 tok/s), consistent with PVC write overhead dominating at this model's token rate.

#### Qwen3-14B

Throughput curves are flat across all configurations and concurrency levels (54–57 tok/s range), consistent with Qwen3-14B being GPU KV-cache constrained on this hardware. All offload configurations are within ±5.6% of the no-offload baseline. native-offload-20k and lmcache-valkey achieve −1.9%; fs-offload reaches −3.5%; cpu+fs-offload-20k −5.6%. lmcache-local shows −3.6%.

#### Qwen3-32B-AWQ

No-offload peaks at rate=1 (50.1 tok/s), declining to 16–20 tok/s at rate=50+. All offload configurations match no-offload within ±2% at rate=1: native-offload-20k, lmcache-local, and lmcache-valkey are all 50.1 tok/s (0.0%); fs-offload and cpu+fs-offload-20k reach 49.1 tok/s (−2.0%). This represents a large improvement for native and lmcache configs vs v0.6.0, where these were at 18.1 tok/s due to vLLM #38515.

### Throughput Delta vs No-Offload

![Overhead Heatmap](analysis/v0.7.0_overhead_heatmap.png)
*Throughput % delta vs no-offload baseline (magma colormap, gmu=0.9). Positive values (lighter) indicate gains; negative values (darker) indicate overhead.*

### Version Comparison

![Version Comparison](analysis/v0.7.0_version_comparison.png)
*Peak throughput v0.5.1 / v0.6.0 / v0.7.0 for configurations with valid baselines across all versions. fs-offload excluded — v0.6.0 results invalid (system misconfiguration, PVC); see Figure 6.*

---

## Latency Results (gmu=0.9)

### Latency at rate=50

Rate=50 represents a moderate concurrency level where latency differences between configurations are most interpretable. Rate=1 used for Qwen3-32B-AWQ (peak throughput rate for this model).

| Model | Config | TTFT (s) | ITL (ms) |
|-------|--------|:--------:|:--------:|
| Qwen3-0.6B | no-offload | 1.183 | 50.6 |
| | native-offload-20k | 1.147 | 50.8 |
| | lmcache-local | 1.761 | 61.6 |
| | lmcache-valkey | 1.197 | 51.4 |
| | fs-offload | 1.325 | 55.5 |
| | cpu+fs-offload-20k | 1.355 | 56.2 |
| Qwen3-8B | no-offload | 5.954 | 111.9 |
| | native-offload-20k | 11.463 | 170.7 |
| | lmcache-local | 8.880 | 145.1 |
| | lmcache-valkey | 7.046 | 146.9 |
| | fs-offload | 12.024 | 174.5 |
| | cpu+fs-offload-20k | 12.411 | 175.2 |
| Qwen3-14B | no-offload | 40.431 | 349.7 |
| | native-offload-20k | 38.116 | 349.7 |
| | lmcache-local | 41.576 | 355.3 |
| | lmcache-valkey | 34.895 | 349.7 |
| | fs-offload | 40.757 | 354.5 |
| | cpu+fs-offload-20k | 40.978 | 359.6 |
| Qwen3-32B-AWQ | no-offload | 31.050 | 537.1 |
| | native-offload-20k | 31.306 | 532.6 |
| | lmcache-local | 30.471 | 547.5 |
| | lmcache-valkey | 28.443 | 528.6 |
| | fs-offload | 29.963 | 537.6 |
| | cpu+fs-offload-20k | 30.200 | 542.7 |

### TTFT and ITL Curves

![TTFT Curves](analysis/v0.7.0_ttft_curves.png)
*Time to first token vs concurrency for all configurations (gmu=0.9). Note differing y-axis scales per model.*

![ITL Curves](analysis/v0.7.0_itl_curves.png)
*Inter-token latency vs concurrency for all configurations (gmu=0.9).*

### Latency Observations

**Qwen3-0.6B:** All configurations show low TTFT at rate=50 (1.1–1.8 s). lmcache-local shows the largest TTFT (+48.9% vs no-offload at rate=50, 1.761 s vs 1.183 s) and ITL (+21.7 ms). native-offload-20k, lmcache-valkey, and no-offload are within 2% of each other on both TTFT and ITL.

**Qwen3-8B:** native-offload-20k TTFT at rate=50 is 11.463 s, 1.92× the no-offload baseline (5.954 s). ITL increases from 111.9 to 170.7 ms (+52.5%). fs-offload and cpu+fs-offload-20k show similar latency profiles to native-offload (12.0–12.4 s TTFT). lmcache-valkey (7.046 s TTFT, +18.3%) and lmcache-local (8.880 s, +49.1%) show intermediate increases. The TTFT elevation for native/fs configs at rate=50 is consistent with KV block transfer latency adding to prefill time.

**Qwen3-14B:** TTFT is inherently high (34–41 s at rate=50) due to KV-cache saturation causing queuing. lmcache-valkey shows the lowest TTFT among offload configs at rate=50 (34.895 s, −13.7% vs no-offload 40.431 s), consistent with remote cache hits reducing prefill computation. ITL is stable across all configs (349–360 ms).

**Qwen3-32B-AWQ:** At rate=1 (peak throughput rate), TTFT differences are small (28–31 s range). lmcache-valkey shows slightly lower TTFT (28.443 s vs 31.050 s no-offload, −8.4%), consistent with prefix cache hits at this prefix-heavy workload. ITL ranges from 528 to 547 ms across configs (within ±2%).

---

## Filesystem Offload — First Valid Baseline

`fs-offload` and `cpu+fs-offload-20k` results represent the first valid measurements with `llmd_fs_connector` correctly loaded. Prior evaluations (v0.5.1, v0.6.0) produced invalid filesystem offload results due to system misconfiguration (PVC).

![Filesystem Offload Baseline](analysis/v0.7.0_fs_offload_baseline.png)
*v0.7.0 fs-offload first baseline (gmu=0.9): no-offload, fs-offload, and cpu+fs-offload-20k for all four models. Annotations show % vs no-offload.*

At gmu=0.9, Qwen3-0.6B and Qwen3-8B GPU KV caches are not saturated on this hardware, so filesystem offload provides no cache-extension benefit and write overhead reduces throughput. Qwen3-14B and Qwen3-32B-AWQ are near KV cache saturation, resulting in smaller overhead (−2.0% to −5.6%).

The storage backend (IBM VPC block PVC, 6,000 IOPS cap) is a binding constraint for Qwen3-8B: at 264.5 tok/s baseline throughput, the per-step PVC write rate exceeds the IOPS budget, producing −41.9% overhead. Qwen3-14B at 57.6 tok/s baseline generates fewer writes per second, resulting in only −3.5% overhead.

![Filesystem Offload Mempress Baseline](analysis/v0.7.0_fs_offload_mempress_baseline.png)
*v0.7.0 filesystem offload first baseline under memory pressure.*

Under memory pressure, Qwen3-0.6B cpu+fs-offload-20k reaches +34.6% vs the mempress no-offload baseline (705.1 vs 523.7 tok/s). The CPU tier provides primary cache extension for this model; the filesystem tier contributes additional capacity. Qwen3-8B incurs overhead for both fs configs (−21% to −24%), consistent with PVC I/O being a bottleneck at this model's generation rate even under reduced gmu.

---

## Memory-Pressure Results

![Mempress Comparison](analysis/v0.7.0_mempress_comparison.png)
*Memory-pressure peak throughput: v0.6.0 vs v0.7.0, for configurations with valid v0.6.0 baselines.*

Qwen3-0.6B native-offload-20k mempress: +53.1% vs mempress no-offload baseline (802.1 vs 523.7 tok/s), consistent with the trend from v0.5.1 (+22.3%) and v0.6.0 (+51.4%). Qwen3-8B native-offload shows −18.0% under mempress. Qwen3-14B native-offload delivers +4.6% (72.5 vs 69.3 tok/s).

Under memory pressure, LMCache shows −5.5% to −4.3% for Qwen3-0.6B vs mempress no-offload, and −2.1% to +1.9% for Qwen3-8B. Qwen3-14B LMCache results are −1.5%.

---

## Notable Changes vs v0.6.0

- **vLLM 0.19.1 / CUDA 13.0.2**: Requires NVIDIA driver ≥ 580.
- **llmd_fs_connector baked in**: Deployed via image build (`install-offloading-connector.sh`); no runtime pip install required.
- **LMCache v0.4.4**: Installed at pod startup via `pip3.12 install lmcache==0.4.4`. LMCacheConnectorV1 used for vLLM 0.19.x compatibility.
- **EPP v1.5.0**: `kv-cache-utilization-scorer` replaces deprecated `--kv-cache-usage-percentage-metric` flag.
- **vLLM #38515 resolved**: Qwen3-32B-AWQ offload configurations no longer crash. All four models complete all eight concurrency levels across both gmu=0.9 and mempress suites.
- **Qwen3-8B native-offload regression**: Overhead increases from −6.5% (v0.6.0) to −39.9% (v0.7.0) at gmu=0.9. Cause not isolated to a specific vLLM 0.19.1 change.
- **Qwen3-8B no-offload +34.1%**: Consistent with vLLM 0.19.1 throughput improvements; specific change not isolated.

**Supersedes:** REPORT-v0.6.0.md
