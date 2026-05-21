# llm-d v0.7.0 KV-Cache Offload Evaluation

This report evaluates KV-cache offload strategies in llm-d v0.7.0 (vLLM 0.19.1, CUDA 13.0.2) across six configurations, four model sizes, and eight concurrency levels. Results are compared against v0.6.0 (vLLM 0.17.1) where valid baselines exist. This is the first evaluation with a valid filesystem offload configuration; fs-offload and cpu+fs-offload-20k results from v0.6.0 are excluded from comparisons due to a system misconfiguration (PVC) in that version (see REPORT-v0.6.0.md).

**TL;DR:** **Qwen3-8B no-offload throughput increases from 197 to 264 tok/s (+34.1%) with vLLM 0.19.1, the largest cross-version gain in this evaluation.** Qwen3-32B-AWQ native-offload overhead is eliminated (−63.9% in v0.6.0 to 0.0% in v0.7.0). Qwen3-8B native-offload regresses from −6.5% to −39.9% overhead vs no-offload at gmu=0.9. The first valid filesystem offload results show −3.5% overhead for Qwen3-14B and −9.2% for Qwen3-0.6B at gmu=0.9; Qwen3-8B incurs −41.9% overhead. Under memory pressure, Qwen3-0.6B cpu+fs-offload reaches +34.6% vs the mempress no-offload baseline.

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

384 benchmark runs: 192 at default gpu_memory_utilization=0.9 (6 configs × 4 models × 8 rates) and 192 under memory pressure (6 configs × 4 models × 8 rates). All four models ran all six configurations, including filesystem offload, with no exclusions due to crashes (vLLM #38515 resolved in this version).

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

The v0.6.0-to-v0.7.0 percentage changes for native/lmcache configs for Qwen3-32B-AWQ are large because v0.6.0 offload results for this model were 18.1 tok/s — a known misconfiguration in that version (vLLM #38515). The v0.7.0 values represent correct behaviour.

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
- Prompt tokens: 128 per turn
- Output tokens: 128 per turn
- Prefix tokens: 10,000 (shared across requests)
- Turns per conversation: 5
- Random seed: 889 (deterministic)

### Memory-Pressure gpu_memory_utilization Settings

Same values as v0.5.1 and v0.6.0 for cross-version comparability:

| Model | gmu=0.9 (default) | mempress gmu |
|-------|:-----------------:|:------------:|
| Qwen3-0.6B | 0.9 | 0.55 |
| Qwen3-8B | 0.9 | 0.65 |
| Qwen3-14B | 0.9 | 0.70 |
| Qwen3-32B-AWQ | 0.9 | 0.65 |

---

## Observations

### 1. No-Offload Baseline

Qwen3-8B no-offload throughput increases from 197.3 to 264.5 tok/s (+34.1%) between v0.6.0 and v0.7.0. Qwen3-0.6B is stable at 810.7 (+0.4%). Qwen3-14B shows +3.8% (55.5→57.6 tok/s). Qwen3-32B-AWQ is stable at 50.1 tok/s (+0.1%).

The Qwen3-8B gain is consistent with vLLM 0.19.1 improvements; the specific change has not been isolated to a single upstream commit.

### 2. Native CPU Offload (OffloadingConnector)

At gmu=0.9, native-offload-20k overhead vs no-offload:

| Model | v0.5.1 | v0.6.0 | v0.7.0 |
|-------|:------:|:------:|:------:|
| Qwen3-0.6B | −2.2% | +0.3% | −0.1% |
| Qwen3-8B | −29.9% | −6.5% | −39.9% |
| Qwen3-14B | +14.5% | +1.8% | −1.9% |
| Qwen3-32B-AWQ | −58.4% | −63.9% | 0.0% |

Qwen3-0.6B and Qwen3-14B show near-zero native-offload overhead (−0.1% and −1.9%). Qwen3-32B-AWQ native-offload overhead is eliminated in v0.7.0: the model at gmu=0.9 saturates at rate=1 (50.1 tok/s) regardless of config, indicating the AWQ-quantized 32B model's KV cache fits entirely in GPU memory at this utilization.

Qwen3-8B native-offload regresses from −6.5% (v0.6.0) to −39.9% (v0.7.0). Both models fit in GPU KV cache at gmu=0.9, so offloading adds overhead with no benefit. The cause of the larger overhead in v0.7.0 compared to v0.6.0 has not been isolated; the OffloadingConnector behavior may have changed between vLLM 0.17.1 and 0.19.1.

Under memory pressure, Qwen3-0.6B native-offload delivers +53.1% vs the mempress no-offload baseline (802.1 vs 523.7 tok/s), consistent with the trend observed in v0.5.1 (+22.3%) and v0.6.0 (+51.4%). Qwen3-8B native-offload shows −18.0% under mempress, and Qwen3-14B +4.6%.

### 3. LMCache (local and Valkey)

At gmu=0.9, lmcache-local overhead vs no-offload: Qwen3-0.6B −20.1%, Qwen3-8B −25.8%, Qwen3-14B −3.6%, Qwen3-32B-AWQ 0.0%.

lmcache-valkey overhead: Qwen3-0.6B −2.4%, Qwen3-8B −22.6%, Qwen3-14B −1.9%, Qwen3-32B-AWQ 0.0%.

Compared to v0.6.0, lmcache-valkey for Qwen3-8B improves from 141.9 to 204.8 tok/s (+44.3%). The v0.6.0 Qwen3-8B lmcache-valkey result of 141.9 tok/s was below the local-backend result (194.1 tok/s), suggesting an anomaly in the v0.6.0 valkey run rather than a structural difference between backends. The v0.7.0 results show lmcache-local (196.3) and lmcache-valkey (204.8) within 4% of each other for Qwen3-8B.

Under memory pressure, lmcache shows −5.5% to −4.3% for Qwen3-0.6B and −2.1% to +1.9% for Qwen3-8B vs the mempress no-offload baseline. Qwen3-14B lmcache results are stable at −1.5%. LMCache v0.4.4 (vs v0.4.2 in v0.6.0) shows no throughput regression relative to expectations.

### 4. Filesystem Offload — First Valid Baseline

`fs-offload` and `cpu+fs-offload-20k` results in this report represent the first valid measurements. Prior evaluations (v0.5.1, v0.6.0) produced invalid filesystem offload results due to system misconfiguration (PVC).

**At gmu=0.9 (both models fit in GPU KV cache except Qwen3-14B at this hardware scale):**

| Model | fs-offload | vs no-offload | cpu+fs-offload-20k | vs no-offload |
|-------|:----------:|:-------------:|:------------------:|:-------------:|
| Qwen3-0.6B | 736.0 tok/s | −9.2% | 728.5 tok/s | −10.1% |
| Qwen3-8B | 153.6 tok/s | −41.9% | 150.4 tok/s | −43.1% |
| Qwen3-14B | 55.5 tok/s | −3.5% | 54.4 tok/s | −5.6% |
| Qwen3-32B-AWQ | 49.1 tok/s | −2.0% | 49.1 tok/s | −2.0% |

At gmu=0.9, Qwen3-0.6B and Qwen3-8B GPU KV caches are not saturated, so filesystem offload provides no cache-extension benefit and its overhead reduces throughput. Qwen3-14B and Qwen3-32B-AWQ approach KV cache saturation at this scale, resulting in smaller overhead (−2.0% to −5.6%).

The storage backend (IBM VPC block PVC, 6,000 IOPS cap, 256 GiB) is likely a bottleneck for the Qwen3-8B overhead: the fs connector writes KV blocks to the PVC on every generation step, and at 264.5 tok/s baseline throughput, the PVC I/O bound is reached sooner than for the larger, slower models.

**Under memory pressure (gmu=0.55–0.70):**

| Model | fs-offload | vs mempress no-offload | cpu+fs-offload-20k | vs mempress no-offload |
|-------|:----------:|:----------------------:|:------------------:|:----------------------:|
| Qwen3-0.6B | 488.5 tok/s | −6.7% | 705.1 tok/s | +34.6% |
| Qwen3-8B | 81.1 tok/s | −24.0% | 84.3 tok/s | −21.0% |
| Qwen3-14B | 68.3 tok/s | −1.5% | 71.5 tok/s | +3.2% |
| Qwen3-32B-AWQ | 49.1 tok/s | −2.0% | 49.1 tok/s | −2.0% |

Under memory pressure, Qwen3-0.6B cpu+fs-offload-20k reaches +34.6% vs the mempress no-offload baseline (705.1 vs 523.7 tok/s). The MultiConnector writes to CPU and filesystem concurrently; for the smallest model the combined cache extension offsets the write overhead. Qwen3-8B cpu+fs-offload shows −21.0%: the PVC I/O overhead dominates cache extension benefit at this model size. Qwen3-14B cpu+fs-offload is +3.2%, consistent with cache extension benefit at the margin of KV saturation.

fs-offload without CPU offload shows smaller mempress gains than cpu+fs-offload-20k for Qwen3-0.6B (−6.7% vs +34.6%), indicating that the CPU tier provides the primary cache extension benefit at this scale; the filesystem tier adds write overhead for Qwen3-0.6B under mempress conditions.

### 5. MultiConnector Deadlock (vLLM #38515)

The EngineCore shared-memory broadcast deadlock that blocked cpu+fs-offload-20k under memory pressure in v0.6.0 does not recur in v0.7.0. All four models completed all eight concurrency levels across both gmu=0.9 and mempress suites with no errors.

---

## Notable Changes vs v0.6.0

- **vLLM 0.19.1 / CUDA 13.0.2**: Requires NVIDIA driver ≥ 580.
- **llmd_fs_connector baked in**: Deployed via image build (`install-offloading-connector.sh`); no runtime pip install required.
- **LMCache v0.4.4**: Installed at pod startup via `pip3.12 install lmcache==0.4.4`. LMCacheConnectorV1 used for vLLM 0.19.x compatibility.
- **EPP v1.5.0**: `kv-cache-utilization-scorer` replaces deprecated `--kv-cache-usage-percentage-metric` flag.
- **vLLM #38515 resolved**: Qwen3-32B-AWQ offload configurations no longer crash. 32B-AWQ native-offload, lmcache-local, and lmcache-valkey all complete at gmu=0.9 and mempress.

---

## Figures

1. `analysis/v0.7.0_throughput_curves.png` — Throughput vs concurrency for all configs and models (gmu=0.9)
2. `analysis/v0.7.0_version_comparison.png` — Peak throughput v0.5.1 / v0.6.0 / v0.7.0 for no-offload, native, and LMCache configs
3. `analysis/v0.7.0_overhead_heatmap.png` — Throughput % delta vs no-offload baseline (gmu=0.9), all configs × models
4. `analysis/v0.7.0_mempress_comparison.png` — Memory-pressure peak throughput v0.6.0 vs v0.7.0 for configs with valid baselines
5. `analysis/v0.7.0_ttft_comparison.png` — TTFT at rate=650 (gmu=0.9)
6. `analysis/v0.7.0_fs_offload_baseline.png` — Filesystem offload first baseline vs no-offload (gmu=0.9)
7. `analysis/v0.7.0_fs_offload_mempress_baseline.png` — Filesystem offload first baseline under memory pressure

**Supersedes:** REPORT-v0.6.0.md
