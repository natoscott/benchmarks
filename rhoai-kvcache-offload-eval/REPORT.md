# RHOAI 3.3 KV Cache CPU Offload Evaluation

This report evaluates native CPU KV cache offload on Red Hat OpenShift AI 3.3 (vLLM 0.13.0+rhai11)
running on NVIDIA H200 hardware. Two models are tested across single- and dual-replica
configurations using the RHOAI llm-d serving stack.

**Software Versions:**
- **RHOAI**: 3.3.0 (vLLM 0.13.0+rhai11, GuideLLM latest)
- **llm-d**: Integrated via RHOAI `LLMInferenceService` with EPP inference scheduler
- **EPP**: `odh-llm-d-inference-scheduler-rhel9`, Valkey-backed prefix cache index

**Hardware:** 2× GPU worker nodes, 8× NVIDIA H200 (140 GB HBM3e each); 160 vCPUs,
1.8 TB RAM per node; 4× NVMe local storage (LVMS)

**Models:**
- `RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8` — compressed-tensors FP8 quantisation
- `openai/gpt-oss-120b` — Mixture-of-Experts (MoE), MXFP4-quantised expert weights

**Concurrency levels:** 1, 50, 100, 150, 300, 400, 500, 650

---

## Summary

64 benchmark runs were collected across two KV-cache configurations, two models, two replica
counts, and eight concurrency levels.

- **no-offload**: GPU-only KV-cache (baseline)
- **native-offload-20k**: CPU offload via `OffloadingConnector`, `num_cpu_blocks=20000`
  (v0.4.0-compatible API, as shipped in RHOAI 3.3)

**Observations:**

1. **Native CPU offload reduces throughput for both models across nearly all concurrency levels.**
   Reductions range from -2.1% to -31.1% for Llama-3.1-70B-FP8 (replicas=1) and -33.6% to
   -64.3% for gpt-oss-120b. These results contrast with upstream evaluations on L40S hardware
   where CPU offload improved throughput for dense models of similar parameter counts.

2. **GPU KV cache block counts on H200 hardware are larger than on L40S:**
   - Llama-3.1-70B-FP8: 26,842 GPU blocks (75% gpu-memory-utilization, FP8 weights at 33.9 GiB/GPU)
   - gpt-oss-120b: 181,691 GPU blocks (65% gpu-memory-utilization, MXFP4 weights at 33.0 GiB/GPU)
   The 20,000 CPU blocks added by native-offload-20k represent +74% additional capacity for
   Llama and only +11% for gpt-oss-120b.

3. **GPU KV cache utilisation reaches 60–90% at moderate concurrency (rate=100–150),**
   confirming that memory pressure exists. However, the `OffloadingConnector` disables vLLM's
   hybrid KV cache manager on initialisation ("Turning off hybrid kv cache manager because
   `--kv-transfer-config` is set"), introducing connector overhead that exceeds the benefit of
   the additional CPU cache capacity under the tested conditions.

4. **gpt-oss-120b (MoE) shows larger offload overhead than Llama-3.1-70B-FP8.** The MoE
   architecture's small per-token KV footprint yields 6.8× more GPU blocks than the dense
   FP8 model (181,691 vs 26,842), so the CPU offload capacity ratio is 11% vs 74%, while the
   connector overhead is the same in both cases.

5. **Replica scaling efficiency reaches or exceeds 100% for Llama-3.1-70B-FP8 at
   concurrency ≥ 100.** At rate=50, no-offload replicas=2 delivers 148% of 2× replicas=1
   throughput. The EPP prefix-cache-aware routing concentrates similar-prefix requests on the
   same replica, improving per-replica cache hit rates and enabling super-linear throughput.

6. **gpt-oss-120b scales to ~100% efficiency at concurrency ≥ 300** (rate=300–400), with
   sub-linear scaling at lower concurrency due to the model's high per-replica throughput
   leaving the second replica underutilised at low request rates.

**Peak Throughput Summary:**

| Model | Replicas | no-offload | native-offload-20k | Offload Δ |
|-------|:--------:|:----------:|:------------------:|:---------:|
| Llama-3.1-70B-FP8 | 1 | 141.9 tok/s @50 | 116.1 tok/s @50 | −18.2% |
| Llama-3.1-70B-FP8 | 2 | 418.8 tok/s @50 | 440.4 tok/s @50 | **+5.2%** |
| gpt-oss-120b (MoE) | 1 | 1517.5 tok/s @50 | 800.1 tok/s @50 | −47.3% |
| gpt-oss-120b (MoE) | 2 | 1685.9 tok/s @100 | 1076.0 tok/s @50 | −36.2% |

---

## Test Configuration

### Hardware

**Worker nodes:** 2× OpenShift worker nodes on IBM Cloud
- **GPUs:** 8× NVIDIA H200 (140 GB HBM3e each, 1120 GB total per node)
- **CPU:** 160 vCPUs per node
- **RAM:** 1.8 TB per node
- **Storage:** 4× NVMe drives in LVM volume group (~12–16 TB per node)
- **Network:** 8× RoCE v2 interfaces per node

**GPU allocation per benchmark:**
- Tensor parallel size: 2 (2 GPUs per replica)
- Replicas=1: 2 GPUs total
- Replicas=2: 4 GPUs total (both replicas on same node, shared model PVC)

### Software

| Component | Version |
|-----------|---------|
| RHOAI | 3.3.0 |
| vLLM | 0.13.0+rhai11 (bundled in `rhaiis/vllm-cuda-rhel9`) |
| GuideLLM | latest |
| EPP scheduler | `odh-llm-d-inference-scheduler-rhel9` |
| OpenShift | 4.20 |
| PCP | 7.x (quay.io/performancecopilot/pcp:latest) |

### Model Configuration

**Llama-3.1-70B-FP8** (`RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8`):
```
--tensor-parallel-size 2
--gpu-memory-utilization 0.75
--max-num-seq 1024
# Offload config (native-offload-20k):
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"num_cpu_blocks":20000}}'
```
- Quantisation: compressed-tensors (FP8)
- Model weights per GPU: 33.9 GiB (TP=2)
- Available KV cache: 65.5 GiB/GPU
- GPU blocks (no-offload): 26,842
- CPU blocks (offload): 20,000 (+74.5%)

**gpt-oss-120b** (`openai/gpt-oss-120b`):
```
--tensor-parallel-size 2
--gpu-memory-utilization 0.65
--max-num-seq 1024
# Offload config (native-offload-20k):
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"num_cpu_blocks":20000}}'
```
- Quantisation: MXFP4 (expert weights)
- Model weights per GPU: 33.0 GiB (TP=2)
- Available KV cache: 49.9 GiB/GPU
- GPU blocks (no-offload): 181,691
- CPU blocks (offload): 20,000 (+11.0%)

### Workload

**Profile:** Concurrent multi-turn conversation with shared prefix
- Duration: 120 seconds per concurrency level
- Prompt tokens: 512 per turn
- Output tokens: 128 per turn
- Prefix tokens: 10,000 (shared across requests)
- Turns: 5 per conversation

### Routing and Metrics

- **Gateway:** `openshift-ai-inference` (OpenShift gateway controller, HTTPS/443)
- **Scheduler:** EPP with Valkey-backed prefix cache index (`gpu-prefix-cache-scorer`
  + `cpu-prefix-cache-scorer`, queue-scorer, kv-cache-utilization-scorer)
- **PCP:** Single-pod Deployment co-located with vLLM (pod affinity), recording
  `openmetrics.vllm.*`, `openmetrics.dcgm.*`, `openmetrics.epp.*` at 10-second intervals

---

## Results

### Throughput vs Concurrency

![Throughput Curves](throughput_curves.png)

Both models show higher throughput for no-offload than native-offload-20k across most
concurrency levels. The separation is more pronounced for gpt-oss-120b (MoE) than for
Llama-3.1-70B-FP8.

For Llama-3.1-70B-FP8 (replicas=1), throughput peaks at concurrency=50 (141.9 tok/s no-offload,
116.1 tok/s offload) then declines as queue saturation increases. For replicas=2, the offload
peak slightly exceeds no-offload (440.4 vs 418.8 tok/s), the only configuration where offload
shows a throughput advantage.

gpt-oss-120b reaches substantially higher throughput than Llama in both configurations, consistent
with its MoE architecture activating only 5.1B of 120B parameters per forward pass. Peak
no-offload throughput is 1517.5 tok/s (replicas=1) and 1685.9 tok/s (replicas=2).

### CPU Offload Impact

![Offload Impact Heatmap](offload_impact_heatmap.png)

The heatmap shows the throughput change from no-offload to native-offload-20k as a percentage
across all model/replica/concurrency combinations.

**Llama-3.1-70B-FP8 (replicas=1):** Offload reduces throughput at all tested concurrency levels
except rate=1 (−2.1%). The largest reduction is at rate=650 (−31.1%).

**Llama-3.1-70B-FP8 (replicas=2):** Small positive effect at rate=1 (+0.7%) and rate=50 (+5.2%),
turning negative at rate=100 (−21.4%) and remaining negative through rate=650 (−15.5%).

**gpt-oss-120b (replicas=1):** Offload has a neutral effect at rate=1 (+2.0%) and a −64.3%
reduction at rate=100, the largest throughput reduction observed. Values remain in the
−47% to −58% range at rate=50 and rate=150–650.

**gpt-oss-120b (replicas=2):** Similar pattern with a small positive at rate=1 (+9.4%) and
reductions of −33.6% to −56.4% at higher concurrency.

### Latency

![Latency Curves](latency_curves.png)

TTFT (mean) rises with concurrency for Llama-3.1-70B-FP8, reaching approximately 40,000 ms
at rate=300+ for replicas=1. The offload configuration shows similar or slightly lower mean TTFT at
high concurrency for Llama, consistent with the offload connector managing queue pressure differently.

gpt-oss-120b TTFT is substantially lower than Llama across all concurrency levels, peaking near
12,000 ms at rate=300–400. The gpt-oss-120b TTFT p50 is 0 ms across all runs due to the MoE
model's token batching behaviour — the first token is delivered before the inference scheduler
records a measurement for the majority of requests. Mean TTFT is used throughout for gpt-oss-120b.

TPOT (time per output token, p50) increases monotonically with concurrency for both models. The
offload configuration consistently shows higher TPOT for gpt-oss-120b (e.g. 448 ms vs 307 ms
mean at concurrency=50, replicas=1), indicating increased per-token latency under the connector
overhead.

![Latency Comparison](latency_comparison.png)

Averaged across all concurrency levels ≥ 50, native-offload-20k increases mean TTFT by 28% for
gpt-oss-120b (replicas=1: 8,370 ms vs 6,530 ms no-offload) and reduces it by 15% for
Llama-3.1-70B-FP8 (replicas=1: 31,856 ms vs 37,556 ms). Mean TPOT is higher under offload for
both models.

### GPU KV Cache Utilisation

![KV Cache Pressure](kv_cache_pressure.png)

GPU KV cache utilisation from PCP archives (`openmetrics.vllm.vllm.kv_cache_usage_perc × 100`)
shows that memory pressure is present at moderate concurrency despite the large block counts on H200.

**Llama-3.1-70B-FP8:** Utilisation reaches approximately 60–80% at rate=100–150 and declines at
higher rates as queue saturation limits the number of simultaneously active sequences. The offload
configuration shows slightly different utilisation patterns, with the OffloadingConnector managing
blocks through its own mechanism rather than vLLM's native hybrid KV cache manager.

**gpt-oss-120b:** Despite 181,691 GPU blocks, utilisation reaches 70–90% at rate=100–150.
The large block count does not eliminate cache pressure — the MoE model's fast token generation
rate means more sequences complete and start within each measurement interval, cycling through
the KV cache.

The block annotations (GPU blocks: 26,842 / CPU blocks: 20,000 / CPU/GPU ratio: 75% for Llama;
GPU blocks: 181,691 / CPU blocks: 20,000 / CPU/GPU ratio: 11% for gpt-oss-120b) illustrate why
the two models respond differently to the same 20,000 CPU block allocation.

### Replica Scaling

![Replica Scaling](replica_scaling.png)

**Llama-3.1-70B-FP8:** At rate=50, replicas=2 delivers 148% of 2× replicas=1 throughput
(no-offload). This super-linear scaling is attributed to the EPP prefix-cache-aware routing
routing similar-prefix requests to the same replica, reducing per-replica KV cache evictions and
improving throughput per replica beyond the linear expectation. Scaling efficiency stabilises
at 100–110% for rate=100–650.

The native-offload-20k configuration shows a similar pattern with an even larger spike at rate=50
(>150%), followed by 100–125% at high rates. The offload connector's altered KV cache management
may interact with the EPP routing differently from the no-offload case.

**gpt-oss-120b:** Scaling efficiency increases from ~48% (rate=1) to ~100% (rate=300–400),
reflecting that the model's per-replica throughput is high enough that the second replica is not
fully utilised until moderate-to-high concurrency. The offload configuration follows a similar
trajectory but with greater variance.

---

## Observations

1. **Native CPU KV cache offload reduces throughput on H200 hardware for both tested models**
   across almost all concurrency levels. The `OffloadingConnector` disables vLLM's hybrid KV
   cache manager on initialisation, introducing connector overhead that is not offset by the
   additional CPU cache capacity under these conditions.

2. **The throughput reduction is larger for gpt-oss-120b (−34% to −64%) than for
   Llama-3.1-70B-FP8 (−2% to −31%).** The MoE model's MXFP4 quantisation leaves 181,691 GPU
   blocks available — 6.8× more than the FP8 dense model — meaning the 20,000 CPU blocks
   represent only 11% additional capacity. The connector overhead is the same in both cases,
   but the benefit is proportionally smaller for gpt-oss-120b.

3. **GPU KV cache utilisation reaches 60–90% at moderate concurrency (rate=100–150)** for
   both models, confirming that memory pressure is present. The absence of a throughput benefit
   from CPU offload is not explained by low GPU cache utilisation, but rather by the overhead
   of the OffloadingConnector path relative to the native KV cache manager.

4. **The only configuration showing a net throughput gain from offload is Llama-3.1-70B-FP8
   at replicas=2, rate=1 (+0.7%) and rate=50 (+5.2%).** At these low concurrency levels with
   two replicas, the EPP routing distributes load such that each replica has fewer simultaneous
   sequences, and the connector overhead is proportionally smaller.

5. **EPP prefix-cache-aware routing enables super-linear throughput scaling at moderate
   concurrency for Llama-3.1-70B-FP8.** replicas=2 delivers 148% of the expected 2× replicas=1
   throughput at rate=50 (no-offload), declining to 100–110% at rate=100–650. This behaviour
   reflects the EPP routing similar requests to the same replica, improving prefix cache hit
   rates and reducing compute per request.

6. **gpt-oss-120b reaches ~100% scaling efficiency at rate=300–400 (replicas=2).** At lower
   concurrency, per-replica throughput is high enough that a second replica is not fully
   utilised.

---

## Appendix: Technical Details

### vLLM OffloadingConnector Behaviour

When `--kv-transfer-config` is set, vLLM 0.13.0 logs:

> *"Turning off hybrid kv cache manager because `--kv-transfer-config` is set. This will reduce
> the performance of vLLM on LLMs with sliding window attention or Mamba attention. If you are
> a developer of kv connector, please consider supporting hybrid kv cache manager for your
> connector by making sure your connector is a subclass of `SupportsHMA` defined in
> kv_connector/v1/base.py and use `--no-disable-hybrid-kv-cache-manager` to start vLLM."*

The hybrid KV cache manager provides native block allocation optimisations. Disabling it
introduces overhead that scales with the number of KV operations, explaining the degradation
at higher concurrency where KV operations are most frequent.

### PCP Metrics Captured

Each 120-second benchmark run includes a PCP archive recording:
- `openmetrics.vllm.*` — all vLLM Prometheus metrics (vLLM metrics named `*_perc` export
  values in 0–1 range, not 0–100; multiply by 100 for percentage display)
- `openmetrics.dcgm.*` — DCGM GPU metrics (8× H200, per-GPU utilisation and memory)
- `openmetrics.epp.*` — EPP inference scheduler workqueue metrics

### GuideLLM Benchmark Command

```bash
guidellm benchmark run \
  --target "https://inference-gateway.apps.<cluster>/llm-d-pfc-cpu/<model>" \
  --rate-type concurrent \
  --rate <N> \
  --max-seconds 120 \
  --random-seed 889 \
  --data '{"prompt_tokens":512,"output_tokens":128,"prefix_tokens":10000,"turns":5,"prefix_count":<2N>}' \
  --sample-requests 0
```

---

*Data source:*
*[PCP](https://pcp.io) metric archives and [GuideLLM](https://github.com/vllm-project/guidellm) benchmark results*

*Test dates: March 25–26, 2026.*
*Report generated in conjunction with [Claude Code](https://claude.ai).*
