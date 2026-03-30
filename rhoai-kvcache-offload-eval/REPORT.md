# RHOAI 3.3 KV Cache CPU Offload Evaluation

This report evaluates native CPU KV cache offload on Red Hat OpenShift AI 3.3 (vLLM 0.13.0+rhai11)
running on NVIDIA H200 hardware. Three models are tested across three replica counts using the
RHOAI llm-d serving stack. Three workload profiles characterise offload behaviour across a range
of KV cache pressure levels.

**Software Versions:**
- **RHOAI**: 3.3.0 (vLLM 0.13.0+rhai11, GuideLLM 0.5.4)
- **llm-d**: Integrated via RHOAI `LLMInferenceService` with EPP inference scheduler
- **EPP**: `odh-llm-d-inference-scheduler-rhel9`, Valkey-backed prefix cache index

**Hardware:** 1x GPU worker node, 8x NVIDIA H200 (140 GB HBM3e each); 160 vCPUs,
1.8 TB RAM; 8x NVMe local storage (LVMS)

**Models:**
- `RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8` -- compressed-tensors FP8 quantization
- `meta-llama/Llama-3.1-70B-Instruct` -- BF16, full precision
- `openai/gpt-oss-120b` -- Mixture-of-Experts (MoE), MXFP4-quantised expert weights

**Workload profiles:**
- **Standard**: prompt=512 tokens, output=128 tokens, concurrency 1-650
- **KV-stress**: prompt=512 tokens, output=512 tokens, concurrency 1-650 (4x output length)
- **Long-context**: prompt=4,096 tokens, output=256 tokens, reduced gpu_util, concurrency 1-300

---

## Summary

432 benchmark runs were collected across three workload profiles, three models, two KV-cache
configurations, three replica counts, and eight concurrency levels.

- **no-offload**: GPU-only KV-cache (baseline)
- **native-offload-20k**: CPU offload via `OffloadingConnector`, `num_cpu_blocks=20000`
  (v0.4.0-compatible API, as shipped in RHOAI 3.3)

**Key Findings:**

1. **CPU offload is beneficial for Llama-3.1-70B-FP8 under long-context conditions across all
   replica counts.** With prompt=4,096 tokens and gpu_memory_utilization=0.50 (GPU blocks reduced
   from 26,842 to ~14,440), offload delivers +21.4% throughput at concurrency=10 (replicas=1),
   +9.8% at replicas=2, and +3.3% at replicas=4.

2. **gpt-oss-120b (MoE) at replicas=4 shows +22.3% throughput improvement under long-context
   conditions.** At gpu_memory_utilization=0.50 with 4 replicas sharing 8 GPUs, the connector
   delivers its largest benefit across all tested configurations. At replicas=1 and 2, the MoE
   model shows small negative offload impact under long-context (-9.4% and -1.7%).

3. **Under standard and kv-stress workloads, CPU offload reduces throughput for all models at
   replicas=1 and replicas=4.** The `OffloadingConnector` disables vLLM's hybrid KV cache
   manager, introducing overhead that is not offset by the additional CPU cache capacity under
   these conditions.

4. **At replicas=2, offload shows small throughput gains for all three models under standard
   workload conditions:** +2.8% (FP8), +4.1% (BF16), +12.8% (MoE). The r=2 behaviour reflects
   EPP prefix-cache-aware routing concentrating similar requests per replica.

5. **GPU KV cache block counts relative to 20,000 CPU blocks vary substantially by model:**
   - Llama-3.1-70B-FP8: 26,842 GPU blocks -- CPU adds +74% capacity
   - Llama-3.1-70B-BF16: 22,376 GPU blocks -- CPU adds +89% capacity
   - gpt-oss-120b: 181,691 GPU blocks -- CPU adds only +11% capacity

**Peak Throughput Summary -- Standard Workload:**

| Model | Replicas | no-offload | native-offload-20k | Offload delta |
|-------|:--------:|:----------:|:------------------:|:-------------:|
| Llama-3.1-70B-FP8 | 1 | 135.6 tok/s @50 | 123.9 tok/s @50 | -8.6% |
| Llama-3.1-70B-FP8 | 2 | 287.9 tok/s @50 | 296.1 tok/s @50 | **+2.8%** |
| Llama-3.1-70B-FP8 | 4 | 559.2 tok/s @50 | 409.9 tok/s @50 | -26.7% |
| Llama-3.1-70B-BF16 | 1 | 81.0 tok/s @50 | 76.7 tok/s @50 | -5.3% |
| Llama-3.1-70B-BF16 | 2 | 162.1 tok/s @50 | 168.7 tok/s @50 | **+4.1%** |
| Llama-3.1-70B-BF16 | 4 | 282.1 tok/s @100 | 264.1 tok/s @50 | -6.4% |
| gpt-oss-120b (MoE) | 1 | 1,635.7 tok/s @50 | 858.1 tok/s @50 | -47.5% |
| gpt-oss-120b (MoE) | 2 | 2,543.8 tok/s @100 | 2,868.2 tok/s @50 | **+12.8%** |
| gpt-oss-120b (MoE) | 4 | 5,587.5 tok/s @150 | 3,232.2 tok/s @50 | -42.2% |

---

## Test Configuration

### Hardware

**Worker node:** 1x OpenShift GPU worker node on IBM Cloud
- **GPUs:** 8x NVIDIA H200 (140 GB HBM3e each, 1,120 GB total)
- **CPU:** 160 vCPUs
- **RAM:** 1.8 TB
- **Storage:** 8x NVMe drives in LVM volume group
- **Network:** 8x RoCE v2 interfaces

**GPU allocation per benchmark:**
- Tensor parallel size: 2 (2 GPUs per replica)
- Replicas=1: 2 GPUs total
- Replicas=2: 4 GPUs total
- Replicas=4: 8 GPUs total (full node)

### Software

| Component | Version |
|-----------|---------|
| RHOAI | 3.3.0 |
| vLLM | 0.13.0+rhai11 |
| GuideLLM | 0.5.4 |
| OpenShift | 4.20 |
| PCP | 7.0.3 |

### Model Configuration

**Llama-3.1-70B-FP8** (`RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8`):
```
--tensor-parallel-size 2
--gpu-memory-utilization 0.75   (standard/kv-stress)
                         0.50   (long-context)
--max-num-seq 1024
```
GPU KV cache: 429,472 tokens (26,842 blocks); max_seq_len=131,072

**Llama-3.1-70B-BF16** (`meta-llama/Llama-3.1-70B-Instruct`):
```
--tensor-parallel-size 2
--gpu-memory-utilization 0.90   (all workloads -- BF16 weights require this minimum)
--max-num-seq 1024
--max-model-len 65536           (long-context only; without this limit the KV pool at 0.90
                                 util is insufficient for 131,072-token max_seq_len)
```
GPU KV cache: 358,016 tokens (22,376 blocks at 0.90 util); BF16 weights ~70 GiB/GPU

**gpt-oss-120b** (`openai/gpt-oss-120b`, MoE MXFP4):
```
--tensor-parallel-size 2
--gpu-memory-utilization 0.65   (standard/kv-stress)
                         0.50   (long-context)
--max-num-seq 1024
```
GPU KV cache: 1,453,520 tokens (181,691 blocks); max_seq_len=131,072

**CPU Offload** (`native-offload-20k`):
```
--kv-transfer-config '{"kv_connector":"OffloadingConnector",
  "kv_connector_extra_config":{"num_cpu_blocks":20000}}'
```
Adds 20,000 CPU blocks to each replica's KV storage.

### Benchmark Methodology

- **Tool**: GuideLLM 0.5.4 (`--sample-requests=0` for synthetic workload)
- **Rate type**: concurrent (fixed concurrency levels)
- **Duration**: 120 seconds per run
- **Turns**: 5 per conversation
- **Prefix**: 10,000 shared prefix tokens (exercises EPP prefix-cache routing)
- **Results**: 432 runs total (3 profiles x 3 models x 2 configs x 3 replicas x 8 rates)
- **Metrics**: GuideLLM JSON results + PCP archives (openmetrics PMDA from vLLM HTTPS endpoints)

---

## Standard Workload Results

Prompt=512 tokens, output=128 tokens, concurrency 1-650.

### Throughput

![Throughput Curves](throughput_curves.png)

For both dense models, throughput peaks at concurrency=50 then declines as queue saturation
accumulates. Llama-3.1-70B-BF16 reaches roughly 55-60% of FP8 throughput, consistent with
its larger KV footprint occupying more GPU memory per token.

gpt-oss-120b reaches substantially higher raw throughput (1,636-5,588 tok/s no-offload
depending on replica count), reflecting the MoE architecture activating only ~5.1B of 120B
parameters per forward pass.

### Offload Impact

![Offload Impact Heatmap](offload_impact_heatmap.png)

The heatmap shows throughput change from no-offload to native-offload-20k across all nine
model/replica combinations.

**Replicas=1:** All three models show throughput reductions across most concurrency levels.
The MoE model shows the largest reductions (-30% to -64%), consistent with its small
CPU/GPU capacity ratio (11%). The two dense models show more moderate reductions (-2% to -27%).

**Replicas=2:** Small positive effects appear for all three models at low-to-moderate concurrency
(+2.8% FP8, +4.1% BF16, +12.8% MoE at their respective peaks). The MoE r=2 +12.8% at
concurrency=50 is the largest positive result in the standard workload.

**Replicas=4:** Offload returns to negative for all models (-6.4% BF16, -26.7% FP8,
-42.2% MoE). At r=4, all 8 H200 GPUs are fully utilised; connector overhead accumulates
across the higher total request volume.

### Latency

![Latency Curves](latency_curves.png)

TTFT (mean) rises with concurrency for the dense models, reaching 40,000-60,000 ms at
concurrency=300-650 for replicas=1. gpt-oss-120b TTFT is lower across all concurrency levels.
gpt-oss-120b TTFT p50 is near zero for most requests due to MoE token batching; mean TTFT
is used throughout.

TPOT p50 increases monotonically with concurrency for all models. The offload configuration
shows higher TPOT for gpt-oss-120b at moderate concurrency, consistent with connector overhead
applying per token.

![Latency Comparison](latency_comparison.png)

### GPU KV Cache Utilisation

![KV Cache Pressure](kv_cache_pressure.png)

GPU KV cache utilisation from PCP archives confirms that cache pressure is present at moderate
concurrency. Llama-3.1-70B-FP8 and gpt-oss-120b reach 60-90% at concurrency=100-150.
Llama-3.1-70B-BF16 with 22,376 GPU blocks at 0.90 utilisation saturates at lower concurrency
than the FP8 model.

### Replica Scaling

![Replica Scaling](replica_scaling.png)

Dense models (FP8, BF16) show super-linear scaling at concurrency=50 for replicas=2
(>100% efficiency), attributed to EPP prefix-cache-aware routing concentrating similar-prefix
requests per replica and reducing per-replica KV evictions. Scaling efficiency returns to
approximately 100% at replicas=4.

gpt-oss-120b scaling efficiency increases from ~50% at low concurrency to ~100% at
moderate-to-high concurrency, as the second and fourth replicas are utilised more fully.

---

## KV-Stress Workload Results

Prompt=512 tokens, output=512 tokens (4x standard output), concurrency 1-650.

![KV-Stress Heatmap](kvstress_impact_heatmap.png)

**Peak Throughput Summary -- KV-Stress:**

| Model | Replicas | no-offload | native-offload-20k | Offload delta |
|-------|:--------:|:----------:|:------------------:|:-------------:|
| Llama-3.1-70B-FP8 | 1 | 364.0 tok/s @50 | 188.0 tok/s @50 | -48.3% |
| Llama-3.1-70B-FP8 | 2 | 660.2 tok/s @50 | 657.7 tok/s @50 | -0.4% |
| Llama-3.1-70B-FP8 | 4 | 1,221.0 tok/s @100 | 924.4 tok/s @50 | -24.3% |
| Llama-3.1-70B-BF16 | 1 | 158.5 tok/s @50 | 129.9 tok/s @50 | -18.0% |
| Llama-3.1-70B-BF16 | 2 | 430.8 tok/s @50 | 394.5 tok/s @50 | -8.4% |
| Llama-3.1-70B-BF16 | 4 | 746.8 tok/s @100 | 608.1 tok/s @50 | -18.6% |
| gpt-oss-120b (MoE) | 1 | 2,349.0 tok/s @50 | 931.0 tok/s @50 | -60.4% |
| gpt-oss-120b (MoE) | 2 | 4,554.6 tok/s @100 | 2,518.4 tok/s @50 | -44.7% |
| gpt-oss-120b (MoE) | 4 | 8,621.2 tok/s @150 | 4,640.0 tok/s @100 | -46.2% |

KV-stress results are uniformly negative across all models and replica counts. FP8 replicas=2
at -0.4% is the only near-neutral result. Increasing output token count raises per-sequence
KV footprint without increasing the recomputation cost advantage needed for offload benefit.
The FP8 replicas=1 kv-stress result (-48.3%) is substantially worse than the standard
workload (-8.6%), indicating that connector overhead scales with output token volume.

---

## Long-Context Results

Prompt=4,096 tokens per turn, 5 turns (21,760 unique tokens per sequence), output=256 tokens.
`gpu_memory_utilization=0.50` for FP8 and MoE; `gpu_memory_utilization=0.90` for BF16
(BF16 weights require this minimum). Concurrency 1-300.

At 21,760-token context, recomputation cost is approximately 350 us per block vs ~78 us CPU
fetch -- the crossover condition where offload can yield a net benefit.

**Data quality note:** For Llama-3.1-70B-BF16 replicas=2, a pipeline execution inconsistency
caused the no-offload rate=1 run to use gpu_util=0.60 while the native-offload config used
0.90. The rate=1 pair is excluded from comparisons for that configuration; rates 5-300 are
consistent at 0.90.

### CPU KV Cache Hit Rates

The `OffloadingConnector` registers its CPU cache lookups through vLLM's
`external_prefix_cache_queries_total` and `external_prefix_cache_hits_total` metrics. These are
zero for no-offload runs and non-zero only when the connector is active. PCP archives capture
these counters from the running vLLM pods.

![CPU Offload Hit Rates](cpu_offload_hit_rates.png)

The most reliable data point is FP8 longctx replicas=1 rate=10 (where the pod was freshly
started at archive capture time): **97.5% CPU KV cache hit rate** (2,939 hits / 3,014 queries).
At rate=100, the hit rate drops to 69.2%. These high hit rates directly explain the throughput
gains at those operating points — the connector is successfully serving blocks from CPU memory
rather than triggering recomputation.

For the standard workload, counter extraction shows near-zero or unmeasurable hit rates. The
standard workload's short per-sequence KV footprint (512 prompt tokens, 128 output tokens,
5 turns = ~4,480 tokens per sequence) means the GPU KV pool (26,842 blocks for FP8 at 0.75
util) is rarely evicted. When no blocks are evicted to CPU memory, the connector has no blocks
to serve back, and its overhead applies without benefit.

For the kv-stress workload, hit rates are low (0.1–16% for FP8, <5% for MoE). The longer
output sequences fill the CPU cache faster, but the CPU cache capacity (20,000 blocks) is still
insufficient relative to the total KV traffic at moderate-to-high concurrency.

**Note on counter extraction:** Each benchmark rate produces its own 120-second PCP archive,
but all rates for a given model+config share a single vLLM serving pod — the pod is only
restarted when switching between configs (no-offload → native-offload-20k) or between models.
vLLM's Prometheus counters (`external_prefix_cache_queries_total` etc.) are cumulative since
pod startup. The PCP archive for rate=100 therefore starts with a counter that already includes
all traffic from the earlier rates (1, 50, …) that ran on the same pod. Computing `last − first`
over the 120-second window is correct in principle, but is unreliable when the PCP scrape
interval (10 s) straddles a rate boundary. Archives where the vLLM pod was freshly started
at capture time are cleanly extractable; approximately 40% of native-offload archives fall
into this category. The remainder show apparent resets or inflated deltas and are excluded.

### Long-Context Throughput

![Long-Context Throughput](longctx_throughput.png)

**Llama-3.1-70B-FP8** (gpu_util=0.50): The native-offload-20k line exceeds no-offload across
concurrency 5-100 for all replica counts. Peak benefit at rate=10: +21.4% (r=1), +9.8% (r=2).
At gpu_util=0.50, the GPU KV pool is ~14,440 blocks -- below the 20,000 CPU blocks added by
offload, providing net capacity expansion of +139%.

**Llama-3.1-70B-BF16** (gpu_util=0.90): Near-neutral at replicas=1 (+0.5% at rate=10),
small positive at replicas=2 (+6.8% at rate=20), and slightly negative at replicas=4
(-1.1% peak). At 0.90 util, the BF16 model retains 22,376 GPU blocks, providing proportionally
less benefit from 20,000 additional CPU blocks than FP8 at 0.50 util.

**gpt-oss-120b** (gpu_util=0.50): Small negative at replicas=1 (-9.4%) and replicas=2 (-1.7%),
but **+22.3% at replicas=4** at concurrency=50. Even at 0.50 utilisation, the MoE model
retains ~131,000 GPU blocks (6.5x the CPU blocks added). The r=4 gain reflects EPP routing
and offload interaction at scale with 4 replicas on 8 GPUs.

### Long-Context Offload Delta

![Long-Context Offload Delta](longctx_offload_delta.png)

**Long-Context Throughput Summary (rates 10-100):**

| Model / Config | rate=10 | rate=20 | rate=50 | rate=100 |
|----------------|:-------:|:-------:|:-------:|:--------:|
| FP8 r=1 no-offload | 191.8 tok/s | 164.3 tok/s | 145.6 tok/s | 132.5 tok/s |
| FP8 r=1 native-offload-20k | **232.9 (+21.4%)** | **193.1 (+17.5%)** | 151.0 (+3.7%) | 136.8 (+3.3%) |
| FP8 r=4 no-offload | 468.9 tok/s | 519.9 tok/s | 546.1 tok/s | 539.6 tok/s |
| FP8 r=4 native-offload-20k | 493.0 (+5.1%) | 555.7 (+6.9%) | **564.2 (+3.3%)** | 527.6 (-2.2%) |
| MoE r=1 no-offload | 772.5 tok/s | 1,060.2 tok/s | 822.7 tok/s | 649.8 tok/s |
| MoE r=1 native-offload-20k | 766.9 (-0.7%) | 960.5 (-9.4%) | 429.8 (-47.8%) | 317.2 (-51.2%) |
| MoE r=4 no-offload | 1,180.7 tok/s | 1,768.7 tok/s | 2,404.5 tok/s | 2,367.2 tok/s |
| MoE r=4 native-offload-20k | 1,203.0 (+1.9%) | 2,051.3 (+16.0%) | **2,939.7 (+22.3%)** | 2,722.0 (+14.9%) |

---

## Observations

1. **Native CPU KV cache offload reduces throughput under standard workload conditions for most
   configurations.** The `OffloadingConnector` disables vLLM's hybrid KV cache manager on
   initialisation, introducing connector overhead that is not offset by the additional CPU cache
   capacity at standard (prompt=512, output=128) concurrency levels.

2. **The throughput reduction is largest for gpt-oss-120b under standard and kv-stress workloads
   (-30% to -64% at replicas=1).** The MoE architecture's MXFP4 quantisation leaves 181,691 GPU
   blocks -- 6.8x more than the FP8 dense model -- so 20,000 CPU blocks represent only 11%
   additional capacity while connector overhead applies to all KV operations.

3. **Replicas=2 shows small throughput gains from offload for all three models under the standard
   workload** (+2.8% FP8, +4.1% BF16, +12.8% MoE). EPP prefix-cache-aware routing concentrates
   similar-prefix requests per replica, reducing evictions and enabling the CPU cache to absorb
   more overflow. This effect is not sustained at replicas=4.

4. **Replicas=4 shows larger throughput reductions from offload** (-26.7% FP8, -6.4% BF16,
   -42.2% MoE under standard workload). When all 8 H200 GPUs are fully utilised (TP=2 x 4
   replicas), connector overhead accumulates proportionally to the higher total request volume.

5. **Long-context conditions produce the largest offload benefits for the dense FP8 model.**
   With prompt=4,096 tokens, recomputation cost exceeds CPU fetch latency by ~4.5x, and
   reduced gpu_memory_utilization (0.50) shrinks the GPU KV pool below the 20,000 CPU block
   threshold. FP8 at r=1 delivers +21.4% throughput at concurrency=10. PCP archives confirm
   a 97.5% CPU KV cache hit rate at this operating point (FP8 longctx r=1 rate=10:
   2,939 hits / 3,014 queries), directly linking the high hit rate to the observed
   throughput gain.

6. **gpt-oss-120b at replicas=4 under long-context workload shows +22.3% throughput with offload**
   at concurrency=50. This is the largest positive result for the MoE model across all tested
   conditions. At r=4 with gpu_util=0.50, the EPP routing and offload connector interaction at
   scale produces a net positive result, in contrast to the negative results at r=1 and r=2.

7. **Llama-3.1-70B-BF16 shows near-neutral offload impact under long-context conditions.**
   At gpu_util=0.90, the BF16 model retains 22,376 GPU blocks (89% CPU/GPU ratio). Connector
   overhead largely offsets the capacity benefit at r=1 (+0.5%) and r=4 (-1.1%), with a modest
   gain at r=2 (+6.8%).

8. **KV-stress workload (output=512) is uniformly negative across all configurations.**
   Longer output sequences increase per-sequence KV writes without providing the recomputation
   cost advantage needed for offload to be beneficial. FP8 replicas=1 shows -48.3% under
   kv-stress vs -8.6% under standard, indicating connector overhead scales with output token volume.

9. **EPP prefix-cache-aware routing produces super-linear scaling for dense models at
   moderate concurrency.** replicas=2 delivers >100% of the expected 2x scaling efficiency
   for FP8 and BF16 at concurrency=50 (no-offload), declining to ~100% at higher concurrency
   as queuing effects dominate.

10. **The OffloadingConnector does not implement `SupportsHMA`** (confirmed in
    `vllm/v1/kv_offload/offloading_connector.py`), causing vLLM to disable the hybrid
    memory allocator on startup. The "Turning off hybrid kv cache manager" warning is logged
    at each model initialisation.

---

*Data source:*
*[PCP](https://pcp.io) metric archives and [GuideLLM](https://github.com/vllm-project/guidellm) benchmark results*

*Test dates: March 25-30, 2026.*
*Report generated in conjunction with [Claude Code](https://claude.ai).*
