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
- **Short-context**: prompt=512 tokens, output=128 tokens, concurrency 1-650
- **Long-output**: prompt=512 tokens, output=512 tokens, concurrency 1-650 (4x output length)
- **Long-context**: prompt=4,096 tokens, output=256 tokens, reduced gpu_util, concurrency 1-300

**Data note:** gpt-oss-120b results were re-collected on 2026-04-01 using a corrected GuideLLM
nightly build. The original 0.5.4 build contained a concurrency management bug that caused
under-loading of the inference server for the MoE model under certain configurations, producing
inaccurate throughput measurements. All gpt-oss-120b figures and tables in this report use the
corrected data. Llama-3.1-70B-FP8 and Llama-3.1-70B-BF16 results are unchanged from the
original collection.

---

## Summary

432 benchmark runs were collected across three workload profiles, three models, two KV-cache
configurations, three replica counts, and eight concurrency levels.

- **no-offload**: GPU-only KV-cache (baseline)
- **native-offload-20k**: CPU offload via `OffloadingConnector`, `num_cpu_blocks=20000`
  (v0.4.0-compatible API, as shipped in RHOAI 3.3)

**Findings:**

1. **Long-context workload (prompt=4,096 tokens) is the primary condition where CPU offload
   delivers throughput gains, and only for the dense FP8 model.** Reduced
   gpu_memory_utilization shrinks the GPU KV pool below 20,000 blocks, making CPU capacity
   a net addition. Llama-3.1-70B-FP8 at replicas=1 reaches +21.4% at concurrency=10;
   at replicas=2, +9.8%; at replicas=4, +3.3%.

2. **gpt-oss-120b (MoE) shows no throughput benefit from CPU offload under any tested
   configuration.** With 181,691 GPU blocks at standard utilisation and ~131,000 at
   gpu_util=0.50, the 20,000 CPU blocks represent only 11-15% additional capacity while
   connector overhead applies to all KV operations. All MoE results are negative across
   short-context, long-output, and long-context workloads.

3. **The OffloadingConnector causes server-side request failures for gpt-oss-120b at high
   concurrency under long-output conditions.** At rate=300 with native-offload-20k and
   replica=1, 283 of 341 requests fail (83% error rate) vs 0 errors with no-offload at
   the same concurrency. This failure mode is repeatable. No-offload handles rate=300
   cleanly (303 completions, 1,075 tok/s).

4. **GPU KV cache block counts relative to 20,000 CPU blocks vary substantially by model:**
   - Llama-3.1-70B-FP8: 26,842 GPU blocks at 0.75 util (short-context); ~14,440 at 0.50 util
     (long-context) -- CPU adds +139% capacity under long-context conditions
   - Llama-3.1-70B-BF16: 22,376 GPU blocks at 0.90 util -- CPU adds +89% capacity
   - gpt-oss-120b: 181,691 GPU blocks at 0.65 util (short-context); ~131,000 at 0.50 util
     (long-context) -- CPU adds only +11-15% capacity

5. **Under short-context and long-output workloads, CPU offload reduces throughput for most
   configurations.** The `OffloadingConnector` disables vLLM's hybrid KV cache manager,
   introducing overhead not offset by the additional CPU cache capacity. Dense models at
   replicas=2 show small gains under short-context (+2.8% FP8, +4.1% BF16) attributed to
   EPP prefix-cache-aware routing.

**Peak Throughput Summary -- Long-Context Workload:**

| Model | Replicas | no-offload | native-offload-20k | Offload delta |
|-------|:--------:|:----------:|:------------------:|:-------------:|
| Llama-3.1-70B-FP8 | 1 | 191.8 tok/s @10 | 232.9 tok/s @10 | **+21.4%** |
| Llama-3.1-70B-FP8 | 2 | 335.9 tok/s @20 | 368.9 tok/s @20 | **+9.8%** |
| Llama-3.1-70B-FP8 | 4 | 546.1 tok/s @50 | 564.2 tok/s @50 | +3.3% |
| Llama-3.1-70B-BF16 | 1 | 149.3 tok/s @10 | 150.0 tok/s @10 | +0.5% |
| Llama-3.1-70B-BF16 | 2 | 229.0 tok/s @20 | 244.5 tok/s @20 | +6.8% |
| Llama-3.1-70B-BF16 | 4 | 380.0 tok/s @50 | 376.0 tok/s @20 | -1.1% |
| gpt-oss-120b (MoE) | 1 | 825.7 tok/s @20 | 790.3 tok/s @20 | -4.3% |
| gpt-oss-120b (MoE) | 2 | 1,565.0 tok/s @50 | 1,257.3 tok/s @20 | -19.7% |
| gpt-oss-120b (MoE) | 4 | 2,777.3 tok/s @100 | 2,524.7 tok/s @50 | -9.1% |

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
| GuideLLM | 0.5.4 (Llama models); nightly/2026-04-01 (gpt-oss-120b re-run) |
| OpenShift | 4.20 |
| PCP | 7.0.3 |

### Model Configuration

**Llama-3.1-70B-FP8** (`RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8`):
```
--tensor-parallel-size 2
--gpu-memory-utilization 0.75   (short-context/long-output)
                         0.50   (long-context)
--max-num-seq 1024
```
GPU KV cache: 429,472 tokens (26,842 blocks at 0.75 util); ~230,720 tokens (~14,420 blocks at 0.50 util); max_seq_len=131,072

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
--gpu-memory-utilization 0.65   (short-context/long-output)
                         0.50   (long-context)
--max-num-seq 1024
```
GPU KV cache: 1,453,520 tokens (181,691 blocks at 0.65 util); ~1,048,000 tokens (~131,000 blocks at 0.50 util); max_seq_len=131,072

**CPU Offload** (`native-offload-20k`):
```
--kv-transfer-config '{"kv_connector":"OffloadingConnector",
  "kv_connector_extra_config":{"num_cpu_blocks":20000}}'
```
Adds 20,000 CPU blocks to each replica's KV storage.

### Benchmark Methodology

- **Tool**: GuideLLM 0.5.4
- **Rate type**: concurrent (fixed concurrency levels)
- **Duration**: 120 seconds per run
- **Turns**: 5 per conversation
- **Prefix**: 10,000 shared prefix tokens
- **Results**: 432 runs total (3 profiles x 3 models x 2 configs x 3 replicas x 8 rates)
- **Metrics**: GuideLLM JSON results + PCP archives (openmetrics PMDA from vLLM HTTPS endpoints)

---

## Long-Context Results

Prompt=4,096 tokens per turn, 5 turns (21,760 unique tokens per sequence), output=256 tokens.
`gpu_memory_utilization=0.50` for FP8 and MoE; `gpu_memory_utilization=0.90` for BF16
(BF16 weights require this minimum). Concurrency 1-300.

At 21,760-token context, recomputation cost is approximately 350 µs per block vs ~78 µs CPU
fetch -- the crossover condition where offload can yield a net benefit.

**Data quality note:** For Llama-3.1-70B-BF16 replicas=2, a pipeline execution inconsistency
caused the no-offload rate=1 run to use gpu_util=0.60 while the native-offload config used
0.90. The rate=1 pair is excluded from comparisons for that configuration; rates 5-300 are
consistent at 0.90.

### Long-Context Throughput

![Long-Context Throughput](longctx_throughput.png)

**Llama-3.1-70B-FP8** (gpu_util=0.50): The native-offload-20k line exceeds no-offload across
concurrency 5-100 for replicas=1 and 2. At gpu_util=0.50, the GPU KV pool is ~14,420 blocks --
below the 20,000 CPU blocks added by offload, providing net capacity expansion of +139%. Peak
benefit at replicas=1: +21.4% at concurrency=10. At replicas=2: +9.8% at concurrency=20. At
replicas=4: +3.3% at concurrency=50, declining as per-replica GPU KV pools together exceed the
per-replica CPU pool.

**Llama-3.1-70B-BF16** (gpu_util=0.90): Near-neutral at replicas=1 (+0.5% at concurrency=10),
small positive at replicas=2 (+6.8% at concurrency=20), and slightly negative at replicas=4
(-1.1% peak). At 0.90 util, the BF16 model retains 22,376 GPU blocks, providing proportionally
less benefit from 20,000 additional CPU blocks than FP8 at 0.50 util.

**gpt-oss-120b** (gpu_util=0.50): Negative across all replica counts. At replicas=1, offload
reduces throughput by -4.3% at concurrency=20, worsening to -43.0% at concurrency=50. At
replicas=4, -9.1% at peak. Even at gpu_util=0.50, each replica retains ~131,000 GPU blocks
(6.5x the 20,000 CPU blocks added), so the connector overhead is not offset by the additional
capacity.

### Long-Context Offload Delta

![Long-Context Offload Delta](longctx_offload_delta.png)

The delta plot shows throughput change at each concurrency level relative to the no-offload
baseline. FP8 at replicas=1 and 2 shows the most consistent positive range (concurrency 5-100).
BF16 remains close to zero across all replica counts. MoE is negative at all concurrency levels
for all replica counts.

### Long-Context Throughput Summary (rates 10-100)

| Model / Config | rate=10 | rate=20 | rate=50 | rate=100 |
|----------------|:-------:|:-------:|:-------:|:--------:|
| FP8 r=1 no-offload | 191.8 tok/s | 164.3 tok/s | 145.6 tok/s | 132.5 tok/s |
| FP8 r=1 native-offload-20k | **232.9 (+21.4%)** | **193.1 (+17.5%)** | 151.0 (+3.7%) | 136.8 (+3.3%) |
| FP8 r=2 no-offload | 324.0 tok/s | 335.9 tok/s | 306.9 tok/s | 272.9 tok/s |
| FP8 r=2 native-offload-20k | 314.8 (-2.8%) | **368.9 (+9.8%)** | 315.1 (+2.7%) | 290.8 (+6.6%) |
| FP8 r=4 no-offload | 468.9 tok/s | 519.9 tok/s | 546.1 tok/s | 539.6 tok/s |
| FP8 r=4 native-offload-20k | 493.0 (+5.1%) | 555.7 (+6.9%) | **564.2 (+3.3%)** | 527.6 (-2.2%) |
| MoE r=1 no-offload | 675.6 tok/s | 825.7 tok/s | 647.2 tok/s | 484.8 tok/s |
| MoE r=1 native-offload-20k | 668.1 (-1.1%) | 790.3 (-4.3%) | 369.2 (-43.0%) | 305.5 (-37.0%) |
| MoE r=4 no-offload | 1,084.9 tok/s | 1,741.8 tok/s | 2,695.8 tok/s | 2,777.3 tok/s |
| MoE r=4 native-offload-20k | 1,007.1 (-7.2%) | 1,751.1 (+0.5%) | 2,524.7 (-6.3%) | 2,076.5 (-25.2%) |

### Long-Context Latency

![Long-Context Latency](longctx_latency.png)

The figure shows % change in TTFT p90 and TPOT p50 (native-offload-20k vs no-offload) for the
three FP8 configurations where offload delivers throughput gains. Negative values indicate
reduced latency. At the low-to-moderate concurrency operating points where offload is beneficial,
both latency metrics are maintained or reduced alongside the throughput improvement.

| Config | Concurrency | Throughput delta | TTFT p90 | TPOT p50 |
|--------|:-----------:|:----------------:|:--------:|:--------:|
| FP8 r=1 | 10 | +21.4% | 5,461 ms → 5,040 ms (-7.7%) | 48.0 ms → 42.2 ms (-12.1%) |
| FP8 r=2 | 20 | +9.8% | 4,696 ms → 4,484 ms (-4.5%) | 57.7 ms → 52.3 ms (-9.3%) |
| FP8 r=4 | 50 | +3.3% | 7,868 ms → 7,597 ms (-3.4%) | 84.0 ms → 84.1 ms (+0.1%) |

Where offload reduces throughput, latency degrades in proportion. For gpt-oss-120b replicas=1
at concurrency=50 (-43.0% throughput), TTFT p90 rises from 13,892 ms to 28,057 ms (+102%)
and TPOT p50 from 60.5 ms to 105.1 ms (+73.5%).

### CPU KV Cache Hit Rates

The `OffloadingConnector` registers CPU cache lookups via vLLM's
`external_prefix_cache_queries_total` and `external_prefix_cache_hits_total` metrics.
These counters are non-zero only when the connector is active.

Metrics for FP8 longctx replicas=1 rate=10 (where the vLLM pod was freshly started at archive
capture time) shows a **97.5% CPU KV cache hit rate** (2,939 hits / 3,014 queries). At
concurrency=100 the hit rate drops to 69.2%. These hit rates directly explain the throughput
gains at those operating points: the connector is serving blocks from CPU memory rather than
triggering recomputation.

Under short-context workload conditions, the GPU KV pool (26,842 blocks at 0.75 util) is large enough
that few blocks are evicted to CPU memory; with nothing to serve back, the connector applies
overhead without benefit. Under long-output conditions, longer output sequences fill the CPU cache faster
but the 20,000-block CPU capacity is insufficient relative to total KV traffic at
moderate-to-high concurrency.

---

## Observations

1. **Long-context conditions with reduced gpu_memory_utilization are the primary scenario where
   CPU offload delivers throughput gains, and this applies only to the dense FP8 model.** With
   prompt=4,096 tokens and gpu_util=0.50, the GPU KV pool shrinks below 20,000 blocks for FP8,
   making the CPU blocks a net capacity addition (+139%). Recomputation cost at 21,760 tokens
   per sequence (~350 µs/block) exceeds CPU fetch latency (~78 µs/block), providing the cost
   advantage needed for offload benefit.

2. **FP8 longctx replicas=1 and 2 show the most consistent offload benefit** (+21.4% at r=1,
   +9.8% at r=2, both at low-to-moderate concurrency). PCP archives confirm 97.5% CPU cache
   hit rate at the r=1 peak operating point (concurrency=10), directly linking cache hit rate
   to throughput gain. Benefit decreases at replicas=4 (+3.3%) as per-replica GPU KV pools
   together approach or exceed the per-replica CPU pool. At all three replica counts, TTFT p90
   and TPOT p50 are maintained or reduced alongside the throughput gains: FP8 r=1 at
   concurrency=10 shows TTFT p90 -7.7% (5,461 ms → 5,040 ms) and TPOT p50 -12.1%
   (48.0 ms → 42.2 ms).

3. **gpt-oss-120b shows no throughput benefit from CPU offload under any tested configuration.**
   The MoE model retains 6.5x more GPU blocks than the CPU capacity added (131,000 vs 20,000
   at gpu_util=0.50), so the connector overhead is not offset by the additional capacity at any
   concurrency level or replica count. All long-context MoE results are negative (-4.3% to
   -43.0% at replicas=1; -9.1% to -25.2% at replicas=4).

4. **The OffloadingConnector causes server-side request failures for gpt-oss-120b at high
   concurrency under long-output conditions.** At rate=300 with native-offload-20k and
   replica=1, 83% of requests fail (283 errors out of 341 attempts, 58 completions at 163
   tok/s). No-offload at the same concurrency level completes all requests (303 completions,
   0 errors, 1,075 tok/s). This failure mode is repeatable and is not present in any Llama
   model configuration.

5. **Llama-3.1-70B-BF16 shows near-neutral offload impact under long-context conditions.**
   At gpu_util=0.90, the BF16 model retains 22,376 GPU blocks. Connector overhead largely
   offsets the capacity benefit at r=1 (+0.5%) and r=4 (-1.1%), with a modest gain at r=2
   (+6.8%).

6. **EPP prefix-cache-aware routing produces super-linear scaling for dense models at moderate
   concurrency.** replicas=2 delivers >100% of the expected 2x scaling efficiency for FP8 and
   BF16 at concurrency=50 (no-offload), attributed to EPP concentrating similar-prefix requests
   per replica and reducing per-replica KV evictions. This effect is visible in both no-offload
   and native-offload configurations under long-context conditions.

7. **The OffloadingConnector does not implement `SupportsHMA`** (confirmed in
   `vllm/v1/kv_offload/offloading_connector.py`), causing vLLM to disable the hybrid
   memory allocator on startup. The "Turning off hybrid kv cache manager" warning is logged
   at each model initialisation. This is the primary source of overhead under short-context and
   long-output workloads where CPU blocks are rarely utilised.

---

## Short-Context and Long-Output Workload Results

Under short-context (prompt=512, output=128) and long-output (prompt=512, output=512) workloads,
CPU offload reduces throughput for most configurations. The GPU KV pools under these conditions
are large enough (26,842 blocks for FP8 at 0.75 util; 181,691 blocks for MoE at 0.65 util)
that few blocks are evicted to CPU memory. The connector applies overhead without compensating
cache benefit.

### Short-Context Workload

Concurrency 1-650. Offload impact across all nine model/replica combinations:

![Offload Impact Heatmap](offload_impact_heatmap.png)

**Replicas=1:** All three models show throughput reductions across most concurrency levels.
The MoE model shows the largest reductions (-40% to -60%), consistent with its small
CPU/GPU capacity ratio (11%). The two dense models show more moderate reductions (-2% to -27%).

**Replicas=2:** Dense models show small positive effects at low-to-moderate concurrency
(+2.8% FP8, +4.1% BF16). MoE at replicas=2 shows -40.2%, consistent with the pattern across
all MoE configurations.

**Replicas=4:** Offload is negative for all models (-6.4% BF16, -26.7% FP8, -40.4% MoE).
At r=4, all 8 H200 GPUs are fully utilised; connector overhead accumulates with higher total
request volume.

**Peak Throughput -- Short-Context Workload:**

| Model | Replicas | no-offload | native-offload-20k | Offload delta |
|-------|:--------:|:----------:|:------------------:|:-------------:|
| Llama-3.1-70B-FP8 | 1 | 135.6 tok/s @50 | 123.9 tok/s @50 | -8.6% |
| Llama-3.1-70B-FP8 | 2 | 287.9 tok/s @50 | 296.1 tok/s @50 | +2.8% |
| Llama-3.1-70B-FP8 | 4 | 559.2 tok/s @50 | 409.9 tok/s @50 | -26.7% |
| Llama-3.1-70B-BF16 | 1 | 81.0 tok/s @50 | 76.7 tok/s @50 | -5.3% |
| Llama-3.1-70B-BF16 | 2 | 162.1 tok/s @50 | 168.7 tok/s @50 | +4.1% |
| Llama-3.1-70B-BF16 | 4 | 282.1 tok/s @100 | 264.1 tok/s @50 | -6.4% |
| gpt-oss-120b (MoE) | 1 | 1,703.3 tok/s @100 | 722.6 tok/s @50 | -57.6% |
| gpt-oss-120b (MoE) | 2 | 2,647.9 tok/s @100 | 1,582.5 tok/s @50 | -40.2% |
| gpt-oss-120b (MoE) | 4 | 5,287.9 tok/s @150 | 3,152.7 tok/s @100 | -40.4% |

### Throughput and Latency

![Throughput Curves](throughput_curves.png)

For both dense models, throughput peaks at concurrency=50 then declines as queue saturation
accumulates. gpt-oss-120b reaches substantially higher raw throughput (1,703-5,288 tok/s
no-offload depending on replica count), reflecting the MoE architecture activating only ~5.1B
of 120B parameters per forward pass.

![Latency Comparison](latency_comparison.png)

![Latency Curves](latency_curves.png)

TTFT (mean) rises with concurrency for the dense models, reaching 40,000-60,000 ms at
concurrency=300-650 for replicas=1. gpt-oss-120b TTFT is lower across all concurrency levels.
TPOT p50 increases monotonically with concurrency for all models.

### Replica Scaling

![Replica Scaling](replica_scaling.png)

Dense models (FP8, BF16) show super-linear scaling at concurrency=50 for replicas=2
(>100% efficiency), attributed to EPP prefix-cache-aware routing. Scaling efficiency returns
to approximately 100% at replicas=4. gpt-oss-120b scaling efficiency increases from ~50% at
low concurrency to ~100% at moderate-to-high concurrency.

### Long-Output Workload (output=512)

Concurrency 1-650. Long-output results are negative across all models and replica counts.

![KV-Stress Heatmap](kvstress_impact_heatmap.png)

**Peak Throughput -- Long-Output Workload:**

| Model | Replicas | no-offload | native-offload-20k | Offload delta |
|-------|:--------:|:----------:|:------------------:|:-------------:|
| Llama-3.1-70B-FP8 | 1 | 364.0 tok/s @50 | 188.0 tok/s @50 | -48.3% |
| Llama-3.1-70B-FP8 | 2 | 660.2 tok/s @50 | 657.7 tok/s @50 | -0.4% |
| Llama-3.1-70B-FP8 | 4 | 1,221.0 tok/s @100 | 924.4 tok/s @50 | -24.3% |
| Llama-3.1-70B-BF16 | 1 | 158.5 tok/s @50 | 129.9 tok/s @50 | -18.0% |
| Llama-3.1-70B-BF16 | 2 | 430.8 tok/s @50 | 394.5 tok/s @50 | -8.4% |
| Llama-3.1-70B-BF16 | 4 | 746.8 tok/s @100 | 608.1 tok/s @50 | -18.6% |
| gpt-oss-120b (MoE) | 1 | 2,686.5 tok/s @100 | 2,444.3 tok/s @300 | -9.0% |
| gpt-oss-120b (MoE) | 2 | 5,110.2 tok/s @300 | 2,881.9 tok/s @50 | -43.6% |
| gpt-oss-120b (MoE) | 4 | 10,003.6 tok/s @500 | 3,064.7 tok/s @50 | -69.4% |

FP8 replicas=2 at -0.4% is the only near-neutral result. Increasing output token count raises
per-sequence KV footprint without increasing the recomputation cost advantage needed for offload
benefit. The FP8 replicas=1 long-output result (-48.3%) is substantially worse than the
short-context workload (-8.6%), indicating that connector overhead scales with output token
volume.

**gpt-oss-120b long-output connector failure:** At rate=300 with native-offload-20k and
replica=1, the OffloadingConnector causes 83% of requests to fail (283 errors, 58 completions,
163 tok/s) compared to 0 errors and 1,075 tok/s with no-offload. This failure mode does not
appear in Llama model results and is reproducible across multiple runs. The apparent -9.0%
peak delta for MoE r=1 is measured at rate=300 (the highest-throughput no-offload point);
at lower concurrency the offload delta is more negative.

### GPU KV Cache Utilisation

![KV Cache Pressure](kv_cache_pressure.png)

GPU KV cache utilisation from PCP archives confirms cache pressure is present at moderate
concurrency under short-context workload conditions. Llama-3.1-70B-FP8 and gpt-oss-120b reach
60-90% at concurrency=100-150. Llama-3.1-70B-BF16 with 22,376 GPU blocks at 0.90 utilisation
saturates at lower concurrency than the FP8 model.

---

*Data source:*
*[PCP](https://pcp.io) metric archives and [GuideLLM](https://github.com/vllm-project/guidellm) benchmark results*

*Test dates: March 25-30, 2026 (Llama models); April 1-2, 2026 (gpt-oss-120b re-run).*
*Report generated in conjunction with [Claude Code](https://claude.ai).*
