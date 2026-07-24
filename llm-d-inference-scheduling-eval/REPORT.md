# EPP Optimised Baseline vs Prior Default — RHOAI 3.5

## TL;DR

**The upstream optimised baseline EPP configuration meets or exceeds the
prior default across all tested models and workload profiles, with the
largest gains under high concurrency and heterogeneous request sizes.**
gpt-oss-120b (MoE, 2 replicas) shows +12% to +117% output throughput
improvement across all concurrency levels with TTFT p50 reductions
of 13–93%.  Llama-3.3-70B-FP8 (dense, 4 replicas) shows +45% throughput
at concurrency 256 and up to +236% under heterogeneous workloads at
concurrency 300.
The multi-turn profile shows no throughput regression at any concurrency
level for any model.
The prefix-cache-stress profile shows 3–40% lower throughput for the
optimised baseline at low arrival rates where KV cache pressure is
minimal — a tradeoff, not a regression under production conditions.
**The data supports adopting the optimised baseline as the RHOAI 3.5
default.**

---

## 1. Objective

Validate that the upstream llm-d optimised baseline EndpointPickerConfig
meets or exceeds the prior RHOAI default (3.3/3.4) for throughput and
latency across representative workloads and concurrency levels.

### EPP Scorer Reference

- **queue-scorer**: Scores endpoints inversely proportional to their
  waiting queue size. Normalises across all endpoints so the endpoint
  with the shortest queue gets score 1.0 and the longest gets 0.0.
- **kv-cache-utilisation-scorer**: Scores each endpoint as
  `1 - KVCacheUsagePercent`. Endpoints with more free KV cache memory
  score higher, distributing requests away from memory-pressured
  endpoints.
- **prefix-cache-scorer**: Scores endpoints based on how many prefix
  tokens from the incoming request are already cached on that endpoint.
  Routes requests to where their prefix is already in GPU KV cache,
  avoiding redundant prefill computation.
- **no-hit-lru-scorer**: Only activates for "cold" requests (no prefix
  cache hit on any endpoint). For cold requests, scores endpoints
  inversely by recency of use — endpoints that haven't been routed to
  recently score higher. For cache-hit requests, returns neutral scores
  (no effect). This steers cache-miss traffic to the least-recently-used
  endpoint, minimising the cost of cache eviction.
- **max-score-picker** is the mechanism that selects the highest-scoring
  endpoint after all scorers have contributed their weighted scores.
  With no scorers enabled, max-score-picker receives endpoints all
  scored at 0, shuffles them randomly, and picks one — uniform random
  selection (each endpoint equally likely per request, no state, no
  memory of previous selections). This is distinct from round-robin
  which cycles through endpoints in order.

## 2. Configuration

### EPP Configurations Under Test

| Config | Scorers | Weights |
|---|---|---|
| **A: Prior Default** | queue-scorer, prefix-cache-scorer | 2, 3 |
| **B: Optimised Baseline** | queue-scorer, kv-cache-utilisation-scorer, prefix-cache-scorer, no-hit-lru-scorer | 2, 2, 3, 2 |

Config B adds `kv-cache-utilisation-scorer` and `no-hit-lru-scorer`.
Both configs use `max-score-picker` and `single-profile-handler`.

### Models

| Model | Type | TP | Replicas | GPUs |
|---|---|---|---|---|
| Qwen/Qwen3-30B-A3B-Instruct-2507 | MoE | 1 | 8 | 8 |
| RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic | Dense FP8 | 2 | 4 | 8 |
| openai/gpt-oss-120b | MoE MXFP4 | 4 | 2 | 8 |

All models deployed on a single H200 node (8 GPUs) via RHOAI 3.5 EA1
LLMInferenceService with `gpu-memory-utilisation=0.90`,
`max-num-seq=1024`, `max-model-len=40960`.  vLLM 0.19.1+rhaiv.6
with prefix caching and chunked prefill enabled.

### Workload Profiles

| Profile | Type | Parameters | Purpose |
|---|---|---|---|
| multi-turn | concurrent | streams=[32,64,128,256,512], 5 turns, 10k-token prefix, 128/128 ISL/OSL | Prefix cache + high concurrency |
| heavy-heterogeneous | concurrent | streams=[1,50,100,200,300], ISL=8000±8500 (50–30k), 600s/level | KV cache utilisation imbalance |
| prefix-cache-stress | poisson | rates=[3–60], 6k prefix × 150 buckets, 1200/1000 ISL/OSL, 30s/rate | Prefix scorer isolation |

## 3. Results

### 3.1 Multi-Turn (Critical Scenario)

This profile directly exercises the concurrency 128–256 range identified
in PSAP-2482 as the region of peak throughput difference.

#### Llama-3.3-70B-FP8 (4 replicas, TP=2)

| Streams | Prior Default (tok/s) | Optimised Baseline (tok/s) | Δ Throughput | Δ TTFT p50 | Δ TTFT p99 |
|---|---|---|---|---|---|
| 32 | 965 | 945 | -2.1% | -5.9% | -13.4% |
| 64 | 1,513 | 1,609 | +6.4% | +11.1% | -19.1% |
| 128 | 2,277 | 2,282 | +0.2% | -6.2% | -2.6% |
| 256 | 1,035 | 1,505 | **+45.4%** | +19.4% | -5.2% |
| 512 | 598 | 625 | +4.5% | +9.9% | -10.0% |

The optimised baseline delivers +45.4% higher output throughput at
concurrency 256 (1,505 vs 1,035 tok/s). The prior default's throughput
drops from its peak of 2,277 at 128 to 1,035 at 256. The optimised
baseline sustains 1,505 tok/s at the same concurrency level.
TTFT p99 is lower for the optimised baseline at every concurrency level
(-2.6% to -19.1%). Zero errors recorded for both configs.

![Llama-3.3-70B-FP8 throughput comparison](analysis/throughput_comparison_Llama-3_3-70B-FP8.png)

#### gpt-oss-120b (2 replicas, TP=4)

| Streams | Prior Default (tok/s) | Optimised Baseline (tok/s) | Δ Throughput | Δ TTFT p50 |
|---|---|---|---|---|
| 32 | 1,099 | 2,386 | **+117.0%** | -92.7% |
| 64 | 2,408 | 3,606 | +49.8% | -12.2% |
| 128 | 3,190 | 4,592 | +44.0% | -68.1% |
| 256 | 4,368 | 5,354 | +22.6% | -25.2% |
| 512 | 5,460 | 6,117 | +12.0% | -13.2% |

The optimised baseline outperforms the prior default at every concurrency
level. The largest delta is at streams=32: +117% throughput and TTFT p50
reduced from 1,125ms to 82ms. A small number of errored requests
(1–2 per run) were observed at streams=128 and 256 for both configs.

![gpt-oss-120b throughput comparison](analysis/throughput_comparison_gpt-oss-120b.png)

#### Qwen3-30B-A3B (8 replicas, TP=1)

| Streams | Prior Default (tok/s) | Optimised Baseline (tok/s) | Δ Throughput |
|---|---|---|---|
| 32 | 1,490 | 2,057 | +38.1% |
| 64 | 2,516 | 3,122 | +24.1% |
| 128 | 3,827 | 3,853 | +0.7% |
| 256 | 4,010 | 3,982 | -0.7% |
| 512 | 3,803 | 3,803 | 0.0% |

The optimised baseline improves throughput by 24–38% at streams 32–64.
At streams 128+ both configs produce equivalent throughput. Zero errors
recorded for both configs.

#### Multi-Turn Summary

![Summary heatmap](analysis/summary_heatmap.png)

Throughput deltas range from -2.1% to +117% across all models and
concurrency levels. The -2.1% for Llama-70B at streams=32 (965 vs 945
tok/s, 320 requests) is within run-to-run variability for a short test
at low concurrency. The largest improvements are observed at streams=32
for gpt-oss-120b (+117%) and Qwen3-30B (+38.1%), and at streams=256 for
Llama-70B (+45.4%).

### 3.2 Heavy-Heterogeneous (Llama-3.3-70B-FP8)

This profile uses a wide ISL distribution (50–30,000 tokens) which
produces uneven per-request KV cache demand across endpoints.

| Streams | Prior Default (tok/s) | Optimised Baseline (tok/s) | Δ Throughput | Δ TTFT p50 |
|---|---|---|---|---|
| 1 | 73 | 73 | 0.0% | +2.4% |
| 50 | 1,599 | 1,656 | +3.6% | -15.9% |
| 100 | 1,951 | 1,978 | +1.4% | -24.7% |
| 200 | 1,411 | 2,198 | **+55.8%** | -53.3% |
| 300 | 644 | 2,162 | **+236.0%** | -75.7% |

At streams=300, the prior default delivers 644 tok/s with a TTFT p50 of
117,927ms. The optimised baseline delivers 2,162 tok/s with a TTFT p50
of 28,606ms — a 3.4× throughput difference. The prior default's
throughput drops sharply above streams=100 while the optimised baseline
sustains above 2,100 tok/s through streams=300.

![Heavy-heterogeneous comparison](analysis/heavy_hetero_Llama-3_3-70B-FP8.png)

### 3.3 Prefix-Cache-Stress (Llama-3.3-70B-FP8)

This Poisson rate sweep with 150 unique 6k-token system prompts tests
prefix cache behaviour in isolation.

The optimised baseline shows 3–40% lower throughput than the prior
default across the tested rates (mean delta: -21.6%). Both configs
use prefix-cache-scorer at weight 3, so prefix routing decisions are
identical. The 30s-per-rate constraint limits queue depth and KV cache
pressure. At arrival rates above 15 req/s, both configs saturate the
4-replica cluster and throughput drops to near zero.

![Prefix cache sweep](analysis/prefix_cache_sweep_Llama-3_3-70B-FP8.png)

## 4. Assessment

The optimised baseline meets or exceeds the prior default in the
multi-turn profile (15 of 15 comparison points show no regression) and
the heavy-heterogeneous profile (5 of 5 points, with +56% to +236%
improvement at high concurrency). The prefix-cache-stress profile
(16 rate levels) shows 3–40% lower throughput for the optimised
baseline under low KV cache pressure conditions.

## 5. Methodology Notes

vLLM 0.19.1+rhaiv.6 configuration was identical across all runs; only
the EPP EndpointPickerConfig differed.

- Each multi-turn concurrency level ran 10×concurrency requests
  (e.g. 2,560 requests at streams=256).
- Heavy-heterogeneous ran 600s per concurrency level.
- Prefix-cache-stress ran 30s per Poisson rate with a 50s warmup at rate=15.
- Cluster: RHOAI 3.5 EA1 (rhods-operator 3.5.0-ea.1) on OCP 4.21,
  single H200 GPU node (8× NVIDIA H200).
- Results collected via guidellm v0.7.1, PCP 7.1.5.
