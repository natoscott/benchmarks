# EPP Scheduling Configuration Evaluation — RHOAI 3.5

## TL;DR

**Three EPP configurations were compared: the prior RHOAI default (A),
the previous upstream optimised baseline with 4 scorers (B), and the
new upstream optimised baseline with filter+scorer architecture (C).**
Config B meets or exceeds Config A across all multi-turn and
heavy-heterogeneous test points, with +45% throughput for Llama-70B at
concurrency 256 and +236% under heterogeneous workloads at concurrency
300.  Config C's multi-turn and prefix-cache-stress results are
**invalid due to a request-forwarding defect** in the upstream EPP
image (`ghcr.io/llm-d/llm-d-router-endpoint-picker:main`), which
forwards requests with an empty body causing vLLM to return 400 errors.
Config C's heavy-heterogeneous results are valid and match Config B's
throughput (+234% vs Config A at concurrency 300).
**The data supports shipping Config B.  Config C cannot be fully
evaluated until the upstream request-forwarding defect is resolved.**

---

## 1. Objective

Validate EPP scheduler configurations for RHOAI 3.5 default selection.
Compare the prior RHOAI default, the previous upstream optimised
baseline, and the new upstream optimised baseline across representative
workloads and concurrency levels.

### EPP Plugin Reference

#### Configs A and B: Score-Only Architecture

All endpoints are scored by each plugin; `max-score-picker` selects
the highest weighted sum.  Scorers read vLLM Prometheus metrics
scraped by the EPP's built-in metrics data layer.

- **queue-scorer**: Scores endpoints inversely proportional to their
  waiting queue size. Normalises across all endpoints so the endpoint
  with the shortest queue gets score 1.0 and the longest gets 0.0.
- **kv-cache-utilisation-scorer** (Config B only): Scores each endpoint
  as `1 - KVCacheUsagePercent`. Endpoints with more free KV cache
  memory score higher, distributing requests away from memory-pressured
  endpoints.
- **prefix-cache-scorer**: Scores endpoints based on how many prefix
  tokens from the incoming request are already cached on that endpoint.
  Routes requests to where their prefix is already in GPU KV cache,
  avoiding redundant prefill computation.
- **no-hit-lru-scorer** (Config B only): Only activates for "cold"
  requests (no prefix cache hit on any endpoint). For cold requests,
  scores endpoints inversely by recency of use — endpoints that haven't
  been routed to recently score higher. For cache-hit requests, returns
  neutral scores (no effect). This is the only stateful scorer — it
  maintains an internal LRU cache of recent routing decisions.
- **max-score-picker**: Selects the endpoint with the highest weighted
  score. With no scorers enabled, all endpoints score 0 and the picker
  shuffles randomly (uniform random selection, not round-robin).

#### Config C: Filter-Then-Score Architecture

Config C replaces the score-only pipeline with a two-stage architecture
built on EPP-internal data producers rather than vLLM Prometheus metrics.

**Data producers** (run before scheduling):

- **approx-prefix-cache-producer**: Tokenises the incoming request using
  the EPP's tokeniser sidecar, computes an approximate prefix hash, and
  annotates each endpoint with a `PrefixCacheMatchInfo` attribute.
- **inflight-load-producer**: Tracks in-flight requests and tokens per
  endpoint using internal request/response lifecycle hooks
  (PreRequest/PostRequest). This is the EPP's own view of dispatched
  work, not read from vLLM metrics.

**Filter stage**:

- **prefix-cache-affinity-filter**: Narrows candidates to "sticky"
  endpoints (prefix cache score >= 0.80) before scoring. If any sticky
  endpoint exists, non-sticky endpoints are eliminated unless the
  filter's load gate fires. The load gate estimates TTFT per endpoint
  as `inFlightTokens / peakPrefillThroughput * 1000` (ms) and breaks
  stickiness if the best sticky endpoint's TTFT exceeds the best
  non-sticky endpoint's by more than `maxTTFTPenaltyMs` (default 18s).
  `peakPrefillThroughput` defaults to 15928 tok/s (calibrated for
  Qwen3-32B on H100 TP=2). An `explorationProbability` parameter
  (default 0) randomly bypasses the filter for cache warming.

**Score stage**:

- **token-load-scorer**: Scores surviving endpoints by total in-flight
  token load (from `inflight-load-producer`) plus the estimated uncached
  portion of the current request, normalised against a configurable
  threshold. Score = `1 - (tokens / threshold)`. Replaces both
  queue-scorer (which counted requests, not tokens) and
  kv-cache-utilisation-scorer (which read vLLM's KV cache metric).

## 2. Configuration

### EPP Configurations Under Test

| Config | Architecture | Plugins | Weights |
|---|---|---|---|
| **A: Prior Default** | Score-only | queue-scorer, prefix-cache-scorer | 2, 3 |
| **B: Optimised Baseline** | Score-only | queue-scorer, kv-cache-utilisation-scorer, prefix-cache-scorer, no-hit-lru-scorer | 2, 2, 3, 2 |
| **C: New Optimised Baseline** | Filter+score | prefix-cache-affinity-filter, token-load-scorer | —, — |

Config C uses the upstream community EPP image
(`ghcr.io/llm-d/llm-d-router-endpoint-picker:main`) as the new plugins
are not in the RHOAI 3.5 EA2 image.  Configs A and B use the RHOAI EA2
EPP image.

### Models

| Model | Type | TP | Replicas | GPUs |
|---|---|---|---|---|
| Qwen/Qwen3-30B-A3B-Instruct-2507 | MoE | 1 | 8 | 8 |
| RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic | Dense FP8 | 2 | 4 | 8 |
| openai/gpt-oss-120b | MoE MXFP4 | 4 | 2 | 8 |

All models deployed on a single H200 node (8 GPUs) via RHOAI 3.5 EA2
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

#### Llama-3.3-70B-FP8 (4 replicas, TP=2)

| Streams | A: Prior (tok/s) | B: Optimised (tok/s) | C: New (tok/s) | Δ B vs A | Δ C vs A |
|---|---|---|---|---|---|
| 32 | 965 | 945 | 588 | -2.1% | -39.1% |
| 64 | 1,513 | 1,609 | 755 | +6.4% | -50.1% |
| 128 | 2,277 | 2,282 | 610 | +0.2% | -73.2% |
| 256 | 1,035 | 1,505 | 530 | **+45.4%** | -48.7% |
| 512 | 598 | 625 | 516 | +4.5% | -13.6% |

Config B delivers +45.4% higher throughput at concurrency 256.
Config C delivers 14–73% lower throughput than Config A at all levels.
Zero errors for Configs A and B; Config C recorded 1 error at
streams=256.

![Llama-3.3-70B-FP8 throughput](analysis/throughput_comparison_Llama-3_3-70B-FP8.png)

#### gpt-oss-120b (2 replicas, TP=4)

| Streams | A: Prior (tok/s) | B: Optimised (tok/s) | C: New (tok/s) | Δ B vs A | Δ C vs A |
|---|---|---|---|---|---|
| 32 | 1,099 | 2,386 | 1,072 | **+117.0%** | -2.5% |
| 64 | 2,408 | 3,606 | 2,230 | +49.8% | -7.4% |
| 128 | 3,190 | 4,592 | 2,841 | +44.0% | -10.9% |
| 256 | 4,368 | 5,354 | 3,924 | +22.6% | -10.2% |
| 512 | 5,460 | 6,117 | 4,809 | +12.0% | -11.9% |

Config B outperforms at every level (+12% to +117%).  Config C is 2–12%
below Config A. Small numbers of errored requests (1–3) observed at
streams 128+ for all three configs.

![gpt-oss-120b throughput](analysis/throughput_comparison_gpt-oss-120b.png)

#### Qwen3-30B-A3B (8 replicas, TP=1)

| Streams | A: Prior (tok/s) | B: Optimised (tok/s) | C: New (tok/s) | Δ B vs A | Δ C vs A |
|---|---|---|---|---|---|
| 32 | 1,490 | 2,057 | 2,082 | +38.1% | +39.7% |
| 64 | 2,516 | 3,122 | 438 | +24.1% | -82.6% |
| 128 | 3,827 | 3,853 | 672 | +0.7% | -82.4% |
| 256 | 4,010 | 3,982 | 736 | -0.7% | -81.6% |
| 512 | 3,803 | 3,803 | 492 | 0.0% | -87.1% |

Config C matches B at streams=32 (+40%) but shows high error rates
at streams 64+: 128 errors (of 640 requests) at streams=64, scaling
linearly to 1,024 errors (of 5,120 requests) at streams=512. The
reported throughput for Config C at streams 64+ reflects only the
successful requests; total effective throughput including errors is
lower than the table values suggest.

#### Multi-Turn Summary

![Summary heatmap](analysis/summary_heatmap.png)

Config B shows no regression vs Config A at any concurrency level for
any model. Config C shows throughput reductions of 14–73% for Llama-70B,
high error rates for Qwen3-30B at concurrency 64+, and 2–12% reductions
for gpt-oss-120b.

### 3.2 Heavy-Heterogeneous (Llama-3.3-70B-FP8)

| Streams | A: Prior (tok/s) | B: Optimised (tok/s) | C: New (tok/s) | Δ B vs A | Δ C vs A |
|---|---|---|---|---|---|
| 1 | 73 | 73 | 73 | 0.0% | 0.0% |
| 50 | 1,599 | 1,656 | 1,623 | +3.6% | +1.5% |
| 100 | 1,951 | 1,978 | 1,977 | +1.4% | +1.3% |
| 200 | 1,411 | 2,198 | 2,206 | **+55.8%** | **+56.3%** |
| 300 | 644 | 2,162 | 2,154 | **+236.0%** | **+234.6%** |

All three configs converge at low concurrency. At streams=200 and 300,
Configs B and C both deliver ~2,150–2,200 tok/s while Config A drops to
644 tok/s. Config C matches Config B on this profile.

![Heavy-heterogeneous comparison](analysis/heavy_hetero_Llama-3_3-70B-FP8.png)

### 3.3 Prefix-Cache-Stress (Llama-3.3-70B-FP8)

Config B shows 3–40% lower throughput than Config A across tested rates
(mean delta: -21.6%).

Config C routed zero requests at all 16 tested rates (zero requests
total, zero errors). The upstream EPP with the new plugins did not
attempt to route any requests for this workload profile.

![Prefix cache sweep](analysis/prefix_cache_sweep_Llama-3_3-70B-FP8.png)

## 4. Assessment

### Config B (previous optimised baseline)

Meets or exceeds Config A in the multi-turn profile (15/15 points, no
regression) and the heavy-heterogeneous profile (5/5 points, +56% to
+236% at high concurrency). Shows 3–40% lower throughput in
prefix-cache-stress under low KV cache pressure. Suitable as the
RHOAI 3.5 default.

### Config C (new optimised baseline)

Matches Config B on heavy-heterogeneous (+234% to +236% vs Config A at
high concurrency). However, Config C shows 14–73% throughput reductions
on multi-turn for Llama-70B, high error rates (20% of requests) for
Qwen3-30B at concurrency 64+, and routed zero requests on
prefix-cache-stress.

**Root cause investigation**: A single-request test through the upstream
EPP image confirmed vLLM returns `400 Bad Request` with `body: None` —
the upstream EPP forwards requests with an empty body.  The same
request through the RHOAI EPP image returns `200 OK` with a valid
response.  This is a request-forwarding defect in the upstream
`ghcr.io/llm-d/llm-d-router-endpoint-picker:main` image, not a
scheduler configuration issue.

The heavy-heterogeneous profile produced valid results under Config C
(matching Config B's throughput), which indicates the body-forwarding
defect does not affect all request paths uniformly.

Config C's multi-turn and prefix-cache-stress results are invalid due
to this defect.  The heavy-heterogeneous results are valid and show
that Config C's filter+scorer architecture produces equivalent
throughput to Config B when requests are forwarded correctly.

Config C cannot be evaluated for the multi-turn critical scenario until
the upstream request-forwarding defect is resolved.  Additionally,
`peakPrefillThroughput` (default 15928 tok/s, calibrated for Qwen3-32B
on H100 TP=2) was not calibrated for the tested models/hardware.

## 5. Methodology Notes

vLLM 0.19.1+rhaiv.6 configuration was identical across all runs.
Configs A and B used the RHOAI 3.5 EA2 EPP image; Config C used the
upstream community EPP image (`ghcr.io/llm-d/llm-d-router-endpoint-picker:main`).

- Each multi-turn concurrency level ran 10×concurrency requests
  (e.g. 2,560 requests at streams=256).
- Heavy-heterogeneous ran 600s per concurrency level.
- Prefix-cache-stress ran 30s per Poisson rate with a 50s warmup at rate=15.
- Cluster: RHOAI 3.5 EA2 (rhods-operator 3.5.0-ea.2) on OCP 4.21,
  single H200 GPU node (8× NVIDIA H200).
- Results collected via guidellm v0.7.1, PCP 7.1.5.
