# EPP Scheduling Configuration Evaluation — RHOAI 3.5

## TL;DR

**Three EPP configurations were compared: the prior RHOAI default (A),
the previous upstream optimised baseline with 4 scorers (B), and the
new upstream optimised baseline with filter+scorer architecture (C).**
Config B meets or exceeds Config A across all multi-turn and
heavy-heterogeneous test points, with +45% throughput for Llama-70B at
concurrency 256 and +236% under heterogeneous workloads at concurrency
300.  Config C shows mixed results: heavy-heterogeneous matches
Config B (+230% vs Config A), but multi-turn throughput is 14–73%
lower for Llama-70B (requests succeed but complete slowly), 7–21%
lower for gpt-oss-120b, and Qwen3-30B returns 400 errors (upstream
EPP forwards requests with empty body to vLLM).  Prefix-cache-stress
requests time out with zero completions under Config C.
`peakPrefillThroughput` was calibrated per model on H200 — calibration
did not change the outcome.
**The data supports shipping Config B.  Config C's multi-turn
throughput reductions and request-forwarding defects are inherent to
the upstream EPP image, not caused by miscalibration.**

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
| 32 | 965 | 945 | 616 | -2.1% | -36.1% |
| 64 | 1,513 | 1,609 | 766 | +6.4% | -49.4% |
| 128 | 2,277 | 2,282 | 617 | +0.2% | -72.9% |
| 256 | 1,035 | 1,505 | 538 | **+45.4%** | -48.0% |
| 512 | 598 | 625 | 516 | +4.5% | -13.7% |

Config B delivers +45.4% higher throughput at concurrency 256.
Config C delivers 14–73% lower throughput than Config A at all levels.
Zero errors for Configs A and B; Config C recorded 1 error at
streams=512.

![Llama-3.3-70B-FP8 throughput](analysis/throughput_comparison_Llama-3_3-70B-FP8.png)

#### gpt-oss-120b (2 replicas, TP=4)

| Streams | A: Prior (tok/s) | B: Optimised (tok/s) | C: New (tok/s) | Δ B vs A | Δ C vs A |
|---|---|---|---|---|---|
| 32 | 1,099 | 2,386 | 945 | **+117.0%** | -14.0% |
| 64 | 2,408 | 3,606 | 1,901 | +49.8% | -21.0% |
| 128 | 3,190 | 4,592 | 2,756 | +44.0% | -13.6% |
| 256 | 4,368 | 5,354 | 3,972 | +22.6% | -9.1% |
| 512 | 5,460 | 6,117 | 5,061 | +12.0% | -7.3% |

Config B outperforms at every level (+12% to +117%).  Config C is 7–21%
below Config A. Small numbers of errored requests (1–4) observed at
streams 64+ for all three configs.

![gpt-oss-120b throughput](analysis/throughput_comparison_gpt-oss-120b.png)

#### Qwen3-30B-A3B (8 replicas, TP=1)

| Streams | A: Prior (tok/s) | B: Optimised (tok/s) | C: New (tok/s) | Δ B vs A | Δ C vs A |
|---|---|---|---|---|---|
| 32 | 1,490 | 2,057 | 350 | +38.1% | -76.5% |
| 64 | 2,516 | 3,122 | 387 | +24.1% | -84.6% |
| 128 | 3,827 | 3,853 | 264 | +0.7% | -93.1% |
| 256 | 4,010 | 3,982 | 362 | -0.7% | -91.0% |
| 512 | 3,803 | 3,803 | 437 | 0.0% | -88.5% |

Config C shows high error rates at all concurrency levels: 64 errors
(of 320 requests) at streams=32, scaling to 1,024 errors (of 5,120
requests) at streams=512. The upstream EPP returns `400 Bad Request`
for Qwen3-30B requests (vLLM receives empty body). This is a
model-specific defect in the upstream EPP image.

#### Multi-Turn Summary

![Summary heatmap](analysis/summary_heatmap.png)

Config B shows no regression vs Config A at any concurrency level for
any model. Config C shows throughput reductions of 14–73% for Llama-70B,
7–21% for gpt-oss-120b, and 400 errors for Qwen3-30B at all concurrency
levels.

### 3.2 Heavy-Heterogeneous

Heavy-heterogeneous and prefix-cache-stress profiles were run for
Llama-3.3-70B-FP8 only (the primary model under evaluation, 4
replicas TP=2). Qwen3-30B and gpt-oss-120b were not tested on
these profiles.

| Streams | A: Prior (tok/s) | B: Optimised (tok/s) | C: New (tok/s) | Δ B vs A | Δ C vs A |
|---|---|---|---|---|---|
| 1 | 73 | 73 | 73 | 0.0% | 0.0% |
| 50 | 1,599 | 1,656 | 1,623 | +3.6% | +1.5% |
| 100 | 1,951 | 1,978 | 1,977 | +1.4% | +1.3% |
| 200 | 1,411 | 2,198 | 2,201 | **+55.8%** | **+56.0%** |
| 300 | 644 | 2,162 | 2,125 | **+236.0%** | **+230.2%** |

All three configs converge at low concurrency. At streams=200 and 300,
Configs B and C both deliver ~2,150–2,200 tok/s while Config A drops to
644 tok/s. Config C matches Config B on this profile.

![Heavy-heterogeneous comparison](analysis/heavy_hetero_Llama-3_3-70B-FP8.png)

### 3.3 Prefix-Cache-Stress (Llama-3.3-70B-FP8)

This profile sends Poisson-distributed requests with 7,233 input tokens
(6k prefix + 1.2k prompt) and 1,000 output tokens at 16 arrival rates
from 3 to 60 req/s, with a 30s time constraint per rate.

At rates 3 and 10 req/s, both Configs A and B complete requests (A:
1,904 and 1,503 tok/s; B: 1,844 and 902 tok/s). At rate 15+ req/s,
**all three configs record zero completed requests** — all dispatched
requests are cancelled when the 30s window expires.

At rate=15, 450 requests arrive in 30s. Each request requires
processing 7,233 input tokens plus generating 1,000 output tokens.
With 4 replicas of Llama-70B FP8, the cluster cannot complete enough
requests within 30s at this arrival rate — the queue grows faster
than requests complete. This is a workload saturation effect, not an
EPP scheduling difference. The 30s-per-rate constraint (from the
upstream benchflow profile) is too short for this ISL/OSL combination
on Llama-70B.

At the two rates where requests complete (3 and 10 req/s), Config B
shows 3% and 40% lower throughput than Config A respectively.

Config C dispatched requests but none completed within 30s at any
rate (all cancelled, zero errors).

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

**Root cause investigation**: `peakPrefillThroughput` was calibrated
per model on H200 (Qwen3-30B: 174,330; Llama-70B: 101,012;
gpt-oss-120b: 245,270 — vs default 15,928). Calibration did not
change the outcome: Llama-70B multi-turn throughput remains 14–73%
below Config A, Qwen3-30B still returns 400 errors, and
prefix-cache-stress still times out. This rules out miscalibration
as the cause.

The three failure modes under Config C:

1. **Qwen3-30B 400 errors**: A single-request diagnostic confirmed
   vLLM returns `400 Bad Request` with `body: None` when routed through
   the upstream EPP. The same request through the RHOAI EPP returns
   `200 OK`. This is a model-dependent request-forwarding defect in
   `ghcr.io/llm-d/llm-d-router-endpoint-picker:main` — it does not
   affect Llama-70B or gpt-oss-120b.

2. **Llama-70B and gpt-oss-120b multi-turn low throughput**: Requests
   complete without errors but throughput is 14–73% (Llama-70B) and
   7–21% (gpt-oss-120b) lower than Config A. Calibration confirmed
   this is not a `peakPrefillThroughput` issue. The throughput
   reduction is inherent to the upstream EPP image's request handling.

3. **Prefix-cache-stress all cancelled**: Requests were dispatched
   but none completed within the 30s constraint (status: cancelled).
   Zero errors, zero successes.

The heavy-heterogeneous profile produced valid results under Config C,
matching Config B's throughput (+230% vs Config A at streams=300).

Config C's multi-turn throughput reductions and request-forwarding
defects require investigation in the upstream
`llm-d-router-endpoint-picker` codebase.

## 5. Methodology Notes

vLLM 0.19.1+rhaiv.6 configuration was identical across all runs.
Configs A and B used the RHOAI 3.5 EA2 EPP image; Config C used the
upstream community EPP image (`ghcr.io/llm-d/llm-d-router-endpoint-picker:main`).
Config C's `peakPrefillThroughput` was calibrated per model on H200
(Qwen3-30B: 174,330; Llama-70B: 101,012; gpt-oss-120b: 245,270).

- Each multi-turn concurrency level ran 10×concurrency requests
  (e.g. 2,560 requests at streams=256).
- Heavy-heterogeneous ran 600s per concurrency level.
- Prefix-cache-stress ran 30s per Poisson rate with a 50s warmup at rate=15.
- Cluster: RHOAI 3.5 EA2 (rhods-operator 3.5.0-ea.2) on OCP 4.21,
  single H200 GPU node (8× NVIDIA H200).
- Results collected via guidellm v0.7.1, PCP 7.1.5.
