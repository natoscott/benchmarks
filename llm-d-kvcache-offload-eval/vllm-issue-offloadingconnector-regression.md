# vLLM GitHub Issue: OffloadingConnector request convoy / TTFT regression

**Title:**
`[Bug][OffloadingConnector] _blocks_being_loaded serialises all concurrent requests through a single load, causing 12× TTFT inflation and 34% throughput loss`

---

## Summary

When multiple concurrent requests share prefix blocks that are being loaded from
CPU offload storage, `OffloadingConnector` delays ALL of them (`return None`) until
the first load completes. With prefix caching and a shared 10,000-token prefix, 50
concurrent requests are fully serialised behind a single load job, collapsing
throughput and inflating TTFT by up to 12×.

The regression was introduced in PR #29087 (commit `7013e9ac8`,
"Prevent redundant loads") and remains unfixed in `HEAD` (`8a9eb408`, 2026-05-25).

---

## Environment

```
GPU:     2× NVIDIA L40S (24 GB each), tensor_parallel_size=2
Model:   Qwen/Qwen3-8B (BF16)
vLLM:    0.19.1 (regression first observed vs 0.17.1)
Config:  OffloadingConnector, cpu_bytes_to_use=57616986275 (~20k blocks)
         enable_prefix_caching=True (default)
         gpu_memory_utilization=0.9
Workload: 50 concurrent requests, 10,000-token shared prefix, 128 output tokens
```

---

## Observed behaviour

Throughput and TTFT for Qwen3-8B native CPU offload vs no-offload baseline,
measured at `gpu_memory_utilization=0.9` (GPU KV-cache NOT constrained):

| vLLM version | Throughput overhead | TTFT at concurrency=50 |
|---|:---:|:---:|
| **0.17.1** | **−6.5%** | **0.90 s** |
| **0.19.1** | **−39.9%** | **11.46 s** |

No-offload TTFT at concurrency=50 is 5.95 s. Native-offload TTFT is 11.46 s —
**1.92× higher** — not from load latency but from artificial queueing.

The regression is specific to `enable_prefix_caching=True`; when prefix caching
is disabled `_blocks_being_loaded` is `None` and the blocking path is skipped.

---

## Root cause

PR #29087 introduced `_blocks_being_loaded` to prevent duplicate CPU→GPU load jobs
for the same blocks. The mechanism is correct in intent but too aggressive in scope:
it delays **every request** that references blocks currently being loaded, even
requests that could proceed independently.

**Relevant code** (`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`,
confirmed in HEAD `8a9eb408`):

```python
# __init__ (line ~293)
self._blocks_being_loaded: set[OffloadKey] | None = (
    set() if spec.vllm_config.cache_config.enable_prefix_caching else None
)

# get_num_new_matched_tokens (line ~491-511)
if self._blocks_being_loaded:
    for group_config, group_state in zip(...):
        ...
        offload_keys = offload_keys[start_block_idx:num_blocks]
        if any(key in self._blocks_being_loaded for key in offload_keys):
            logger.debug("Delaying request %s ...", req_status.req.request_id)
            return None   # ← delays the ENTIRE request, not just the load
```

**Scenario with 50 concurrent requests sharing a 10,000-token prefix:**

1. Request A misses GPU cache for prefix blocks → load job issued →
   blocks added to `_blocks_being_loaded`
2. Requests B–AZ arrive, all sharing the same prefix → each hits
   `_blocks_being_loaded` → all delayed (`return None`)
3. All 49 requests queue behind request A's single load job
4. Load completes → next scheduler step processes one more request →
   repeat (the load now runs for B, delaying C–AZ)

This creates a request convoy: O(n) sequential loads instead of O(1) concurrent
processing.

---

## Proposed fix

Return `0` (no CPU hit, proceed normally) instead of `None` (delay) when a
request finds its blocks already being loaded. This allows the request to proceed
without a load job — it re-prefills this turn — while the in-flight load warms the
GPU cache for subsequent requests. The "no duplicate loads" property is preserved
since the guard still prevents a second load job from being issued.

```python
# scheduler.py ~line 507
if any(key in self._blocks_being_loaded for key in offload_keys):
    logger.debug(
        "Skipping CPU hit for request %s — blocks already loading",
        req_status.req.request_id,
    )
-   return None   # causes request convoy at high concurrency
+   return 0      # skip CPU hit this turn, avoid serialisation
```

**Trade-off:** the second request re-prefills once instead of waiting for the first
request's load. At high concurrency (where the regression bites), the cost of one
re-prefill is far less than O(n) serialisation. At low concurrency (rate=1), there
is no convoy and `_blocks_being_loaded` is rarely populated, so behaviour is
unchanged.

---

## Reproducer

```bash
# Configure model server with OffloadingConnector (20k-block equivalent)
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"cpu_bytes_to_use":57616986275}}'

# Run GuideLLM at concurrency=50, shared 10K-token prefix
guidellm benchmark run --target http://localhost:8000 \
  --rate-type concurrent --rate 50 --max-seconds 120 \
  --data '{"prompt_tokens":128,"output_tokens":128,"prefix_tokens":10000,"turns":5}'
```

Expected: throughput close to no-offload (GPU cache unconstrained at gmu=0.9)  
Actual: 39.9% throughput loss, 11.5 s TTFT (vs 5.95 s no-offload)

---

## Bisect

The regression is introduced by:
```
7013e9ac8 OffloadingConnector: Prevent redundant loads (#29087)
```

The fix was not in vLLM 0.17.1; it landed between 0.17.1 and 0.19.1, and remains
present in `HEAD` (`8a9eb408`, 2026-05-25).

---

## Related

- PR #29087 — introduced `_blocks_being_loaded`
- PR #34805 (`fcf0687b2`) — fixed 32B-AWQ preemption crash (separate issue, confirmed fixed)
- No existing open issue found for this throughput/TTFT regression
