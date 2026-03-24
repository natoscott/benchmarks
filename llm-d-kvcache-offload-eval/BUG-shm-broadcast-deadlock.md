# Bug: EngineCore deadlock under SharedStorageOffloadingSpec + mp executor at high concurrency

## Summary

When using `SharedStorageOffloadingSpec` (from `llmd_fs_connector`) with
`--distributed-executor-backend mp`, the vLLM EngineCore deadlocks under
sustained high concurrency. The API server remains alive and `/health` returns
200 OK, but the engine stops dispatching requests, causing the gateway to return
HTTP 503 for all subsequent requests.

## Environment

| Component | Version |
|-----------|---------|
| vLLM | 0.15.1 |
| llm-d | v0.5.1 (`ghcr.io/llm-d/llm-d-cuda:v0.5.1`) |
| llmd_fs_connector | 0.15.1 (`llmd_fs_connector-0.15.1-cp312-cp312-linux_x86_64.whl`) |
| Python | 3.12.12 |
| CUDA | 12.9 |
| GPUs | 2× NVIDIA L40S (tensor-parallel-size=2) |
| OS | RHEL 9 (kernel 5.14) |
| Executor backend | `--distributed-executor-backend mp` |
| Connector | `OffloadingConnector` / `SharedStorageOffloadingSpec` |

## Reproduction

### vLLM command

```bash
pip3.12 install --target /tmp/llmd_packages \
    llmd_fs_connector-0.15.1-cp312-cp312-linux_x86_64.whl

PYTHONPATH=/tmp/llmd_packages \
LD_PRELOAD=/opt/nvidia/nsight-compute/.../libstdc++.so.6 \
PYTHONHASHSEED=42 \
exec vllm serve Qwen/Qwen3-0.6B \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max-num-seq 1024 \
    --distributed-executor-backend mp \
    --kv-transfer-config '{
      "kv_connector": "OffloadingConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "spec_name": "SharedStorageOffloadingSpec",
        "shared_storage_path": "/kvcache/kv-cache/",
        "block_size": 256,
        "threads_per_gpu": 64,
        "spec_module_path": "llmd_fs_backend.spec"
      }
    }'
```

### Workload to trigger the deadlock

Send ≥300 concurrent requests with a large shared prefix (10,000 tokens),
5 conversation turns, 128 prompt + 128 output tokens per turn:

```bash
guidellm benchmark run \
    --target http://localhost:8000 \
    --rate-type concurrent \
    --rate 300 \
    --max-seconds 120 \
    --data '{"prompt_tokens":128,"output_tokens":128,"prefix_tokens":10000,"turns":5,"prefix_count":600}'
```

The deadlock manifests within ~8 minutes of the high-concurrency load starting.
At `rate=150` the first ~39 requests succeed before the deadlock sets in.
At `rate=300` and `rate=500` zero requests succeed.

The issue does **not** reproduce with:
- `--distributed-executor-backend ray` (not tested but likely avoids the shm path)
- native-offload-20k (`CPUOffloadingSpec`) under identical concurrency
- no-offload under identical concurrency
- Larger models (Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ) under identical concurrency
  (slower token generation rate reduces KV write pressure sufficiently)

## Observed Symptoms

### 1. vLLM log: shm_broadcast timeout

Appears in the vLLM log approximately 8 minutes after load begins, then repeats
every 60 seconds until the process is restarted:

```
[EngineCore_DP0 pid=288] INFO shm_broadcast.py:542 No available shared memory
broadcast block found in 60 seconds. This typically happens when some processes
are hanging or doing some time-consuming work (e.g. compilation, weight/kv
cache quantization).
```

### 2. HTTP 503 from gateway

All client requests receive `HTTP 503 Service Unavailable` once the deadlock
sets in. The `/health` endpoint continues to return 200 OK (the API server
process is alive; only the EngineCore is stalled).

### 3. Request error distribution (from GuideLLM)

At rate=300 over 120 seconds:
- Successful: 0
- Errored: 53,415 (`HTTPStatusError: 503 Service Unavailable`)
- Incomplete: 3 (cancelled at benchmark end)

Error traceback from client:
```
httpx.HTTPStatusError: Server error '503 Service Unavailable' for url
  'http://<gateway>/v1/chat/completions'
```

### 4. Connector initialisation is correct

The startup log confirms `SharedStorageOffloadingSpec` initialises successfully
on all TP workers before the deadlock occurs:

```
[Worker_TP0] INFO factory.py:51 Creating offloading spec with name: SharedStorageOffloadingSpec
[Worker_TP1] INFO factory.py:51 Creating offloading spec with name: SharedStorageOffloadingSpec
[Worker_TP0] INFO worker.py:223 StorageOffloadingHandlers: threads_per_gpu=64,
    offloading block_size=256, staging_buffer_size_mb=14, max_staging_memory_gb=150
[Worker_TP1] INFO worker.py:223 StorageOffloadingHandlers: threads_per_gpu=64,
    offloading block_size=256, staging_buffer_size_mb=14, max_staging_memory_gb=150
```

## Root Cause Analysis

The `--distributed-executor-backend mp` executor uses shared memory buffers for
inter-process communication between the EngineCore scheduler and the worker
processes. The `shm_broadcast.py` module implements a circular broadcast
mechanism where the EngineCore writes to a slot and waits for all workers to
acknowledge before reusing it.

Under high concurrency with `SharedStorageOffloadingSpec`, the filesystem I/O
worker threads (64 per GPU) become saturated with KV block write operations.
While these threads are blocked on I/O, the main worker process loop cannot
drain the shared memory broadcast queue fast enough. The EngineCore times out
waiting for broadcast acknowledgement (60-second timeout), logs the warning, and
the engine stalls. No further requests are dispatched until the process is
restarted.

**Why Qwen3-0.6B is affected but larger models are not:**

Qwen3-0.6B generates tokens at ~636 tok/s (no-offload, rate=50). Under
rate=300 concurrent requests, the volume of KV blocks written per second
substantially exceeds what larger models produce — Qwen3-8B generates ~114
tok/s, Qwen3-14B ~59 tok/s, Qwen3-32B-AWQ ~51 tok/s. The smaller model's
high throughput creates disproportionate write pressure on the connector's I/O
threads, crossing the threshold where broadcast starvation occurs.

**Storage medium note:**

The storage path (`/kvcache/kv-cache/`) was backed by an IBM VPC block PVC
mounted via the kernel filesystem. During the 120-second benchmark window, disk
I/O was negligible (≤0.04 MB/s) — all writes went to the OS page cache. The
deadlock therefore occurs before any physical I/O, purely from the threading
and IPC interaction between the connector's I/O pool and the mp executor's
broadcast mechanism.

## Impact

- **Scope**: Single-node deployments using `SharedStorageOffloadingSpec` with
  `--distributed-executor-backend mp` under high request concurrency with
  fast (small) models
- **Effect**: Complete request failure (100% 503 errors) with no vLLM crash or
  restart required to trigger — the process must be restarted to recover
- **Workaround**: None identified within the current vLLM 0.15.1 +
  llmd_fs_connector 0.15.1 combination. Reducing concurrency below the
  threshold (rate < ~150 for Qwen3-0.6B on 2× L40S) avoids the deadlock.

## Suggested Investigation

1. **Increase broadcast buffer size** in `shm_broadcast.py` to reduce
   sensitivity to slow broadcast acknowledgements
2. **Non-blocking I/O** in `StorageOffloadingHandlers`: ensure I/O threads
   cannot block the worker process's main loop (e.g. async I/O or a separate
   process for I/O)
3. **Broadcast timeout tuning**: the 60-second timeout may be appropriate for
   compilation/quantisation but too long for production request handling;
   a shorter timeout with graceful degradation may be preferable
4. **Backpressure from connector**: the EngineCore should limit request dispatch
   rate when the connector's I/O queue depth exceeds a threshold

## Data Location

Full benchmark data is in the llm-d benchmarks repository:

- GuideLLM results: `results/1x2xL40S_upstream-llm-d-0.5.1_Qwen3-0.6B_fs-offload_replica1_rate300/guidellm-results.json.zst`
- vLLM startup log: `results/1x2xL40S_upstream-llm-d-0.5.1_Qwen3-0.6B_fs-offload_replica1_rate300/vllm-startup.log.zst`
- PCP archives: `results/1x2xL40S_upstream-llm-d-0.5.1_Qwen3-0.6B_fs-offload_replica1_rate300/pcp-archives/`
- Same pattern for `rate500` and `cpu+fs-offload-20k` variant

Related: REPORT-v0.5.1.md §"Qwen3-0.6B EngineCore Deadlock"
