# PCP Archive Summary

## Archive Collection

PCP archives were collected for **16 benchmark configurations** testing llm-d distributed KV-cache management across 4 model sizes and 4 caching strategies:

- **Models**: Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ
- **Configurations**: no-offload, native-offload, llm-d-redis, llm-d-valkey
- **Duration per benchmark**: ~3-15 minutes (varies by model size and concurrency)
- **Total benchmark time**: ~4 hours (16 configurations × ~15 minutes average)

## Storage Statistics

| Metric | Value |
|--------|-------|
| Total archives | 16 configurations |
| Combined storage (compressed) | 9.3M |
| Metrics per archive | 2,074 unique metrics |
| Sample frequency | 10 seconds (openmetrics), 1 second (system) |
| Archive format | Compressed with zstd |

## Benchmark Coverage

Each archive captures metrics for one complete benchmark run:

| Model | Configurations | Concurrency Levels | Duration/Config |
|-------|---------------|-------------------|-----------------|
| Qwen3-0.6B | 4 | 1,50,100 | ~3 min |
| Qwen3-8B | 4 | 1,50,100,150,300,400,500,650 | ~16 min |
| Qwen3-14B | 4 | 1,50,100,150,300,400,500,650 | ~16 min |
| Qwen3-32B-AWQ | 4 | 1,50,100,150,300 | ~10 min |

## Metric Groups

### Utilized in Report

**vLLM Application Metrics (openmetrics.vllm.*): 113 metrics**
- `vllm.kv_cache_usage_perc` - KV-cache utilization percentage
- `vllm.prefix_cache_hits_total` - Prefix cache effectiveness
- `vllm.prefix_cache_queries_total` - Total prefix cache queries
- `vllm.num_requests_running` - Active request queue depth
- `vllm.num_requests_waiting` - Waiting request queue depth
- `process_resident_memory_bytes` - Process memory consumption
- HTTP request/response metrics

**llm-d EPP Metrics (openmetrics.epp.*): 496 metrics**
- Go runtime memory statistics
- HTTP endpoint metrics
- Request routing telemetry

**System Memory (mem.*): 8 metrics**
- Memory utilization
- Free/used memory tracking

**CPU (kernel.all.cpu.*): 5 metrics**
- CPU idle/user/system time
- Overall CPU utilization

### Available but Unused

The archives contain extensive additional system-level metrics:

- **network.***: ~378 metrics (interface statistics, TCP/IP metrics)
- **disk.***: ~132 metrics (I/O statistics, disk utilization)
- **xfs.***: ~210 metrics (XFS filesystem operations)
- **proc.***: ~56 metrics (per-process statistics)
- **ipc.***: ~49 metrics (inter-process communication)
- **hinv.***: ~38 metrics (hardware inventory)
- **kvm.***: ~34 metrics (kernel virtual machine)
- **rpc.***: ~21 metrics (remote procedure calls)
- **filesys.***: ~12 metrics (filesystem statistics)
- **vfs.***: ~7 metrics (virtual filesystem layer)
- **swap.***: ~7 metrics (swap space usage)

## Key Findings from PCP Metrics

The Performance Co-Pilot archives enabled detailed resource utilization analysis:

1. **Prefix Cache Hit Rates**: Consistent ~98.5% across all configurations, demonstrating excellent cache effectiveness

2. **KV-Cache Utilization**:
   - Normal range: 33-60% mean usage for most configs
   - Anomaly detected: 289% mean usage for 0.6B native-offload (correlates with -29.4% throughput degradation)

3. **Memory Efficiency**:
   - Process RSS: 1.4-1.9 GB across all configurations
   - 32B-AWQ quantized model uses less memory (1.4 GB) than smaller full-precision models
   - Minimal overhead from distributed caching (<10% variance)

4. **Request Queue Patterns**:
   - Running requests: 119-343 concurrent average
   - Waiting queue depth: 36-289 average (native-offload 0.6B shows pathological queueing)

## Analysis Tools

PCP archives were processed using:

- **pcp2arrow**: Convert time-series to Parquet format for pandas analysis
- **pmval**: Extract individual metric time series
- **pminfo**: Query metric metadata and availability
- **Python script** (`scripts/extract-pcp-metrics.py`): Automated extraction and visualization
  - Output: `analysis/pcp_metrics.csv`
  - Extracts: KV-cache utilization, GPU metrics, CPU usage, memory, request queues

## Archive Organization

Archives are organized by benchmark configuration:

```
results/
├── 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-0.6B_no-offload/pcp-archives/
├── 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-0.6B_native-offload/pcp-archives/
├── 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-0.6B_llm-d-redis/pcp-archives/
├── 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-0.6B_llm-d-valkey/pcp-archives/
├── ... (12 more configurations)
```

Each archive directory contains:
- `*.0.zst` - Compressed data archive
- `*.index.zst` - Compressed metric index
- `*.meta.zst` - Compressed metadata

## Future Analysis Opportunities

The comprehensive PCP archives enable additional analysis beyond the current report:

- **Network I/O patterns**: Analyze Redis/Valkey network traffic during distributed indexing
- **Disk I/O correlation**: Understand storage impact during CPU offloading
- **Per-process resource tracking**: Detailed vLLM worker process analysis
- **Filesystem cache effects**: XFS metadata operations during benchmark runs
- **IPC patterns**: Inter-process communication overhead in multi-GPU deployments

The archives provide a complete system-level view of performance characteristics, supporting both current analysis and future deep-dive investigations.

---

*PCP archives compressed with zstd, stored in `results/*/pcp-archives/` directories*
