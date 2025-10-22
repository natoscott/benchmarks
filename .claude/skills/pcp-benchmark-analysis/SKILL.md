# PCP Benchmark Data Analysis

Performance Co-Pilot (PCP) provides system-level performance monitoring for benchmark analysis. This skill guides analysis of PCP archives for benchmarking workloads.

## Critical Rules

1. **Always use UTC**: PCP archives and benchmark timestamps should be in UTC to avoid timezone confusion.  This is common for cloud instances anyway.
2. **Validate coverage**: Check that PCP data exists for benchmark time ranges before deeper analysis
3. **Prefer pmrep over pmdumptext**: pmrep is a more capable python utility
4. **Convert JSON benchmark results to PCP archives**: Use vllmbench2pcp, guidellm2pcp, or create new converters with PMI
5. **Merge before Parquet**: Combine benchmark and system archives with pmlogextract, then convert to Parquet using pcp2arrow
6. **Set adequate timeouts**: pcp2arrow can take significant time for large archives

## Benchmark Data Integration Workflow

When benchmark results are in JSON format, integrate them into PCP archives:

```bash
# 1. Convert JSON benchmark results to PCP archive
guidellm2pcp benchmark-results.json benchmark-metrics

# Or for vllm bench:
vllmbench2pcp vllm-results.json vllm-metrics

# 2. Set hostname to match system metrics (for single-system benchmarks)
# This ensures proper alignment when merging

# 3. Merge benchmark metrics with system metrics
pmlogextract system-metrics benchmark-metrics unified-archive

# 4. Convert unified archive to Parquet
# NOTE: This can take significant time for large archives
# Set timeout appropriately (e.g., 120000ms = 2 minutes)
pcp2arrow -a unified-archive -o unified-data.parquet
```

**Result**: Single timeseries view with benchmark results and system metrics aligned by timestamp.

### Creating New Benchmark Converters

If no converter exists for your benchmark tool, create one using PCP's PMI (Performance Metrics Interface) module.  Use guidellm2pcp and vllmbench2pcp as references.

## Pre-Analysis Validation

Before analyzing PCP data, verify coverage:

```bash
# Check archive time range
pmlogdump -l archive-name

# Verify key metrics exist
pminfo -a archive-name mem.util       # Memory metrics
pminfo -a archive-name network.interface # Networking
pminfo -a archive-name kernel.all.cpu  # CPU metrics
pminfo -a archive-name nvidia amdgpu  # GPU metrics
pminfo -a archive-name hinv   # Hardware inventory

# Check specific time window has data
pmval -a archive-name -S @START_TIME -T @END_TIME nvidia.memused
```

## Data Extraction Workflow

1. **Merge archives first** (if separate benchmark and system archives):
```bash
pmlogextract system-archive benchmark-archive unified-archive
```

2. **Convert unified archive to Parquet** for pandas analysis:
```bash
# WARNING: pcp2arrow can take considerable time for large, compressed archives
# When using via Bash tool, set timeout appropriately (decompressed archives):
#   - Small archives (<100MB): 60000ms (1 min)
#   - Medium archives (100MB-1GB): 120000ms (2 min)
#   - Large archives (>1GB): 300000ms+ (5+ min)
pcp2arrow -a unified-archive -o data.parquet
```

3. **Query metrics** using pmrep (not pmdumptext):
```bash
# Time-series data with UTC timestamps
pmrep -a archive-name -t 1sec -z UTC metric.name

# Summary statistics
pmlogsummary -I -N archive-name metric.name
```

4. **Explore metrics hierarchy**:
```bash
# Top-level metric groups
pminfo -a archive-name | awk -F. '{print $1}' | sort -u

# Drill down into specific group search for metrics
pminfo -a archive-name | grep metric-pattern

# Metric labels provide additional information
pminfo -a archive-name -l hinv.ncpu | sed -e 's/ *labels //g' -e '/^hinv/d' | jq
```

## Key Metrics for Benchmarks

**Hardware Inventory**:
- `hinv.*` (hardware configuration, CPU/GPU counts, memory capacity)

**GPU Metrics**:
- `nvidia.*` (NVIDIA GPUs)
- `amdgpu.*` (AMD GPUs)

**vLLM/llm-d Metrics**:
- `openmetrics.*` (all Prometheus metrics)

**System Metrics**:
- `mem.util.*` (memory utilisation)
- `kernel.all.cpu.*` (CPU utilisation)
- `kernel.all.pressure.*` (PSI - pressure stall information)

**I/O and Network**:
- `disk.dev.*` (disk I/O per device)
- `network.interface.*` (network I/O per interface)
- `infiniband.*` (InfiniBand fabric metrics)
- `rocestat.*` (RoCE statistics)

**Benchmark Results** (after conversion to PCP):
- `guidellm.*` (throughput, latency, requests)
- `vllmbench.*` (vllm bench results)

## Best Practices

❌ **Don't skip merging**: Benchmark and system metrics should be in one archive
❌ **Don't ignore timezones**: Convert everything to UTC if not already
❌ **Don't skip coverage checks**: Gaps discovered during analysis waste time
❌ **Don't use default timeout for pcp2arrow**: Large archives need extended timeouts

✅ **Do convert JSON to PCP**: Use guidellm2pcp, vllmbench2pcp, or create PMI converter
✅ **Do merge archives**: Use pmlogextract to combine benchmark + system metrics
✅ **Do set adequate timeouts**: pcp2arrow can take 2+ minutes for large archives
✅ **Do validate first**: Check data coverage before analysis
✅ **Do use pmrep**: Modern python replacement for pmdumptext
✅ **Do consider labels**: Check labels on metrics too, `pminfo --labels` option

When capturing PCP data for benchmarks:
- Install the pcp-zeroconf package on all systems under test
- High resolution sampling: PMLOGGER_INTERVAL=1 in /usr/share/pcp/zeroconf/pmlogger
- Use system loggers for auto-configuration, continual logging and active compression
- Enable benchmark-relevant metrics (nvidia, amdgpu, openmetrics, infiniband, rocestat)
- Convert JSON benchmark results to PCP archives with matching hostname
- Merge all archives before analysis
- Budget adequate time for pcp2arrow conversion of large archives

## Analysis Pattern

```python
import pandas as pd

# Load unified Parquet data (benchmark + system metrics merged)
df = pd.read_parquet('unified-data.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Validate time range
print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Now can correlate benchmark metrics with system metrics
# Example: GPU usage during high throughput periods
high_throughput = df[df['guidellm.throughput'] > threshold]
print(f"GPU usage during high throughput: {high_throughput['nvidia.gpuactive'].mean()}")
```

Good visuals are obtained from dataframes with seaborn or matplotlib modules, and are crucial in helping to explain benchmark results.  Compare different runs to each other using summaries of time windows within the dataframes.
