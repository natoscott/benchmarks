# Process Improvements

**Session Duration**: ~2.5 hours (2025-10-13, 14:15-16:45 UTC)

**Key bottlenecks**: Understanding nested GuideLLM JSON format, discovering data gaps, matching up system metrics with benchmark results (timezone confusion), generate Parquet file(s) beforehard as its very resource intensive (Claude times it out).

---

## Data Collection

1. **Timezone confusion**: Always use UTC for cloud testing (data gaps from local/UTC mismatch)

2. Verify vLLM (openmetrics), GPU (nvidia/amd) metrics are enabled and logged

3. **Sampling interval**: Configure pmlogger for 1-second sampling for benchmarking (default is relatively coarse 10s or 60s, intended for 24x7 production systems)

## Claude Code Tooling

**Missing Tools**:
1. Native PCP query tool (vs Bash wrapper commands)
2. Parquet query tool (vs writing full Python scripts)
3. Inline visualization (vs save-then-describe)
4. JSON schema inspector (vs iterative print debugging)

**Workflow Issues**:
1. Reload Parquet data across multiple scripts (need session persistence)
2. Complete script rewrites (need incremental execution)
