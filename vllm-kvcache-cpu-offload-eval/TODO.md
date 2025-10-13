# TODO: Process Improvements

**Session Duration**: ~2.5 hours (2025-10-13, 14:15-16:45 UTC)

**Deliverables**: REPORT.md, 5 analysis scripts, 3 CSV files, 1 visualization

**Key bottlenecks**: Understanding nested GuideLLM JSON format, discovering data gaps late

---

## Data Collection Issues

1. **Timezone confusion**: Always use UTC for cloud testing (data gaps from local/UTC mismatch)

2. **PCP recording gaps**: Ensure continuous logging throughout benchmark session

3. **Missing metrics**: Verify vLLM (openmetrics), GPU (nvidia/amd) metrics are enabled and logged

4. **Sampling interval**: Configure pmlogger for 1-second sampling (default may be too slow)

5. **Late validation**: Data gaps discovered during analysis; need pre-flight checks

## Claude Code Tool Improvements

**Missing Tools**:
1. Native PCP query tool (vs Bash wrapper commands)
2. Parquet query tool (vs writing full Python scripts)
3. Inline visualization (vs save-then-describe)
4. JSON schema inspector (vs iterative print debugging)

**Workflow Issues**:
1. Reload Parquet data across multiple scripts (need session persistence)
2. Complete script rewrites (need incremental execution)
3. Late structure discovery (need early validation)

**What Worked**:
- Iterative refinement
- Breaking into manageable steps
- Proactive error handling

---

*Session date: 2025-10-13*
