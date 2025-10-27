# PCP Archive Summary

## Archive Details

| Archive | Models | Duration | Size | Metrics | Samples/Metric |
|---------|--------|----------|------|---------|----------------|
| pcp-archive-20251023 | Qwen3-0.6B, 8B | 9.3 hours | 49M | 1,632 | ~33,600 |
| pcp-archive-20251026 | Qwen3-14B | 55 minutes | 5.3M | 1,632 | ~3,300 |

## Combined Statistics

- **Total recording time**: ~10.2 hours
- **Combined storage**: 54.3M
- **Metrics captured**: 1,632 unique metrics per archive
- **Sample frequency**: 1 second (default for most metrics)
- **Total benchmark runs**: 60 configurations
  - Archive 1: 40 runs (2 models × 2 configs × 10 concurrency levels)
  - Archive 2: 20 runs (1 model × 2 configs × 10 concurrency levels)

## Metric Groups

### Utilized in Report
- **guidellm.***: 70 metrics (benchmark results)
- **nvidia.***: 23 metrics (GPU hardware)
- **openmetrics.vllm.***: 113 metrics (vLLM Prometheus exporters)

### Available but Unused
- **network.***: 378 metrics (interface statistics)
- **mem.***: 340 metrics (memory details)
- **disk.***: 132 metrics (I/O and storage)
- **kernel.***: 80 metrics (Linux kernel)
- **proc.***: 56 metrics (process-level)

The archives contain extensive system-level performance data beyond what was analyzed in the report, providing opportunities for deeper analysis of resource consumption patterns, network activity, and process-level behaviors during benchmarking.

---

*PCP archives available in git at https://git.new/J5Cx50P*
