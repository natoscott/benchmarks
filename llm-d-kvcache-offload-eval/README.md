# LLM-D KV-Cache Offload Evaluation

Benchmark suite for evaluating KV-cache offloading and distributed caching strategies in llm-d deployments. See [REPORT.md](REPORT.md) for comprehensive evaluation results and analysis.

## Overview

This repository contains automation scripts and Kubernetes manifests for benchmarking various KV-cache offloading configurations:

- **No offloading**: Baseline vLLM deployment without KV-cache offloading
- **Native vLLM offloading**: Using vLLM's built-in OffloadingConnector
- **LMCache (local)**: LMCache with local CPU backend
- **LMCache (Redis)**: LMCache with distributed Redis backend
- **LMCache (Valkey)**: LMCache with distributed Valkey backend
- **llm-d (Redis)**: llm-d EPP with Redis-based KV-block indexing
- **llm-d (Valkey)**: llm-d EPP with Valkey-based KV-block indexing

## Directory Structure

```
llm-d-kvcache-offload-eval/
├── scripts/
│   ├── run-benchmark.sh                      # Main benchmark automation script
│   ├── run-all.sh                            # Run all baseline configurations
│   ├── run-increased-cpu-memory.sh           # Benchmark for increased CPU offload capacity
│   ├── run-multi-replica.sh                  # Multi-replica distributed cache testing
│   ├── comprehensive-analysis.py             # Primary GuideLLM analysis with visualizations
│   ├── extract-pcp-peak-metrics.py           # Extract PCP metrics at peak throughput
│   ├── create-pcp-visualizations.py          # Generate PCP metric visualizations
│   ├── extract-pcp-cpu-memory-analysis.py    # CPU utilization and memory pressure analysis
│   ├── analyze-per-cpu-utilization.py        # Per-CPU saturation detection
│   └── visualize-percpu-saturation.py        # Per-CPU saturation visualizations
├── manifests/
│   ├── valkey-deployment.yaml
│   ├── llm-d-model-cache-pvc.yaml
│   ├── llm-d-model-server-deployment.yaml
│   ├── epp-deployment.yaml
│   ├── interactive-pod-deployment.yaml
│   ├── pcp-daemonset.yaml
│   ├── openmetrics-pmda-configmap.yaml
│   ├── openmetrics-pmlogconf-configmap.yaml
│   ├── redis-pmda-configmap.yaml
│   └── pcp-pmlogconf/
│       ├── vllm            # PCP pmlogconf for vLLM metrics
│       ├── epp             # PCP pmlogconf for EPP metrics
│       ├── istio           # PCP pmlogconf for Istio metrics
│       └── dcgm            # PCP pmlogconf for DCGM GPU metrics
├── results/                # Benchmark results (gitignored)
├── analysis/               # Generated analysis outputs (CSV, PNG)
├── archive/                # Archived obsolete scripts
├── REPORT.md               # Comprehensive benchmark evaluation report
└── PCPLOGS.md              # PCP archive documentation
```

## Prerequisites

1. Kubernetes/OpenShift cluster with:
   - llm-d installed (Gateway API + Inference Extension)
   - GPU nodes with NVIDIA GPUs
   - Storage class for PVCs (e.g., `ibmc-vpc-block-10iops-tier`)

2. Required secrets:
   - `llm-d-hf-token`: HuggingFace token for model downloads

3. Deployed components:
   - llm-d InferencePool with EPP
   - PCP DaemonSet for metrics collection
   - Interactive pod for running benchmarks
   - Redis or Valkey deployment (for distributed caching tests)

## Quick Start

### Run a single benchmark

```bash
cd scripts/

# Set required environment variables
export KUBECONFIG=/path/to/kubeconfig
export NAMESPACE=llm-d-pfc-cpu
export MODEL=Qwen/Qwen3-0.6B

# Run baseline (no offloading)
bash run-benchmark.sh

# Run with native vLLM offloading
export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
bash run-benchmark.sh

# Run with llm-d Redis indexing
export EPP_BACKEND_CONFIG=redis
bash run-benchmark.sh
```

### Run complete test matrix

```bash
cd scripts/

# Configure base parameters
export KUBECONFIG=/path/to/kubeconfig
export NAMESPACE=llm-d-pfc-cpu

# Run all configurations across all models
bash run-all.sh
```

This will run benchmarks across multiple configurations and models:
- Configurations: no-offload, native-offload, lmcache-local, lmcache-redis, lmcache-valkey, llm-d-redis, llm-d-valkey
- Models: Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B-AWQ
- Concurrency levels: 1, 50, 100, 150, 300, 400, 500, 650

## Configuration Parameters

### run-benchmark.sh

Key environment variables:

```bash
# Cluster configuration
KUBECONFIG              # Path to kubeconfig file
NAMESPACE               # Kubernetes namespace (default: llm-d-pfc-cpu)

# Model configuration
MODEL                   # HuggingFace model ID (default: Qwen/Qwen3-0.6B)
TENSOR_PARALLEL_SIZE    # TP size (default: 1)
MAX_NUM_SEQ             # Max concurrent sequences (default: 1024)

# vLLM configuration
VLLM_EXTRA_ARGS         # Additional vLLM arguments (for KV-transfer-config)
VLLM_ENV_VARS           # Environment variables for vLLM pod (e.g., LMCACHE_REMOTE_URL)

# EPP configuration
EPP_BACKEND_CONFIG      # EPP indexer backend: in-memory, redis, valkey
EPP_CONFIGMAP           # EPP ConfigMap name (default: llm-d-infpool-epp)
EPP_DEPLOYMENT          # EPP Deployment name (default: llm-d-infpool-epp)

# Benchmark configuration
PARAMETERS              # Test name/label (default: default)
MAX_REQUESTS            # Number of benchmark requests (default: 10)
RATE                    # Request rate limit (default: null/unlimited)
```

### EPP Backend Configuration

When `EPP_BACKEND_CONFIG` is set to `redis` or `valkey`, the script will:
1. Update the EPP ConfigMap with appropriate indexerConfig
2. Restart the EPP deployment
3. Wait for EPP to be ready

Example EPP configuration (Redis):
```yaml
parameters:
  indexerConfig:
    kvBlockIndexConfig:
      redisConfig:
        address: "redis://redis.llm-d-pfc-cpu.svc.cluster.local:6379"
      enableMetrics: true
      metricsLoggingInterval: "1m0s"
```

## PCP Metrics Collection

The PCP DaemonSet collects metrics from:
- **vLLM**: OpenMetrics from model server pods
- **EPP**: OpenMetrics from EPP pods
- **Istio**: Envoy proxy metrics
- **DCGM**: NVIDIA GPU metrics

Metrics are logged to compressed PCP archives on each node at `/var/log/pcp/pmlogger/`.

Configure pmlogconf files in `manifests/pcp-pmlogconf/` to force metric collection intervals.

## Benchmark Outputs

Each benchmark run generates:
1. **guidellm JSON reports**: Detailed benchmark results with throughput, latency, and request/response statistics
2. **PCP archives**: Compressed system and application metrics from Performance Co-Pilot

Results are organized by experiment name:
```
results/<experiment-name>/
├── guidellm-results.json.zst    # Compressed benchmark results
├── benchmark-config.txt          # Run configuration parameters
└── pcp-archives/                 # Compressed PCP metric archives
    └── <node-name>/
        ├── YYYYMMDD.HH.MM.SS.0.zst
        ├── YYYYMMDD.HH.MM.SS.index.zst
        └── YYYYMMDD.HH.MM.SS.meta.zst
```

## Analysis

After completing benchmark runs, use the analysis scripts to generate insights:

### Comprehensive Analysis

```bash
# Extract metrics from all benchmark runs and generate visualizations
python3 scripts/comprehensive-analysis.py
```

This generates:
- `analysis/complete_metrics.csv` - All extracted metrics
- `analysis/peak_throughput_all.csv` - Peak performance summary
- `analysis/peak_throughput_all_scenarios.png` - Bar chart comparison
- `analysis/performance_delta_heatmap_all.png` - Performance delta heatmap
- `analysis/throughput_curve_*.png` - Throughput vs concurrency curves
- `analysis/latency_comparison_all.png` - TTFT and TPOT comparison

### PCP Metrics Analysis

```bash
# Extract PCP metrics at peak throughput (rate=50)
python3 scripts/extract-pcp-peak-metrics.py

# Create PCP visualizations
python3 scripts/create-pcp-visualizations.py

# Deep dive into CPU and memory pressure
python3 scripts/extract-pcp-cpu-memory-analysis.py

# Analyze per-CPU utilization patterns
python3 scripts/analyze-per-cpu-utilization.py

# Create per-CPU saturation visualizations
python3 scripts/visualize-percpu-saturation.py
```

This generates:
- `analysis/pcp_metrics_peak.csv` - PCP metrics at peak throughput
- `analysis/pcp_cpu_memory_analysis.csv` - CPU utilization and pressure stall metrics
- `analysis/percpu_analysis.csv` - Per-CPU saturation analysis
- `analysis/pcp_gpu_vs_throughput.png` - GPU utilization correlation
- `analysis/pcp_kv_cache_usage.png` - KV-cache utilization patterns
- `analysis/pcp_memory_usage.png` - Process memory consumption
- `analysis/pcp_request_queues.png` - Request queue dynamics
- `analysis/pcp_prefix_cache_hits.png` - Prefix cache effectiveness
- `analysis/pcp_correlation_heatmap.png` - Metric correlation matrix
- `analysis/percpu_saturation_*.png` - Per-CPU saturation visualizations

### Results Report

See [REPORT.md](REPORT.md) for a comprehensive evaluation of KV-cache management strategies with detailed analysis, visualizations, and insights.

## Model Cache PVC

The `llm-d-model-cache` PVC is shared between:
- **llm-d-model-server**: Mounted at `/data` for model weights
- **interactive-pod**: Mounted at `/models` for pre-downloading models

This avoids re-downloading models on every deployment change.

## Notes

- Single-node clusters should use `Recreate` deployment strategy for model-server to avoid resource conflicts
- HF_TOKEN secret is required for downloading gated models
- PCP archives are automatically compressed to save disk space
- Valkey backend is Redis-compatible (BSD license) and may support RDMA in future

## References

- [llm-d Documentation](https://github.com/kubernetes-sigs/llm-d)
- [vLLM KV Transfer](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)
- [LMCache](https://github.com/LMCache/LMCache)
- [Performance Co-Pilot](https://pcp.io/)
