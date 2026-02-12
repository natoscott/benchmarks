# LLM-D KV-Cache Offload Evaluation

Benchmark suite for evaluating KV-cache offloading and distributed caching strategies in llm-d deployments.

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
│   ├── run-benchmark.sh    # Main benchmark automation script
│   └── run-all.sh          # Wrapper to run all test configurations
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
└── docs/                   # Additional documentation
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

This will run 21 benchmarks (7 configurations × 3 models):
- Configurations: no-offload, native-offload, lmcache-local, lmcache-redis, lmcache-valkey, llm-d-redis, llm-d-valkey
- Models: Qwen3-0.6B, Qwen3-7B, Qwen3-14B

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

Benchmarks generate:
1. **guidellm JSON reports**: Detailed benchmark results with request/response stats
2. **PCP archives**: Compressed system and application metrics (`.xz` format)

Both are saved with timestamps in the results directory:
```
results/
├── benchmark_YYYYMMDD_HHMMSS_<parameters>_guidellm.json
└── benchmark_YYYYMMDD_HHMMSS_<parameters>_pcp.tar.xz
```

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
