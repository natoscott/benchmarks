# Performance Benchmarks

This repository contains benchmark results and analysis for vLLM and llm-d KV cache
offload evaluations on NVIDIA GPU hardware.

## Repository Structure

```
benchmarks/
├── llm-d-kvcache-offload-eval/   # llm-d v0.4.0–v0.5.1 KV cache offload (L40S)
├── rhoai-kvcache-offload-eval/   # RHOAI 3.3 KV cache offload (H200)
├── vllm-kvcache-cpu-offload-eval/# vLLM upstream CPU offload benchmarks
└── scripts/                      # Shared analysis utilities
```

## Git LFS

This repository uses [Git Large File Storage (LFS)](https://git-lfs.github.com) for
benchmark result archives and figures. The following file types are tracked via LFS:

| Pattern | Contents |
|---------|----------|
| `*.zst` | Compressed GuideLLM JSON results, PCP metric archives, vLLM startup logs |
| `*.png` | Analysis figures |
| `*.zstd`| Alternative zstd extension |
| `*.xz`  | XZ-compressed archives |
| `*.parquet` | Columnar metric data |

### Cloning

Git LFS must be installed before cloning. Without it, LFS-tracked files will appear
as small text pointer files rather than their actual content.

**Install git-lfs:**
```bash
# Fedora / RHEL
sudo dnf install git-lfs

# Ubuntu / Debian
sudo apt install git-lfs

# macOS (Homebrew)
brew install git-lfs

# After installing, initialise once per user account:
git lfs install
```

**Clone with LFS content:**
```bash
git clone git@github.com:natoscott/benchmarks.git
cd benchmarks
# LFS files are downloaded automatically on clone if git-lfs is installed.
```

**If you cloned without git-lfs installed** (files are pointer text):
```bash
git lfs install
git lfs pull
```

**Partial checkout** (skip LFS downloads, work with pointers only):
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:natoscott/benchmarks.git
# Later, fetch only what you need:
git lfs pull --include="rhoai-kvcache-offload-eval/results/*/guidellm-results.json.zst"
```

### Working with LFS files

Compressed result files can be read directly without extracting:
```bash
# Decompress a single result to stdout
zstd -d -q -c results/1x8xH200_.../guidellm-results.json.zst | python3 -m json.tool

# Run analysis (script handles decompression internally)
python3 scripts/analyze-results.py
```

PCP archives require decompression to a temporary directory before use with PCP tools:
```bash
tmpdir=$(mktemp -d)
for f in results/.../pcp-archives/*/*.zst; do
    zstd -d -q -c "$f" > "$tmpdir/$(basename ${f%.zst})"
done
pmrep -a "$tmpdir/<archive-name>" openmetrics.vllm.vllm.kv_cache_usage_perc
rm -rf "$tmpdir"
```

## Benchmark Scenarios

### rhoai-kvcache-offload-eval

RHOAI 3.3 (vLLM 0.13.0+rhai11) on 1×8×H200. Evaluates `OffloadingConnector`
CPU KV cache offload across three workload profiles, three models, and three
replica counts. See [rhoai-kvcache-offload-eval/REPORT.md](rhoai-kvcache-offload-eval/REPORT.md).

### llm-d-kvcache-offload-eval

llm-d v0.5.1 on 2×L40S. Evaluates CPU offload, filesystem offload, and
MultiConnector (CPU+filesystem) configurations. See
[llm-d-kvcache-offload-eval/REPORT.md](llm-d-kvcache-offload-eval/REPORT.md).

## Data Collection

- **Benchmarks**: [GuideLLM](https://github.com/vllm-project/guidellm)
- **Metrics**: [PCP](https://pcp.io) (Performance Co-Pilot) Linux kernel and hardware metrics, plus openmetrics PMDA
  captures all vLLM Prometheus metrics, DCGM GPU metrics, and EPP
  scheduler metrics at 10-second intervals during each benchmark run.
- **Results**: Each run directory contains `guidellm-results.json.zst`,
  `vllm-startup.log.zst`, `pcp-archives/`, and `benchmark-config.txt`.
