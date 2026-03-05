#!/usr/bin/env python3
"""
Extract KV-cache allocation info from benchmark run directories.
Each benchmark run has logs that contain KV-cache allocation information.
"""

import subprocess
import re
from pathlib import Path
import pandas as pd

def find_pcp_archives():
    """Find all PCP archives in results directories."""
    results_dir = Path("results")
    archives = []

    for result_dir in results_dir.iterdir():
        if not result_dir.is_dir():
            continue

        # Look for PCP archives
        pcp_dir = result_dir / "pcp-archives"
        if pcp_dir.exists():
            for node_dir in pcp_dir.iterdir():
                if node_dir.is_dir():
                    # Decompress if needed
                    subprocess.run(f"cd {node_dir} && zstd -d -f *.zst 2>/dev/null",
                                 shell=True, capture_output=True)

                    meta_files = list(node_dir.glob("*.meta"))
                    if meta_files:
                        archive_path = str(meta_files[0]).replace(".meta", "")
                        archives.append((result_dir.name, archive_path))
                        break

    return archives

def extract_vllm_logs_from_pcp(archive_path):
    """Extract vLLM startup logs from PCP archive."""
    try:
        # Try to get vLLM logs from PCP using pmlogdump or similar
        # This is a placeholder - PCP doesn't directly store application logs
        # We'll need to use kubectl logs instead
        return None
    except Exception as e:
        return None

def parse_experiment_name(dirname):
    """Parse model, configuration, and other details from directory name."""
    # Format: 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_lmcache-local-20kcpu_replica1_rate50
    parts = dirname.split("_")

    model = None
    config = None

    for i, part in enumerate(parts):
        if part.startswith("Qwen"):
            model = part
        if any(x in part for x in ["offload", "lmcache", "llm-d"]):
            # Capture config and potentially next part
            config_parts = [part]
            if i + 1 < len(parts) and parts[i + 1].startswith("replica"):
                pass  # Don't include replica part
            elif i + 1 < len(parts) and not parts[i + 1].startswith("rate"):
                config_parts.append(parts[i + 1])
            config = "_".join(config_parts)

    return model, config

def extract_kvcache_from_kubectl_logs(pod_logs):
    """Extract KV-cache allocation info from vLLM logs."""
    if not pod_logs:
        return {}

    data = {}

    # Extract GPU KV-cache memory
    match = re.search(r"Available KV cache memory: ([\d.]+) GiB", pod_logs)
    if match:
        data['gpu_kv_memory_gb'] = float(match.group(1))

    # Extract GPU KV-cache tokens
    match = re.search(r"GPU KV cache size: ([\d,]+) tokens", pod_logs)
    if match:
        data['gpu_kv_tokens'] = int(match.group(1).replace(',', ''))

    # Extract CPU KV-cache blocks
    match = re.search(r"CPU KV cache size: ([\d,]+) blocks", pod_logs)
    if match:
        data['cpu_kv_blocks'] = int(match.group(1).replace(',', ''))

    # Extract model loading memory
    match = re.search(r"Loading model weights took ([\d.]+) GiB", pod_logs)
    if match:
        data['model_memory_gb'] = float(match.group(1))

    # Extract max concurrency
    match = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", pod_logs)
    if match:
        data['max_concurrency'] = float(match.group(1))

    return data

def main():
    print("=" * 80)
    print("KV-CACHE DATA EXTRACTION FROM EXISTING BENCHMARKS")
    print("=" * 80)
    print()
    print("This approach won't work - PCP doesn't store application logs.")
    print("We need a different strategy: manually document from known configurations.")
    print()
    print("Creating manual KV-cache allocation table from benchmark configurations...")
    print("=" * 80)

    # Manual data based on known configurations from benchmarks
    # This data can be filled in by examining a few representative runs

    data = []

    # We'll create placeholder entries for now
    # User will need to fill in actual values from pod logs

    models = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B-AWQ"]
    configs = [
        ("no-offload", 2, None, "GPU-only baseline"),
        ("native-offload-10k", 2, 10000, "vLLM native offload, 10K CPU blocks"),
        ("native-offload-20k", 2, 20000, "vLLM native offload, 20K CPU blocks"),
        ("lmcache-local-20kcpu", 2, 20000, "LMCache local CPU, ~58GB"),
    ]

    for model in models:
        for config_name, tp, cpu_blocks, desc in configs:
            data.append({
                'model': model,
                'configuration': config_name,
                'tp_size': tp,
                'cpu_kv_blocks': cpu_blocks if cpu_blocks else 0,
                'description': desc,
                'gpu_kv_memory_gb': 'TBD',
                'gpu_kv_tokens': 'TBD',
                'model_memory_gb': 'TBD',
                'max_concurrency': 'TBD',
            })

    df = pd.DataFrame(data)
    output_file = 'analysis/kvcache_allocations_template.csv'
    df.to_csv(output_file, index=False)

    print(f"Template saved to: {output_file}")
    print()
    print("NOTE: This is a template. Actual values need to be filled in manually")
    print("by examining pod logs from benchmark runs.")
    print("=" * 80)

if __name__ == '__main__':
    main()
