#!/usr/bin/env python3
"""Extract KV-cache allocation information from vLLM v0.5.0 startup logs."""

import os
import re
import subprocess
import csv
from pathlib import Path

def extract_kvcache_info(log_file):
    """Extract KV-cache information from a vLLM startup log."""
    try:
        # Decompress and read log file
        result = subprocess.run(
            ['zstdcat', log_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        log_content = result.stdout

        info = {
            'log_file': str(log_file),
            'model': None,
            'gpu_kv_cache_gib': None,
            'gpu_kv_cache_tokens': None,
            'cpu_bytes_configured': None,
            'cpu_blocks_allocated': None,
            'cpu_kv_cache_gib': None,
        }

        # Extract model name from log file path
        # Format: 1x2xL40S_upstream-llm-d-0.5.0_Qwen3-14B_native-offload-10k_replica1_rate50
        path_parts = log_file.parent.name.split('_')
        if len(path_parts) >= 3:
            info['model'] = path_parts[2]  # Qwen3-14B
            info['configuration'] = path_parts[3]  # native-offload-10k, no-offload, etc

        # Extract GPU KV cache memory
        # Format: Available KV cache memory: 20.6 GiB
        gpu_mem_match = re.search(r'Available KV cache memory: ([\d.]+) GiB', log_content)
        if gpu_mem_match:
            info['gpu_kv_cache_gib'] = float(gpu_mem_match.group(1))

        # Extract GPU KV cache size in tokens
        # Format: GPU KV cache size: 269,968 tokens
        gpu_tokens_match = re.search(r'GPU KV cache size: ([\d,]+) tokens', log_content)
        if gpu_tokens_match:
            info['gpu_kv_cache_tokens'] = int(gpu_tokens_match.group(1).replace(',', ''))

        # Extract CPU bytes configured
        # Format: 'cpu_bytes_to_use': 22097606737
        cpu_bytes_match = re.search(r"'cpu_bytes_to_use': (\d+)", log_content)
        if cpu_bytes_match:
            cpu_bytes = int(cpu_bytes_match.group(1))
            info['cpu_bytes_configured'] = cpu_bytes
            info['cpu_kv_cache_gib'] = cpu_bytes / (1024**3)  # Convert to GiB

        # Extract CPU blocks allocated
        # Format: Allocating a cross layer KV cache of shape (16873, 40, 2, 16, 4, 128)
        cpu_blocks_match = re.search(r'Allocating a cross layer KV cache of shape \((\d+),', log_content)
        if cpu_blocks_match:
            info['cpu_blocks_allocated'] = int(cpu_blocks_match.group(1))

        return info

    except Exception as e:
        print(f"Error processing {log_file}: {e}")
        return None

def main():
    # Find all v0.5.0 vLLM startup logs
    results_dir = Path('results')
    log_files = sorted(results_dir.glob('1x2xL40S_upstream-llm-d-0.5.0_*/vllm-startup.log.zst'))

    print(f"Found {len(log_files)} v0.5.0 vLLM startup logs")

    # Extract KV-cache info from all logs
    all_info = []
    for log_file in log_files:
        info = extract_kvcache_info(log_file)
        if info:
            all_info.append(info)

    # Write to CSV
    output_file = 'analysis/v0.5.0_kvcache_allocations.csv'
    os.makedirs('analysis', exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['log_file', 'model', 'configuration', 'gpu_kv_cache_gib', 'gpu_kv_cache_tokens',
                     'cpu_bytes_configured', 'cpu_blocks_allocated', 'cpu_kv_cache_gib']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_info)

    print(f"\nWrote {len(all_info)} entries to {output_file}")

    # Print summary statistics
    print("\n=== Summary by Model and Configuration ===")
    by_model_config = {}
    for info in all_info:
        key = (info['model'], info['configuration'])
        if key not in by_model_config:
            by_model_config[key] = []
        by_model_config[key].append(info)

    for (model, config), infos in sorted(by_model_config.items()):
        # Take first entry as representative (all should be same for same model/config)
        rep = infos[0]
        print(f"\n{model} - {config}:")
        print(f"  GPU KV cache: {rep['gpu_kv_cache_gib']:.2f} GiB ({rep['gpu_kv_cache_tokens']:,} tokens)")
        if rep['cpu_bytes_configured']:
            print(f"  CPU bytes configured: {rep['cpu_kv_cache_gib']:.2f} GiB")
            print(f"  CPU blocks allocated: {rep['cpu_blocks_allocated']:,}")

if __name__ == '__main__':
    main()
