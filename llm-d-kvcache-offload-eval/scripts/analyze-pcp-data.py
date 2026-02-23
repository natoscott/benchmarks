#!/usr/bin/env python3
"""
Comprehensive PCP metrics analysis for llm-d benchmark results.
Extracts system, GPU, vLLM, and EPP metrics from PCP archives.
"""

import subprocess
import pandas as pd
from pathlib import Path
import json
import re
import numpy as np
from collections import defaultdict

RESULTS_DIR = Path('results')
ANALYSIS_DIR = Path('analysis')

# Key metrics to extract
METRICS = {
    'system': {
        'cpu_user': 'kernel.all.cpu.user',
        'cpu_sys': 'kernel.all.cpu.sys',
        'cpu_idle': 'kernel.all.cpu.idle',
        'mem_free': 'mem.freemem',
        'mem_used': 'mem.util.used',
    },
    'vllm': {
        'kv_cache_usage': 'openmetrics.vllm.vllm.kv_cache_usage_perc',
        'num_requests_running': 'openmetrics.vllm.vllm.num_requests_running',
        'num_requests_waiting': 'openmetrics.vllm.vllm.num_requests_waiting',
        'prefix_cache_hits': 'openmetrics.vllm.vllm.prefix_cache_hits_total',
        'prefix_cache_queries': 'openmetrics.vllm.vllm.prefix_cache_queries_total',
        'process_rss': 'openmetrics.vllm.process_resident_memory_bytes',
    },
    'gpu': {
        'gpu_util': 'openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL',
        'gpu_mem_used': 'openmetrics.dcgm.DCGM_FI_DEV_FB_USED',
        'gpu_mem_free': 'openmetrics.dcgm.DCGM_FI_DEV_FB_FREE',
        'gpu_power': 'openmetrics.dcgm.DCGM_FI_DEV_POWER_USAGE',
    }
}

def find_pcp_archive(result_dir):
    """Find PCP archive base name in result directory."""
    pcp_dir = result_dir / 'pcp-archives'
    if not pcp_dir.exists():
        return None

    # Find first node directory
    for node_dir in pcp_dir.iterdir():
        if node_dir.is_dir():
            # Find .meta.zst file (compressed) or .meta file
            meta_files = list(node_dir.glob('*.meta.zst')) or list(node_dir.glob('*.meta'))
            if meta_files:
                # Return base archive name (without extension)
                archive_base = str(meta_files[0]).replace('.meta.zst', '').replace('.meta', '')
                return archive_base
    return None

def extract_metric_stats(archive_path, metric_name):
    """Extract statistics for a metric using pmrep."""
    try:
        # Use pmrep with -z for archive timezone, -a for archive
        cmd = [
            'pmrep', '-z', '-a', archive_path,
            '-t', '10s',  # 10-second intervals
            '-o', 'csv',
            metric_name
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None

        # Parse CSV output
        lines = [l for l in result.stdout.strip().split('\n') if l and not l.startswith('Time')]
        if len(lines) < 2:  # Need at least header + data
            return None

        # Extract numeric values (skip N/A)
        values = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            for part in parts[1:]:  # Skip timestamp column
                try:
                    val = float(part.strip())
                    if not np.isnan(val):
                        values.append(val)
                except (ValueError, AttributeError):
                    continue

        if not values:
            return None

        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
        }
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Error extracting {metric_name}: {e}")
        return None

def parse_experiment_name(dir_name):
    """Parse experiment directory name to extract metadata."""
    # Format: 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_llm-d-redis_replica1_rate50
    parts = dir_name.split('_')

    model = None
    scenario = None
    rate = None

    for i, part in enumerate(parts):
        if 'Qwen' in part:
            model = part
        if part.startswith('rate'):
            rate = int(part.replace('rate', ''))
        # Detect scenario
        if 'no-offload' in dir_name:
            scenario = 'no-offload'
        elif 'native-offload' in dir_name:
            scenario = 'native-offload'
        elif 'lmcache-local-20kcpu' in dir_name:
            scenario = 'lmcache-local-20kcpu'
        elif 'lmcache-local' in dir_name:
            scenario = 'lmcache-local'
        elif 'lmcache-redis' in dir_name:
            scenario = 'lmcache-redis'
        elif 'lmcache-valkey-20kcpu' in dir_name:
            scenario = 'lmcache-valkey-20kcpu'
        elif 'lmcache-valkey' in dir_name:
            scenario = 'lmcache-valkey'
        elif 'llm-d-redis' in dir_name:
            scenario = 'llm-d-redis'
        elif 'llm-d-valkey' in dir_name:
            scenario = 'llm-d-valkey'

    return model, scenario, rate

def main():
    """Main analysis function."""
    print("Extracting PCP metrics from benchmark results...")
    print("=" * 80)

    results = []

    # Iterate through all result directories
    for result_dir in sorted(RESULTS_DIR.iterdir()):
        if not result_dir.is_dir():
            continue

        model, scenario, rate = parse_experiment_name(result_dir.name)
        if not all([model, scenario, rate]):
            continue

        print(f"\nProcessing: {model} / {scenario} / rate={rate}")

        # Find PCP archive
        archive_path = find_pcp_archive(result_dir)
        if not archive_path:
            print("  No PCP archive found")
            continue

        print(f"  Archive: {Path(archive_path).name}")

        # Extract metrics
        row = {
            'model': model,
            'scenario': scenario,
            'rate': rate,
        }

        # Extract each metric category
        for category, metrics in METRICS.items():
            for metric_key, metric_name in metrics.items():
                stats = extract_metric_stats(archive_path, metric_name)
                if stats:
                    for stat_name, stat_val in stats.items():
                        row[f'{metric_key}_{stat_name}'] = stat_val
                    print(f"    ✓ {metric_key}: mean={stats['mean']:.2f}")
                else:
                    print(f"    ✗ {metric_key}: no data")

        results.append(row)

    # Create DataFrame
    print("\n" + "=" * 80)
    print(f"Extracted metrics from {len(results)} benchmark runs")

    if results:
        df = pd.DataFrame(results)

        # Save to CSV
        ANALYSIS_DIR.mkdir(exist_ok=True)
        output_file = ANALYSIS_DIR / 'pcp_metrics_comprehensive.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")

        # Print summary statistics
        print("\nSummary by scenario:")
        print(df.groupby('scenario').agg({
            'kv_cache_usage_mean': 'mean',
            'num_requests_running_mean': 'mean',
            'cpu_idle_mean': 'mean',
            'process_rss_mean': lambda x: x.mean() / 1e9  # Convert to GB
        }).round(2))

    print("\nDone!")

if __name__ == '__main__':
    main()
