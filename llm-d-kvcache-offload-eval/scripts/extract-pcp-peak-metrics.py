#!/usr/bin/env python3
"""
Fast PCP metrics extraction focusing on peak throughput scenarios (rate=50).
Extracts key metrics to correlate with GuideLLM benchmark results.
"""

import subprocess
import pandas as pd
from pathlib import Path
import json
import numpy as np

RESULTS_DIR = Path('results')
ANALYSIS_DIR = Path('analysis')

# Focus on rate=50 (peak throughput for most models)
TARGET_RATE = 50

# Key metrics
METRICS = [
    ('cpu_idle', 'kernel.all.cpu.idle'),
    ('cpu_user', 'kernel.all.cpu.user'),
    ('mem_used_gb', 'mem.util.used'),
    ('kv_cache_pct', 'openmetrics.vllm.vllm.kv_cache_usage_perc'),
    ('requests_running', 'openmetrics.vllm.vllm.num_requests_running'),
    ('requests_waiting', 'openmetrics.vllm.vllm.num_requests_waiting'),
    ('prefix_hits_rate', 'openmetrics.vllm.vllm.prefix_cache_hits_total'),
    ('prefix_queries_rate', 'openmetrics.vllm.vllm.prefix_cache_queries_total'),
    ('process_rss_gb', 'openmetrics.vllm.process_resident_memory_bytes'),
    ('gpu_util', 'openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL'),
]

def find_pcp_archive(result_dir):
    """Find PCP archive in result directory."""
    pcp_dir = result_dir / 'pcp-archives'
    if not pcp_dir.exists():
        return None

    for node_dir in pcp_dir.iterdir():
        if node_dir.is_dir():
            archives = list(node_dir.glob('*.0.zst')) or list(node_dir.glob('*.0'))
            if archives:
                return str(archives[0]).replace('.zst', '')
    return None

def extract_metric(archive, metric_name):
    """Extract metric statistics using pmrep."""
    try:
        cmd = ['pmrep', '-z', '-a', archive, '-t', '10s', '-o', 'csv', metric_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if result.returncode != 0:
            return None

        lines = [l for l in result.stdout.strip().split('\n') if l and not l.startswith('Time')]
        if len(lines) < 2:
            return None

        values = []
        for line in lines[1:]:
            parts = line.split(',')
            for part in parts[1:]:
                try:
                    val = float(part.strip())
                    if not np.isnan(val):
                        values.append(val)
                except:
                    pass

        if not values:
            return None

        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
        }
    except Exception:
        return None

def parse_dir_name(dir_name):
    """Parse model, scenario, rate from directory name."""
    if f'rate{TARGET_RATE}' not in dir_name and 'rate050' not in dir_name:
        return None, None, None

    model = None
    scenario = None

    for part in ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']:
        if part in dir_name:
            model = part
            break

    if 'no-offload' in dir_name:
        scenario = 'no-offload'
    elif 'native-offload-20kcpu' in dir_name:
        scenario = 'native-offload-20kcpu'
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

    return model, scenario, TARGET_RATE

def main():
    print("Extracting PCP metrics at peak throughput (rate=50)...")
    print("=" * 80)

    results = []

    for result_dir in sorted(RESULTS_DIR.iterdir()):
        if not result_dir.is_dir():
            continue

        model, scenario, rate = parse_dir_name(result_dir.name)
        if not all([model, scenario, rate]):
            continue

        print(f"\n{model} / {scenario}")

        archive = find_pcp_archive(result_dir)
        if not archive:
            print("  No PCP archive")
            continue

        row = {'model': model, 'scenario': scenario, 'rate': rate}

        for metric_key, metric_name in METRICS:
            stats = extract_metric(archive, metric_name)
            if stats:
                row[f'{metric_key}_mean'] = stats['mean']
                row[f'{metric_key}_median'] = stats['median']
                row[f'{metric_key}_p95'] = stats['p95']
                print(f"  ✓ {metric_key}: {stats['mean']:.2f}")
            else:
                print(f"  ✗ {metric_key}")

        # Convert bytes to GB for memory metrics
        if 'process_rss_gb_mean' in row:
            row['process_rss_gb_mean'] /= 1e9
            row['process_rss_gb_median'] /= 1e9
            row['process_rss_gb_p95'] /= 1e9
        if 'mem_used_gb_mean' in row:
            row['mem_used_gb_mean'] /= 1e9
            row['mem_used_gb_median'] /= 1e9
            row['mem_used_gb_p95'] /= 1e9

        # Calculate prefix cache hit rate
        if 'prefix_hits_rate_mean' in row and 'prefix_queries_rate_mean' in row:
            if row['prefix_queries_rate_mean'] > 0:
                row['prefix_hit_rate_pct'] = (row['prefix_hits_rate_mean'] / row['prefix_queries_rate_mean']) * 100

        results.append(row)

    print("\n" + "=" * 80)
    print(f"Extracted {len(results)} benchmark runs at rate={TARGET_RATE}")

    if results:
        df = pd.DataFrame(results)
        ANALYSIS_DIR.mkdir(exist_ok=True)
        output_file = ANALYSIS_DIR / 'pcp_metrics_peak.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")

        # Print summary
        print("\nKey findings:")
        print(df.groupby('scenario').agg({
            'kv_cache_pct_mean': 'mean',
            'gpu_util_mean': 'mean',
            'process_rss_gb_mean': 'mean',
            'requests_running_mean': 'mean',
        }).round(2))

    print("\nDone!")

if __name__ == '__main__':
    main()
