#!/usr/bin/env python3
"""
Extract PCP metrics from benchmark result archives.
Focuses on key system metrics: CPU, GPU, memory, KV-cache, network.
"""

import subprocess
import pandas as pd
from pathlib import Path
import numpy as np

SCENARIOS = [
    'no-offload',
    'native-offload',
    'lmcache-local',
    'lmcache-redis',
    'lmcache-valkey',
    'llm-d-redis',
    'llm-d-valkey'
]

MODELS = [
    'Qwen3-0.6B',
    'Qwen3-8B',
    'Qwen3-14B',
    'Qwen3-32B-AWQ'
]

def extract_pcp_metric(archive_dir, metric_name, instance=None):
    """Extract a single PCP metric from archive."""
    # Find PCP archive files
    archive_files = []
    for node_dir in archive_dir.iterdir():
        if node_dir.is_dir():
            meta_files = list(node_dir.glob('*.meta'))
            if meta_files:
                # Use first archive found
                archive_base = str(meta_files[0]).replace('.meta', '')
                archive_files.append(archive_base)
                break

    if not archive_files:
        return None

    try:
        # Build pmrep command
        cmd = ['pmrep', '-a', archive_files[0], '-t', '5s', '-o', 'csv', '-F', metric_name]
        if instance:
            cmd.extend(['-i', instance])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0 or not result.stdout:
            return None

        # Parse CSV output
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 2:  # Header only or no data
            return None

        values = []
        for line in lines[2:]:  # Skip header lines
            parts = line.split(',')
            if len(parts) > 1:
                try:
                    val = float(parts[-1])
                    if not np.isnan(val):
                        values.append(val)
                except (ValueError, IndexError):
                    continue

        if values:
            return {
                'mean': np.mean(values),
                'median': np.median(values),
                'max': np.max(values),
                'min': np.min(values),
                'p95': np.percentile(values, 95)
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return None

def main():
    results_dir = Path('results')

    print("="  * 80)
    print("PCP METRICS EXTRACTION")
    print("=" * 80)
    print()
    print("Extracting system metrics from PCP archives...")
    print()

    all_metrics = []
    processed = 0

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue

        # Parse directory name
        dir_name = result_dir.name
        parts = dir_name.split('_')

        model = None
        scenario = None
        rate = None

        for part in parts:
            if part in MODELS:
                model = part
            if part in SCENARIOS:
                scenario = part
            if part.startswith('rate'):
                rate = int(part[4:])

        if not all([model, scenario, rate]):
            continue

        pcp_dir = result_dir / 'pcp-archives'
        if not pcp_dir.exists():
            continue

        print(f"Processing {model} {scenario} rate={rate}...", end=' ')

        metrics = {
            'model': model,
            'scenario': scenario,
            'rate': rate
        }

        # Extract key metrics
        # GPU utilization (DCGM)
        gpu_util = extract_pcp_metric(pcp_dir, 'dcgm.gpu_utilization')
        if gpu_util:
            metrics['gpu_util_mean'] = gpu_util['mean']
            metrics['gpu_util_max'] = gpu_util['max']

        # CPU utilization (kernel)
        cpu_idle = extract_pcp_metric(pcp_dir, 'kernel.all.cpu.idle')
        if cpu_idle:
            metrics['cpu_util_mean'] = 100.0 - cpu_idle['mean']
            metrics['cpu_util_max'] = 100.0 - cpu_idle['min']

        # vLLM KV-cache usage
        kv_cache = extract_pcp_metric(pcp_dir, 'vllm.gpu_cache_usage_perc')
        if kv_cache:
            metrics['kv_cache_usage_mean'] = kv_cache['mean']
            metrics['kv_cache_usage_max'] = kv_cache['max']

        # vLLM prefix cache hit rate
        hit_rate = extract_pcp_metric(pcp_dir, 'vllm.cache_config_prefix_cache_hit_rate')
        if hit_rate:
            metrics['prefix_cache_hit_rate'] = hit_rate['mean']

        # vLLM request queue depths
        running = extract_pcp_metric(pcp_dir, 'vllm.num_requests_running')
        if running:
            metrics['requests_running_mean'] = running['mean']
            metrics['requests_running_max'] = running['max']

        waiting = extract_pcp_metric(pcp_dir, 'vllm.num_requests_waiting')
        if waiting:
            metrics['requests_waiting_mean'] = waiting['mean']
            metrics['requests_waiting_max'] = waiting['max']

        # GPU memory utilization
        gpu_mem = extract_pcp_metric(pcp_dir, 'dcgm.memory_utilization')
        if gpu_mem:
            metrics['gpu_mem_util_mean'] = gpu_mem['mean']
            metrics['gpu_mem_util_max'] = gpu_mem['max']

        # Network metrics (if available)
        net_rx = extract_pcp_metric(pcp_dir, 'network.interface.in.bytes', instance='eth0')
        if net_rx:
            metrics['net_rx_mean_mbps'] = (net_rx['mean'] * 8) / 1_000_000  # Convert to Mbps

        net_tx = extract_pcp_metric(pcp_dir, 'network.interface.out.bytes', instance='eth0')
        if net_tx:
            metrics['net_tx_mean_mbps'] = (net_tx['mean'] * 8) / 1_000_000

        all_metrics.append(metrics)
        processed += 1
        print(f"✓ ({len(metrics)} metrics)")

    print()
    print(f"Processed {processed} benchmark runs")

    # Convert to DataFrame and save
    df = pd.DataFrame(all_metrics)
    df.to_csv('analysis/pcp_metrics.csv', index=False)

    print(f"Saved PCP metrics to analysis/pcp_metrics.csv")
    print()

    # Print summary statistics
    print("=" * 80)
    print("PCP METRICS SUMMARY")
    print("=" * 80)
    print()

    for model in MODELS:
        print(f"\n{model}:")
        print("-" * 80)
        model_data = df[df['model'] == model]

        for scenario in SCENARIOS:
            scenario_data = model_data[model_data['scenario'] == scenario]
            if len(scenario_data) == 0:
                continue

            # Get peak throughput row (highest rate with data)
            peak_row = scenario_data.loc[scenario_data['rate'].idxmax()]

            print(f"\n  {scenario}:")
            if 'cpu_util_mean' in peak_row and not pd.isna(peak_row['cpu_util_mean']):
                print(f"    CPU util:      {peak_row['cpu_util_mean']:6.1f}%")
            if 'gpu_util_mean' in peak_row and not pd.isna(peak_row['gpu_util_mean']):
                print(f"    GPU util:      {peak_row['gpu_util_mean']:6.1f}%")
            if 'kv_cache_usage_mean' in peak_row and not pd.isna(peak_row['kv_cache_usage_mean']):
                print(f"    KV-cache:      {peak_row['kv_cache_usage_mean']:6.1f}%")
            if 'prefix_cache_hit_rate' in peak_row and not pd.isna(peak_row['prefix_cache_hit_rate']):
                print(f"    Cache hits:    {peak_row['prefix_cache_hit_rate']:6.2f}%")

if __name__ == '__main__':
    main()
