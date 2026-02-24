#!/usr/bin/env python3
"""
Deep dive into CPU utilization, memory pressure, and prefix cache metrics from PCP archives.
Focuses on metrics that weren't fully explored in the initial analysis.
"""

import subprocess
import pandas as pd
from pathlib import Path
import numpy as np

RESULTS_DIR = Path('results')
ANALYSIS_DIR = Path('analysis')
TARGET_RATE = 50

# Extended metrics focusing on CPU, memory pressure, and prefix cache
METRICS = [
    # CPU utilization breakdown
    ('cpu_user', 'kernel.all.cpu.user'),
    ('cpu_sys', 'kernel.all.cpu.sys'),
    ('cpu_idle', 'kernel.all.cpu.idle'),
    ('cpu_wait', 'kernel.all.cpu.wait.total'),
    ('cpu_steal', 'kernel.all.cpu.steal'),

    # Memory pressure (PSI - Pressure Stall Information)
    ('mem_pressure_some_avg10', 'kernel.all.pressure.memory.some.avg10'),
    ('mem_pressure_some_avg60', 'kernel.all.pressure.memory.some.avg60'),
    ('mem_pressure_full_avg10', 'kernel.all.pressure.memory.full.avg10'),
    ('mem_pressure_full_avg60', 'kernel.all.pressure.memory.full.avg60'),

    # CPU pressure
    ('cpu_pressure_some_avg10', 'kernel.all.pressure.cpu.some.avg10'),
    ('cpu_pressure_some_avg60', 'kernel.all.pressure.cpu.some.avg60'),

    # I/O pressure
    ('io_pressure_some_avg10', 'kernel.all.pressure.io.some.avg10'),
    ('io_pressure_full_avg10', 'kernel.all.pressure.io.full.avg10'),

    # vLLM process metrics
    ('process_cpu_seconds', 'openmetrics.vllm.process_cpu_seconds_total'),
    ('process_rss_bytes', 'openmetrics.vllm.process_resident_memory_bytes'),
    ('process_vms_bytes', 'openmetrics.vllm.process_virtual_memory_bytes'),
    ('process_open_fds', 'openmetrics.vllm.process_open_fds'),

    # Prefix cache metrics - regular prefix cache
    ('prefix_cache_hits', 'openmetrics.vllm.vllm.prefix_cache_hits_total'),
    ('prefix_cache_queries', 'openmetrics.vllm.vllm.prefix_cache_queries_total'),
    ('prefix_cache_blocks', 'openmetrics.vllm.vllm.prefix_cache_blocks'),

    # External prefix cache (llm-d EPP)
    ('external_prefix_cache_hits', 'openmetrics.vllm.vllm.external_prefix_cache_hits_total'),
    ('external_prefix_cache_queries', 'openmetrics.vllm.vllm.external_prefix_cache_queries_total'),
    ('external_prefix_cache_blocks', 'openmetrics.vllm.vllm.external_prefix_cache_blocks'),

    # KV-cache metrics
    ('kv_cache_usage_pct', 'openmetrics.vllm.vllm.kv_cache_usage_perc'),
    ('gpu_cache_usage_pct', 'openmetrics.vllm.vllm.gpu_cache_usage_perc'),
    ('cpu_cache_usage_pct', 'openmetrics.vllm.vllm.cpu_cache_usage_perc'),
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

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
            'max': np.max(values),
            'min': np.min(values),
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
    print("Deep dive: CPU, Memory Pressure, and Prefix Cache Analysis")
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
                row[f'{metric_key}_max'] = stats['max']
                row[f'{metric_key}_min'] = stats['min']
                print(f"  ✓ {metric_key}: mean={stats['mean']:.2f}")
            else:
                print(f"  ✗ {metric_key}")

        # Calculate derived metrics

        # CPU utilization percentage (converting from centiseconds to percentage)
        if 'cpu_user_mean' in row and 'cpu_idle_mean' in row:
            total_cpu = row.get('cpu_user_mean', 0) + row.get('cpu_sys_mean', 0) + row.get('cpu_idle_mean', 0) + row.get('cpu_wait_mean', 0)
            if total_cpu > 0:
                row['cpu_util_pct_mean'] = ((total_cpu - row['cpu_idle_mean']) / total_cpu) * 100
                print(f"  → CPU util: {row['cpu_util_pct_mean']:.1f}%")

        # Prefix cache hit rates
        if 'prefix_cache_hits_mean' in row and 'prefix_cache_queries_mean' in row:
            if row['prefix_cache_queries_mean'] > 0:
                row['prefix_hit_rate_pct'] = (row['prefix_cache_hits_mean'] / row['prefix_cache_queries_mean']) * 100
                print(f"  → Prefix hit rate: {row['prefix_hit_rate_pct']:.1f}%")

        # External prefix cache hit rates
        if 'external_prefix_cache_hits_mean' in row and 'external_prefix_cache_queries_mean' in row:
            if row['external_prefix_cache_queries_mean'] > 0:
                row['external_prefix_hit_rate_pct'] = (row['external_prefix_cache_hits_mean'] / row['external_prefix_cache_queries_mean']) * 100
                print(f"  → External prefix hit rate: {row['external_prefix_hit_rate_pct']:.1f}%")

        # Memory conversion
        if 'process_rss_bytes_mean' in row:
            row['process_rss_gb_mean'] = row['process_rss_bytes_mean'] / 1e9
            row['process_vms_gb_mean'] = row.get('process_vms_bytes_mean', 0) / 1e9

        results.append(row)

    print("\n" + "=" * 80)
    print(f"Extracted {len(results)} benchmark runs at rate={TARGET_RATE}")

    if results:
        df = pd.DataFrame(results)
        ANALYSIS_DIR.mkdir(exist_ok=True)
        output_file = ANALYSIS_DIR / 'pcp_cpu_memory_analysis.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")

        # Print insights by scenario
        print("\n" + "=" * 80)
        print("CPU and Memory Insights by Scenario:")
        print("=" * 80)

        for scenario in sorted(df['scenario'].unique()):
            scenario_df = df[df['scenario'] == scenario]
            print(f"\n{scenario}:")

            if 'cpu_util_pct_mean' in scenario_df.columns:
                cpu_util = scenario_df['cpu_util_pct_mean'].mean()
                print(f"  CPU Utilization: {cpu_util:.1f}%")

            if 'mem_pressure_some_avg10_mean' in scenario_df.columns:
                mem_pressure = scenario_df['mem_pressure_some_avg10_mean'].mean()
                if not np.isnan(mem_pressure):
                    print(f"  Memory Pressure (10s): {mem_pressure:.2f}")

            if 'cpu_pressure_some_avg10_mean' in scenario_df.columns:
                cpu_pressure = scenario_df['cpu_pressure_some_avg10_mean'].mean()
                if not np.isnan(cpu_pressure):
                    print(f"  CPU Pressure (10s): {cpu_pressure:.2f}")

            if 'prefix_hit_rate_pct' in scenario_df.columns:
                hit_rate = scenario_df['prefix_hit_rate_pct'].mean()
                if not np.isnan(hit_rate):
                    print(f"  Prefix Cache Hit Rate: {hit_rate:.1f}%")

            if 'external_prefix_hit_rate_pct' in scenario_df.columns:
                ext_hit_rate = scenario_df['external_prefix_hit_rate_pct'].mean()
                if not np.isnan(ext_hit_rate):
                    print(f"  External Prefix Hit Rate: {ext_hit_rate:.1f}%")

    print("\nDone!")

if __name__ == '__main__':
    main()
