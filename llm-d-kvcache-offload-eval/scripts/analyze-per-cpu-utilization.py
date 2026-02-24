#!/usr/bin/env python3
"""
Analyze per-CPU utilization patterns to identify CPU saturation that may be hidden by averaging.
Examines kernel.percpu metrics to detect hotspots and individual CPU bottlenecks.
"""

import subprocess
import pandas as pd
from pathlib import Path
import numpy as np
import re

RESULTS_DIR = Path('results')
ANALYSIS_DIR = Path('analysis')
TARGET_RATE = 50

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

def get_cpu_count(archive):
    """Determine number of CPUs from pmrep output."""
    try:
        cmd = ['pmrep', '-z', '-a', archive, '-s', '1', '-o', 'csv', 'kernel.percpu.cpu.user']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return None

        # Parse CSV header to count CPU columns
        lines = result.stdout.strip().split('\n')
        if len(lines) < 1:
            return None

        # Header format: "Time","cpu0","cpu1",...
        header = lines[0]
        cpu_cols = [col for col in header.split(',') if col.strip('"').startswith('cpu')]
        return len(cpu_cols)
    except Exception:
        return None

def analyze_percpu_metrics(archive, metric_name):
    """Analyze per-CPU metric to find max, min, and saturation patterns."""
    try:
        cmd = ['pmrep', '-z', '-a', archive, '-t', '10s', '-o', 'csv', metric_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return None

        lines = [l for l in result.stdout.strip().split('\n') if l and not l.startswith('Time')]
        if len(lines) < 2:
            return None

        # Parse CSV data
        # Format: "timestamp","cpu0","cpu1","cpu2",...
        cpu_data = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            cpu_values = []
            for part in parts[1:]:  # Skip timestamp
                try:
                    val = float(part.strip().strip('"'))
                    if not np.isnan(val):
                        cpu_values.append(val)
                except:
                    pass
            if cpu_values:
                cpu_data.append(cpu_values)

        if not cpu_data:
            return None

        # Convert to numpy array for easier analysis
        cpu_array = np.array(cpu_data)  # shape: (time_samples, num_cpus)

        # Calculate statistics
        # Per-CPU averages over time
        per_cpu_mean = np.mean(cpu_array, axis=0)
        per_cpu_max = np.max(cpu_array, axis=0)

        # Overall statistics
        global_mean = np.mean(per_cpu_mean)
        max_cpu_mean = np.max(per_cpu_mean)
        min_cpu_mean = np.min(per_cpu_mean)

        # Find hottest CPU
        hottest_cpu = np.argmax(per_cpu_mean)

        # Count saturated CPUs (>80% utilization on average)
        saturated_cpus = np.sum(per_cpu_mean > 80)

        # Check for peak saturation (any CPU hit >95% at any point)
        peak_saturated = np.any(cpu_array > 95)

        return {
            'global_mean': global_mean,
            'max_cpu_mean': max_cpu_mean,
            'min_cpu_mean': min_cpu_mean,
            'hottest_cpu': hottest_cpu,
            'saturated_cpus': saturated_cpus,
            'peak_saturated': peak_saturated,
            'cpu_count': cpu_array.shape[1],
            'per_cpu_mean': per_cpu_mean,
        }
    except Exception as e:
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
    print("Analyzing per-CPU utilization patterns...")
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

        # Analyze per-CPU user time
        user_stats = analyze_percpu_metrics(archive, 'kernel.percpu.cpu.user')

        if user_stats:
            row = {
                'model': model,
                'scenario': scenario,
                'rate': rate,
                'cpu_count': user_stats['cpu_count'],
                'global_mean_user': user_stats['global_mean'],
                'max_cpu_mean_user': user_stats['max_cpu_mean'],
                'min_cpu_mean_user': user_stats['min_cpu_mean'],
                'hottest_cpu_user': user_stats['hottest_cpu'],
                'saturated_cpus_user': user_stats['saturated_cpus'],
                'peak_saturated_user': user_stats['peak_saturated'],
            }

            print(f"  CPUs: {user_stats['cpu_count']}")
            print(f"  Global mean user: {user_stats['global_mean']:.1f}%")
            print(f"  Max CPU mean: {user_stats['max_cpu_mean']:.1f}% (CPU {user_stats['hottest_cpu']})")
            print(f"  Min CPU mean: {user_stats['min_cpu_mean']:.1f}%")
            print(f"  Saturated CPUs (>80%): {user_stats['saturated_cpus']}")
            if user_stats['peak_saturated']:
                print(f"  ⚠ Peak saturation detected (>95%)")

            # Analyze sys time
            sys_stats = analyze_percpu_metrics(archive, 'kernel.percpu.cpu.sys')
            if sys_stats:
                row['max_cpu_mean_sys'] = sys_stats['max_cpu_mean']
                row['saturated_cpus_sys'] = sys_stats['saturated_cpus']
                print(f"  Max CPU sys: {sys_stats['max_cpu_mean']:.1f}%")

            # Calculate total utilization distribution
            if user_stats['per_cpu_mean'] is not None:
                # Convert centiseconds to percentage if needed
                per_cpu_user = user_stats['per_cpu_mean']

                # Statistics on CPU load distribution
                row['cpu_stdev'] = np.std(per_cpu_user)
                row['cpu_range'] = np.max(per_cpu_user) - np.min(per_cpu_user)

                print(f"  CPU load std dev: {row['cpu_stdev']:.1f}%")
                print(f"  CPU load range: {row['cpu_range']:.1f}%")

            results.append(row)
        else:
            print("  No per-CPU data available")

    print("\n" + "=" * 80)
    print(f"Analyzed {len(results)} benchmark runs")

    if results:
        df = pd.DataFrame(results)
        ANALYSIS_DIR.mkdir(exist_ok=True)
        output_file = ANALYSIS_DIR / 'percpu_analysis.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")

        # Summary by scenario
        print("\n" + "=" * 80)
        print("Per-CPU Saturation Summary by Scenario:")
        print("=" * 80)

        for scenario in sorted(df['scenario'].unique()):
            scenario_df = df[df['scenario'] == scenario]
            print(f"\n{scenario}:")
            print(f"  Avg global CPU: {scenario_df['global_mean_user'].mean():.1f}%")
            print(f"  Avg max CPU: {scenario_df['max_cpu_mean_user'].mean():.1f}%")
            print(f"  Avg saturated CPUs: {scenario_df['saturated_cpus_user'].mean():.1f}")
            print(f"  Peak saturation events: {scenario_df['peak_saturated_user'].sum()}")
            print(f"  Avg CPU load std dev: {scenario_df['cpu_stdev'].mean():.1f}%")

    print("\nDone!")

if __name__ == '__main__':
    main()
