#!/usr/bin/env python3
"""
Quick power consumption analysis for 14B model.
Focused on answering: Does CPU offload reduce GPU power consumption?
"""

import subprocess
import pandas as pd
from pathlib import Path

# Focus on 14B model at rate=50 (peak throughput)
TARGET_DIRS = [
    "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_no-offload_replica1_rate50",
    "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_native-offload_replica1_rate50",
    "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_lmcache-local_replica1_rate50",
    "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_lmcache-local-20kcpu_replica1_rate50",
    "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_lmcache-redis_replica1_rate50",
    "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_lmcache-valkey_replica1_rate50",
    "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_llm-d-redis_replica1_rate50",
    "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_llm-d-valkey_replica1_rate50",
    "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_native-offload-20kcpu_replica1_rate50",
]

def find_archive(result_dir):
    """Find PCP archive in result directory."""
    pcp_dir = result_dir / "pcp-archives"
    if not pcp_dir.exists():
        return None

    for node_dir in pcp_dir.iterdir():
        if node_dir.is_dir():
            # Decompress if needed
            subprocess.run(f"cd {node_dir} && zstd -d -f *.zst 2>/dev/null", shell=True, capture_output=True)

            meta_files = list(node_dir.glob("*.meta"))
            if meta_files:
                return str(meta_files[0]).replace(".meta", "")

    return None

def extract_power_stats(archive_path):
    """Extract GPU power consumption stats."""
    try:
        cmd = f"pmrep -a {archive_path} -t 10s -o csv openmetrics.dcgm.DCGM_FI_DEV_POWER_USAGE"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            return None

        # Parse CSV - sum across both GPUs (columns 1 and 2)
        power_values = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    gpu0 = float(parts[1]) if parts[1] else 0
                    gpu1 = float(parts[2]) if parts[2] else 0
                    if gpu0 > 0 or gpu1 > 0:
                        power_values.append(gpu0 + gpu1)  # Total power across both GPUs
                except ValueError:
                    continue

        if power_values:
            import numpy as np
            return {
                'mean': np.mean(power_values),
                'median': np.median(power_values),
                'min': np.min(power_values),
                'max': np.max(power_values),
            }

        return None

    except Exception as e:
        print(f"Error extracting power: {e}")
        return None

def extract_util_stats(archive_path):
    """Extract GPU utilization stats."""
    try:
        cmd = f"pmrep -a {archive_path} -t 10s -o csv openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            return None

        # Parse CSV - average across both GPUs
        util_values = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    gpu0 = float(parts[1]) if parts[1] else 0
                    gpu1 = float(parts[2]) if parts[2] else 0
                    if gpu0 > 0 or gpu1 > 0:
                        util_values.append((gpu0 + gpu1) / 2.0)  # Average util
                except ValueError:
                    continue

        if util_values:
            import numpy as np
            return {
                'mean': np.mean(util_values),
                'median': np.median(util_values),
            }

        return None

    except Exception as e:
        print(f"Error extracting utilization: {e}")
        return None

def main():
    results_dir = Path("results")

    print("=" * 80)
    print("POWER CONSUMPTION ANALYSIS: Qwen3-14B @ Rate=50")
    print("=" * 80)
    print()
    print("Question: Does CPU KV-cache offload reduce GPU power consumption?")
    print()

    data = []

    for dir_name in TARGET_DIRS:
        result_dir = results_dir / dir_name
        if not result_dir.exists():
            continue

        # Extract scenario name
        if "no-offload" in dir_name:
            scenario = "no-offload"
        elif "native-offload-20kcpu" in dir_name:
            scenario = "native-offload-20kcpu"
        elif "native-offload" in dir_name:
            scenario = "native-offload"
        elif "lmcache-local-20kcpu" in dir_name:
            scenario = "lmcache-local-20kcpu"
        elif "lmcache-local" in dir_name:
            scenario = "lmcache-local"
        elif "lmcache-redis" in dir_name:
            scenario = "lmcache-redis"
        elif "lmcache-valkey" in dir_name:
            scenario = "lmcache-valkey"
        elif "llm-d-redis" in dir_name:
            scenario = "llm-d-redis"
        elif "llm-d-valkey" in dir_name:
            scenario = "llm-d-valkey"
        else:
            continue

        print(f"Processing: {scenario}")

        archive_path = find_archive(result_dir)
        if not archive_path:
            print(f"  No archive found")
            continue

        power_stats = extract_power_stats(archive_path)
        util_stats = extract_util_stats(archive_path)

        if power_stats:
            print(f"  Power: {power_stats['mean']:.1f}W (avg), {power_stats['median']:.1f}W (median)")
            data.append({
                'scenario': scenario,
                'power_mean_w': power_stats['mean'],
                'power_median_w': power_stats['median'],
                'power_min_w': power_stats['min'],
                'power_max_w': power_stats['max'],
                'gpu_util_mean': util_stats['mean'] if util_stats else None,
            })
        else:
            print(f"  No power data")

    if not data:
        print("\nNo data extracted!")
        return

    df = pd.DataFrame(data)
    df.to_csv('analysis/power_14b_rate50.csv', index=False)

    print()
    print("=" * 80)
    print("POWER CONSUMPTION SUMMARY")
    print("=" * 80)
    print()

    # Find baseline
    baseline = df[df['scenario'] == 'no-offload']
    if len(baseline) > 0:
        baseline_power = baseline['power_mean_w'].values[0]
        baseline_util = baseline['gpu_util_mean'].values[0] if baseline['gpu_util_mean'].notna().values[0] else None

        print(f"Baseline (no-offload):")
        print(f"  Power: {baseline_power:.1f}W")
        if baseline_util:
            print(f"  GPU Util: {baseline_util:.1f}%")
        print()

        print(f"{'Scenario':<25} {'Power (W)':<12} {'Delta (W)':<12} {'Savings (%)':<12} {'GPU Util (%)':<12}")
        print("-" * 85)

        for _, row in df.iterrows():
            scenario = row['scenario']
            power = row['power_mean_w']
            delta = power - baseline_power
            savings = (delta / baseline_power) * 100
            util = row['gpu_util_mean'] if pd.notna(row['gpu_util_mean']) else 0

            print(f"{scenario:<25} {power:<12.1f} {delta:<+12.1f} {savings:<+12.1f} {util:<12.1f}")

        print()
        print("=" * 80)
        print("KEY FINDINGS:")
        print("=" * 80)

        # Find scenarios with best offload performance
        offload_scenarios = df[df['scenario'].str.contains('lmcache-local')]
        if len(offload_scenarios) > 0:
            best = offload_scenarios.loc[offload_scenarios['power_mean_w'].idxmin()]
            savings = ((best['power_mean_w'] - baseline_power) / baseline_power) * 100

            print(f"\nBest CPU offload scenario: {best['scenario']}")
            print(f"  Power consumption: {best['power_mean_w']:.1f}W (vs {baseline_power:.1f}W baseline)")
            print(f"  Power savings: {-savings:.1f}% reduction" if savings < 0 else f"  Power increase: {savings:.1f}%")
            print(f"  Absolute savings: {baseline_power - best['power_mean_w']:.1f}W")

            if pd.notna(best['gpu_util_mean']) and baseline_util:
                util_delta = best['gpu_util_mean'] - baseline_util
                print(f"  GPU utilization: {best['gpu_util_mean']:.1f}% (vs {baseline_util:.1f}% baseline, {util_delta:+.1f}%)")

if __name__ == '__main__':
    main()
