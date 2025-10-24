#!/usr/bin/env python3
"""
Detailed GPU utilization analysis with proper metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load GPU active metrics (1 sec resolution)
print("Loading GPU active metrics...")
df_gpu = pd.read_csv('gpu_active_metrics.csv', parse_dates=['Time'])

# Load final config mapping
df_results = pd.read_csv('final_config_mapping.csv', parse_dates=['start_time', 'end_time'])

print(f"GPU data shape: {df_gpu.shape}")
print(f"Time range: {df_gpu['Time'].min()} to {df_gpu['Time'].max()}")

# Extract GPU utilization for each benchmark
gpu_data = []

for idx, row in df_results[~df_results['is_warmup']].iterrows():
    start_time = row['start_time']
    end_time = row['end_time']

    # Find GPU data during this benchmark
    time_mask = (df_gpu['Time'] >= start_time) & (df_gpu['Time'] <= end_time)

    if time_mask.sum() == 0:
        continue

    benchmark_gpu = df_gpu[time_mask]

    # Get GPU active percentages across both GPUs
    gpu0_active = benchmark_gpu['nvidia.gpuactive-gpu0'].dropna()
    gpu1_active = benchmark_gpu['nvidia.gpuactive-gpu1'].dropna()
    mem0_active = benchmark_gpu['nvidia.memactive-gpu0'].dropna()
    mem1_active = benchmark_gpu['nvidia.memactive-gpu1'].dropna()
    mem0_used = benchmark_gpu['nvidia.memused-gpu0'].dropna()
    mem1_used = benchmark_gpu['nvidia.memused-gpu1'].dropna()

    # Combine both GPUs
    all_gpu_active = pd.concat([gpu0_active, gpu1_active])
    all_mem_active = pd.concat([mem0_active, mem1_active])
    all_mem_used = pd.concat([mem0_used, mem1_used])

    gpu_data.append({
        'model': row['model'],
        'config_label': row['config_label'],
        'concurrency': row['concurrency'],
        'throughput': row['tokens_per_second'],
        'ttft_ms': row['ttft_ms'],
        'tpot_ms': row['tpot_ms'],
        'gpu_active_mean': all_gpu_active.mean() if len(all_gpu_active) > 0 else np.nan,
        'gpu_active_median': all_gpu_active.median() if len(all_gpu_active) > 0 else np.nan,
        'gpu_active_max': all_gpu_active.max() if len(all_gpu_active) > 0 else np.nan,
        'gpu_active_std': all_gpu_active.std() if len(all_gpu_active) > 0 else np.nan,
        'mem_active_mean': all_mem_active.mean() if len(all_mem_active) > 0 else np.nan,
        'mem_active_median': all_mem_active.median() if len(all_mem_active) > 0 else np.nan,
        'mem_active_max': all_mem_active.max() if len(all_mem_active) > 0 else np.nan,
        'mem_used_mean_gb': (all_mem_used.mean() / 1024**3) if len(all_mem_used) > 0 else np.nan,
        'mem_used_max_gb': (all_mem_used.max() / 1024**3) if len(all_mem_used) > 0 else np.nan,
        'num_samples': len(all_gpu_active)
    })

df_gpu_analysis = pd.DataFrame(gpu_data)

print("\n=== GPU Utilization Summary (Non-Warmup Runs) ===")
summary = df_gpu_analysis.groupby(['model', 'config_label']).agg({
    'gpu_active_mean': 'mean',
    'gpu_active_max': 'mean',
    'mem_active_mean': 'mean',
    'mem_used_mean_gb': 'mean',
    'throughput': 'mean'
}).round(2)

print(summary)

# Save detailed GPU analysis
df_gpu_analysis.to_csv('gpu_utilization_detailed.csv', index=False)
print("\nSaved to gpu_utilization_detailed.csv")

# Print detailed comparison
print("\n" + "="*80)
print("GPU UTILIZATION COMPARISON: Default vs Offload")
print("="*80)

for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    print(f"\n{model}:")
    print("-" * 80)
    model_data = df_gpu_analysis[df_gpu_analysis['model'] == model]

    baseline = model_data[model_data['config_label'] == 'default']
    offload = model_data[model_data['config_label'] == 'offload']
    lmcache = model_data[model_data['config_label'] == 'lmcache']

    if len(baseline) > 0 and len(offload) > 0:
        # GPU Compute Active
        baseline_gpu = baseline['gpu_active_mean'].mean()
        offload_gpu = offload['gpu_active_mean'].mean()
        lmcache_gpu = lmcache['gpu_active_mean'].mean() if len(lmcache) > 0 else np.nan

        # Memory Active
        baseline_mem = baseline['mem_active_mean'].mean()
        offload_mem = offload['mem_active_mean'].mean()
        lmcache_mem = lmcache['mem_active_mean'].mean() if len(lmcache) > 0 else np.nan

        # Memory Used
        baseline_mem_used = baseline['mem_used_mean_gb'].mean()
        offload_mem_used = offload['mem_used_mean_gb'].mean()
        lmcache_mem_used = lmcache['mem_used_mean_gb'].mean() if len(lmcache) > 0 else np.nan

        # Throughput
        baseline_tput = baseline['throughput'].mean()
        offload_tput = offload['throughput'].mean()

        print(f"\n  GPU Compute Active (%):")
        print(f"    Baseline:  {baseline_gpu:.1f}%")
        print(f"    Offload:   {offload_gpu:.1f}% ({(offload_gpu-baseline_gpu)/baseline_gpu*100:+.1f}%)")
        if not np.isnan(lmcache_gpu):
            print(f"    LMCache:   {lmcache_gpu:.1f}% ({(lmcache_gpu-baseline_gpu)/baseline_gpu*100:+.1f}%)")

        print(f"\n  GPU Memory Active (%):")
        print(f"    Baseline:  {baseline_mem:.1f}%")
        print(f"    Offload:   {offload_mem:.1f}% ({(offload_mem-baseline_mem)/baseline_mem*100:+.1f}%)")
        if not np.isnan(lmcache_mem):
            print(f"    LMCache:   {lmcache_mem:.1f}% ({(lmcache_mem-baseline_mem)/baseline_mem*100:+.1f}%)")

        print(f"\n  GPU Memory Used (GB):")
        print(f"    Baseline:  {baseline_mem_used:.1f} GB")
        print(f"    Offload:   {offload_mem_used:.1f} GB ({(offload_mem_used-baseline_mem_used)/baseline_mem_used*100:+.1f}%)")
        if not np.isnan(lmcache_mem_used):
            print(f"    LMCache:   {lmcache_mem_used:.1f} GB ({(lmcache_mem_used-baseline_mem_used)/baseline_mem_used*100:+.1f}%)")

        print(f"\n  Throughput (tok/s):")
        print(f"    Baseline:  {baseline_tput:.1f} tok/s")
        print(f"    Offload:   {offload_tput:.1f} tok/s ({(offload_tput-baseline_tput)/baseline_tput*100:+.1f}%)")

print("\n" + "="*80)

# Create a comparison summary
comparison_data = []
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    model_data = df_gpu_analysis[df_gpu_analysis['model'] == model]

    for config in ['default', 'offload', 'lmcache']:
        config_data = model_data[model_data['config_label'] == config]
        if len(config_data) > 0:
            comparison_data.append({
                'model': model,
                'config': config,
                'gpu_compute_avg': config_data['gpu_active_mean'].mean(),
                'gpu_memory_avg': config_data['mem_active_mean'].mean(),
                'memory_used_gb': config_data['mem_used_mean_gb'].mean(),
                'throughput': config_data['throughput'].mean()
            })

df_comparison = pd.DataFrame(comparison_data)
df_comparison.to_csv('gpu_comparison_summary.csv', index=False)
print("Saved comparison summary to gpu_comparison_summary.csv")
