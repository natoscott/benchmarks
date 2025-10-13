#!/usr/bin/env python3
"""
GPU Memory and KV Cache Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load Parquet data
print("Loading PCP data...")
df = pd.read_parquet('benchmark-data.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Load run_id mappings
run_id_map = {}
configs = {
    'benchmark-Qwen3-0.6B-offload.json': ('Qwen3-0.6B', 'Offload'),
    'benchmark-Qwen3-0.6B-lmcache.json': ('Qwen3-0.6B', 'LMCache'),
    'benchmark-Qwen3-0.6B-default.json': ('Qwen3-0.6B', 'Baseline'),
    'benchmark-Qwen3-8B-offload.json': ('Qwen3-8B', 'Offload'),
    'benchmark-Qwen3-8B-lmcache.json': ('Qwen3-8B', 'LMCache'),
    'benchmark-Qwen3-8B-default.json': ('Qwen3-8B', 'Baseline'),
}

for json_file, (model, config) in configs.items():
    with open(json_file) as f:
        data = json.load(f)
    run_id = data['benchmarks'][0]['run_id']
    run_id_map[run_id] = f"{model}-{config}"

# Find GPU memory columns
gpu_mem_cols = [c for c in df.columns if 'nvidia.memused' in c and 'accum' not in c]
gpu_util_cols = [c for c in df.columns if 'nvidia.gpuactive' in c and 'accum' not in c]
kv_cache_cols = [c for c in df.columns if 'vllm.kv_cache_usage_perc' in c]

print(f"\nFound {len(gpu_mem_cols)} GPU memory metrics")
print(f"Found {len(gpu_util_cols)} GPU utilization metrics")
print(f"Found {len(kv_cache_cols)} KV cache usage metrics")

# Get run_id columns to determine time ranges
run_id_cols = [c for c in df.columns if 'guidellm.run_id[' in c]

# Analyze each run
print("\n" + "="*120)
print("GPU MEMORY AND KV CACHE ANALYSIS")
print("="*120)

for run_id, name in sorted(run_id_map.items(), key=lambda x: x[1]):
    # Find instances for this run_id
    instance_ids = []
    for col in run_id_cols:
        instance_id = col.split('[')[1].split(']')[0]
        vals = df[col].dropna().unique()
        if len(vals) > 0 and vals[0] == run_id:
            instance_ids.append(instance_id)

    if not instance_ids:
        continue

    # Get time range where this run was active
    first_instance = instance_ids[0]
    duration_col = f'guidellm.duration[{first_instance}]'
    if duration_col in df.columns:
        active_mask = df[duration_col].notna()
        active_data = df[active_mask]

        if len(active_data) > 0:
            print(f"\n{name}:")
            print(f"  Time range: {active_data.index[0]} to {active_data.index[-1]}")
            print(f"  Duration: {(active_data.index[-1] - active_data.index[0]).total_seconds():.0f}s")

            # GPU memory analysis
            if len(gpu_mem_cols) > 0:
                mem_vals = active_data[gpu_mem_cols[0]].dropna()
                if len(mem_vals) > 0:
                    mem_mean_mb = mem_vals.mean() / (1024**2)
                    mem_max_mb = mem_vals.max() / (1024**2)
                    print(f"  GPU Memory: mean={mem_mean_mb:.0f} MB, max={mem_max_mb:.0f} MB")

            # GPU utilization
            if len(gpu_util_cols) > 0:
                util_vals = active_data[gpu_util_cols[0]].dropna()
                if len(util_vals) > 0:
                    util_mean = util_vals.mean() * 100
                    print(f"  GPU Util: mean={util_mean:.1f}%")

            # KV cache usage
            if len(kv_cache_cols) > 0:
                for kv_col in kv_cache_cols:
                    kv_vals = active_data[kv_col].dropna()
                    if len(kv_vals) > 0:
                        kv_mean = kv_vals.mean()
                        kv_max = kv_vals.max()
                        print(f"  KV Cache: mean={kv_mean:.1f}%, max={kv_max:.1f}%")
                        break

print("\n" + "="*120)
print("Analysis complete!")
