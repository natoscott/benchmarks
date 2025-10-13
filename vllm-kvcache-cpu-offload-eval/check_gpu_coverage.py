#!/usr/bin/env python3
"""
Check GPU metrics coverage across all benchmark runs
"""

import pandas as pd
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
    run_id_map[run_id] = {
        'name': f"{model}-{config}",
        'json_file': json_file
    }

# Find metric columns
run_id_cols = [c for c in df.columns if 'guidellm.run_id[' in c]
gpu_mem_cols = [c for c in df.columns if 'nvidia.memused' in c and 'accum' not in c]
gpu_util_cols = [c for c in df.columns if 'nvidia.gpuactive' in c and 'accum' not in c]
kv_cache_cols = [c for c in df.columns if 'vllm.kv_cache_usage_perc' in c]

print("\n" + "="*120)
print("GPU METRICS COVERAGE ANALYSIS")
print("="*120)

print(f"\nAvailable GPU metric types:")
print(f"  GPU Memory: {len(gpu_mem_cols)} metrics - {gpu_mem_cols}")
print(f"  GPU Util: {len(gpu_util_cols)} metrics - {gpu_util_cols}")
print(f"  KV Cache: {len(kv_cache_cols)} metrics - {kv_cache_cols}")

print("\n" + "-"*120)
print(f"{'Configuration':<30} {'Run ID':<40} {'Has Data':<10} {'Time Range':<40}")
print("-"*120)

for run_id, info in sorted(run_id_map.items(), key=lambda x: x[1]['name']):
    name = info['name']

    # Find instances for this run_id
    instance_ids = []
    for col in run_id_cols:
        instance_id = col.split('[')[1].split(']')[0]
        vals = df[col].dropna().unique()
        if len(vals) > 0 and vals[0] == run_id:
            instance_ids.append(instance_id)

    has_data = "NO"
    time_range = "N/A"

    if instance_ids:
        first_instance = instance_ids[0]
        duration_col = f'guidellm.duration[{first_instance}]'
        if duration_col in df.columns:
            active_mask = df[duration_col].notna()
            active_data = df[active_mask]

            if len(active_data) > 0:
                start_time = active_data.index[0]
                end_time = active_data.index[-1]
                time_range = f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"

                # Check if GPU data exists during this time
                if len(gpu_mem_cols) > 0:
                    gpu_data = active_data[gpu_mem_cols[0]].dropna()
                    if len(gpu_data) > 0:
                        has_data = "YES"

    print(f"{name:<30} {run_id[:36]:<40} {has_data:<10} {time_range:<40}")

# Now check what time ranges have GPU data
print("\n" + "="*120)
print("GPU DATA TIME RANGES")
print("="*120)

if len(gpu_mem_cols) > 0:
    gpu_col = gpu_mem_cols[0]
    gpu_data = df[gpu_col].dropna()

    if len(gpu_data) > 0:
        print(f"\nGPU Memory metric '{gpu_col}':")
        print(f"  First sample: {gpu_data.index[0]}")
        print(f"  Last sample: {gpu_data.index[-1]}")
        print(f"  Total samples: {len(gpu_data)}")

        # Find gaps in GPU data
        print(f"\n  Checking for continuous coverage...")

        # Group by date-hour to show coverage
        gpu_data_df = pd.DataFrame({'value': gpu_data})
        gpu_data_df['hour'] = gpu_data_df.index.floor('H')
        hourly_counts = gpu_data_df.groupby('hour').size()

        print(f"\n  Hourly sample counts:")
        for hour, count in hourly_counts.items():
            print(f"    {hour}: {count} samples")

# Check which run_ids are missing GPU data
print("\n" + "="*120)
print("MISSING GPU DATA SUMMARY")
print("="*120)

missing_configs = []
has_data_configs = []

for run_id, info in sorted(run_id_map.items(), key=lambda x: x[1]['name']):
    name = info['name']

    instance_ids = []
    for col in run_id_cols:
        instance_id = col.split('[')[1].split(']')[0]
        vals = df[col].dropna().unique()
        if len(vals) > 0 and vals[0] == run_id:
            instance_ids.append(instance_id)

    if instance_ids:
        first_instance = instance_ids[0]
        duration_col = f'guidellm.duration[{first_instance}]'
        if duration_col in df.columns:
            active_mask = df[duration_col].notna()
            active_data = df[active_mask]

            if len(active_data) > 0 and len(gpu_mem_cols) > 0:
                gpu_data = active_data[gpu_mem_cols[0]].dropna()
                if len(gpu_data) > 0:
                    has_data_configs.append(name)
                else:
                    missing_configs.append(name)

print(f"\nConfigurations WITH GPU metrics ({len(has_data_configs)}):")
for config in has_data_configs:
    print(f"  ✓ {config}")

print(f"\nConfigurations MISSING GPU metrics ({len(missing_configs)}):")
for config in missing_configs:
    print(f"  ✗ {config}")

print("\n" + "="*120)
