#!/usr/bin/env python3
"""
Analyze system-level CPU and Memory metrics across benchmark runs
Focus on metrics relevant to KV cache CPU offload
"""

import pandas as pd
import numpy as np
import json

print("Loading PCP data...")
df = pd.read_parquet('benchmark-data.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])

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
        'json_file': json_file,
        'start_time': pd.to_datetime(data['benchmarks'][0]['start_time'], unit='s'),
        'end_time': pd.to_datetime(data['benchmarks'][-1]['end_time'], unit='s'),
    }

# Key metrics for CPU offload analysis
key_metrics = {
    # Memory metrics - critical for understanding CPU offload
    'mem.util.used': 'Total Memory Used',
    'mem.util.free': 'Free Memory',
    'mem.util.cached': 'Cached Memory',
    'mem.util.active': 'Active Memory',
    'mem.util.inactive': 'Inactive Memory',
    'mem.util.active_anon': 'Active Anonymous (heap allocations)',
    'mem.util.inactive_anon': 'Inactive Anonymous',
    'mem.util.slab': 'Kernel Slab (cache)',

    # CPU metrics - understanding compute overhead
    'kernel.all.cpu.user': 'User CPU Time',
    'kernel.all.cpu.sys': 'System CPU Time',
    'kernel.all.cpu.idle': 'Idle CPU Time',
    'kernel.all.cpu.wait.total': 'IO Wait Time',
}

# Check which metrics exist
available_metrics = {}
for metric, desc in key_metrics.items():
    if metric in df.columns:
        available_metrics[metric] = desc

print(f"\nFound {len(available_metrics)} of {len(key_metrics)} key system metrics")
print("\nAvailable metrics:")
for metric, desc in available_metrics.items():
    print(f"  ✓ {metric:<30} - {desc}")

missing = set(key_metrics.keys()) - set(available_metrics.keys())
if missing:
    print("\nMissing metrics:")
    for metric in missing:
        print(f"  ✗ {metric:<30} - {key_metrics[metric]}")

print("\n" + "="*120)
print("SYSTEM RESOURCE UTILIZATION ANALYSIS")
print("="*120)

# Analyze each benchmark run
results = []

for run_id, info in sorted(run_id_map.items(), key=lambda x: x[1]['name']):
    name = info['name']
    start = info['start_time']
    end = info['end_time']

    # Get data for this time window
    # Use timestamp column for comparison since index is integer
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    run_data = df[mask]

    if len(run_data) == 0:
        print(f"\n{name}: NO SYSTEM DATA (no samples in time range)")
        continue

    print(f"\n{name}:")
    print(f"  Time: {start.strftime('%H:%M:%S')} - {end.strftime('%H:%M:%S')} ({len(run_data)} samples)")

    row = {'config': name, 'start': start, 'end': end, 'samples': len(run_data)}

    # Memory analysis
    if 'mem.util.used' in df.columns:
        mem_used = run_data['mem.util.used'].dropna()
        if len(mem_used) > 0:
            mem_mean_gb = mem_used.mean() / (1024**2)
            mem_max_gb = mem_used.max() / (1024**2)
            mem_min_gb = mem_used.min() / (1024**2)
            print(f"  Memory Used: mean={mem_mean_gb:.2f} GB, max={mem_max_gb:.2f} GB, min={mem_min_gb:.2f} GB")
            row['mem_mean_gb'] = mem_mean_gb
            row['mem_max_gb'] = mem_max_gb
            row['mem_delta_gb'] = mem_max_gb - mem_min_gb

    # Active anonymous memory (heap allocations - key for CPU offload)
    if 'mem.util.active_anon' in df.columns:
        active_anon = run_data['mem.util.active_anon'].dropna()
        if len(active_anon) > 0:
            aa_mean_gb = active_anon.mean() / (1024**2)
            aa_max_gb = active_anon.max() / (1024**2)
            print(f"  Active Anon (heap): mean={aa_mean_gb:.2f} GB, max={aa_max_gb:.2f} GB")
            row['active_anon_mean_gb'] = aa_mean_gb
            row['active_anon_max_gb'] = aa_max_gb

    # Cached memory
    if 'mem.util.cached' in df.columns:
        cached = run_data['mem.util.cached'].dropna()
        if len(cached) > 0:
            cached_mean_gb = cached.mean() / (1024**2)
            print(f"  Cached Memory: mean={cached_mean_gb:.2f} GB")
            row['cached_mean_gb'] = cached_mean_gb

    # CPU utilization
    cpu_metrics = ['kernel.all.cpu.user', 'kernel.all.cpu.sys', 'kernel.all.cpu.idle']
    if all(m in df.columns for m in cpu_metrics):
        cpu_user = run_data['kernel.all.cpu.user'].dropna()
        cpu_sys = run_data['kernel.all.cpu.sys'].dropna()
        cpu_idle = run_data['kernel.all.cpu.idle'].dropna()

        if len(cpu_user) > 0 and len(cpu_sys) > 0 and len(cpu_idle) > 0:
            # Calculate percentages (these are in milliseconds per interval)
            total = cpu_user + cpu_sys + cpu_idle
            user_pct = (cpu_user / total * 100).mean()
            sys_pct = (cpu_sys / total * 100).mean()
            idle_pct = (cpu_idle / total * 100).mean()

            print(f"  CPU Utilization: user={user_pct:.1f}%, sys={sys_pct:.1f}%, idle={idle_pct:.1f}%")
            row['cpu_user_pct'] = user_pct
            row['cpu_sys_pct'] = sys_pct
            row['cpu_busy_pct'] = 100 - idle_pct

    results.append(row)

# Create comparison table
results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print("\n" + "="*120)
    print("COMPARATIVE SUMMARY")
    print("="*120)

    # Group by model
    for model in ['Qwen3-0.6B', 'Qwen3-8B']:
        model_data = results_df[results_df['config'].str.contains(model)]

        if len(model_data) > 0:
            print(f"\n{model}:")
            print(f"{'Configuration':<20} {'Mem Mean':<12} {'Mem Max':<12} {'Active Anon':<15} {'CPU Busy':<10}")
            print("-" * 80)

            for _, row in model_data.iterrows():
                config = row['config'].split('-')[-1]
                mem_mean = f"{row.get('mem_mean_gb', 0):.2f} GB" if 'mem_mean_gb' in row else 'N/A'
                mem_max = f"{row.get('mem_max_gb', 0):.2f} GB" if 'mem_max_gb' in row else 'N/A'
                active_anon = f"{row.get('active_anon_mean_gb', 0):.2f} GB" if 'active_anon_mean_gb' in row else 'N/A'
                cpu_busy = f"{row.get('cpu_busy_pct', 0):.1f}%" if 'cpu_busy_pct' in row else 'N/A'

                print(f"{config:<20} {mem_mean:<12} {mem_max:<12} {active_anon:<15} {cpu_busy:<10}")

    # Calculate deltas vs baseline
    print("\n" + "="*120)
    print("MEMORY USAGE DELTA vs BASELINE")
    print("="*120)

    for model in ['Qwen3-0.6B', 'Qwen3-8B']:
        model_data = results_df[results_df['config'].str.contains(model)]
        baseline = model_data[model_data['config'].str.contains('Baseline')]

        if len(baseline) > 0 and 'mem_mean_gb' in baseline.iloc[0]:
            baseline_mem = baseline.iloc[0]['mem_mean_gb']
            baseline_anon = baseline.iloc[0].get('active_anon_mean_gb', 0)

            print(f"\n{model} (Baseline: {baseline_mem:.2f} GB total, {baseline_anon:.2f} GB active anon):")

            for _, row in model_data.iterrows():
                if 'Baseline' in row['config']:
                    continue

                config = row['config'].split('-')[-1]
                if 'mem_mean_gb' in row:
                    mem_delta = row['mem_mean_gb'] - baseline_mem
                    mem_pct = (mem_delta / baseline_mem) * 100

                    anon_delta = row.get('active_anon_mean_gb', 0) - baseline_anon if 'active_anon_mean_gb' in row else 0

                    print(f"  {config:<15}: {mem_delta:+.2f} GB ({mem_pct:+.1f}%), active anon: {anon_delta:+.2f} GB")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)

# Assessment for report
print("\n" + "="*120)
print("RECOMMENDATION: VALUE FOR REPORT")
print("="*120)

print("""
**High Value Metrics to Add:**

1. **Active Anonymous Memory (mem.util.active_anon)**
   - CRITICAL for CPU offload analysis
   - Shows heap allocations where KV cache would be stored
   - Should show INCREASE for Offload/LMCache vs Baseline
   - Directly validates whether CPU offload is working

2. **Total Memory Usage (mem.util.used)**
   - Shows overall memory pressure
   - Helps explain why CPU offload improves performance
   - Delta vs baseline quantifies memory benefit

3. **CPU System Time (kernel.all.cpu.sys)**
   - Reveals kernel overhead from memory management
   - CPU offload may increase sys% due to memory transfers
   - Important for understanding performance tradeoffs

**Medium Value Metrics:**

4. **Cached Memory (mem.util.cached)**
   - Shows file cache usage
   - May change if offload uses disk-backed memory

5. **CPU User Time (kernel.all.cpu.user)**
   - Application-level CPU usage
   - Should remain similar across configs

**Lower Priority:**

6. **IO Wait (kernel.all.cpu.wait.total)**
   - Only relevant if swapping to disk
   - Probably not happening in these tests

**Recommended Addition to Report:**

Add a new section "System Resource Utilization" with:
- Memory usage comparison table (total + active anon)
- CPU utilization breakdown (user/sys/idle)
- Delta analysis showing memory overhead/savings
- Correlation with performance metrics

This would validate whether:
- CPU offload actually uses more system memory ✓
- Memory transfers add measurable CPU overhead ✓
- Trade-offs align with performance results ✓
""")
