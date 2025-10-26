#!/usr/bin/env python3
"""
Extract cache usage metrics for Qwen3-14B benchmark runs.
Metric is in 0-1 range, multiply by 100 for percentage.
"""

import pandas as pd
import subprocess
import io

# Load benchmark results to get time ranges
df_bench = pd.read_csv('qwen3_14b_results.csv', parse_dates=['start_time', 'end_time'])

results = []

for idx, row in df_bench.iterrows():
    uuid = row['uuid']
    start = row['start_time'].strftime('%Y-%m-%d %H:%M:%S')
    end = row['end_time'].strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract cache usage for this run
    cmd = [
        'pmrep', '-a', 'pcp-archive-20251026', '-z',
        '-S', f'@{start}', '-T', f'@{end}',
        '-t', '1sec', '-o', 'csv',
        'openmetrics.vllm.vllm.kv_cache_usage_perc'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Parse CSV output
        cache_df = pd.read_csv(io.StringIO(result.stdout), comment='#')
        
        # Get 14B column (should be the last column with "14B" in the name)
        col_14b = [c for c in cache_df.columns if '14B' in c]
        
        if col_14b:
            # Metric is 0-1 range, multiply by 100 for percentage
            cache_values = cache_df[col_14b[0]].dropna() * 100
            cache_mean = cache_values.mean() if len(cache_values) > 0 else 0.0
            cache_max = cache_values.max() if len(cache_values) > 0 else 0.0
        else:
            cache_mean = 0.0
            cache_max = 0.0
    else:
        cache_mean = 0.0
        cache_max = 0.0
    
    results.append({
        'uuid': uuid,
        'cache_usage_mean': cache_mean,
        'cache_usage_max': cache_max
    })
    
    print(f"Run {idx+1}/20: UUID {uuid[:8]}... cache_usage_mean={cache_mean:.2f}%, max={cache_max:.2f}%")

df_cache_usage = pd.DataFrame(results)
df_cache_usage.to_csv('cache_usage_14b.csv', index=False)
print(f"\nSaved cache usage data to cache_usage_14b.csv")
print(f"\nOverall average cache usage: {df_cache_usage['cache_usage_mean'].mean():.2f}%")
print(f"Overall max cache usage: {df_cache_usage['cache_usage_max'].max():.2f}%")
