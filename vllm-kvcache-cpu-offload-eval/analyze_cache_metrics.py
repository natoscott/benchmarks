#!/usr/bin/env python3
"""
Analyze KV cache metrics per benchmark run.
Note: The connector cache metrics appear to be rates (per second) not counters.
"""

import pandas as pd
import numpy as np

# Load the benchmark mapping to get time ranges
df_benchmarks = pd.read_csv('final_config_mapping.csv', parse_dates=['start_time', 'end_time'])

# Load cache metrics
df_cache = pd.read_csv('cache_metrics_timeseries.csv', parse_dates=['Time'])

# Simplify column names
column_mapping = {}
for col in df_cache.columns:
    if 'kv_cache_usage_perc' in col and 'Qwen3-0.6B' in col:
        column_mapping[col] = 'cache_usage_0.6B'
    elif 'kv_cache_usage_perc' in col and 'Qwen3-8B' in col:
        column_mapping[col] = 'cache_usage_8B'
    elif 'prefix_cache_queries_total' in col and 'Qwen3-0.6B' in col and 'connector' not in col:
        column_mapping[col] = 'prefix_queries_0.6B'
    elif 'prefix_cache_queries_total' in col and 'Qwen3-8B' in col and 'connector' not in col:
        column_mapping[col] = 'prefix_queries_8B'
    elif 'prefix_cache_hits_total' in col and 'Qwen3-0.6B' in col and 'connector' not in col:
        column_mapping[col] = 'prefix_hits_0.6B'
    elif 'prefix_cache_hits_total' in col and 'Qwen3-8B' in col and 'connector' not in col:
        column_mapping[col] = 'prefix_hits_8B'
    elif 'connector_prefix_cache_queries_total' in col and 'Qwen3-0.6B' in col:
        column_mapping[col] = 'connector_queries_rate_0.6B'
    elif 'connector_prefix_cache_queries_total' in col and 'Qwen3-8B' in col:
        column_mapping[col] = 'connector_queries_rate_8B'
    elif 'connector_prefix_cache_hits_total' in col and 'Qwen3-0.6B' in col:
        column_mapping[col] = 'connector_hits_rate_0.6B'
    elif 'connector_prefix_cache_hits_total' in col and 'Qwen3-8B' in col:
        column_mapping[col] = 'connector_hits_rate_8B'

df_cache = df_cache.rename(columns=column_mapping)

print(f"Loaded {len(df_cache)} cache metric samples")
print(f"Analyzing {len(df_benchmarks)} benchmark runs")

# Analyze each benchmark run
results = []

for idx, bench in df_benchmarks.iterrows():
    # Filter cache data for this benchmark time range
    mask = (df_cache['Time'] >= bench['start_time']) & (df_cache['Time'] <= bench['end_time'])
    cache_data = df_cache[mask].copy()

    if len(cache_data) == 0:
        print(f"Warning: No cache data for benchmark {bench['uuid'][:8]}")
        continue

    # Determine which model column to use
    model_suffix = '0.6B' if bench['model'] == 'Qwen3-0.6B' else '8B'

    # Calculate cache usage statistics
    cache_usage_col = f'cache_usage_{model_suffix}'
    cache_usage_mean = cache_data[cache_usage_col].mean()
    cache_usage_max = cache_data[cache_usage_col].max()

    # Calculate connector cache metrics (these are rates, so we average them)
    connector_queries_rate_col = f'connector_queries_rate_{model_suffix}'
    connector_hits_rate_col = f'connector_hits_rate_{model_suffix}'

    # Get the data where connector is active (non-zero, non-null)
    connector_data = cache_data[cache_data[connector_queries_rate_col] > 0]

    if len(connector_data) > 0:
        connector_query_rate_mean = connector_data[connector_queries_rate_col].mean()
        connector_query_rate_max = connector_data[connector_queries_rate_col].max()
        connector_hit_rate_mean = connector_data[connector_hits_rate_col].mean()
        connector_hit_rate_max = connector_data[connector_hits_rate_col].max()

        # Calculate hit ratio (hits/queries)
        connector_hit_ratio = (connector_hit_rate_mean / connector_query_rate_mean * 100) if connector_query_rate_mean > 0 else 0
    else:
        connector_query_rate_mean = 0
        connector_query_rate_max = 0
        connector_hit_rate_mean = 0
        connector_hit_rate_max = 0
        connector_hit_ratio = 0

    results.append({
        'uuid': bench['uuid'],
        'model': bench['model'],
        'config_label': bench['config_label'],
        'concurrency': bench['concurrency'],
        'is_warmup': bench['is_warmup'],
        'throughput': bench['tokens_per_second'],
        'duration_sec': (bench['end_time'] - bench['start_time']).total_seconds(),

        # Cache usage
        'cache_usage_mean': cache_usage_mean,
        'cache_usage_max': cache_usage_max,

        # Connector cache rates (queries/sec and hits/sec)
        'connector_query_rate_mean': connector_query_rate_mean,
        'connector_query_rate_max': connector_query_rate_max,
        'connector_hit_rate_mean': connector_hit_rate_mean,
        'connector_hit_rate_max': connector_hit_rate_max,
        'connector_hit_ratio': connector_hit_ratio,
    })

df_results = pd.DataFrame(results)

# Save results
df_results.to_csv('cache_analysis_results.csv', index=False)
print(f"\nSaved cache analysis to cache_analysis_results.csv")

# Print summary by configuration
print("\n=== Cache Metrics Summary ===\n")

for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    print(f"\n{model}:")
    model_data = df_results[(df_results['model'] == model) & (~df_results['is_warmup'])]

    for config in ['default', 'offload']:
        config_data = model_data[model_data['config_label'] == config]
        if len(config_data) == 0:
            continue

        print(f"\n  {config}:")
        print(f"    Cache usage: {config_data['cache_usage_mean'].mean():.1f}% avg, {config_data['cache_usage_max'].mean():.1f}% max")
        print(f"    Connector cache query rate: {config_data['connector_query_rate_mean'].mean():.1f} queries/sec avg, {config_data['connector_query_rate_max'].mean():.1f} max")
        print(f"    Connector cache hit rate: {config_data['connector_hit_rate_mean'].mean():.1f} hits/sec avg, {config_data['connector_hit_rate_max'].mean():.1f} max")
        print(f"    Connector cache hit ratio: {config_data['connector_hit_ratio'].mean():.1f}%")
        print(f"    Throughput: {config_data['throughput'].mean():.1f} tok/s avg")

print("\nAnalysis complete!")
