#!/usr/bin/env python3
"""
Analyze cache hit rates from vLLM metrics
"""

import pandas as pd
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
        'start_time': pd.to_datetime(data['benchmarks'][0]['start_time'], unit='s'),
        'end_time': pd.to_datetime(data['benchmarks'][-1]['end_time'], unit='s'),
    }

# Find cache metric columns
cache_cols = [c for c in df.columns if 'cache' in c.lower() and 'vllm' in c]

print(f"\nFound {len(cache_cols)} cache-related metrics:")
for col in sorted(cache_cols):
    print(f"  {col}")

# Analyze cache metrics
print("\n" + "="*120)
print("CACHE HIT RATE ANALYSIS")
print("="*120)

# Get prefix cache metrics (generic across all configs)
prefix_queries_cols = [c for c in cache_cols if 'prefix_cache_queries_total' in c]
prefix_hits_cols = [c for c in cache_cols if 'prefix_cache_hits_total' in c]

# Get connector-specific cache metrics
connector_queries_cols = [c for c in cache_cols if 'connector_prefix_cache_queries_total' in c]
connector_hits_cols = [c for c in cache_cols if 'connector_prefix_cache_hits_total' in c]

for run_id, info in sorted(run_id_map.items(), key=lambda x: x[1]['name']):
    name = info['name']
    start = info['start_time']
    end = info['end_time']

    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    run_data = df[mask]

    if len(run_data) == 0:
        continue

    print(f"\n{name}:")

    # Check prefix cache
    for queries_col in prefix_queries_cols:
        if queries_col in run_data.columns:
            queries = run_data[queries_col].dropna()
            if len(queries) > 0:
                total_queries = queries.max() - queries.min()

                # Find corresponding hits column
                hits_col = queries_col.replace('queries_total', 'hits_total')
                if hits_col in run_data.columns:
                    hits = run_data[hits_col].dropna()
                    if len(hits) > 0:
                        total_hits = hits.max() - hits.min()
                        hit_rate = (total_hits / total_queries * 100) if total_queries > 0 else 0

                        print(f"  Prefix Cache: {total_queries:.0f} queries, {total_hits:.0f} hits ({hit_rate:.1f}%)")

    # Check connector cache
    for queries_col in connector_queries_cols:
        if queries_col in run_data.columns:
            queries = run_data[queries_col].dropna()
            if len(queries) > 0:
                total_queries = queries.max() - queries.min()

                # Find corresponding hits column
                hits_col = queries_col.replace('queries_total', 'hits_total')
                if hits_col in run_data.columns:
                    hits = run_data[hits_col].dropna()
                    if len(hits) > 0:
                        total_hits = hits.max() - hits.min()
                        hit_rate = (total_hits / total_queries * 100) if total_queries > 0 else 0

                        print(f"  Connector Cache: {total_queries:.0f} queries, {total_hits:.0f} hits ({hit_rate:.1f}%)")

print("\n" + "="*120)
