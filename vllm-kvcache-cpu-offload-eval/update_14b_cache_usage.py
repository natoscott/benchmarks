#!/usr/bin/env python3
"""
Update cache_analysis_results_with_14b.csv with correct cache usage values.
"""

import pandas as pd

# Load existing cache analysis
df_cache = pd.read_csv('cache_analysis_results_with_14b.csv')

# Load 14B cache usage data
df_usage = pd.read_csv('cache_usage_14b.csv')

# Update the 14B rows with correct cache usage
for idx, usage_row in df_usage.iterrows():
    uuid = usage_row['uuid']
    mask = df_cache['uuid'] == uuid
    df_cache.loc[mask, 'cache_usage_mean'] = usage_row['cache_usage_mean']
    df_cache.loc[mask, 'cache_usage_max'] = usage_row['cache_usage_max']

# Save updated file
df_cache.to_csv('cache_analysis_results_with_14b.csv', index=False)

# Print summary
print("Updated cache_analysis_results_with_14b.csv with 14B cache usage data")
print("\n=== Updated 14B Cache Usage Summary ===")
df_14b = df_cache[df_cache['model'] == 'Qwen3-14B']
for config in df_14b['config_label'].unique():
    config_data = df_14b[(df_14b['config_label'] == config) & (~df_14b['is_warmup'])]
    if len(config_data) > 0:
        print(f"\n{config}:")
        print(f"  Avg cache usage: {config_data['cache_usage_mean'].mean():.2f}%")
        print(f"  Max cache usage: {config_data['cache_usage_max'].max():.2f}%")
