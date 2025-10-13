#!/usr/bin/env python3
"""
vLLM KV Cache CPU Offload Benchmark Analysis
Uses PCP Parquet data as primary source
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

print("Loading PCP Parquet data...")
df = pd.read_parquet('benchmark-data.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Map run_ids to configurations based on JSON files
run_id_config_map = {}
configs_info = {
    'benchmark-Qwen3-0.6B-offload.json': ('Qwen3-0.6B', 'offload', 'OffloadingConnector'),
    'benchmark-Qwen3-0.6B-lmcache.json': ('Qwen3-0.6B', 'lmcache', 'LMCacheConnectorV1'),
    'benchmark-Qwen3-0.6B-default.json': ('Qwen3-0.6B', 'default', 'Baseline'),
    'benchmark-Qwen3-8B-offload.json': ('Qwen3-8B', 'offload', 'OffloadingConnector'),
    'benchmark-Qwen3-8B-lmcache.json': ('Qwen3-8B', 'lmcache', 'LMCacheConnectorV1'),
    'benchmark-Qwen3-8B-default.json': ('Qwen3-8B', 'default', 'Baseline'),
}

for json_file, (model, config, config_full) in configs_info.items():
    with open(json_file) as f:
        data = json.load(f)
    run_id = data['benchmarks'][0]['run_id']
    run_id_config_map[run_id] = {
        'model': model,
        'config': config,
        'config_full': config_full,
        'json_file': json_file
    }

print(f"\nMapped {len(run_id_config_map)} configurations:")
for run_id, info in run_id_config_map.items():
    print(f"  {run_id[:8]}... => {info['model']} - {info['config']}")

# Extract metrics for each run_id
results = []

for run_id, cfg in run_id_config_map.items():
    # Find instance IDs associated with this run_id
    instance_ids = []
    for col in df.columns:
        if f'guidellm.run_id[' in col:
            instance_id = col.split('[')[1].split(']')[0]
            run_id_vals = df[col].dropna().unique()
            if len(run_id_vals) > 0 and run_id_vals[0] == run_id:
                instance_ids.append(instance_id)

    print(f"\n{cfg['model']}-{cfg['config']}: {len(instance_ids)} strategy instances")

    # For each instance (strategy), get the final metrics
    for instance_id in instance_ids:
        metrics = {}

        # Extract key metrics
        metric_cols = {
            'output_tps': f'guidellm.output_tokens_per_second.total.mean[{instance_id}]',
            'total_tps': f'guidellm.tokens_per_second.total.mean[{instance_id}]',
            'ttft_mean': f'guidellm.time_to_first_token_ms.total.mean[{instance_id}]',
            'ttft_p50': f'guidellm.time_to_first_token_ms.total.median[{instance_id}]',
            'ttft_p99': f'guidellm.time_to_first_token_ms.total.max[{instance_id}]',
            'tpot_mean': f'guidellm.inter_token_latency_ms.total.mean[{instance_id}]',
            'tpot_p50': f'guidellm.inter_token_latency_ms.total.median[{instance_id}]',
            'tpot_p99': f'guidellm.inter_token_latency_ms.total.max[{instance_id}]',
            'latency_mean': f'guidellm.request_latency.total.mean[{instance_id}]',
            'rps': f'guidellm.requests_per_second.total.mean[{instance_id}]',
            'req_success': f'guidellm.run_stats.requests_made.successful[{instance_id}]',
            'req_error': f'guidellm.run_stats.requests_made.errored[{instance_id}]',
        }

        for key, col in metric_cols.items():
            if col in df.columns:
                val = df[col].dropna()
                metrics[key] = val.iloc[-1] if len(val) > 0 else np.nan
            else:
                metrics[key] = np.nan

        results.append({
            'run_id': run_id,
            'instance_id': instance_id,
            'model': cfg['model'],
            'config': cfg['config'],
            'config_full': cfg['config_full'],
            **metrics
        })

results_df = pd.DataFrame(results)

# Calculate summary statistics per configuration
summary_data = []
for run_id, cfg in run_id_config_map.items():
    run_data = results_df[results_df['run_id'] == run_id]

    summary = {
        'model': cfg['model'],
        'config': cfg['config'],
        'config_full': cfg['config_full'],
        'max_output_tps': run_data['output_tps'].max(),
        'mean_output_tps': run_data['output_tps'].mean(),
        'max_rps': run_data['rps'].max(),
        'mean_ttft': run_data['ttft_mean'].mean(),
        'p50_ttft': run_data['ttft_p50'].mean(),
        'p99_ttft': run_data['ttft_p99'].mean(),
        'mean_tpot': run_data['tpot_mean'].mean(),
        'p50_tpot': run_data['tpot_p50'].mean(),
        'p99_tpot': run_data['tpot_p99'].mean(),
        'mean_latency': run_data['latency_mean'].mean(),
        'total_success': run_data['req_success'].sum(),
        'total_error': run_data['req_error'].sum(),
        'num_strategies': len(run_data),
    }
    summary_data.append(summary)

summary_df = pd.DataFrame(summary_data)

# Print results
print("\n" + "="*130)
print("VLLM KV CACHE CPU OFFLOAD EVALUATION - RESULTS")
print("="*130)
print("\nTest Configuration:")
print("  - Models: Qwen3-0.6B, Qwen3-8B")
print("  - Configurations: OffloadingConnector, LMCacheConnectorV1, Baseline")
print("  - Workload: 256 input tokens, 128 output tokens")
print("  - Duration: 30 seconds per run")
print("  - Rate: Sweep (10 strategies per configuration)")
print("\n" + "-"*130)
print(f"{'Model':<15} {'Configuration':<22} {'Max TPS':>10} {'Avg TTFT':>10} {'Avg TPOT':>10} {'Success':>10}")
print("-"*130)

for _, row in summary_df.iterrows():
    print(f"{row['model']:<15} {row['config_full']:<22} {row['max_output_tps']:>10.1f} "
          f"{row['mean_ttft']:>10.1f} {row['mean_tpot']:>10.1f} {row['total_success']:>10.0f}")

# Performance comparisons
print("\n" + "="*130)
print("PERFORMANCE vs BASELINE")
print("="*130)

for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    print(f"\n{model}:")
    baseline = summary_df[(summary_df['model'] == model) & (summary_df['config'] == 'default')].iloc[0]

    for config in ['offload', 'lmcache']:
        cfg_row = summary_df[(summary_df['model'] == model) & (summary_df['config'] == config)].iloc[0]

        tps_delta = ((cfg_row['max_output_tps'] / baseline['max_output_tps']) - 1) * 100
        ttft_delta = ((cfg_row['mean_ttft'] / baseline['mean_ttft']) - 1) * 100
        tpot_delta = ((cfg_row['mean_tpot'] / baseline['mean_tpot']) - 1) * 100

        print(f"\n  {cfg_row['config_full']}:")
        print(f"    Throughput: {tps_delta:+6.1f}% ({cfg_row['max_output_tps']:.1f} vs {baseline['max_output_tps']:.1f} tok/s)")
        print(f"    TTFT:       {ttft_delta:+6.1f}% ({cfg_row['mean_ttft']:.1f} vs {baseline['mean_ttft']:.1f} ms)")
        print(f"    TPOT:       {tpot_delta:+6.1f}% ({cfg_row['mean_tpot']:.1f} vs {baseline['mean_tpot']:.1f} ms)")

# Save results
results_df.to_csv('detailed_results.csv', index=False)
summary_df.to_csv('summary_results.csv', index=False)
print("\n\nSaved detailed_results.csv and summary_results.csv")

print("\nAnalysis complete!")
