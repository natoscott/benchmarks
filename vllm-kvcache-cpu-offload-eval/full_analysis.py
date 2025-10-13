#!/usr/bin/env python3
"""
Comprehensive analysis of vLLM KV Cache CPU Offload Benchmarks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)

# Load data
print("Loading Parquet data...")
df = pd.read_parquet('benchmark-data.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# The 10 benchmark IDs we found - need to map to 6 configurations
benchmark_ids = [
    '545714d3-a25f-4d22-afa3-0c1370e7d14c',  # 09:59
    'b564c4b3-6e2e-4c1e-b3c4-898eadc4f725',  # 10:11
    '4e740361-4ca0-4c8e-9508-0b239a4d1fbe',  # 10:12
    '73e0d116-351d-4f29-812c-42f237f6f31f',  # 10:17
    '4656ce81-09ac-4c46-b645-6f32785ceb95',  # 10:19
    'a1988645-e241-4537-9da6-b8e1b1d87f34',  # 10:21
    'a8672574-4708-48fd-9f83-c26ed95259fc',  # 10:32
    'dd923c82-bbfb-4abe-9f57-3e4115b2ddd7',  # 10:32
    'fdb6f648-cd99-45ce-b44b-3eb6c271cc66',  # 10:55
    '6814f513-c9c1-4b8d-88b2-0a8b780d8a8f',  # 10:56
]

# Map benchmark IDs to configurations based on timing order
# Qwen3-0.6B: offload, lmcache, default (first 3 configurations)
# Qwen3-8B: offload, lmcache, default (last 3 configurations)
config_map = {
    benchmark_ids[0]: 'Qwen3-0.6B-offload',
    benchmark_ids[1]: 'Qwen3-0.6B-lmcache',
    benchmark_ids[2]: 'Qwen3-0.6B-default',
    benchmark_ids[3]: 'Qwen3-8B-offload',
    benchmark_ids[4]: 'Qwen3-8B-lmcache',
    benchmark_ids[5]: 'Qwen3-8B-default',
}

# Extract key metrics for each benchmark run
results = []
for bid, config in config_map.items():
    # Get metrics for this benchmark ID
    output_tps_col = f'guidellm.output_tokens_per_second.total.mean[{bid}]'
    total_tps_col = f'guidellm.tokens_per_second.total.mean[{bid}]'
    ttft_col = f'guidellm.time_to_first_token_ms.total.mean[{bid}]'
    tpot_col = f'guidellm.inter_token_latency_ms.total.mean[{bid}]'
    latency_col = f'guidellm.request_latency.total.mean[{bid}]'
    rps_col = f'guidellm.requests_per_second.total.mean[{bid}]'
    req_success_col = f'guidellm.run_stats.requests_made.successful[{bid}]'
    req_error_col = f'guidellm.run_stats.requests_made.errored[{bid}]'

    # Get final values (last non-null value)
    output_tps = df[output_tps_col].dropna().iloc[-1] if output_tps_col in df.columns else np.nan
    total_tps = df[total_tps_col].dropna().iloc[-1] if total_tps_col in df.columns else np.nan
    ttft = df[ttft_col].dropna().iloc[-1] if ttft_col in df.columns else np.nan
    tpot = df[tpot_col].dropna().iloc[-1] if tpot_col in df.columns else np.nan
    latency = df[latency_col].dropna().iloc[-1] if latency_col in df.columns else np.nan
    rps = df[rps_col].dropna().iloc[-1] if rps_col in df.columns else np.nan
    req_success = df[req_success_col].dropna().iloc[-1] if req_success_col in df.columns else 0
    req_error = df[req_error_col].dropna().iloc[-1] if req_error_col in df.columns else 0

    model, config_type = config.rsplit('-', 1)
    results.append({
        'model': model,
        'config': config_type,
        'output_throughput': output_tps,
        'total_throughput': total_tps,
        'ttft_ms': ttft,
        'tpot_ms': tpot,
        'request_latency_s': latency,
        'requests_per_sec': rps,
        'successful_requests': req_success,
        'errored_requests': req_error,
    })

results_df = pd.DataFrame(results)

# Print summary
print("\n" + "="*100)
print("BENCHMARK RESULTS SUMMARY")
print("="*100)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('benchmark_summary.csv', index=False)
print("\nSaved summary to benchmark_summary.csv")

# Create comparison visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('vLLM KV Cache CPU Offload Evaluation', fontsize=16, fontweight='bold')

# Plot 1: Output Throughput
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    model_data = results_df[results_df['model'] == model]
    x = range(len(model_data))
    axes[0, 0].bar([i + (0 if model == 'Qwen3-0.6B' else 0.35) for i in x],
                   model_data['output_throughput'],
                   width=0.35, label=model)
axes[0, 0].set_ylabel('Output Tokens/sec')
axes[0, 0].set_title('Output Token Throughput')
axes[0, 0].set_xticks([0.175, 1.175, 2.175])
axes[0, 0].set_xticklabels(['Offload', 'LMCache', 'Default'])
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: TTFT
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    model_data = results_df[results_df['model'] == model]
    x = range(len(model_data))
    axes[0, 1].bar([i + (0 if model == 'Qwen3-0.6B' else 0.35) for i in x],
                   model_data['ttft_ms'],
                   width=0.35, label=model)
axes[0, 1].set_ylabel('Time to First Token (ms)')
axes[0, 1].set_title('Time to First Token (TTFT)')
axes[0, 1].set_xticks([0.175, 1.175, 2.175])
axes[0, 1].set_xticklabels(['Offload', 'LMCache', 'Default'])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: TPOT
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    model_data = results_df[results_df['model'] == model]
    x = range(len(model_data))
    axes[0, 2].bar([i + (0 if model == 'Qwen3-0.6B' else 0.35) for i in x],
                   model_data['tpot_ms'],
                   width=0.35, label=model)
axes[0, 2].set_ylabel('Inter-Token Latency (ms)')
axes[0, 2].set_title('Time Per Output Token (TPOT)')
axes[0, 2].set_xticks([0.175, 1.175, 2.175])
axes[0, 2].set_xticklabels(['Offload', 'LMCache', 'Default'])
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Request Latency
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    model_data = results_df[results_df['model'] == model]
    x = range(len(model_data))
    axes[1, 0].bar([i + (0 if model == 'Qwen3-0.6B' else 0.35) for i in x],
                   model_data['request_latency_s'],
                   width=0.35, label=model)
axes[1, 0].set_ylabel('Request Latency (s)')
axes[1, 0].set_title('End-to-End Request Latency')
axes[1, 0].set_xticks([0.175, 1.175, 2.175])
axes[1, 0].set_xticklabels(['Offload', 'LMCache', 'Default'])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Requests Per Second
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    model_data = results_df[results_df['model'] == model]
    x = range(len(model_data))
    axes[1, 1].bar([i + (0 if model == 'Qwen3-0.6B' else 0.35) for i in x],
                   model_data['requests_per_sec'],
                   width=0.35, label=model)
axes[1, 1].set_ylabel('Requests/sec')
axes[1, 1].set_title('Request Throughput')
axes[1, 1].set_xticks([0.175, 1.175, 2.175])
axes[1, 1].set_xticklabels(['Offload', 'LMCache', 'Default'])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Success Rate
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    model_data = results_df[results_df['model'] == model]
    x = range(len(model_data))
    total_requests = model_data['successful_requests'] + model_data['errored_requests']
    success_rate = (model_data['successful_requests'] / total_requests * 100).fillna(0)
    axes[1, 2].bar([i + (0 if model == 'Qwen3-0.6B' else 0.35) for i in x],
                   success_rate,
                   width=0.35, label=model)
axes[1, 2].set_ylabel('Success Rate (%)')
axes[1, 2].set_title('Request Success Rate')
axes[1, 2].set_xticks([0.175, 1.175, 2.175])
axes[1, 2].set_xticklabels(['Offload', 'LMCache', 'Default'])
axes[1, 2].set_ylim([0, 105])
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to benchmark_comparison.png")

print("\nAnalysis complete!")
