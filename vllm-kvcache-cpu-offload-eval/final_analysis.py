#!/usr/bin/env python3
"""
Comprehensive vLLM KV Cache CPU Offload Analysis
Primary: GuideLLM JSON files
Secondary: PCP data for system metrics
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Configuration mapping
configs = [
    ('benchmark-Qwen3-0.6B-offload.json', 'Qwen3-0.6B', 'Offload', 'OffloadingConnector'),
    ('benchmark-Qwen3-0.6B-lmcache.json', 'Qwen3-0.6B', 'LMCache', 'LMCacheConnectorV1'),
    ('benchmark-Qwen3-0.6B-default.json', 'Qwen3-0.6B', 'Baseline', 'GPU Only'),
    ('benchmark-Qwen3-8B-offload.json', 'Qwen3-8B', 'Offload', 'OffloadingConnector'),
    ('benchmark-Qwen3-8B-lmcache.json', 'Qwen3-8B', 'LMCache', 'LMCacheConnectorV1'),
    ('benchmark-Qwen3-8B-default.json', 'Qwen3-8B', 'Baseline', 'GPU Only'),
]

def load_guidellm_json(filename):
    """Load and extract metrics from GuideLLM JSON"""
    with open(filename) as f:
        data = json.load(f)

    results = []
    for idx, bench in enumerate(data['benchmarks']):
        m = bench.get('metrics', {})
        args = bench.get('args', {})
        profile = args.get('profile', {})

        # Extract request rate from profile
        measured_rates = profile.get('measured_rates', [])
        request_rate = measured_rates[idx] if idx < len(measured_rates) else 0

        # Helper to extract metric values
        def get_metric(metric_dict, stat='mean'):
            if isinstance(metric_dict, dict):
                total = metric_dict.get('total', {})
                if isinstance(total, dict):
                    return total.get(stat, 0)
            return 0

        results.append({
            'request_rate': request_rate,
            'output_tps': get_metric(m.get('output_tokens_per_second')),
            'total_tps': get_metric(m.get('tokens_per_second')),
            'ttft_mean': get_metric(m.get('time_to_first_token_ms')),
            'ttft_p50': get_metric(m.get('time_to_first_token_ms'), 'median'),
            'ttft_p99': get_metric(m.get('time_to_first_token_ms'), 'p99'),
            'tpot_mean': get_metric(m.get('inter_token_latency_ms')),
            'tpot_p50': get_metric(m.get('inter_token_latency_ms'), 'median'),
            'tpot_p99': get_metric(m.get('inter_token_latency_ms'), 'p99'),
            'latency_mean': get_metric(m.get('request_latency')),
            'rps': get_metric(m.get('requests_per_second')),
            'completed': bench.get('request_totals', {}).get('completed', 0),
            'errored': bench.get('request_totals', {}).get('errored', 0),
        })
    return pd.DataFrame(results)

# Load all data
all_data = {}
summary_data = []

for json_file, model, config_short, config_full in configs:
    df = load_guidellm_json(json_file)
    df['model'] = model
    df['config'] = config_short
    all_data[f'{model}-{config_short}'] = df

    # Calculate summary
    summary = {
        'model': model,
        'config': config_short,
        'config_full': config_full,
        'max_output_tps': df['output_tps'].max(),
        'mean_output_tps': df['output_tps'].mean(),
        'max_rps': df['rps'].max(),
        'mean_ttft': df['ttft_mean'].mean(),
        'p50_ttft': df['ttft_p50'].mean(),
        'p99_ttft': df['ttft_p99'].mean(),
        'mean_tpot': df['tpot_mean'].mean(),
        'p50_tpot': df['tpot_p50'].mean(),
        'p99_tpot': df['tpot_p99'].mean(),
        'mean_latency': df['latency_mean'].mean(),
        'total_completed': df['completed'].sum(),
        'total_errored': df['errored'].sum(),
    }
    summary_data.append(summary)

summary_df = pd.DataFrame(summary_data)
combined_df = pd.concat(all_data.values(), ignore_index=True)

# Print summary table
print("\n" + "="*140)
print("vLLM KV CACHE CPU OFFLOAD EVALUATION - BENCHMARK RESULTS")
print("="*140)
print("\nConfiguration:")
print("  Workload: 256 input tokens → 128 output tokens")
print("  Duration: 30 seconds per benchmark")
print("  Rate Strategy: Sweep (10 different request rates)")
print("  Hardware: 2x GPU (Tensor Parallel)")
print("\n" + "-"*140)
print(f"{'Model':<15} {'Configuration':<25} {'Max TPS':>12} {'Avg TTFT':>12} {'Avg TPOT':>12} {'Requests':>12}")
print("-"*140)

for _, row in summary_df.iterrows():
    print(f"{row['model']:<15} {row['config_full']:<25} {row['max_output_tps']:>12.1f} "
          f"{row['mean_ttft']:>12.1f} {row['mean_tpot']:>12.1f} {row['total_completed']:>12.0f}")

# Performance analysis
print("\n" + "="*140)
print("PERFORMANCE COMPARISON vs BASELINE")
print("="*140)

for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    baseline = summary_df[(summary_df['model'] == model) & (summary_df['config'] == 'Baseline')].iloc[0]

    print(f"\n{model} (Baseline: {baseline['max_output_tps']:.1f} tok/s, TTFT: {baseline['mean_ttft']:.1f}ms, TPOT: {baseline['mean_tpot']:.1f}ms)")
    print("-" * 100)

    for config in ['Offload', 'LMCache']:
        cfg_row = summary_df[(summary_df['model'] == model) & (summary_df['config'] == config)].iloc[0]

        tps_pct = ((cfg_row['max_output_tps'] / baseline['max_output_tps']) - 1) * 100
        ttft_pct = ((cfg_row['mean_ttft'] / baseline['mean_ttft']) - 1) * 100
        tpot_pct = ((cfg_row['mean_tpot'] / baseline['mean_tpot']) - 1) * 100

        print(f"\n{cfg_row['config_full']:>25}:")
        print(f"{'':>10} Throughput: {tps_pct:>7.1f}%  ({cfg_row['max_output_tps']:>8.1f} tok/s)")
        print(f"{'':>10} TTFT:       {ttft_pct:>7.1f}%  ({cfg_row['mean_ttft']:>8.1f} ms)")
        print(f"{'':>10} TPOT:       {tpot_pct:>7.1f}%  ({cfg_row['mean_tpot']:>8.1f} ms)")

# Save results
summary_df.to_csv('summary.csv', index=False)
combined_df.to_csv('detailed.csv', index=False)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('vLLM KV Cache CPU Offload Evaluation', fontsize=18, fontweight='bold', y=0.995)

# Plot 1: Throughput comparison
ax = axes[0, 0]
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    data = summary_df[summary_df['model'] == model]
    x = np.arange(len(data))
    offset = -0.2 if model == 'Qwen3-0.6B' else 0.2
    ax.bar(x + offset, data['max_output_tps'], width=0.35, label=model, alpha=0.85)

ax.set_ylabel('Output Tokens/sec', fontsize=13, fontweight='bold')
ax.set_title('Maximum Output Throughput', fontsize=14, fontweight='bold')
ax.set_xticks(range(3))
ax.set_xticklabels(['Baseline', 'LMCache', 'Offload'], fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: TTFT
ax = axes[0, 1]
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    data = summary_df[summary_df['model'] == model]
    x = np.arange(len(data))
    offset = -0.2 if model == 'Qwen3-0.6B' else 0.2
    ax.bar(x + offset, data['mean_ttft'], width=0.35, label=model, alpha=0.85)

ax.set_ylabel('Time to First Token (ms)', fontsize=13, fontweight='bold')
ax.set_title('Average TTFT', fontsize=14, fontweight='bold')
ax.set_xticks(range(3))
ax.set_xticklabels(['Baseline', 'LMCache', 'Offload'], fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: TPOT
ax = axes[0, 2]
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    data = summary_df[summary_df['model'] == model]
    x = np.arange(len(data))
    offset = -0.2 if model == 'Qwen3-0.6B' else 0.2
    ax.bar(x + offset, data['mean_tpot'], width=0.35, label=model, alpha=0.85)

ax.set_ylabel('Inter-Token Latency (ms)', fontsize=13, fontweight='bold')
ax.set_title('Average TPOT', fontsize=14, fontweight='bold')
ax.set_xticks(range(3))
ax.set_xticklabels(['Baseline', 'LMCache', 'Offload'], fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Plots 4-5: Throughput vs Request Rate
for idx, model in enumerate(['Qwen3-0.6B', 'Qwen3-8B']):
    ax = axes[1, idx]

    for config in ['Baseline', 'LMCache', 'Offload']:
        data = combined_df[(combined_df['model'] == model) & (combined_df['config'] == config)]
        data = data.sort_values('request_rate')
        ax.plot(data['request_rate'], data['output_tps'],
                marker='o', label=config, linewidth=2.5, markersize=7, alpha=0.85)

    ax.set_xlabel('Request Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Output Throughput (tok/s)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

# Plot 6: Relative performance heatmap
ax = axes[1, 2]
rel_perf = []
for model in ['Qwen3-0.6B', 'Qwen3-8B']:
    baseline_tps = summary_df[(summary_df['model'] == model) & (summary_df['config'] == 'Baseline')]['max_output_tps'].iloc[0]
    row = []
    for config in ['Baseline', 'Offload', 'LMCache']:
        cfg_tps = summary_df[(summary_df['model'] == model) & (summary_df['config'] == config)]['max_output_tps'].iloc[0]
        row.append((cfg_tps / baseline_tps) * 100)
    rel_perf.append(row)

rel_df = pd.DataFrame(rel_perf, index=['Qwen3-0.6B', 'Qwen3-8B'], columns=['Baseline', 'Offload', 'LMCache'])
sns.heatmap(rel_df, annot=True, fmt='.1f', cmap='RdYlGn', center=100, vmin=0, vmax=150,
            cbar_kws={'label': '% of Baseline'}, ax=ax, linewidths=1, linecolor='white')
ax.set_title('Relative Throughput Performance', fontsize=14, fontweight='bold')
ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')

print("\n\nFiles created:")
print("  - summary.csv (aggregate metrics)")
print("  - detailed.csv (all strategy results)")
print("  - benchmark_results.png (visualizations)")
print("\n" + "="*140)
print("Analysis complete!")
print("="*140 + "\n")
