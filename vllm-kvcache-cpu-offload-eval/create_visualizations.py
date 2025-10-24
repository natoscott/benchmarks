#!/usr/bin/env python3
"""
Create visualizations for the benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set consistent style per visualization-palette skill
sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Load results
df = pd.read_csv('final_config_mapping.csv', parse_dates=['start_time', 'end_time'])

# Filter out warmup runs and lmcache (excluded from analysis)
df_plot = df[(~df['is_warmup']) & (df['config_label'] != 'lmcache')].copy()

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('vLLM KV Cache CPU Offload Benchmark Results - Concurrency-Based Testing',
             fontsize=16, fontweight='bold')

# Get colors from muted palette
palette_colors = sns.color_palette("muted")
colors = {'default': palette_colors[0], 'offload': palette_colors[1]}
config_order = ['default', 'offload']

# Plot 1: Throughput vs Concurrency - Qwen3-0.6B
ax = axes[0, 0]
for config in config_order:
    data = df_plot[(df_plot['model'] == 'Qwen3-0.6B') & (df_plot['config_label'] == config)]
    ax.plot(data['concurrency'], data['tokens_per_second'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax.set_xlabel('Concurrency', fontsize=12)
ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
ax.set_title('Qwen3-0.6B: Throughput vs Concurrency', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 2: Throughput vs Concurrency - Qwen3-8B
ax = axes[0, 1]
for config in config_order:
    data = df_plot[(df_plot['model'] == 'Qwen3-8B') & (df_plot['config_label'] == config)]
    ax.plot(data['concurrency'], data['tokens_per_second'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax.set_xlabel('Concurrency', fontsize=12)
ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
ax.set_title('Qwen3-8B: Throughput vs Concurrency', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 3: Bar chart - Average Throughput Comparison
ax = axes[0, 2]
summary = df_plot.groupby(['model', 'config_label'])['tokens_per_second'].mean().reset_index()
x = np.arange(len(summary[summary['model'] == 'Qwen3-0.6B']))
width = 0.35
models = ['Qwen3-0.6B', 'Qwen3-8B']
for i, model in enumerate(models):
    model_data = summary[summary['model'] == model]
    ax.bar(x + i*width, model_data['tokens_per_second'], width, label=model, alpha=0.8)
ax.set_xlabel('Configuration', fontsize=12)
ax.set_ylabel('Avg Throughput (tokens/sec)', fontsize=12)
ax.set_title('Average Throughput Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(config_order)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: TTFT vs Concurrency - Qwen3-0.6B
ax = axes[1, 0]
for config in config_order:
    data = df_plot[(df_plot['model'] == 'Qwen3-0.6B') & (df_plot['config_label'] == config)]
    ax.plot(data['concurrency'], data['ttft_ms'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax.set_xlabel('Concurrency', fontsize=12)
ax.set_ylabel('TTFT (ms)', fontsize=12)
ax.set_title('Qwen3-0.6B: Time to First Token vs Concurrency', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 5: TTFT vs Concurrency - Qwen3-8B
ax = axes[1, 1]
for config in config_order:
    data = df_plot[(df_plot['model'] == 'Qwen3-8B') & (df_plot['config_label'] == config)]
    ax.plot(data['concurrency'], data['ttft_ms'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax.set_xlabel('Concurrency', fontsize=12)
ax.set_ylabel('TTFT (ms)', fontsize=12)
ax.set_title('Qwen3-8B: Time to First Token vs Concurrency', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 6: TPOT vs Concurrency - Both Models
ax = axes[1, 2]
for model in models:
    for config in config_order:
        data = df_plot[(df_plot['model'] == model) & (df_plot['config_label'] == config)]
        label = f"{model.split('-')[1]} - {config}"
        linestyle = '-' if model == 'Qwen3-0.6B' else '--'
        ax.plot(data['concurrency'], data['tpot_ms'], marker='o', label=label,
                color=colors[config], linewidth=2, markersize=6, linestyle=linestyle, alpha=0.7)
ax.set_xlabel('Concurrency', fontsize=12)
ax.set_ylabel('TPOT (ms)', fontsize=12)
ax.set_title('Time Per Output Token vs Concurrency', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
print("Saved visualization to benchmark_results.png")

# Create a second figure for performance improvements
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('CPU Offload Performance Improvements vs Baseline',
              fontsize=14, fontweight='bold')

# Calculate percentage improvements
for idx, model in enumerate(models):
    ax = axes2[idx]

    baseline = df_plot[(df_plot['model'] == model) & (df_plot['config_label'] == 'default')]
    offload = df_plot[(df_plot['model'] == model) & (df_plot['config_label'] == 'offload')]

    # Merge on concurrency
    merged = baseline.merge(offload, on='concurrency', suffixes=('_baseline', '_offload'))

    # Calculate improvements (positive = better)
    merged['throughput_improvement'] = ((merged['tokens_per_second_offload'] - merged['tokens_per_second_baseline']) /
                                        merged['tokens_per_second_baseline'] * 100)
    merged['ttft_improvement'] = -((merged['ttft_ms_offload'] - merged['ttft_ms_baseline']) /
                                   merged['ttft_ms_baseline'] * 100)  # negative = improvement
    merged['tpot_improvement'] = -((merged['tpot_ms_offload'] - merged['tpot_ms_baseline']) /
                                   merged['tpot_ms_baseline'] * 100)  # negative = improvement

    x = np.arange(len(merged))
    width = 0.25

    ax.bar(x - width, merged['throughput_improvement'], width, label='Throughput', color='#2E86AB', alpha=0.8)
    ax.bar(x, merged['ttft_improvement'], width, label='TTFT (lower is better)', color='#A23B72', alpha=0.8)
    ax.bar(x + width, merged['tpot_improvement'], width, label='TPOT (lower is better)', color='#F18F01', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Concurrency', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title(f'{model}: Offload vs Default', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) if pd.notna(c) else '' for c in merged['concurrency']])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('performance_improvements.png', dpi=300, bbox_inches='tight')
print("Saved improvements visualization to performance_improvements.png")

plt.show()
