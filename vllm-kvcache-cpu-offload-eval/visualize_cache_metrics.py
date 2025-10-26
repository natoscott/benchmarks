#!/usr/bin/env python3
"""
Create KV cache metrics visualizations.
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

# Load data
df = pd.read_csv('cache_analysis_results_with_14b.csv')

# Filter out warmup runs and lmcache
df = df[(~df['is_warmup']) & (df['config_label'] != 'lmcache')].copy()

# Get colors from muted palette
palette_colors = sns.color_palette("muted")
colors = {'default': palette_colors[0], 'offload': palette_colors[1]}
models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B']

print("Creating KV cache visualizations...")

# ============================================================================
# Figure 1: Comprehensive cache metrics comparison
# ============================================================================

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Plot 1: Cache query rate vs Concurrency - Qwen3-0.6B
ax1 = fig.add_subplot(gs[0, 0])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-0.6B') & (df['config_label'] == config)]
    ax1.plot(data['concurrency'], data['connector_query_rate_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax1.set_xlabel('Concurrency', fontsize=11)
ax1.set_ylabel('Cache Queries/sec', fontsize=11)
ax1.set_title('Qwen3-0.6B: Cache Query Rate', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Cache query rate vs Concurrency - Qwen3-8B
ax2 = fig.add_subplot(gs[0, 1])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-8B') & (df['config_label'] == config)]
    ax2.plot(data['concurrency'], data['connector_query_rate_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax2.set_xlabel('Concurrency', fontsize=11)
ax2.set_ylabel('Cache Queries/sec', fontsize=11)
ax2.set_title('Qwen3-8B: Cache Query Rate', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# Plot 3: Cache query rate vs Concurrency - Qwen3-14B
ax3 = fig.add_subplot(gs[0, 2])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-14B') & (df['config_label'] == config)]
    ax3.plot(data['concurrency'], data['connector_query_rate_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax3.set_xlabel('Concurrency', fontsize=11)
ax3.set_ylabel('Cache Queries/sec', fontsize=11)
ax3.set_title('Qwen3-14B: Cache Query Rate', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')

# Plot 4: Average cache query rate comparison
ax4 = fig.add_subplot(gs[0, 3])
x = np.arange(len(models))
width = 0.25
for i, config in enumerate(['default', 'offload']):
    values = []
    for model in models:
        avg = df[(df['model'] == model) & (df['config_label'] == config)]['connector_query_rate_mean'].mean()
        values.append(avg)
    ax4.bar(x + i*width, values, width, label=config, color=colors[config], alpha=0.8)
ax4.set_xlabel('Model', fontsize=11)
ax4.set_ylabel('Avg Cache Queries/sec', fontsize=11)
ax4.set_title('Average Cache Query Rate', fontsize=12, fontweight='bold')
ax4.set_xticks(x + width)
ax4.set_xticklabels([m.split('-')[1] for m in models])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Cache hit rate vs Concurrency - Qwen3-0.6B
ax5 = fig.add_subplot(gs[1, 0])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-0.6B') & (df['config_label'] == config)]
    ax5.plot(data['concurrency'], data['connector_hit_rate_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax5.set_xlabel('Concurrency', fontsize=11)
ax5.set_ylabel('Cache Hits/sec', fontsize=11)
ax5.set_title('Qwen3-0.6B: Cache Hit Rate', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xscale('log')

# Plot 6: Cache hit rate vs Concurrency - Qwen3-8B
ax6 = fig.add_subplot(gs[1, 1])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-8B') & (df['config_label'] == config)]
    ax6.plot(data['concurrency'], data['connector_hit_rate_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax6.set_xlabel('Concurrency', fontsize=11)
ax6.set_ylabel('Cache Hits/sec', fontsize=11)
ax6.set_title('Qwen3-8B: Cache Hit Rate', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_xscale('log')

# Plot 7: Cache hit rate vs Concurrency - Qwen3-14B
ax7 = fig.add_subplot(gs[1, 2])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-14B') & (df['config_label'] == config)]
    ax7.plot(data['concurrency'], data['connector_hit_rate_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax7.set_xlabel('Concurrency', fontsize=11)
ax7.set_ylabel('Cache Hits/sec', fontsize=11)
ax7.set_title('Qwen3-14B: Cache Hit Rate', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.set_xscale('log')

# Plot 8: Average cache hit rate comparison
ax8 = fig.add_subplot(gs[1, 3])
x = np.arange(len(models))
width = 0.25
for i, config in enumerate(['default', 'offload']):
    values = []
    for model in models:
        avg = df[(df['model'] == model) & (df['config_label'] == config)]['connector_hit_rate_mean'].mean()
        values.append(avg)
    ax8.bar(x + i*width, values, width, label=config, color=colors[config], alpha=0.8)
ax8.set_xlabel('Model', fontsize=11)
ax8.set_ylabel('Avg Cache Hits/sec', fontsize=11)
ax8.set_title('Average Cache Hit Rate', fontsize=12, fontweight='bold')
ax8.set_xticks(x + width)
ax8.set_xticklabels([m.split('-')[1] for m in models])
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Cache hit ratio vs Concurrency - Qwen3-0.6B
ax9 = fig.add_subplot(gs[2, 0])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-0.6B') & (df['config_label'] == config)]
    ax9.plot(data['concurrency'], data['connector_hit_ratio'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax9.set_xlabel('Concurrency', fontsize=11)
ax9.set_ylabel('Cache Hit Ratio (%)', fontsize=11)
ax9.set_title('Qwen3-0.6B: Cache Hit Ratio', fontsize=12, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.set_xscale('log')
ax9.set_ylim(0, 100)

# Plot 10: Cache hit ratio vs Concurrency - Qwen3-8B
ax10 = fig.add_subplot(gs[2, 1])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-8B') & (df['config_label'] == config)]
    ax10.plot(data['concurrency'], data['connector_hit_ratio'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax10.set_xlabel('Concurrency', fontsize=11)
ax10.set_ylabel('Cache Hit Ratio (%)', fontsize=11)
ax10.set_title('Qwen3-8B: Cache Hit Ratio', fontsize=12, fontweight='bold')
ax10.legend()
ax10.grid(True, alpha=0.3)
ax10.set_xscale('log')
ax10.set_ylim(0, 100)

# Plot 11: Cache hit ratio vs Concurrency - Qwen3-14B
ax11 = fig.add_subplot(gs[2, 2])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-14B') & (df['config_label'] == config)]
    ax11.plot(data['concurrency'], data['connector_hit_ratio'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax11.set_xlabel('Concurrency', fontsize=11)
ax11.set_ylabel('Cache Hit Ratio (%)', fontsize=11)
ax11.set_title('Qwen3-14B: Cache Hit Ratio', fontsize=12, fontweight='bold')
ax11.legend()
ax11.grid(True, alpha=0.3)
ax11.set_xscale('log')
ax11.set_ylim(0, 100)

# Plot 12: Average cache hit ratio comparison
ax12 = fig.add_subplot(gs[2, 3])
x = np.arange(len(models))
width = 0.25
for i, config in enumerate(['default', 'offload']):
    values = []
    for model in models:
        avg = df[(df['model'] == model) & (df['config_label'] == config)]['connector_hit_ratio'].mean()
        values.append(avg)
    bars = ax12.bar(x + i*width, values, width, label=config, color=colors[config], alpha=0.8)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax12.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
ax12.set_xlabel('Model', fontsize=11)
ax12.set_ylabel('Avg Cache Hit Ratio (%)', fontsize=11)
ax12.set_title('Average Cache Hit Ratio', fontsize=12, fontweight='bold')
ax12.set_xticks(x + width)
ax12.set_xticklabels([m.split('-')[1] for m in models])
ax12.legend()
ax12.grid(True, alpha=0.3, axis='y')
ax12.set_ylim(0, 100)

fig.suptitle('KV Cache Metrics Analysis: CPU Offload vs Baseline',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('cache_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("Saved cache metrics comparison to cache_metrics_comparison.png")

# ============================================================================
# Figure 2: Focused cache performance summary
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('KV Cache Performance: Offload vs Baseline',
              fontsize=14, fontweight='bold')

# Chart 1: Qwen3-8B Cache Query Rate
ax = axes[0, 0]
configs = ['Baseline', 'Offload']
query_vals = [758.1, 11256.4]
bars = ax.bar(configs, query_vals, color=[colors['default'], colors['offload']], alpha=0.8, width=0.5)
ax.set_ylabel('Cache Queries/sec', fontsize=12)
ax.set_title('Qwen3-8B: Cache Query Rate', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, query_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Chart 2: Qwen3-8B Cache Hit Rate
ax = axes[0, 1]
hit_vals = [244.3, 5135.0]
bars = ax.bar(configs, hit_vals, color=[colors['default'], colors['offload']], alpha=0.8, width=0.5)
ax.set_ylabel('Cache Hits/sec', fontsize=12)
ax.set_title('Qwen3-8B: Cache Hit Rate', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, hit_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Chart 3: Cache Hit Ratio Comparison
ax = axes[1, 0]
hit_ratio_06b = [0.0, 24.8]
hit_ratio_8b = [3.6, 46.2]
hit_ratio_14b = [6.0, 6.0]
x = np.arange(2)
width = 0.25
bars1 = ax.bar(x - width, hit_ratio_06b, width, label='0.6B', color=palette_colors[2], alpha=0.8)
bars2 = ax.bar(x, hit_ratio_8b, width, label='8B', color=palette_colors[3], alpha=0.8)
bars3 = ax.bar(x + width, hit_ratio_14b, width, label='14B', color=palette_colors[4], alpha=0.8)
ax.set_ylabel('Cache Hit Ratio (%)', fontsize=12)
ax.set_title('Cache Hit Ratio by Model', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Baseline', 'Offload'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 60)
# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Chart 4: Cache usage percentage
ax = axes[1, 1]
cache_usage_data = df.groupby(['model', 'config_label'])['cache_usage_mean'].mean().reset_index()
models_short = ['0.6B', '8B', '14B']
x = np.arange(len(models_short))
width = 0.35
for i, config in enumerate(['default', 'offload']):
    values = []
    for model in models:
        val = cache_usage_data[(cache_usage_data['model'] == model) &
                               (cache_usage_data['config_label'] == config)]['cache_usage_mean'].values
        values.append(val[0] if len(val) > 0 else 0)
    ax.bar(x + i*width, values, width, label=config, color=colors[config], alpha=0.8)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Avg Cache Usage (%)', fontsize=12)
ax.set_title('KV Cache Memory Usage', fontsize=12, fontweight='bold')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(models_short)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cache_summary.png', dpi=300, bbox_inches='tight')
print("Saved cache summary to cache_summary.png")

print("\nVisualization complete!")
