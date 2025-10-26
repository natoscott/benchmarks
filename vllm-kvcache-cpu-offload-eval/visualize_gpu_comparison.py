#!/usr/bin/env python3
"""
Create GPU utilization comparison visualizations.
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
df = pd.read_csv('gpu_utilization_detailed_with_14b.csv')
df_comparison = pd.read_csv('gpu_comparison_summary_with_14b.csv')

# Filter valid data and exclude lmcache
df = df[(df['gpu_active_mean'].notna()) & (df['config_label'] != 'lmcache')]
df_comparison = df_comparison[df_comparison['config'] != 'lmcache']

print("Creating GPU utilization visualizations...")

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Get colors from muted palette
palette_colors = sns.color_palette("muted")
colors = {'default': palette_colors[0], 'offload': palette_colors[1]}
models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B']

# ============================================================================
# Row 1: GPU Compute Utilization
# ============================================================================

# Plot 1: GPU Compute Active vs Concurrency - Qwen3-0.6B
ax1 = fig.add_subplot(gs[0, 0])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-0.6B') & (df['config_label'] == config)]
    ax1.plot(data['concurrency'], data['gpu_active_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax1.set_xlabel('Concurrency', fontsize=11)
ax1.set_ylabel('GPU Compute Active (%)', fontsize=11)
ax1.set_title('Qwen3-0.6B: GPU Compute Utilization', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_ylim(0, 100)

# Plot 2: GPU Compute Active vs Concurrency - Qwen3-8B
ax2 = fig.add_subplot(gs[0, 1])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-8B') & (df['config_label'] == config)]
    ax2.plot(data['concurrency'], data['gpu_active_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax2.set_xlabel('Concurrency', fontsize=11)
ax2.set_ylabel('GPU Compute Active (%)', fontsize=11)
ax2.set_title('Qwen3-8B: GPU Compute Utilization', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_ylim(0, 100)

# Plot 3: Average GPU Compute Comparison
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(len(models))
width = 0.35
for i, config in enumerate(['default', 'offload']):
    values = [df_comparison[(df_comparison['model'] == m) &
                           (df_comparison['config'] == config)]['gpu_compute_avg'].values[0]
             for m in models]
    ax3.bar(x + i*width, values, width, label=config, color=colors[config], alpha=0.8)
ax3.set_xlabel('Model', fontsize=11)
ax3.set_ylabel('Avg GPU Compute Active (%)', fontsize=11)
ax3.set_title('Average GPU Compute Utilization', fontsize=12, fontweight='bold')
ax3.set_xticks(x + width)
ax3.set_xticklabels([m.split('-')[1] for m in models])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 100)

# ============================================================================
# Row 2: GPU Memory Utilization
# ============================================================================

# Plot 4: GPU Memory Active vs Concurrency - Qwen3-0.6B
ax4 = fig.add_subplot(gs[1, 0])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-0.6B') & (df['config_label'] == config)]
    ax4.plot(data['concurrency'], data['mem_active_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax4.set_xlabel('Concurrency', fontsize=11)
ax4.set_ylabel('GPU Memory Active (%)', fontsize=11)
ax4.set_title('Qwen3-0.6B: GPU Memory Bandwidth', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log')
ax4.set_ylim(0, 100)

# Plot 5: GPU Memory Active vs Concurrency - Qwen3-8B
ax5 = fig.add_subplot(gs[1, 1])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-8B') & (df['config_label'] == config)]
    ax5.plot(data['concurrency'], data['mem_active_mean'], marker='o', label=config,
            color=colors[config], linewidth=2, markersize=8)
ax5.set_xlabel('Concurrency', fontsize=11)
ax5.set_ylabel('GPU Memory Active (%)', fontsize=11)
ax5.set_title('Qwen3-8B: GPU Memory Bandwidth', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xscale('log')
ax5.set_ylim(0, 100)

# Plot 6: Memory Used Comparison
ax6 = fig.add_subplot(gs[1, 2])
width = 0.35
for i, config in enumerate(['default', 'offload']):
    values = [df_comparison[(df_comparison['model'] == m) &
                           (df_comparison['config'] == config)]['memory_used_gb'].values[0]
             for m in models]
    ax6.bar(x + i*width, values, width, label=config, color=colors[config], alpha=0.8)
ax6.set_xlabel('Model', fontsize=11)
ax6.set_ylabel('Avg GPU Memory Used (GB)', fontsize=11)
ax6.set_title('Average GPU Memory Usage', fontsize=12, fontweight='bold')
ax6.set_xticks(x + width)
ax6.set_xticklabels([m.split('-')[1] for m in models])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Row 3: Performance vs GPU Utilization Trade-off
# ============================================================================

# Plot 7: Throughput vs GPU Compute - Qwen3-0.6B
ax7 = fig.add_subplot(gs[2, 0])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-0.6B') & (df['config_label'] == config)]
    ax7.scatter(data['gpu_active_mean'], data['throughput'],
               label=config, color=colors[config], s=100, alpha=0.7)
ax7.set_xlabel('GPU Compute Active (%)', fontsize=11)
ax7.set_ylabel('Throughput (tok/s)', fontsize=11)
ax7.set_title('Qwen3-0.6B: Throughput vs GPU Utilization', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: Throughput vs GPU Compute - Qwen3-8B
ax8 = fig.add_subplot(gs[2, 1])
for config in ['default', 'offload']:
    data = df[(df['model'] == 'Qwen3-8B') & (df['config_label'] == config)]
    ax8.scatter(data['gpu_active_mean'], data['throughput'],
               label=config, color=colors[config], s=100, alpha=0.7)
ax8.set_xlabel('GPU Compute Active (%)', fontsize=11)
ax8.set_ylabel('Throughput (tok/s)', fontsize=11)
ax8.set_title('Qwen3-8B: Throughput vs GPU Utilization ⭐', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)


# Plot 9: Efficiency Comparison (Throughput per GPU%)
ax9 = fig.add_subplot(gs[2, 2])
efficiency_data = []
for model in models:
    for config in ['default', 'offload']:
        row = df_comparison[(df_comparison['model'] == model) & (df_comparison['config'] == config)]
        if len(row) > 0:
            efficiency = row['throughput'].values[0] / row['gpu_compute_avg'].values[0]
            efficiency_data.append({
                'model': model.split('-')[1],
                'config': config,
                'efficiency': efficiency
            })

df_efficiency = pd.DataFrame(efficiency_data)
pivot_efficiency = df_efficiency.pivot(index='model', columns='config', values='efficiency')
pivot_efficiency[['default', 'offload']].plot(kind='bar', ax=ax9,
                                                color=[colors['default'], colors['offload']],
                                                alpha=0.8, width=0.7)
ax9.set_xlabel('Model', fontsize=11)
ax9.set_ylabel('Throughput per GPU% (tok/s per %)', fontsize=11)
ax9.set_title('GPU Efficiency (Higher is Better)', fontsize=12, fontweight='bold')
ax9.legend(title='Config')
ax9.grid(True, alpha=0.3, axis='y')
ax9.set_xticklabels(ax9.get_xticklabels(), rotation=0)

fig.suptitle('GPU Utilization Analysis: CPU Offload vs Baseline',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('gpu_utilization_comparison.png', dpi=300, bbox_inches='tight')
print("Saved GPU utilization comparison to gpu_utilization_comparison.png")

# ============================================================================
# Create a focused comparison figure for the report
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Qwen3-8B Offload Performance',
              fontsize=14, fontweight='bold')

# Chart 1: GPU Compute reduction
ax = axes[0, 0]
configs = ['Baseline', 'Offload']
gpu_vals = [88.3, 80.3]
bars = ax.bar(configs, gpu_vals, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.5)
ax.set_ylabel('GPU Compute Active (%)', fontsize=12)
ax.set_title('GPU Compute Utilization\n(Lower is Better for Efficiency)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')
# Add value labels
for bar, val in zip(bars, gpu_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
# Add improvement label
ax.text(0.5, 85, '-9.1%', ha='center', fontsize=14, fontweight='bold',
        color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Chart 2: Throughput improvement
ax = axes[0, 1]
tput_vals = [52.7, 69.9]
bars = ax.bar(configs, tput_vals, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.5)
ax.set_ylabel('Throughput (tok/s)', fontsize=12)
ax.set_title('Throughput Performance\n(Higher is Better)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, tput_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
# Add improvement label
ax.text(0.5, 62, '+32.7%', ha='center', fontsize=14, fontweight='bold',
        color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Chart 3: Efficiency comparison
ax = axes[1, 0]
efficiency_vals = [52.7/88.3, 69.9/80.3]
bars = ax.bar(configs, efficiency_vals, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.5)
ax.set_ylabel('Throughput per GPU% Usage', fontsize=12)
ax.set_title('GPU Efficiency\n(tok/s per % GPU utilization)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, efficiency_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
# Add improvement label
improvement = ((efficiency_vals[1] - efficiency_vals[0]) / efficiency_vals[0] * 100)
ax.text(0.5, 0.75, f'+{improvement:.1f}%', ha='center', fontsize=14, fontweight='bold',
        color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Chart 4: Cache hit ratio comparison
ax = axes[1, 1]
# Add cache hit ratio data from cache analysis
cache_hit_06b = [0.0, 24.8]
cache_hit_8b = [3.6, 46.2]
cache_hit_14b = [6.0, 6.0]  # 14B showed 6% cache hit ratio for both configs
x_pos = np.arange(2)
bar_width = 0.25
bars1 = ax.bar(x_pos - bar_width, cache_hit_06b, bar_width, label='0.6B', alpha=0.8)
bars2 = ax.bar(x_pos, cache_hit_8b, bar_width, label='8B', alpha=0.8)
bars3 = ax.bar(x_pos + bar_width, cache_hit_14b, bar_width, label='14B', alpha=0.8)
ax.set_ylabel('Cache Hit Ratio (%)', fontsize=12)
ax.set_title('Cache Hit Ratio by Configuration', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Baseline', 'Offload'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 60)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('gpu_key_findings.png', dpi=300, bbox_inches='tight')
print("Saved key findings to gpu_key_findings.png")

print("\nVisualization complete!")
