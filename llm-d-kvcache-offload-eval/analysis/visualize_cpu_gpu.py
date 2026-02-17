#!/usr/bin/env python3
"""
Create visualization of CPU and GPU utilization metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Read the data
df = pd.read_csv('analysis/cpu_gpu_utilization.csv')

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Prepare data for plotting
models = df['model'].unique()
configs = df['config'].unique()

x = np.arange(len(models))
width = 0.2

# Plot CPU Utilization
for i, config in enumerate(configs):
    config_data = df[df['config'] == config]
    cpu_values = [config_data[config_data['model'] == m]['cpu_util_pct'].values[0]
                  for m in models]
    ax1.bar(x + i * width, cpu_values, width, label=config)

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('CPU Utilization (%)', fontsize=12, fontweight='bold')
ax1.set_title('CPU Utilization Across Configurations', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot GPU Utilization (cap at 100% for visualization)
for i, config in enumerate(configs):
    config_data = df[df['config'] == config]
    gpu_values = [min(config_data[config_data['model'] == m]['gpu_util_pct'].values[0], 100)
                  for m in models]
    ax2.bar(x + i * width, gpu_values, width, label=config)

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('GPU Utilization (% per GPU)', fontsize=12, fontweight='bold')
ax2.set_title('GPU Utilization Across Configurations', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(models, rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 100)

# Add note about 0.6B native-offload
ax2.text(0.02, 0.98, 'Note: 0.6B native-offload GPU util exceeds 100% (112%)\ndue to multi-GPU aggregation',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('analysis/cpu_gpu_utilization.png', dpi=150, bbox_inches='tight')
print("Saved CPU and GPU utilization graph to analysis/cpu_gpu_utilization.png")

# Create a second figure for GPU Memory Copy Utilization
fig2, ax = plt.subplots(figsize=(14, 6))

for i, config in enumerate(configs):
    config_data = df[df['config'] == config]
    mem_values = [config_data[config_data['model'] == m]['gpu_mem_copy_util_pct'].values[0]
                  for m in models]
    ax.bar(x + i * width, mem_values, width, label=config)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('GPU Memory Copy Utilization (%)', fontsize=12, fontweight='bold')
ax.set_title('GPU Memory Copy Utilization Across Configurations', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/gpu_memory_copy_utilization.png', dpi=150, bbox_inches='tight')
print("Saved GPU memory copy utilization graph to analysis/gpu_memory_copy_utilization.png")

# Create summary table with deltas vs baseline
print("\nGenerating comparison table...")

summary_rows = []
for model in models:
    baseline_row = df[(df['model'] == model) & (df['config'] == 'no-offload')].iloc[0]

    for config in configs:
        if config == 'no-offload':
            continue
        config_row = df[(df['model'] == model) & (df['config'] == config)].iloc[0]

        cpu_delta = config_row['cpu_util_pct'] - baseline_row['cpu_util_pct']
        gpu_delta = config_row['gpu_util_pct'] - baseline_row['gpu_util_pct']
        mem_delta = config_row['gpu_mem_copy_util_pct'] - baseline_row['gpu_mem_copy_util_pct']

        summary_rows.append({
            'Model': model,
            'Configuration': config,
            'CPU Util': f"{config_row['cpu_util_pct']:.1f}%",
            'CPU Δ': f"{cpu_delta:+.1f}%",
            'GPU Util': f"{config_row['gpu_util_pct']:.1f}%",
            'GPU Δ': f"{gpu_delta:+.1f}%",
            'Mem Copy Util': f"{config_row['gpu_mem_copy_util_pct']:.1f}%",
            'Mem Copy Δ': f"{mem_delta:+.1f}%",
        })

summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 120)
print("CPU and GPU Utilization Comparison (vs no-offload baseline)")
print("=" * 120)
print(summary_df.to_string(index=False))
print("=" * 120)
