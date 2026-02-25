#!/usr/bin/env python3
"""Create visualizations showing KV-cache memory allocation and its impact on performance."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style according to visualization-palette skill
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Load KV-cache allocation data
kvcache_df = pd.read_csv('analysis/kvcache_allocations_actual.csv')

# Load performance data
perf_df = pd.read_csv('analysis/peak_throughput_all.csv')

# Create model short names for plotting
kvcache_df['model_short'] = kvcache_df['model'].str.replace('Qwen/Qwen3-', '').str.replace('-AWQ', '-AWQ')

# Figure 1: GPU Memory and Token Capacity by Model
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Prepare data - get one row per model (no-offload config)
gpu_only = kvcache_df[kvcache_df['is_offload'] == 'no'].copy()
gpu_only = gpu_only.sort_values('gpu_memory_gib')

# Top left: GPU KV-Cache Memory Available
ax = axes[0, 0]
bars = ax.barh(gpu_only['model_short'], gpu_only['gpu_memory_gib'].astype(float), color=sns.color_palette("muted")[0])
ax.set_xlabel('Available GPU KV-Cache Memory (GiB)')
ax.set_title('GPU Memory Available for KV-Cache by Model')
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, (idx, row) in enumerate(gpu_only.iterrows()):
    ax.text(row['gpu_memory_gib'] + 0.5, i, f"{row['gpu_memory_gib']:.1f} GiB", va='center')

# Top right: GPU KV-Cache Token Capacity
ax = axes[0, 1]
token_counts = gpu_only['gpu_kv_tokens'].astype(float) / 1000  # Convert to thousands
bars = ax.barh(gpu_only['model_short'], token_counts, color=sns.color_palette("muted")[1])
ax.set_xlabel('GPU KV-Cache Token Capacity (thousands)')
ax.set_title('GPU KV-Cache Token Capacity by Model')
ax.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(gpu_only.iterrows()):
    tokens_k = int(row['gpu_kv_tokens']) // 1000
    ax.text(tokens_k + 10, i, f"{tokens_k}K", va='center')

# Bottom left: Max Concurrency
ax = axes[1, 0]
bars = ax.barh(gpu_only['model_short'], gpu_only['max_concurrency'].astype(float), color=sns.color_palette("muted")[2])
ax.set_xlabel('Maximum Concurrency')
ax.set_title('vLLM Maximum Concurrency by Model')
ax.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(gpu_only.iterrows()):
    ax.text(row['max_concurrency'] + 0.3, i, f"{row['max_concurrency']:.1f}x", va='center')

# Bottom right: CPU Offload Memory (for configs that use it)
ax = axes[1, 1]
offload_configs = kvcache_df[(kvcache_df['is_offload'] == 'yes') & (kvcache_df['config_name'].str.contains('10k'))].copy()
offload_configs = offload_configs.sort_values('cpu_memory_gib')
if not offload_configs.empty:
    width = 0.35
    x = np.arange(len(offload_configs))

    ax.barh(x - width/2, offload_configs['gpu_memory_gib'].astype(float), width,
            label='GPU Memory', color=sns.color_palette("muted")[0])
    ax.barh(x + width/2, offload_configs['cpu_memory_gib'].astype(float), width,
            label='CPU Offload Memory', color=sns.color_palette("muted")[3])

    ax.set_yticks(x)
    ax.set_yticklabels(offload_configs['model_short'])
    ax.set_xlabel('Memory (GiB)')
    ax.set_title('GPU + CPU Offload Memory (10K blocks config)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/kvcache_memory_capacity.png', dpi=150, bbox_inches='tight')
print("Created: analysis/kvcache_memory_capacity.png")
plt.close()

# Figure 2: Memory Allocation vs Performance Impact
# Merge with performance data
# Get peak throughput for baseline and offload configs
baseline_perf = perf_df[perf_df['scenario'].str.contains('no-offload')].copy()
baseline_perf['model_short'] = baseline_perf['model'].str.replace('Qwen/Qwen3-', '').str.replace('-AWQ', '-AWQ')

# Merge GPU memory data
gpu_only_dict = gpu_only.set_index('model_short')['gpu_memory_gib'].to_dict()

# Create scatter plot showing relationship
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Get offload performance deltas from the report data
model_data = {
    '0.6B': {'gpu_mem': 33.92, 'native_delta': -29.1, 'lmcache_delta': -13.6},
    '8B': {'gpu_mem': 26.83, 'native_delta': -36.5, 'lmcache_delta': -5.6},
    '14B': {'gpu_mem': 20.58, 'native_delta': 0.6, 'lmcache_delta': 11.8},
    '32B-AWQ': {'gpu_mem': 25.40, 'native_delta': -1.0, 'lmcache_delta': -12.7},
}

models = list(model_data.keys())
gpu_mems = [model_data[m]['gpu_mem'] for m in models]
native_deltas = [model_data[m]['native_delta'] for m in models]
lmcache_deltas = [model_data[m]['lmcache_delta'] for m in models]

# Create scatter plot
ax.scatter(gpu_mems, native_deltas, s=200, alpha=0.7,
          label='Native Offload (10K blocks)', color=sns.color_palette("muted")[0], marker='o')
ax.scatter(gpu_mems, lmcache_deltas, s=200, alpha=0.7,
          label='LMCache Local (10K blocks)', color=sns.color_palette("muted")[1], marker='s')

# Add zero line
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Add labels for each point
for i, model in enumerate(models):
    ax.annotate(model, (gpu_mems[i], native_deltas[i]),
               xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.annotate(model, (gpu_mems[i], lmcache_deltas[i]),
               xytext=(5, -15), textcoords='offset points', fontsize=9)

ax.set_xlabel('Available GPU KV-Cache Memory (GiB)')
ax.set_ylabel('Performance Delta vs Baseline (%)')
ax.set_title('CPU Offload Performance Impact vs GPU Memory Availability\n(Lower GPU memory → Higher benefit from CPU offload)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/kvcache_memory_vs_performance.png', dpi=150, bbox_inches='tight')
print("Created: analysis/kvcache_memory_vs_performance.png")
plt.close()

# Figure 3: Actual vs Configured CPU Blocks
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

offload_all = kvcache_df[kvcache_df['is_offload'] == 'yes'].copy()
offload_all = offload_all.sort_values(['model_short', 'config_name'])

# Group by model and config
grouped = offload_all.groupby('model_short')
x_pos = 0
x_ticks = []
x_labels = []

colors = sns.color_palette("muted")
for i, (model, group) in enumerate(grouped):
    for j, (idx, row) in enumerate(group.iterrows()):
        configured = int(row['cpu_blocks_configured'])
        actual = int(row['cpu_blocks_actual'])

        # Plot configured and actual
        ax.barh(x_pos, configured, height=0.4, alpha=0.4,
               color=colors[i % len(colors)], label=f'{model} (configured)' if j == 0 else '')
        ax.barh(x_pos, actual, height=0.4, alpha=0.8,
               color=colors[i % len(colors)], label=f'{model} (actual)' if j == 0 else '')

        # Add labels
        config_label = row['config_name'].split('-')[-1]
        x_ticks.append(x_pos)
        x_labels.append(f"{model}\n{config_label}")

        # Add value text
        ratio = actual / configured
        ax.text(actual + 500, x_pos, f"{actual:,} ({ratio:.2f}x)", va='center', fontsize=9)

        x_pos += 1
    x_pos += 0.5  # Add spacing between models

ax.set_yticks(x_ticks)
ax.set_yticklabels(x_labels, fontsize=10)
ax.set_xlabel('CPU Blocks')
ax.set_title('CPU KV-Cache Blocks: Configured vs Actual Allocated\n(vLLM allocates based on available memory, not just configuration)')
ax.grid(axis='x', alpha=0.3)

# Create custom legend
from matplotlib.patches import Rectangle
legend_elements = [
    Rectangle((0, 0), 1, 1, fc='gray', alpha=0.4, label='Configured'),
    Rectangle((0, 0), 1, 1, fc='gray', alpha=0.8, label='Actual Allocated')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('analysis/kvcache_configured_vs_actual.png', dpi=150, bbox_inches='tight')
print("Created: analysis/kvcache_configured_vs_actual.png")
plt.close()

# Figure 4: Comprehensive Memory Summary
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Create a comprehensive table view
models_sorted = ['14B', '32B-AWQ', '8B', '0.6B']  # Order by GPU memory (lowest to highest)
offload_benefit = [11.8, -12.7, -5.6, -13.6]  # LMCache local 10K performance delta
gpu_memory = [20.58, 25.40, 26.83, 33.92]
gpu_tokens = [269.7, 208.1, 390.7, 635.2]  # in thousands
max_concurrency = [6.58, 5.08, 9.54, 15.51]

# Create stacked data view
y_pos = np.arange(len(models_sorted))

# Setup subplot grid
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Panel 1: GPU Memory
ax = axes[0]
colors_benefit = ['green' if x > 0 else 'red' for x in offload_benefit]
bars = ax.barh(y_pos, gpu_memory, color=sns.color_palette("muted")[0])
ax.set_yticks(y_pos)
ax.set_yticklabels(models_sorted)
ax.set_xlabel('GPU KV-Cache Memory (GiB)')
ax.set_title('GPU Memory Available\n(Lower = Higher pressure)')
ax.invert_yaxis()
for i, v in enumerate(gpu_memory):
    ax.text(v + 0.5, i, f'{v:.1f} GiB', va='center')
ax.grid(axis='x', alpha=0.3)

# Panel 2: Token Capacity
ax = axes[1]
bars = ax.barh(y_pos, gpu_tokens, color=sns.color_palette("muted")[1])
ax.set_yticks(y_pos)
ax.set_yticklabels([])
ax.set_xlabel('Token Capacity (thousands)')
ax.set_title('GPU KV-Cache Token Capacity\n(Lower = More constrained)')
ax.invert_yaxis()
for i, v in enumerate(gpu_tokens):
    ax.text(v + 10, i, f'{v:.0f}K', va='center')
ax.grid(axis='x', alpha=0.3)

# Panel 3: Offload Benefit
ax = axes[2]
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in offload_benefit]
bars = ax.barh(y_pos, offload_benefit, color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([])
ax.set_xlabel('Performance Delta (%)')
ax.set_title('CPU Offload Benefit\n(LMCache 10K blocks)')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.invert_yaxis()
for i, v in enumerate(offload_benefit):
    sign = '+' if v > 0 else ''
    ax.text(v + (1 if v > 0 else -1), i, f'{sign}{v:.1f}%', va='center', ha='left' if v > 0 else 'right')
ax.grid(axis='x', alpha=0.3)

plt.suptitle('KV-Cache Memory Pressure Explains CPU Offload Performance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('analysis/kvcache_memory_pressure_summary.png', dpi=150, bbox_inches='tight')
print("Created: analysis/kvcache_memory_pressure_summary.png")
plt.close()

print("\nAll visualizations created successfully!")
