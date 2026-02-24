#!/usr/bin/env python3
"""
Create visualizations for per-CPU saturation analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style per visualization-palette skill
sns.set_style("whitegrid")
sns.set_palette("muted")  # Qualitative palette for categorical data
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

ANALYSIS_DIR = Path('analysis')

# Scenario ordering (consistent with comprehensive-analysis.py)
SCENARIO_ORDER = [
    'no-offload',
    'native-offload',
    'lmcache-local',
    'lmcache-redis',
    'lmcache-valkey',
    'llm-d-redis',
    'llm-d-valkey'
]

def load_data():
    """Load per-CPU analysis data."""
    return pd.read_csv(ANALYSIS_DIR / 'percpu_analysis.csv')

def plot_saturated_cpus_by_scenario(df):
    """Bar chart showing number of saturated CPUs by scenario."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by scenario
    scenario_summary = df.groupby('scenario').agg({
        'saturated_cpus_user': 'mean',
        'global_mean_user': 'mean'
    })

    # Reorder by consistent scenario ordering
    scenario_summary = scenario_summary.reindex([s for s in SCENARIO_ORDER if s in scenario_summary.index])

    x = np.arange(len(scenario_summary))
    width = 0.35

    ax.bar(x - width/2, scenario_summary['saturated_cpus_user'], width,
           label='Saturated CPUs (>80%)', color='#d62728', alpha=0.8)
    ax.bar(x + width/2, scenario_summary['global_mean_user'] / 100 * 48, width,
           label='Expected from avg CPU%', color='#1f77b4', alpha=0.8)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Number of CPUs', fontsize=12)
    ax.set_title('CPU Saturation: Saturated CPUs vs Expected from Average',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_summary.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'percpu_saturation_by_scenario.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'percpu_saturation_by_scenario.png'}")
    plt.close()

def plot_cpu_load_distribution(df):
    """Box plot showing CPU load distribution (std dev and range)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Use consistent scenario ordering
    scenario_order = [s for s in SCENARIO_ORDER if s in df['scenario'].values]

    # Standard deviation
    data_stdev = [df[df['scenario'] == s]['cpu_stdev'].values for s in scenario_order]
    bp1 = ax1.boxplot(data_stdev, labels=scenario_order, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('#ff7f0e')
        patch.set_alpha(0.7)

    ax1.set_xlabel('Scenario', fontsize=12)
    ax1.set_ylabel('CPU Load Standard Deviation (%)', fontsize=12)
    ax1.set_title('CPU Load Variance Across CPUs', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Range (max - min)
    data_range = [df[df['scenario'] == s]['cpu_range'].values for s in scenario_order]
    bp2 = ax2.boxplot(data_range, labels=scenario_order, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('#2ca02c')
        patch.set_alpha(0.7)

    ax2.set_xlabel('Scenario', fontsize=12)
    ax2.set_ylabel('CPU Load Range (Max - Min) (%)', fontsize=12)
    ax2.set_title('CPU Load Imbalance (Hotspot Severity)', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'percpu_load_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'percpu_load_distribution.png'}")
    plt.close()

def plot_avg_vs_max_cpu(df):
    """Scatter plot showing average CPU vs max CPU utilization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use consistent scenario ordering
    scenarios = [s for s in SCENARIO_ORDER if s in df['scenario'].values]
    colors = sns.color_palette("muted", len(scenarios))

    for scenario, color in zip(scenarios, colors):
        scenario_df = df[df['scenario'] == scenario]
        ax.scatter(
            scenario_df['global_mean_user'],
            scenario_df['max_cpu_mean_user'],
            label=scenario,
            alpha=0.7,
            s=100,
            color=color
        )

    # Add diagonal reference line (where avg = max)
    max_val = max(df['global_mean_user'].max(), df['max_cpu_mean_user'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.3, linewidth=2,
            label='Equal distribution')

    ax.set_xlabel('Global Average CPU Utilization (%)', fontsize=12)
    ax.set_ylabel('Maximum Single CPU Utilization (%)', fontsize=12)
    ax.set_title('CPU Load Concentration: Average vs Peak Single CPU',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'percpu_avg_vs_max.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'percpu_avg_vs_max.png'}")
    plt.close()

def plot_saturation_heatmap(df):
    """Heatmap showing CPU saturation metrics across scenarios."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create summary matrix
    metrics = ['saturated_cpus_user', 'max_cpu_mean_user', 'cpu_stdev', 'cpu_range']
    metric_labels = ['Saturated CPUs', 'Max CPU %', 'Load Std Dev', 'Load Range']

    summary = df.groupby('scenario')[metrics].mean()

    # Reorder by consistent scenario ordering
    summary = summary.reindex([s for s in SCENARIO_ORDER if s in summary.index])

    # Normalize for heatmap (scale each metric to 0-1)
    summary_norm = summary.copy()
    for col in summary_norm.columns:
        min_val = summary_norm[col].min()
        max_val = summary_norm[col].max()
        summary_norm[col] = (summary_norm[col] - min_val) / (max_val - min_val)

    summary_norm.columns = metric_labels

    sns.heatmap(
        summary_norm.T,
        annot=summary.T,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Normalized Severity'},
        linewidths=1,
        ax=ax
    )

    ax.set_title('CPU Saturation Patterns by Scenario', fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'percpu_saturation_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'percpu_saturation_heatmap.png'}")
    plt.close()

def plot_offload_impact_on_saturation(df):
    """Compare CPU saturation between offload and non-offload scenarios."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Categorize scenarios
    offload_scenarios = ['lmcache-local', 'lmcache-local-20kcpu', 'lmcache-redis',
                         'lmcache-valkey', 'lmcache-valkey-20kcpu', 'native-offload',
                         'native-offload-20kcpu']

    df['category'] = df['scenario'].apply(
        lambda x: 'CPU Offload' if x in offload_scenarios else
                  ('Distributed Index' if 'llm-d' in x else 'No Offload')
    )

    # Box plot by category
    categories = ['No Offload', 'Distributed Index', 'CPU Offload']
    data = [df[df['category'] == cat]['saturated_cpus_user'].values for cat in categories]
    colors = ['#2ca02c', '#ff7f0e', '#d62728']

    bp = ax.boxplot(data, labels=categories, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Number of Saturated CPUs (>80%)', fontsize=12)
    ax.set_title('CPU Saturation by Configuration Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean values as text
    for i, cat in enumerate(categories):
        mean_val = df[df['category'] == cat]['saturated_cpus_user'].mean()
        ax.text(i+1, mean_val, f'{mean_val:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'percpu_offload_impact.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'percpu_offload_impact.png'}")
    plt.close()

def main():
    print("Creating per-CPU saturation visualizations...")
    print("=" * 80)

    df = load_data()
    print(f"Loaded {len(df)} benchmark runs")

    print("\nGenerating visualizations...")
    plot_saturated_cpus_by_scenario(df)
    plot_cpu_load_distribution(df)
    plot_avg_vs_max_cpu(df)
    plot_saturation_heatmap(df)
    plot_offload_impact_on_saturation(df)

    print("\n" + "=" * 80)
    print("All visualizations created successfully!")
    print(f"Output directory: {ANALYSIS_DIR}/")

if __name__ == '__main__':
    main()
