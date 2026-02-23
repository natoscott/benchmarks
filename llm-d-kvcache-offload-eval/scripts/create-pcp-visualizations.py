#!/usr/bin/env python3
"""
Create visualizations correlating PCP system metrics with GuideLLM benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

ANALYSIS_DIR = Path('analysis')

def load_data():
    """Load PCP metrics and GuideLLM complete metrics."""
    pcp = pd.read_csv(ANALYSIS_DIR / 'pcp_metrics_peak.csv')
    guidellm = pd.read_csv(ANALYSIS_DIR / 'complete_metrics.csv')

    # Filter guidellm to rate=50 to match PCP data
    guidellm = guidellm[guidellm['rate'] == 50].copy()

    # Merge on model and scenario
    merged = pd.merge(
        guidellm,
        pcp,
        on=['model', 'scenario'],
        how='inner',
        suffixes=('_guidellm', '_pcp')
    )

    return merged

def plot_gpu_util_vs_throughput(df):
    """Plot GPU utilization vs output token throughput."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by scenario for coloring
    scenarios = df['scenario'].unique()
    colors = sns.color_palette("husl", len(scenarios))

    for scenario, color in zip(scenarios, colors):
        scenario_df = df[df['scenario'] == scenario]
        ax.scatter(
            scenario_df['gpu_util_mean'],
            scenario_df['output_tps_mean'],
            label=scenario,
            alpha=0.7,
            s=100,
            color=color
        )

    ax.set_xlabel('GPU Utilization (%)', fontsize=12)
    ax.set_ylabel('Output Tokens/Second', fontsize=12)
    ax.set_title('GPU Utilization vs Throughput (Peak Load, Rate=50)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'pcp_gpu_vs_throughput.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'pcp_gpu_vs_throughput.png'}")
    plt.close()

def plot_kv_cache_usage_by_scenario(df):
    """Plot KV-cache usage by scenario and model."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Pivot for grouped bar chart
    pivot = df.pivot_table(
        values='kv_cache_pct_mean',
        index='scenario',
        columns='model',
        aggfunc='mean'
    )

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('KV-Cache Usage (%)', fontsize=12)
    ax.set_title('KV-Cache Utilization by Scenario and Model (Rate=50)', fontsize=14, fontweight='bold')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'pcp_kv_cache_usage.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'pcp_kv_cache_usage.png'}")
    plt.close()

def plot_memory_usage_comparison(df):
    """Plot process memory usage across scenarios."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Pivot for grouped bar chart
    pivot = df.pivot_table(
        values='process_rss_gb_mean',
        index='scenario',
        columns='model',
        aggfunc='mean'
    )

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Process RSS (GB)', fontsize=12)
    ax.set_title('vLLM Process Memory Usage by Scenario (Rate=50)', fontsize=14, fontweight='bold')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'pcp_memory_usage.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'pcp_memory_usage.png'}")
    plt.close()

def plot_request_queue_patterns(df):
    """Plot request queue depths by scenario."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Running requests
    pivot_running = df.pivot_table(
        values='requests_running_mean',
        index='scenario',
        columns='model',
        aggfunc='mean'
    )

    pivot_running.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_xlabel('Scenario', fontsize=12)
    ax1.set_ylabel('Running Requests', fontsize=12)
    ax1.set_title('Mean Running Requests (Rate=50)', fontsize=13, fontweight='bold')
    ax1.legend(title='Model')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    # Waiting requests
    pivot_waiting = df.pivot_table(
        values='requests_waiting_mean',
        index='scenario',
        columns='model',
        aggfunc='mean'
    )

    pivot_waiting.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_xlabel('Scenario', fontsize=12)
    ax2.set_ylabel('Waiting Requests', fontsize=12)
    ax2.set_title('Mean Waiting Requests (Rate=50)', fontsize=13, fontweight='bold')
    ax2.legend(title='Model')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'pcp_request_queues.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'pcp_request_queues.png'}")
    plt.close()

def plot_prefix_cache_effectiveness(df):
    """Plot prefix cache hit rates."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Calculate hit rate percentage
    df['prefix_hit_rate'] = (df['prefix_hits_rate_mean'] / df['prefix_queries_rate_mean'] * 100).fillna(0)

    # Pivot for grouped bar chart
    pivot = df.pivot_table(
        values='prefix_hit_rate',
        index='scenario',
        columns='model',
        aggfunc='mean'
    )

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Prefix Cache Hit Rate (%)', fontsize=12)
    ax.set_title('Prefix Cache Effectiveness by Scenario (Rate=50)', fontsize=14, fontweight='bold')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'pcp_prefix_cache_hits.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'pcp_prefix_cache_hits.png'}")
    plt.close()

def create_correlation_heatmap(df):
    """Create heatmap showing correlations between system metrics and performance."""
    # Select key metrics for correlation
    metrics = [
        'output_tps_mean',
        'ttft_mean_ms',
        'tpot_mean_ms',
        'gpu_util_mean',
        'kv_cache_pct_mean',
        'requests_running_mean',
        'requests_waiting_mean',
        'process_rss_gb_mean',
    ]

    # Filter to available columns
    available_metrics = [m for m in metrics if m in df.columns]
    corr_df = df[available_metrics].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_df,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title('Correlation: System Metrics vs Performance', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'pcp_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {ANALYSIS_DIR / 'pcp_correlation_heatmap.png'}")
    plt.close()

def generate_summary_stats(df):
    """Generate summary statistics table."""
    summary = df.groupby('scenario').agg({
        'output_tps_mean': 'mean',
        'gpu_util_mean': 'mean',
        'kv_cache_pct_mean': 'mean',
        'requests_running_mean': 'mean',
        'requests_waiting_mean': 'mean',
        'process_rss_gb_mean': 'mean',
    }).round(2)

    summary.columns = [
        'Avg Throughput (tok/s)',
        'Avg GPU Util (%)',
        'Avg KV-Cache (%)',
        'Avg Running Reqs',
        'Avg Waiting Reqs',
        'Avg Process RSS (GB)'
    ]

    output_file = ANALYSIS_DIR / 'pcp_summary_stats.csv'
    summary.to_csv(output_file)
    print(f"\nSaved summary statistics: {output_file}")
    print("\nSummary Statistics:")
    print(summary)

    return summary

def main():
    print("Creating PCP metric visualizations...")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Merged dataset: {len(df)} benchmark runs")

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_gpu_util_vs_throughput(df)
    plot_kv_cache_usage_by_scenario(df)
    plot_memory_usage_comparison(df)
    plot_request_queue_patterns(df)
    plot_prefix_cache_effectiveness(df)
    create_correlation_heatmap(df)

    # Generate summary stats
    print("\nGenerating summary statistics...")
    summary = generate_summary_stats(df)

    print("\n" + "=" * 80)
    print("All visualizations created successfully!")
    print(f"Output directory: {ANALYSIS_DIR}/")

if __name__ == '__main__':
    main()
