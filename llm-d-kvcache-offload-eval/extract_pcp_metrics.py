#!/usr/bin/env python3
"""
Extract and visualize PCP metrics from benchmark archives.
Focuses on KV-cache utilization, memory usage, and vLLM metrics.
"""

import subprocess
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

CONFIGS = ['no-offload', 'native-offload', 'llm-d-redis', 'llm-d-valkey']
MODELS = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']

# Key metrics to extract from PCP archives
PCP_METRICS = [
    'openmetrics.vllm.vllm.kv_cache_usage_perc',
    'openmetrics.vllm.vllm.num_requests_running',
    'openmetrics.vllm.vllm.num_requests_waiting',
    'openmetrics.vllm.vllm.prefix_cache_hits_total',
    'openmetrics.vllm.vllm.prefix_cache_queries_total',
    'openmetrics.vllm.process_resident_memory_bytes',
    'openmetrics.vllm.process_virtual_memory_bytes',
    'mem.util.used',
    'mem.util.free',
    'kernel.all.cpu.idle',
    'kernel.all.cpu.user',
    'kernel.all.cpu.sys',
]

def decompress_pcp_archives(results_dir):
    """Decompress all PCP archives in a results directory."""
    pcp_dir = results_dir / 'pcp-archives'
    if not pcp_dir.exists():
        return None

    # Find all compressed archive files
    for node_dir in pcp_dir.iterdir():
        if not node_dir.is_dir():
            continue

        for zst_file in node_dir.glob('*.zst'):
            decompressed = zst_file.with_suffix('')
            if not decompressed.exists():
                print(f"  Decompressing: {zst_file.name}")
                subprocess.run(['zstd', '-d', '-k', str(zst_file)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Find the decompressed archive base name
    for node_dir in pcp_dir.iterdir():
        if not node_dir.is_dir():
            continue

        # Look for .meta file to find archive base name
        meta_files = list(node_dir.glob('*.meta'))
        if meta_files:
            # Return path without extension
            return str(meta_files[0]).replace('.meta', '')

    return None

def extract_pcp_metric_timeseries(archive_path, metric_name):
    """Extract time series data for a specific metric from PCP archive."""
    try:
        cmd = ['pmval', '-a', archive_path, '-t', '10sec', '-f', '3', '-r', metric_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return None

        # Parse pmval output
        lines = result.stdout.strip().split('\n')
        timestamps = []
        values = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('metric:') or line.startswith('host:') or line.startswith('semantics:') or line.startswith('units:') or line.startswith('samples:'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    # First part is timestamp, last part is value
                    timestamp = parts[0]
                    value_str = parts[-1]

                    # Handle '?' for missing values
                    if value_str == '?':
                        continue

                    value = float(value_str)
                    timestamps.append(timestamp)
                    values.append(value)
                except (ValueError, IndexError):
                    continue

        if not timestamps:
            return None

        return pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
        print(f"    Warning: Failed to extract {metric_name}: {e}")
        return None

def calculate_prefix_cache_hit_rate(archive_path):
    """Calculate prefix cache hit rate from PCP metrics."""
    hits_df = extract_pcp_metric_timeseries(archive_path, 'openmetrics.vllm.vllm.prefix_cache_hits_total')
    queries_df = extract_pcp_metric_timeseries(archive_path, 'openmetrics.vllm.vllm.prefix_cache_queries_total')

    if hits_df is None or queries_df is None:
        return None

    # Calculate hit rate as percentage
    # These are cumulative counters, so we need deltas
    hits = hits_df['value'].iloc[-1] - hits_df['value'].iloc[0] if len(hits_df) > 1 else 0
    queries = queries_df['value'].iloc[-1] - queries_df['value'].iloc[0] if len(queries_df) > 1 else 0

    if queries == 0:
        return 0.0

    return (hits / queries) * 100.0

def extract_all_metrics(results_dir):
    """Extract all metrics from a benchmark results directory."""
    print(f"Extracting metrics from: {results_dir.name}")

    # Decompress archives
    archive_path = decompress_pcp_archives(results_dir)
    if not archive_path:
        print("  No PCP archives found")
        return None

    print(f"  Archive: {Path(archive_path).name}")

    metrics = {}

    # Extract KV cache usage
    kv_cache_df = extract_pcp_metric_timeseries(archive_path, 'openmetrics.vllm.vllm.kv_cache_usage_perc')
    if kv_cache_df is not None:
        metrics['kv_cache_mean'] = kv_cache_df['value'].mean()
        metrics['kv_cache_max'] = kv_cache_df['value'].max()
        metrics['kv_cache_p95'] = kv_cache_df['value'].quantile(0.95)

    # Calculate prefix cache hit rate
    hit_rate = calculate_prefix_cache_hit_rate(archive_path)
    if hit_rate is not None:
        metrics['prefix_cache_hit_rate'] = hit_rate

    # Extract request queue metrics
    running_df = extract_pcp_metric_timeseries(archive_path, 'openmetrics.vllm.vllm.num_requests_running')
    if running_df is not None:
        metrics['requests_running_mean'] = running_df['value'].mean()
        metrics['requests_running_max'] = running_df['value'].max()

    waiting_df = extract_pcp_metric_timeseries(archive_path, 'openmetrics.vllm.vllm.num_requests_waiting')
    if waiting_df is not None:
        metrics['requests_waiting_mean'] = waiting_df['value'].mean()
        metrics['requests_waiting_max'] = waiting_df['value'].max()

    # Extract memory usage
    rss_df = extract_pcp_metric_timeseries(archive_path, 'openmetrics.vllm.process_resident_memory_bytes')
    if rss_df is not None:
        # Convert to GB
        metrics['memory_rss_mean_gb'] = rss_df['value'].mean() / (1024**3)
        metrics['memory_rss_max_gb'] = rss_df['value'].max() / (1024**3)

    return metrics

def create_comparison_plots(all_metrics_df, output_dir):
    """Create comparative visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Plot 1: KV Cache Usage Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('KV-Cache Utilization Across Configurations', fontsize=16, fontweight='bold')

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 2, idx % 2]
        model_data = all_metrics_df[all_metrics_df['model'] == model]

        if model_data.empty:
            ax.text(0.5, 0.5, f'No data for {model}', ha='center', va='center')
            ax.set_title(model)
            continue

        x_pos = np.arange(len(model_data))
        ax.bar(x_pos, model_data['kv_cache_mean'], yerr=model_data['kv_cache_max'] - model_data['kv_cache_mean'],
               capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_data['config'], rotation=45, ha='right')
        ax.set_ylabel('KV Cache Usage (%)')
        ax.set_title(model)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'kv_cache_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'kv_cache_comparison.png'}")
    plt.close()

    # Plot 2: Prefix Cache Hit Rate Comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    # Pivot data for grouped bar chart
    hit_rate_data = all_metrics_df.pivot(index='model', columns='config', values='prefix_cache_hit_rate')

    hit_rate_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Prefix Cache Hit Rate (%)')
    ax.set_title('Prefix Cache Hit Rate Across Configurations', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'prefix_cache_hit_rate.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'prefix_cache_hit_rate.png'}")
    plt.close()

    # Plot 3: Memory Usage Comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    memory_data = all_metrics_df.pivot(index='model', columns='config', values='memory_rss_mean_gb')

    memory_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Memory Usage (GB)')
    ax.set_title('Process Memory Usage Across Configurations', fontsize=14, fontweight='bold')
    ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'memory_usage.png'}")
    plt.close()

    # Plot 4: Request Queue Depth
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    running_data = all_metrics_df.pivot(index='model', columns='config', values='requests_running_mean')
    waiting_data = all_metrics_df.pivot(index='model', columns='config', values='requests_waiting_mean')

    running_data.plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Running Requests (avg)')
    axes[0].set_title('Average Running Requests', fontsize=12, fontweight='bold')
    axes[0].legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    waiting_data.plot(kind='bar', ax=axes[1], width=0.8)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Waiting Requests (avg)')
    axes[1].set_title('Average Waiting Requests', fontsize=12, fontweight='bold')
    axes[1].legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    fig.suptitle('Request Queue Depth Across Configurations', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'request_queues.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'request_queues.png'}")
    plt.close()

def main():
    results_base = Path('results')

    all_metrics = []

    for model in MODELS:
        for config in CONFIGS:
            dir_name = f"1x2xL40S_upstream-llm-d-0.4.0_{model}_{config}"
            results_dir = results_base / dir_name

            if not results_dir.exists():
                print(f"Skipping {dir_name} (not found)")
                continue

            metrics = extract_all_metrics(results_dir)
            if metrics:
                metrics['model'] = model
                metrics['config'] = config
                all_metrics.append(metrics)

    if not all_metrics:
        print("No metrics extracted!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_metrics)

    # Save to CSV
    analysis_dir = Path('analysis')
    analysis_dir.mkdir(exist_ok=True)

    csv_path = analysis_dir / 'pcp_metrics_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics summary to: {csv_path}")

    # Print summary
    print("\n" + "="*80)
    print("PCP Metrics Summary")
    print("="*80)
    print(df.to_string(index=False))

    # Create visualization plots
    print("\n" + "="*80)
    print("Creating visualization plots...")
    print("="*80)
    create_comparison_plots(df, analysis_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
