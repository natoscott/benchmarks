#!/usr/bin/env python3
"""Generate visualizations for v0.5.0 native offload evaluation."""

import json
import subprocess
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Visualization standards from .claude/skills/visualization-palette
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

def extract_all_metrics(results_dir):
    """Extract comprehensive metrics from all GuideLLM result files."""
    results_dir = Path(results_dir)
    all_data = []

    for result_file in results_dir.glob('*/guidellm-results.json.zst'):
        try:
            # Decompress and read
            result = subprocess.run(
                ['zstdcat', result_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            data = json.loads(result.stdout)

            # Parse directory name
            # Format: 1x2xL40S_upstream-llm-d-0.5.0_Qwen3-14B_native-offload-10k_replica1_rate50
            dir_name = result_file.parent.name
            parts = dir_name.split('_')

            model = parts[2]
            config = parts[3]
            rate = int(parts[5].replace('rate', ''))

            # Extract metrics
            benchmark = data['benchmarks'][0]
            metrics = benchmark['metrics']

            # Throughput
            total_output_tokens = metrics['output_token_count']['successful']['total_sum']
            duration = benchmark['duration']
            throughput = total_output_tokens / duration

            # Latency metrics (all in milliseconds from GuideLLM)
            ttft_median = metrics.get('time_to_first_token_ms', {}).get('successful', {}).get('median', 0) / 1000  # ms to s
            itl_median = metrics.get('inter_token_latency_ms', {}).get('successful', {}).get('median', 0)
            tpot_median = metrics.get('time_per_output_token_ms', {}).get('successful', {}).get('median', 0)

            # Request counts
            request_totals = metrics['request_totals']
            completed = request_totals.get('completed', 0)

            all_data.append({
                'model': model,
                'config': config,
                'rate': rate,
                'throughput': throughput,
                'completed_requests': completed,
                'ttft_median': ttft_median,
                'itl_median': itl_median,
                'tpot_median': tpot_median,
            })

        except Exception as e:
            print(f"Warning: Failed to process {result_file}: {e}")
            continue

    return pd.DataFrame(all_data)

def find_peak_throughput(df):
    """Find peak throughput for each model/config combination."""
    peak_data = []

    for (model, config), group in df.groupby(['model', 'config']):
        peak_row = group.loc[group['throughput'].idxmax()]
        peak_data.append({
            'model': model,
            'config': config,
            'peak_throughput': peak_row['throughput'],
            'optimal_rate': peak_row['rate'],
            'ttft_median': peak_row['ttft_median'],
            'itl_median': peak_row['itl_median'],
            'tpot_median': peak_row['tpot_median'],
        })

    return pd.DataFrame(peak_data)

def plot_peak_throughput(peak_df, output_file):
    """Bar chart showing peak throughput across models and configs."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for grouped bar chart
    models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
    configs = ['no-offload', 'native-offload-10k', 'native-offload-20k']

    x = np.arange(len(models))
    width = 0.25

    for i, config in enumerate(configs):
        config_data = peak_df[peak_df['config'] == config]
        throughputs = [
            config_data[config_data['model'] == model]['peak_throughput'].values[0]
            if len(config_data[config_data['model'] == model]) > 0 else 0
            for model in models
        ]
        ax.bar(x + i*width, throughputs, width, label=config)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Peak Throughput (tokens/s)', fontsize=12)
    ax.set_title('Peak Throughput: v0.5.0 Native Offload vs Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def plot_throughput_vs_concurrency(df, output_file):
    """Line plots showing throughput vs concurrency for all models."""
    models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, model in enumerate(models):
        model_data = df[df['model'] == model]

        for config in model_data['config'].unique():
            config_data = model_data[model_data['config'] == config].sort_values('rate')
            axes[i].plot(config_data['rate'], config_data['throughput'],
                        marker='o', label=config, linewidth=2)

        axes[i].set_xlabel('Concurrency (rate)', fontsize=11)
        axes[i].set_ylabel('Throughput (tokens/s)', fontsize=11)
        axes[i].set_title(f'{model}', fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)

    fig.suptitle('Throughput vs Concurrency: v0.5.0 Native Offload',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def plot_latency_comparison(peak_df, output_file):
    """Bar charts comparing latency metrics at peak throughput."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
    configs = ['no-offload', 'native-offload-10k', 'native-offload-20k']

    x = np.arange(len(models))
    width = 0.25

    # TTFT
    for i, config in enumerate(configs):
        config_data = peak_df[peak_df['config'] == config]
        ttfts = [
            config_data[config_data['model'] == model]['ttft_median'].values[0]
            if len(config_data[config_data['model'] == model]) > 0 else 0
            for model in models
        ]
        axes[0].bar(x + i*width, ttfts, width, label=config)

    axes[0].set_xlabel('Model', fontsize=11)
    axes[0].set_ylabel('TTFT (seconds)', fontsize=11)
    axes[0].set_title('Time to First Token (median)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # ITL
    for i, config in enumerate(configs):
        config_data = peak_df[peak_df['config'] == config]
        itls = [
            config_data[config_data['model'] == model]['itl_median'].values[0]
            if len(config_data[config_data['model'] == model]) > 0 else 0
            for model in models
        ]
        axes[1].bar(x + i*width, itls, width, label=config)

    axes[1].set_xlabel('Model', fontsize=11)
    axes[1].set_ylabel('ITL (milliseconds)', fontsize=11)
    axes[1].set_title('Inter-Token Latency (median)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def plot_version_comparison_heatmap(peak_df, v040_data, output_file):
    """Heatmap showing v0.5.0 vs v0.4.0 percentage point changes."""
    # Load v0.4.0 native offload results
    v040_native = {
        'Qwen3-0.6B': {'no-offload': 602.0, 'native-offload-10k': 426.8},
        'Qwen3-8B': {'no-offload': 113.0, 'native-offload-10k': 71.8},
        'Qwen3-14B': {'no-offload': 58.7, 'native-offload-10k': 59.0},
        'Qwen3-32B-AWQ': {'no-offload': 49.2, 'native-offload-10k': 48.7},
    }

    # Calculate percentage changes
    models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
    configs = ['no-offload', 'native-offload-10k', 'native-offload-20k']

    # Build comparison matrix
    comparison_data = []

    for model in models:
        row = []
        v040_baseline = v040_native[model]['no-offload']
        v040_offload = v040_native[model]['native-offload-10k']
        v040_delta = ((v040_offload - v040_baseline) / v040_baseline) * 100

        # Get v0.5.0 data
        model_data = peak_df[peak_df['model'] == model]

        for config in configs:
            config_row = model_data[model_data['config'] == config]
            if len(config_row) > 0:
                v050_throughput = config_row['peak_throughput'].values[0]
                v050_baseline = peak_df[(peak_df['model'] == model) &
                                       (peak_df['config'] == 'no-offload')]['peak_throughput'].values[0]
                v050_delta = ((v050_throughput - v050_baseline) / v050_baseline) * 100

                # Calculate improvement: v0.5.0 delta - v0.4.0 delta
                if 'offload' in config:
                    improvement = v050_delta - v040_delta
                    row.append(improvement)
                else:
                    # For baseline, show version-to-version change
                    version_change = ((v050_throughput - v040_baseline) / v040_baseline) * 100
                    row.append(version_change)
            else:
                row.append(0)

        comparison_data.append(row)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    df_heatmap = pd.DataFrame(comparison_data,
                              index=models,
                              columns=['Baseline\n(v0.5.0 vs v0.4.0)',
                                      'Native Offload 10K\n(improvement vs v0.4.0)',
                                      'Native Offload 20K\n(improvement vs v0.4.0)'])

    sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Percentage Point Change'}, ax=ax,
                vmin=-60, vmax=30)

    ax.set_title('v0.5.0 vs v0.4.0: Native Offload Performance Change',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12)
    ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def main():
    """Generate all v0.5.0 visualizations."""
    results_dir = Path('results')
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)

    print("Extracting metrics from GuideLLM results...")
    df = extract_all_metrics(results_dir)

    if df.empty:
        print("ERROR: No data extracted from results")
        return

    print(f"Extracted {len(df)} data points")

    # Find peak throughput for each config
    print("Calculating peak throughput...")
    peak_df = find_peak_throughput(df)

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_peak_throughput(peak_df, output_dir / 'v0.5.0_peak_throughput.png')
    plot_throughput_vs_concurrency(df, output_dir / 'v0.5.0_throughput_vs_concurrency.png')
    plot_latency_comparison(peak_df, output_dir / 'v0.5.0_latency_comparison.png')
    plot_version_comparison_heatmap(peak_df, None, output_dir / 'v0.5.0_vs_v0.4.0_heatmap.png')

    # Save peak data to CSV
    peak_csv = output_dir / 'v0.5.0_peak_throughput_all.csv'
    peak_df.to_csv(peak_csv, index=False)
    print(f"\nSaved peak throughput data to: {peak_csv}")

    print("\nVisualization generation complete!")

if __name__ == '__main__':
    main()
