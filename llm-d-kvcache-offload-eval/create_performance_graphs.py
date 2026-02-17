#!/usr/bin/env python3
"""
Create performance visualization graphs from GuideLLM benchmark results.
Generates throughput vs concurrency curves and performance comparison charts.
"""

import json
import zstandard as zstd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

def decompress_and_load_json(file_path):
    """Decompress zstd file and load JSON data."""
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        data = dctx.decompress(f.read())
        return json.loads(data)

def extract_benchmark_data(results_dir):
    """Extract throughput vs concurrency data from all benchmark results."""
    results_dir = Path(results_dir)
    all_data = []

    # Find all guidellm-results.json.zst files
    for result_file in results_dir.glob('*/guidellm-results.json.zst'):
        try:
            data = decompress_and_load_json(result_file)

            # Extract configuration from directory name
            dir_name = result_file.parent.name
            # Format: 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-8B_llm-d-redis
            parts = dir_name.split('_')

            # Handle model names that may contain underscores or dashes
            # Find where model starts (after "llm-d-0.4.0")
            model_start_idx = None
            for i, part in enumerate(parts):
                if part.startswith('Qwen'):
                    model_start_idx = i
                    break

            if model_start_idx is None:
                print(f"Could not parse model from {dir_name}")
                continue

            # Model may be in multiple parts (e.g., Qwen3-32B-AWQ)
            config_start_idx = model_start_idx + 1
            for i in range(model_start_idx + 1, len(parts)):
                if parts[i] in ['no-offload', 'native-offload', 'llm-d-redis', 'llm-d-valkey']:
                    config_start_idx = i
                    break

            model = '_'.join(parts[model_start_idx:config_start_idx])
            config = '_'.join(parts[config_start_idx:])

            # GuideLLM stores results as a list of test runs with different concurrency levels
            if 'benchmarks' in data:
                for benchmark in data['benchmarks']:
                    if 'config' in benchmark and 'metrics' in benchmark:
                        # Extract concurrency from strategy
                        concurrency = benchmark['config']['strategy'].get('max_concurrency', 1)
                        metrics = benchmark['metrics']

                        # Extract metrics from successful requests
                        def get_metric(name):
                            metric_data = metrics.get(name, {}).get('successful', {})
                            return metric_data.get('mean', 0) if metric_data else 0

                        throughput = get_metric('output_tokens_per_second')
                        ttft_mean = get_metric('time_to_first_token_ms')
                        tpot_mean = get_metric('time_per_output_token_ms')

                        all_data.append({
                            'model': model,
                            'config': config,
                            'concurrency': concurrency,
                            'throughput': throughput,
                            'ttft_mean': ttft_mean,
                            'tpot_mean': tpot_mean
                        })
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return pd.DataFrame(all_data)

def create_throughput_curves(df, output_dir):
    """Create throughput vs concurrency curves for each model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    models = df['model'].unique()
    configs = ['no-offload', 'native-offload', 'llm-d-redis', 'llm-d-valkey']
    config_colors = {
        'no-offload': '#1f77b4',
        'native-offload': '#2ca02c',
        'llm-d-redis': '#ff7f0e',
        'llm-d-valkey': '#d62728'
    }
    config_labels = {
        'no-offload': 'No Offload (Baseline)',
        'native-offload': 'vLLM Native Offload',
        'llm-d-redis': 'llm-d Redis',
        'llm-d-valkey': 'llm-d Valkey'
    }

    # Create combined figure with all models
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, model in enumerate(sorted(models)):
        ax = axes[idx]
        model_data = df[df['model'] == model]

        for config in configs:
            config_data = model_data[model_data['config'] == config].sort_values('concurrency')
            if len(config_data) > 0:
                ax.plot(config_data['concurrency'], config_data['throughput'],
                       marker='o', linewidth=2, markersize=6,
                       color=config_colors.get(config, None),
                       label=config_labels.get(config, config))

        ax.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Output Throughput (tokens/s)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model} Throughput vs Concurrency', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_vs_concurrency_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create individual plots for each model
    for model in sorted(models):
        fig, ax = plt.subplots(figsize=(12, 7))
        model_data = df[df['model'] == model]

        for config in configs:
            config_data = model_data[model_data['config'] == config].sort_values('concurrency')
            if len(config_data) > 0:
                ax.plot(config_data['concurrency'], config_data['throughput'],
                       marker='o', linewidth=2.5, markersize=8,
                       color=config_colors.get(config, None),
                       label=config_labels.get(config, config))

        ax.set_xlabel('Concurrency Level', fontsize=13, fontweight='bold')
        ax.set_ylabel('Output Throughput (tokens/s)', fontsize=13, fontweight='bold')
        ax.set_title(f'{model} Output Throughput vs Concurrency', fontsize=15, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_dir / f'throughput_curve_{model}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Created throughput curves in {output_dir}")

def create_peak_performance_comparison(df, output_dir):
    """Create bar charts comparing peak throughput across configurations."""
    output_dir = Path(output_dir)

    # Get peak throughput for each model+config
    peak_data = df.loc[df.groupby(['model', 'config'])['throughput'].idxmax()]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    models = sorted(peak_data['model'].unique())
    configs = ['no-offload', 'llm-d-redis', 'llm-d-valkey', 'native-offload']
    config_labels = {
        'no-offload': 'No Offload',
        'native-offload': 'Native Offload',
        'llm-d-redis': 'llm-d Redis',
        'llm-d-valkey': 'llm-d Valkey'
    }
    config_colors = {
        'no-offload': '#5588cc',
        'native-offload': '#66cc66',
        'llm-d-redis': '#ffaa55',
        'llm-d-valkey': '#ee6666'
    }

    x = np.arange(len(models))
    width = 0.2

    for idx, config in enumerate(configs):
        offsets = (idx - 1.5) * width
        values = []
        for model in models:
            model_config_data = peak_data[(peak_data['model'] == model) & (peak_data['config'] == config)]
            if len(model_config_data) > 0:
                values.append(model_config_data['throughput'].values[0])
            else:
                values.append(0)

        bars = ax.bar(x + offsets, values, width, label=config_labels[config],
                     color=config_colors[config], edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Peak Output Throughput (tokens/s)', fontsize=13, fontweight='bold')
    ax.set_title('Peak Throughput Comparison Across Configurations', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'peak_throughput_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Created peak performance comparison in {output_dir}")

def create_latency_comparison(df, output_dir):
    """Create latency comparison charts."""
    output_dir = Path(output_dir)

    # Get metrics at peak throughput
    peak_data = df.loc[df.groupby(['model', 'config'])['throughput'].idxmax()]

    # Create TTFT comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    models = sorted(peak_data['model'].unique())
    configs = ['no-offload', 'llm-d-redis', 'llm-d-valkey', 'native-offload']
    config_labels = {
        'no-offload': 'No Offload',
        'native-offload': 'Native Offload',
        'llm-d-redis': 'llm-d Redis',
        'llm-d-valkey': 'llm-d Valkey'
    }
    config_colors = {
        'no-offload': '#5588cc',
        'native-offload': '#66cc66',
        'llm-d-redis': '#ffaa55',
        'llm-d-valkey': '#ee6666'
    }

    x = np.arange(len(models))
    width = 0.2

    # TTFT comparison
    for idx, config in enumerate(configs):
        offsets = (idx - 1.5) * width
        values = []
        for model in models:
            model_config_data = peak_data[(peak_data['model'] == model) & (peak_data['config'] == config)]
            if len(model_config_data) > 0:
                values.append(model_config_data['ttft_mean'].values[0])
            else:
                values.append(0)

        ax1.bar(x + offsets, values, width, label=config_labels[config],
               color=config_colors[config], edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean TTFT (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Time to First Token at Peak Throughput', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # TPOT comparison
    for idx, config in enumerate(configs):
        offsets = (idx - 1.5) * width
        values = []
        for model in models:
            model_config_data = peak_data[(peak_data['model'] == model) & (peak_data['config'] == config)]
            if len(model_config_data) > 0:
                values.append(model_config_data['tpot_mean'].values[0])
            else:
                values.append(0)

        ax2.bar(x + offsets, values, width, label=config_labels[config],
               color=config_colors[config], edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean TPOT (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Time Per Output Token at Peak Throughput', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Created latency comparison in {output_dir}")

def create_performance_delta_heatmap(df, output_dir):
    """Create heatmap showing performance delta vs baseline."""
    output_dir = Path(output_dir)

    # Get peak throughput for each model+config
    peak_data = df.loc[df.groupby(['model', 'config'])['throughput'].idxmax()]

    # Create pivot table
    pivot = peak_data.pivot(index='config', columns='model', values='throughput')

    # Calculate percentage difference from no-offload baseline
    baseline = pivot.loc['no-offload']
    delta_pct = ((pivot - baseline) / baseline * 100).round(1)

    # Reorder rows
    row_order = ['no-offload', 'llm-d-redis', 'llm-d-valkey', 'native-offload']
    delta_pct = delta_pct.reindex(row_order)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(delta_pct, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Performance Delta (%)'},
                linewidths=0.5, linecolor='gray',
                vmin=-35, vmax=5, ax=ax)

    ax.set_title('Throughput Performance vs Baseline (no-offload)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')

    # Update y-tick labels
    yticklabels = ['No Offload (Baseline)', 'llm-d Redis', 'llm-d Valkey', 'vLLM Native Offload']
    ax.set_yticklabels(yticklabels, rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_delta_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Created performance delta heatmap in {output_dir}")

def main():
    results_dir = Path('results')
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)

    print("Extracting benchmark data from GuideLLM results...")
    df = extract_benchmark_data(results_dir)

    if len(df) == 0:
        print("ERROR: No benchmark data found!")
        return

    print(f"Found {len(df)} data points across {df['model'].nunique()} models")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Configs: {sorted(df['config'].unique())}")

    print("\nCreating throughput vs concurrency curves...")
    create_throughput_curves(df, output_dir)

    print("Creating peak performance comparison...")
    create_peak_performance_comparison(df, output_dir)

    print("Creating latency comparison charts...")
    create_latency_comparison(df, output_dir)

    print("Creating performance delta heatmap...")
    create_performance_delta_heatmap(df, output_dir)

    print("\nAll performance graphs created successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
