#!/usr/bin/env python3
"""Extract and visualize PCP metrics for v0.5.0 benchmarks."""

import subprocess
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Visualization standards
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

def extract_pcp_metrics(archive_base, metrics_list, start_time=None, end_time=None):
    """Extract metrics from PCP archive using pmrep."""
    cmd = ['pmrep', '-a', archive_base, '-o', 'csv']

    if start_time and end_time:
        cmd.extend(['-S', start_time, '-T', end_time])

    cmd.extend(metrics_list)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None

        # Parse CSV output
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return None

        # Skip header and timestamp column
        data = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) > 1:
                try:
                    values = [float(x) if x else 0.0 for x in parts[1:]]
                    data.append(values)
                except ValueError:
                    continue

        if not data:
            return None

        df = pd.DataFrame(data, columns=metrics_list)
        return df.mean()

    except Exception as e:
        print(f"  Warning: Failed to extract metrics: {e}")
        return None

def find_peak_throughput_configs():
    """Find result directories for peak throughput scenarios (rate=50 for most models)."""
    results_dir = Path('results')
    configs = []

    # For v0.5.0, we want rate=50 for 0.6B, 8B, 14B and rate=1 for 32B-AWQ
    models_rate50 = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B']
    model_rate1 = 'Qwen3-32B-AWQ'

    config_types = ['no-offload', 'native-offload-10k', 'native-offload-20k']

    for model in models_rate50:
        for config in config_types:
            pattern = f"1x2xL40S_upstream-llm-d-0.5.0_{model}_{config}_replica1_rate50"
            dirs = list(results_dir.glob(pattern))
            if dirs:
                configs.append({
                    'model': model,
                    'config': config,
                    'rate': 50,
                    'path': dirs[0]
                })

    for config in config_types:
        pattern = f"1x2xL40S_upstream-llm-d-0.5.0_{model_rate1}_{config}_replica1_rate1"
        dirs = list(results_dir.glob(pattern))
        if dirs:
            configs.append({
                'model': model_rate1,
                'config': config,
                'rate': 1,
                'path': dirs[0]
            })

    return configs

def extract_all_pcp_metrics(configs):
    """Extract PCP metrics from all configurations."""
    all_data = []

    # Metrics to extract
    gpu_metrics = [
        'openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL',
        'openmetrics.dcgm.DCGM_FI_DEV_MEM_COPY_UTIL',
    ]

    vllm_metrics = [
        'openmetrics.vllm.vllm.kv_cache_usage_perc',
        'openmetrics.vllm.vllm.num_requests_running',
        'openmetrics.vllm.vllm.num_requests_waiting',
        'openmetrics.vllm.vllm.prefix_cache_hits_total',
        'openmetrics.vllm.vllm.prefix_cache_queries_total',
    ]

    for cfg in configs:
        print(f"Processing {cfg['model']} {cfg['config']} (rate={cfg['rate']})...")

        # Find PCP archives
        archive_dirs = list(cfg['path'].glob('pcp-archives/*/'))
        if not archive_dirs:
            print(f"  No PCP archives found")
            continue

        # Use first node's archives
        archive_dir = archive_dirs[0]

        # Clean up any spurious files (decompressed archives from testing)
        for spurious in archive_dir.glob('202*'):
            if not str(spurious).endswith('.zst'):
                spurious.unlink()

        # Look for .meta.zst files (compressed PCP archives)
        archive_files = list(archive_dir.glob('*.meta.zst'))

        if not archive_files:
            print(f"  No PCP archive files found")
            continue

        # Get archive base path (remove .meta.zst extension)
        archive_base = str(archive_files[0]).replace('.meta.zst', '')

        # Extract GPU metrics
        gpu_data = extract_pcp_metrics(archive_base, gpu_metrics)
        vllm_data = extract_pcp_metrics(archive_base, vllm_metrics)

        if gpu_data is not None or vllm_data is not None:
            row = {
                'model': cfg['model'],
                'config': cfg['config'],
                'rate': cfg['rate'],
            }

            if gpu_data is not None:
                row['gpu_util'] = gpu_data.get('openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL', 0)
                row['gpu_mem_copy_util'] = gpu_data.get('openmetrics.dcgm.DCGM_FI_DEV_MEM_COPY_UTIL', 0)

            if vllm_data is not None:
                row['kv_cache_usage'] = vllm_data.get('openmetrics.vllm.vllm.kv_cache_usage_perc', 0)
                row['requests_running'] = vllm_data.get('openmetrics.vllm.vllm.num_requests_running', 0)
                row['requests_waiting'] = vllm_data.get('openmetrics.vllm.vllm.num_requests_waiting', 0)

                # Calculate prefix cache hit rate from hits and queries
                hits = vllm_data.get('openmetrics.vllm.vllm.prefix_cache_hits_total', 0)
                queries = vllm_data.get('openmetrics.vllm.vllm.prefix_cache_queries_total', 0)
                row['prefix_cache_hit_rate'] = (hits / queries * 100) if queries > 0 else 0

            all_data.append(row)
            print(f"  Extracted metrics successfully")

    return pd.DataFrame(all_data)

def plot_gpu_utilization(df, output_file):
    """Bar chart of GPU utilization by model and config."""
    fig, ax = plt.subplots(figsize=(14, 6))

    models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
    configs = df['config'].unique()

    x = np.arange(len(models))
    width = 0.25

    for i, config in enumerate(sorted(configs)):
        config_data = df[df['config'] == config]
        utils = [
            config_data[config_data['model'] == model]['gpu_util'].values[0]
            if len(config_data[config_data['model'] == model]) > 0 else 0
            for model in models
        ]
        ax.bar(x + i*width, utils, width, label=config)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization at Peak Throughput: v0.5.0', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def plot_kv_cache_usage(df, output_file):
    """Bar chart of KV-cache usage by model and config."""
    fig, ax = plt.subplots(figsize=(14, 6))

    models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
    configs = df['config'].unique()

    x = np.arange(len(models))
    width = 0.25

    for i, config in enumerate(sorted(configs)):
        config_data = df[df['config'] == config]
        usage = [
            config_data[config_data['model'] == model]['kv_cache_usage'].values[0]
            if len(config_data[config_data['model'] == model]) > 0 else 0
            for model in models
        ]
        ax.bar(x + i*width, usage, width, label=config)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('KV-Cache Usage (%)', fontsize=12)
    ax.set_title('GPU KV-Cache Utilization at Peak Throughput: v0.5.0', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def plot_request_queues(df, output_file):
    """Grouped bar chart showing running and waiting requests."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
    configs = df['config'].unique()

    x = np.arange(len(models))
    width = 0.25

    # Running requests
    for i, config in enumerate(sorted(configs)):
        config_data = df[df['config'] == config]
        running = [
            config_data[config_data['model'] == model]['requests_running'].values[0]
            if len(config_data[config_data['model'] == model]) > 0 else 0
            for model in models
        ]
        axes[0].bar(x + i*width, running, width, label=config)

    axes[0].set_xlabel('Model', fontsize=11)
    axes[0].set_ylabel('Running Requests', fontsize=11)
    axes[0].set_title('Mean Running Requests', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # Waiting requests
    for i, config in enumerate(sorted(configs)):
        config_data = df[df['config'] == config]
        waiting = [
            config_data[config_data['model'] == model]['requests_waiting'].values[0]
            if len(config_data[config_data['model'] == model]) > 0 else 0
            for model in models
        ]
        axes[1].bar(x + i*width, waiting, width, label=config)

    axes[1].set_xlabel('Model', fontsize=11)
    axes[1].set_ylabel('Waiting Requests', fontsize=11)
    axes[1].set_title('Mean Waiting Requests', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def plot_prefix_cache_hits(df, output_file):
    """Bar chart of prefix cache hit rates."""
    fig, ax = plt.subplots(figsize=(14, 6))

    models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
    configs = df['config'].unique()

    x = np.arange(len(models))
    width = 0.25

    for i, config in enumerate(sorted(configs)):
        config_data = df[df['config'] == config]
        hit_rates = [
            config_data[config_data['model'] == model]['prefix_cache_hit_rate'].values[0]
            if len(config_data[config_data['model'] == model]) > 0 else 0
            for model in models
        ]
        ax.bar(x + i*width, hit_rates, width, label=config)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Prefix Cache Hit Rate (%)', fontsize=12)
    ax.set_title('Prefix Cache Effectiveness at Peak Throughput: v0.5.0', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def extract_v04_throughput_for_comparison():
    """Extract v0.4.0 throughput data for 14B model comparison."""
    import json

    v04_data = {}
    configs = {
        'no-offload': 'results/1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_no-offload_replica1_rate50',
        'native-offload': 'results/1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_native-offload_replica1_rate50',
    }

    for config_name, path_str in configs.items():
        path = Path(path_str)
        json_file = path / 'guidellm-results.json.zst'
        if json_file.exists():
            try:
                result = subprocess.run(
                    ['zstdcat', str(json_file)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    throughput = data['benchmarks'][0]['metrics']['output_tokens_per_second']['successful']['mean']
                    v04_data[config_name] = throughput
            except Exception as e:
                print(f"  Warning: Could not extract v0.4.0 {config_name}: {e}")

    return v04_data

def extract_v05_throughput_for_comparison():
    """Extract v0.5.0 throughput data for 14B model comparison."""
    import json

    v05_data = {}
    configs = {
        'no-offload': 'results/1x2xL40S_upstream-llm-d-0.5.0_Qwen3-14B_no-offload_replica1_rate50',
        'native-offload-10k': 'results/1x2xL40S_upstream-llm-d-0.5.0_Qwen3-14B_native-offload-10k_replica1_rate50',
        'native-offload-20k': 'results/1x2xL40S_upstream-llm-d-0.5.0_Qwen3-14B_native-offload-20k_replica1_rate50',
    }

    for config_name, path_str in configs.items():
        path = Path(path_str)
        json_file = path / 'guidellm-results.json.zst'
        if json_file.exists():
            try:
                result = subprocess.run(
                    ['zstdcat', str(json_file)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    throughput = data['benchmarks'][0]['metrics']['output_tokens_per_second']['successful']['mean']
                    v05_data[config_name] = throughput
            except Exception as e:
                print(f"  Warning: Could not extract v0.5.0 {config_name}: {e}")

    return v05_data

def plot_v04_v05_regression(output_file):
    """Create visualization comparing v0.4.0 vs v0.5.0 for 14B model."""
    v04_data = extract_v04_throughput_for_comparison()
    v05_data = extract_v05_throughput_for_comparison()

    if not v04_data or not v05_data:
        print("  Warning: Could not create v0.4.0 vs v0.5.0 comparison - missing data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Absolute throughput
    configs = ['no-offload', 'native-offload']
    x = np.arange(len(configs))
    width = 0.35

    v04_values = [v04_data.get('no-offload', 0), v04_data.get('native-offload', 0)]
    v05_10k_values = [v05_data.get('no-offload', 0), v05_data.get('native-offload-10k', 0)]

    ax1.bar(x - width/2, v04_values, width, label='v0.4.0 (vLLM 0.11.2)', color='#5DA5DA')
    ax1.bar(x + width/2, v05_10k_values, width, label='v0.5.0 (vLLM 0.14.1)', color='#FAA43A')

    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax1.set_title('Qwen3-14B: v0.4.0 vs v0.5.0 Throughput', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Baseline\n(no-offload)', 'Native Offload\n(10K blocks)'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (v04, v05) in enumerate(zip(v04_values, v05_10k_values)):
        ax1.text(i - width/2, v04 + 1, f'{v04:.1f}', ha='center', va='bottom', fontsize=10)
        ax1.text(i + width/2, v05 + 1, f'{v05:.1f}', ha='center', va='bottom', fontsize=10)

    # Right plot: Performance delta vs baseline
    if 'no-offload' in v04_data and 'native-offload' in v04_data:
        v04_baseline = v04_data['no-offload']
        v04_delta = ((v04_data['native-offload'] - v04_baseline) / v04_baseline) * 100
    else:
        v04_delta = 0

    if 'no-offload' in v05_data and 'native-offload-10k' in v05_data:
        v05_baseline = v05_data['no-offload']
        v05_delta = ((v05_data['native-offload-10k'] - v05_baseline) / v05_baseline) * 100
    else:
        v05_delta = 0

    versions = ['v0.4.0\n(vLLM 0.11.2)', 'v0.5.0\n(vLLM 0.14.1)']
    deltas = [v04_delta, v05_delta]
    colors = ['green' if d >= 0 else 'red' for d in deltas]

    bars = ax2.bar(versions, deltas, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Performance Delta vs Baseline (%)', fontsize=12)
    ax2.set_title('Native Offload Performance Change', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        label_y = height + (1 if height >= 0 else -2)
        ax2.text(bar.get_x() + bar.get_width()/2, label_y,
                f'{delta:+.2f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=11, fontweight='bold')

    # Add regression annotation
    regression = v05_delta - v04_delta
    ax2.text(0.5, 0.95, f'Regression: {regression:.2f} percentage points',
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")

def main():
    """Main execution."""
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)

    print("Finding peak throughput configurations...")
    configs = find_peak_throughput_configs()
    print(f"Found {len(configs)} configurations to analyze")

    if not configs:
        print("ERROR: No configurations found")
        return

    print("\nExtracting PCP metrics...")
    df = extract_all_pcp_metrics(configs)

    if df.empty:
        print("ERROR: No metrics extracted")
        return

    print(f"\nExtracted metrics from {len(df)} scenarios")

    # Save metrics to CSV
    csv_file = output_dir / 'v0.5.0_pcp_metrics.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved metrics to: {csv_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    if 'gpu_util' in df.columns:
        plot_gpu_utilization(df, output_dir / 'v0.5.0_pcp_gpu_utilization.png')

    if 'kv_cache_usage' in df.columns:
        plot_kv_cache_usage(df, output_dir / 'v0.5.0_pcp_kv_cache_usage.png')

    if 'requests_running' in df.columns and 'requests_waiting' in df.columns:
        plot_request_queues(df, output_dir / 'v0.5.0_pcp_request_queues.png')

    if 'prefix_cache_hit_rate' in df.columns:
        plot_prefix_cache_hits(df, output_dir / 'v0.5.0_pcp_prefix_cache_hits.png')

    # Generate v0.4.0 vs v0.5.0 comparison
    print("\nGenerating v0.4.0 vs v0.5.0 regression analysis...")
    plot_v04_v05_regression(output_dir / 'v0.5.0_regression_comparison.png')

    print("\nPCP analysis complete!")
    print(f"\nSummary statistics:")

    # Only show columns that exist
    summary_cols = [col for col in ['kv_cache_usage', 'requests_running', 'requests_waiting', 'prefix_cache_hit_rate'] if col in df.columns]
    if summary_cols:
        print(df.groupby('config')[summary_cols].mean())

if __name__ == '__main__':
    main()
