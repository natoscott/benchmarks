#!/usr/bin/env python3
"""
Comprehensive analysis of KV-cache management strategies.
Extracts metrics from guidellm results and PCP archives.
"""

import json
import zstandard as zstd
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configuration
SCENARIOS = [
    'no-offload',
    'native-offload',
    'lmcache-local',
    'lmcache-redis',
    'lmcache-valkey',
    'llm-d-redis',
    'llm-d-valkey'
]

MODELS = [
    'Qwen3-0.6B',
    'Qwen3-8B',
    'Qwen3-14B',
    'Qwen3-32B-AWQ'
]

# Model ordering for plots (already in correct order)
MODEL_ORDER = MODELS

# Set plotting style per visualization-palette skill
sns.set_style("whitegrid")
sns.set_palette("muted")  # Qualitative palette for categorical data
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

def decompress_zstd(file_path):
    """Decompress zstd file and return JSON data."""
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(f.read())
        return json.loads(decompressed)

def extract_guidellm_metrics(result_dir):
    """Extract metrics from guidellm results."""
    # Parse directory name
    dir_name = result_dir.name
    parts = dir_name.split('_')

    model = None
    scenario = None
    rate = None

    for i, part in enumerate(parts):
        if part in MODELS:
            model = part
        if part in SCENARIOS:
            scenario = part
        if part.startswith('rate'):
            rate = int(part[4:])

    if not all([model, scenario, rate]):
        return None

    # Load guidellm results
    guidellm_file = result_dir / 'guidellm-results.json.zst'
    if not guidellm_file.exists():
        return None

    try:
        data = decompress_zstd(guidellm_file)
    except Exception as e:
        print(f"Error decompressing {guidellm_file}: {e}")
        return None

    if 'benchmarks' not in data or len(data['benchmarks']) == 0:
        return None

    benchmark = data['benchmarks'][0]

    # Get metrics object (new guidellm format)
    metrics = benchmark.get('metrics', {})

    # Extract throughput metrics (tokens per second)
    output_tps_stats = metrics.get('output_tokens_per_second', {}).get('successful', {})
    output_tps_mean = output_tps_stats.get('mean')
    output_tps_median = output_tps_stats.get('median')

    # Extract latency metrics (milliseconds)
    ttft_stats = metrics.get('time_to_first_token_ms', {}).get('successful', {})
    ttft_mean = ttft_stats.get('mean')
    ttft_median = ttft_stats.get('median')

    tpot_stats = metrics.get('time_per_output_token_ms', {}).get('successful', {})
    tpot_mean = tpot_stats.get('mean')
    tpot_median = tpot_stats.get('median')

    # Extract request metrics
    request_totals = metrics.get('request_totals', {})
    completed = request_totals.get('successful', 0)

    return {
        'model': model,
        'scenario': scenario,
        'rate': rate,
        'output_tps_mean': output_tps_mean,
        'output_tps_median': output_tps_median,
        'ttft_mean_ms': ttft_mean,
        'ttft_median_ms': ttft_median,
        'tpot_mean_ms': tpot_mean,
        'tpot_median_ms': tpot_median,
        'completed_requests': completed
    }

def extract_pcp_metrics(result_dir):
    """Extract metrics from PCP archives using pmrep."""
    pcp_dir = result_dir / 'pcp-archives'
    if not pcp_dir.exists():
        return {}

    # Find PCP archive files
    archive_files = []
    for node_dir in pcp_dir.iterdir():
        if node_dir.is_dir():
            # Look for .meta.zst files
            meta_files = list(node_dir.glob('*.meta.zst'))
            if meta_files:
                # Decompress archives first
                for meta_file in meta_files:
                    base_name = str(meta_file).replace('.zst', '')
                    # Check if already decompressed
                    if not Path(base_name).exists():
                        subprocess.run(['zstd', '-d', '-f', str(meta_file)],
                                     stderr=subprocess.DEVNULL, check=False)
                    # Also decompress .index and data files
                    index_file = str(meta_file).replace('.meta.zst', '.index.zst')
                    if Path(index_file).exists():
                        subprocess.run(['zstd', '-d', '-f', index_file],
                                     stderr=subprocess.DEVNULL, check=False)

                    # Find data files (*.0.zst, *.1.zst, etc.)
                    data_files = list(node_dir.glob(f"{meta_file.stem.split('.')[0]}*.zst"))
                    for data_file in data_files:
                        if not data_file.name.endswith('.meta.zst') and not data_file.name.endswith('.index.zst'):
                            base_data = str(data_file).replace('.zst', '')
                            if not Path(base_data).exists():
                                subprocess.run(['zstd', '-d', '-f', str(data_file)],
                                             stderr=subprocess.DEVNULL, check=False)

                # Add base archive path (without extension)
                archive_base = str(meta_file).replace('.meta.zst', '').replace('.meta', '')
                archive_files.append(archive_base)

    if not archive_files:
        return {}

    metrics = {}

    # Use pmrep to extract key metrics
    # GPU utilization, memory, KV-cache usage, CPU utilization
    metric_specs = {
        'gpu_util': 'dcgm.gpu_utilization',
        'gpu_mem_util': 'dcgm.memory_utilization',
        'cpu_util': 'kernel.all.cpu.idle',  # We'll convert to utilization
        'vllm_kv_cache_usage': 'vllm.gpu_cache_usage_perc',
        'vllm_prefix_cache_hit_rate': 'vllm.cache_config_prefix_cache_hit_rate',
        'vllm_running_requests': 'vllm.num_requests_running',
        'vllm_waiting_requests': 'vllm.num_requests_waiting',
    }

    for metric_name, pcp_metric in metric_specs.items():
        try:
            cmd = ['pmrep', '-a', archive_files[0], '-t', '1s', '-o', 'csv',
                   '-F', pcp_metric]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout:
                # Parse CSV output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 2:  # Header + data
                    values = []
                    for line in lines[2:]:  # Skip header lines
                        parts = line.split(',')
                        if len(parts) > 1:
                            try:
                                val = float(parts[-1])
                                values.append(val)
                            except (ValueError, IndexError):
                                continue

                    if values:
                        metrics[f'{metric_name}_mean'] = np.mean(values)
                        metrics[f'{metric_name}_median'] = np.median(values)
                        metrics[f'{metric_name}_max'] = np.max(values)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    # Convert CPU idle to utilization
    if 'cpu_util_mean' in metrics:
        metrics['cpu_util_mean'] = 100.0 - metrics['cpu_util_mean']

    return metrics

def main():
    results_dir = Path('results')

    print("=" * 80)
    print("COMPREHENSIVE KV-CACHE MANAGEMENT ANALYSIS")
    print("=" * 80)
    print()
    print("Extracting metrics from 224 benchmark runs...")
    print()

    # Extract all metrics
    all_metrics = []

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue

        guidellm_metrics = extract_guidellm_metrics(result_dir)
        if guidellm_metrics:
            # Also extract PCP metrics
            pcp_metrics = extract_pcp_metrics(result_dir)
            guidellm_metrics.update(pcp_metrics)
            all_metrics.append(guidellm_metrics)

    print(f"Extracted metrics from {len(all_metrics)} runs")

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Save complete dataset
    df.to_csv('analysis/complete_metrics.csv', index=False)
    print(f"Saved complete metrics to analysis/complete_metrics.csv")
    print()

    # Find peak throughput for each model-scenario
    # Filter out rows with NaN throughput values first
    df_valid = df[df['output_tps_mean'].notna()].copy()

    peak_results = df_valid.loc[df_valid.groupby(['model', 'scenario'])['output_tps_mean'].idxmax()]
    peak_results.to_csv('analysis/peak_throughput_all.csv', index=False)

    # ANALYSIS AREA 1: llm-d EPP KV-block indexing overhead
    print("=" * 80)
    print("AREA 1: llm-d EPP KV-BLOCK INDEXING OVERHEAD")
    print("=" * 80)
    print()
    print("Comparing baseline (no-offload) vs llm-d-redis vs llm-d-valkey")
    print("This measures the cost of distributed KV-block INDEXING for request routing")
    print()

    indexing_scenarios = ['no-offload', 'llm-d-redis', 'llm-d-valkey']
    indexing_data = peak_results[peak_results['scenario'].isin(indexing_scenarios)].copy()

    for model in MODELS:
        print(f"\n{model}:")
        print("-" * 60)
        model_data = indexing_data[indexing_data['model'] == model]
        baseline = model_data[model_data['scenario'] == 'no-offload']

        if len(baseline) == 0:
            continue

        baseline_tps = baseline['output_tps_mean'].values[0]

        for _, row in model_data.iterrows():
            scenario = row['scenario']
            tps = row['output_tps_mean']
            rate = row['rate']
            delta_pct = ((tps - baseline_tps) / baseline_tps) * 100

            print(f"  {scenario:15s}: {tps:8.1f} tok/s @ rate={rate:3d}  ({delta_pct:+6.2f}%)")

    # ANALYSIS AREA 2: KV-cache offloading strategies
    print()
    print("=" * 80)
    print("AREA 2: KV-CACHE OFFLOADING STRATEGIES")
    print("=" * 80)
    print()
    print("Comparing baseline vs CPU offload vs distributed KV-cache sharing")
    print()

    offload_scenarios = ['no-offload', 'native-offload', 'lmcache-local',
                         'lmcache-redis', 'lmcache-valkey']
    offload_data = peak_results[peak_results['scenario'].isin(offload_scenarios)].copy()

    for model in MODELS:
        print(f"\n{model}:")
        print("-" * 60)
        model_data = offload_data[offload_data['model'] == model]
        baseline = model_data[model_data['scenario'] == 'no-offload']

        if len(baseline) == 0:
            continue

        baseline_tps = baseline['output_tps_mean'].values[0]

        for _, row in model_data.iterrows():
            scenario = row['scenario']
            tps = row['output_tps_mean']
            rate = row['rate']
            delta_pct = ((tps - baseline_tps) / baseline_tps) * 100

            print(f"  {scenario:15s}: {tps:8.1f} tok/s @ rate={rate:3d}  ({delta_pct:+6.2f}%)")

    print()
    print("=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    print()

    # Create visualizations
    create_visualizations(df, peak_results)

    print()
    print("Analysis complete! Generated files:")
    print("  - analysis/complete_metrics.csv")
    print("  - analysis/peak_throughput_all.csv")
    print("  - analysis/*.png (visualizations)")

def create_visualizations(df, peak_df):
    """Create comprehensive visualizations."""

    # 1. Peak throughput comparison - all scenarios
    fig, ax = plt.subplots(figsize=(14, 8))

    pivot = peak_df.pivot(index='model', columns='scenario', values='output_tps_mean')
    # Reorder rows (models) and columns (scenarios)
    pivot = pivot.reindex(MODEL_ORDER)
    pivot = pivot[SCENARIOS]

    pivot.plot(kind='bar', ax=ax)
    ax.set_title('Peak Throughput Comparison Across All Scenarios', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Output Tokens/Second', fontsize=12)
    ax.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('analysis/peak_throughput_all_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Performance delta heatmap vs baseline
    fig, ax = plt.subplots(figsize=(12, 6))

    # Exclude 'no-offload' from heatmap (all zeros, comparing baseline to itself)
    scenarios_for_heatmap = [s for s in SCENARIOS if s != 'no-offload']

    delta_data = []
    for model in MODEL_ORDER:
        model_peak = peak_df[peak_df['model'] == model]
        baseline = model_peak[model_peak['scenario'] == 'no-offload']

        if len(baseline) == 0:
            continue

        baseline_tps = baseline['output_tps_mean'].values[0]

        row = []
        for scenario in scenarios_for_heatmap:
            scenario_data = model_peak[model_peak['scenario'] == scenario]
            if len(scenario_data) > 0:
                tps = scenario_data['output_tps_mean'].values[0]
                delta_pct = ((tps - baseline_tps) / baseline_tps) * 100
                row.append(delta_pct)
            else:
                row.append(np.nan)
        delta_data.append(row)

    delta_df = pd.DataFrame(delta_data, index=MODEL_ORDER, columns=scenarios_for_heatmap)

    sns.heatmap(delta_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': '% Change vs Baseline'},
                linewidths=0.5, ax=ax)
    ax.set_title('Performance Delta vs Baseline (no-offload)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig('analysis/performance_delta_heatmap_all.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Throughput vs concurrency curves for each model
    for model in MODEL_ORDER:
        fig, ax = plt.subplots(figsize=(14, 8))

        model_data = df[df['model'] == model]

        for scenario in SCENARIOS:
            scenario_data = model_data[model_data['scenario'] == scenario]
            if len(scenario_data) > 0:
                scenario_data = scenario_data.sort_values('rate')
                ax.plot(scenario_data['rate'], scenario_data['output_tps_mean'],
                       marker='o', label=scenario, linewidth=2)

        ax.set_title(f'{model} - Throughput vs Concurrency', fontsize=14, fontweight='bold')
        ax.set_xlabel('Concurrency (Rate)', fontsize=12)
        ax.set_ylabel('Output Tokens/Second', fontsize=12)
        ax.legend(title='Scenario')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'analysis/throughput_curve_{model}_all.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Latency comparison - 4-panel: peak throughput + fixed rate=50
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Top row: Latency at peak throughput (variable concurrency)
    # TTFT at peak
    ttft_pivot = peak_df.pivot(index='model', columns='scenario', values='ttft_mean_ms')
    ttft_pivot = ttft_pivot.reindex(MODEL_ORDER)
    ttft_pivot = ttft_pivot[SCENARIOS]
    ttft_pivot.plot(kind='bar', ax=ax1, legend=False)
    ax1.set_title('TTFT at Peak Throughput (variable concurrency)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=10)
    ax1.set_ylabel('TTFT (milliseconds)', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=0)

    # TPOT at peak
    tpot_pivot = peak_df.pivot(index='model', columns='scenario', values='tpot_mean_ms')
    tpot_pivot = tpot_pivot.reindex(MODEL_ORDER)
    tpot_pivot = tpot_pivot[SCENARIOS]
    tpot_pivot.plot(kind='bar', ax=ax2)
    ax2.set_title('TPOT at Peak Throughput (variable concurrency)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=10)
    ax2.set_ylabel('TPOT (milliseconds)', fontsize=10)
    ax2.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=0)

    # Bottom row: Latency at rate=50 (fixed concurrency for comparison)
    rate50_df = complete_df[complete_df['rate'] == 50].copy()

    # TTFT at rate=50
    ttft_rate50_pivot = rate50_df.pivot(index='model', columns='scenario', values='ttft_mean_ms')
    ttft_rate50_pivot = ttft_rate50_pivot.reindex(MODEL_ORDER)
    ttft_rate50_pivot = ttft_rate50_pivot[SCENARIOS]
    ttft_rate50_pivot.plot(kind='bar', ax=ax3, legend=False)
    ax3.set_title('TTFT at Rate=50 (fixed concurrency)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Model', fontsize=10)
    ax3.set_ylabel('TTFT (milliseconds)', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=0)

    # TPOT at rate=50
    tpot_rate50_pivot = rate50_df.pivot(index='model', columns='scenario', values='tpot_mean_ms')
    tpot_rate50_pivot = tpot_rate50_pivot.reindex(MODEL_ORDER)
    tpot_rate50_pivot = tpot_rate50_pivot[SCENARIOS]
    tpot_rate50_pivot.plot(kind='bar', ax=ax4, legend=False)
    ax4.set_title('TPOT at Rate=50 (fixed concurrency)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Model', fontsize=10)
    ax4.set_ylabel('TPOT (milliseconds)', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('analysis/latency_comparison_all.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. CPU vs GPU utilization comparison (if PCP metrics available)
    if 'cpu_util_mean' in peak_df.columns:
        cpu_data = peak_df[peak_df['cpu_util_mean'].notna()]
    else:
        cpu_data = []

    if len(cpu_data) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        cpu_pivot = peak_df.pivot(index='model', columns='scenario', values='cpu_util_mean')
        cpu_pivot = cpu_pivot.reindex(MODEL_ORDER)
        cpu_pivot = cpu_pivot[[s for s in SCENARIOS if s in cpu_pivot.columns]]
        cpu_pivot.plot(kind='bar', ax=ax1)
        ax1.set_title('CPU Utilization at Peak Throughput', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Model', fontsize=10)
        ax1.set_ylabel('CPU Utilization (%)', fontsize=10)
        ax1.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=0)

        gpu_pivot = peak_df.pivot(index='model', columns='scenario', values='gpu_util_mean')
        gpu_pivot = gpu_pivot.reindex(MODEL_ORDER)
        gpu_pivot = gpu_pivot[[s for s in SCENARIOS if s in gpu_pivot.columns]]
        gpu_pivot.plot(kind='bar', ax=ax2)
        ax2.set_title('GPU Utilization at Peak Throughput', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Model', fontsize=10)
        ax2.set_ylabel('GPU Utilization (%)', fontsize=10)
        ax2.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.savefig('analysis/cpu_gpu_utilization_all.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
