#!/usr/bin/env python3
"""
Analyze GPU power consumption patterns across KV-cache offload scenarios.
Focuses on 14B model where CPU offload showed performance improvements.
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

SCENARIOS = [
    'no-offload',
    'native-offload',
    'native-offload-20kcpu',
    'lmcache-local',
    'lmcache-local-20kcpu',
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

# Key metrics to extract
METRICS = {
    'gpu_power': 'openmetrics.dcgm.DCGM_FI_DEV_POWER_USAGE',
    'gpu_energy': 'openmetrics.dcgm.DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION',
    'gpu_util': 'openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL',
    'kv_cache_usage': 'openmetrics.vllm.gpu_cache_usage_perc',
    'prefix_cache_hit_rate': 'openmetrics.vllm.cache_config_prefix_cache_hit_rate',
}

def decompress_archive(archive_dir):
    """Decompress zstd-compressed PCP archives."""
    zst_files = list(archive_dir.glob('*.zst'))
    if not zst_files:
        return None

    # Decompress all .zst files
    for zst_file in zst_files:
        base_name = str(zst_file).replace('.zst', '')
        if not Path(base_name).exists():
            subprocess.run(['zstd', '-d', '-f', str(zst_file)],
                         stderr=subprocess.DEVNULL, check=False)

    # Find the base archive path (without extension)
    meta_files = list(archive_dir.glob('*.meta'))
    if meta_files:
        return str(meta_files[0]).replace('.meta', '')
    return None

def extract_metric(archive_path, metric_name):
    """Extract a metric from PCP archive using pmrep."""
    try:
        cmd = ['pmrep', '-a', archive_path, '-t', '10s', '-o', 'csv',
               '-F', metric_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0 or not result.stdout:
            return []

        # Parse CSV output
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 2:  # Just headers
            return []

        values = []
        for line in lines[2:]:  # Skip header lines
            parts = line.split(',')
            if len(parts) > 1:
                try:
                    # Handle multiple instances (e.g., multiple GPUs)
                    # Sum across all instances for total power/energy
                    val_sum = 0
                    for part in parts[1:]:
                        if part.strip():
                            val_sum += float(part)
                    values.append(val_sum)
                except (ValueError, IndexError):
                    continue

        return values
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

def analyze_result_dir(result_dir):
    """Extract power and performance metrics from a result directory."""
    # Parse directory name
    dir_name = result_dir.name
    parts = dir_name.split('_')

    model = None
    scenario = None
    rate = None

    for part in parts:
        if part in MODELS:
            model = part
        # Handle scenario names (including compound ones like lmcache-local-20kcpu)
        for s in SCENARIOS:
            if s in dir_name:
                scenario = s
                break
        if part.startswith('rate'):
            rate = int(part[4:])

    if not all([model, scenario, rate]):
        return None

    # Find PCP archive
    pcp_dir = result_dir / 'pcp-archives'
    if not pcp_dir.exists():
        return None

    archive_dirs = [d for d in pcp_dir.iterdir() if d.is_dir()]
    if not archive_dirs:
        return None

    archive_path = decompress_archive(archive_dirs[0])
    if not archive_path:
        return None

    # Extract metrics
    metrics_data = {
        'model': model,
        'scenario': scenario,
        'rate': rate,
    }

    for metric_key, metric_name in METRICS.items():
        values = extract_metric(archive_path, metric_name)
        if values:
            metrics_data[f'{metric_key}_mean'] = np.mean(values)
            metrics_data[f'{metric_key}_median'] = np.median(values)
            metrics_data[f'{metric_key}_max'] = np.max(values)
            metrics_data[f'{metric_key}_min'] = np.min(values)
            metrics_data[f'{metric_key}_std'] = np.std(values)

            # For energy, also calculate total consumption (delta)
            if metric_key == 'gpu_energy' and len(values) > 1:
                metrics_data['gpu_energy_total_kwh'] = (values[-1] - values[0]) / 1000.0  # Convert to kWh

    return metrics_data

def main():
    results_dir = Path('results')

    print("=" * 80)
    print("GPU POWER CONSUMPTION ANALYSIS")
    print("=" * 80)
    print()
    print("Extracting power metrics from PCP archives...")
    print("Target metrics:")
    print("  - GPU power usage (Watts)")
    print("  - GPU total energy consumption (Joules)")
    print("  - GPU utilization (%)")
    print("  - KV-cache usage (%)")
    print("  - Prefix cache hit rate (%)")
    print()

    all_metrics = []

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue

        print(f"Processing: {result_dir.name}")
        metrics = analyze_result_dir(result_dir)
        if metrics:
            all_metrics.append(metrics)

    print()
    print(f"Extracted power metrics from {len(all_metrics)} runs")

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Save complete dataset
    output_file = 'analysis/gpu_power_metrics.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved power metrics to {output_file}")
    print()

    # Focus on peak throughput (rate=50 for most models, rate=1 for 32B-AWQ)
    # Create visualizations
    create_visualizations(df)

    # Print summary for 14B model (where offload showed benefits)
    print("=" * 80)
    print("14B MODEL POWER ANALYSIS")
    print("=" * 80)
    print()

    df_14b = df[df['model'] == 'Qwen3-14B']
    if len(df_14b) > 0:
        # Group by scenario and show power metrics at peak throughput
        df_14b_rate50 = df_14b[df_14b['rate'] == 50]

        if len(df_14b_rate50) > 0:
            print("Power metrics at rate=50 (peak throughput for most scenarios):")
            print()
            print(f"{'Scenario':<25} {'Avg Power (W)':<15} {'GPU Util (%)':<15} {'KV-Cache (%)':<15}")
            print("-" * 70)

            for _, row in df_14b_rate50.iterrows():
                scenario = row['scenario']
                power = row.get('gpu_power_mean', np.nan)
                util = row.get('gpu_util_mean', np.nan)
                kv = row.get('kv_cache_usage_mean', np.nan)

                print(f"{scenario:<25} {power:<15.1f} {util:<15.1f} {kv:<15.1f}")

            print()

            # Calculate power savings
            baseline = df_14b_rate50[df_14b_rate50['scenario'] == 'no-offload']
            if len(baseline) > 0:
                baseline_power = baseline['gpu_power_mean'].values[0]

                print("Power reduction vs baseline (no-offload):")
                print()
                print(f"{'Scenario':<25} {'Power Delta (W)':<20} {'Power Savings (%)':<20}")
                print("-" * 65)

                for _, row in df_14b_rate50.iterrows():
                    scenario = row['scenario']
                    power = row.get('gpu_power_mean', np.nan)

                    if scenario != 'no-offload' and not np.isnan(power):
                        delta = power - baseline_power
                        pct = (delta / baseline_power) * 100
                        print(f"{scenario:<25} {delta:<+20.1f} {pct:<+20.1f}")

    print()
    print("=" * 80)
    print("Visualizations saved to analysis/ directory")
    print("=" * 80)

def create_visualizations(df):
    """Create power consumption visualizations."""

    # Filter to peak throughput scenarios (rate=50)
    df_rate50 = df[df['rate'] == 50].copy()

    if len(df_rate50) == 0:
        print("Warning: No rate=50 data found")
        return

    # 1. GPU Power vs Throughput correlation
    # (Need to merge with guidellm data for throughput)

    # 2. Power consumption by scenario (4-panel, one per model)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    axes = [ax1, ax2, ax3, ax4]

    for idx, (model, ax) in enumerate(zip(MODELS, axes)):
        model_data = df_rate50[df_rate50['model'] == model]

        if len(model_data) == 0:
            continue

        # Sort by scenario order
        model_data['scenario_cat'] = pd.Categorical(
            model_data['scenario'],
            categories=[s for s in SCENARIOS if s in model_data['scenario'].values],
            ordered=True
        )
        model_data = model_data.sort_values('scenario_cat')

        x_pos = np.arange(len(model_data))

        # Bar chart with GPU power
        bars = ax.bar(x_pos, model_data['gpu_power_mean'],
                      color='steelblue', alpha=0.7, label='GPU Power')

        # Add error bars (std dev)
        if 'gpu_power_std' in model_data.columns:
            ax.errorbar(x_pos, model_data['gpu_power_mean'],
                       yerr=model_data['gpu_power_std'],
                       fmt='none', ecolor='black', capsize=3, alpha=0.5)

        ax.set_title(f'{model} - GPU Power Consumption', fontsize=12, fontweight='bold')
        ax.set_xlabel('Scenario', fontsize=10)
        ax.set_ylabel('Average Power (Watts)', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_data['scenario'], rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Add baseline reference line for no-offload
        baseline = model_data[model_data['scenario'] == 'no-offload']
        if len(baseline) > 0:
            baseline_power = baseline['gpu_power_mean'].values[0]
            ax.axhline(baseline_power, color='red', linestyle='--',
                      linewidth=1, alpha=0.5, label='Baseline')
            ax.legend(fontsize=8)

    plt.suptitle('GPU Power Consumption at Peak Throughput (Rate=50)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('analysis/gpu_power_by_scenario.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Power vs GPU Utilization scatter
    fig, ax = plt.subplots(figsize=(12, 8))

    for model in MODELS:
        model_data = df_rate50[df_rate50['model'] == model]
        if len(model_data) > 0 and 'gpu_util_mean' in model_data.columns:
            ax.scatter(model_data['gpu_util_mean'], model_data['gpu_power_mean'],
                      label=model, s=100, alpha=0.6)

    ax.set_title('GPU Power Consumption vs GPU Utilization', fontsize=14, fontweight='bold')
    ax.set_xlabel('GPU Utilization (%)', fontsize=12)
    ax.set_ylabel('Average Power (Watts)', fontsize=12)
    ax.legend(title='Model', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/gpu_power_vs_utilization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 14B Model detailed analysis (power, util, kv-cache)
    df_14b = df_rate50[df_rate50['model'] == 'Qwen3-14B']

    if len(df_14b) > 0:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        df_14b['scenario_cat'] = pd.Categorical(
            df_14b['scenario'],
            categories=[s for s in SCENARIOS if s in df_14b['scenario'].values],
            ordered=True
        )
        df_14b = df_14b.sort_values('scenario_cat')

        x_pos = np.arange(len(df_14b))

        # GPU Power
        ax1.bar(x_pos, df_14b['gpu_power_mean'], color='steelblue', alpha=0.7)
        ax1.set_title('GPU Power Consumption', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Average Power (Watts)', fontsize=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df_14b['scenario'], rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)

        # GPU Utilization
        if 'gpu_util_mean' in df_14b.columns:
            ax2.bar(x_pos, df_14b['gpu_util_mean'], color='coral', alpha=0.7)
            ax2.set_title('GPU Utilization', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Utilization (%)', fontsize=10)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(df_14b['scenario'], rotation=45, ha='right', fontsize=8)
            ax2.grid(axis='y', alpha=0.3)

        # KV-Cache Usage
        if 'kv_cache_usage_mean' in df_14b.columns:
            ax3.bar(x_pos, df_14b['kv_cache_usage_mean'], color='seagreen', alpha=0.7)
            ax3.set_title('KV-Cache Usage', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Cache Usage (%)', fontsize=10)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(df_14b['scenario'], rotation=45, ha='right', fontsize=8)
            ax3.grid(axis='y', alpha=0.3)

        plt.suptitle('Qwen3-14B: Power, Utilization, and KV-Cache at Rate=50',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('analysis/gpu_power_14b_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
