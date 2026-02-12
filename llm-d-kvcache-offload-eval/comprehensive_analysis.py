#!/usr/bin/env python3
"""
Comprehensive analysis of vLLM KV-cache offload benchmarks.
Combines guidellm results with PCP metrics using pandas.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
from datetime import datetime

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_guidellm_data(result_dir):
    """Load guidellm JSON results into pandas DataFrame."""
    json_file = result_dir / "guidellm-results.json"
    config_file = result_dir / "benchmark-config.txt"

    with open(json_file) as f:
        data = json.load(f)

    # Parse config file for metadata
    config = {}
    with open(config_file) as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                config[key.strip()] = value.strip()

    # Extract benchmark data
    benchmarks_data = []

    for bench in data.get('benchmarks', []):
        concurrency = bench['config'].get('streams', bench['config'].get('rate', 0))

        # Calculate metrics from requests
        requests = bench.get('requests', [])

        # Filter to actual request objects (not strings)
        if not requests or len(requests) == 0:
            continue

        if isinstance(requests, list) and len(requests) > 0 and isinstance(requests[0], str):
            continue

        # Get completed requests
        completed = [r for r in requests if isinstance(r, dict)]

        if not completed:
            continue

        # Calculate latencies
        latencies = []
        ttfts = []
        for req in completed:
            if 'start_time' in req and 'end_time' in req:
                latencies.append(req['end_time'] - req['start_time'])
            if 'output' in req and 'first_token_time' in req['output']:
                ttfts.append(req['output']['first_token_time'] * 1000)  # Convert to ms

        # Calculate throughput
        duration = bench.get('duration', 30)
        throughput = len(completed) / duration if duration > 0 else 0

        bench_data = {
            'config': config.get('Parameters', 'unknown'),
            'model': config.get('Model Name', 'unknown'),
            'concurrency': concurrency,
            'duration_sec': duration,
            'completed_requests': len(completed),
            'throughput_req_per_sec': throughput,
            'latency_median_sec': np.median(latencies) if latencies else np.nan,
            'latency_p95_sec': np.percentile(latencies, 95) if latencies else np.nan,
            'latency_mean_sec': np.mean(latencies) if latencies else np.nan,
            'ttft_median_ms': np.median(ttfts) if ttfts else np.nan,
            'ttft_p95_ms': np.percentile(ttfts, 95) if ttfts else np.nan,
            'start_time': datetime.fromisoformat(bench['start_time'].replace('Z', '+00:00')),
            'end_time': datetime.fromisoformat(bench['end_time'].replace('Z', '+00:00')),
        }

        benchmarks_data.append(bench_data)

    return pd.DataFrame(benchmarks_data)


def convert_pcp_to_parquet(result_dir):
    """Convert PCP archives to parquet using pcp2arrow."""
    pcp_dir = result_dir / "pcp-archives"
    output_dir = result_dir / "pcp-data.parquet"

    if output_dir.exists():
        print(f"  Parquet already exists: {output_dir}")
        return output_dir

    # Find all PCP archive base files (not .zst, not .index, not .meta)
    archive_files = []
    for node_dir in pcp_dir.iterdir():
        if node_dir.is_dir():
            # Find the archive base names
            for f in node_dir.glob("*.zst"):
                fname = f.name.replace('.zst', '')
                # Skip .index and .meta files
                if not fname.endswith('.index') and not fname.endswith('.meta'):
                    # Check if this looks like a base archive file (YYYYMMDD.HH.MM)
                    if fname.count('.') == 3:  # e.g., 20260212.07.17.0
                        archive_files.append(str(f.parent / fname))

    if not archive_files:
        print(f"  No PCP archives found in {pcp_dir}")
        return None

    # Use first archive for now (they should all be from same benchmark run)
    archive = archive_files[0]
    print(f"  Converting PCP archive: {archive}")

    try:
        # pcp2arrow command
        cmd = f"pcp2arrow -a {archive} -O {output_dir}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print(f"  Created parquet: {output_dir}")
            return output_dir
        else:
            print(f"  pcp2arrow failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"  Error running pcp2arrow: {e}")
        return None


def load_pcp_metrics(parquet_dir, start_time, end_time):
    """Load PCP metrics from parquet files."""
    if not parquet_dir or not parquet_dir.exists():
        return None

    try:
        # Load parquet data
        df = pd.read_parquet(parquet_dir)

        # Filter to benchmark time range
        if 'timestamp' in df.columns:
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

        return df
    except Exception as e:
        print(f"  Error loading parquet: {e}")
        return None


def analyze_configuration(config_name, result_dir):
    """Analyze a single configuration."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {config_name}")
    print(f"{'='*80}")

    # Load guidellm data
    print("Loading guidellm data...")
    df_guidellm = load_guidellm_data(result_dir)

    if df_guidellm.empty:
        print("  No guidellm data found")
        return None

    print(f"  Found {len(df_guidellm)} benchmark runs")

    # Convert PCP archives to parquet
    print("Converting PCP archives...")
    parquet_dir = convert_pcp_to_parquet(result_dir)

    # Add PCP metrics if available
    if parquet_dir:
        print("Loading PCP metrics...")
        # For now, we'll just note they're available
        # Full integration would require matching timestamps
        df_guidellm['pcp_data_available'] = True
    else:
        df_guidellm['pcp_data_available'] = False

    return df_guidellm


def main():
    """Main analysis workflow."""
    base_dir = Path("results")

    # Find result directories
    configs = {
        'no-offload': base_dir / "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-0.6B_no-offload",
        'native-offload': base_dir / "1x2xL40S_upstream-llm-d-0.4.0_Qwen3-0.6B_native-offload",
    }

    # Load all configurations
    all_data = {}
    for config_name, result_dir in configs.items():
        if result_dir.exists():
            df = analyze_configuration(config_name, result_dir)
            if df is not None:
                all_data[config_name] = df

    if not all_data:
        print("\nNo data loaded!")
        return

    # Combine data
    print(f"\n{'='*80}")
    print("Combining Results")
    print(f"{'='*80}")

    for config_name, df in all_data.items():
        df['configuration'] = config_name

    df_combined = pd.concat(all_data.values(), ignore_index=True)

    # Save combined data
    output_file = "combined_benchmark_results.csv"
    df_combined.to_csv(output_file, index=False)
    print(f"\nSaved combined results to: {output_file}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")

    summary = df_combined.groupby(['configuration', 'concurrency']).agg({
        'throughput_req_per_sec': 'mean',
        'latency_median_sec': 'mean',
        'ttft_median_ms': 'mean',
    }).round(3)

    print("\n", summary)

    # Create visualization
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('vLLM KV-Cache Offload Benchmark Comparison', fontsize=16, fontweight='bold')

    # Throughput comparison
    ax = axes[0, 0]
    for config in df_combined['configuration'].unique():
        data = df_combined[df_combined['configuration'] == config]
        ax.plot(data['concurrency'], data['throughput_req_per_sec'], marker='o', label=config, linewidth=2)
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Throughput (requests/sec)')
    ax.set_title('Throughput vs Concurrency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Latency comparison
    ax = axes[0, 1]
    for config in df_combined['configuration'].unique():
        data = df_combined[df_combined['configuration'] == config]
        ax.plot(data['concurrency'], data['latency_median_sec'], marker='s', label=config, linewidth=2)
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Median Latency (seconds)')
    ax.set_title('Latency vs Concurrency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TTFT comparison
    ax = axes[1, 0]
    for config in df_combined['configuration'].unique():
        data = df_combined[df_combined['configuration'] == config]
        ax.plot(data['concurrency'], data['ttft_median_ms'], marker='^', label=config, linewidth=2)
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Median TTFT (milliseconds)')
    ax.set_title('Time to First Token vs Concurrency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Performance difference
    ax = axes[1, 1]
    pivot_throughput = df_combined.pivot(index='concurrency', columns='configuration', values='throughput_req_per_sec')
    if 'no-offload' in pivot_throughput.columns and 'native-offload' in pivot_throughput.columns:
        pct_diff = ((pivot_throughput['native-offload'] - pivot_throughput['no-offload']) /
                    pivot_throughput['no-offload'] * 100)
        ax.bar(pct_diff.index, pct_diff.values, color=['red' if x < 0 else 'green' for x in pct_diff.values])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Concurrency Level')
        ax.set_ylabel('Throughput Change (%)')
        ax.set_title('Native Offload vs No Offload Performance')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_plot = "benchmark_comparison.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_plot}")

    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    print(f"\nOutputs:")
    print(f"  - Combined data: {output_file}")
    print(f"  - Visualization: {output_plot}")


if __name__ == '__main__':
    main()
