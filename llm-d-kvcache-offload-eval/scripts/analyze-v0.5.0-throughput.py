#!/usr/bin/env python3
"""Analyze v0.5.0 throughput results and compare to v0.4.0 baseline."""

import os
import json
import subprocess
import csv
from pathlib import Path
import pandas as pd

def extract_guidellm_metrics(result_file):
    """Extract key metrics from a GuideLLM result file."""
    try:
        # Decompress and read result file
        result = subprocess.run(
            ['zstdcat', result_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        data = json.loads(result.stdout)

        # Extract directory name to parse configuration
        # Format: 1x2xL40S_upstream-llm-d-0.5.0_Qwen3-14B_native-offload-10k_replica1_rate50
        dir_name = result_file.parent.name
        parts = dir_name.split('_')

        hardware = parts[0]
        software = parts[1]
        model = parts[2]
        config = parts[3]
        replicas = int(parts[4].replace('replica', ''))
        rate = int(parts[5].replace('rate', ''))

        # Extract throughput metrics from the benchmark results
        benchmarks = data.get('benchmarks', [])
        if not benchmarks:
            return None

        benchmark = benchmarks[0]  # Should only be one benchmark per file

        # Get server throughput statistics
        server_throughput = benchmark.get('server_throughput', {})
        output_tokens_mean = server_throughput.get('output_token_throughput_per_second_mean', 0)

        # Get request counts
        completed_requests = benchmark.get('completed_request_count', 0)

        # Get latency statistics
        latency = benchmark.get('latency', {})
        ttft_median = latency.get('time_to_first_token_median_ms', 0) / 1000  # Convert to seconds
        tpot_median = latency.get('time_per_output_token_median_ms', 0) / 1000  # Convert to seconds

        # Get request latency
        request_latency_median = latency.get('request_median_seconds', 0)

        return {
            'hardware': hardware,
            'software': software,
            'model': model,
            'configuration': config,
            'replicas': replicas,
            'rate': rate,
            'output_tokens_per_second': output_tokens_mean,
            'completed_requests': completed_requests,
            'ttft_median_s': ttft_median,
            'tpot_median_s': tpot_median,
            'request_latency_median_s': request_latency_median,
        }

    except Exception as e:
        print(f"Error processing {result_file}: {e}")
        return None

def main():
    # Find all v0.5.0 GuideLLM result files
    results_dir = Path('results')
    result_files = sorted(results_dir.glob('1x2xL40S_upstream-llm-d-0.5.0_*/guidellm-results.json.zst'))

    print(f"Found {len(result_files)} v0.5.0 GuideLLM result files")

    # Extract metrics from all result files
    all_metrics = []
    for result_file in result_files:
        metrics = extract_guidellm_metrics(result_file)
        if metrics:
            all_metrics.append(metrics)

    # Write to CSV
    output_file = 'analysis/v0.5.0_throughput_results.csv'
    os.makedirs('analysis', exist_ok=True)

    df = pd.DataFrame(all_metrics)
    df.to_csv(output_file, index=False)

    print(f"\nWrote {len(all_metrics)} entries to {output_file}")

    # Calculate peak throughput for each model/configuration
    print("\n=== Peak Throughput by Model and Configuration ===")
    peak_results = df.loc[df.groupby(['model', 'configuration'])['output_tokens_per_second'].idxmax()]
    peak_results = peak_results.sort_values(['model', 'configuration'])

    print("\nModel | Configuration | Peak Throughput (tok/s) | Optimal Rate | Completed Requests")
    print("-" * 90)
    for _, row in peak_results.iterrows():
        print(f"{row['model']:15} | {row['configuration']:20} | {row['output_tokens_per_second']:23.1f} | "
              f"{row['rate']:12} | {row['completed_requests']:18}")

    # Compare to v0.4.0 baseline (no-offload and native-offload results from REPORT.md)
    # v0.4.0 baseline peak throughput (from REPORT.md):
    v04_baseline = {
        ('Qwen3-0.6B', 'no-offload'): 602.0,
        ('Qwen3-0.6B', 'native-offload'): 426.8,  # 10k blocks
        ('Qwen3-8B', 'no-offload'): 113.0,
        ('Qwen3-8B', 'native-offload'): 71.8,  # 10k blocks
        ('Qwen3-14B', 'no-offload'): 58.7,
        ('Qwen3-14B', 'native-offload'): 59.0,  # 10k blocks
        ('Qwen3-14B', 'native-offload-20k'): 68.5,  # 20k blocks
    }

    print("\n=== v0.5.0 vs v0.4.0 Comparison ===")
    print("\nModel | Configuration | v0.4.0 (tok/s) | v0.5.0 (tok/s) | Delta | Delta %")
    print("-" * 95)

    for _, row in peak_results.iterrows():
        model = row['model']
        config = row['configuration']
        v05_throughput = row['output_tokens_per_second']

        # Map v0.5.0 config names to v0.4.0 equivalents
        if config == 'native-offload-10k':
            v04_config = 'native-offload'
        elif config == 'native-offload-20k':
            v04_config = 'native-offload-20k'
        else:
            v04_config = config

        key = (model, v04_config)
        if key in v04_baseline:
            v04_throughput = v04_baseline[key]
            delta = v05_throughput - v04_throughput
            delta_pct = (delta / v04_throughput) * 100

            print(f"{model:15} | {config:20} | {v04_throughput:14.1f} | {v05_throughput:14.1f} | "
                  f"{delta:+6.1f} | {delta_pct:+7.1f}%")

    # Save comparison to CSV
    comparison_file = 'analysis/v0.5.0_vs_v0.4.0_comparison.csv'
    comparison_data = []

    for _, row in peak_results.iterrows():
        model = row['model']
        config = row['configuration']
        v05_throughput = row['output_tokens_per_second']

        if config == 'native-offload-10k':
            v04_config = 'native-offload'
        elif config == 'native-offload-20k':
            v04_config = 'native-offload-20k'
        else:
            v04_config = config

        key = (model, v04_config)
        if key in v04_baseline:
            v04_throughput = v04_baseline[key]
            delta = v05_throughput - v04_throughput
            delta_pct = (delta / v04_throughput) * 100

            comparison_data.append({
                'model': model,
                'configuration': config,
                'v04_throughput_tok_s': v04_throughput,
                'v05_throughput_tok_s': v05_throughput,
                'delta_tok_s': delta,
                'delta_percent': delta_pct,
                'v05_optimal_rate': row['rate'],
            })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\nWrote comparison to {comparison_file}")

if __name__ == '__main__':
    main()
