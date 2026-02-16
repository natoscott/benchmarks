#!/usr/bin/env python3
"""
Analyze benchmark results from guidellm JSON files.
Extract key performance metrics for comparison across configurations.
"""

import json
import zstandard as zstd
from pathlib import Path
import pandas as pd
import numpy as np

def load_guidellm_results(json_path):
    """Load and decompress guidellm results JSON."""
    if json_path.suffix == '.zst':
        with open(json_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                data = json.loads(reader.read())
    else:
        with open(json_path, 'r') as fh:
            data = json.load(fh)
    return data

def extract_benchmark_metrics(data):
    """Extract key metrics from guidellm benchmark data."""
    results = []

    # Iterate through all benchmark reports
    for report in data.get('reports', []):
        for benchmark in report.get('benchmarks', []):
            # Get configuration info
            config_name = benchmark.get('name', 'unknown')
            mode = benchmark.get('mode', 'unknown')

            # Extract performance metrics
            metrics = benchmark.get('metrics', {})

            # Request-level metrics
            req_metrics = metrics.get('request', {})

            # Token-level metrics
            token_metrics = metrics.get('token', {})

            result = {
                'config': config_name,
                'mode': mode,
                # Request metrics
                'completed_requests': req_metrics.get('completed', 0),
                'failed_requests': req_metrics.get('errors', 0),
                'request_latency_mean': req_metrics.get('latency_mean'),
                'request_latency_p50': req_metrics.get('latency_p50'),
                'request_latency_p95': req_metrics.get('latency_p95'),
                'request_latency_p99': req_metrics.get('latency_p99'),
                'ttft_mean': req_metrics.get('time_to_first_token_mean'),
                'ttft_p50': req_metrics.get('time_to_first_token_p50'),
                'ttft_p95': req_metrics.get('time_to_first_token_p95'),
                'ttft_p99': req_metrics.get('time_to_first_token_p99'),
                'tpot_mean': req_metrics.get('time_per_output_token_mean'),
                'tpot_p50': req_metrics.get('time_per_output_token_p50'),
                'tpot_p95': req_metrics.get('time_per_output_token_p95'),
                'tpot_p99': req_metrics.get('time_per_output_token_p99'),
                # Token metrics
                'output_throughput': token_metrics.get('throughput_output'),
                'total_throughput': token_metrics.get('throughput_total'),
                'input_tokens': token_metrics.get('count_input'),
                'output_tokens': token_metrics.get('count_output'),
            }

            results.append(result)

    return pd.DataFrame(results)

def analyze_results(results_dir):
    """Analyze all benchmark results in the directory."""
    results_path = Path(results_dir)

    all_data = {}

    # Process each result directory
    for result_dir in sorted(results_path.glob('1x2xL40S_upstream-llm-d-0.4.0_Qwen3-*')):
        # Extract model and config from directory name
        parts = result_dir.name.split('_')
        model = parts[3]  # Qwen3-0.6B, Qwen3-8B, Qwen3-14B
        config = parts[4] if len(parts) > 4 else 'unknown'  # no-offload, native-offload, etc.

        # Skip if not no-offload or native-offload
        if config not in ['no-offload', 'native-offload']:
            continue

        # Find guidellm results file
        json_files = list(result_dir.glob('guidellm-results.json*'))
        if not json_files:
            print(f"Warning: No guidellm results found in {result_dir}")
            continue

        json_path = json_files[0]
        print(f"Processing: {model} / {config}")

        # Load and extract metrics
        data = load_guidellm_results(json_path)
        df = extract_benchmark_metrics(data)

        # Store in dictionary
        key = f"{model}_{config}"
        all_data[key] = df

        # Print summary statistics
        if not df.empty:
            print(f"  Configurations found: {df['config'].unique().tolist()}")
            print(f"  Total requests: {df['completed_requests'].sum():.0f}")
            if df['output_throughput'].notna().any():
                print(f"  Throughput range: {df['output_throughput'].min():.2f} - {df['output_throughput'].max():.2f} tok/s")

    return all_data

def compare_configurations(all_data):
    """Compare performance metrics across configurations."""
    comparisons = []

    models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B']

    for model in models:
        no_offload_key = f"{model}_no-offload"
        native_offload_key = f"{model}_native-offload"

        if no_offload_key not in all_data or native_offload_key not in all_data:
            print(f"Warning: Missing data for {model}")
            continue

        no_off = all_data[no_offload_key]
        native_off = all_data[native_offload_key]

        # Get peak throughput for each config
        no_off_peak = no_off['output_throughput'].max()
        native_off_peak = native_off['output_throughput'].max()

        # Get metrics at peak throughput
        no_off_peak_row = no_off.loc[no_off['output_throughput'].idxmax()]
        native_off_peak_row = native_off.loc[native_off['output_throughput'].idxmax()]

        comparison = {
            'model': model,
            'no_offload_throughput': no_off_peak,
            'native_offload_throughput': native_off_peak,
            'throughput_delta_pct': ((native_off_peak - no_off_peak) / no_off_peak * 100),
            'no_offload_ttft_mean': no_off_peak_row['ttft_mean'],
            'native_offload_ttft_mean': native_off_peak_row['ttft_mean'],
            'ttft_delta_pct': ((native_off_peak_row['ttft_mean'] - no_off_peak_row['ttft_mean']) / no_off_peak_row['ttft_mean'] * 100),
            'no_offload_tpot_mean': no_off_peak_row['tpot_mean'],
            'native_offload_tpot_mean': native_off_peak_row['tpot_mean'],
            'tpot_delta_pct': ((native_off_peak_row['tpot_mean'] - no_off_peak_row['tpot_mean']) / no_off_peak_row['tpot_mean'] * 100),
            'no_offload_concurrency': no_off_peak_row['config'],
            'native_offload_concurrency': native_off_peak_row['config'],
        }

        comparisons.append(comparison)

    return pd.DataFrame(comparisons)

def main():
    results_dir = Path('/home/nathans/git/benchmarks/llm-d-kvcache-offload-eval/results')

    print("=" * 80)
    print("Analyzing Benchmark Results")
    print("=" * 80)
    print()

    # Analyze all results
    all_data = analyze_results(results_dir)

    print()
    print("=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print()

    # Compare configurations
    comparison_df = compare_configurations(all_data)

    # Display comparison
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    print(comparison_df.to_string(index=False))

    # Save detailed results
    output_file = results_dir / 'performance_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    print()
    print(f"Saved comparison to: {output_file}")

    # Save all extracted data
    for key, df in all_data.items():
        output_file = results_dir / f'{key}_metrics.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved detailed metrics to: {output_file}")

if __name__ == '__main__':
    main()
