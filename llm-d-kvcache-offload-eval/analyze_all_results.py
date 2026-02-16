#!/usr/bin/env python3
"""
Comprehensive analysis of all benchmark configurations.
Extracts and compares performance metrics across:
- no-offload (baseline)
- native-offload (vLLM OffloadingConnector)
- llm-d-redis (llm-d EPP with Redis)
- llm-d-valkey (llm-d EPP with Valkey)
"""

import json
import zstandard as zstd
from pathlib import Path
import pandas as pd
import numpy as np

CONFIGS = ['no-offload', 'native-offload', 'llm-d-redis', 'llm-d-valkey']
MODELS = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']

def load_guidellm_results(json_path):
    """Load and decompress guidellm results JSON."""
    print(f"  Loading: {json_path.name}")
    try:
        with open(json_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                data = json.loads(reader.read())
        return data
    except Exception as e:
        print(f"  ERROR loading {json_path}: {e}")
        return None

def extract_peak_metrics(data):
    """Extract peak performance metrics from guidellm benchmark data."""
    if not data or 'benchmarks' not in data:
        return None

    benchmarks = data['benchmarks']
    if not benchmarks:
        return None

    # Find benchmark with highest output throughput
    max_throughput = 0
    peak_benchmark = None

    for bm in benchmarks:
        m = bm.get('metrics', {})
        throughput_data = m.get('output_tokens_per_second', {}).get('total', {})
        throughput = throughput_data.get('mean', 0)

        if throughput > max_throughput:
            max_throughput = throughput
            peak_benchmark = bm

    if not peak_benchmark:
        return None

    # Extract metrics from the peak benchmark
    m = peak_benchmark.get('metrics', {})
    config = peak_benchmark.get('config', {})
    strategy = config.get('strategy', {})

    # Helper function to safely extract metric values
    def get_metric(metric_name, field='mean'):
        return m.get(metric_name, {}).get('total', {}).get(field, 0)

    # Extract key metrics
    metrics = {
        'concurrency': strategy.get('streams', 0),
        'completed_requests': m.get('request_totals', {}).get('successful', 0),
        'failed_requests': m.get('request_totals', {}).get('errored', 0),

        # Throughput metrics (tokens/sec)
        'output_throughput': get_metric('output_tokens_per_second', 'mean'),
        'total_throughput': get_metric('tokens_per_second', 'mean'),
        'request_throughput': get_metric('requests_per_second', 'mean'),

        # Latency metrics (already in ms)
        'ttft_mean': get_metric('time_to_first_token_ms', 'mean'),
        'ttft_p50': get_metric('time_to_first_token_ms', 'median'),
        'ttft_p95': get_metric('time_to_first_token_ms', 'percentiles').get('p95', 0) if isinstance(get_metric('time_to_first_token_ms', 'percentiles'), dict) else 0,
        'ttft_p99': get_metric('time_to_first_token_ms', 'percentiles').get('p99', 0) if isinstance(get_metric('time_to_first_token_ms', 'percentiles'), dict) else 0,

        'tpot_mean': get_metric('time_per_output_token_ms', 'mean'),
        'tpot_p50': get_metric('time_per_output_token_ms', 'median'),
        'tpot_p95': get_metric('time_per_output_token_ms', 'percentiles').get('p95', 0) if isinstance(get_metric('time_per_output_token_ms', 'percentiles'), dict) else 0,
        'tpot_p99': get_metric('time_per_output_token_ms', 'percentiles').get('p99', 0) if isinstance(get_metric('time_per_output_token_ms', 'percentiles'), dict) else 0,

        'itl_mean': get_metric('inter_token_latency_ms', 'mean'),
        'itl_p50': get_metric('inter_token_latency_ms', 'median'),
        'itl_p95': get_metric('inter_token_latency_ms', 'percentiles').get('p95', 0) if isinstance(get_metric('inter_token_latency_ms', 'percentiles'), dict) else 0,

        'request_latency_mean': get_metric('request_latency', 'mean'),
        'request_latency_p50': get_metric('request_latency', 'median'),
        'request_latency_p95': get_metric('request_latency', 'percentiles').get('p95', 0) if isinstance(get_metric('request_latency', 'percentiles'), dict) else 0,

        # Token counts
        'input_tokens': get_metric('prompt_token_count', 'mean'),
        'output_tokens': get_metric('output_token_count', 'mean'),
    }

    return metrics

def analyze_all_results():
    """Analyze all benchmark results."""
    results_dir = Path('results')
    all_metrics = []

    print("=" * 80)
    print("Extracting Metrics from All Benchmarks")
    print("=" * 80)
    print()

    for model in MODELS:
        print(f"Model: {model}")
        print("-" * 80)

        for config in CONFIGS:
            result_dir = results_dir / f'1x2xL40S_upstream-llm-d-0.4.0_{model}_{config}'

            if not result_dir.exists():
                print(f"  {config}: NOT FOUND")
                continue

            json_file = result_dir / 'guidellm-results.json.zst'
            if not json_file.exists():
                print(f"  {config}: No guidellm results")
                continue

            data = load_guidellm_results(json_file)
            if not data:
                continue

            metrics = extract_peak_metrics(data)
            if metrics:
                metrics['model'] = model
                metrics['config'] = config
                all_metrics.append(metrics)

                print(f"  {config}: ✓ Peak throughput: {metrics['output_throughput']:.1f} tok/s @ concurrency={metrics['concurrency']}")
            else:
                print(f"  {config}: ERROR - No metrics extracted")

        print()

    return pd.DataFrame(all_metrics)

def calculate_comparisons(df):
    """Calculate performance comparisons against baseline (no-offload)."""
    comparisons = []

    for model in MODELS:
        model_data = df[df['model'] == model]

        baseline = model_data[model_data['config'] == 'no-offload']
        if baseline.empty:
            print(f"Warning: No baseline data for {model}")
            continue

        baseline_row = baseline.iloc[0]

        for config in ['native-offload', 'llm-d-redis', 'llm-d-valkey']:
            config_data = model_data[model_data['config'] == config]
            if config_data.empty:
                continue

            config_row = config_data.iloc[0]

            comparison = {
                'model': model,
                'config': config,

                # Throughput comparison
                'baseline_throughput': baseline_row['output_throughput'],
                'config_throughput': config_row['output_throughput'],
                'throughput_delta': config_row['output_throughput'] - baseline_row['output_throughput'],
                'throughput_delta_pct': ((config_row['output_throughput'] - baseline_row['output_throughput']) / baseline_row['output_throughput'] * 100),

                # TTFT comparison
                'baseline_ttft': baseline_row['ttft_mean'],
                'config_ttft': config_row['ttft_mean'],
                'ttft_delta': config_row['ttft_mean'] - baseline_row['ttft_mean'],
                'ttft_delta_pct': ((config_row['ttft_mean'] - baseline_row['ttft_mean']) / baseline_row['ttft_mean'] * 100),

                # TPOT comparison
                'baseline_tpot': baseline_row['tpot_mean'],
                'config_tpot': config_row['tpot_mean'],
                'tpot_delta': config_row['tpot_mean'] - baseline_row['tpot_mean'],
                'tpot_delta_pct': ((config_row['tpot_mean'] - baseline_row['tpot_mean']) / baseline_row['tpot_mean'] * 100),

                # Concurrency
                'baseline_concurrency': baseline_row['concurrency'],
                'config_concurrency': config_row['concurrency'],
            }

            comparisons.append(comparison)

    return pd.DataFrame(comparisons)

def main():
    print("=" * 80)
    print("Comprehensive Benchmark Analysis")
    print("=" * 80)
    print()
    print(f"Configurations: {', '.join(CONFIGS)}")
    print(f"Models: {', '.join(MODELS)}")
    print()

    # Extract all metrics
    df = analyze_all_results()

    if df.empty:
        print("ERROR: No data extracted")
        return

    print("=" * 80)
    print("Peak Performance Summary")
    print("=" * 80)
    print()

    # Display summary
    summary = df[['model', 'config', 'output_throughput', 'ttft_mean', 'tpot_mean', 'concurrency']]
    summary = summary.sort_values(['model', 'output_throughput'], ascending=[True, False])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    print(summary.to_string(index=False))
    print()

    # Calculate comparisons
    print("=" * 80)
    print("Performance vs Baseline (no-offload)")
    print("=" * 80)
    print()

    comp_df = calculate_comparisons(df)

    if not comp_df.empty:
        comp_summary = comp_df[['model', 'config', 'throughput_delta_pct', 'ttft_delta_pct', 'tpot_delta_pct']]
        print(comp_summary.to_string(index=False))
        print()

    # Save results
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / 'all_peak_metrics.csv', index=False)
    print(f"✓ Saved: analysis/all_peak_metrics.csv")

    if not comp_df.empty:
        comp_df.to_csv(output_dir / 'performance_comparisons.csv', index=False)
        print(f"✓ Saved: analysis/performance_comparisons.csv")

    # Save detailed per-config results
    for model in MODELS:
        for config in CONFIGS:
            model_config = df[(df['model'] == model) & (df['config'] == config)]
            if not model_config.empty:
                filename = output_dir / f'{model}_{config}_metrics.csv'
                model_config.to_csv(filename, index=False)
                print(f"✓ Saved: {filename}")

    print()
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
