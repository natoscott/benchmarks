#!/usr/bin/env python3
"""
Comprehensive analysis of all benchmark results across 7 scenarios and 4 models.
Extracts peak throughput, latency metrics, and performance comparisons.
"""

import json
import zstandard as zstd
import pandas as pd
from pathlib import Path
import re

# Define scenarios and models
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

def decompress_zstd(file_path):
    """Decompress zstd file and return JSON data."""
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(f.read())
        return json.loads(decompressed)

def extract_benchmark_metrics(result_dir):
    """Extract key metrics from a single benchmark result directory."""
    # Parse directory name: 1x2xL40S_upstream-llm-d-0.4.0_Qwen3-0.6B_no-offload_replica1_rate50
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

    # Extract metrics from guidellm output
    # guidellm results structure: list of benchmark objects
    if not data or len(data) == 0:
        return None

    benchmark = data[0]

    # Extract throughput metrics (output tokens per second)
    output_throughput_mean = benchmark.get('output_throughput', {}).get('mean')
    output_throughput_median = benchmark.get('output_throughput', {}).get('median')

    # Extract latency metrics
    ttft_mean = benchmark.get('time_to_first_token', {}).get('mean')
    ttft_median = benchmark.get('time_to_first_token', {}).get('median')
    tpot_mean = benchmark.get('inter_token_latency', {}).get('mean')
    tpot_median = benchmark.get('inter_token_latency', {}).get('median')

    # Extract request counts
    completed_requests = benchmark.get('completed_request_count', 0)

    return {
        'model': model,
        'scenario': scenario,
        'rate': rate,
        'output_throughput_mean': output_throughput_mean,
        'output_throughput_median': output_throughput_median,
        'ttft_mean': ttft_mean,
        'ttft_median': ttft_median,
        'tpot_mean': tpot_mean,
        'tpot_median': tpot_median,
        'completed_requests': completed_requests
    }

def main():
    results_dir = Path('results')

    # Extract metrics from all benchmark results
    all_metrics = []

    for result_dir in results_dir.iterdir():
        if not result_dir.is_dir():
            continue

        metrics = extract_benchmark_metrics(result_dir)
        if metrics:
            all_metrics.append(metrics)

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)

    # Find peak throughput for each model-scenario combination
    peak_results = df.loc[df.groupby(['model', 'scenario'])['output_throughput_mean'].idxmax()]

    print("=" * 80)
    print("PEAK THROUGHPUT ANALYSIS")
    print("=" * 80)
    print()

    # Print comprehensive peak throughput table
    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"{'='*80}")
        print()
        print(f"{'Scenario':<20} {'Peak Throughput (tok/s)':>25} {'Optimal Rate':>15} {'vs Baseline':>15}")
        print("-" * 80)

        model_data = peak_results[peak_results['model'] == model].sort_values('scenario')
        baseline_throughput = model_data[model_data['scenario'] == 'no-offload']['output_throughput_mean'].values

        if len(baseline_throughput) > 0:
            baseline_throughput = baseline_throughput[0]
        else:
            baseline_throughput = None

        for _, row in model_data.iterrows():
            scenario = row['scenario']
            throughput = row['output_throughput_mean']
            rate = row['rate']

            if baseline_throughput:
                delta_pct = ((throughput - baseline_throughput) / baseline_throughput) * 100
                delta_str = f"{delta_pct:+.1f}%"
            else:
                delta_str = "—"

            print(f"{scenario:<20} {throughput:>20,.1f} {rate:>15} {delta_str:>15}")

    print("\n")
    print("=" * 80)
    print("LATENCY COMPARISON AT PEAK THROUGHPUT")
    print("=" * 80)
    print()

    # TTFT comparison
    print("\nTime to First Token (TTFT) - milliseconds")
    print("-" * 80)
    print(f"{'Model':<15}", end='')
    for scenario in SCENARIOS:
        print(f"{scenario:>15}", end='')
    print()
    print("-" * 80)

    for model in MODELS:
        print(f"{model:<15}", end='')
        model_data = peak_results[peak_results['model'] == model]

        for scenario in SCENARIOS:
            scenario_data = model_data[model_data['scenario'] == scenario]
            if len(scenario_data) > 0:
                ttft = scenario_data['ttft_mean'].values[0]
                if ttft:
                    print(f"{ttft*1000:>14.0f}ms", end='')
                else:
                    print(f"{'N/A':>15}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        print()

    # TPOT comparison
    print("\nTime Per Output Token (TPOT) - milliseconds")
    print("-" * 80)
    print(f"{'Model':<15}", end='')
    for scenario in SCENARIOS:
        print(f"{scenario:>15}", end='')
    print()
    print("-" * 80)

    for model in MODELS:
        print(f"{model:<15}", end='')
        model_data = peak_results[peak_results['model'] == model]

        for scenario in SCENARIOS:
            scenario_data = model_data[model_data['scenario'] == scenario]
            if len(scenario_data) > 0:
                tpot = scenario_data['tpot_mean'].values[0]
                if tpot:
                    print(f"{tpot*1000:>14.1f}ms", end='')
                else:
                    print(f"{'N/A':>15}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        print()

    # Save detailed CSV for further analysis
    peak_results.to_csv('analysis/peak_throughput_all_scenarios.csv', index=False)
    df.to_csv('analysis/all_benchmarks_detailed.csv', index=False)

    print("\n")
    print("=" * 80)
    print("Detailed CSV files saved to analysis/ directory:")
    print("  - peak_throughput_all_scenarios.csv")
    print("  - all_benchmarks_detailed.csv")
    print("=" * 80)

if __name__ == '__main__':
    main()
