#!/usr/bin/env python3
"""Analyze v0.5.0 results comparing no-offload vs native-offload within v0.5.0."""

import json
import subprocess
from pathlib import Path
from collections import defaultdict

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

        model = parts[2]
        config = parts[3]
        rate = int(parts[5].replace('rate', ''))

        # Extract metrics from the benchmark results
        benchmark = data['benchmarks'][0]
        metrics = benchmark['metrics']

        # Calculate aggregate throughput
        total_output_tokens = metrics['output_token_count']['successful']['total_sum']
        duration = benchmark['duration']
        aggregate_throughput = total_output_tokens / duration

        # Get request metrics
        request_totals = metrics['request_totals']
        completed_requests = request_totals.get('completed', 0)

        return {
            'model': model,
            'configuration': config,
            'rate': rate,
            'throughput': aggregate_throughput,
            'completed_requests': completed_requests,
            'request_rate_mean': metrics['requests_per_second']['successful']['mean'],
            'output_tokens_per_sec_mean': metrics['output_tokens_per_second']['successful']['mean'],
            'ttft_median_ms': metrics['time_to_first_token_ms']['successful']['median'],
            'ttft_p95_ms': metrics['time_to_first_token_ms']['successful']['percentiles']['p95'],
            'itl_median_ms': metrics['inter_token_latency_ms']['successful']['median'],
            'itl_p95_ms': metrics['inter_token_latency_ms']['successful']['percentiles']['p95'],
            'tpot_median_ms': metrics['time_per_output_token_ms']['successful']['median'],
            'tpot_p95_ms': metrics['time_per_output_token_ms']['successful']['percentiles']['p95'],
        }
    except Exception as e:
        print(f"Error processing {result_file}: {e}")
        return None

def main():
    results_dir = Path('results')

    # Collect all v0.5.0 results
    v050_results = []
    for result_file in sorted(results_dir.glob('1x2xL40S_upstream-llm-d-0.5.0_*/guidellm-results.json.zst')):
        metrics = extract_guidellm_metrics(result_file)
        if metrics:
            v050_results.append(metrics)

    # Organize by model and rate
    by_model_rate = defaultdict(lambda: defaultdict(list))
    for result in v050_results:
        key = (result['model'], result['rate'])
        by_model_rate[key][result['configuration']].append(result['throughput'])

    # Find peak throughput for each model+config
    print("=" * 80)
    print("v0.5.0 (vLLM 0.14.1) Performance Analysis")
    print("=" * 80)
    print()
    print("NOTE: Corrected parameters - PREFIX_COUNT = rate × 2 (matching v0.4.0)")
    print("Now we can make apples-to-apples comparison between v0.4.0 and v0.5.0!")
    print()

    # Find peak for each model/config combination
    peak_results = defaultdict(lambda: defaultdict(dict))
    for (model, rate), configs in sorted(by_model_rate.items()):
        for config, throughputs in configs.items():
            avg_throughput = sum(throughputs) / len(throughputs)
            if not peak_results[model][config] or avg_throughput > peak_results[model][config]['throughput']:
                peak_results[model][config] = {
                    'throughput': avg_throughput,
                    'rate': rate
                }

    # Print peak results and comparison
    print(f"{'Model':<15s} {'Configuration':<25s} {'Peak (tok/s)':>12s} {'@ Rate':>8s}")
    print("-" * 80)

    for model in sorted(peak_results.keys()):
        configs = peak_results[model]

        # Get baseline (no-offload)
        baseline_throughput = configs.get('no-offload', {}).get('throughput', 0)
        baseline_rate = configs.get('no-offload', {}).get('rate', 0)

        if baseline_throughput > 0:
            print(f"{model:<15s} {'no-offload':<25s} {baseline_throughput:12.1f} {baseline_rate:8d}")

            # Print native-offload with comparison
            if 'native-offload-10k' in configs:
                offload_throughput = configs['native-offload-10k']['throughput']
                offload_rate = configs['native-offload-10k']['rate']
                delta = ((offload_throughput - baseline_throughput) / baseline_throughput) * 100
                print(f"{model:<15s} {'native-offload-10k':<25s} {offload_throughput:12.1f} {offload_rate:8d}  ({delta:+.1f}%)")
            print()

    print()
    print("=" * 80)
    print("Comparison with v0.4.0 Relative Performance")
    print("=" * 80)
    print()

    # v0.4.0 degradation percentages (from REPORT.md)
    v040_degradation = {
        'Qwen3-0.6B': -29.1,
        'Qwen3-8B': -36.5,
        'Qwen3-14B': +0.6,
    }

    print(f"{'Model':<15s} {'v0.4.0 Impact':>15s} {'v0.5.0 Impact':>15s} {'Change':>15s}")
    print("-" * 80)

    for model in sorted(['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B']):
        if model in peak_results and 'no-offload' in peak_results[model] and 'native-offload-10k' in peak_results[model]:
            baseline = peak_results[model]['no-offload']['throughput']
            offload = peak_results[model]['native-offload-10k']['throughput']
            v050_impact = ((offload - baseline) / baseline) * 100

            v040_impact = v040_degradation.get(model, 0)
            change = v050_impact - v040_impact

            print(f"{model:<15s} {v040_impact:14.1f}% {v050_impact:14.1f}% {change:+14.1f}%")

if __name__ == '__main__':
    main()
