#!/usr/bin/env python3
"""Analyze v0.5.0 benchmark results and compare with v0.4.0 baseline."""

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

        benchmark = benchmarks[0]

        # GuideLLM v0.5.3 structure
        metrics = benchmark.get('metrics', {})
        output_tokens_data = metrics.get('output_tokens_per_second', {})
        successful = output_tokens_data.get('successful', {})
        output_tokens_mean = successful.get('mean', 0)

        # Get request metrics
        request_totals = metrics.get('request_totals', {})
        completed_requests = request_totals.get('successful', 0)

        return {
            'hardware': hardware,
            'software': software,
            'model': model,
            'configuration': config,
            'replicas': replicas,
            'rate': rate,
            'output_tokens_per_second': output_tokens_mean,
            'completed_requests': completed_requests,
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

    # Organize by model and configuration
    by_model_config = defaultdict(lambda: defaultdict(list))
    for result in v050_results:
        key = (result['model'], result['configuration'])
        by_model_config[key][result['rate']].append(result['output_tokens_per_second'])

    # Find peak throughput for each model+config
    print("=" * 80)
    print("v0.5.0 (vLLM 0.14.1) Peak Throughput Results")
    print("=" * 80)
    print()

    peak_results = {}
    for (model, config), rates in sorted(by_model_config.items()):
        max_throughput = 0
        max_rate = 0
        for rate, throughputs in sorted(rates.items()):
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput > max_throughput:
                max_throughput = avg_throughput
                max_rate = rate

        peak_results[(model, config)] = {
            'throughput': max_throughput,
            'rate': max_rate
        }

        print(f"{model:20s} {config:25s} Peak: {max_throughput:7.1f} tok/s @ rate={max_rate}")

    print()
    print("=" * 80)
    print("Comparison with v0.4.0 Baseline")
    print("=" * 80)
    print()

    # v0.4.0 baseline results (from REPORT.md)
    v040_baseline = {
        ('Qwen3-0.6B', 'no-offload'): 120.9,
        ('Qwen3-0.6B', 'native-offload-10k'): 86.0,  # -28.9%
        ('Qwen3-8B', 'no-offload'): 90.1,
        ('Qwen3-8B', 'native-offload-10k'): 64.1,     # -28.9%
        ('Qwen3-14B', 'no-offload'): 59.0,
        ('Qwen3-14B', 'native-offload-10k'): 59.0,    # +0.0% (actually beneficial per report)
    }

    # Map config names
    config_map = {
        'no-offload': 'no-offload',
        'native-offload-10k': 'native-offload-10k',
    }

    print(f"{'Model':<20s} {'Config':<25s} {'v0.4.0':>10s} {'v0.5.0':>10s} {'Delta':>10s}")
    print("-" * 80)

    for (model, config), v050_data in sorted(peak_results.items()):
        if config not in config_map:
            continue

        v040_key = (model, config_map[config])
        if v040_key in v040_baseline:
            v040_throughput = v040_baseline[v040_key]
            v050_throughput = v050_data['throughput']
            delta = ((v050_throughput - v040_throughput) / v040_throughput) * 100

            print(f"{model:<20s} {config:<25s} {v040_throughput:9.1f}  {v050_throughput:9.1f}  {delta:+9.1f}%")

if __name__ == '__main__':
    main()
