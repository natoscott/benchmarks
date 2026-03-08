#!/usr/bin/env python3
"""Comprehensive v0.5.0 analysis showing all LLM metrics (TTFT, ITL, TPOT, throughput)."""

import json
import subprocess
from pathlib import Path
from collections import defaultdict

def extract_guidellm_metrics(result_file):
    """Extract comprehensive LLM metrics from a GuideLLM result file."""
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
        # Silently skip files with errors (JSON corruption from kubectl EOF issues)
        return None

def main():
    results_dir = Path('results')

    # Collect all v0.5.0 results
    v050_results = []
    for result_file in sorted(results_dir.glob('1x2xL40S_upstream-llm-d-0.5.0_*/guidellm-results.json.zst')):
        metrics = extract_guidellm_metrics(result_file)
        if metrics:
            v050_results.append(metrics)

    print(f"Successfully loaded {len(v050_results)} benchmark results")
    print()

    # Organize by model, config, and rate for detailed analysis
    by_model_config_rate = defaultdict(lambda: defaultdict(list))
    for result in v050_results:
        key = (result['model'], result['configuration'])
        by_model_config_rate[key][result['rate']].append(result)

    # Analyze each model
    for model in sorted(['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']):
        print("=" * 120)
        print(f"Model: {model}")
        print("=" * 120)
        print()

        # Get configs for this model
        configs = [c for c in ['no-offload', 'native-offload-10k', 'native-offload-20k']
                   if (model, c) in by_model_config_rate]

        if not configs:
            print(f"No data found for {model}")
            print()
            continue

        # For each configuration, show metrics at peak rate
        for config in configs:
            key = (model, config)
            if key not in by_model_config_rate:
                continue

            # Find the rate with highest throughput
            peak_rate = None
            peak_throughput = 0
            peak_metrics = None

            for rate, results_list in sorted(by_model_config_rate[key].items()):
                # Average across replicas if multiple
                avg_throughput = sum(r['throughput'] for r in results_list) / len(results_list)
                if avg_throughput > peak_throughput:
                    peak_throughput = avg_throughput
                    peak_rate = rate
                    # Use first result's metrics (or could average across replicas)
                    peak_metrics = results_list[0]

            if peak_metrics:
                print(f"Configuration: {config}")
                print(f"  Peak throughput: {peak_metrics['throughput']:.1f} tok/s @ rate={peak_rate}")
                print(f"  Request rate:    {peak_metrics['request_rate_mean']:.2f} req/s")
                print(f"  Output tokens/s: {peak_metrics['output_tokens_per_sec_mean']:.1f} tok/s")
                print(f"  TTFT median:     {peak_metrics['ttft_median_ms']:.1f} ms")
                print(f"  TTFT p95:        {peak_metrics['ttft_p95_ms']:.1f} ms")
                print(f"  ITL median:      {peak_metrics['itl_median_ms']:.1f} ms")
                print(f"  ITL p95:         {peak_metrics['itl_p95_ms']:.1f} ms")
                print(f"  TPOT median:     {peak_metrics['tpot_median_ms']:.1f} ms")
                print(f"  TPOT p95:        {peak_metrics['tpot_p95_ms']:.1f} ms")
                print()

        # Calculate offload impact for this model
        no_offload_key = (model, 'no-offload')
        offload_10k_key = (model, 'native-offload-10k')

        if no_offload_key in by_model_config_rate and offload_10k_key in by_model_config_rate:
            # Find peak throughput for each config
            no_offload_peak = max(
                sum(r['throughput'] for r in results_list) / len(results_list)
                for results_list in by_model_config_rate[no_offload_key].values()
            )
            offload_10k_peak = max(
                sum(r['throughput'] for r in results_list) / len(results_list)
                for results_list in by_model_config_rate[offload_10k_key].values()
            )

            impact_pct = ((offload_10k_peak - no_offload_peak) / no_offload_peak) * 100

            print(f"Native KV-cache CPU Offload Impact:")
            print(f"  Baseline (no-offload):     {no_offload_peak:.1f} tok/s")
            print(f"  With offload (10k):        {offload_10k_peak:.1f} tok/s")
            print(f"  Impact:                    {impact_pct:+.1f}%")
            print()

    print()
    print("=" * 120)
    print("Summary: v0.5.0 Native KV-cache CPU Offload vs v0.4.0 LMCache")
    print("=" * 120)
    print()

    # v0.4.0 degradation percentages (from REPORT.md)
    v040_degradation = {
        'Qwen3-0.6B': -29.1,
        'Qwen3-8B': -36.5,
        'Qwen3-14B': +0.6,
    }

    print(f"{'Model':<15s} {'v0.4.0 Impact':>15s} {'v0.5.0 Impact':>15s} {'Improvement':>15s}")
    print("-" * 65)

    for model in sorted(['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B']):
        no_offload_key = (model, 'no-offload')
        offload_10k_key = (model, 'native-offload-10k')

        if no_offload_key in by_model_config_rate and offload_10k_key in by_model_config_rate:
            # Calculate v0.5.0 impact
            no_offload_peak = max(
                sum(r['throughput'] for r in results_list) / len(results_list)
                for results_list in by_model_config_rate[no_offload_key].values()
            )
            offload_10k_peak = max(
                sum(r['throughput'] for r in results_list) / len(results_list)
                for results_list in by_model_config_rate[offload_10k_key].values()
            )

            v050_impact = ((offload_10k_peak - no_offload_peak) / no_offload_peak) * 100
            v040_impact = v040_degradation.get(model, 0)
            improvement = v050_impact - v040_impact

            print(f"{model:<15s} {v040_impact:14.1f}% {v050_impact:14.1f}% {improvement:+14.1f}%")

    print()
    print("NOTE: Positive improvement means v0.5.0 native offload performs better than v0.4.0 LMCache")
    print("      Parameters are now apples-to-apples: PREFIX_COUNT = rate × 2 for both versions")
    print()

if __name__ == '__main__':
    main()
