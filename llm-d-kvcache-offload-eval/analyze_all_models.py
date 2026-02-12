#!/usr/bin/env python3
"""
Comprehensive analysis of vLLM KV-cache offload benchmarks for all models
Compares GPU-only vs CPU-offload configurations across Qwen3-0.6B, Qwen3-8B, and Qwen3-14B
"""

import re
import sys

# Parse benchmark results from log file
def parse_benchmark_log(log_file):
    """Extract metrics from benchmark log for all models"""
    with open(log_file) as f:
        content = f.read()

    # Split by benchmark iterations
    iterations = re.split(r'\*{40}\nBenchmark Iteration:', content)[1:]

    results = {}
    for iteration in iterations:
        config_match = re.search(r'([^\n]+)\nModel: ([^\n]+)\nvLLM Args: ([^\n]*)', iteration)
        if not config_match:
            continue

        config_name = config_match.group(1).strip()
        model = config_match.group(2).strip()
        vllm_args = config_match.group(3).strip()

        # Determine configuration type
        if 'no-offload' in config_name:
            config_type = 'no-offload'
        elif 'native-offload' in config_name:
            config_type = 'native-offload'
        else:
            continue

        # Extract latency statistics
        latency_section = re.search(
            r'Request Latency Statistics.*?\n\|=+\|(.*?)\n\|=+\|',
            iteration, re.DOTALL
        )

        latencies = []
        if latency_section:
            rows = re.findall(
                r'\| concurrent \| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\|',
                latency_section.group(1)
            )
            for row in rows:
                latencies.append({
                    'req_latency_median': float(row[0]),
                    'req_latency_p95': float(row[1]),
                    'ttft_median': float(row[2]),
                    'ttft_p95': float(row[3]),
                    'itl_median': float(row[4]),
                    'itl_p95': float(row[5]),
                    'tpot_median': float(row[6]),
                    'tpot_p95': float(row[7]),
                })

        # Extract throughput statistics
        throughput_section = re.search(
            r'Server Throughput Statistics.*?\n\|=+\|(.*?)\n\|=+\|',
            iteration, re.DOTALL
        )

        throughputs = []
        if throughput_section:
            rows = re.findall(
                r'\| concurrent \| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\| ([\d.]+)\s+\|',
                throughput_section.group(1)
            )
            for row in rows:
                throughputs.append({
                    'req_per_sec_median': float(row[0]),
                    'req_per_sec_mean': float(row[1]),
                    'concurrency_median': float(row[2]),
                    'concurrency_mean': float(row[3]),
                    'input_tok_per_sec_median': float(row[4]),
                    'input_tok_per_sec_mean': float(row[5]),
                    'output_tok_per_sec_median': float(row[6]),
                    'output_tok_per_sec_mean': float(row[7]),
                    'total_tok_per_sec_median': float(row[8]),
                    'total_tok_per_sec_mean': float(row[9]),
                })

        if model not in results:
            results[model] = {}

        results[model][config_type] = {
            'config': config_type,
            'model': model,
            'vllm_args': vllm_args,
            'latencies': latencies,
            'throughputs': throughputs,
        }

    return results


def print_model_comparison(model, model_data):
    """Print comparison for a single model"""
    if 'no-offload' not in model_data or 'native-offload' not in model_data:
        print(f"  Missing data for {model}")
        return

    no_offload = model_data['no-offload']
    native_offload = model_data['native-offload']

    if not no_offload['throughputs'] or not native_offload['throughputs']:
        print(f"  Incomplete data for {model}")
        return

    concurrency_levels = [5, 25, 50, 100, 250]

    print(f"\n{'='*100}")
    print(f"Model: {model}")
    print(f"{'='*100}")

    print(f"\n{'THROUGHPUT COMPARISON (requests/sec)':<100}")
    print(f"{'-'*100}")
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print(f"{'-'*100}")

    for i, conc in enumerate(concurrency_levels):
        if i >= len(no_offload['throughputs']) or i >= len(native_offload['throughputs']):
            break
        no_tput = no_offload['throughputs'][i]['req_per_sec_mean']
        nat_tput = native_offload['throughputs'][i]['req_per_sec_mean']
        diff = nat_tput - no_tput
        pct = (diff / no_tput * 100) if no_tput > 0 else 0

        print(f"{conc:<12} {no_tput:>8.1f} req/s      {nat_tput:>8.1f} req/s      {diff:>+8.1f} req/s      {pct:>+6.1f}%")

    print(f"\n{'LATENCY COMPARISON - Median (seconds)':<100}")
    print(f"{'-'*100}")
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print(f"{'-'*100}")

    for i, conc in enumerate(concurrency_levels):
        if i >= len(no_offload['latencies']) or i >= len(native_offload['latencies']):
            break
        no_lat = no_offload['latencies'][i]['req_latency_median']
        nat_lat = native_offload['latencies'][i]['req_latency_median']
        diff = nat_lat - no_lat
        pct = (diff / no_lat * 100) if no_lat > 0 else 0

        print(f"{conc:<12} {no_lat:>8.2f}s           {nat_lat:>8.2f}s           {diff:>+8.2f}s           {pct:>+6.1f}%")

    print(f"\n{'TIME TO FIRST TOKEN (TTFT) - Median (milliseconds)':<100}")
    print(f"{'-'*100}")
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print(f"{'-'*100}")

    for i, conc in enumerate(concurrency_levels):
        if i >= len(no_offload['latencies']) or i >= len(native_offload['latencies']):
            break
        no_ttft = no_offload['latencies'][i]['ttft_median']
        nat_ttft = native_offload['latencies'][i]['ttft_median']
        diff = nat_ttft - no_ttft
        pct = (diff / no_ttft * 100) if no_ttft > 0 else 0

        print(f"{conc:<12} {no_ttft:>8.1f}ms          {nat_ttft:>8.1f}ms          {diff:>+8.1f}ms          {pct:>+6.1f}%")

    # Calculate averages
    n = len(no_offload['throughputs'])
    avg_tput_diff = sum((native_offload['throughputs'][i]['req_per_sec_mean'] - no_offload['throughputs'][i]['req_per_sec_mean']) / no_offload['throughputs'][i]['req_per_sec_mean'] * 100 for i in range(n)) / n
    avg_lat_diff = sum((native_offload['latencies'][i]['req_latency_median'] - no_offload['latencies'][i]['req_latency_median']) / no_offload['latencies'][i]['req_latency_median'] * 100 for i in range(n)) / n
    avg_ttft_diff = sum((native_offload['latencies'][i]['ttft_median'] - no_offload['latencies'][i]['ttft_median']) / no_offload['latencies'][i]['ttft_median'] * 100 for i in range(n)) / n

    print(f"\n{'KEY METRICS':<100}")
    print(f"{'-'*100}")
    print(f"  Average throughput change: {avg_tput_diff:>+6.1f}%")
    print(f"  Average latency change:    {avg_lat_diff:>+6.1f}%")
    print(f"  Average TTFT change:       {avg_ttft_diff:>+6.1f}%")


def print_summary_comparison(results):
    """Print cross-model summary"""
    print(f"\n\n{'='*100}")
    print(f"CROSS-MODEL SUMMARY - Average Performance Impact of CPU Offload")
    print(f"{'='*100}")
    print(f"\n{'Model':<20} {'Throughput Δ':<18} {'Latency Δ':<18} {'TTFT Δ':<18}")
    print(f"{'-'*100}")

    for model in sorted(results.keys()):
        if 'no-offload' not in results[model] or 'native-offload' not in results[model]:
            continue

        no_offload = results[model]['no-offload']
        native_offload = results[model]['native-offload']

        n = len(no_offload['throughputs'])
        avg_tput_diff = sum((native_offload['throughputs'][i]['req_per_sec_mean'] - no_offload['throughputs'][i]['req_per_sec_mean']) / no_offload['throughputs'][i]['req_per_sec_mean'] * 100 for i in range(n)) / n
        avg_lat_diff = sum((native_offload['latencies'][i]['req_latency_median'] - no_offload['latencies'][i]['req_latency_median']) / no_offload['latencies'][i]['req_latency_median'] * 100 for i in range(n)) / n
        avg_ttft_diff = sum((native_offload['latencies'][i]['ttft_median'] - no_offload['latencies'][i]['ttft_median']) / no_offload['latencies'][i]['ttft_median'] * 100 for i in range(n)) / n

        print(f"{model:<20} {avg_tput_diff:>+6.1f}%           {avg_lat_diff:>+6.1f}%           {avg_ttft_diff:>+6.1f}%")

    print(f"\n{'='*100}")
    print(f"Note: Negative values indicate improvement with CPU offload, positive values indicate degradation")
    print(f"{'='*100}")


if __name__ == '__main__':
    log_file = 'benchmark-run-larger-models.log'
    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    print(f"{'='*100}")
    print(f"vLLM KV-Cache CPU Offload - Multi-Model Benchmark Comparison")
    print(f"{'='*100}")
    print(f"\nAnalyzing: {log_file}")
    print(f"Test Configuration: 30-second runs at concurrency levels [5, 25, 50, 100, 250]")
    print(f"Workload: 128 prompt tokens, 128 output tokens, 10K prefix tokens, 5 turns")

    results = parse_benchmark_log(log_file)

    # Print individual model comparisons
    for model in sorted(results.keys()):
        print_model_comparison(model, results[model])

    # Print cross-model summary
    print_summary_comparison(results)
