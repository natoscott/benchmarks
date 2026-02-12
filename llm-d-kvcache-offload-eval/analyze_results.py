#!/usr/bin/env python3
"""
Comprehensive analysis of vLLM KV-cache offload benchmarks
Compares GPU-only vs CPU-offload configurations
"""

import re

# Parse benchmark results from log file
def parse_benchmark_log(log_file):
    """Extract metrics from benchmark log"""
    with open(log_file) as f:
        content = f.read()

    # Split by benchmark iterations
    iterations = re.split(r'\*{40}\nBenchmark Iteration:', content)[1:]

    results = []
    for iteration in iterations:
        config_match = re.search(r'([^\n]+)\nModel: ([^\n]+)\nvLLM Args: ([^\n]*)', iteration)
        if not config_match:
            continue

        config_name = config_match.group(1).strip()
        model = config_match.group(2).strip()
        vllm_args = config_match.group(3).strip()

        # Determine configuration type
        if 'no-offload' in config_name:
            config_type = 'NO-OFFLOAD (GPU-only)'
        elif 'native-offload' in config_name:
            config_type = 'NATIVE-OFFLOAD (CPU offload)'
        else:
            config_type = config_name

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

        results.append({
            'config': config_type,
            'model': model,
            'vllm_args': vllm_args,
            'latencies': latencies,
            'throughputs': throughputs,
        })

    return results


def print_comparison(results):
    """Print side-by-side comparison"""

    # Group by configuration
    no_offload = [r for r in results if 'NO-OFFLOAD' in r['config']][0]
    native_offload = [r for r in results if 'NATIVE-OFFLOAD' in r['config']][0]

    print("=" * 100)
    print("vLLM KV-Cache CPU Offload Benchmark Comparison")
    print("=" * 100)
    print(f"\nModel: {no_offload['model']}")
    print(f"Test Configuration: 30-second runs at concurrency levels [5, 25, 50, 100, 250]")
    print(f"Workload: 128 prompt tokens, 128 output tokens, 10K prefix tokens, 5 turns\n")

    concurrency_levels = [5, 25, 50, 100, 250]

    print("\n" + "=" * 100)
    print("THROUGHPUT COMPARISON (requests/sec)")
    print("=" * 100)
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print("-" * 100)

    for i, conc in enumerate(concurrency_levels):
        no_tput = no_offload['throughputs'][i]['req_per_sec_mean']
        nat_tput = native_offload['throughputs'][i]['req_per_sec_mean']
        diff = nat_tput - no_tput
        pct = (diff / no_tput * 100) if no_tput > 0 else 0

        print(f"{conc:<12} {no_tput:>8.1f} req/s      {nat_tput:>8.1f} req/s      {diff:>+8.1f} req/s      {pct:>+6.1f}%")

    print("\n" + "=" * 100)
    print("TOKEN THROUGHPUT COMPARISON (output tokens/sec)")
    print("=" * 100)
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print("-" * 100)

    for i, conc in enumerate(concurrency_levels):
        no_tok = no_offload['throughputs'][i]['output_tok_per_sec_mean']
        nat_tok = native_offload['throughputs'][i]['output_tok_per_sec_mean']
        diff = nat_tok - no_tok
        pct = (diff / no_tok * 100) if no_tok > 0 else 0

        print(f"{conc:<12} {no_tok:>8.1f} tok/s      {nat_tok:>8.1f} tok/s      {diff:>+8.1f} tok/s      {pct:>+6.1f}%")

    print("\n" + "=" * 100)
    print("LATENCY COMPARISON - Median (seconds)")
    print("=" * 100)
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print("-" * 100)

    for i, conc in enumerate(concurrency_levels):
        no_lat = no_offload['latencies'][i]['req_latency_median']
        nat_lat = native_offload['latencies'][i]['req_latency_median']
        diff = nat_lat - no_lat
        pct = (diff / no_lat * 100) if no_lat > 0 else 0

        print(f"{conc:<12} {no_lat:>8.2f}s           {nat_lat:>8.2f}s           {diff:>+8.2f}s           {pct:>+6.1f}%")

    print("\n" + "=" * 100)
    print("LATENCY COMPARISON - P95 (seconds)")
    print("=" * 100)
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print("-" * 100)

    for i, conc in enumerate(concurrency_levels):
        no_lat = no_offload['latencies'][i]['req_latency_p95']
        nat_lat = native_offload['latencies'][i]['req_latency_p95']
        diff = nat_lat - no_lat
        pct = (diff / no_lat * 100) if no_lat > 0 else 0

        print(f"{conc:<12} {no_lat:>8.2f}s           {nat_lat:>8.2f}s           {diff:>+8.2f}s           {pct:>+6.1f}%")

    print("\n" + "=" * 100)
    print("TIME TO FIRST TOKEN (TTFT) COMPARISON - Median (milliseconds)")
    print("=" * 100)
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print("-" * 100)

    for i, conc in enumerate(concurrency_levels):
        no_ttft = no_offload['latencies'][i]['ttft_median']
        nat_ttft = native_offload['latencies'][i]['ttft_median']
        diff = nat_ttft - no_ttft
        pct = (diff / no_ttft * 100) if no_ttft > 0 else 0

        print(f"{conc:<12} {no_ttft:>8.1f}ms          {nat_ttft:>8.1f}ms          {diff:>+8.1f}ms          {pct:>+6.1f}%")

    print("\n" + "=" * 100)
    print("TIME TO FIRST TOKEN (TTFT) COMPARISON - P95 (milliseconds)")
    print("=" * 100)
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print("-" * 100)

    for i, conc in enumerate(concurrency_levels):
        no_ttft = no_offload['latencies'][i]['ttft_p95']
        nat_ttft = native_offload['latencies'][i]['ttft_p95']
        diff = nat_ttft - no_ttft
        pct = (diff / no_ttft * 100) if no_ttft > 0 else 0

        print(f"{conc:<12} {no_ttft:>8.1f}ms          {nat_ttft:>8.1f}ms          {diff:>+8.1f}ms          {pct:>+6.1f}%")

    print("\n" + "=" * 100)
    print("INTER-TOKEN LATENCY (ITL) COMPARISON - Median (milliseconds)")
    print("=" * 100)
    print(f"{'Concurrency':<12} {'NO-OFFLOAD':<20} {'NATIVE-OFFLOAD':<20} {'Difference':<20} {'% Change':<15}")
    print("-" * 100)

    for i, conc in enumerate(concurrency_levels):
        no_itl = no_offload['latencies'][i]['itl_median']
        nat_itl = native_offload['latencies'][i]['itl_median']
        diff = nat_itl - no_itl
        pct = (diff / no_itl * 100) if no_itl > 0 else 0

        print(f"{conc:<12} {no_itl:>8.1f}ms          {nat_itl:>8.1f}ms          {diff:>+8.1f}ms          {pct:>+6.1f}%")

    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    # Calculate overall differences
    avg_tput_diff = sum((native_offload['throughputs'][i]['req_per_sec_mean'] - no_offload['throughputs'][i]['req_per_sec_mean']) / no_offload['throughputs'][i]['req_per_sec_mean'] * 100 for i in range(5)) / 5
    avg_lat_diff = sum((native_offload['latencies'][i]['req_latency_median'] - no_offload['latencies'][i]['req_latency_median']) / no_offload['latencies'][i]['req_latency_median'] * 100 for i in range(5)) / 5
    avg_ttft_diff = sum((native_offload['latencies'][i]['ttft_median'] - no_offload['latencies'][i]['ttft_median']) / no_offload['latencies'][i]['ttft_median'] * 100 for i in range(5)) / 5

    print(f"\n1. Throughput Impact:")
    print(f"   - Average change: {avg_tput_diff:+.1f}%")
    print(f"   - At low concurrency (5): {((native_offload['throughputs'][0]['req_per_sec_mean'] - no_offload['throughputs'][0]['req_per_sec_mean']) / no_offload['throughputs'][0]['req_per_sec_mean'] * 100):+.1f}%")
    print(f"   - At high concurrency (250): {((native_offload['throughputs'][4]['req_per_sec_mean'] - no_offload['throughputs'][4]['req_per_sec_mean']) / no_offload['throughputs'][4]['req_per_sec_mean'] * 100):+.1f}%")

    print(f"\n2. Latency Impact:")
    print(f"   - Average median latency change: {avg_lat_diff:+.1f}%")
    print(f"   - At low concurrency (5): {((native_offload['latencies'][0]['req_latency_median'] - no_offload['latencies'][0]['req_latency_median']) / no_offload['latencies'][0]['req_latency_median'] * 100):+.1f}%")
    print(f"   - At high concurrency (250): {((native_offload['latencies'][4]['req_latency_median'] - no_offload['latencies'][4]['req_latency_median']) / no_offload['latencies'][4]['req_latency_median'] * 100):+.1f}%")

    print(f"\n3. TTFT Impact:")
    print(f"   - Average median TTFT change: {avg_ttft_diff:+.1f}%")
    print(f"   - At low concurrency (5): {((native_offload['latencies'][0]['ttft_median'] - no_offload['latencies'][0]['ttft_median']) / no_offload['latencies'][0]['ttft_median'] * 100):+.1f}%")
    print(f"   - At high concurrency (250): {((native_offload['latencies'][4]['ttft_median'] - no_offload['latencies'][4]['ttft_median']) / no_offload['latencies'][4]['ttft_median'] * 100):+.1f}%")

    print("\n" + "=" * 100)


if __name__ == '__main__':
    results = parse_benchmark_log('benchmark-run-complete.log')
    print_comparison(results)
