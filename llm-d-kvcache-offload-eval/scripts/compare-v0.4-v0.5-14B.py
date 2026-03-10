#!/usr/bin/env python3
"""Compare v0.4.0 vs v0.5.0 throughput for 14B model."""

import json
import subprocess
from pathlib import Path

def extract_throughput(result_dir):
    """Extract throughput from guidellm results."""
    json_file = result_dir / 'guidellm-results.json.zst'
    if not json_file.exists():
        return None

    try:
        result = subprocess.run(
            ['zstdcat', str(json_file)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        benchmarks = data.get('benchmarks', [{}])
        if not benchmarks:
            return None

        results = benchmarks[0].get('results', {})
        return results.get('output_token_throughput', 0)

    except Exception as e:
        print(f"  Error processing {result_dir.name}: {e}")
        return None

def main():
    """Extract and compare v0.4.0 vs v0.5.0 results for 14B model."""
    results_dir = Path('results')

    print("=" * 80)
    print("Qwen3-14B Model: v0.4.0 vs v0.5.0 Throughput Comparison (rate=50)")
    print("=" * 80)

    # v0.4.0 results (native-offload and no-offload at 10K blocks)
    v04_patterns = {
        'no-offload': '1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_no-offload_replica1_rate50',
        'native-offload': '1x2xL40S_upstream-llm-d-0.4.0_Qwen3-14B_native-offload_replica1_rate50',
    }

    # v0.5.0 results
    v05_patterns = {
        'no-offload': '1x2xL40S_upstream-llm-d-0.5.0_Qwen3-14B_no-offload_replica1_rate50',
        'native-offload-10k': '1x2xL40S_upstream-llm-d-0.5.0_Qwen3-14B_native-offload-10k_replica1_rate50',
        'native-offload-20k': '1x2xL40S_upstream-llm-d-0.5.0_Qwen3-14B_native-offload-20k_replica1_rate50',
    }

    print("\nv0.4.0 Results:")
    print("-" * 60)
    v04_data = {}
    for config, pattern in v04_patterns.items():
        dirs = list(results_dir.glob(pattern))
        if dirs:
            throughput = extract_throughput(dirs[0])
            if throughput:
                v04_data[config] = throughput
                print(f"  {config:25s}: {throughput:7.2f} tok/s")
            else:
                print(f"  {config:25s}: No data")
        else:
            print(f"  {config:25s}: Not found")

    print("\nv0.5.0 Results:")
    print("-" * 60)
    v05_data = {}
    for config, pattern in v05_patterns.items():
        dirs = list(results_dir.glob(pattern))
        if dirs:
            throughput = extract_throughput(dirs[0])
            if throughput:
                v05_data[config] = throughput
                print(f"  {config:25s}: {throughput:7.2f} tok/s")
            else:
                print(f"  {config:25s}: No data")
        else:
            print(f"  {config:25s}: Not found")

    # Comparison
    if 'no-offload' in v04_data and 'no-offload' in v05_data:
        print("\nComparison:")
        print("=" * 80)

        v04_baseline = v04_data['no-offload']
        v05_baseline = v05_data['no-offload']

        print(f"\nBaseline (no-offload):")
        print(f"  v0.4.0: {v04_baseline:7.2f} tok/s")
        print(f"  v0.5.0: {v05_baseline:7.2f} tok/s")
        baseline_delta = ((v05_baseline - v04_baseline) / v04_baseline) * 100
        print(f"  Change: {baseline_delta:+6.2f}%")

        if 'native-offload' in v04_data:
            v04_offload = v04_data['native-offload']
            v04_offload_delta = ((v04_offload - v04_baseline) / v04_baseline) * 100
            print(f"\nv0.4.0 Native Offload (10K blocks):")
            print(f"  Throughput: {v04_offload:7.2f} tok/s")
            print(f"  vs Baseline: {v04_offload_delta:+6.2f}%")

        if 'native-offload-10k' in v05_data:
            v05_offload_10k = v05_data['native-offload-10k']
            v05_offload_10k_delta = ((v05_offload_10k - v05_baseline) / v05_baseline) * 100
            print(f"\nv0.5.0 Native Offload (10K blocks):")
            print(f"  Throughput: {v05_offload_10k:7.2f} tok/s")
            print(f"  vs Baseline: {v05_offload_10k_delta:+6.2f}%")

        if 'native-offload-20k' in v05_data:
            v05_offload_20k = v05_data['native-offload-20k']
            v05_offload_20k_delta = ((v05_offload_20k - v05_baseline) / v05_baseline) * 100
            print(f"\nv0.5.0 Native Offload (20K blocks):")
            print(f"  Throughput: {v05_offload_20k:7.2f} tok/s")
            print(f"  vs Baseline: {v05_offload_20k_delta:+6.2f}%")

        # Regression analysis
        print("\n" + "=" * 80)
        print("REGRESSION ANALYSIS")
        print("=" * 80)

        if 'native-offload' in v04_data and 'native-offload-10k' in v05_data:
            v04_gain = v04_offload_delta
            v05_gain = v05_offload_10k_delta

            print(f"\nNative Offload Performance (10K blocks):")
            print(f"  v0.4.0: {v04_gain:+6.2f}% vs baseline")
            print(f"  v0.5.0: {v05_gain:+6.2f}% vs baseline")

            if v04_gain > 0 and v05_gain < v04_gain:
                regression = v05_gain - v04_gain
                print(f"\n  ⚠️  REGRESSION DETECTED: {regression:+.2f} percentage points")
                print(f"      Offload benefit reduced from {v04_gain:+.2f}% to {v05_gain:+.2f}%")
            elif v04_gain > 0 and v05_gain < 0:
                print(f"\n  ⛔ SEVERE REGRESSION: Offload shifted from +{v04_gain:.2f}% gain to {v05_gain:.2f}% loss")
            else:
                print(f"\n  ✓ No significant regression detected")

if __name__ == '__main__':
    main()
