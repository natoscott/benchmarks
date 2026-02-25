#!/usr/bin/env python3
"""Extract KV-cache allocation data from vLLM startup logs."""

import re
import sys
import os
import glob
import csv

def parse_log(log_file):
    """Parse a vLLM startup log and extract KV-cache data."""
    with open(log_file, 'r') as f:
        content = f.read()

    data = {}

    # Extract basic info
    patterns = {
        'model': r'model_tag[\'"]:\s*[\'"]([^"\']+)[\'"]',
        'tensor_parallel': r'tensor_parallel_size[\'"]:\s*(\d+)',
        'gpu_kv_tokens': r'GPU KV cache size:\s*([\d,]+)\s*tokens',
        'gpu_memory_gib': r'Available KV cache memory:\s*([\d.]+)\s*GiB',
        'max_concurrency': r'Maximum concurrency.*:\s*([\d.]+)x',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            # Remove commas from numbers
            if ',' in value:
                value = value.replace(',', '')
            data[key] = value

    # Check if this is an offload config
    is_offload = 'OffloadingConnector' in content
    data['is_offload'] = is_offload

    if is_offload:
        # Extract CPU offload config
        match = re.search(r'num_cpu_blocks[\'"]:\s*(\d+)', content)
        if match:
            data['cpu_blocks_configured'] = match.group(1)

        # Extract actual CPU allocation shape
        match = re.search(r'Allocating a cross layer KV cache of shape \((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', content)
        if match:
            shape = tuple(int(x) for x in match.groups())
            data['cpu_allocation_shape'] = shape

            # Calculate CPU blocks and memory
            blocks, layers, heads_per_tp, kv_heads, head_size_parts, embedding_dim = shape
            data['cpu_blocks_actual'] = blocks

            # Calculate total elements and memory
            total_elements = blocks * layers * heads_per_tp * kv_heads * head_size_parts * embedding_dim
            bytes_per_element = 2  # bfloat16
            total_bytes = total_elements * bytes_per_element
            cpu_memory_gib = total_bytes / (1024**3)
            data['cpu_memory_gib'] = f"{cpu_memory_gib:.2f}"

            # Calculate tokens per block based on GPU allocation
            if 'gpu_kv_tokens' in data:
                gpu_tokens = int(data['gpu_kv_tokens'])
                tokens_per_block = gpu_tokens / blocks
                data['tokens_per_block'] = f"{tokens_per_block:.1f}"

                # Calculate what the configured blocks would be in tokens
                if 'cpu_blocks_configured' in data:
                    configured_blocks = int(data['cpu_blocks_configured'])
                    configured_tokens = configured_blocks * tokens_per_block
                    data['cpu_tokens_configured'] = f"{configured_tokens:.0f}"

    return data

def main():
    log_dir = "vllm-startup-logs"

    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' not found")
        sys.exit(1)

    # Find all log files
    log_files = glob.glob(f"{log_dir}/vllm-*.log")

    if not log_files:
        print(f"Error: No log files found in '{log_dir}'")
        sys.exit(1)

    print(f"Found {len(log_files)} log files")
    print()

    # Parse all logs
    results = []
    for log_file in sorted(log_files):
        print(f"Parsing: {os.path.basename(log_file)}")
        data = parse_log(log_file)
        data['log_file'] = os.path.basename(log_file)
        results.append(data)

    print()
    print("=" * 120)
    print("KV-CACHE ALLOCATION DATA")
    print("=" * 120)
    print()

    # Print summary table
    for result in results:
        model = result.get('model', 'Unknown')
        config = result['log_file'].replace('vllm-', '').replace('.log', '')

        print(f"Model: {model}")
        print(f"Config: {config}")
        print(f"Tensor Parallel: {result.get('tensor_parallel', 'N/A')}")
        print()

        print(f"GPU:")
        print(f"  Available Memory: {result.get('gpu_memory_gib', 'N/A')} GiB")
        print(f"  KV Cache Tokens: {result.get('gpu_kv_tokens', 'N/A')}")
        print(f"  Max Concurrency: {result.get('max_concurrency', 'N/A')}x")
        print()

        if result.get('is_offload'):
            print(f"CPU Offload:")
            print(f"  Configured Blocks: {result.get('cpu_blocks_configured', 'N/A')}")
            print(f"  Actual Blocks: {result.get('cpu_blocks_actual', 'N/A')}")
            print(f"  CPU Memory: {result.get('cpu_memory_gib', 'N/A')} GiB")
            print(f"  Tokens/Block: {result.get('tokens_per_block', 'N/A')}")

            if 'cpu_blocks_configured' in result and 'cpu_blocks_actual' in result:
                configured = int(result['cpu_blocks_configured'])
                actual = int(result['cpu_blocks_actual'])
                ratio = actual / configured
                print(f"  Block Ratio (actual/configured): {ratio:.2f}x")

            print()
        else:
            print("CPU Offload: No (GPU-only)")
            print()

        print("-" * 120)
        print()

    # Save to CSV
    csv_file = "analysis/kvcache_allocations_actual.csv"
    os.makedirs("analysis", exist_ok=True)

    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            'model', 'config_name', 'tensor_parallel', 'is_offload',
            'gpu_memory_gib', 'gpu_kv_tokens', 'max_concurrency',
            'cpu_blocks_configured', 'cpu_blocks_actual', 'cpu_memory_gib',
            'tokens_per_block', 'log_file'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            config_name = result['log_file'].replace('vllm-', '').replace('.log', '')
            row = {
                'model': result.get('model', ''),
                'config_name': config_name,
                'tensor_parallel': result.get('tensor_parallel', ''),
                'is_offload': 'yes' if result.get('is_offload') else 'no',
                'gpu_memory_gib': result.get('gpu_memory_gib', ''),
                'gpu_kv_tokens': result.get('gpu_kv_tokens', ''),
                'max_concurrency': result.get('max_concurrency', ''),
                'cpu_blocks_configured': result.get('cpu_blocks_configured', ''),
                'cpu_blocks_actual': result.get('cpu_blocks_actual', ''),
                'cpu_memory_gib': result.get('cpu_memory_gib', ''),
                'tokens_per_block': result.get('tokens_per_block', ''),
                'log_file': result.get('log_file', ''),
            }
            writer.writerow(row)

    print(f"Data saved to: {csv_file}")

if __name__ == '__main__':
    main()
