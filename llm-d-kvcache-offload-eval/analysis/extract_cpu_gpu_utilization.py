#!/usr/bin/env python3
"""
Extract CPU and GPU utilization metrics from PCP archives and generate summary.
"""

import subprocess
import pandas as pd
import json
from pathlib import Path
import sys

# Models and configurations to analyze
MODELS = ["Qwen3-0.6B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B-AWQ"]
CONFIGS = ["no-offload", "native-offload", "llm-d-redis", "llm-d-valkey"]

# Metrics to extract
CPU_METRICS = [
    "kernel.all.cpu.user",
    "kernel.all.cpu.sys",
    "kernel.all.cpu.idle",
]

GPU_METRICS = [
    "openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL",
    "openmetrics.dcgm.DCGM_FI_DEV_MEM_COPY_UTIL",
]


def find_pcp_archive(model, config):
    """Find the PCP archive directory for a given model and config."""
    results_dir = Path("results")
    pattern = f"1x2xL40S_upstream-llm-d-0.4.0_{model}_{config}"

    for result_dir in results_dir.glob(pattern):
        pcp_archives = result_dir / "pcp-archives"
        if pcp_archives.exists():
            # Find the first archive directory
            for archive_host in pcp_archives.iterdir():
                if archive_host.is_dir():
                    # Find the archive files (there should be a .meta.zst file)
                    meta_files = list(archive_host.glob("*.meta.zst"))
                    if meta_files:
                        # Return the archive base (without extension)
                        archive_base = str(meta_files[0]).replace(".meta.zst", "")
                        return archive_base
    return None


def extract_metrics(archive_path, metrics):
    """Extract metrics from PCP archive using pmval."""
    results = {}

    for metric in metrics:
        try:
            # Use pmval to get the metric values
            cmd = ["pmval", "-a", archive_path, "-t", "1sec", "-w", "20", metric]
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)

            # Parse pmval output - skip header lines and extract values
            values = []
            for line in output.split('\n'):
                line = line.strip()
                if not line or line.startswith('metric:') or line.startswith('host:') or \
                   line.startswith('semantics:') or line.startswith('units:') or \
                   line.startswith('samples:') or line.startswith('interval:'):
                    continue
                # Try to extract the value (last column typically)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # Handle instance values
                        value_str = parts[-1]
                        # For multiple instances, we might see multiple values
                        # For now, just try to parse the last one
                        value = float(value_str)
                        values.append(value)
                    except (ValueError, IndexError):
                        continue

            if values:
                results[metric] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        except subprocess.CalledProcessError:
            # Metric might not exist in this archive
            pass

    return results


def main():
    all_results = []

    print("Extracting CPU and GPU utilization metrics from PCP archives...")
    print("=" * 80)

    for model in MODELS:
        for config in CONFIGS:
            print(f"\nProcessing {model} / {config}...")

            archive_path = find_pcp_archive(model, config)
            if not archive_path:
                print(f"  WARNING: Archive not found for {model} / {config}")
                continue

            print(f"  Archive: {archive_path}")

            # Extract CPU metrics
            cpu_results = extract_metrics(archive_path, CPU_METRICS)

            # Extract GPU metrics
            gpu_results = extract_metrics(archive_path, GPU_METRICS)

            # Calculate CPU utilization percentage
            if all(m in cpu_results for m in ["kernel.all.cpu.user", "kernel.all.cpu.sys", "kernel.all.cpu.idle"]):
                user_mean = cpu_results["kernel.all.cpu.user"]["mean"]
                sys_mean = cpu_results["kernel.all.cpu.sys"]["mean"]
                idle_mean = cpu_results["kernel.all.cpu.idle"]["mean"]
                total = user_mean + sys_mean + idle_mean
                if total > 0:
                    cpu_util_pct = ((user_mean + sys_mean) / total) * 100
                else:
                    cpu_util_pct = 0
            else:
                cpu_util_pct = None

            # Get GPU utilization
            if "openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL" in gpu_results:
                gpu_util_pct = gpu_results["openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL"]["mean"]
            else:
                gpu_util_pct = None

            # Get GPU memory copy utilization
            if "openmetrics.dcgm.DCGM_FI_DEV_MEM_COPY_UTIL" in gpu_results:
                gpu_mem_util_pct = gpu_results["openmetrics.dcgm.DCGM_FI_DEV_MEM_COPY_UTIL"]["mean"]
            else:
                gpu_mem_util_pct = None

            result = {
                'model': model,
                'config': config,
                'cpu_util_pct': cpu_util_pct,
                'gpu_util_pct': gpu_util_pct,
                'gpu_mem_copy_util_pct': gpu_mem_util_pct,
            }

            all_results.append(result)

            print(f"  CPU Utilization: {cpu_util_pct:.1f}%" if cpu_util_pct is not None else "  CPU Utilization: N/A")
            print(f"  GPU Utilization: {gpu_util_pct:.1f}%" if gpu_util_pct is not None else "  GPU Utilization: N/A")
            print(f"  GPU Memory Copy Util: {gpu_mem_util_pct:.1f}%" if gpu_mem_util_pct is not None else "  GPU Memory Copy Util: N/A")

    # Save results to CSV
    df = pd.DataFrame(all_results)
    output_file = "analysis/cpu_gpu_utilization.csv"
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
