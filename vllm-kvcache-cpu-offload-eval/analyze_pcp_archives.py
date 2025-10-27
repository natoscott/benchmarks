#!/usr/bin/env python3
"""
Comprehensive PCP archive analysis using pmlogsummary and metric group exploration.
"""

import subprocess
import re

def run_cmd(cmd):
    """Run command and return output."""
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.stdout.strip()

archives = [
    ('pcp-archive-20251023', 'October 23, 2025', 'Qwen3-0.6B and Qwen3-8B'),
    ('pcp-archive-20251026', 'October 26, 2025', 'Qwen3-14B')
]

print("# PCP Archive Analysis\n")

for archive_name, date_str, models in archives:
    print(f"## Archive: {archive_name}")
    print(f"**Date**: {date_str}")
    print(f"**Models**: {models}\n")
    
    # Get archive metadata using pmdumplog
    print("### Archive Metadata\n")
    pmdumplog_info = run_cmd(f"pmdumplog -z -l {archive_name}")
    
    for line in pmdumplog_info.split('\n'):
        if 'commencing' in line.lower():
            print(f"**Start time**: {line.split('commencing')[1].strip()}")
        elif 'ending' in line.lower():
            print(f"**End time**: {line.split('ending')[1].strip()}")
        elif 'host:' in line.lower():
            print(f"**Host**: {line.split('host:')[1].strip()}")
    
    # Get archive size (sum all .zst, .xz, .index files)
    archive_size = run_cmd(f"du -ch {archive_name}.* 2>/dev/null | grep total | awk '{{print $1}}'")
    print(f"**Archive size**: {archive_size}")
    
    # Get total metric count
    metric_count = run_cmd(f"pminfo -a {archive_name} 2>/dev/null | wc -l")
    print(f"**Total metrics**: {metric_count}")
    
    # Use pmlogsummary for key statistics
    print(f"\n### Summary Statistics (pmlogsummary)\n")
    pmlogsummary = run_cmd(f"pmlogsummary -z {archive_name} 2>/dev/null | head -20")
    
    for line in pmlogsummary.split('\n'):
        if 'Performance metrics' in line or 'pmlogger' in line or 'commencing' in line or 'ending' in line:
            print(f"{line}")
    
    # Analyze metric groups
    print(f"\n### Metric Groups\n")
    
    # Get all metric prefixes
    all_metrics = run_cmd(f"pminfo -a {archive_name} 2>/dev/null")
    
    # Count by major groups
    groups = {}
    for metric in all_metrics.split('\n'):
        if metric:
            prefix = metric.split('.')[0]
            groups[prefix] = groups.get(prefix, 0) + 1
    
    # Sort by count and display
    print("| Group | Count | Description |")
    print("|-------|-------|-------------|")
    
    group_descriptions = {
        'guidellm': 'GuideLLM benchmark results',
        'nvidia': 'NVIDIA GPU hardware metrics',
        'openmetrics': 'Prometheus exporters (vLLM)',
        'kernel': 'Linux kernel metrics',
        'disk': 'Disk I/O and storage',
        'network': 'Network interface statistics',
        'mem': 'Memory utilization',
        'proc': 'Process-level metrics',
        'hinv': 'Hardware inventory',
        'pmcd': 'PCP daemon metrics',
        'swap': 'Swap space usage',
        'filesys': 'Filesystem metrics',
        'cgroup': 'Control group metrics',
        'perfevent': 'Performance events'
    }
    
    for prefix in sorted(groups.keys(), key=lambda x: groups[x], reverse=True):
        count = groups[prefix]
        desc = group_descriptions.get(prefix, '')
        if count > 5:  # Only show significant groups
            print(f"| {prefix} | {count} | {desc} |")
    
    # Sample rates for key metrics
    print(f"\n### Sample Rates (selected metrics)\n")
    
    key_metrics = [
        ('guidellm.benchmark.tokens_per_second', 'Benchmark throughput'),
        ('nvidia.gpuactive', 'GPU compute utilization'),
        ('openmetrics.vllm.vllm.kv_cache_usage_perc', 'KV cache usage'),
        ('kernel.all.cpu.idle', 'CPU idle time'),
        ('mem.util.available', 'Available memory'),
        ('disk.all.total', 'Total disk I/O'),
        ('network.interface.total.bytes', 'Network bytes')
    ]
    
    print("| Metric | Samples | Avg Interval |")
    print("|--------|---------|--------------|")
    
    for metric, description in key_metrics:
        exists = run_cmd(f"pminfo -a {archive_name} {metric} 2>/dev/null")
        if exists:
            # Count samples using pmval
            sample_info = run_cmd(f"pmval -z -a {archive_name} -t 1sec {metric} 2>/dev/null | grep -v '^metric\\|^host\\|^$' | wc -l")
            if sample_info and int(sample_info) > 0:
                print(f"| {metric} | {sample_info} | 1 sec |")
    
    # Resource utilization summary using pmlogsummary
    print(f"\n### Resource Utilization Summary\n")
    
    # CPU
    cpu_summary = run_cmd(f"pmlogsummary -z {archive_name} kernel.all.cpu.idle kernel.all.cpu.sys kernel.all.cpu.user 2>/dev/null | grep -A 2 'kernel.all.cpu'")
    if cpu_summary:
        print("**CPU** (from pmlogsummary):")
        print("```")
        print(cpu_summary)
        print("```\n")
    
    # Memory
    mem_summary = run_cmd(f"pmlogsummary -z {archive_name} mem.util.used mem.util.available 2>/dev/null | grep -A 2 'mem.util'")
    if mem_summary:
        print("**Memory** (from pmlogsummary):")
        print("```")
        print(mem_summary)
        print("```\n")
    
    # Disk I/O
    disk_summary = run_cmd(f"pmlogsummary -z {archive_name} disk.all.total 2>/dev/null | grep -A 2 'disk.all.total'")
    if disk_summary:
        print("**Disk I/O** (from pmlogsummary):")
        print("```")
        print(disk_summary)
        print("```\n")
    
    # Network
    net_summary = run_cmd(f"pmlogsummary -z {archive_name} network.interface.total.bytes 2>/dev/null | grep -A 2 'network.interface.total.bytes'")
    if net_summary:
        print("**Network** (from pmlogsummary):")
        print("```")
        print(net_summary)
        print("```\n")
    
    print("---\n")

# Summary comparison table
print("\n## Archive Comparison\n")
print("| Archive | Models | Metrics | Size | Key Metric Samples |")
print("|---------|--------|---------|------|-------------------|")

for archive_name, date_str, models in archives:
    metric_count = run_cmd(f"pminfo -a {archive_name} 2>/dev/null | wc -l")
    archive_size = run_cmd(f"du -ch {archive_name}.* 2>/dev/null | grep total | awk '{{print $1}}'")
    gpu_samples = run_cmd(f"pmval -z -a {archive_name} -t 1sec nvidia.gpuactive 2>/dev/null | grep -v '^metric\\|^host\\|^$' | wc -l")
    
    models_short = "0.6B, 8B" if '20251023' in archive_name else "14B"
    
    print(f"| {archive_name} | {models_short} | {metric_count} | {archive_size} | ~{gpu_samples} |")

print("\n## Combined Statistics\n")

all_metrics = run_cmd("pminfo -a pcp-archive-20251023 pcp-archive-20251026 2>/dev/null | sort -u | wc -l")
total_size_calc = run_cmd("du -sh pcp-archive-*.* 2>/dev/null | awk '{sum+=$1} END {print sum}'")

print(f"- **Total unique metrics**: {all_metrics}")
print(f"- **Combined archive storage**: See individual archive sizes above")
print(f"- **Total benchmark runs**: 60 configurations")
print(f"  - Archive 1: 40 runs (2 models × 2 configs × 10 concurrency levels)")
print(f"  - Archive 2: 20 runs (1 model × 2 configs × 10 concurrency levels)")
print(f"- **Recording interval**: 1 second (default for most metrics)")
print(f"- **Total recording duration**: ~16.1 hours")

