#!/usr/bin/env python3
"""Comprehensive analysis of llm-d v0.5.1 KV cache offload benchmarks.

Analyzes 128 benchmark runs across 4 configs x 4 models x 8 rates.
Generates visualizations and summary statistics for report writing.
"""

import json
import subprocess
import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Visualization standards (visualization-palette skill) ────────────────────
sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

RESULTS_DIR = Path('results')
OUTPUT_DIR = Path('analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
CONFIGS = ['no-offload', 'native-offload-20k', 'fs-offload', 'cpu+fs-offload-20k']
RATES = [1, 50, 100, 150, 300, 400, 500, 650]

CONFIG_LABELS = {
    'no-offload': 'No Offload',
    'native-offload-20k': 'Native Offload (20k)',
    'fs-offload': 'Filesystem Offload',
    'cpu+fs-offload-20k': 'CPU+FS Offload (20k)',
}

# ── Historical baseline data ─────────────────────────────────────────────────
# v0.4.0 baselines (from REPORT-v0.4.0.md)
V040_BASELINES = {
    ('Qwen3-0.6B', 'no-offload'):        602.0,
    ('Qwen3-8B',   'no-offload'):        113.0,
    ('Qwen3-14B',  'no-offload'):         58.7,
    ('Qwen3-32B-AWQ', 'no-offload'):      49.2,
    ('Qwen3-0.6B', 'native-offload'):    426.8,
    ('Qwen3-8B',   'native-offload'):     71.8,
    ('Qwen3-14B',  'native-offload'):     59.0,
    ('Qwen3-32B-AWQ', 'native-offload'): 48.7,
}

# v0.5.0 baselines (from REPORT-v0.5.0.md)
V050_BASELINES = {
    ('Qwen3-0.6B', 'no-offload'):           634.7,
    ('Qwen3-8B',   'no-offload'):           114.1,
    ('Qwen3-14B',  'no-offload'):            66.1,
    ('Qwen3-32B-AWQ', 'no-offload'):         51.2,
    ('Qwen3-0.6B', 'native-offload-20k'):   632.5,
    ('Qwen3-8B',   'native-offload-20k'):    84.3,
    ('Qwen3-14B',  'native-offload-20k'):    65.1,
    ('Qwen3-32B-AWQ', 'native-offload-20k'): 21.3,
}


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Parse GuideLLM JSON results
# ══════════════════════════════════════════════════════════════════════════════

def parse_dir_name(dir_name):
    """Parse benchmark directory name into components.

    Format: 1x2xL40S_upstream-llm-d-0.5.1_Qwen3-0.6B_no-offload_replica1_rate50
    Config names can contain '+' and '-', so split on _ but be careful:
    parts[0]=hardware, parts[1]=software, parts[2]=model, parts[-2]=replicas, parts[-1]=rate
    parts[3:-2] joined with '_' = config (handles cpu+fs-offload-20k etc.)
    """
    parts = dir_name.split('_')
    hardware = parts[0]
    software = parts[1]
    model = parts[2]
    replicas = int(parts[-2].replace('replica', ''))
    rate = int(parts[-1].replace('rate', ''))
    # Config is everything between model and replicas
    config = '_'.join(parts[3:-2])
    return hardware, software, model, config, replicas, rate


def extract_guidellm_metrics(result_file):
    """Extract throughput and latency metrics from a GuideLLM result file."""
    try:
        proc = subprocess.run(
            ['zstdcat', str(result_file)],
            capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0:
            print(f"  zstdcat failed for {result_file}: {proc.stderr[:200]}")
            return None

        data = json.loads(proc.stdout)

        dir_name = result_file.parent.name
        _, software, model, config, replicas, rate = parse_dir_name(dir_name)

        # Only process v0.5.1 results
        if '0.5.1' not in software:
            return None

        benchmarks = data.get('benchmarks', [])
        if not benchmarks:
            return None

        benchmark = benchmarks[0]
        metrics = benchmark.get('metrics', {})
        duration = benchmark.get('duration', 0)

        # ── Throughput ──────────────────────────────────────────────────────
        # Compute from total successful output tokens / benchmark duration.
        # NOTE: output_tokens_per_second.successful.mean is a per-streaming-
        # iteration stat (tokens/s within individual requests), not aggregate
        # benchmark throughput. Always use total_sum / duration instead.
        throughput = 0.0
        otc = metrics.get('output_token_count', {})
        if otc and duration > 0:
            total_tokens = otc.get('successful', {}).get('total_sum', 0)
            if total_tokens == 0:
                total_tokens = otc.get('successful', {}).get('sum', 0)
            throughput = total_tokens / duration

        # ── Latency ─────────────────────────────────────────────────────────
        # TTFT (time to first token) in seconds
        ttft_ms = metrics.get('time_to_first_token_ms', {})
        ttft_median = ttft_ms.get('successful', {}).get('median', 0.0) / 1000.0  # ms->s

        # ITL (inter-token latency) in ms
        itl_ms = metrics.get('inter_token_latency_ms', {})
        itl_median = itl_ms.get('successful', {}).get('median', 0.0)

        # TPOT (time per output token) in ms
        tpot_ms = metrics.get('time_per_output_token_ms', {})
        tpot_median = tpot_ms.get('successful', {}).get('median', 0.0)

        # ── Request counts ───────────────────────────────────────────────────
        request_totals = metrics.get('request_totals', {})
        completed = request_totals.get('completed', 0)
        if completed == 0:
            completed = request_totals.get('successful', 0)

        return {
            'model': model,
            'config': config,
            'rate': rate,
            'throughput': throughput,
            'duration': duration,
            'completed_requests': completed,
            'ttft_median_s': ttft_median,
            'itl_median_ms': itl_median,
            'tpot_median_ms': tpot_median,
        }

    except Exception as e:
        print(f"  Error parsing {result_file}: {e}")
        return None


def load_all_guidellm_results():
    """Load all v0.5.1 GuideLLM results."""
    print("Loading GuideLLM results...")
    pattern = '1x2xL40S_upstream-llm-d-0.5.1_*/guidellm-results.json.zst'
    files = sorted(RESULTS_DIR.glob(pattern))
    print(f"  Found {len(files)} result files")

    records = []
    for f in files:
        row = extract_guidellm_metrics(f)
        if row:
            records.append(row)

    df = pd.DataFrame(records)
    print(f"  Successfully parsed {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Extract PCP metrics
# ══════════════════════════════════════════════════════════════════════════════

def find_archive_base(run_dir):
    """Find the PCP archive base path in a run directory."""
    archive_dirs = list(Path(run_dir).glob('pcp-archives/*/'))
    if not archive_dirs:
        return None
    archive_dir = archive_dirs[0]
    meta_files = list(archive_dir.glob('*.meta.zst'))
    if not meta_files:
        return None
    return str(meta_files[0]).replace('.meta.zst', '')


def run_pmrep(archive_base, metrics):
    """Run pmrep to extract mean values for a list of metrics from an archive.

    Returns a dict mapping each requested metric name to its column mean.
    pmrep CSV column headers include label strings (e.g. "metric-0 label:val"),
    so we match columns by metric-name prefix rather than exact name.
    When multiple instances exist (e.g. two GPUs), we sum them.
    """
    cmd = ['pmrep', '-a', archive_base, '-o', 'csv', '-z', '-t', '10s'] + metrics
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not result.stdout.strip():
            return None

        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return None

        # Parse CSV header: first column is Time, rest are metric columns
        # Header format: "Time","metric-0 label1:v1 label2:v2","metric-1 ..."
        header_line = lines[0]
        # Use csv module to handle quoted headers properly
        import csv as csv_mod
        reader = csv_mod.reader([header_line])
        raw_headers = next(reader)

        # Build mapping: metric_name -> list of column indices
        # Column header starts with "metric_name-N" where N is instance index
        metric_col_map = defaultdict(list)
        for col_idx, col_header in enumerate(raw_headers):
            if col_idx == 0:
                continue  # skip Time column
            # Strip instance suffix: "openmetrics.foo.bar-0 labels..." -> "openmetrics.foo.bar"
            base = col_header.split('-')[0].strip()
            metric_col_map[base].append(col_idx)

        # Parse data rows
        data_by_col = defaultdict(list)
        for line in lines[1:]:
            reader2 = csv_mod.reader([line])
            parts = next(reader2)
            if len(parts) != len(raw_headers):
                continue
            for col_idx, val_str in enumerate(parts):
                if col_idx == 0:
                    continue
                try:
                    data_by_col[col_idx].append(float(val_str))
                except (ValueError, TypeError):
                    pass  # skip empty/non-numeric

        if not data_by_col:
            return None

        # Compute mean per column, then aggregate instances per metric
        result_means = {}
        for metric in metrics:
            col_indices = metric_col_map.get(metric, [])
            if not col_indices:
                continue
            col_means = []
            for ci in col_indices:
                vals = data_by_col.get(ci, [])
                if vals:
                    col_means.append(np.nanmean(vals))
            if col_means:
                # Sum across instances (e.g. multi-GPU util, multi-disk bytes)
                result_means[metric] = sum(col_means)

        return result_means if result_means else None

    except subprocess.TimeoutExpired:
        print(f"    pmrep timeout for {archive_base}")
        return None
    except Exception as e:
        print(f"    pmrep error: {e}")
        return None


def run_pmlogsummary(archive_base, metrics):
    """Use pmlogsummary -I -N to get mean values."""
    cmd = ['pmlogsummary', '-I', '-N', archive_base] + metrics
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}

        values = {}
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                metric = parts[0]
                try:
                    val = float(parts[1])
                    values[metric] = val
                except ValueError:
                    pass
        return values
    except Exception as e:
        return {}


def extract_pcp_metrics_for_run(run_dir):
    """Extract PCP metrics from a single run directory."""
    archive_base = find_archive_base(run_dir)
    if not archive_base:
        return None

    result = {'archive': archive_base}

    # ── GPU utilization via DCGM ─────────────────────────────────────────────
    gpu_metrics = [
        'openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL',
        'openmetrics.dcgm.DCGM_FI_DEV_MEM_COPY_UTIL',
    ]
    gpu_data = run_pmrep(archive_base, gpu_metrics)
    if gpu_data:
        result.update(gpu_data)

    # Fall back to nvidia.* metrics if DCGM not available
    if 'openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL' not in result:
        nvidia_metrics = ['nvidia.gpuactive', 'nvidia.memused']
        nvidia_data = run_pmrep(archive_base, nvidia_metrics)
        if nvidia_data:
            result.update(nvidia_data)

    # ── vLLM metrics ─────────────────────────────────────────────────────────
    # In these archives metric names use dots throughout (no colons)
    vllm_metrics = [
        'openmetrics.vllm.vllm.kv_cache_usage_perc',
        'openmetrics.vllm.vllm.num_requests_running',
        'openmetrics.vllm.vllm.num_requests_waiting',
        'openmetrics.vllm.vllm.prefix_cache_hits_total',
        'openmetrics.vllm.vllm.prefix_cache_queries_total',
        'openmetrics.vllm.vllm.external_prefix_cache_hits_total',
        'openmetrics.vllm.vllm.external_prefix_cache_queries_total',
    ]
    vllm_data = run_pmrep(archive_base, vllm_metrics)
    if vllm_data:
        result.update(vllm_data)

    # ── Disk I/O ─────────────────────────────────────────────────────────────
    disk_metrics = ['disk.dev.read_bytes', 'disk.dev.write_bytes']
    disk_data = run_pmrep(archive_base, disk_metrics)
    if disk_data:
        result.update(disk_data)

    return result


def extract_all_pcp_metrics(df_throughput):
    """Extract PCP metrics for all run directories."""
    print("\nExtracting PCP metrics from archives...")

    pcp_records = []
    dirs = sorted(RESULTS_DIR.glob('1x2xL40S_upstream-llm-d-0.5.1_*/'))

    for run_dir in dirs:
        dir_name = run_dir.name
        try:
            _, software, model, config, replicas, rate = parse_dir_name(dir_name)
        except Exception:
            continue

        if '0.5.1' not in software:
            continue

        print(f"  {model} / {config} / rate={rate} ...", end=' ', flush=True)
        pcp_data = extract_pcp_metrics_for_run(run_dir)

        if pcp_data:
            row = {
                'model': model,
                'config': config,
                'rate': rate,
            }
            row.update(pcp_data)
            pcp_records.append(row)
            print("OK")
        else:
            print("no archive")

    if pcp_records:
        df_pcp = pd.DataFrame(pcp_records)
        # Simplify metric column names
        rename_map = {
            'openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL': 'gpu_util_pct',
            'openmetrics.dcgm.DCGM_FI_DEV_MEM_COPY_UTIL': 'gpu_mem_copy_pct',
            'nvidia.gpuactive': 'gpu_util_pct',
            'openmetrics.vllm.vllm.kv_cache_usage_perc': 'kv_cache_usage_pct',
            'openmetrics.vllm.vllm.num_requests_running': 'requests_running',
            'openmetrics.vllm.vllm.num_requests_waiting': 'requests_waiting',
            'openmetrics.vllm.vllm.prefix_cache_hits_total': 'prefix_cache_hits',
            'openmetrics.vllm.vllm.prefix_cache_queries_total': 'prefix_cache_queries',
            'openmetrics.vllm.vllm.external_prefix_cache_hits_total': 'ext_prefix_hits',
            'openmetrics.vllm.vllm.external_prefix_cache_queries_total': 'ext_prefix_queries',
            'disk.dev.read_bytes': 'disk_read_bytes_s',
            'disk.dev.write_bytes': 'disk_write_bytes_s',
        }
        df_pcp = df_pcp.rename(columns={k: v for k, v in rename_map.items() if k in df_pcp.columns})
        print(f"  Extracted PCP metrics for {len(df_pcp)} runs")
        return df_pcp
    else:
        print("  WARNING: No PCP metrics extracted")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Analysis
# ══════════════════════════════════════════════════════════════════════════════

def find_peak_throughput(df):
    """Find peak throughput row for each model+config combination."""
    peak_rows = []
    for (model, config), group in df.groupby(['model', 'config']):
        if group['throughput'].max() > 0:
            idx = group['throughput'].idxmax()
            peak_rows.append(group.loc[idx].copy())
    return pd.DataFrame(peak_rows).reset_index(drop=True)


def compute_pcp_summary(df_pcp):
    """Compute per-(model,config) PCP summary at peak throughput rates."""
    if df_pcp.empty:
        return pd.DataFrame()

    numeric_cols = [c for c in df_pcp.columns if c not in ('model', 'config', 'rate', 'archive')]
    summary = df_pcp.groupby(['model', 'config', 'rate'])[numeric_cols].mean().reset_index()

    # Compute prefix cache hit rate
    if 'prefix_cache_hits' in summary.columns and 'prefix_cache_queries' in summary.columns:
        summary['prefix_hit_rate_pct'] = np.where(
            summary['prefix_cache_queries'] > 0,
            summary['prefix_cache_hits'] / summary['prefix_cache_queries'] * 100,
            np.nan
        )

    # Compute external prefix cache hit rate
    if 'ext_prefix_hits' in summary.columns and 'ext_prefix_queries' in summary.columns:
        summary['ext_prefix_hit_rate_pct'] = np.where(
            summary['ext_prefix_queries'] > 0,
            summary['ext_prefix_hits'] / summary['ext_prefix_queries'] * 100,
            np.nan
        )

    # Convert disk bytes/s to MB/s
    if 'disk_read_bytes_s' in summary.columns:
        summary['disk_read_MB_s'] = summary['disk_read_bytes_s'] / 1e6
    if 'disk_write_bytes_s' in summary.columns:
        summary['disk_write_MB_s'] = summary['disk_write_bytes_s'] / 1e6

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def get_muted_colors(n):
    """Get n colors from seaborn muted palette."""
    return sns.color_palette("muted", n)


def plot_throughput_curves(df, output_file):
    """4-panel throughput vs rate curves, one per model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = get_muted_colors(len(CONFIGS))

    for i, model in enumerate(MODELS):
        ax = axes[i]
        model_data = df[df['model'] == model]

        for j, config in enumerate(CONFIGS):
            cfg_data = model_data[model_data['config'] == config].sort_values('rate')
            if cfg_data.empty:
                continue
            label = CONFIG_LABELS.get(config, config)
            ax.plot(cfg_data['rate'], cfg_data['throughput'],
                    marker='o', label=label, linewidth=2,
                    color=colors[j])

        ax.set_xlabel('Concurrency (requests)', fontsize=10)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=10)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Throughput vs Concurrency: llm-d v0.5.1',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_peak_throughput(peak_df, output_file):
    """Grouped bar chart of peak throughput by model and config."""
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(MODELS))
    n_configs = len(CONFIGS)
    width = 0.8 / n_configs
    colors = get_muted_colors(n_configs)

    for j, config in enumerate(CONFIGS):
        cfg_data = peak_df[peak_df['config'] == config]
        throughputs = []
        for model in MODELS:
            row = cfg_data[cfg_data['model'] == model]
            throughputs.append(row['throughput'].values[0] if len(row) > 0 else 0)

        offset = (j - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, throughputs, width,
                      label=CONFIG_LABELS.get(config, config),
                      color=colors[j])

        # Add value labels on bars
        for bar, val in zip(bars, throughputs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Peak Throughput (tokens/s)', fontsize=12)
    ax.set_title('Peak Throughput by Model and Configuration: llm-d v0.5.1',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_version_comparison(peak_df, output_file):
    """Grouped bar chart comparing v0.4.0, v0.5.0, v0.5.1 for no-offload and native-offload."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    versions = ['v0.4.0', 'v0.5.0', 'v0.5.1']
    colors = get_muted_colors(3)
    x = np.arange(len(MODELS))
    width = 0.25

    for panel_idx, cfg_label in enumerate(['no-offload', 'native-offload-20k']):
        ax = axes[panel_idx]

        # v0.4.0 values
        v040_key_suffix = 'no-offload' if cfg_label == 'no-offload' else 'native-offload'
        v040_vals = [V040_BASELINES.get((m, v040_key_suffix), np.nan) for m in MODELS]

        # v0.5.0 values
        v050_vals = [V050_BASELINES.get((m, cfg_label), np.nan) for m in MODELS]

        # v0.5.1 values
        cfg_data = peak_df[peak_df['config'] == cfg_label]
        v051_vals = []
        for model in MODELS:
            row = cfg_data[cfg_data['model'] == model]
            v051_vals.append(row['throughput'].values[0] if len(row) > 0 else np.nan)

        all_vals = [v040_vals, v050_vals, v051_vals]
        for vi, (version, vals, color) in enumerate(zip(versions, all_vals, colors)):
            # Replace nan with 0 for plotting
            plot_vals = [v if not np.isnan(v) else 0 for v in vals]
            bars = ax.bar(x + (vi - 1) * width, plot_vals, width,
                          label=version, color=color)
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Peak Throughput (tokens/s)', fontsize=11)
        title = 'No Offload Baseline' if cfg_label == 'no-offload' else 'Native Offload (20k blocks)'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Version Comparison: v0.4.0 vs v0.5.0 vs v0.5.1',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_latency(df, metric_col, ylabel, title_suffix, output_file):
    """Bar chart of median latency at rate=50 across configs and models."""
    # For 32B-AWQ, use rate=1 (it's the typical peak)
    rate50_data = df[df['rate'] == 50].copy()
    rate1_data = df[(df['rate'] == 1) & (df['model'] == 'Qwen3-32B-AWQ')].copy()
    plot_data = pd.concat([
        rate50_data[rate50_data['model'] != 'Qwen3-32B-AWQ'],
        rate1_data
    ])

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(MODELS))
    n_configs = len(CONFIGS)
    width = 0.8 / n_configs
    colors = get_muted_colors(n_configs)

    for j, config in enumerate(CONFIGS):
        cfg_data = plot_data[plot_data['config'] == config]
        latencies = []
        for model in MODELS:
            row = cfg_data[cfg_data['model'] == model]
            latencies.append(row[metric_col].values[0] if len(row) > 0 and metric_col in row.columns else 0)

        offset = (j - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, latencies, width,
               label=CONFIG_LABELS.get(config, config),
               color=colors[j])

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{title_suffix} at rate=50 (32B-AWQ at rate=1): llm-d v0.5.1',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_delta_heatmap(peak_df, output_file):
    """Heatmap of % delta vs no-offload baseline (magma colormap)."""
    # Build matrix: rows=configs (offload only), cols=models
    offload_configs = [c for c in CONFIGS if c != 'no-offload']

    matrix = []
    annot = []
    for config in offload_configs:
        row_vals = []
        row_annot = []
        for model in MODELS:
            no_off_row = peak_df[(peak_df['model'] == model) & (peak_df['config'] == 'no-offload')]
            cfg_row = peak_df[(peak_df['model'] == model) & (peak_df['config'] == config)]

            if len(no_off_row) > 0 and len(cfg_row) > 0:
                baseline = no_off_row['throughput'].values[0]
                offload = cfg_row['throughput'].values[0]
                if baseline > 0:
                    delta = (offload - baseline) / baseline * 100
                    row_vals.append(delta)
                    row_annot.append(f'{delta:+.1f}%')
                else:
                    row_vals.append(np.nan)
                    row_annot.append('N/A')
            else:
                row_vals.append(np.nan)
                row_annot.append('N/A')

        matrix.append(row_vals)
        annot.append(row_annot)

    df_heat = pd.DataFrame(matrix,
                           index=[CONFIG_LABELS.get(c, c) for c in offload_configs],
                           columns=MODELS)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(df_heat, annot=annot, fmt='', cmap='magma',
                center=0, vmin=-60, vmax=10,
                cbar_kws={'label': '% vs no-offload baseline'},
                ax=ax, linewidths=0.5)

    ax.set_title('Throughput % Delta vs No-Offload Baseline: llm-d v0.5.1',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_pcp_gpu_util(df_pcp_summary, peak_df, output_file):
    """GPU utilization at peak throughput rates by config and model."""
    if df_pcp_summary.empty or 'gpu_util_pct' not in df_pcp_summary.columns:
        print(f"  Skipping {output_file}: no GPU util data")
        return

    # Get peak rates
    peak_rates = peak_df.set_index(['model', 'config'])['rate'].to_dict()

    # Filter to peak rates
    rows = []
    for (model, config), rate in peak_rates.items():
        row = df_pcp_summary[(df_pcp_summary['model'] == model) &
                              (df_pcp_summary['config'] == config) &
                              (df_pcp_summary['rate'] == rate)]
        if len(row) > 0 and 'gpu_util_pct' in row.columns:
            rows.append({
                'model': model,
                'config': config,
                'gpu_util_pct': row['gpu_util_pct'].values[0]
            })

    if not rows:
        print(f"  Skipping {output_file}: no matched data")
        return

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(MODELS))
    n_configs = len(CONFIGS)
    width = 0.8 / n_configs
    colors = get_muted_colors(n_configs)

    for j, config in enumerate(CONFIGS):
        cfg_data = plot_df[plot_df['config'] == config]
        utils = []
        for model in MODELS:
            row = cfg_data[cfg_data['model'] == model]
            utils.append(row['gpu_util_pct'].values[0] if len(row) > 0 else 0)

        offset = (j - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, utils, width,
               label=CONFIG_LABELS.get(config, config), color=colors[j])

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization at Peak Throughput: llm-d v0.5.1',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_pcp_kvcache(df_pcp_summary, peak_df, output_file):
    """KV cache usage % at peak throughput."""
    if df_pcp_summary.empty or 'kv_cache_usage_pct' not in df_pcp_summary.columns:
        print(f"  Skipping {output_file}: no KV cache data")
        return

    peak_rates = peak_df.set_index(['model', 'config'])['rate'].to_dict()
    rows = []
    for (model, config), rate in peak_rates.items():
        row = df_pcp_summary[(df_pcp_summary['model'] == model) &
                              (df_pcp_summary['config'] == config) &
                              (df_pcp_summary['rate'] == rate)]
        if len(row) > 0 and 'kv_cache_usage_pct' in row.columns:
            rows.append({
                'model': model,
                'config': config,
                'kv_cache_usage_pct': row['kv_cache_usage_pct'].values[0]
            })

    if not rows:
        print(f"  Skipping {output_file}: no matched data")
        return

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(MODELS))
    n_configs = len(CONFIGS)
    width = 0.8 / n_configs
    colors = get_muted_colors(n_configs)

    for j, config in enumerate(CONFIGS):
        cfg_data = plot_df[plot_df['config'] == config]
        usages = []
        for model in MODELS:
            row = cfg_data[cfg_data['model'] == model]
            usages.append(row['kv_cache_usage_pct'].values[0] * 100
                         if len(row) > 0 else 0)  # fraction -> %

        offset = (j - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, usages, width,
               label=CONFIG_LABELS.get(config, config), color=colors[j])

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('KV Cache Usage (%)', fontsize=12)
    ax.set_title('GPU KV Cache Utilization at Peak Throughput: llm-d v0.5.1',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_pcp_external_hits(df_pcp_summary, peak_df, output_file):
    """External prefix cache hit rate by config and model at peak throughput."""
    if df_pcp_summary.empty:
        print(f"  Skipping {output_file}: no PCP data")
        return

    hit_col = 'ext_prefix_hit_rate_pct'
    if hit_col not in df_pcp_summary.columns:
        # Try internal prefix cache
        hit_col = 'prefix_hit_rate_pct'
        if hit_col not in df_pcp_summary.columns:
            print(f"  Skipping {output_file}: no prefix cache hit rate data")
            return

    peak_rates = peak_df.set_index(['model', 'config'])['rate'].to_dict()
    rows = []
    for (model, config), rate in peak_rates.items():
        row = df_pcp_summary[(df_pcp_summary['model'] == model) &
                              (df_pcp_summary['config'] == config) &
                              (df_pcp_summary['rate'] == rate)]
        if len(row) > 0 and hit_col in row.columns:
            rows.append({
                'model': model,
                'config': config,
                'hit_rate': row[hit_col].values[0]
            })

    if not rows:
        print(f"  Skipping {output_file}: no matched data")
        return

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(MODELS))
    n_configs = len(CONFIGS)
    width = 0.8 / n_configs
    colors = get_muted_colors(n_configs)

    for j, config in enumerate(CONFIGS):
        cfg_data = plot_df[plot_df['config'] == config]
        rates_vals = []
        for model in MODELS:
            row = cfg_data[cfg_data['model'] == model]
            rates_vals.append(row['hit_rate'].values[0] if len(row) > 0 else 0)

        offset = (j - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, rates_vals, width,
               label=CONFIG_LABELS.get(config, config), color=colors[j])

    title_metric = 'External Prefix Cache' if hit_col == 'ext_prefix_hit_rate_pct' else 'Prefix Cache'
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Hit Rate (%)', fontsize=12)
    ax.set_title(f'{title_metric} Hit Rate at Peak Throughput: llm-d v0.5.1',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_pcp_disk_io(df_pcp_summary, peak_df, output_file):
    """Disk read/write MB/s for fs-offload configs vs baseline."""
    if df_pcp_summary.empty:
        print(f"  Skipping {output_file}: no PCP data")
        return

    read_col = 'disk_read_MB_s'
    write_col = 'disk_write_MB_s'
    if read_col not in df_pcp_summary.columns and write_col not in df_pcp_summary.columns:
        print(f"  Skipping {output_file}: no disk I/O data")
        return

    fs_configs = ['no-offload', 'fs-offload', 'cpu+fs-offload-20k']
    peak_rates = peak_df.set_index(['model', 'config'])['rate'].to_dict()

    rows = []
    for (model, config), rate in peak_rates.items():
        if config not in fs_configs:
            continue
        row = df_pcp_summary[(df_pcp_summary['model'] == model) &
                              (df_pcp_summary['config'] == config) &
                              (df_pcp_summary['rate'] == rate)]
        if len(row) > 0:
            r = {
                'model': model,
                'config': config,
                'read_MB_s': row[read_col].values[0] if read_col in row.columns else 0,
                'write_MB_s': row[write_col].values[0] if write_col in row.columns else 0,
            }
            rows.append(r)

    if not rows:
        print(f"  Skipping {output_file}: no matched data")
        return

    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    x = np.arange(len(MODELS))
    n_fs_configs = len(fs_configs)
    width = 0.8 / n_fs_configs
    colors = get_muted_colors(n_fs_configs)

    for panel_idx, (io_col, io_label) in enumerate([('read_MB_s', 'Disk Read'), ('write_MB_s', 'Disk Write')]):
        ax = axes[panel_idx]
        for j, config in enumerate(fs_configs):
            cfg_data = plot_df[plot_df['config'] == config]
            vals = []
            for model in MODELS:
                row = cfg_data[cfg_data['model'] == model]
                vals.append(row[io_col].values[0] if len(row) > 0 else 0)

            offset = (j - n_fs_configs / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=CONFIG_LABELS.get(config, config), color=colors[j])

        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel(f'{io_label} (MB/s)', fontsize=11)
        ax.set_title(f'{io_label} MB/s at Peak Throughput', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Disk I/O: Filesystem Offload Configs: llm-d v0.5.1',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Summary CSV and console report
# ══════════════════════════════════════════════════════════════════════════════

def build_summary_csv(peak_df, df_pcp_summary):
    """Build and save summary CSV with all key metrics."""
    summary_rows = []

    for _, peak_row in peak_df.iterrows():
        model = peak_row['model']
        config = peak_row['config']
        rate = peak_row['rate']

        row = {
            'model': model,
            'config': config,
            'peak_throughput_tok_s': peak_row['throughput'],
            'optimal_rate': rate,
            'ttft_median_s': peak_row.get('ttft_median_s', np.nan),
            'itl_median_ms': peak_row.get('itl_median_ms', np.nan),
            'tpot_median_ms': peak_row.get('tpot_median_ms', np.nan),
        }

        # Delta vs v0.5.1 no-offload baseline
        no_off_row = peak_df[(peak_df['model'] == model) & (peak_df['config'] == 'no-offload')]
        if len(no_off_row) > 0:
            baseline = no_off_row['throughput'].values[0]
            if baseline > 0:
                row['delta_vs_no_offload_pct'] = (peak_row['throughput'] - baseline) / baseline * 100
            else:
                row['delta_vs_no_offload_pct'] = np.nan
        else:
            row['delta_vs_no_offload_pct'] = np.nan

        # Delta vs v0.5.0 baseline
        v050_key = (model, config)
        if v050_key in V050_BASELINES:
            v050_val = V050_BASELINES[v050_key]
            row['v050_peak_tok_s'] = v050_val
            row['delta_vs_v050_pct'] = (peak_row['throughput'] - v050_val) / v050_val * 100
        else:
            row['v050_peak_tok_s'] = np.nan
            row['delta_vs_v050_pct'] = np.nan

        # PCP metrics at peak rate
        if not df_pcp_summary.empty:
            pcp_row = df_pcp_summary[(df_pcp_summary['model'] == model) &
                                     (df_pcp_summary['config'] == config) &
                                     (df_pcp_summary['rate'] == rate)]
            if len(pcp_row) > 0:
                for col in ['gpu_util_pct', 'kv_cache_usage_pct', 'requests_running',
                            'requests_waiting', 'prefix_hit_rate_pct', 'ext_prefix_hit_rate_pct',
                            'disk_read_MB_s', 'disk_write_MB_s']:
                    if col in pcp_row.columns:
                        row[col] = pcp_row[col].values[0]

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    out_file = OUTPUT_DIR / 'v0.5.1_summary.csv'
    df_summary.to_csv(out_file, index=False)
    print(f"\nSaved summary CSV: {out_file}")
    return df_summary


def print_peak_throughput_table(peak_df):
    """Print formatted peak throughput table."""
    print("\n" + "=" * 90)
    print("PEAK THROUGHPUT RESULTS: llm-d v0.5.1")
    print("=" * 90)
    print(f"{'Model':<20} {'Config':<25} {'Peak (tok/s)':>12} {'Rate':>6} {'vs No-Offload':>14}")
    print("-" * 90)

    for model in MODELS:
        model_data = peak_df[peak_df['model'] == model]
        no_off_row = model_data[model_data['config'] == 'no-offload']
        baseline = no_off_row['throughput'].values[0] if len(no_off_row) > 0 else None

        for config in CONFIGS:
            row = model_data[model_data['config'] == config]
            if len(row) == 0:
                continue
            tput = row['throughput'].values[0]
            rate = row['rate'].values[0]

            if baseline and baseline > 0 and config != 'no-offload':
                delta = (tput - baseline) / baseline * 100
                delta_str = f'{delta:+.1f}%'
            else:
                delta_str = 'baseline'

            print(f"  {model:<18} {config:<25} {tput:>12.1f} {rate:>6} {delta_str:>14}")
        print()


def print_version_comparison(peak_df):
    """Print version comparison table."""
    print("=" * 90)
    print("VERSION COMPARISON: v0.4.0 vs v0.5.0 vs v0.5.1")
    print("=" * 90)
    print(f"{'Model':<20} {'Config':<22} {'v0.4.0':>8} {'v0.5.0':>8} {'v0.5.1':>8} {'v050→v051':>10}")
    print("-" * 90)

    compare_pairs = [
        ('no-offload', 'no-offload', 'no-offload'),
        ('native-offload-20k', 'native-offload', 'native-offload-20k'),
    ]

    for v051_cfg, v040_cfg, v050_cfg in compare_pairs:
        for model in MODELS:
            v040_val = V040_BASELINES.get((model, v040_cfg), np.nan)
            v050_val = V050_BASELINES.get((model, v050_cfg), np.nan)

            peak_row = peak_df[(peak_df['model'] == model) & (peak_df['config'] == v051_cfg)]
            v051_val = peak_row['throughput'].values[0] if len(peak_row) > 0 else np.nan

            if not np.isnan(v050_val) and not np.isnan(v051_val) and v050_val > 0:
                delta_str = f'{(v051_val - v050_val) / v050_val * 100:+.1f}%'
            else:
                delta_str = 'N/A'

            v040_str = f'{v040_val:.1f}' if not np.isnan(v040_val) else 'N/A'
            v050_str = f'{v050_val:.1f}' if not np.isnan(v050_val) else 'N/A'
            v051_str = f'{v051_val:.1f}' if not np.isnan(v051_val) else 'N/A'

            print(f"  {model:<18} {v051_cfg:<22} {v040_str:>8} {v050_str:>8} {v051_str:>8} {delta_str:>10}")
        print()


def print_pcp_summary(df_pcp_summary, peak_df):
    """Print PCP metric summary by config."""
    if df_pcp_summary.empty:
        print("\nNo PCP metrics available.")
        return

    print("=" * 90)
    print("PCP METRICS SUMMARY (at peak throughput rates)")
    print("=" * 90)

    peak_rates = peak_df.set_index(['model', 'config'])['rate'].to_dict()

    pcp_cols = {
        'gpu_util_pct': 'GPU Util %',
        'kv_cache_usage_pct': 'KV Cache %',
        'requests_running': 'Running Req',
        'requests_waiting': 'Waiting Req',
        'prefix_hit_rate_pct': 'Prefix Hit %',
        'ext_prefix_hit_rate_pct': 'Ext Prefix Hit %',
        'disk_read_MB_s': 'Disk Read MB/s',
        'disk_write_MB_s': 'Disk Write MB/s',
    }

    available_cols = [c for c in pcp_cols if c in df_pcp_summary.columns]
    if not available_cols:
        print("  No PCP metric columns found in summary")
        return

    header = f"{'Model':<18} {'Config':<22}" + "".join(f" {pcp_cols[c]:>16}" for c in available_cols)
    print(header)
    print("-" * len(header))

    for model in MODELS:
        for config in CONFIGS:
            rate = peak_rates.get((model, config))
            if rate is None:
                continue
            row = df_pcp_summary[(df_pcp_summary['model'] == model) &
                                  (df_pcp_summary['config'] == config) &
                                  (df_pcp_summary['rate'] == rate)]
            if len(row) == 0:
                continue

            vals = f"  {model:<18} {config:<22}"
            for col in available_cols:
                if col in row.columns:
                    val = row[col].values[0]
                    if col == 'kv_cache_usage_pct':
                        val = val * 100  # fraction to percent
                    if np.isnan(val):
                        vals += f" {'N/A':>16}"
                    else:
                        vals += f" {val:>16.2f}"
                else:
                    vals += f" {'N/A':>16}"
            print(vals)
        print()


def print_latency_rate50(df):
    """Print latency at rate=50 for all models and configs."""
    print("=" * 90)
    print("LATENCY AT RATE=50 (TTFT / ITL / TPOT)")
    print("=" * 90)
    print(f"{'Model':<18} {'Config':<22} {'TTFT (s)':>10} {'ITL (ms)':>10} {'TPOT (ms)':>10}")
    print("-" * 75)

    for model in MODELS:
        rate_val = 1 if model == 'Qwen3-32B-AWQ' else 50
        model_data = df[(df['model'] == model) & (df['rate'] == rate_val)]
        for config in CONFIGS:
            row = model_data[model_data['config'] == config]
            if len(row) == 0:
                continue
            ttft = row['ttft_median_s'].values[0]
            itl = row['itl_median_ms'].values[0]
            tpot = row['tpot_median_ms'].values[0]
            print(f"  {model:<16} {config:<22} {ttft:>10.3f} {itl:>10.1f} {tpot:>10.1f}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.chdir(Path(__file__).parent.parent)
    print(f"Working directory: {os.getcwd()}")

    # ── Step 1: Parse GuideLLM results ──────────────────────────────────────
    df = load_all_guidellm_results()

    if df.empty:
        print("ERROR: No GuideLLM results loaded. Check results directory.")
        sys.exit(1)

    # Filter to v0.5.1 configs
    v051_configs = df['config'].unique()
    print(f"Configs found: {sorted(v051_configs)}")
    print(f"Models found:  {sorted(df['model'].unique())}")
    print(f"Rates found:   {sorted(df['rate'].unique())}")

    # Save all throughput data
    throughput_csv = OUTPUT_DIR / 'v0.5.1_throughput_all.csv'
    df.to_csv(throughput_csv, index=False)
    print(f"Saved: {throughput_csv}")

    # Find peak throughput
    peak_df = find_peak_throughput(df)

    # ── Step 2: Extract PCP metrics ──────────────────────────────────────────
    df_pcp = extract_all_pcp_metrics(df)
    df_pcp_summary = compute_pcp_summary(df_pcp)

    if not df_pcp.empty:
        pcp_csv = OUTPUT_DIR / 'v0.5.1_pcp_metrics.csv'
        df_pcp.to_csv(pcp_csv, index=False)
        print(f"Saved: {pcp_csv}")

    # ── Step 3: Generate visualizations ─────────────────────────────────────
    print("\nGenerating visualizations...")

    plot_throughput_curves(df, OUTPUT_DIR / 'v0.5.1_throughput_curves.png')
    plot_peak_throughput(peak_df, OUTPUT_DIR / 'v0.5.1_peak_throughput.png')
    plot_version_comparison(peak_df, OUTPUT_DIR / 'v0.5.1_version_comparison.png')

    plot_latency(df, 'ttft_median_s', 'TTFT (seconds)', 'Time to First Token',
                 OUTPUT_DIR / 'v0.5.1_latency_ttft.png')
    plot_latency(df, 'itl_median_ms', 'ITL (milliseconds)', 'Inter-Token Latency',
                 OUTPUT_DIR / 'v0.5.1_latency_itl.png')

    plot_delta_heatmap(peak_df, OUTPUT_DIR / 'v0.5.1_delta_heatmap.png')

    if not df_pcp_summary.empty:
        plot_pcp_gpu_util(df_pcp_summary, peak_df, OUTPUT_DIR / 'v0.5.1_pcp_gpu_util.png')
        plot_pcp_kvcache(df_pcp_summary, peak_df, OUTPUT_DIR / 'v0.5.1_pcp_kvcache.png')
        plot_pcp_external_hits(df_pcp_summary, peak_df, OUTPUT_DIR / 'v0.5.1_pcp_external_hits.png')
        plot_pcp_disk_io(df_pcp_summary, peak_df, OUTPUT_DIR / 'v0.5.1_pcp_disk_io.png')

    # ── Step 4: Save summary CSV ─────────────────────────────────────────────
    df_summary = build_summary_csv(peak_df, df_pcp_summary)

    # ── Step 5: Console report ───────────────────────────────────────────────
    print_peak_throughput_table(peak_df)
    print_version_comparison(peak_df)
    print_pcp_summary(df_pcp_summary, peak_df)
    print_latency_rate50(df)

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 90)


if __name__ == '__main__':
    main()
