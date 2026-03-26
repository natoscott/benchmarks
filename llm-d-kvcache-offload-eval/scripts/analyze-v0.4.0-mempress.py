#!/usr/bin/env python3
"""Analysis of v0.4.0-mempress benchmark results vs original v0.4.0 (gmu=0.9).

Parses guidellm-results.json.zst from both suites, extracts PCP metrics,
and generates visualisations for reporting.
"""

import csv as csv_mod
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ── Visualization standards (visualization-palette skill) ─────────────────────
sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

RESULTS_DIR = Path('results')
OUTPUT_DIR = Path('analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Dataset definitions ───────────────────────────────────────────────────────
MODELS = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
MODEL_LABELS = {
    'Qwen3-0.6B': '0.6B',
    'Qwen3-8B': '8B',
    'Qwen3-14B': '14B',
    'Qwen3-32B-AWQ': '32B-AWQ',
}

# mempress configs (5)
MEMPRESS_CONFIGS = ['no-offload', 'native-offload', 'lmcache-local', 'lmcache-valkey', 'llm-d-valkey']

# original 0.4.0 configs (5 that match mempress, excluding redis variants)
ORIG_CONFIGS = ['no-offload', 'native-offload', 'lmcache-local', 'lmcache-valkey', 'llm-d-valkey']

CONFIG_LABELS = {
    'no-offload':     'No Offload',
    'native-offload': 'Native Offload',
    'lmcache-local':  'LMCache Local',
    'lmcache-valkey': 'LMCache Valkey',
    'llm-d-valkey':   'llm-d Valkey',
}

RATES = [1, 50, 100, 150, 300, 400, 500, 650]

# gpu_memory_utilization per model for mempress
GMU_MEMPRESS = {
    'Qwen3-0.6B': 0.55,
    'Qwen3-8B': 0.65,
    'Qwen3-14B': 0.70,
    'Qwen3-32B-AWQ': 0.65,
}

# Known original gmu=0.9 peak throughput (from REPORT-v0.4.0.md)
ORIG_PEAK = {
    ('Qwen3-0.6B',    'no-offload'):     602.0,
    ('Qwen3-8B',      'no-offload'):     113.0,
    ('Qwen3-14B',     'no-offload'):      58.7,
    ('Qwen3-32B-AWQ', 'no-offload'):      49.2,
    ('Qwen3-0.6B',    'native-offload'): 426.8,
    ('Qwen3-8B',      'native-offload'):  71.8,
    ('Qwen3-14B',     'native-offload'):  59.0,
    ('Qwen3-32B-AWQ', 'native-offload'):  48.7,
    ('Qwen3-0.6B',    'lmcache-local'):  520.4,
    ('Qwen3-8B',      'lmcache-local'):  106.6,
    ('Qwen3-14B',     'lmcache-local'):   65.6,
    ('Qwen3-32B-AWQ', 'lmcache-local'):   43.0,
    ('Qwen3-0.6B',    'lmcache-valkey'): 523.9,
    ('Qwen3-8B',      'lmcache-valkey'): 105.7,
    ('Qwen3-14B',     'lmcache-valkey'):  66.3,
    ('Qwen3-32B-AWQ', 'lmcache-valkey'):  43.0,
    ('Qwen3-0.6B',    'llm-d-valkey'):   592.9,
    ('Qwen3-8B',      'llm-d-valkey'):   113.4,
    ('Qwen3-14B',     'llm-d-valkey'):    64.5,
    ('Qwen3-32B-AWQ', 'llm-d-valkey'):    49.2,
}


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Parse GuideLLM results
# ══════════════════════════════════════════════════════════════════════════════

def parse_dir_name(dir_name):
    """Parse benchmark directory name into components.

    Format: 1x2xL40S_upstream-llm-d-0.4.0[-mempress]_Qwen3-0.6B_no-offload_replica1_rate50
    Config name is everything between model (parts[2]) and replicas (parts[-2]).
    """
    parts = dir_name.split('_')
    hardware = parts[0]
    software = parts[1]
    model = parts[2]
    replicas = int(parts[-2].replace('replica', ''))
    rate = int(parts[-1].replace('rate', ''))
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

        benchmarks = data.get('benchmarks', [])
        if not benchmarks:
            return None

        benchmark = benchmarks[0]
        metrics = benchmark.get('metrics', {})
        duration = benchmark.get('duration', 0)

        # Throughput: total_sum output tokens / duration
        throughput = 0.0
        otc = metrics.get('output_token_count', {})
        if otc and duration > 0:
            total_tokens = otc.get('successful', {}).get('total_sum', 0)
            if total_tokens == 0:
                total_tokens = otc.get('successful', {}).get('sum', 0)
            throughput = total_tokens / duration

        # Latency metrics
        ttft_ms = metrics.get('time_to_first_token_ms', {})
        ttft_median = ttft_ms.get('successful', {}).get('median', 0.0)

        itl_ms = metrics.get('inter_token_latency_ms', {})
        itl_median = itl_ms.get('successful', {}).get('median', 0.0)

        tpot_ms = metrics.get('time_per_output_token_ms', {})
        tpot_median = tpot_ms.get('successful', {}).get('median', 0.0)

        request_totals = metrics.get('request_totals', {})
        completed = request_totals.get('completed', 0) or request_totals.get('successful', 0)

        suite = 'mempress' if 'mempress' in software else 'original'

        return {
            'suite': suite,
            'software': software,
            'model': model,
            'config': config,
            'rate': rate,
            'throughput': throughput,
            'duration': duration,
            'completed_requests': completed,
            'ttft_median_ms': ttft_median,
            'itl_median_ms': itl_median,
            'tpot_median_ms': tpot_median,
        }

    except Exception as e:
        print(f"  Error parsing {result_file}: {e}")
        return None


def load_guidellm_results():
    """Load all mempress and matching original v0.4.0 guidellm results."""
    print("Loading GuideLLM results...")

    records = []
    patterns = [
        '1x2xL40S_upstream-llm-d-0.4.0-mempress_*/guidellm-results.json.zst',
        '1x2xL40S_upstream-llm-d-0.4.0_*/guidellm-results.json.zst',
    ]

    for pattern in patterns:
        files = sorted(RESULTS_DIR.glob(pattern))
        print(f"  Pattern '{pattern}': {len(files)} files")
        for f in files:
            row = extract_guidellm_metrics(f)
            if row and row['config'] in MEMPRESS_CONFIGS:
                records.append(row)

    df = pd.DataFrame(records)
    print(f"  Successfully parsed {len(df)} records "
          f"({(df['suite']=='mempress').sum()} mempress, "
          f"{(df['suite']=='original').sum()} original)")
    return df


def find_peak_throughput(df):
    """Find peak throughput row for each suite+model+config combination.

    For models where peak genuinely occurs at higher concurrency (0.6B, 8B, 14B),
    rate=1 is a low-sample warm-up run.  For large models (32B-AWQ) where the
    GPU is saturated at concurrency=1, rate=1 may be the true peak.

    Strategy:
    - For rate=1 to be accepted as the peak, its throughput must be plausibly
      consistent with higher-rate results.  If it exceeds the rate=50 value by
      more than 2× it is likely a warm-up / prefix-cache artefact (too few
      requests, multi-turn caching), and we prefer the rate>1 peak instead.
    - If rate=1 is within 2× of the next-highest rate, accept it as a genuine
      low-concurrency peak (typical for large memory-bound models like 32B-AWQ).
    """
    peak_rows = []
    for (suite, model, config), group in df.groupby(['suite', 'model', 'config']):
        if group.empty:
            continue

        rate1 = group[group['rate'] == 1]
        rate1_tp = rate1.iloc[0]['throughput'] if not rate1.empty else 0

        higher = group[group['rate'] > 1]
        higher_max_tp = higher['throughput'].max() if not higher.empty else 0

        if rate1_tp > 0 and higher_max_tp > 0:
            # Determine whether rate=1 is a genuine peak or a prefix-cache artefact.
            # Artefact signature: at concurrency=1 with 10 sample requests and
            # shared 10K-token prefix + 5 turns, cache hits inflate throughput.
            # Observed: 0.6B & 14B mempress rate=1 = 186.7 tok/s (both identical,
            # physically implausible for 14B single-stream: expected ~40-65 tok/s).
            #
            # For genuine low-concurrency peaks (32B-AWQ):
            #   no-offload rate=1/rate=50 ≈ 1.7×
            #   lmcache rate=1/rate=50 ≈ 2.6× (model+lmcache overhead kills concurrency)
            #
            # For artefacts (14B mempress):
            #   no-offload rate=1/rate=50 ≈ 2.8×
            #
            # The known original peak for each (model, config) provides a sanity
            # bound: if rate=1 in mempress far exceeds the known original rate=1,
            # it is an artefact.  Use ORIG_PEAK to validate.
            orig_ref = ORIG_PEAK.get((model, config))
            rate1_is_artefact = False
            if orig_ref is not None:
                # If rate=1 exceeds the known original PEAK (not just rate=1) by
                # more than 2×, it is unphysical → artefact
                if rate1_tp > orig_ref * 2.0:
                    rate1_is_artefact = True
            else:
                # No reference: fall back to ratio heuristic
                if rate1_tp > higher_max_tp * 2.5:
                    rate1_is_artefact = True

            if rate1_is_artefact:
                idx = higher['throughput'].idxmax()
            else:
                idx = group['throughput'].idxmax()
        elif higher.empty:
            idx = group['throughput'].idxmax()
        else:
            idx = higher['throughput'].idxmax()

        peak_rows.append(group.loc[idx].copy())

    return pd.DataFrame(peak_rows).reset_index(drop=True)


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
    """Run pmrep and return mean values per metric, summing instances."""
    cmd = ['pmrep', '-a', archive_base, '-o', 'csv', '-z', '-t', '10s'] + metrics
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not result.stdout.strip():
            return None

        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return None

        reader = csv_mod.reader([lines[0]])
        raw_headers = next(reader)

        metric_col_map = defaultdict(list)
        for col_idx, col_header in enumerate(raw_headers):
            if col_idx == 0:
                continue
            base = col_header.split('-')[0].strip()
            metric_col_map[base].append(col_idx)

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
                    pass

        if not data_by_col:
            return None

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
                result_means[metric] = sum(col_means)

        return result_means if result_means else None

    except subprocess.TimeoutExpired:
        print(f"    pmrep timeout for {archive_base}")
        return None
    except Exception as e:
        print(f"    pmrep error: {e}")
        return None


def extract_pcp_metrics_for_run(run_dir):
    """Extract vLLM KV cache and GPU metrics from a single run."""
    archive_base = find_archive_base(run_dir)
    if not archive_base:
        return None

    result = {}

    # GPU utilization
    gpu_metrics = [
        'openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL',
        'openmetrics.dcgm.DCGM_FI_DEV_MEM_COPY_UTIL',
    ]
    gpu_data = run_pmrep(archive_base, gpu_metrics)
    if gpu_data:
        result.update(gpu_data)
    if 'openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL' not in result:
        nvidia_data = run_pmrep(archive_base, ['nvidia.gpuactive'])
        if nvidia_data:
            result.update(nvidia_data)

    # vLLM metrics
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

    return result if result else None


def extract_all_pcp_metrics(df_peak):
    """Extract PCP metrics for all mempress peak-rate runs."""
    print("\nExtracting PCP metrics from archives (mempress only)...")

    pcp_records = []
    dirs = sorted(RESULTS_DIR.glob('1x2xL40S_upstream-llm-d-0.4.0-mempress_*/'))

    for run_dir in dirs:
        dir_name = run_dir.name
        try:
            _, software, model, config, replicas, rate = parse_dir_name(dir_name)
        except Exception:
            continue
        if config not in MEMPRESS_CONFIGS:
            continue

        # Only extract for peak rates (reduces runtime)
        peak_rate_rows = df_peak[
            (df_peak['suite'] == 'mempress') &
            (df_peak['model'] == model) &
            (df_peak['config'] == config)
        ]
        if peak_rate_rows.empty:
            continue
        peak_rate = int(peak_rate_rows.iloc[0]['rate'])
        if rate != peak_rate:
            continue

        print(f"  mempress {model} / {config} / rate={rate} ...", end=' ', flush=True)
        pcp_data = extract_pcp_metrics_for_run(run_dir)

        if pcp_data:
            row = {'suite': 'mempress', 'model': model, 'config': config, 'rate': rate}
            row.update(pcp_data)
            pcp_records.append(row)
            print("OK")
        else:
            print("no archive")

    # Also extract for original no-offload peak rates for comparison
    print("\nExtracting PCP metrics from archives (original no-offload)...")
    orig_dirs = sorted(RESULTS_DIR.glob('1x2xL40S_upstream-llm-d-0.4.0_*/'))
    for run_dir in orig_dirs:
        dir_name = run_dir.name
        try:
            _, software, model, config, replicas, rate = parse_dir_name(dir_name)
        except Exception:
            continue
        if config != 'no-offload':
            continue

        peak_rate_rows = df_peak[
            (df_peak['suite'] == 'original') &
            (df_peak['model'] == model) &
            (df_peak['config'] == 'no-offload')
        ]
        if peak_rate_rows.empty:
            continue
        peak_rate = int(peak_rate_rows.iloc[0]['rate'])
        if rate != peak_rate:
            continue

        print(f"  original {model} / no-offload / rate={rate} ...", end=' ', flush=True)
        pcp_data = extract_pcp_metrics_for_run(run_dir)
        if pcp_data:
            row = {'suite': 'original', 'model': model, 'config': config, 'rate': rate}
            row.update(pcp_data)
            pcp_records.append(row)
            print("OK")
        else:
            print("no archive")

    if pcp_records:
        df_pcp = pd.DataFrame(pcp_records)
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
        }
        df_pcp = df_pcp.rename(columns={k: v for k, v in rename_map.items() if k in df_pcp.columns})
        print(f"  Extracted PCP metrics for {len(df_pcp)} runs")
        return df_pcp
    else:
        print("  WARNING: No PCP metrics extracted")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Print analysis tables
# ══════════════════════════════════════════════════════════════════════════════

def print_peak_throughput_table(df_peak):
    """Print peak throughput tables for mempress and comparison."""
    print("\n" + "=" * 90)
    print("MEMPRESS PEAK THROUGHPUT (gmu reduced per model)")
    print("=" * 90)
    print(f"\n{'Config':<20} {'0.6B':>10} {'8B':>10} {'14B':>10} {'32B-AWQ':>10}")
    print("-" * 60)

    mempress = df_peak[df_peak['suite'] == 'mempress']

    for config in MEMPRESS_CONFIGS:
        row_str = f"{config:<20}"
        for model in MODELS:
            rows = mempress[(mempress['model'] == model) & (mempress['config'] == config)]
            if not rows.empty:
                t = rows.iloc[0]['throughput']
                row_str += f" {t:>9.1f}"
            else:
                row_str += f" {'N/A':>9}"
        print(row_str)

    print("\n" + "=" * 90)
    print("MEMPRESS % DELTA VS MEMPRESS NO-OFFLOAD BASELINE")
    print("=" * 90)
    print(f"\n{'Config':<20} {'0.6B':>12} {'8B':>12} {'14B':>12} {'32B-AWQ':>12}")
    print("-" * 68)

    for config in MEMPRESS_CONFIGS:
        if config == 'no-offload':
            continue
        row_str = f"{config:<20}"
        for model in MODELS:
            baseline_rows = mempress[(mempress['model'] == model) & (mempress['config'] == 'no-offload')]
            config_rows = mempress[(mempress['model'] == model) & (mempress['config'] == config)]
            if not baseline_rows.empty and not config_rows.empty:
                baseline = baseline_rows.iloc[0]['throughput']
                val = config_rows.iloc[0]['throughput']
                delta = (val - baseline) / baseline * 100
                row_str += f" {delta:>+11.1f}%"
            else:
                row_str += f" {'N/A':>12}"
        print(row_str)

    print("\n" + "=" * 90)
    print("COMPARISON: ORIGINAL gmu=0.9 vs MEMPRESS (delta vs respective no-offload baseline)")
    print("=" * 90)

    for model in MODELS:
        print(f"\n  {model} (mempress gmu={GMU_MEMPRESS[model]}):")
        print(f"  {'Config':<20} {'orig delta':>12} {'mempress delta':>16} {'change (pp)':>12}")
        print(f"  {'-'*62}")
        baseline_orig = ORIG_PEAK.get((model, 'no-offload'), None)
        baseline_mp_rows = mempress[(mempress['model'] == model) & (mempress['config'] == 'no-offload')]
        baseline_mp = baseline_mp_rows.iloc[0]['throughput'] if not baseline_mp_rows.empty else None

        for config in MEMPRESS_CONFIGS:
            if config == 'no-offload':
                orig_t = ORIG_PEAK.get((model, 'no-offload'), None)
                mp_rows = mempress[(mempress['model'] == model) & (mempress['config'] == config)]
                mp_t = mp_rows.iloc[0]['throughput'] if not mp_rows.empty else None
                orig_str = f"{orig_t:.1f}" if orig_t else "N/A"
                mp_str = f"{mp_t:.1f}" if mp_t else "N/A"
                print(f"  {config:<20} {orig_str:>12} {mp_str:>16}")
                continue

            orig_t = ORIG_PEAK.get((model, config), None)
            mp_rows = mempress[(mempress['model'] == model) & (mempress['config'] == config)]
            mp_t = mp_rows.iloc[0]['throughput'] if not mp_rows.empty else None

            if orig_t is not None and baseline_orig:
                orig_delta = (orig_t - baseline_orig) / baseline_orig * 100
                orig_str = f"{orig_delta:+.1f}%"
            else:
                orig_str = "N/A"

            if mp_t is not None and baseline_mp:
                mp_delta = (mp_t - baseline_mp) / baseline_mp * 100
                mp_str = f"{mp_delta:+.1f}%"
            else:
                mp_str = "N/A"

            if orig_t and baseline_orig and mp_t and baseline_mp:
                orig_pp = (orig_t - baseline_orig) / baseline_orig * 100
                mp_pp = (mp_t - baseline_mp) / baseline_mp * 100
                change = mp_pp - orig_pp
                change_str = f"{change:+.1f} pp"
            else:
                change_str = "N/A"

            print(f"  {config:<20} {orig_str:>12} {mp_str:>16} {change_str:>12}")


def print_pcp_table(df_pcp):
    """Print PCP KV cache usage table."""
    if df_pcp.empty:
        print("\nNo PCP data available.")
        return

    print("\n" + "=" * 90)
    print("PCP: GPU KV CACHE USAGE % AT PEAK RATE")
    print("=" * 90)

    if 'kv_cache_usage_pct' not in df_pcp.columns:
        print("  kv_cache_usage_pct not available in PCP data")
        return

    for suite in ['original', 'mempress']:
        gmu_label = "gmu=0.9" if suite == 'original' else "gmu=mempress"
        print(f"\n  Suite: {suite} ({gmu_label})")
        print(f"  {'Config':<20} {'0.6B':>10} {'8B':>10} {'14B':>10} {'32B-AWQ':>10}")
        print(f"  {'-'*52}")
        suite_data = df_pcp[df_pcp['suite'] == suite]
        configs_to_show = ['no-offload'] if suite == 'original' else MEMPRESS_CONFIGS
        for config in configs_to_show:
            row_str = f"  {config:<20}"
            for model in MODELS:
                rows = suite_data[(suite_data['model'] == model) & (suite_data['config'] == config)]
                if not rows.empty and 'kv_cache_usage_pct' in rows.columns:
                    v = rows.iloc[0]['kv_cache_usage_pct']
                    if pd.notna(v):
                        row_str += f" {v*100:>9.1f}%"
                    else:
                        row_str += f" {'N/A':>10}"
                else:
                    row_str += f" {'N/A':>10}"
            print(row_str)


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Visualisations
# ══════════════════════════════════════════════════════════════════════════════

def plot_peak_throughput_grouped(df_peak):
    """Grouped bar chart: mempress vs original peak throughput for all configs/models."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=False)
    fig.suptitle('v0.4.0 Peak Throughput: Original (gmu=0.9) vs Memory Pressure',
                 fontsize=13, fontweight='bold', y=1.01)

    palette = sns.color_palette("muted")
    colors = {
        ('original', 'no-offload'):     palette[0],
        ('mempress', 'no-offload'):      palette[1],
        ('mempress', 'native-offload'):  palette[2],
        ('mempress', 'lmcache-local'):   palette[3],
        ('mempress', 'lmcache-valkey'):  palette[4],
        ('mempress', 'llm-d-valkey'):    palette[5],
    }

    bar_groups = [
        ('original', 'no-offload', 'Orig no-offload\n(gmu=0.9)'),
        ('mempress', 'no-offload', 'MP no-offload\n(reduced gmu)'),
        ('mempress', 'native-offload', 'MP native\noffload'),
        ('mempress', 'lmcache-local', 'MP lmcache\nlocal'),
        ('mempress', 'lmcache-valkey', 'MP lmcache\nvalkey'),
        ('mempress', 'llm-d-valkey', 'MP llm-d\nvalkey'),
    ]

    mempress = df_peak[df_peak['suite'] == 'mempress']
    original = df_peak[df_peak['suite'] == 'original']

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        x = np.arange(len(bar_groups))
        bar_h = 0.6

        vals = []
        bar_colors = []
        for suite, config, label in bar_groups:
            if suite == 'original':
                t = ORIG_PEAK.get((model, config), 0)
            else:
                rows = mempress[(mempress['model'] == model) & (mempress['config'] == config)]
                t = rows.iloc[0]['throughput'] if not rows.empty else 0
            vals.append(t)
            bar_colors.append(colors.get((suite, config), palette[0]))

        bars = ax.barh(x, vals, bar_h, color=bar_colors, edgecolor='white', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:.0f}', va='center', ha='left', fontsize=8)

        ax.set_yticks(x)
        ax.set_yticklabels([g[2] for g in bar_groups], fontsize=8)
        ax.set_xlabel('Output tokens/s', fontsize=9)
        gmu_str = f"gmu={GMU_MEMPRESS[model]}"
        ax.set_title(f'{MODEL_LABELS[model]}\n({gmu_str})', fontsize=10, fontweight='bold')
        ax.set_xlim(0, max(v for v in vals if v > 0) * 1.15 if any(v > 0 for v in vals) else 100)

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'v0.4.0-mempress_peak_throughput.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


def plot_delta_heatmap(df_peak):
    """Heatmap: % delta vs no-offload baseline, side by side original and mempress."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('% Delta vs No-Offload Baseline: Original (gmu=0.9) vs Memory Pressure',
                 fontsize=13, fontweight='bold')

    configs_offload = ['native-offload', 'lmcache-local', 'lmcache-valkey', 'llm-d-valkey']
    model_labels = [MODEL_LABELS[m] for m in MODELS]

    for ax_idx, (suite_label, suite_key) in enumerate([
        ('Original (gmu=0.9)', 'original'),
        ('Memory Pressure (reduced gmu)', 'mempress'),
    ]):
        ax = axes[ax_idx]
        data_matrix = np.full((len(configs_offload), len(MODELS)), np.nan)

        suite_df = df_peak[df_peak['suite'] == suite_key]

        for row_i, config in enumerate(configs_offload):
            for col_j, model in enumerate(MODELS):
                if suite_key == 'original':
                    baseline = ORIG_PEAK.get((model, 'no-offload'))
                    val = ORIG_PEAK.get((model, config))
                else:
                    bl_rows = suite_df[(suite_df['model'] == model) & (suite_df['config'] == 'no-offload')]
                    v_rows = suite_df[(suite_df['model'] == model) & (suite_df['config'] == config)]
                    baseline = bl_rows.iloc[0]['throughput'] if not bl_rows.empty else None
                    val = v_rows.iloc[0]['throughput'] if not v_rows.empty else None

                if baseline and val:
                    data_matrix[row_i, col_j] = (val - baseline) / baseline * 100

        # Clip for colour scale
        vmax = max(abs(np.nanmin(data_matrix)), abs(np.nanmax(data_matrix)), 5)
        vmax = min(vmax, 50)

        sns.heatmap(
            data_matrix,
            ax=ax,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            xticklabels=model_labels,
            yticklabels=[CONFIG_LABELS[c] for c in configs_offload],
            cbar_kws={'label': '% vs no-offload baseline'},
            linewidths=0.5,
        )
        ax.set_title(suite_label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Configuration')

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'v0.4.0-mempress_delta_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_throughput_curves(df):
    """4-panel throughput vs rate curves: orig no-offload vs mempress no-offload vs mempress offload configs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    fig.suptitle('Throughput vs Request Rate: Original No-Offload vs Memory Pressure Configs',
                 fontsize=13, fontweight='bold')

    palette = sns.color_palette("muted")
    line_styles = {
        ('original', 'no-offload'):     {'color': palette[0], 'ls': '--', 'lw': 2, 'marker': 'o', 'label': 'Orig no-offload (gmu=0.9)'},
        ('mempress', 'no-offload'):      {'color': palette[1], 'ls': '-',  'lw': 2, 'marker': 's', 'label': 'MP no-offload (reduced gmu)'},
        ('mempress', 'native-offload'):  {'color': palette[2], 'ls': '-',  'lw': 2, 'marker': '^', 'label': 'MP native offload'},
        ('mempress', 'lmcache-local'):   {'color': palette[3], 'ls': '-',  'lw': 1.5, 'marker': 'D', 'label': 'MP lmcache-local'},
        ('mempress', 'lmcache-valkey'):  {'color': palette[4], 'ls': '-',  'lw': 1.5, 'marker': 'v', 'label': 'MP lmcache-valkey'},
        ('mempress', 'llm-d-valkey'):    {'color': palette[5], 'ls': '-',  'lw': 2, 'marker': 'P', 'label': 'MP llm-d-valkey'},
    }

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        gmu_str = f"gmu={GMU_MEMPRESS[model]}"
        ax.set_title(f'{MODEL_LABELS[model]} ({gmu_str})', fontsize=11, fontweight='bold')

        for (suite, config), style in line_styles.items():
            subset = df[(df['suite'] == suite) & (df['model'] == model) & (df['config'] == config)]
            if subset.empty:
                continue
            subset_sorted = subset.sort_values('rate')
            rates = subset_sorted['rate'].values
            throughputs = subset_sorted['throughput'].values
            ax.plot(rates, throughputs, **{k: v for k, v in style.items() if k != 'label'},
                    label=style['label'], markersize=5)

        ax.set_xlabel('Request rate (concurrency)', fontsize=9)
        ax.set_ylabel('Output tokens/s', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'v0.4.0-mempress_throughput_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_pcp_kvcache(df_pcp):
    """Bar chart: GPU KV cache usage % at peak rate, original no-offload vs mempress configs."""
    if df_pcp.empty or 'kv_cache_usage_pct' not in df_pcp.columns:
        print("  Skipping PCP KV cache plot: no data")
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=False)
    fig.suptitle('GPU KV Cache Usage at Peak Rate: Original (gmu=0.9) vs Memory Pressure',
                 fontsize=13, fontweight='bold', y=1.01)

    palette = sns.color_palette("muted")

    bar_groups = [
        ('original', 'no-offload', 'Orig\nno-offload'),
        ('mempress', 'no-offload', 'MP\nno-offload'),
        ('mempress', 'native-offload', 'MP native\noffload'),
        ('mempress', 'lmcache-local', 'MP lmcache\nlocal'),
        ('mempress', 'lmcache-valkey', 'MP lmcache\nvalkey'),
        ('mempress', 'llm-d-valkey', 'MP llm-d\nvalkey'),
    ]

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        labels = []
        vals = []
        colors = []

        for i, (suite, config, label) in enumerate(bar_groups):
            rows = df_pcp[(df_pcp['suite'] == suite) & (df_pcp['model'] == model) & (df_pcp['config'] == config)]
            if not rows.empty and 'kv_cache_usage_pct' in rows.columns:
                v = rows.iloc[0]['kv_cache_usage_pct']
                if pd.notna(v):
                    vals.append(v * 100)
                    labels.append(label)
                    colors.append(palette[i % len(palette)])

        if not vals:
            ax.text(0.5, 0.5, 'No PCP data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(MODEL_LABELS[model], fontsize=10)
            continue

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=0)
        ax.set_ylabel('KV Cache Usage (%)', fontsize=9)
        ax.set_ylim(0, 105)
        ax.axhline(80, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='80% target')
        gmu_str = f"gmu={GMU_MEMPRESS[model]}"
        ax.set_title(f'{MODEL_LABELS[model]}\n({gmu_str})', fontsize=10, fontweight='bold')
        if ax_idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'v0.4.0-mempress_pcp_kvcache.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def print_all_rates_table(df):
    """Print full throughput-vs-rate table for all mempress runs (diagnostic)."""
    print("\n" + "=" * 90)
    print("ALL RATES - MEMPRESS THROUGHPUT (diagnostic)")
    print("=" * 90)
    mempress = df[df['suite'] == 'mempress']
    for model in MODELS:
        print(f"\n  {model}:")
        print(f"  {'Rate':>6}", end='')
        for config in MEMPRESS_CONFIGS:
            print(f"  {config:>16}", end='')
        print()
        print(f"  {'-'*100}")
        for rate in RATES:
            print(f"  {rate:>6}", end='')
            for config in MEMPRESS_CONFIGS:
                rows = mempress[(mempress['model'] == model) & (mempress['config'] == config) & (mempress['rate'] == rate)]
                if not rows.empty:
                    t = rows.iloc[0]['throughput']
                    completed = rows.iloc[0]['completed_requests']
                    print(f"  {t:>12.1f} ({completed:>3.0f})", end='')
                else:
                    print(f"  {'N/A':>18}", end='')
            print()


def main():
    # Load guidellm results
    df = load_guidellm_results()
    if df.empty:
        print("ERROR: No results loaded. Check results directory.")
        sys.exit(1)

    # Print all rates for diagnostics
    print_all_rates_table(df)

    # Find peak throughput
    df_peak = find_peak_throughput(df)
    print(f"\nPeak throughput entries: {len(df_peak)}")

    # Print analysis tables
    print_peak_throughput_table(df_peak)

    # Save to CSV
    df_peak.to_csv(OUTPUT_DIR / 'v0.4.0-mempress_peak_throughput.csv', index=False)
    df.to_csv(OUTPUT_DIR / 'v0.4.0-mempress_all_throughput.csv', index=False)
    print(f"\nSaved CSVs to {OUTPUT_DIR}/")

    # Extract PCP metrics
    df_pcp = extract_all_pcp_metrics(df_peak)
    if not df_pcp.empty:
        df_pcp.to_csv(OUTPUT_DIR / 'v0.4.0-mempress_pcp_metrics.csv', index=False)
        print_pcp_table(df_pcp)
    else:
        print("No PCP metrics extracted - skipping PCP visualisation")

    # Generate visualisations
    print("\nGenerating visualisations...")
    plot_peak_throughput_grouped(df_peak)
    plot_delta_heatmap(df_peak)
    plot_throughput_curves(df)
    plot_pcp_kvcache(df_pcp)

    print("\nDone.")


if __name__ == '__main__':
    main()
