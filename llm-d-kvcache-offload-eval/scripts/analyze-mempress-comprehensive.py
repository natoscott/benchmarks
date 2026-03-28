#!/usr/bin/env python3
"""Comprehensive memory-pressure benchmark analysis.

Analyses all v0.4.0-mempress (192 runs) and v0.5.1-mempress (128 runs) data.
Generates all visualisations requested in the task spec, including the
previously under-analysed external KV cache hit rate metrics.

Outputs to analysis/:
  v0.4.0-mempress_peak_throughput_v2.png
  v0.4.0-mempress_external_cache_hits.png
  v0.4.0-mempress_gpu_kvcache_util.png
  v0.5.1-mempress_peak_throughput.png
  v0.5.1-mempress_delta_heatmap.png
  v0.5.1-mempress_throughput_curves.png
  v0.5.1-mempress_external_cache_hits.png
  v0.5.1-mempress_gpu_kvcache.png
  v0.5.1-mempress_vs_original.png
  mempress_crossversion_native_offload.png
  v0.5.1-mempress_peak_throughput.csv
  v0.4.0-mempress_native20k_peak.csv
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

MODELS = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B-AWQ']
MODEL_LABELS = {
    'Qwen3-0.6B': '0.6B',
    'Qwen3-8B': '8B',
    'Qwen3-14B': '14B',
    'Qwen3-32B-AWQ': '32B-AWQ',
}
RATES = [1, 50, 100, 150, 300, 400, 500, 650]

GMU_MEMPRESS = {
    'Qwen3-0.6B': 0.55,
    'Qwen3-8B': 0.65,
    'Qwen3-14B': 0.70,
    'Qwen3-32B-AWQ': 0.65,
}

# v0.4.0 configs (old: without 20k; new: including 20k)
V040_CONFIGS_OLD = ['no-offload', 'native-offload', 'lmcache-local', 'lmcache-valkey', 'llm-d-valkey']
V040_CONFIGS_ALL = ['no-offload', 'native-offload', 'native-offload-20k', 'lmcache-local', 'lmcache-valkey', 'llm-d-valkey']

# v0.5.1-mempress configs
V051_CONFIGS = ['no-offload', 'native-offload-20k', 'fs-offload', 'cpu+fs-offload-20k']

CONFIG_LABELS = {
    'no-offload':         'No Offload',
    'native-offload':     'Native Offload (10k)',
    'native-offload-20k': 'Native Offload (20k)',
    'lmcache-local':      'LMCache Local',
    'lmcache-valkey':     'LMCache Valkey',
    'llm-d-valkey':       'llm-d Valkey',
    'fs-offload':         'FS Offload',
    'cpu+fs-offload-20k': 'CPU+FS Offload (20k)',
}

# Known v0.4.0 original (gmu=0.9) peak throughput for reference
ORIG_040_PEAK = {
    ('Qwen3-0.6B',    'no-offload'):      602.0,
    ('Qwen3-8B',      'no-offload'):      113.0,
    ('Qwen3-14B',     'no-offload'):       58.7,
    ('Qwen3-32B-AWQ', 'no-offload'):       49.2,
    ('Qwen3-0.6B',    'native-offload'):  426.8,
    ('Qwen3-8B',      'native-offload'):   71.8,
    ('Qwen3-14B',     'native-offload'):   59.0,
    ('Qwen3-32B-AWQ', 'native-offload'):   48.7,
    ('Qwen3-0.6B',    'native-offload-20k'): 426.8,  # same as native-offload for comparison
    ('Qwen3-8B',      'native-offload-20k'):  71.8,
    ('Qwen3-14B',     'native-offload-20k'):  59.0,
    ('Qwen3-32B-AWQ', 'native-offload-20k'):  48.7,
    ('Qwen3-0.6B',    'lmcache-local'):   520.4,
    ('Qwen3-8B',      'lmcache-local'):   106.6,
    ('Qwen3-14B',     'lmcache-local'):    65.6,
    ('Qwen3-32B-AWQ', 'lmcache-local'):    43.0,
    ('Qwen3-0.6B',    'lmcache-valkey'):  523.9,
    ('Qwen3-8B',      'lmcache-valkey'):  105.7,
    ('Qwen3-14B',     'lmcache-valkey'):   66.3,
    ('Qwen3-32B-AWQ', 'lmcache-valkey'):   43.0,
    ('Qwen3-0.6B',    'llm-d-valkey'):    592.9,
    ('Qwen3-8B',      'llm-d-valkey'):    113.4,
    ('Qwen3-14B',     'llm-d-valkey'):     64.5,
    ('Qwen3-32B-AWQ', 'llm-d-valkey'):     49.2,
}

# Known v0.5.1 original (gmu=0.9) peak throughput
ORIG_051_PEAK = {
    ('Qwen3-0.6B',    'no-offload'):         634.7,
    ('Qwen3-8B',      'no-offload'):         114.1,
    ('Qwen3-14B',     'no-offload'):          66.1,
    ('Qwen3-32B-AWQ', 'no-offload'):          51.2,
    ('Qwen3-0.6B',    'native-offload-20k'): 632.5,
    ('Qwen3-8B',      'native-offload-20k'):  84.3,
    ('Qwen3-14B',     'native-offload-20k'):  65.1,
    ('Qwen3-32B-AWQ', 'native-offload-20k'):  21.3,
}


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Parse GuideLLM results
# ══════════════════════════════════════════════════════════════════════════════

def parse_dir_name(dir_name):
    parts = dir_name.split('_')
    hardware = parts[0]
    software = parts[1]
    model = parts[2]
    replicas = int(parts[-2].replace('replica', ''))
    rate = int(parts[-1].replace('rate', ''))
    config = '_'.join(parts[3:-2])
    return hardware, software, model, config, replicas, rate


def extract_guidellm_metrics(result_file):
    try:
        proc = subprocess.run(
            ['zstdcat', str(result_file)],
            capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0:
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

        throughput = 0.0
        otc = metrics.get('output_token_count', {})
        if otc and duration > 0:
            total_tokens = otc.get('successful', {}).get('total_sum', 0)
            if total_tokens == 0:
                total_tokens = otc.get('successful', {}).get('sum', 0)
            throughput = total_tokens / duration

        ttft_ms = metrics.get('time_to_first_token_ms', {})
        ttft_median = ttft_ms.get('successful', {}).get('median', 0.0)

        itl_ms = metrics.get('inter_token_latency_ms', {})
        itl_median = itl_ms.get('successful', {}).get('median', 0.0)

        tpot_ms = metrics.get('time_per_output_token_ms', {})
        tpot_median = tpot_ms.get('successful', {}).get('median', 0.0)

        request_totals = metrics.get('request_totals', {})
        completed = request_totals.get('completed', 0) or request_totals.get('successful', 0)

        if 'mempress' in software:
            if '0.4.0' in software:
                suite = 'v040_mempress'
            else:
                suite = 'v051_mempress'
        else:
            suite = 'original'

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


def load_v040_mempress():
    print("Loading v0.4.0-mempress GuideLLM results (including native-offload-20k)...")
    records = []
    pattern = '1x2xL40S_upstream-llm-d-0.4.0-mempress_*/guidellm-results.json.zst'
    files = sorted(RESULTS_DIR.glob(pattern))
    print(f"  Found {len(files)} files")
    for f in files:
        row = extract_guidellm_metrics(f)
        if row:
            records.append(row)
    df = pd.DataFrame(records)
    print(f"  Parsed {len(df)} records, configs: {sorted(df['config'].unique())}")
    return df


def load_v051_mempress():
    print("Loading v0.5.1-mempress GuideLLM results...")
    records = []
    pattern = '1x2xL40S_upstream-llm-d-0.5.1-mempress_*/guidellm-results.json.zst'
    files = sorted(RESULTS_DIR.glob(pattern))
    print(f"  Found {len(files)} files")
    for f in files:
        row = extract_guidellm_metrics(f)
        if row:
            records.append(row)
    df = pd.DataFrame(records)
    print(f"  Parsed {len(df)} records, configs: {sorted(df['config'].unique())}")
    return df


def find_peak_throughput(df, all_configs, orig_peak_dict=None):
    """Find peak throughput per model+config, applying artefact filtering for rate=1."""
    peak_rows = []
    for (model, config), group in df.groupby(['model', 'config']):
        if group.empty:
            continue

        rate1 = group[group['rate'] == 1]
        rate1_tp = rate1.iloc[0]['throughput'] if not rate1.empty else 0
        higher = group[group['rate'] > 1]
        higher_max_tp = higher['throughput'].max() if not higher.empty else 0

        if rate1_tp > 0 and higher_max_tp > 0:
            # Artefact detection: rate=1 throughput implausibly high vs known reference
            orig_ref = orig_peak_dict.get((model, config)) if orig_peak_dict else None
            rate1_is_artefact = False
            if orig_ref is not None:
                if rate1_tp > orig_ref * 2.0:
                    rate1_is_artefact = True
            else:
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
# Part 2: Load existing PCP metrics CSVs
# ══════════════════════════════════════════════════════════════════════════════

def load_v040_pcp():
    """Load existing v0.4.0-mempress PCP metrics CSV."""
    path = OUTPUT_DIR / 'v0.4.0-mempress_pcp_metrics.csv'
    if path.exists():
        df = pd.read_csv(path)
        print(f"Loaded v0.4.0 PCP metrics: {len(df)} rows, columns: {list(df.columns)}")
        return df
    print("WARNING: v0.4.0-mempress_pcp_metrics.csv not found")
    return pd.DataFrame()


def load_v051_mempress_pcp():
    """Load PCP metrics for v0.5.1-mempress runs.

    The existing v0.5.1_pcp_metrics.csv was generated from the original
    (non-mempress, gmu=0.9) v0.5.1 runs.  It is NOT suitable as a proxy
    for v0.5.1-mempress because the workload characteristics differ.
    We must use the v0.5.1-mempress_pcp_metrics.csv (generated by
    extract_v051_mempress_pcp), or fall back to per-archive extraction.
    """
    mempress_path = OUTPUT_DIR / 'v0.5.1-mempress_pcp_metrics.csv'
    if mempress_path.exists():
        df = pd.read_csv(mempress_path)
        print(f"Loaded v0.5.1-mempress PCP metrics from cache: {len(df)} rows")
        return df

    print("No v0.5.1-mempress PCP CSV found — will extract from archives")
    return pd.DataFrame()


def _run_pmrep_for_archive(archive_base, metrics):
    """Run pmrep for a list of metrics, return dict of metric->mean value."""
    cmd = ['pmrep', '-a', archive_base, '-o', 'csv', '-z', '-t', '10s'] + metrics
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return {}
        reader = csv_mod.reader([lines[0]])
        headers = next(reader)
        col_map = defaultdict(list)
        for ci, h in enumerate(headers):
            if ci == 0:
                continue
            base = h.split('-')[0].strip()
            col_map[base].append(ci)
        col_data = defaultdict(list)
        for line in lines[1:]:
            r = csv_mod.reader([line])
            parts = next(r)
            if len(parts) != len(headers):
                continue
            for ci, val in enumerate(parts):
                if ci == 0:
                    continue
                try:
                    col_data[ci].append(float(val))
                except (ValueError, TypeError):
                    pass
        out = {}
        for metric in metrics:
            indices = col_map.get(metric, [])
            vals = []
            for ci in indices:
                if col_data.get(ci):
                    vals.append(np.nanmean(col_data[ci]))
            if vals:
                out[metric] = sum(vals)
        return out
    except subprocess.TimeoutExpired:
        return {}
    except Exception:
        return {}


def extract_v051_mempress_pcp(df_peak):
    """Extract PCP metrics for v0.5.1-mempress peak-rate runs using pmrep.

    Tries two metric naming conventions (colon and dot) to handle different
    PMDA configurations across archive dates.
    """
    print("\nExtracting PCP metrics for v0.5.1-mempress runs...")
    pcp_records = []

    # Two naming conventions observed across PCP versions
    METRIC_SETS = [
        # Convention 1: dot-colon (newer)
        [
            'openmetrics.vllm.vllm:kv_cache_usage_perc',
            'openmetrics.vllm.vllm:num_requests_running',
            'openmetrics.vllm.vllm:num_requests_waiting',
            'openmetrics.vllm.vllm:external_prefix_cache_hits_total',
            'openmetrics.vllm.vllm:external_prefix_cache_queries_total',
        ],
        # Convention 2: dot-dot (older)
        [
            'openmetrics.vllm.vllm.kv_cache_usage_perc',
            'openmetrics.vllm.vllm.num_requests_running',
            'openmetrics.vllm.vllm.num_requests_waiting',
            'openmetrics.vllm.vllm.external_prefix_cache_hits_total',
            'openmetrics.vllm.vllm.external_prefix_cache_queries_total',
        ],
    ]
    RENAME_MAPS = [
        {
            'openmetrics.vllm.vllm:kv_cache_usage_perc': 'kv_cache_usage_pct',
            'openmetrics.vllm.vllm:num_requests_running': 'requests_running',
            'openmetrics.vllm.vllm:num_requests_waiting': 'requests_waiting',
            'openmetrics.vllm.vllm:external_prefix_cache_hits_total': 'ext_prefix_hits',
            'openmetrics.vllm.vllm:external_prefix_cache_queries_total': 'ext_prefix_queries',
        },
        {
            'openmetrics.vllm.vllm.kv_cache_usage_perc': 'kv_cache_usage_pct',
            'openmetrics.vllm.vllm.num_requests_running': 'requests_running',
            'openmetrics.vllm.vllm.num_requests_waiting': 'requests_waiting',
            'openmetrics.vllm.vllm.external_prefix_cache_hits_total': 'ext_prefix_hits',
            'openmetrics.vllm.vllm.external_prefix_cache_queries_total': 'ext_prefix_queries',
        },
    ]

    dirs = sorted(RESULTS_DIR.glob('1x2xL40S_upstream-llm-d-0.5.1-mempress_*/'))
    for run_dir in dirs:
        dir_name = run_dir.name
        try:
            _, software, model, config, replicas, rate = parse_dir_name(dir_name)
        except Exception:
            continue

        # Only process peak rates
        peak_row = df_peak[(df_peak['model'] == model) & (df_peak['config'] == config)]
        if peak_row.empty:
            continue
        peak_rate = int(peak_row.iloc[0]['rate'])
        if rate != peak_rate:
            continue

        print(f"  {model}/{config}/rate={rate}...", end=' ', flush=True)

        archive_dirs = list(run_dir.glob('pcp-archives/*/'))
        if not archive_dirs:
            print("no archive dir")
            continue
        meta_files = list(archive_dirs[0].glob('*.meta.zst'))
        if not meta_files:
            print("no meta.zst")
            continue
        archive_base = str(meta_files[0]).replace('.meta.zst', '')

        row = {'model': model, 'config': config, 'rate': rate}
        got_data = False
        for metrics, rename in zip(METRIC_SETS, RENAME_MAPS):
            data = _run_pmrep_for_archive(archive_base, metrics)
            if data:
                for raw_key, friendly_key in rename.items():
                    if raw_key in data:
                        row[friendly_key] = data[raw_key]
                got_data = True
                break

        if got_data:
            pcp_records.append(row)
            print("OK")
        else:
            print("no data")

    if pcp_records:
        df_pcp = pd.DataFrame(pcp_records)
        df_pcp.to_csv(OUTPUT_DIR / 'v0.5.1-mempress_pcp_metrics.csv', index=False)
        print(f"  Saved {len(df_pcp)} PCP records")
        return df_pcp
    return pd.DataFrame()


def compute_ext_hit_rate(df_pcp):
    """Add ext_hit_rate_pct column (%) to a PCP dataframe.

    Handles two possible column naming conventions:
      - ext_prefix_hits / ext_prefix_queries  (v0.4.0 PCP CSV)
      - ext_prefix_hits / ext_prefix_queries  (v0.5.1 PCP CSV — same names)
    """
    df = df_pcp.copy()
    hits_col = None
    queries_col = None
    for h in ['ext_prefix_hits']:
        if h in df.columns:
            hits_col = h
            break
    for q in ['ext_prefix_queries']:
        if q in df.columns:
            queries_col = q
            break

    if hits_col and queries_col:
        df['ext_hit_rate_pct'] = np.nan
        # Cast to numeric, coerce errors
        hits = pd.to_numeric(df[hits_col], errors='coerce').fillna(0)
        queries = pd.to_numeric(df[queries_col], errors='coerce').fillna(0)
        mask = queries > 0
        df.loc[mask, 'ext_hit_rate_pct'] = hits[mask] / queries[mask] * 100
        df.loc[~mask, 'ext_hit_rate_pct'] = 0.0
    else:
        df['ext_hit_rate_pct'] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Print data tables
# ══════════════════════════════════════════════════════════════════════════════

def print_table(title, df_peak, configs, models=MODELS, ref_config='no-offload'):
    print(f"\n{'='*90}")
    print(title)
    print('='*90)
    print(f"\n{'Config':<24}", end='')
    for m in models:
        print(f"  {MODEL_LABELS[m]:>10}", end='')
    print()
    print('-'*70)

    for config in configs:
        rows = {m: df_peak[(df_peak['model']==m) & (df_peak['config']==config)] for m in models}
        vals = {m: (r.iloc[0]['throughput'] if not r.empty else None) for m, r in rows.items()}
        print(f"  {config:<22}", end='')
        for m in models:
            v = vals[m]
            print(f"  {v:>9.1f}" if v else f"  {'N/A':>9}", end='')
        print()

    # Delta vs no-offload
    print(f"\n{'Config (% vs no-offload)':<24}", end='')
    for m in models:
        print(f"  {MODEL_LABELS[m]:>10}", end='')
    print()
    print('-'*70)

    baselines = {}
    for m in models:
        bl = df_peak[(df_peak['model']==m) & (df_peak['config']==ref_config)]
        baselines[m] = bl.iloc[0]['throughput'] if not bl.empty else None

    for config in configs:
        if config == ref_config:
            continue
        print(f"  {config:<22}", end='')
        for m in models:
            rows = df_peak[(df_peak['model']==m) & (df_peak['config']==config)]
            bl = baselines[m]
            if not rows.empty and bl:
                v = rows.iloc[0]['throughput']
                delta = (v - bl) / bl * 100
                print(f"  {delta:>+9.1f}%", end='')
            else:
                print(f"  {'N/A':>10}", end='')
        print()


def print_pcp_ext_cache(df_pcp, title, configs):
    print(f"\n{'='*90}")
    print(title)
    print('='*90)
    print(f"\n{'Config':<24}", end='')
    for m in MODELS:
        print(f"  {MODEL_LABELS[m]:>12}", end='')
    print()
    print('-'*80)

    for config in configs:
        print(f"  {config:<22}", end='')
        for m in MODELS:
            rows = df_pcp[(df_pcp['model']==m) & (df_pcp['config']==config)]
            if not rows.empty and 'ext_hit_rate_pct' in rows.columns:
                v = rows.iloc[0]['ext_hit_rate_pct']
                if pd.notna(v):
                    print(f"  {v:>11.1f}%", end='')
                else:
                    print(f"  {'N/A':>12}", end='')
            else:
                print(f"  {'N/A':>12}", end='')
        print()


def print_pcp_kvcache(df_pcp, title, configs):
    print(f"\n{'='*90}")
    print(title)
    print('='*90)
    print(f"\n{'Config':<24}", end='')
    for m in MODELS:
        print(f"  {MODEL_LABELS[m]:>12}", end='')
    print()
    print('-'*80)

    for config in configs:
        print(f"  {config:<22}", end='')
        for m in MODELS:
            rows = df_pcp[(df_pcp['model']==m) & (df_pcp['config']==config)]
            if not rows.empty and 'kv_cache_usage_pct' in rows.columns:
                v = rows.iloc[0]['kv_cache_usage_pct']
                if pd.notna(v):
                    print(f"  {v*100:>11.1f}%", end='')
                else:
                    print(f"  {'N/A':>12}", end='')
            else:
                print(f"  {'N/A':>12}", end='')
        print()


def print_cross_version_table(df040, df051):
    """Cross-version native-offload comparison: v0.4.0-10k, v0.4.0-20k, v0.5.1-20k."""
    print(f"\n{'='*90}")
    print("CROSS-VERSION NATIVE OFFLOAD COMPARISON (mempress gmu)")
    print("All values: peak throughput tok/s | delta vs respective no-offload baseline")
    print('='*90)

    rows_data = []
    for model in MODELS:
        bl_040 = df040[(df040['model']==model) & (df040['config']=='no-offload')]
        bl_051 = df051[(df051['model']==model) & (df051['config']=='no-offload')]
        bl_040_val = bl_040.iloc[0]['throughput'] if not bl_040.empty else None
        bl_051_val = bl_051.iloc[0]['throughput'] if not bl_051.empty else None

        v040_10k = df040[(df040['model']==model) & (df040['config']=='native-offload')]
        v040_20k = df040[(df040['model']==model) & (df040['config']=='native-offload-20k')]
        v051_20k = df051[(df051['model']==model) & (df051['config']=='native-offload-20k')]

        def fmt(df_sub, bl):
            if df_sub.empty or bl is None:
                return 'N/A', 'N/A'
            v = df_sub.iloc[0]['throughput']
            d = (v - bl) / bl * 100
            return f"{v:.1f}", f"{d:+.1f}%"

        t040_10k, d040_10k = fmt(v040_10k, bl_040_val)
        t040_20k, d040_20k = fmt(v040_20k, bl_040_val)
        t051_20k, d051_20k = fmt(v051_20k, bl_051_val)

        rows_data.append({
            'model': MODEL_LABELS[model],
            'v040_no_offload': f"{bl_040_val:.1f}" if bl_040_val else "N/A",
            'v040_10k': t040_10k, 'v040_10k_d': d040_10k,
            'v040_20k': t040_20k, 'v040_20k_d': d040_20k,
            'v051_no_offload': f"{bl_051_val:.1f}" if bl_051_val else "N/A",
            'v051_20k': t051_20k, 'v051_20k_d': d051_20k,
        })

    print(f"\n  {'Model':<12} {'v0.4.0 no-off':>14} {'v0.4.0 nat-10k':>16} {'v0.4.0 nat-20k':>16} "
          f"{'v0.5.1 no-off':>14} {'v0.5.1 nat-20k':>16}")
    print(f"  {'-'*92}")
    for r in rows_data:
        print(f"  {r['model']:<12} {r['v040_no_offload']:>14} "
              f"{r['v040_10k']:>8} {r['v040_10k_d']:>8} "
              f"{r['v040_20k']:>8} {r['v040_20k_d']:>8} "
              f"{r['v051_no_offload']:>14} "
              f"{r['v051_20k']:>8} {r['v051_20k_d']:>8}")


def print_latency_table(df, configs, suite_label, rate=50):
    print(f"\n{'='*90}")
    print(f"LATENCY AT RATE={rate}: {suite_label}")
    print(f"{'='*90}")
    print(f"\n{'Config':<24} {'Model':<16} {'TTFT median (ms)':>18} {'ITL median (ms)':>16}")
    print('-'*78)
    for config in configs:
        for model in MODELS:
            rows = df[(df['model']==model) & (df['config']==config) & (df['rate']==rate)]
            if not rows.empty:
                ttft = rows.iloc[0]['ttft_median_ms']
                itl = rows.iloc[0]['itl_median_ms']
                print(f"  {config:<22} {MODEL_LABELS[model]:<16} {ttft:>16.0f} ms {itl:>14.1f} ms")


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Visualisations
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = sns.color_palette("muted")


def _palette_map(keys):
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}


def plot_v040_peak_throughput_v2(df040_peak):
    """Updated grouped bar including native-offload-20k."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=False)
    fig.suptitle('v0.4.0 Memory-Pressure Peak Throughput by Config and Model',
                 fontsize=13, fontweight='bold', y=1.01)

    configs = V040_CONFIGS_ALL
    colors = _palette_map(configs)

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        vals, clrs, labels = [], [], []
        for cfg in configs:
            rows = df040_peak[(df040_peak['model']==model) & (df040_peak['config']==cfg)]
            t = rows.iloc[0]['throughput'] if not rows.empty else 0
            vals.append(t)
            clrs.append(colors[cfg])
            labels.append(CONFIG_LABELS.get(cfg, cfg))

        x = np.arange(len(vals))
        bars = ax.barh(x, vals, 0.65, color=clrs, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(val + max(v for v in vals if v > 0) * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.0f}', va='center', ha='left', fontsize=8)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Output tokens/s', fontsize=9)
        gmu = GMU_MEMPRESS[model]
        ax.set_title(f'{MODEL_LABELS[model]}\n(gmu={gmu})', fontsize=10, fontweight='bold')
        max_val = max(v for v in vals if v > 0) if any(v > 0 for v in vals) else 100
        ax.set_xlim(0, max_val * 1.15)

    plt.tight_layout()
    path = OUTPUT_DIR / 'v0.4.0-mempress_peak_throughput_v2.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_v040_external_cache_hits(df040_pcp):
    """External cache hit rate by config and model for v0.4.0-mempress."""
    if df040_pcp.empty or 'ext_hit_rate_pct' not in df040_pcp.columns:
        print("  Skipping v0.4.0 external cache hits: no data")
        return

    # Only mempress suite rows
    df = df040_pcp[df040_pcp['suite'] == 'mempress'].copy() if 'suite' in df040_pcp.columns else df040_pcp.copy()

    configs_show = ['no-offload', 'native-offload', 'native-offload-20k',
                    'lmcache-local', 'lmcache-valkey', 'llm-d-valkey']
    # Filter to configs present in data
    configs_present = [c for c in configs_show if c in df['config'].values]

    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
    fig.suptitle('v0.4.0 Memory-Pressure: External KV Cache Hit Rate at Peak Rate\n'
                 '(Shows proportion of queries served from offloaded cache)',
                 fontsize=12, fontweight='bold', y=1.02)

    colors = _palette_map(configs_present)

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        vals, clrs, labels = [], [], []
        for cfg in configs_present:
            rows = df[(df['model']==model) & (df['config']==cfg)]
            if not rows.empty and 'ext_hit_rate_pct' in rows.columns:
                v = rows.iloc[0]['ext_hit_rate_pct']
                vals.append(v if pd.notna(v) else 0.0)
            else:
                vals.append(0.0)
            clrs.append(colors[cfg])
            labels.append(CONFIG_LABELS.get(cfg, cfg))

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=clrs, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('External Cache Hit Rate (%)', fontsize=9)
        ax.set_ylim(0, 105)
        ax.axhline(0, color='black', linewidth=0.5)
        gmu = GMU_MEMPRESS[model]
        ax.set_title(f'{MODEL_LABELS[model]}\n(gmu={gmu})', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = OUTPUT_DIR / 'v0.4.0-mempress_external_cache_hits.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_v040_gpu_kvcache(df040_pcp):
    """GPU KV cache utilisation at peak rate for v0.4.0-mempress."""
    if df040_pcp.empty or 'kv_cache_usage_pct' not in df040_pcp.columns:
        print("  Skipping v0.4.0 GPU KV cache plot: no data")
        return

    df = df040_pcp[df040_pcp['suite'] == 'mempress'].copy() if 'suite' in df040_pcp.columns else df040_pcp.copy()
    configs_show = [c for c in V040_CONFIGS_OLD if c in df['config'].values]

    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
    fig.suptitle('v0.4.0 Memory-Pressure: GPU KV Cache Utilisation at Peak Rate\n'
                 '(Shows whether GPU KV cache is under pressure)',
                 fontsize=12, fontweight='bold', y=1.02)

    colors = _palette_map(configs_show)

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        vals, clrs, labels = [], [], []
        for cfg in configs_show:
            rows = df[(df['model']==model) & (df['config']==cfg)]
            if not rows.empty and 'kv_cache_usage_pct' in rows.columns:
                v = rows.iloc[0]['kv_cache_usage_pct']
                vals.append(v * 100 if pd.notna(v) else 0.0)
            else:
                vals.append(0.0)
            clrs.append(colors[cfg])
            labels.append(CONFIG_LABELS.get(cfg, cfg))

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=clrs, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('GPU KV Cache Usage (%)', fontsize=9)
        ax.set_ylim(0, 115)
        ax.axhline(80, color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label='80% reference')
        ax.axhline(100, color='red', linestyle=':', linewidth=0.8, alpha=0.6, label='100%')
        gmu = GMU_MEMPRESS[model]
        ax.set_title(f'{MODEL_LABELS[model]}\n(gmu={gmu})', fontsize=10, fontweight='bold')
        if ax_idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    path = OUTPUT_DIR / 'v0.4.0-mempress_gpu_kvcache_util.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_v051_peak_throughput(df051_peak):
    """Grouped bar: v0.5.1-mempress peak throughput."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=False)
    fig.suptitle('v0.5.1 Memory-Pressure Peak Throughput by Config and Model',
                 fontsize=13, fontweight='bold', y=1.01)

    colors = _palette_map(V051_CONFIGS)

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        vals, clrs, labels = [], [], []
        for cfg in V051_CONFIGS:
            rows = df051_peak[(df051_peak['model']==model) & (df051_peak['config']==cfg)]
            t = rows.iloc[0]['throughput'] if not rows.empty else 0
            vals.append(t)
            clrs.append(colors[cfg])
            labels.append(CONFIG_LABELS.get(cfg, cfg))

        x = np.arange(len(vals))
        bars = ax.barh(x, vals, 0.65, color=clrs, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(val + max(v for v in vals if v > 0) * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.0f}', va='center', ha='left', fontsize=8)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Output tokens/s', fontsize=9)
        gmu = GMU_MEMPRESS[model]
        ax.set_title(f'{MODEL_LABELS[model]}\n(gmu={gmu})', fontsize=10, fontweight='bold')
        max_val = max(v for v in vals if v > 0) if any(v > 0 for v in vals) else 100
        ax.set_xlim(0, max_val * 1.15)

    plt.tight_layout()
    path = OUTPUT_DIR / 'v0.5.1-mempress_peak_throughput.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_v051_delta_heatmap(df051_peak):
    """Heatmap: % delta vs no-offload for v0.5.1-mempress, magma colormap."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('v0.5.1 Memory-Pressure: % Delta vs No-Offload Baseline',
                 fontsize=13, fontweight='bold')

    configs_offload = [c for c in V051_CONFIGS if c != 'no-offload']
    model_labels = [MODEL_LABELS[m] for m in MODELS]

    data_matrix = np.full((len(configs_offload), len(MODELS)), np.nan)
    for row_i, config in enumerate(configs_offload):
        for col_j, model in enumerate(MODELS):
            bl_rows = df051_peak[(df051_peak['model']==model) & (df051_peak['config']=='no-offload')]
            v_rows = df051_peak[(df051_peak['model']==model) & (df051_peak['config']==config)]
            if not bl_rows.empty and not v_rows.empty:
                bl = bl_rows.iloc[0]['throughput']
                v = v_rows.iloc[0]['throughput']
                if bl > 0:
                    data_matrix[row_i, col_j] = (v - bl) / bl * 100

    vmax = max(abs(np.nanmin(data_matrix)), abs(np.nanmax(data_matrix)), 5)
    vmax = min(vmax, 80)

    sns.heatmap(
        data_matrix,
        ax=ax,
        annot=True,
        fmt='.1f',
        cmap='magma',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=model_labels,
        yticklabels=[CONFIG_LABELS.get(c, c) for c in configs_offload],
        cbar_kws={'label': '% delta vs no-offload'},
        linewidths=0.5,
    )
    ax.set_title('% Delta vs No-Offload Baseline (positive = faster)', fontsize=11)
    ax.set_xlabel('Model')
    ax.set_ylabel('Configuration')

    plt.tight_layout()
    path = OUTPUT_DIR / 'v0.5.1-mempress_delta_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_v051_throughput_curves(df051):
    """4-panel throughput vs rate for v0.5.1-mempress."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    fig.suptitle('v0.5.1 Memory-Pressure: Throughput vs Request Rate',
                 fontsize=13, fontweight='bold')

    colors = _palette_map(V051_CONFIGS)
    markers = {'no-offload': 'o', 'native-offload-20k': 's', 'fs-offload': '^', 'cpu+fs-offload-20k': 'D'}
    ls_map = {'no-offload': '--', 'native-offload-20k': '-', 'fs-offload': '-', 'cpu+fs-offload-20k': '-'}

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        for cfg in V051_CONFIGS:
            subset = df051[(df051['model']==model) & (df051['config']==cfg)].sort_values('rate')
            if subset.empty:
                continue
            ax.plot(subset['rate'], subset['throughput'],
                    color=colors[cfg],
                    linestyle=ls_map.get(cfg, '-'),
                    marker=markers.get(cfg, 'o'),
                    linewidth=2,
                    markersize=5,
                    label=CONFIG_LABELS.get(cfg, cfg))
        gmu = GMU_MEMPRESS[model]
        ax.set_title(f'{MODEL_LABELS[model]} (gmu={gmu})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Request rate (concurrency)', fontsize=9)
        ax.set_ylabel('Output tokens/s', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    path = OUTPUT_DIR / 'v0.5.1-mempress_throughput_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_v051_external_cache_hits(df051_pcp):
    """External cache hit rate for v0.5.1-mempress."""
    if df051_pcp.empty or 'ext_hit_rate_pct' not in df051_pcp.columns:
        print("  Skipping v0.5.1 external cache hits: no data")
        return

    configs_show = [c for c in V051_CONFIGS if c in df051_pcp['config'].values]

    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
    fig.suptitle('v0.5.1 Memory-Pressure: External KV Cache Hit Rate at Peak Rate\n'
                 '(Shows proportion of queries served from offloaded cache)',
                 fontsize=12, fontweight='bold', y=1.02)

    colors = _palette_map(configs_show)

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        vals, clrs, labels = [], [], []
        for cfg in configs_show:
            rows = df051_pcp[(df051_pcp['model']==model) & (df051_pcp['config']==cfg)]
            if not rows.empty and 'ext_hit_rate_pct' in rows.columns:
                v = rows.iloc[0]['ext_hit_rate_pct']
                vals.append(v if pd.notna(v) else 0.0)
            else:
                vals.append(0.0)
            clrs.append(colors[cfg])
            labels.append(CONFIG_LABELS.get(cfg, cfg))

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=clrs, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('External Cache Hit Rate (%)', fontsize=9)
        ax.set_ylim(0, 105)
        gmu = GMU_MEMPRESS[model]
        ax.set_title(f'{MODEL_LABELS[model]}\n(gmu={gmu})', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = OUTPUT_DIR / 'v0.5.1-mempress_external_cache_hits.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_v051_gpu_kvcache(df051_pcp):
    """GPU KV cache utilisation for v0.5.1-mempress."""
    if df051_pcp.empty or 'kv_cache_usage_pct' not in df051_pcp.columns:
        print("  Skipping v0.5.1 GPU KV cache plot: no data")
        return

    configs_show = [c for c in V051_CONFIGS if c in df051_pcp['config'].values]

    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
    fig.suptitle('v0.5.1 Memory-Pressure: GPU KV Cache Utilisation at Peak Rate',
                 fontsize=12, fontweight='bold', y=1.01)

    colors = _palette_map(configs_show)

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        vals, clrs, labels = [], [], []
        for cfg in configs_show:
            rows = df051_pcp[(df051_pcp['model']==model) & (df051_pcp['config']==cfg)]
            if not rows.empty and 'kv_cache_usage_pct' in rows.columns:
                v = rows.iloc[0]['kv_cache_usage_pct']
                vals.append(v * 100 if pd.notna(v) else 0.0)
            else:
                vals.append(0.0)
            clrs.append(colors[cfg])
            labels.append(CONFIG_LABELS.get(cfg, cfg))

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=clrs, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('GPU KV Cache Usage (%)', fontsize=9)
        ax.set_ylim(0, 115)
        ax.axhline(80, color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label='80%')
        gmu = GMU_MEMPRESS[model]
        ax.set_title(f'{MODEL_LABELS[model]}\n(gmu={gmu})', fontsize=10, fontweight='bold')
        if ax_idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    path = OUTPUT_DIR / 'v0.5.1-mempress_gpu_kvcache.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_v051_vs_original(df051_peak):
    """Side-by-side: gmu=0.9 (original) vs gmu=mempress for v0.5.1 native-offload-20k."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=False)
    fig.suptitle('v0.5.1 Native Offload (20k): Original (gmu=0.9) vs Memory-Pressure\n'
                 'Shows whether reduced GPU KV capacity improves offload benefit',
                 fontsize=12, fontweight='bold', y=1.03)

    orig_color = PALETTE[0]
    mempress_color = PALETTE[2]

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]

        orig_no = ORIG_051_PEAK.get((model, 'no-offload'), 0)
        orig_nat = ORIG_051_PEAK.get((model, 'native-offload-20k'), 0)
        mp_no_rows = df051_peak[(df051_peak['model']==model) & (df051_peak['config']=='no-offload')]
        mp_nat_rows = df051_peak[(df051_peak['model']==model) & (df051_peak['config']=='native-offload-20k')]
        mp_no = mp_no_rows.iloc[0]['throughput'] if not mp_no_rows.empty else 0
        mp_nat = mp_nat_rows.iloc[0]['throughput'] if not mp_nat_rows.empty else 0

        bars_data = [
            ('No Offload\n(gmu=0.9)', orig_no, orig_color, 0.6),
            ('Nat Off-20k\n(gmu=0.9)', orig_nat, orig_color, 0.6),
            (f'No Offload\n(gmu={GMU_MEMPRESS[model]})', mp_no, mempress_color, 1.0),
            (f'Nat Off-20k\n(gmu={GMU_MEMPRESS[model]})', mp_nat, mempress_color, 1.0),
        ]

        x = np.arange(len(bars_data))
        # Plot bars individually (alpha must be scalar per bar in matplotlib)
        bars = []
        for xi, (label, val, color, alpha) in zip(x, bars_data):
            b = ax.bar(xi, val, color=color, edgecolor='white', linewidth=0.5, alpha=alpha)
            bars.append(b[0])
        for bar, (label, val, _, __) in zip(bars, bars_data):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + max(b[1] for b in bars_data) * 0.01,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)

        # Add delta annotation for native-offload vs no-offload in each condition
        if orig_no > 0 and orig_nat > 0:
            d_orig = (orig_nat - orig_no) / orig_no * 100
            ax.annotate(f'{d_orig:+.1f}%', xy=(1, orig_nat), xytext=(1.3, orig_nat * 1.05),
                        fontsize=8, color='gray',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        if mp_no > 0 and mp_nat > 0:
            d_mp = (mp_nat - mp_no) / mp_no * 100
            ax.annotate(f'{d_mp:+.1f}%', xy=(3, mp_nat), xytext=(3.2, mp_nat * 1.05),
                        fontsize=8, color='black',
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars_data], fontsize=7)
        ax.set_ylabel('Output tokens/s', fontsize=9)
        ax.set_title(f'{MODEL_LABELS[model]}', fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(b[1] for b in bars_data) * 1.25 if any(b[1] > 0 for b in bars_data) else 100)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=orig_color, alpha=0.6, label='Original (gmu=0.9)'),
        Patch(facecolor=mempress_color, label='Memory-Pressure (reduced gmu)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / 'v0.5.1-mempress_vs_original.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_crossversion_native_offload(df040_peak, df051_peak):
    """Cross-version native offload comparison: v0.4.0-10k, v0.4.0-20k, v0.5.1-20k."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=False)
    fig.suptitle('Cross-Version Native Offload Comparison (all at memory-pressure gmu)\n'
                 'v0.4.0 native-10k | v0.4.0 native-20k | v0.5.1 native-20k',
                 fontsize=12, fontweight='bold', y=1.03)

    # Colors: no-offload baselines greyed; native-offload variants colored
    c_040_no   = PALETTE[7] if len(PALETTE) > 7 else PALETTE[0]
    c_040_10k  = PALETTE[1]
    c_040_20k  = PALETTE[2]
    c_051_no   = PALETTE[6] if len(PALETTE) > 6 else PALETTE[3]
    c_051_20k  = PALETTE[4]

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]

        def get_tp(df, config):
            rows = df[(df['model']==model) & (df['config']==config)]
            return rows.iloc[0]['throughput'] if not rows.empty else 0

        v040_no  = get_tp(df040_peak, 'no-offload')
        v040_10k = get_tp(df040_peak, 'native-offload')
        v040_20k = get_tp(df040_peak, 'native-offload-20k')
        v051_no  = get_tp(df051_peak, 'no-offload')
        v051_20k = get_tp(df051_peak, 'native-offload-20k')

        gmu = GMU_MEMPRESS[model]
        bars_data = [
            (f'v0.4.0\nno-off\n(gmu={gmu})',    v040_no,  c_040_no,  0.6),
            (f'v0.4.0\nnat-10k\n(gmu={gmu})',   v040_10k, c_040_10k, 1.0),
            (f'v0.4.0\nnat-20k\n(gmu={gmu})',   v040_20k, c_040_20k, 1.0),
            (f'v0.5.1\nno-off\n(gmu={gmu})',    v051_no,  c_051_no,  0.6),
            (f'v0.5.1\nnat-20k\n(gmu={gmu})',   v051_20k, c_051_20k, 1.0),
        ]

        x = np.arange(len(bars_data))
        # Plot bars individually so alpha can vary per bar
        bars = []
        for xi, (label, val, color, alpha) in zip(x, bars_data):
            b = ax.bar(xi, val, color=color, alpha=alpha, edgecolor='white', linewidth=0.5)
            bars.append(b[0])

        all_vals = [b[1] for b in bars_data if b[1] > 0]
        max_val = max(all_vals) if all_vals else 100

        for bar, (label, val, _, _) in zip(bars, bars_data):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + max_val * 0.01,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)

        # Delta annotations vs respective no-offload baseline
        for i, (cfg, val, _, _) in enumerate(bars_data):
            if 'nat-10k' in cfg and v040_no > 0 and val > 0:
                d = (val - v040_no) / v040_no * 100
                ax.text(x[i], val + max_val * 0.08, f'{d:+.1f}%', ha='center', fontsize=7.5,
                        color='darkgreen' if d > 0 else 'darkred')
            elif 'nat-20k' in cfg and 'v0.4.0' in cfg and v040_no > 0 and val > 0:
                d = (val - v040_no) / v040_no * 100
                ax.text(x[i], val + max_val * 0.08, f'{d:+.1f}%', ha='center', fontsize=7.5,
                        color='darkgreen' if d > 0 else 'darkred')
            elif 'nat-20k' in cfg and 'v0.5.1' in cfg and v051_no > 0 and val > 0:
                d = (val - v051_no) / v051_no * 100
                ax.text(x[i], val + max_val * 0.08, f'{d:+.1f}%', ha='center', fontsize=7.5,
                        color='darkgreen' if d > 0 else 'darkred')

        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars_data], fontsize=7)
        ax.set_ylabel('Output tokens/s', fontsize=9)
        ax.set_title(f'{MODEL_LABELS[model]}', fontsize=10, fontweight='bold')
        ax.set_ylim(0, max_val * 1.25)

        # Vertical separator between v040 and v051 groups
        ax.axvline(2.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.text(1.0, max_val * 1.15, 'v0.4.0', ha='center', fontsize=8, color='gray')
        ax.text(3.5, max_val * 1.15, 'v0.5.1', ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    path = OUTPUT_DIR / 'mempress_crossversion_native_offload.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Comprehensive memory-pressure benchmark analysis")
    print("=" * 70)

    # ── Load guidellm results ─────────────────────────────────────────────────
    df040 = load_v040_mempress()
    df051 = load_v051_mempress()

    # ── Find peak throughput ──────────────────────────────────────────────────
    print("\nFinding peak throughput...")
    df040_peak = find_peak_throughput(df040, V040_CONFIGS_ALL, ORIG_040_PEAK)
    df051_peak = find_peak_throughput(df051, V051_CONFIGS, ORIG_051_PEAK)
    print(f"  v0.4.0-mempress peak rows: {len(df040_peak)}")
    print(f"  v0.5.1-mempress peak rows: {len(df051_peak)}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    # v0.4.0 native-offload-20k peak data
    df040_20k_peak = df040_peak[df040_peak['config'] == 'native-offload-20k'].copy()
    df040_no_peak  = df040_peak[df040_peak['config'] == 'no-offload'].copy()
    df040_20k_merged = pd.merge(df040_20k_peak, df040_no_peak[['model', 'throughput']],
                                 on='model', suffixes=('', '_no_offload'))
    if 'throughput_no_offload' in df040_20k_merged.columns:
        df040_20k_merged['delta_vs_no_offload_pct'] = (
            (df040_20k_merged['throughput'] - df040_20k_merged['throughput_no_offload'])
            / df040_20k_merged['throughput_no_offload'] * 100
        )
    df040_20k_merged.to_csv(OUTPUT_DIR / 'v0.4.0-mempress_native20k_peak.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'v0.4.0-mempress_native20k_peak.csv'}")

    # v0.5.1-mempress peak table with deltas
    df051_summary_rows = []
    for model in MODELS:
        bl_row = df051_peak[(df051_peak['model']==model) & (df051_peak['config']=='no-offload')]
        bl = bl_row.iloc[0]['throughput'] if not bl_row.empty else None
        for cfg in V051_CONFIGS:
            row = df051_peak[(df051_peak['model']==model) & (df051_peak['config']==cfg)]
            if row.empty:
                continue
            r = row.iloc[0].to_dict()
            r['delta_vs_no_offload_pct'] = (r['throughput'] - bl) / bl * 100 if bl else None
            orig_ref = ORIG_051_PEAK.get((model, cfg))
            r['orig_v051_peak'] = orig_ref
            r['delta_vs_orig_v051_pct'] = (
                (r['throughput'] - orig_ref) / orig_ref * 100 if orig_ref else None
            )
            df051_summary_rows.append(r)
    df051_summary = pd.DataFrame(df051_summary_rows)
    df051_summary.to_csv(OUTPUT_DIR / 'v0.5.1-mempress_peak_throughput.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'v0.5.1-mempress_peak_throughput.csv'}")

    # ── Print analysis tables ─────────────────────────────────────────────────
    print_table("v0.4.0-MEMPRESS PEAK THROUGHPUT (including native-offload-20k)",
                df040_peak, V040_CONFIGS_ALL)

    print_table("v0.5.1-MEMPRESS PEAK THROUGHPUT",
                df051_peak, V051_CONFIGS)

    print_cross_version_table(df040_peak, df051_peak)

    # Latency at rate=50
    print_latency_table(df040, V040_CONFIGS_OLD, "v0.4.0-mempress", rate=50)
    print_latency_table(df051, V051_CONFIGS, "v0.5.1-mempress", rate=50)

    # ── Load PCP metrics ──────────────────────────────────────────────────────
    print("\nLoading PCP metrics...")
    df040_pcp = load_v040_pcp()
    if not df040_pcp.empty:
        df040_pcp = compute_ext_hit_rate(df040_pcp)
        print_pcp_ext_cache(
            df040_pcp[df040_pcp['suite']=='mempress'] if 'suite' in df040_pcp.columns else df040_pcp,
            "v0.4.0-MEMPRESS EXTERNAL CACHE HIT RATE AT PEAK RATE",
            V040_CONFIGS_OLD
        )
        print_pcp_kvcache(
            df040_pcp[df040_pcp['suite']=='mempress'] if 'suite' in df040_pcp.columns else df040_pcp,
            "v0.4.0-MEMPRESS GPU KV CACHE USAGE AT PEAK RATE",
            V040_CONFIGS_OLD
        )

    # Load or extract v0.5.1-mempress PCP
    df051_pcp = load_v051_mempress_pcp()
    if df051_pcp.empty:
        print("Attempting to extract v0.5.1-mempress PCP from archives...")
        df051_pcp = extract_v051_mempress_pcp(df051_peak)

    if not df051_pcp.empty:
        df051_pcp = compute_ext_hit_rate(df051_pcp)
        print_pcp_ext_cache(
            df051_pcp,
            "v0.5.1-MEMPRESS EXTERNAL CACHE HIT RATE AT PEAK RATE",
            V051_CONFIGS
        )
        print_pcp_kvcache(
            df051_pcp,
            "v0.5.1-MEMPRESS GPU KV CACHE USAGE AT PEAK RATE",
            V051_CONFIGS
        )

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("Generating visualisations...")
    print("="*70)

    # v0.4.0-mempress
    plot_v040_peak_throughput_v2(df040_peak)
    plot_v040_external_cache_hits(df040_pcp)
    plot_v040_gpu_kvcache(df040_pcp)

    # v0.5.1-mempress
    plot_v051_peak_throughput(df051_peak)
    plot_v051_delta_heatmap(df051_peak)
    plot_v051_throughput_curves(df051)
    plot_v051_external_cache_hits(df051_pcp)
    plot_v051_gpu_kvcache(df051_pcp)
    plot_v051_vs_original(df051_peak)

    # Cross-version
    plot_crossversion_native_offload(df040_peak, df051_peak)

    print("\nDone.")


if __name__ == '__main__':
    main()
