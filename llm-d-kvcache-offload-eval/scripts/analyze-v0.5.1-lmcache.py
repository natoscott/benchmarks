#!/usr/bin/env python3
"""Analysis of llm-d v0.5.1 LMCache benchmarks.

Analyzes lmcache-local and lmcache-valkey configs at gmu=0.9 and mempress,
comparing against no-offload and native-offload-20k baselines and v0.4.0
historical data.
"""

import json
import subprocess
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
RATES = [1, 50, 100, 150, 300, 400, 500, 650]

# Configs to include in lmcache analysis
LMCACHE_CONFIGS = ['no-offload', 'lmcache-local', 'lmcache-valkey', 'native-offload-20k']

CONFIG_LABELS = {
    'no-offload':        'No Offload',
    'lmcache-local':     'LMCache Local',
    'lmcache-valkey':    'LMCache Valkey',
    'native-offload-20k': 'Native Offload (20k)',
}

# ── v0.4.0 historical data (from REPORT-v0.4.0.md) ──────────────────────────
V040_GMU09 = {
    ('Qwen3-0.6B',    'no-offload'):      602.0,
    ('Qwen3-8B',      'no-offload'):      113.0,
    ('Qwen3-14B',     'no-offload'):       58.7,
    ('Qwen3-32B-AWQ', 'no-offload'):       49.2,
    ('Qwen3-0.6B',    'lmcache-local'):   520.4,
    ('Qwen3-8B',      'lmcache-local'):   106.6,
    ('Qwen3-14B',     'lmcache-local'):    65.6,
    ('Qwen3-32B-AWQ', 'lmcache-local'):    43.0,
    ('Qwen3-0.6B',    'lmcache-valkey'):  523.9,
    ('Qwen3-8B',      'lmcache-valkey'):  105.7,
    ('Qwen3-14B',     'lmcache-valkey'):   66.3,
    ('Qwen3-32B-AWQ', 'lmcache-valkey'):   43.0,
}

V040_MEMPRESS = {
    ('Qwen3-0.6B',    'no-offload'):      437.3,
    ('Qwen3-8B',      'no-offload'):      116.3,
    ('Qwen3-14B',     'no-offload'):       66.1,
    ('Qwen3-32B-AWQ', 'no-offload'):       46.9,
    ('Qwen3-0.6B',    'lmcache-local'):   518.4,
    ('Qwen3-8B',      'lmcache-local'):   105.6,
    ('Qwen3-14B',     'lmcache-local'):    57.6,
    ('Qwen3-32B-AWQ', 'lmcache-local'):    41.6,
    ('Qwen3-0.6B',    'lmcache-valkey'):  522.7,
    ('Qwen3-8B',      'lmcache-valkey'):  107.7,
    ('Qwen3-14B',     'lmcache-valkey'):   54.4,
    ('Qwen3-32B-AWQ', 'lmcache-valkey'):   41.6,
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def parse_dir_name(dir_name):
    """Parse benchmark directory name into components.

    Format: 1x2xL40S_upstream-llm-d-0.5.1_Qwen3-0.6B_lmcache-local_replica1_rate50
    Or:     1x2xL40S_upstream-llm-d-0.5.1-mempress_Qwen3-0.6B_lmcache-local_replica1_rate50

    Returns: (hardware, software, model, config, replicas, rate)
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

        # Throughput: total successful output tokens / benchmark duration
        throughput = 0.0
        otc = metrics.get('output_token_count', {})
        if otc and duration > 0:
            total_tokens = otc.get('successful', {}).get('total_sum', 0)
            if total_tokens == 0:
                total_tokens = otc.get('successful', {}).get('sum', 0)
            throughput = total_tokens / duration

        # TTFT (ms -> s)
        ttft_ms = metrics.get('time_to_first_token_ms', {})
        ttft_median = ttft_ms.get('successful', {}).get('median', 0.0) / 1000.0

        # ITL (ms)
        itl_ms = metrics.get('inter_token_latency_ms', {})
        itl_median = itl_ms.get('successful', {}).get('median', 0.0)

        # Completed requests
        request_totals = metrics.get('request_totals', {})
        completed = request_totals.get('completed', 0)
        if completed == 0:
            completed = request_totals.get('successful', 0)

        # Determine gmu variant from software string
        is_mempress = 'mempress' in software

        return {
            'software':    software,
            'is_mempress': is_mempress,
            'model':       model,
            'config':      config,
            'rate':        rate,
            'throughput':  throughput,
            'duration':    duration,
            'completed':   completed,
            'ttft_median_s': ttft_median,
            'itl_median_ms': itl_median,
        }

    except Exception as e:
        print(f"  Error parsing {result_file}: {e}")
        return None


def load_baseline_from_csv():
    """Load gmu=0.9 and mempress baseline data from existing summary CSVs.

    These files are tracked in Git LFS and not fetched locally, so we
    use the pre-computed CSVs from prior analysis runs instead.
    Returns (gmu09_baselines_df, mempress_baselines_df).
    """
    gmu09_csv = OUTPUT_DIR / 'v0.5.1_summary.csv'
    mempress_csv = OUTPUT_DIR / 'v0.5.1-mempress_peak_throughput.csv'

    baseline_configs = ['no-offload', 'native-offload-20k']

    gmu09_rows = []
    if gmu09_csv.exists():
        df_csv = pd.read_csv(gmu09_csv)
        for _, row in df_csv.iterrows():
            if row['config'] in baseline_configs:
                gmu09_rows.append({
                    'software':    'upstream-llm-d-0.5.1',
                    'is_mempress': False,
                    'model':       row['model'],
                    'config':      row['config'],
                    'rate':        int(row['optimal_rate']),
                    'throughput':  row['peak_throughput_tok_s'],
                    'duration':    120.0,
                    'completed':   0,
                    'ttft_median_s': row.get('ttft_median_s', np.nan),
                    'itl_median_ms': row.get('itl_median_ms', np.nan),
                })
        print(f"  Loaded {len(gmu09_rows)} gmu=0.9 baseline rows from {gmu09_csv}")
    else:
        print(f"  WARNING: {gmu09_csv} not found; no-offload/native baselines will be missing (gmu=0.9)")

    mempress_rows = []
    if mempress_csv.exists():
        df_mp = pd.read_csv(mempress_csv)
        for _, row in df_mp.iterrows():
            if row['config'] in baseline_configs:
                mempress_rows.append({
                    'software':    'upstream-llm-d-0.5.1-mempress',
                    'is_mempress': True,
                    'model':       row['model'],
                    'config':      row['config'],
                    'rate':        int(row['rate']),
                    'throughput':  row['throughput'],
                    'duration':    120.0,
                    'completed':   0,
                    'ttft_median_s': row.get('ttft_median_ms', np.nan) / 1000.0
                                    if 'ttft_median_ms' in row else np.nan,
                    'itl_median_ms': row.get('itl_median_ms', np.nan),
                })
        print(f"  Loaded {len(mempress_rows)} mempress baseline rows from {mempress_csv}")
    else:
        print(f"  WARNING: {mempress_csv} not found; no-offload/native baselines will be missing (mempress)")

    return pd.DataFrame(gmu09_rows), pd.DataFrame(mempress_rows)


def load_results():
    """Load all v0.5.1 lmcache + baseline results (both gmu=0.9 and mempress)."""
    print("Loading GuideLLM results...")

    # Match both 0.5.1 (non-mempress) and 0.5.1-mempress directories
    patterns = [
        '1x2xL40S_upstream-llm-d-0.5.1_*/guidellm-results.json.zst',
        '1x2xL40S_upstream-llm-d-0.5.1-mempress_*/guidellm-results.json.zst',
    ]

    files = []
    for pattern in patterns:
        files.extend(sorted(RESULTS_DIR.glob(pattern)))

    print(f"  Found {len(files)} result files")

    records = []
    for f in files:
        row = extract_guidellm_metrics(f)
        if row:
            records.append(row)

    df_lmcache = pd.DataFrame(records)
    print(f"  Successfully parsed {len(df_lmcache)} records from result files")

    # Keep only lmcache configs (baselines come from CSV)
    df_lmcache = df_lmcache[
        df_lmcache['config'].isin(['lmcache-local', 'lmcache-valkey'])
    ].reset_index(drop=True)
    print(f"  After filtering to lmcache configs: {len(df_lmcache)} records")

    # Load baselines from pre-computed CSVs (Git LFS files not fetched locally)
    df_gmu09_base, df_mempress_base = load_baseline_from_csv()

    # Combine all data
    df = pd.concat([df_lmcache, df_gmu09_base, df_mempress_base], ignore_index=True)
    print(f"  Total records after adding baselines: {len(df)}")
    print(f"  Configs:  {sorted(df['config'].unique())}")
    print(f"  Models:   {sorted(df['model'].unique())}")
    print(f"  Rates:    {sorted(df['rate'].unique())}")
    return df


def find_peak_throughput(df):
    """Return the row with max throughput for each (model, config, is_mempress)."""
    peak_rows = []
    for (model, config, is_mempress), group in df.groupby(['model', 'config', 'is_mempress']):
        if group['throughput'].max() > 0:
            idx = group['throughput'].idxmax()
            peak_rows.append(group.loc[idx].copy())
    return pd.DataFrame(peak_rows).reset_index(drop=True)


def get_peak(peak_df, model, config, is_mempress):
    """Look up peak throughput; return NaN if not found."""
    row = peak_df[
        (peak_df['model'] == model) &
        (peak_df['config'] == config) &
        (peak_df['is_mempress'] == is_mempress)
    ]
    return row['throughput'].values[0] if len(row) > 0 else np.nan


def get_muted_colors(n):
    return sns.color_palette("muted", n)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Peak throughput bar chart (gmu=0.9 + mempress, 4-panel)
# ══════════════════════════════════════════════════════════════════════════════

def plot_peak_throughput(peak_df, output_file):
    """4-panel (2×2) bar chart: peak throughput across configs, grouped by gmu variant."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Two variants: gmu=0.9 (is_mempress=False) and mempress (is_mempress=True)
    variant_labels = {False: 'gmu=0.9', True: 'mempress'}
    variant_hatches = {False: '', True: '//'}
    colors = get_muted_colors(len(LMCACHE_CONFIGS))

    x = np.arange(len(LMCACHE_CONFIGS))
    width = 0.35

    for i, model in enumerate(MODELS):
        ax = axes[i]
        for vi, is_mempress in enumerate([False, True]):
            vals = []
            for config in LMCACHE_CONFIGS:
                v = get_peak(peak_df, model, config, is_mempress)
                vals.append(v if not np.isnan(v) else 0)

            offset = (vi - 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                label=variant_labels[is_mempress],
                color=colors[:len(LMCACHE_CONFIGS)],
                hatch=variant_hatches[is_mempress],
                alpha=0.85 if is_mempress else 1.0
            )
            # Annotate bar heights
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(vals) * 0.01,
                            f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Configuration', fontsize=9)
        ax.set_ylabel('Peak Throughput (tok/s)', fontsize=9)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in LMCACHE_CONFIGS],
                           rotation=15, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('LMCache Peak Throughput: llm-d v0.5.1 (gmu=0.9 vs mempress)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Throughput curves (gmu=0.9 only, 4-panel)
# ══════════════════════════════════════════════════════════════════════════════

def plot_throughput_curves(df, output_file):
    """4-panel throughput vs rate, gmu=0.9 only, lines for each lmcache config."""
    gmu09_df = df[~df['is_mempress']]
    lmcache_curve_configs = ['no-offload', 'lmcache-local', 'lmcache-valkey']
    colors = get_muted_colors(len(lmcache_curve_configs))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, model in enumerate(MODELS):
        ax = axes[i]
        model_df = gmu09_df[gmu09_df['model'] == model]

        for j, config in enumerate(lmcache_curve_configs):
            cfg_df = model_df[model_df['config'] == config].sort_values('rate')
            if cfg_df.empty:
                continue
            ax.plot(cfg_df['rate'], cfg_df['throughput'],
                    marker='o', linewidth=2,
                    label=CONFIG_LABELS[config],
                    color=colors[j])

        ax.set_xlabel('Concurrency (requests)', fontsize=9)
        ax.set_ylabel('Throughput (tok/s)', fontsize=9)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Throughput vs Concurrency: llm-d v0.5.1 LMCache (gmu=0.9)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Delta heatmap vs no-offload (gmu=0.9)
# ══════════════════════════════════════════════════════════════════════════════

def plot_delta_heatmap(peak_df, output_file):
    """Heatmap of % throughput delta vs no-offload baseline at gmu=0.9."""
    gmu09_peak = peak_df[~peak_df['is_mempress']]
    offload_configs = ['lmcache-local', 'lmcache-valkey', 'native-offload-20k']

    matrix = []
    annot = []
    for config in offload_configs:
        row_vals = []
        row_annot = []
        for model in MODELS:
            baseline_row = gmu09_peak[
                (gmu09_peak['model'] == model) &
                (gmu09_peak['config'] == 'no-offload')
            ]
            cfg_row = gmu09_peak[
                (gmu09_peak['model'] == model) &
                (gmu09_peak['config'] == config)
            ]
            if len(baseline_row) > 0 and len(cfg_row) > 0:
                baseline = baseline_row['throughput'].values[0]
                val = cfg_row['throughput'].values[0]
                if baseline > 0:
                    delta = (val - baseline) / baseline * 100
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

    df_heat = pd.DataFrame(
        matrix,
        index=[CONFIG_LABELS[c] for c in offload_configs],
        columns=MODELS
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        df_heat, annot=annot, fmt='', cmap='magma',
        center=0, vmin=-30, vmax=30,
        cbar_kws={'label': '% vs no-offload baseline'},
        ax=ax, linewidths=0.5
    )
    ax.set_title('Throughput % Delta vs No-Offload Baseline: llm-d v0.5.1 LMCache (gmu=0.9)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Cross-version comparison gmu=0.9
# ══════════════════════════════════════════════════════════════════════════════

def plot_version_comparison(peak_df, v040_data, is_mempress_flag, title_suffix, output_file):
    """Grouped bar chart: v0.4.0 vs v0.5.1 for lmcache-local and lmcache-valkey."""
    gmu_peak = peak_df[peak_df['is_mempress'] == is_mempress_flag]
    lmcache_cfgs = ['lmcache-local', 'lmcache-valkey']
    versions = ['v0.4.0', 'v0.5.1']
    colors = get_muted_colors(2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for panel_idx, config in enumerate(lmcache_cfgs):
        ax = axes[panel_idx]
        x = np.arange(len(MODELS))
        width = 0.35

        v040_vals = [v040_data.get((m, config), np.nan) for m in MODELS]
        v051_vals = [get_peak(gmu_peak, m, config, is_mempress_flag) for m in MODELS]

        for vi, (version, vals, color) in enumerate(zip(versions, [v040_vals, v051_vals], colors)):
            plot_vals = [v if not np.isnan(v) else 0 for v in vals]
            offset = (vi - 0.5) * width
            bars = ax.bar(x + offset, plot_vals, width, label=version, color=color)

            # Annotate bars
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(plot_vals) * 0.01,
                            f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        # Add delta annotations between bars
        for mi, model in enumerate(MODELS):
            v040 = v040_vals[mi]
            v051 = v051_vals[mi]
            if not np.isnan(v040) and not np.isnan(v051) and v040 > 0:
                delta = (v051 - v040) / v040 * 100
                ypos = max(v040, v051) + max(max(v040_vals), max(v051_vals)) * 0.06
                ax.text(mi, ypos, f'{delta:+.1f}%', ha='center', va='bottom',
                        fontsize=7, color='#444444', fontstyle='italic')

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Peak Throughput (tok/s)', fontsize=10)
        ax.set_title(CONFIG_LABELS[config], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Cross-Version LMCache Comparison ({title_suffix}): v0.4.0 vs v0.5.1',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


# ══════════════════════════════════════════════════════════════════════════════
# Console summary tables
# ══════════════════════════════════════════════════════════════════════════════

def print_peak_table(peak_df, is_mempress, label):
    """Print peak throughput table for one gmu variant."""
    gmu_peak = peak_df[peak_df['is_mempress'] == is_mempress]
    print()
    print("=" * 85)
    print(f"v0.5.1 LMCACHE PEAK THROUGHPUT ({label})")
    print("=" * 85)
    print(f"{'Model':<20} {'Config':<22} {'Peak (tok/s)':>12} {'Peak Rate':>10} {'vs no-offload':>14}")
    print("-" * 85)

    for model in MODELS:
        model_peak = gmu_peak[gmu_peak['model'] == model]
        baseline_row = model_peak[model_peak['config'] == 'no-offload']
        baseline = baseline_row['throughput'].values[0] if len(baseline_row) > 0 else np.nan

        for config in LMCACHE_CONFIGS:
            row = model_peak[model_peak['config'] == config]
            if len(row) == 0:
                print(f"  {model:<18} {config:<22} {'N/A':>12}")
                continue
            tput = row['throughput'].values[0]
            rate = int(row['rate'].values[0])

            if config == 'no-offload':
                delta_str = 'baseline'
            elif not np.isnan(baseline) and baseline > 0:
                delta = (tput - baseline) / baseline * 100
                delta_str = f'{delta:+.1f}%'
            else:
                delta_str = 'N/A'

            print(f"  {model:<18} {config:<22} {tput:>12.1f} {rate:>10} {delta_str:>14}")
        print()


def print_cross_version_table(peak_df, v040_data, is_mempress, label):
    """Print cross-version comparison table."""
    gmu_peak = peak_df[peak_df['is_mempress'] == is_mempress]
    lmcache_cfgs = ['lmcache-local', 'lmcache-valkey']

    print()
    print("=" * 85)
    print(f"CROSS-VERSION LMCACHE COMPARISON ({label}): v0.4.0 vs v0.5.1")
    print("=" * 85)
    print(f"{'Model':<20} {'Config':<22} {'v0.4.0':>10} {'v0.5.1':>10} {'Delta':>10}")
    print("-" * 85)

    for config in lmcache_cfgs:
        for model in MODELS:
            v040 = v040_data.get((model, config), np.nan)
            v051_row = gmu_peak[
                (gmu_peak['model'] == model) &
                (gmu_peak['config'] == config)
            ]
            v051 = v051_row['throughput'].values[0] if len(v051_row) > 0 else np.nan

            v040_str = f'{v040:.1f}' if not np.isnan(v040) else 'N/A'
            v051_str = f'{v051:.1f}' if not np.isnan(v051) else 'N/A'

            if not np.isnan(v040) and not np.isnan(v051) and v040 > 0:
                delta = (v051 - v040) / v040 * 100
                delta_str = f'{delta:+.1f}%'
            else:
                delta_str = 'N/A'

            print(f"  {model:<18} {config:<22} {v040_str:>10} {v051_str:>10} {delta_str:>10}")
        print()


def print_latency_table(df):
    """Print TTFT and ITL at rate=50 (gmu=0.9 only)."""
    gmu09_df = df[~df['is_mempress']]
    lmcache_lat_configs = ['no-offload', 'lmcache-local', 'lmcache-valkey']

    print()
    print("=" * 75)
    print("LATENCY AT RATE=50 (gmu=0.9): TTFT and ITL")
    print("=" * 75)
    print(f"{'Model':<20} {'Config':<22} {'TTFT (s)':>10} {'ITL (ms)':>10}")
    print("-" * 75)

    for model in MODELS:
        rate_val = 1 if model == 'Qwen3-32B-AWQ' else 50
        rate_label = f'rate={rate_val}'
        model_df = gmu09_df[(gmu09_df['model'] == model) & (gmu09_df['rate'] == rate_val)]

        for config in lmcache_lat_configs:
            row = model_df[model_df['config'] == config]
            if len(row) == 0:
                print(f"  {model:<18} {config:<22} {'N/A':>10} {'N/A':>10}")
                continue
            ttft = row['ttft_median_s'].values[0]
            itl = row['itl_median_ms'].values[0]
            print(f"  {model:<18} {config:<22} {ttft:>10.3f} {itl:>10.1f}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.chdir(Path(__file__).parent.parent)
    print(f"Working directory: {os.getcwd()}")

    # Load all data
    df = load_results()

    if df.empty:
        print("ERROR: No results loaded.")
        sys.exit(1)

    peak_df = find_peak_throughput(df)

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    # Figure 1: 4-panel peak throughput (gmu=0.9 + mempress)
    plot_peak_throughput(
        peak_df,
        OUTPUT_DIR / 'v0.5.1_lmcache_peak_throughput.png'
    )

    # Figure 2: Throughput curves gmu=0.9
    plot_throughput_curves(
        df,
        OUTPUT_DIR / 'v0.5.1_lmcache_throughput_curves.png'
    )

    # Figure 3: Delta heatmap gmu=0.9
    plot_delta_heatmap(
        peak_df,
        OUTPUT_DIR / 'v0.5.1_lmcache_delta_heatmap.png'
    )

    # Figure 4: Cross-version comparison gmu=0.9
    plot_version_comparison(
        peak_df, V040_GMU09,
        is_mempress_flag=False,
        title_suffix='gmu=0.9',
        output_file=OUTPUT_DIR / 'v0.5.1_lmcache_version_comparison.png'
    )

    # Figure 5: Cross-version comparison mempress
    plot_version_comparison(
        peak_df, V040_MEMPRESS,
        is_mempress_flag=True,
        title_suffix='mempress',
        output_file=OUTPUT_DIR / 'v0.5.1_lmcache_mempress_comparison.png'
    )

    # ── Summary tables ───────────────────────────────────────────────────────

    # Table 1: gmu=0.9 peak throughput
    print_peak_table(peak_df, is_mempress=False, label='gmu=0.9')

    # Table 2: mempress peak throughput
    print_peak_table(peak_df, is_mempress=True, label='mempress')

    # Table 3: Cross-version gmu=0.9
    print_cross_version_table(peak_df, V040_GMU09, is_mempress=False, label='gmu=0.9')

    # Table 4: Cross-version mempress
    print_cross_version_table(peak_df, V040_MEMPRESS, is_mempress=True, label='mempress')

    # Table 5: Latency at rate=50, gmu=0.9
    print_latency_table(df)

    print()
    print("=" * 85)
    print("ANALYSIS COMPLETE")
    print(f"  Figures saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 85)


if __name__ == '__main__':
    main()
