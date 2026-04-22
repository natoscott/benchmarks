#!/usr/bin/env python3
"""Comprehensive analysis of llm-d v0.6.0 KV cache offload benchmarks.

Analyzes benchmark runs across 6 configs x 4 models x 8 rates (gmu=0.9)
and mempress configs x 3 models x 8 rates. Generates visualizations and
summary statistics for report writing.
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
MODELS_MEMPRESS = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B']  # 32B-AWQ offload not run
CONFIGS = ['no-offload', 'native-offload-20k', 'lmcache-local', 'lmcache-valkey',
           'fs-offload', 'cpu+fs-offload-20k']
RATES = [1, 50, 100, 150, 300, 400, 500, 650]

CONFIG_LABELS = {
    'no-offload':         'No Offload',
    'native-offload-20k': 'Native Offload (20k)',
    'lmcache-local':      'LMCache Local',
    'lmcache-valkey':     'LMCache Valkey',
    'fs-offload':         'FS Offload',
    'cpu+fs-offload-20k': 'CPU+FS Offload (20k)',
}

# ── Historical baselines from prior reports ──────────────────────────────────

# v0.4.0 no-offload baselines (from REPORT-v0.4.0.md)
V040_NO_OFFLOAD = {
    'Qwen3-0.6B':    602.0,
    'Qwen3-8B':      113.0,
    'Qwen3-14B':      58.7,
    'Qwen3-32B-AWQ':  49.2,
}

# v0.5.1 gmu=0.9 peak throughput (tok/s)
V051_GMU09 = {
    ('Qwen3-0.6B',    'no-offload'):         636.8,
    ('Qwen3-8B',      'no-offload'):         114.1,
    ('Qwen3-14B',     'no-offload'):          58.7,
    ('Qwen3-32B-AWQ', 'no-offload'):          51.2,
    ('Qwen3-0.6B',    'native-offload-20k'): 622.9,
    ('Qwen3-8B',      'native-offload-20k'):  80.0,
    ('Qwen3-14B',     'native-offload-20k'):  67.2,
    ('Qwen3-32B-AWQ', 'native-offload-20k'):  21.3,
    ('Qwen3-0.6B',    'lmcache-local'):      605.9,
    ('Qwen3-8B',      'lmcache-local'):      113.1,
    ('Qwen3-14B',     'lmcache-local'):       62.9,
    ('Qwen3-32B-AWQ', 'lmcache-local'):       22.4,
    ('Qwen3-0.6B',    'lmcache-valkey'):     606.9,
    ('Qwen3-8B',      'lmcache-valkey'):     115.2,
    ('Qwen3-14B',     'lmcache-valkey'):      62.9,
    ('Qwen3-32B-AWQ', 'lmcache-valkey'):      21.3,
    # fs-offload results (re-run with threads_per_gpu=128)
    ('Qwen3-0.6B',    'fs-offload'):         842.7,
    ('Qwen3-8B',      'fs-offload'):         212.3,
    ('Qwen3-14B',     'fs-offload'):         183.5,
    ('Qwen3-32B-AWQ', 'fs-offload'):          22.4,
    ('Qwen3-0.6B',    'cpu+fs-offload-20k'): 854.4,
    ('Qwen3-8B',      'cpu+fs-offload-20k'): 188.8,
    ('Qwen3-14B',     'cpu+fs-offload-20k'): 169.6,
    ('Qwen3-32B-AWQ', 'cpu+fs-offload-20k'):  21.3,
}

# v0.5.1 mempress peak throughput (tok/s)
V051_MEMPRESS = {
    ('Qwen3-0.6B',    'no-offload'):         526.9,
    ('Qwen3-8B',      'no-offload'):         117.3,
    ('Qwen3-14B',     'no-offload'):          71.5,
    ('Qwen3-32B-AWQ', 'no-offload'):          51.2,
    ('Qwen3-0.6B',    'native-offload-20k'): 644.3,
    ('Qwen3-8B',      'native-offload-20k'): 113.1,
    ('Qwen3-14B',     'native-offload-20k'):  78.9,
    ('Qwen3-32B-AWQ', 'native-offload-20k'):  34.1,
    ('Qwen3-0.6B',    'lmcache-local'):      502.4,
    ('Qwen3-8B',      'lmcache-local'):      119.5,
    ('Qwen3-14B',     'lmcache-local'):       69.3,
    ('Qwen3-32B-AWQ', 'lmcache-local'):       50.1,
    ('Qwen3-0.6B',    'lmcache-valkey'):     499.2,
    ('Qwen3-8B',      'lmcache-valkey'):     118.4,
    ('Qwen3-14B',     'lmcache-valkey'):      70.4,
    ('Qwen3-32B-AWQ', 'lmcache-valkey'):      50.1,
}


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Parse GuideLLM JSON results
# ══════════════════════════════════════════════════════════════════════════════

def parse_dir_name(dir_name):
    """Parse benchmark directory name into components.

    Format: 1x2xL40S_upstream-llm-d-0.6.0_Qwen3-0.6B_no-offload_replica1_rate50
    Or:     1x2xL40S_upstream-llm-d-0.6.0-mempress_Qwen3-0.6B_lmcache-local_replica1_rate50
    """
    parts = dir_name.split('_')
    hardware = parts[0]
    software = parts[1]
    model = parts[2]
    replicas = int(parts[-2].replace('replica', ''))
    rate = int(parts[-1].replace('rate', ''))
    config = '_'.join(parts[3:-2])
    is_mempress = 'mempress' in software
    return hardware, software, model, config, replicas, rate, is_mempress


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
        _, software, model, config, replicas, rate, is_mempress = parse_dir_name(dir_name)

        # Only process v0.6.0 results
        if '0.6.0' not in software:
            return None

        benchmarks = data.get('benchmarks', [])
        if not benchmarks:
            return None

        benchmark = benchmarks[0]
        metrics = benchmark.get('metrics', {})
        duration = benchmark.get('duration', 0)

        # ── Throughput ──────────────────────────────────────────────────────
        # Compute from total successful output tokens / benchmark duration.
        throughput = 0.0
        otc = metrics.get('output_token_count', {})
        if otc and duration > 0:
            total_tokens = otc.get('successful', {}).get('total_sum', 0)
            if total_tokens == 0:
                total_tokens = otc.get('successful', {}).get('sum', 0)
            throughput = total_tokens / duration

        # ── Latency ─────────────────────────────────────────────────────────
        ttft_ms = metrics.get('time_to_first_token_ms', {})
        ttft_median = ttft_ms.get('successful', {}).get('median', 0.0) / 1000.0  # ms->s

        itl_ms = metrics.get('inter_token_latency_ms', {})
        itl_median = itl_ms.get('successful', {}).get('median', 0.0)

        tpot_ms = metrics.get('time_per_output_token_ms', {})
        tpot_median = tpot_ms.get('successful', {}).get('median', 0.0)

        # ── Request counts ───────────────────────────────────────────────────
        request_totals = metrics.get('request_totals', {})
        completed = request_totals.get('completed', 0)
        if completed == 0:
            completed = request_totals.get('successful', 0)
        errored = request_totals.get('errored', 0)
        total = request_totals.get('total', max(completed, 1))

        # Compute error rate; flag high-error runs as unreliable.
        # vLLM #38515 causes vLLM to crash and restart, resulting in thousands
        # of errors in a single 120-second window. Runs where errored requests
        # exceed 10% of total are flagged; throughput from these runs is
        # unrepresentative (inflated by fast-failing requests).
        error_rate = errored / total if total > 0 else 0.0
        high_error = error_rate > 0.10

        return {
            'model':         model,
            'config':        config,
            'rate':          rate,
            'is_mempress':   is_mempress,
            'throughput':    throughput,
            'duration':      duration,
            'completed':     completed,
            'errored':       errored,
            'error_rate':    error_rate,
            'high_error':    high_error,
            'ttft_median_s': ttft_median,
            'itl_median_ms': itl_median,
            'tpot_median_ms': tpot_median,
        }

    except Exception as e:
        print(f"  Error parsing {result_file}: {e}")
        return None


def load_all_results():
    """Load all v0.6.0 GuideLLM results (gmu=0.9 and mempress)."""
    print("Loading GuideLLM results...")
    patterns = [
        '1x2xL40S_upstream-llm-d-0.6.0_*/guidellm-results.json.zst',
        '1x2xL40S_upstream-llm-d-0.6.0-mempress_*/guidellm-results.json.zst',
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

    df = pd.DataFrame(records)
    print(f"  Successfully parsed {len(df)} records")
    if 'high_error' in df.columns:
        n_high_error = df['high_error'].sum()
        if n_high_error > 0:
            print(f"  WARNING: {n_high_error} runs have error_rate >10% (vLLM crash/restart; excluded from peak selection)")
            bad = df[df['high_error']][['model','config','is_mempress','rate','throughput','error_rate']]
            for _, row in bad.iterrows():
                mp = 'mempress' if row['is_mempress'] else 'gmu=0.9'
                print(f"    {row['model']}/{row['config']}/{mp}/rate={row['rate']}: {row['throughput']:.1f} tok/s (err={row['error_rate']*100:.0f}%)")
    return df


def find_peak_throughput(df, group_cols=None):
    """Find peak throughput row for each combination of group_cols.

    Excludes runs flagged as high_error (error_rate > 10%) to avoid
    selecting artificially high throughput values from vLLM crash/restart
    scenarios (vLLM #38515).
    """
    if group_cols is None:
        group_cols = ['model', 'config', 'is_mempress']
    peak_rows = []
    for keys, group in df.groupby(group_cols):
        # Prefer clean runs; fall back to all runs only if no clean ones exist
        clean = group[~group['high_error']] if 'high_error' in group.columns else group
        if clean.empty:
            clean = group
        if clean['throughput'].max() > 0:
            idx = clean['throughput'].idxmax()
            peak_rows.append(clean.loc[idx].copy())
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
# Part 2: Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def plot_peak_throughput_gmu09(peak_df, output_file):
    """Figure 1: 4-panel bar chart of peak throughput at gmu=0.9, with v0.5.1 comparison."""
    gmu09 = peak_df[~peak_df['is_mempress']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # 2 version groups: v0.5.1 and v0.6.0
    versions = ['v0.5.1', 'v0.6.0']
    colors = get_muted_colors(len(CONFIGS))
    x = np.arange(len(CONFIGS))
    width = 0.35

    for i, model in enumerate(MODELS):
        ax = axes[i]

        v051_vals = [V051_GMU09.get((model, c), np.nan) for c in CONFIGS]
        v060_vals = []
        for c in CONFIGS:
            row = gmu09[(gmu09['model'] == model) & (gmu09['config'] == c)]
            v060_vals.append(row['throughput'].values[0] if len(row) > 0 else np.nan)

        for vi, (version, vals) in enumerate(zip(versions, [v051_vals, v060_vals])):
            plot_vals = [v if not np.isnan(v) else 0 for v in vals]
            offset = (vi - 0.5) * width
            bars = ax.bar(
                x + offset, plot_vals, width,
                label=version,
                color=colors[:len(CONFIGS)],
                alpha=0.7 if vi == 0 else 1.0,
                hatch='//' if vi == 0 else ''
            )
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(plot_vals) * 0.01,
                            f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Configuration', fontsize=9)
        ax.set_ylabel('Peak Throughput (tok/s)', fontsize=9)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS], rotation=15, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Peak Throughput by Configuration: llm-d v0.6.0 vs v0.5.1 (gmu=0.9)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_throughput_curves_gmu09(df, output_file):
    """Figure 2: 4-panel throughput vs concurrency at gmu=0.9, all 4 configs."""
    gmu09 = df[~df['is_mempress']]
    colors = get_muted_colors(len(CONFIGS))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, model in enumerate(MODELS):
        ax = axes[i]
        model_data = gmu09[gmu09['model'] == model]

        for j, config in enumerate(CONFIGS):
            cfg_data = model_data[model_data['config'] == config].sort_values('rate')
            if cfg_data.empty:
                continue
            ax.plot(cfg_data['rate'], cfg_data['throughput'],
                    marker='o', label=CONFIG_LABELS[config], linewidth=2,
                    color=colors[j])

        ax.set_xlabel('Concurrency (requests)', fontsize=10)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=10)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Throughput vs Concurrency: llm-d v0.6.0 (gmu=0.9)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_delta_heatmap(peak_df, output_file):
    """Figure 3: Heatmap of throughput delta (%) vs no-offload baseline at gmu=0.9. magma colormap."""
    gmu09 = peak_df[~peak_df['is_mempress']]
    offload_configs = ['native-offload-20k', 'lmcache-local', 'lmcache-valkey',
                       'fs-offload', 'cpu+fs-offload-20k']

    matrix = []
    annot = []
    for config in offload_configs:
        row_vals = []
        row_annot = []
        for model in MODELS:
            no_off = gmu09[(gmu09['model'] == model) & (gmu09['config'] == 'no-offload')]
            cfg_row = gmu09[(gmu09['model'] == model) & (gmu09['config'] == config)]
            if len(no_off) > 0 and len(cfg_row) > 0:
                baseline = no_off['throughput'].values[0]
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

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(df_heat, annot=annot, fmt='', cmap='magma',
                center=0, vmin=-70, vmax=220,
                cbar_kws={'label': '% vs no-offload baseline'},
                ax=ax, linewidths=0.5)
    ax.set_title('Throughput % Delta vs No-Offload Baseline: llm-d v0.6.0 (gmu=0.9)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_version_comparison(peak_df, configs, title_suffix, output_file):
    """Figure 4/5: v0.5.1 vs v0.6.0 peak throughput for specified configs at gmu=0.9."""
    gmu09 = peak_df[~peak_df['is_mempress']]
    versions = ['v0.5.1', 'v0.6.0']
    colors = get_muted_colors(2)

    n_panels = len(configs)
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 7))
    if n_panels == 1:
        axes = [axes]

    for panel_idx, config in enumerate(configs):
        ax = axes[panel_idx]
        x = np.arange(len(MODELS))
        width = 0.35

        v051_vals = [V051_GMU09.get((m, config), np.nan) for m in MODELS]
        v060_vals = []
        for model in MODELS:
            row = gmu09[(gmu09['model'] == model) & (gmu09['config'] == config)]
            v060_vals.append(row['throughput'].values[0] if len(row) > 0 else np.nan)

        for vi, (version, vals, color) in enumerate(zip(versions, [v051_vals, v060_vals], colors)):
            plot_vals = [v if not np.isnan(v) else 0 for v in vals]
            offset = (vi - 0.5) * width
            bars = ax.bar(x + offset, plot_vals, width, label=version, color=color)
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(plot_vals) * 0.01,
                            f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        # Delta annotations
        for mi, model in enumerate(MODELS):
            v051 = v051_vals[mi]
            v060 = v060_vals[mi]
            if not np.isnan(v051) and not np.isnan(v060) and v051 > 0:
                delta = (v060 - v051) / v051 * 100
                ypos = max([v for v in [v051, v060] if not np.isnan(v)]) + \
                       max([v for v in v060_vals if not np.isnan(v)]) * 0.07
                ax.text(mi, ypos, f'{delta:+.1f}%', ha='center', va='bottom',
                        fontsize=7, color='#444444', fontstyle='italic')

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Peak Throughput (tok/s)', fontsize=10)
        ax.set_title(CONFIG_LABELS[config], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Version Comparison (gmu=0.9): v0.5.1 vs v0.6.0 — {title_suffix}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_mempress_peak_throughput(peak_df, output_file):
    """Figure 6: mempress peak throughput for 3 models, 4 configs, grouped bars."""
    mempress = peak_df[peak_df['is_mempress']]
    colors = get_muted_colors(len(CONFIGS))

    fig, axes = plt.subplots(1, 3, figsize=(14, 7))

    x = np.arange(len(CONFIGS))
    width = 0.35

    for i, model in enumerate(MODELS_MEMPRESS):
        ax = axes[i]

        v051_vals = [V051_MEMPRESS.get((model, c), np.nan) for c in CONFIGS]
        v060_vals = []
        for c in CONFIGS:
            row = mempress[(mempress['model'] == model) & (mempress['config'] == c)]
            v060_vals.append(row['throughput'].values[0] if len(row) > 0 else np.nan)

        for vi, (version, vals) in enumerate(zip(['v0.5.1', 'v0.6.0'], [v051_vals, v060_vals])):
            plot_vals = [v if not np.isnan(v) else 0 for v in vals]
            offset = (vi - 0.5) * width
            bars = ax.bar(
                x + offset, plot_vals, width,
                label=version,
                color=colors[:len(CONFIGS)],
                alpha=0.7 if vi == 0 else 1.0,
                hatch='//' if vi == 0 else ''
            )
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(plot_vals) * 0.01,
                            f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Configuration', fontsize=9)
        ax.set_ylabel('Peak Throughput (tok/s)', fontsize=9)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS], rotation=20, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Memory-Pressure Peak Throughput: llm-d v0.6.0 vs v0.5.1 (3 models)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


def plot_mempress_version_comparison(peak_df, output_file):
    """Figure 7: v0.5.1 vs v0.6.0 mempress native-offload-20k for 3 models."""
    mempress = peak_df[peak_df['is_mempress']]
    compare_configs = ['no-offload', 'native-offload-20k']
    versions = ['v0.5.1', 'v0.6.0']
    colors = get_muted_colors(2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    x = np.arange(len(MODELS_MEMPRESS))
    width = 0.35

    for panel_idx, config in enumerate(compare_configs):
        ax = axes[panel_idx]

        v051_vals = [V051_MEMPRESS.get((m, config), np.nan) for m in MODELS_MEMPRESS]
        v060_vals = []
        for model in MODELS_MEMPRESS:
            row = mempress[(mempress['model'] == model) & (mempress['config'] == config)]
            v060_vals.append(row['throughput'].values[0] if len(row) > 0 else np.nan)

        for vi, (version, vals, color) in enumerate(zip(versions, [v051_vals, v060_vals], colors)):
            plot_vals = [v if not np.isnan(v) else 0 for v in vals]
            offset = (vi - 0.5) * width
            bars = ax.bar(x + offset, plot_vals, width, label=version, color=color)
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(plot_vals) * 0.01,
                            f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        # Delta annotations
        for mi, model in enumerate(MODELS_MEMPRESS):
            v051 = v051_vals[mi]
            v060 = v060_vals[mi]
            if not np.isnan(v051) and not np.isnan(v060) and v051 > 0:
                delta = (v060 - v051) / v051 * 100
                ypos = max([v for v in [v051, v060] if not np.isnan(v)]) + \
                       max([v for v in v060_vals if not np.isnan(v)]) * 0.07
                ax.text(mi, ypos, f'{delta:+.1f}%', ha='center', va='bottom',
                        fontsize=7, color='#444444', fontstyle='italic')

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Peak Throughput (tok/s)', fontsize=10)
        ax.set_title(CONFIG_LABELS[config], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS_MEMPRESS, rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Memory-Pressure Version Comparison: v0.5.1 vs v0.6.0',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_file}")


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Console summary tables
# ══════════════════════════════════════════════════════════════════════════════

def print_peak_throughput_table_gmu09(peak_df):
    """Table 1: v0.6.0 peak throughput (gmu=0.9) with % vs baseline and vs v0.5.1."""
    gmu09 = peak_df[~peak_df['is_mempress']]

    print()
    print("=" * 105)
    print("TABLE 1: v0.6.0 PEAK THROUGHPUT (gmu=0.9)")
    print("=" * 105)
    print(f"{'Model':<20} {'Config':<22} {'v0.6.0 (tok/s)':>14} {'Rate':>6} {'vs no-offload':>14} {'vs v0.5.1':>10}")
    print("-" * 105)

    for model in MODELS:
        model_data = gmu09[gmu09['model'] == model]
        no_off_row = model_data[model_data['config'] == 'no-offload']
        baseline_060 = no_off_row['throughput'].values[0] if len(no_off_row) > 0 else None

        for config in CONFIGS:
            row = model_data[model_data['config'] == config]
            if len(row) == 0:
                continue
            tput = row['throughput'].values[0]
            rate = int(row['rate'].values[0])

            if config == 'no-offload':
                delta_no_off = 'baseline'
            elif baseline_060 and baseline_060 > 0:
                d = (tput - baseline_060) / baseline_060 * 100
                delta_no_off = f'{d:+.1f}%'
            else:
                delta_no_off = 'N/A'

            v051_val = V051_GMU09.get((model, config), np.nan)
            if not np.isnan(v051_val) and v051_val > 0:
                d051 = (tput - v051_val) / v051_val * 100
                delta_051 = f'{d051:+.1f}%'
            else:
                delta_051 = 'N/A'

            print(f"  {model:<18} {config:<22} {tput:>14.1f} {rate:>6} {delta_no_off:>14} {delta_051:>10}")
        print()


def print_mempress_peak_table(peak_df):
    """Table 2: v0.6.0 mempress peak throughput with % vs baseline and vs v0.5.1."""
    mempress = peak_df[peak_df['is_mempress']]

    print()
    print("=" * 105)
    print("TABLE 2: v0.6.0 MEMPRESS PEAK THROUGHPUT")
    print("=" * 105)
    print(f"{'Model':<20} {'Config':<22} {'v0.6.0 (tok/s)':>14} {'Rate':>6} {'vs no-offload':>14} {'vs v0.5.1':>10}")
    print("-" * 105)

    # 3 models for mempress (32B-AWQ offload not run)
    mempress_models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B']
    mempress_configs = ['no-offload', 'native-offload-20k', 'lmcache-local', 'lmcache-valkey']

    for model in mempress_models:
        model_data = mempress[mempress['model'] == model]
        no_off_row = model_data[model_data['config'] == 'no-offload']
        baseline_060 = no_off_row['throughput'].values[0] if len(no_off_row) > 0 else None

        for config in mempress_configs:
            row = model_data[model_data['config'] == config]
            if len(row) == 0:
                print(f"  {model:<18} {config:<22} {'N/A':>14}")
                continue
            tput = row['throughput'].values[0]
            rate = int(row['rate'].values[0])

            if config == 'no-offload':
                delta_no_off = 'baseline'
            elif baseline_060 and baseline_060 > 0:
                d = (tput - baseline_060) / baseline_060 * 100
                delta_no_off = f'{d:+.1f}%'
            else:
                delta_no_off = 'N/A'

            v051_val = V051_MEMPRESS.get((model, config), np.nan)
            if not np.isnan(v051_val) and v051_val > 0:
                d051 = (tput - v051_val) / v051_val * 100
                delta_051 = f'{d051:+.1f}%'
            else:
                delta_051 = 'N/A'

            print(f"  {model:<18} {config:<22} {tput:>14.1f} {rate:>6} {delta_no_off:>14} {delta_051:>10}")

    # 32B-AWQ no-offload only
    model = 'Qwen3-32B-AWQ'
    model_data = mempress[mempress['model'] == model]
    no_off_row = model_data[model_data['config'] == 'no-offload']
    if len(no_off_row) > 0:
        tput = no_off_row['throughput'].values[0]
        rate = int(no_off_row['rate'].values[0])
        v051_val = V051_MEMPRESS.get((model, 'no-offload'), np.nan)
        d051_str = f'{(tput - v051_val) / v051_val * 100:+.1f}%' if not np.isnan(v051_val) else 'N/A'
        print(f"  {model:<18} {'no-offload':<22} {tput:>14.1f} {rate:>6} {'baseline':>14} {d051_str:>10}")
        for config in ['native-offload-20k', 'lmcache-local', 'lmcache-valkey']:
            print(f"  {model:<18} {config:<22} {'not run':>14} {'':>6} {'(vLLM #38515)':>14} {'':>10}")
    print()


def print_no_offload_crossversion(peak_df):
    """Table 3: Cross-version no-offload baseline comparison."""
    gmu09 = peak_df[~peak_df['is_mempress']]

    print()
    print("=" * 90)
    print("TABLE 3: NO-OFFLOAD BASELINE CROSS-VERSION (v0.4.0 → v0.5.1 → v0.6.0, gmu=0.9)")
    print("=" * 90)
    print(f"{'Model':<20} {'v0.4.0':>10} {'v0.5.1':>10} {'v0.6.0':>10} {'v051→v060':>12}")
    print("-" * 90)

    for model in MODELS:
        v040 = V040_NO_OFFLOAD.get(model, np.nan)
        v051 = V051_GMU09.get((model, 'no-offload'), np.nan)
        row = gmu09[(gmu09['model'] == model) & (gmu09['config'] == 'no-offload')]
        v060 = row['throughput'].values[0] if len(row) > 0 else np.nan

        v040_s = f'{v040:.1f}' if not np.isnan(v040) else 'N/A'
        v051_s = f'{v051:.1f}' if not np.isnan(v051) else 'N/A'
        v060_s = f'{v060:.1f}' if not np.isnan(v060) else 'N/A'

        if not np.isnan(v051) and not np.isnan(v060) and v051 > 0:
            delta_s = f'{(v060 - v051) / v051 * 100:+.1f}%'
        else:
            delta_s = 'N/A'

        print(f"  {model:<18} {v040_s:>10} {v051_s:>10} {v060_s:>10} {delta_s:>12}")
    print()


def print_native_offload_crossversion(peak_df):
    """Table 4: Cross-version native-offload-20k comparison."""
    gmu09 = peak_df[~peak_df['is_mempress']]

    print()
    print("=" * 90)
    print("TABLE 4: NATIVE-OFFLOAD-20K CROSS-VERSION (v0.5.1 → v0.6.0, gmu=0.9)")
    print("=" * 90)
    print(f"{'Model':<20} {'v0.5.1':>10} {'v0.6.0':>10} {'Delta':>10} {'v0.5.1 vs nooff':>16} {'v0.6.0 vs nooff':>16}")
    print("-" * 90)

    for model in MODELS:
        v051 = V051_GMU09.get((model, 'native-offload-20k'), np.nan)
        row = gmu09[(gmu09['model'] == model) & (gmu09['config'] == 'no-offload')]
        no_off_060 = row['throughput'].values[0] if len(row) > 0 else np.nan

        row2 = gmu09[(gmu09['model'] == model) & (gmu09['config'] == 'native-offload-20k')]
        v060 = row2['throughput'].values[0] if len(row2) > 0 else np.nan

        v051_s = f'{v051:.1f}' if not np.isnan(v051) else 'N/A'
        v060_s = f'{v060:.1f}' if not np.isnan(v060) else 'N/A'

        if not np.isnan(v051) and not np.isnan(v060) and v051 > 0:
            delta_s = f'{(v060 - v051) / v051 * 100:+.1f}%'
        else:
            delta_s = 'N/A'

        # vs same-version no-offload
        v051_nooff = V051_GMU09.get((model, 'no-offload'), np.nan)
        if not np.isnan(v051) and not np.isnan(v051_nooff) and v051_nooff > 0:
            d051_nooff = f'{(v051 - v051_nooff) / v051_nooff * 100:+.1f}%'
        else:
            d051_nooff = 'N/A'

        if not np.isnan(v060) and not np.isnan(no_off_060) and no_off_060 > 0:
            d060_nooff = f'{(v060 - no_off_060) / no_off_060 * 100:+.1f}%'
        else:
            d060_nooff = 'N/A'

        print(f"  {model:<18} {v051_s:>10} {v060_s:>10} {delta_s:>10} {d051_nooff:>16} {d060_nooff:>16}")
    print()


def print_lmcache_crossversion(peak_df):
    """Table 5: Cross-version lmcache comparison."""
    gmu09 = peak_df[~peak_df['is_mempress']]
    lmcache_cfgs = ['lmcache-local', 'lmcache-valkey']

    print()
    print("=" * 90)
    print("TABLE 5: LMCACHE CROSS-VERSION (v0.5.1 → v0.6.0, gmu=0.9)")
    print("=" * 90)
    print(f"{'Model':<20} {'Config':<22} {'v0.5.1':>10} {'v0.6.0':>10} {'Delta':>10}")
    print("-" * 90)

    for config in lmcache_cfgs:
        for model in MODELS:
            v051 = V051_GMU09.get((model, config), np.nan)
            row = gmu09[(gmu09['model'] == model) & (gmu09['config'] == config)]
            v060 = row['throughput'].values[0] if len(row) > 0 else np.nan

            v051_s = f'{v051:.1f}' if not np.isnan(v051) else 'N/A'
            v060_s = f'{v060:.1f}' if not np.isnan(v060) else 'N/A'
            if not np.isnan(v051) and not np.isnan(v060) and v051 > 0:
                delta_s = f'{(v060 - v051) / v051 * 100:+.1f}%'
            else:
                delta_s = 'N/A'
            print(f"  {model:<18} {config:<22} {v051_s:>10} {v060_s:>10} {delta_s:>10}")
        print()


def print_fs_crossversion(peak_df):
    """Table 6: Cross-version fs-offload and cpu+fs-offload comparison."""
    gmu09 = peak_df[~peak_df['is_mempress']]
    fs_cfgs = ['fs-offload', 'cpu+fs-offload-20k']

    print()
    print("=" * 105)
    print("TABLE 6: FILESYSTEM OFFLOAD CROSS-VERSION (v0.5.1 → v0.6.0, gmu=0.9)")
    print("=" * 105)
    print(f"{'Model':<20} {'Config':<24} {'v0.5.1':>10} {'v0.5.1 vs nooff':>16} {'v0.6.0':>10} {'v0.6.0 vs nooff':>16} {'Delta':>10}")
    print("-" * 105)

    for config in fs_cfgs:
        for model in MODELS:
            v051 = V051_GMU09.get((model, config), np.nan)
            v051_nooff = V051_GMU09.get((model, 'no-offload'), np.nan)
            row = gmu09[(gmu09['model'] == model) & (gmu09['config'] == config)]
            v060 = row['throughput'].values[0] if len(row) > 0 else np.nan
            nooff_row = gmu09[(gmu09['model'] == model) & (gmu09['config'] == 'no-offload')]
            v060_nooff = nooff_row['throughput'].values[0] if len(nooff_row) > 0 else np.nan

            v051_s = f'{v051:.1f}' if not np.isnan(v051) else 'N/A'
            v060_s = f'{v060:.1f}' if not np.isnan(v060) else 'N/A'

            if not np.isnan(v051) and not np.isnan(v051_nooff) and v051_nooff > 0:
                d051_nooff = f'{(v051 - v051_nooff) / v051_nooff * 100:+.1f}%'
            else:
                d051_nooff = 'N/A'
            if not np.isnan(v060) and not np.isnan(v060_nooff) and v060_nooff > 0:
                d060_nooff = f'{(v060 - v060_nooff) / v060_nooff * 100:+.1f}%'
            else:
                d060_nooff = 'N/A'
            if not np.isnan(v051) and not np.isnan(v060) and v051 > 0:
                delta_s = f'{(v060 - v051) / v051 * 100:+.1f}%'
            else:
                delta_s = 'N/A'
            print(f"  {model:<18} {config:<24} {v051_s:>10} {d051_nooff:>16} {v060_s:>10} {d060_nooff:>16} {delta_s:>10}")
        print()


def print_latency_rate50(df):
    """Table 7: Latency at rate=50, gmu=0.9, all configs."""
    gmu09 = df[~df['is_mempress']]

    print()
    print("=" * 90)
    print("TABLE 7: LATENCY AT RATE=50 (gmu=0.9): TTFT and ITL")
    print("=" * 90)
    print(f"{'Model':<20} {'Config':<24} {'TTFT (s)':>10} {'ITL (ms)':>10}")
    print("-" * 90)

    for model in MODELS:
        rate_val = 1 if model == 'Qwen3-32B-AWQ' else 50
        model_data = gmu09[(gmu09['model'] == model) & (gmu09['rate'] == rate_val)]

        for config in CONFIGS:
            row = model_data[model_data['config'] == config]
            if len(row) == 0:
                print(f"  {model:<18} {config:<24} {'N/A':>10} {'N/A':>10}")
                continue
            ttft = row['ttft_median_s'].values[0]
            itl = row['itl_median_ms'].values[0]
            print(f"  {model:<18} {config:<24} {ttft:>10.3f} {itl:>10.1f}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.chdir(Path(__file__).parent.parent)
    print(f"Working directory: {os.getcwd()}")

    # ── Step 1: Load results ─────────────────────────────────────────────────
    df = load_all_results()

    if df.empty:
        print("ERROR: No GuideLLM results loaded. Check results directory.")
        sys.exit(1)

    configs_found = sorted(df['config'].unique())
    models_found = sorted(df['model'].unique())
    rates_found = sorted(df['rate'].unique())
    print(f"Configs found: {configs_found}")
    print(f"Models found:  {models_found}")
    print(f"Rates found:   {rates_found}")
    print(f"Mempress runs: {df['is_mempress'].sum()} / {len(df)} total")

    # Save all data
    df.to_csv(OUTPUT_DIR / 'v0.6.0_all_runs.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'v0.6.0_all_runs.csv'}")

    peak_df = find_peak_throughput(df)
    peak_df.to_csv(OUTPUT_DIR / 'v0.6.0_peak_throughput.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'v0.6.0_peak_throughput.csv'}")

    # ── Step 2: Figures ──────────────────────────────────────────────────────
    print("\nGenerating figures...")

    plot_peak_throughput_gmu09(peak_df, OUTPUT_DIR / 'v0.6.0_peak_throughput.png')
    plot_throughput_curves_gmu09(df, OUTPUT_DIR / 'v0.6.0_throughput_curves.png')
    plot_delta_heatmap(peak_df, OUTPUT_DIR / 'v0.6.0_delta_heatmap.png')
    plot_version_comparison(peak_df,
                            ['no-offload', 'native-offload-20k'],
                            'No Offload and Native Offload',
                            OUTPUT_DIR / 'v0.6.0_version_comparison.png')
    plot_version_comparison(peak_df,
                            ['lmcache-local', 'lmcache-valkey'],
                            'LMCache',
                            OUTPUT_DIR / 'v0.6.0_lmcache_version_comparison.png')
    plot_version_comparison(peak_df,
                            ['fs-offload', 'cpu+fs-offload-20k'],
                            'Filesystem Offload',
                            OUTPUT_DIR / 'v0.6.0_fs_version_comparison.png')
    plot_mempress_peak_throughput(peak_df, OUTPUT_DIR / 'v0.6.0_mempress_peak_throughput.png')
    plot_mempress_version_comparison(peak_df, OUTPUT_DIR / 'v0.6.0_mempress_version_comparison.png')

    # ── Step 3: Console tables ───────────────────────────────────────────────
    print_peak_throughput_table_gmu09(peak_df)
    print_mempress_peak_table(peak_df)
    print_no_offload_crossversion(peak_df)
    print_native_offload_crossversion(peak_df)
    print_lmcache_crossversion(peak_df)
    print_fs_crossversion(peak_df)
    print_latency_rate50(df)

    print()
    print("=" * 90)
    print("ANALYSIS COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 90)


if __name__ == '__main__':
    main()
