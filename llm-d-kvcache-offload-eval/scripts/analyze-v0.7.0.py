#!/usr/bin/env python3
"""Comprehensive analysis of llm-d v0.7.0 KV cache offload benchmarks.

Analyzes benchmark runs across 6 configs x 4 models x 8 rates (gmu=0.9)
and 6 mempress configs x 4 models x 8 rates. Generates visualizations and
summary statistics for report writing.
"""

import json
import subprocess
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

# v0.5.1 gmu=0.9 peak throughput (tok/s) — from REPORT-v0.5.1.md
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
    ('Qwen3-0.6B',    'fs-offload'):         842.7,
    ('Qwen3-8B',      'fs-offload'):         212.3,
    ('Qwen3-14B',     'fs-offload'):         183.5,
    ('Qwen3-32B-AWQ', 'fs-offload'):          22.4,
    ('Qwen3-0.6B',    'cpu+fs-offload-20k'): 854.4,
    ('Qwen3-8B',      'cpu+fs-offload-20k'): 188.8,
    ('Qwen3-14B',     'cpu+fs-offload-20k'): 169.6,
    ('Qwen3-32B-AWQ', 'cpu+fs-offload-20k'):  21.3,
}

# v0.5.1 mempress peak throughput (tok/s) — from REPORT-v0.5.1.md
V051_MEMPRESS = {
    ('Qwen3-0.6B',    'no-offload'):         526.9,
    ('Qwen3-8B',      'no-offload'):         108.4,
    ('Qwen3-14B',     'no-offload'):          55.8,
    ('Qwen3-0.6B',    'native-offload-20k'): 799.0,
    ('Qwen3-8B',      'native-offload-20k'): 134.2,
    ('Qwen3-14B',     'native-offload-20k'): 107.1,
}

# v0.6.0 gmu=0.9 peak throughput (tok/s) — computed from results/.
# fs-offload and cpu+fs-offload-20k excluded: v0.6.0 fs-offload runs used a
# system misconfiguration (PVC) that produced invalid results. v0.7.0 is
# the first properly configured baseline for these configurations.
V060_GMU09 = {
    ('Qwen3-0.6B',    'no-offload'):         807.5,
    ('Qwen3-8B',      'no-offload'):         197.3,
    ('Qwen3-14B',     'no-offload'):          55.5,
    ('Qwen3-32B-AWQ', 'no-offload'):          50.1,
    ('Qwen3-0.6B',    'native-offload-20k'): 809.6,
    ('Qwen3-8B',      'native-offload-20k'): 184.5,
    ('Qwen3-14B',     'native-offload-20k'):  56.5,
    ('Qwen3-32B-AWQ', 'native-offload-20k'):  18.1,
    ('Qwen3-0.6B',    'lmcache-local'):      665.6,
    ('Qwen3-8B',      'lmcache-local'):      194.1,
    ('Qwen3-14B',     'lmcache-local'):       54.4,
    ('Qwen3-32B-AWQ', 'lmcache-local'):       18.1,
    ('Qwen3-0.6B',    'lmcache-valkey'):     789.3,
    ('Qwen3-8B',      'lmcache-valkey'):     141.9,
    ('Qwen3-14B',     'lmcache-valkey'):      56.5,
    ('Qwen3-32B-AWQ', 'lmcache-valkey'):      19.2,
}

# v0.6.0 mempress peak throughput (tok/s) — computed from results/.
# fs-offload and cpu+fs-offload-20k excluded (see V060_GMU09 note above).
V060_MEMPRESS = {
    ('Qwen3-0.6B',    'no-offload'):         524.8,
    ('Qwen3-8B',      'no-offload'):         104.5,
    ('Qwen3-14B',     'no-offload'):          69.3,
    ('Qwen3-32B-AWQ', 'no-offload'):          50.1,
    ('Qwen3-0.6B',    'native-offload-20k'): 794.7,
    ('Qwen3-8B',      'native-offload-20k'):  87.5,
    ('Qwen3-14B',     'native-offload-20k'): 569.8,
    ('Qwen3-32B-AWQ', 'native-offload-20k'):  50.1,
    ('Qwen3-0.6B',    'lmcache-local'):      426.7,
    ('Qwen3-8B',      'lmcache-local'):      104.5,
    ('Qwen3-14B',     'lmcache-local'):       68.3,
    ('Qwen3-0.6B',    'lmcache-valkey'):     474.7,
    ('Qwen3-8B',      'lmcache-valkey'):     107.7,
    ('Qwen3-14B',     'lmcache-valkey'):      68.3,
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_guidellm(path: Path) -> dict | None:
    """Decompress and parse a guidellm-results.json.zst file."""
    try:
        proc = subprocess.run(['zstd', '-d', '-c', str(path)],
                              capture_output=True, timeout=30)
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except Exception:
        return None


def parse_result(result_dir: Path) -> dict | None:
    """Extract key metrics from one benchmark result directory."""
    name = result_dir.name  # e.g. 1x2xL40S_upstream-llm-d-0.7.0_Qwen3-8B_fs-offload_replica1_rate50
    parts = name.split('_')
    if len(parts) < 6:
        return None

    software = parts[1]
    model_name = parts[2]
    config = parts[3]
    replicas = int(parts[4].replace('replica', ''))
    rate = int(parts[5].replace('rate', ''))

    if '0.7.0' not in software:
        return None

    mempress = 'mempress' in software

    result_file = result_dir / 'guidellm-results.json.zst'
    if not result_file.exists():
        return None

    data = load_guidellm(result_file)
    if not data:
        return None

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
        throughput = total_tokens / duration if duration > 0 else 0

    # TTFT (ms)
    ttft = metrics.get('time_to_first_token_ms', {}).get('successful', {}).get('mean', 0)

    # ITL — inter_token_latency_ms (true TPOT excluding TTFT)
    itl = metrics.get('inter_token_latency_ms', {}).get('successful', {}).get('mean', 0)

    # Request count
    req_totals = metrics.get('request_totals', {})
    successful = req_totals.get('successful', 0)

    return {
        'software':   software,
        'model':      model_name,
        'config':     config,
        'rate':       rate,
        'replicas':   replicas,
        'mempress':   mempress,
        'throughput': throughput,
        'ttft_ms':    ttft,
        'itl_ms':     itl,
        'successful': successful,
        'duration':   duration,
    }


def load_all_results() -> pd.DataFrame:
    """Load all v0.7.0 benchmark results."""
    rows = []
    for result_dir in sorted(RESULTS_DIR.iterdir()):
        if not result_dir.is_dir():
            continue
        if 'upstream-llm-d-0.7.0' not in result_dir.name:
            continue
        row = parse_result(result_dir)
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} v0.7.0 results")
    return df


# ── Peak throughput helper ────────────────────────────────────────────────────

def peak_throughput(df: pd.DataFrame, model: str, config: str,
                    mempress: bool = False) -> float:
    """Return peak throughput for a given model/config combination."""
    sub = df[(df['model'] == model) & (df['config'] == config) &
             (df['mempress'] == mempress) & (df['successful'] > 0)]
    if sub.empty:
        return 0.0
    return sub['throughput'].max()


# ── Figures ───────────────────────────────────────────────────────────────────

def fig1_throughput_vs_concurrency(df: pd.DataFrame) -> None:
    """Figure 1: Throughput vs concurrency curves for all configs, gmu=0.9."""
    gmu09 = df[~df['mempress']]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    palette = sns.color_palette("muted", len(CONFIGS))

    for i, model in enumerate(MODELS):
        ax = axes[i]
        mdf = gmu09[gmu09['model'] == model]
        for j, config in enumerate(CONFIGS):
            cdf = mdf[mdf['config'] == config].sort_values('rate')
            if cdf.empty:
                continue
            ax.plot(cdf['rate'], cdf['throughput'],
                    marker='o', markersize=4, color=palette[j],
                    label=CONFIG_LABELS[config])
        ax.set_title(model, fontsize=11)
        ax.set_xlabel('Concurrency')
        ax.set_ylabel('Throughput (tok/s)')
        ax.legend(fontsize=8)

    fig.suptitle('Throughput vs Concurrency: llm-d v0.7.0 (gmu=0.9)', fontsize=13)
    plt.tight_layout()
    out = OUTPUT_DIR / 'v0.7.0_throughput_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def fig2_version_comparison(df: pd.DataFrame) -> None:
    """Figure 2: Peak throughput v0.5.1 vs v0.6.0 vs v0.7.0.

    Only configs with valid v0.6.0 baselines are shown. fs-offload and
    cpu+fs-offload-20k are excluded — the v0.6.0 runs were misconfigured.
    """
    # Only configs with valid v0.6.0 data
    configs_shown = ['no-offload', 'native-offload-20k', 'lmcache-local', 'lmcache-valkey']
    x = np.arange(len(MODELS))
    width = 0.22
    versions = ['v0.5.1', 'v0.6.0', 'v0.7.0']
    version_baselines = [V051_GMU09, V060_GMU09, None]

    fig, axes = plt.subplots(1, len(configs_shown), figsize=(16, 6))
    palette = sns.color_palette("muted", 3)

    for ci, config in enumerate(configs_shown):
        ax = axes[ci]
        for vi, (version, baseline) in enumerate(zip(versions, version_baselines)):
            vals = []
            for model in MODELS:
                if baseline is not None:
                    vals.append(baseline.get((model, config), 0))
                else:
                    vals.append(peak_throughput(df, model, config, mempress=False))
            ax.bar(x + (vi - 1) * width, vals, width, label=version,
                   color=palette[vi])

        ax.set_title(CONFIG_LABELS[config], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('Qwen3-', '') for m in MODELS], fontsize=9)
        ax.set_ylabel('Peak Throughput (tok/s)')
        if ci == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Peak Throughput: v0.5.1 vs v0.6.0 vs v0.7.0 (gmu=0.9)\n'
                 '(fs-offload excluded — v0.6.0 misconfigured; see Figure 6)', fontsize=12)
    plt.tight_layout()
    out = OUTPUT_DIR / 'v0.7.0_version_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def fig6_fs_offload_baseline(df: pd.DataFrame) -> None:
    """Figure 6: v0.7.0 fs-offload and cpu+fs-offload-20k — first valid baseline.

    Compared against no-offload within v0.7.0 only (no prior valid baseline).
    """
    gmu09 = df[~df['mempress']]
    fs_configs = ['no-offload', 'fs-offload', 'cpu+fs-offload-20k']
    x = np.arange(len(MODELS))
    width = 0.25
    palette = sns.color_palette("muted", len(fs_configs))

    fig, ax = plt.subplots(figsize=(12, 6))
    for ci, config in enumerate(fs_configs):
        vals = [peak_throughput(gmu09, m, config) for m in MODELS]
        ax.bar(x + (ci - 1) * width, vals, width,
               label=CONFIG_LABELS[config], color=palette[ci])

    # Annotate % vs no-offload for fs configs
    for ci, config in enumerate(['fs-offload', 'cpu+fs-offload-20k']):
        for mi, model in enumerate(MODELS):
            base = peak_throughput(gmu09, model, 'no-offload')
            val = peak_throughput(gmu09, model, config)
            if base > 0 and val > 0:
                delta = (val - base) / base * 100
                bar_x = x[mi] + (ci) * width
                ax.text(bar_x, val + 2, f'{delta:+.0f}%',
                        ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel('Peak Throughput (tok/s)')
    ax.set_title('Filesystem KV Cache Offload — v0.7.0 First Baseline (gmu=0.9)\n'
                 'llmd_fs_connector baked into llm-d-cuda:v0.7.0', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / 'v0.7.0_fs_offload_baseline.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def fig3_overhead_heatmap(df: pd.DataFrame) -> None:
    """Figure 3: Throughput % delta vs no-offload baseline at gmu=0.9."""
    gmu09 = df[~df['mempress']]
    offload_configs = [c for c in CONFIGS if c != 'no-offload']

    data = np.zeros((len(offload_configs), len(MODELS)))
    for mi, model in enumerate(MODELS):
        baseline = peak_throughput(gmu09.copy(), model, 'no-offload')
        for ci, config in enumerate(offload_configs):
            val = peak_throughput(gmu09.copy(), model, config)
            if baseline > 0 and val > 0:
                data[ci, mi] = (val - baseline) / baseline * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap='magma', aspect='auto',
                   vmin=data.min(), vmax=max(data.max(), 1))
    plt.colorbar(im, ax=ax, label='% vs no-offload baseline')
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=20, ha='right')
    ax.set_yticks(range(len(offload_configs)))
    ax.set_yticklabels([CONFIG_LABELS[c] for c in offload_configs])
    for ci in range(len(offload_configs)):
        for mi in range(len(MODELS)):
            ax.text(mi, ci, f'{data[ci, mi]:+.1f}%',
                    ha='center', va='center', fontsize=9,
                    color='white' if abs(data[ci, mi]) > 30 else 'black')
    ax.set_title('Throughput % Delta vs No-Offload: llm-d v0.7.0 (gmu=0.9)', fontsize=12)
    plt.tight_layout()
    out = OUTPUT_DIR / 'v0.7.0_overhead_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def fig4_mempress_comparison(df: pd.DataFrame) -> None:
    """Figure 4: Mempress peak throughput v0.6.0 vs v0.7.0 — configs with valid baselines."""
    # fs-offload excluded — v0.6.0 runs were misconfigured; see fig7 for v0.7.0 fs mempress
    configs_shown = ['no-offload', 'native-offload-20k', 'lmcache-local', 'lmcache-valkey']
    mempress_df = df[df['mempress']]
    x = np.arange(len(MODELS))
    width = 0.35
    palette = sns.color_palette("muted", 2)

    fig, axes = plt.subplots(1, len(configs_shown), figsize=(16, 6))

    for ci, config in enumerate(configs_shown):
        ax = axes[ci]
        v060_vals = [V060_MEMPRESS.get((m, config), 0) for m in MODELS]
        v070_vals = [peak_throughput(mempress_df, m, config, mempress=True)
                     for m in MODELS]
        ax.bar(x - width/2, v060_vals, width, label='v0.6.0', color=palette[0])
        ax.bar(x + width/2, v070_vals, width, label='v0.7.0', color=palette[1])
        ax.set_title(CONFIG_LABELS[config], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('Qwen3-', '') for m in MODELS], fontsize=9)
        ax.set_ylabel('Peak Throughput (tok/s)')
        if ci == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Memory-Pressure Peak Throughput: v0.6.0 vs v0.7.0\n'
                 '(fs-offload excluded — v0.6.0 misconfigured; see Figure 7)', fontsize=12)
    plt.tight_layout()
    out = OUTPUT_DIR / 'v0.7.0_mempress_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def fig7_fs_offload_mempress_baseline(df: pd.DataFrame) -> None:
    """Figure 7: v0.7.0 fs-offload mempress — first valid baseline."""
    mempress_df = df[df['mempress']]
    fs_configs = ['no-offload', 'fs-offload', 'cpu+fs-offload-20k']
    x = np.arange(len(MODELS))
    width = 0.25
    palette = sns.color_palette("muted", len(fs_configs))

    fig, ax = plt.subplots(figsize=(12, 6))
    for ci, config in enumerate(fs_configs):
        vals = [peak_throughput(mempress_df, m, config, mempress=True) for m in MODELS]
        ax.bar(x + (ci - 1) * width, vals, width,
               label=CONFIG_LABELS[config], color=palette[ci])

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel('Peak Throughput (tok/s)')
    ax.set_title('Filesystem KV Cache Offload — Memory Pressure — v0.7.0 First Baseline', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / 'v0.7.0_fs_offload_mempress_baseline.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def fig5_ttft_comparison(df: pd.DataFrame) -> None:
    """Figure 5: TTFT at peak concurrency for all configs (gmu=0.9)."""
    gmu09 = df[~df['mempress']]
    peak_rate = RATES[-1]  # rate=650

    x = np.arange(len(MODELS))
    width = 0.13
    palette = sns.color_palette("muted", len(CONFIGS))

    fig, ax = plt.subplots(figsize=(14, 6))
    for ci, config in enumerate(CONFIGS):
        vals = []
        for model in MODELS:
            sub = gmu09[(gmu09['model'] == model) & (gmu09['config'] == config) &
                        (gmu09['rate'] == peak_rate)]
            vals.append(sub['ttft_ms'].values[0] if not sub.empty else 0)
        offset = (ci - len(CONFIGS)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=CONFIG_LABELS[config], color=palette[ci])

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel('TTFT (ms) at rate=650')
    ax.set_title('Time to First Token at Peak Concurrency: llm-d v0.7.0 (gmu=0.9)', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / 'v0.7.0_ttft_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print peak throughput table and version delta vs v0.6.0."""
    gmu09 = df[~df['mempress']]

    print("\n" + "="*90)
    print("v0.7.0 PEAK THROUGHPUT (tok/s) — gmu=0.9")
    print("="*90)
    header = f"{'Config':<22}" + "".join(f"{m:>16}" for m in MODELS)
    print(header)
    print("-" * 90)

    for config in CONFIGS:
        row = f"{CONFIG_LABELS[config]:<22}"
        for model in MODELS:
            val = peak_throughput(gmu09, model, config)
            v060 = V060_GMU09.get((model, config), 0)
            if val > 0 and v060 > 0:
                delta = (val - v060) / v060 * 100
                row += f"{val:>8.1f} ({delta:>+5.1f}%)"
            elif val > 0:
                row += f"{val:>8.1f}    (new) "
            else:
                row += f"{'—':>16}"
        print(row)

    print("\n  Values in parentheses: % change vs v0.6.0; (new) = first valid baseline")

    print("\n" + "="*90)
    print("v0.7.0 PEAK THROUGHPUT (tok/s) — MEMPRESS")
    print("="*90)
    print(header)
    print("-" * 90)

    mempress_df = df[df['mempress']]
    for config in CONFIGS:
        row = f"{CONFIG_LABELS[config]:<22}"
        for model in MODELS:
            val = peak_throughput(mempress_df, model, config, mempress=True)
            v060 = V060_MEMPRESS.get((model, config), 0)
            if val > 0 and v060 > 0:
                delta = (val - v060) / v060 * 100
                row += f"{val:>8.1f} ({delta:>+5.1f}%)"
            elif val > 0:
                row += f"{val:>8.1f}         "
            else:
                row += f"{'—':>16}"
        print(row)

    print("\n" + "="*90)
    print("NATIVE OFFLOAD OVERHEAD vs NO-OFFLOAD (gmu=0.9)")
    print("="*90)
    for version_label, baseline_dict, use_df in [
        ('v0.5.1', V051_GMU09, None),
        ('v0.6.0', V060_GMU09, None),
        ('v0.7.0', None,       gmu09),
    ]:
        row = f"{version_label:<10}"
        for model in MODELS:
            if baseline_dict is not None:
                no_off = baseline_dict.get((model, 'no-offload'), 0)
                native = baseline_dict.get((model, 'native-offload-20k'), 0)
            else:
                no_off = peak_throughput(use_df, model, 'no-offload')
                native = peak_throughput(use_df, model, 'native-offload-20k')
            if no_off > 0 and native > 0:
                delta = (native - no_off) / no_off * 100
                row += f"{delta:>+14.1f}%"
            else:
                row += f"{'—':>15}"
        print(row)

    print("\n" + "="*90)
    print("TTFT AT PEAK CONCURRENCY (rate=650, gmu=0.9, ms)")
    print("="*90)
    print(header)
    print("-" * 90)
    for config in CONFIGS:
        row = f"{CONFIG_LABELS[config]:<22}"
        for model in MODELS:
            sub = gmu09[(gmu09['model'] == model) & (gmu09['config'] == config) &
                        (gmu09['rate'] == 650)]
            if not sub.empty:
                val = sub['ttft_ms'].values[0]
                row += f"{val:>16.0f}"
            else:
                row += f"{'—':>16}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading v0.7.0 results...")
    df = load_all_results()

    if df.empty:
        print("No results found. Run benchmarks first.")
        return

    print(f"\nResult counts:")
    print(f"  gmu=0.9:  {len(df[~df['mempress']])} runs")
    print(f"  mempress: {len(df[df['mempress']])} runs")

    print("\nGenerating figures...")
    fig1_throughput_vs_concurrency(df)
    fig2_version_comparison(df)
    fig3_overhead_heatmap(df)
    fig4_mempress_comparison(df)
    fig5_ttft_comparison(df)
    fig6_fs_offload_baseline(df)
    fig7_fs_offload_mempress_baseline(df)

    print_summary(df)

    print("\nDone. Outputs in analysis/v0.7.0_*")


if __name__ == '__main__':
    main()
