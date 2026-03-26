#!/usr/bin/env python3
"""
RHOAI 3.3 KV Cache Offload Benchmark Analysis
Loads all 64 guidellm JSON results and PCP KV cache metrics,
generates visualizations and a summary CSV.
"""
import json
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Palette / style ──────────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11
PALETTE = sns.color_palette("muted")
COL_NO_OFFLOAD  = PALETTE[0]   # blue
COL_OFFLOAD     = PALETTE[1]   # orange
COL_REPLICA1    = PALETTE[2]   # green
COL_REPLICA2    = PALETTE[3]   # red

RESULTS_DIR = Path(__file__).parent.parent / "results"
SCRIPT_DIR  = Path(__file__).parent

# ── Short display names ───────────────────────────────────────────────────────
MODEL_LABELS = {
    "Meta-Llama-3.1-70B-Instruct-FP8": "Llama-3.1-70B-FP8",
    "gpt-oss-120b": "GPT-OSS-120B (MoE)",
}
CONFIG_LABELS = {
    "no-offload": "no-offload",
    "native-offload-20k": "native-offload-20k",
}

# Ordered rate values — used to give each rate equal x-axis spacing
RATES_ORDERED = [1, 50, 100, 150, 300, 400, 500, 650]
RATE_POS = {r: i for i, r in enumerate(RATES_ORDERED)}


def rpos(series):
    """Map a rate Series to ordinal x-axis positions."""
    return series.map(RATE_POS)


def set_rate_xaxis(ax, rates=None):
    """Apply equal-spaced x-axis with rate labels."""
    ordered = rates if rates is not None else RATES_ORDERED
    positions = range(len(ordered))
    ax.set_xticks(list(positions))
    ax.set_xticklabels(ordered)
    ax.set_xlim(-0.5, len(ordered) - 0.5)


# Known GPU/CPU block counts from startup logs and PCP cache_config_info
BLOCK_COUNTS = {
    "Meta-Llama-3.1-70B-Instruct-FP8": {"gpu": 26842, "cpu": 20000},
    "gpt-oss-120b":                     {"gpu": 181691, "cpu": 20000},
}


# ── Data loading ─────────────────────────────────────────────────────────────
def load_guidellm_json(path: Path) -> dict:
    result = subprocess.run(
        ["zstd", "-d", "-q", "-c", str(path)],
        capture_output=True, check=True
    )
    data = json.loads(result.stdout)
    bm = data["benchmarks"][0]
    m  = bm["metrics"]
    return {
        "duration_s":      bm["duration"],
        "start_time":      bm["start_time"],
        "end_time":        bm["end_time"],
        "completed":       m["request_totals"]["successful"],
        "errors":          m["request_totals"]["errored"],
        "gen_tok_s_mean":  m["output_tokens_per_second"]["successful"]["mean"],
        "gen_tok_s_p50":   m["output_tokens_per_second"]["successful"]["percentiles"]["p50"],
        "ttft_ms_p50":     m["time_to_first_token_ms"]["successful"]["percentiles"]["p50"],
        "ttft_ms_p90":     m["time_to_first_token_ms"]["successful"]["percentiles"]["p90"],
        "ttft_ms_p95":     m["time_to_first_token_ms"]["successful"]["percentiles"]["p95"],
        "ttft_ms_mean":    m["time_to_first_token_ms"]["successful"]["mean"],
        "itl_ms_p50":      m["inter_token_latency_ms"]["successful"]["percentiles"]["p50"],
        "itl_ms_p95":      m["inter_token_latency_ms"]["successful"]["percentiles"]["p95"],
        "itl_ms_mean":     m["inter_token_latency_ms"]["successful"]["mean"],
        "tpot_ms_p50":     m["time_per_output_token_ms"]["successful"]["percentiles"]["p50"],
        "tpot_ms_mean":    m["time_per_output_token_ms"]["successful"]["mean"],
        "req_latency_s_mean": m["request_latency"]["successful"]["mean"],
        "req_s_mean":      m["requests_per_second"]["successful"]["mean"],
    }


def parse_run_dir(name: str) -> dict:
    # 2x8xH200_rhoai-3.3_<model>_<config>_replica<N>_rate<R>
    m = re.match(
        r"2x8xH200_rhoai-3\.3_(.+?)_(no-offload|native-offload-20k)_replica(\d+)_rate(\d+)$",
        name,
    )
    if not m:
        return None
    return {
        "model":    m.group(1),
        "config":   m.group(2),
        "replicas": int(m.group(3)),
        "rate":     int(m.group(4)),
    }


def load_all_results() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        meta = parse_run_dir(run_dir.name)
        if meta is None:
            continue
        json_path = run_dir / "guidellm-results.json.zst"
        if not json_path.exists():
            continue
        try:
            metrics = load_guidellm_json(json_path)
        except Exception as e:
            print(f"  WARNING: could not parse {run_dir.name}: {e}")
            continue
        rows.append({**meta, **metrics, "run_dir": str(run_dir)})
    df = pd.DataFrame(rows)
    df["model_label"]  = df["model"].map(MODEL_LABELS)
    df["config_label"] = df["config"].map(CONFIG_LABELS)
    return df


# ── PCP KV cache metrics extraction ──────────────────────────────────────────
def extract_pcp_kv_metrics(run_dir: str) -> pd.DataFrame:
    """Extract vllm.kv_cache_usage_perc timeseries from the PCP archive."""
    archive_base = Path(run_dir) / "pcp-archives"
    node_dirs = list(archive_base.iterdir()) if archive_base.exists() else []
    if not node_dirs:
        return pd.DataFrame()

    node_dir = node_dirs[0]
    zst_files = list(node_dir.glob("*.zst"))
    if not zst_files:
        return pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmpdir:
        for zf in zst_files:
            out = Path(tmpdir) / zf.stem
            subprocess.run(
                ["zstd", "-d", "-q", "-c", str(zf)],
                stdout=open(out, "wb"), stderr=subprocess.DEVNULL, check=False
            )
        meta_files = list(Path(tmpdir).glob("*.meta"))
        if not meta_files:
            return pd.DataFrame()
        arch = str(meta_files[0]).replace(".meta", "")

        try:
            # Use pmrep with separate metric extractions (one metric at a time avoids
            # multi-instance header parsing complexity)
            def get_series(metric):
                r = subprocess.run(
                    ["pmrep", "-a", arch, "-t", "10s", "-z", "-H", "-l", "|", metric],
                    capture_output=True, text=True, timeout=30
                )
                vals = []
                for line in r.stdout.splitlines():
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        v = parts[1].strip()
                        try:
                            vals.append(float(v))
                        except ValueError:
                            vals.append(np.nan)
                return vals

            kv   = get_series("openmetrics.vllm.vllm.kv_cache_usage_perc")
            phit = get_series("openmetrics.vllm.vllm.prefix_cache_hits_total")
            pq   = get_series("openmetrics.vllm.vllm.prefix_cache_queries_total")
            ehit = get_series("openmetrics.vllm.vllm.external_prefix_cache_hits_total")
            eq   = get_series("openmetrics.vllm.vllm.external_prefix_cache_queries_total")

            n = max(len(kv), 1)
            records = [
                {
                    "kv_usage": kv[i]   if i < len(kv) else np.nan,
                    "pfx_hits": phit[i] if i < len(phit) else np.nan,
                    "pfx_q":    pq[i]   if i < len(pq) else np.nan,
                    "ext_hits": ehit[i] if i < len(ehit) else np.nan,
                    "ext_q":    eq[i]   if i < len(eq) else np.nan,
                }
                for i in range(len(kv))
            ]
            return pd.DataFrame(records)
        except Exception:
            return pd.DataFrame()


# ── Figures ───────────────────────────────────────────────────────────────────
def fig_throughput_curves(df: pd.DataFrame):
    """4-panel throughput vs concurrency: 2 models × 2 replica counts."""
    models   = ["Meta-Llama-3.1-70B-Instruct-FP8", "gpt-oss-120b"]
    replicas = [1, 2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle(
        "Output Throughput vs Concurrency — RHOAI 3.3 on 2×8×H200",
        fontsize=13, y=1.01
    )

    for ri, rep in enumerate(replicas):
        for mi, model in enumerate(models):
            ax = axes[ri][mi]
            sub = df[(df["model"] == model) & (df["replicas"] == rep)]

            for config, lbl, col, ls in [
                ("no-offload",        "no-offload",          COL_NO_OFFLOAD, "-"),
                ("native-offload-20k","native-offload-20k",  COL_OFFLOAD,    "--"),
            ]:
                d = sub[sub["config"] == config].sort_values("rate")
                if d.empty:
                    continue
                ax.plot(d["rate"], d["gen_tok_s_mean"], marker="o", ms=5,
                        color=col, ls=ls, lw=2, label=lbl)

            ax.set_title(f"{MODEL_LABELS[model]}  ·  {rep} replica{'s' if rep > 1 else ''}",
                         fontsize=11)
            ax.set_xlabel("Concurrency (requests)")
            ax.set_ylabel("Output throughput (tok/s)")
            set_rate_xaxis(ax)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("throughput_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ throughput_curves.png")


def fig_latency_curves(df: pd.DataFrame):
    """TTFT mean and TPOT p50 vs concurrency, 4-panel (2 models × 2 metrics).
    Uses mean TTFT because gpt-oss-120b p50=0 (most requests get first token
    instantly with the MoE reasoning parser, making p50 uninformative).
    """
    models = ["Meta-Llama-3.1-70B-Instruct-FP8", "gpt-oss-120b"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Latency vs Concurrency (replicas=1) — RHOAI 3.3 on 2×8×H200",
                 fontsize=13, y=1.01)

    metrics_labels = [
        ("ttft_ms_mean", "TTFT mean (ms)"),
        ("tpot_ms_p50",  "TPOT p50 (ms)"),
    ]

    for mi, model in enumerate(models):
        for ki, (metric, ylabel) in enumerate(metrics_labels):
            ax = axes[ki][mi]
            sub = df[(df["model"] == model) & (df["replicas"] == 1)]

            for config, lbl, col, ls in [
                ("no-offload",        "no-offload",         COL_NO_OFFLOAD, "-"),
                ("native-offload-20k","native-offload-20k", COL_OFFLOAD,    "--"),
            ]:
                d = sub[sub["config"] == config].sort_values("rate")
                if d.empty:
                    continue
                ax.plot(rpos(d["rate"]), d[metric], marker="o", ms=5,
                        color=col, ls=ls, lw=2, label=lbl)

            ax.set_title(f"{MODEL_LABELS[model]}  ·  {ylabel}", fontsize=11)
            ax.set_xlabel("Concurrency")
            ax.set_ylabel(ylabel)
            set_rate_xaxis(ax)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("latency_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ latency_curves.png")


def fig_offload_impact_heatmap(df: pd.DataFrame):
    """Throughput delta (%) from no-offload → native-offload-20k, by model/replica/rate."""
    rates    = sorted(df["rate"].unique())
    rows_out = []
    for model in ["Meta-Llama-3.1-70B-Instruct-FP8", "gpt-oss-120b"]:
        for rep in [1, 2]:
            base   = df[(df["model"]==model)&(df["config"]=="no-offload")&(df["replicas"]==rep)]
            offload= df[(df["model"]==model)&(df["config"]=="native-offload-20k")&(df["replicas"]==rep)]
            for rate in rates:
                b = base[base["rate"]==rate]["gen_tok_s_mean"].values
                o = offload[offload["rate"]==rate]["gen_tok_s_mean"].values
                if len(b) and len(o):
                    delta_pct = (o[0] - b[0]) / b[0] * 100
                    rows_out.append({
                        "label": f"{MODEL_LABELS[model]}\nr={rep}",
                        "rate":  rate,
                        "delta": delta_pct,
                    })

    pivot = pd.DataFrame(rows_out).pivot(index="label", columns="rate", values="delta")
    label_order = [
        f"{MODEL_LABELS['Meta-Llama-3.1-70B-Instruct-FP8']}\nr=1",
        f"{MODEL_LABELS['Meta-Llama-3.1-70B-Instruct-FP8']}\nr=2",
        f"{MODEL_LABELS['gpt-oss-120b']}\nr=1",
        f"{MODEL_LABELS['gpt-oss-120b']}\nr=2",
    ]
    pivot = pivot.reindex([l for l in label_order if l in pivot.index])

    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".1f", annot_kws={"size": 10},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Throughput Δ (%)", "shrink": 0.7},
    )
    ax.set_title(
        "Native CPU Offload Throughput Impact vs Baseline\n"
        "(native-offload-20k vs no-offload, % change, positive=higher throughput)",
        fontsize=12
    )
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fig.savefig("offload_impact_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ offload_impact_heatmap.png")


def fig_replica_scaling(df: pd.DataFrame):
    """Throughput at replicas=2 vs ideal 2× replicas=1, by model and config."""
    models  = ["Meta-Llama-3.1-70B-Instruct-FP8", "gpt-oss-120b"]
    configs = ["no-offload", "native-offload-20k"]
    rates   = sorted(df["rate"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Replica Scaling Efficiency — replicas=2 vs 2× replicas=1",
                 fontsize=13)

    for mi, model in enumerate(models):
        ax = axes[mi]
        for config, col, ls in [
            ("no-offload",         COL_NO_OFFLOAD, "-"),
            ("native-offload-20k", COL_OFFLOAD,    "--"),
        ]:
            r1 = df[(df["model"]==model)&(df["config"]==config)&(df["replicas"]==1)]
            r2 = df[(df["model"]==model)&(df["config"]==config)&(df["replicas"]==2)]
            effs = []
            valid_rates = []
            for rate in rates:
                t1 = r1[r1["rate"]==rate]["gen_tok_s_mean"].values
                t2 = r2[r2["rate"]==rate]["gen_tok_s_mean"].values
                if len(t1) and len(t2) and t1[0] > 0:
                    effs.append(t2[0] / (2 * t1[0]) * 100)
                    valid_rates.append(rate)
            if effs:
                ax.plot([RATE_POS[r] for r in valid_rates], effs, marker="o", ms=5,
                        color=col, ls=ls, lw=2, label=CONFIG_LABELS[config])

        ax.axhline(100, color="gray", ls=":", lw=1.5, label="ideal (100%)")
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Scaling efficiency (%)")
        set_rate_xaxis(ax)
        ax.set_ylim(0, 155)
        ax.text(0.97, 0.97, ">100%: EPP prefix-cache\nrouting concentrates\nsimilar requests per replica",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                color="gray", style="italic")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("replica_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ replica_scaling.png")


def fig_kv_cache_pressure(df: pd.DataFrame):
    """KV cache usage % from PCP archives vs concurrency (replicas=1 only)."""
    print("  Extracting PCP KV cache metrics (this may take a minute)...")
    records = []
    for _, row in df[df["replicas"] == 1].iterrows():
        pcp_df = extract_pcp_kv_metrics(row["run_dir"])
        if pcp_df.empty or pcp_df["kv_usage"].isna().all():
            continue
        # kv_cache_usage_perc is 0-1 fraction; convert to percentage
        avg_kv = pcp_df["kv_usage"].dropna().mean() * 100
        max_kv = pcp_df["kv_usage"].dropna().max() * 100
        # Compute prefix cache hit rate if available
        ph = pcp_df["pfx_hits"].dropna()
        pq = pcp_df["pfx_q"].dropna()
        pfx_rate = (ph.iloc[-1] - ph.iloc[0]) / max(pq.iloc[-1] - pq.iloc[0], 1) if len(ph) > 1 else np.nan
        eh = pcp_df["ext_hits"].dropna()
        eq = pcp_df["ext_q"].dropna()
        ext_rate = (eh.iloc[-1] - eh.iloc[0]) / max(eq.iloc[-1] - eq.iloc[0], 1) if len(eh) > 1 else np.nan

        records.append({
            "model":   row["model"],
            "config":  row["config"],
            "rate":    row["rate"],
            "avg_kv":  avg_kv,
            "max_kv":  max_kv,
            "pfx_hit_rate": pfx_rate * 100 if not np.isnan(pfx_rate) else np.nan,
            "ext_hit_rate": ext_rate * 100 if not np.isnan(ext_rate) else np.nan,
        })

    if not records:
        print("  WARNING: no PCP KV cache data available, skipping fig")
        return pd.DataFrame()

    pcp_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GPU KV Cache Usage vs Concurrency (replicas=1) — from PCP Archives",
                 fontsize=12)

    for mi, model in enumerate(["Meta-Llama-3.1-70B-Instruct-FP8", "gpt-oss-120b"]):
        ax = axes[mi]
        sub = pcp_df[pcp_df["model"] == model]
        for config, lbl, col, ls in [
            ("no-offload",        "no-offload",         COL_NO_OFFLOAD, "-"),
            ("native-offload-20k","native-offload-20k", COL_OFFLOAD,    "--"),
        ]:
            d = sub[sub["config"] == config].sort_values("rate")
            if d.empty:
                continue
            ax.plot(rpos(d["rate"]), d["avg_kv"], marker="o", ms=5,
                    color=col, ls=ls, lw=2, label=f"{lbl} (avg)")
            ax.fill_between(rpos(d["rate"]), d["avg_kv"], d["max_kv"],
                            alpha=0.12, color=col)

        blocks = BLOCK_COUNTS.get(model, {})
        gpu_b  = blocks.get("gpu", 0)
        cpu_b  = blocks.get("cpu", 0)
        extra_info = (
            f"GPU blocks: {gpu_b:,}\n"
            f"CPU blocks: {cpu_b:,}\n"
            f"CPU/GPU ratio: {cpu_b/max(gpu_b,1)*100:.0f}%"
        )
        ax.text(0.03, 0.97, extra_info, transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("GPU KV cache usage (%)")
        set_rate_xaxis(ax)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("kv_cache_pressure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ kv_cache_pressure.png")
    return pcp_df


def fig_summary_table(df: pd.DataFrame):
    """Peak throughput summary table as a figure."""
    rows = []
    for model in ["Meta-Llama-3.1-70B-Instruct-FP8", "gpt-oss-120b"]:
        for rep in [1, 2]:
            base = df[(df["model"]==model)&(df["config"]=="no-offload")&(df["replicas"]==rep)]
            off  = df[(df["model"]==model)&(df["config"]=="native-offload-20k")&(df["replicas"]==rep)]
            if base.empty or off.empty:
                continue
            peak_base = base["gen_tok_s_mean"].max()
            peak_off  = off["gen_tok_s_mean"].max()
            rate_base = base.loc[base["gen_tok_s_mean"].idxmax(), "rate"]
            rate_off  = off.loc[off["gen_tok_s_mean"].idxmax(), "rate"]
            delta_pct = (peak_off - peak_base) / peak_base * 100
            rows.append({
                "Model":        MODEL_LABELS[model],
                "Replicas":     rep,
                "no-offload\n(tok/s @ rate)": f"{peak_base:.1f} @{rate_base}",
                "native-offload-20k\n(tok/s @ rate)": f"{peak_off:.1f} @{rate_off}",
                "Offload\nΔ (%)": f"{delta_pct:+.1f}%",
            })

    tbl = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    t = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        cellLoc="center",
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f8f9fa")
        else:
            cell.set_facecolor("white")
        # colour the delta column
        if c == 4 and r > 0:
            val = tbl.iloc[r-1, 4].replace("%", "")
            try:
                v = float(val)
                cell.set_facecolor("#d4edda" if v > 0 else "#f8d7da" if v < -1 else "white")
            except ValueError:
                pass

    ax.set_title("Peak Throughput Summary — RHOAI 3.3 KV Cache Offload Evaluation",
                 fontsize=12, pad=16)
    fig.tight_layout()
    fig.savefig("summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ summary_table.png")


def fig_ttft_itl_comparison(df: pd.DataFrame):
    """Side-by-side bar chart: avg TTFT and ITL across concurrency ≥ 50, per config/model."""
    sub = df[df["rate"] >= 50]
    # Use mean latency (p50 is 0 for gpt-oss-120b due to MoE token batching)
    summary = sub.groupby(["model", "config", "replicas"]).agg(
        ttft=("ttft_ms_mean", "mean"),
        itl=("tpot_ms_mean", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Mean p50 Latency (concurrency ≥ 50) by Model, Config, Replica Count",
        fontsize=12
    )

    for ki, (metric, ylabel, title_sfx) in enumerate([
        ("ttft", "TTFT mean (ms)",    "Time to First Token (mean)"),
        ("itl",  "TPOT mean (ms)", "Time per Output Token (mean)"),
    ]):
        ax = axes[ki]
        summary["label"] = (
            summary["model"].map(MODEL_LABELS) + "\nr=" + summary["replicas"].astype(str)
        )
        pivot = summary.pivot_table(
            index="label", columns="config", values=metric, aggfunc="mean"
        )
        cols_order = [c for c in ["no-offload", "native-offload-20k"] if c in pivot.columns]
        pivot = pivot[cols_order]
        pivot.plot(kind="bar", ax=ax, color=[COL_NO_OFFLOAD, COL_OFFLOAD],
                   width=0.6, edgecolor="white", lw=0.5)
        ax.set_title(title_sfx, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Config", fontsize=9)
        ax.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig("latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ latency_comparison.png")


def save_summary_csv(df: pd.DataFrame):
    cols = ["model", "config", "replicas", "rate",
            "gen_tok_s_mean", "ttft_ms_p50", "ttft_ms_p95",
            "itl_ms_p50", "tpot_ms_p50", "req_s_mean",
            "completed", "errors", "duration_s"]
    df[cols].sort_values(["model", "replicas", "config", "rate"]).to_csv(
        "results_summary.csv", index=False
    )
    print("  ✓ results_summary.csv")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.chdir(Path(__file__).parent.parent)
    print("Loading guidellm results...")
    df = load_all_results()
    print(f"  Loaded {len(df)} runs across {df['model'].nunique()} models")
    print(f"  Models:   {sorted(df['model'].unique())}")
    print(f"  Configs:  {sorted(df['config'].unique())}")
    print(f"  Replicas: {sorted(df['replicas'].unique())}")
    print(f"  Rates:    {sorted(df['rate'].unique())}")
    print(f"  Errors:   {df['errors'].sum()} total across all runs")
    print()

    print("Generating figures...")
    fig_throughput_curves(df)
    fig_latency_curves(df)
    fig_offload_impact_heatmap(df)
    fig_replica_scaling(df)
    pcp_df = fig_kv_cache_pressure(df)
    fig_ttft_itl_comparison(df)
    fig_summary_table(df)
    save_summary_csv(df)

    print()
    print("=== Quick summary ===")
    for model in ["Meta-Llama-3.1-70B-Instruct-FP8", "gpt-oss-120b"]:
        print(f"\n{MODEL_LABELS[model]}:")
        for rep in [1, 2]:
            base_peak = df[(df["model"]==model)&(df["config"]=="no-offload")&(df["replicas"]==rep)]["gen_tok_s_mean"].max()
            off_peak  = df[(df["model"]==model)&(df["config"]=="native-offload-20k")&(df["replicas"]==rep)]["gen_tok_s_mean"].max()
            if base_peak > 0:
                delta = (off_peak - base_peak) / base_peak * 100
                print(f"  replicas={rep}: no-offload peak={base_peak:.1f} tok/s, offload peak={off_peak:.1f} tok/s ({delta:+.1f}%)")


if __name__ == "__main__":
    main()
