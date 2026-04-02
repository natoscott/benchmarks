#!/usr/bin/env python3
"""
RHOAI 3.3 KV Cache Offload Benchmark Analysis
Loads guidellm JSON results across three workload profiles:
  - rhoai-3.3        (standard: prompt=512, output=128)
  - rhoai-3.3-kv-stress (output=512, 4x stress)
  - rhoai-3.3-longctx   (prompt=4096, output=256)

Three models x three replica counts x two configs x 8 rates = 432 total runs.
"""
import json
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# -- Palette / style ----------------------------------------------------------
sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11
PALETTE = sns.color_palette("muted")
COL_NO_OFFLOAD = PALETTE[0]   # blue
COL_OFFLOAD    = PALETTE[1]   # orange
COL_REPLICA1   = PALETTE[2]   # green
COL_REPLICA2   = PALETTE[3]   # red
COL_REPLICA4   = PALETTE[4]   # purple

RESULTS_DIR = Path(__file__).parent.parent / "results"
SCRIPT_DIR  = Path(__file__).parent

# -- Model display names ------------------------------------------------------
MODEL_LABELS = {
    "Meta-Llama-3.1-70B-Instruct-FP8": "Llama-3.1-70B-FP8",
    "gpt-oss-120b":                     "GPT-OSS-120B (MoE)",
    "Llama-3.1-70B-Instruct":           "Llama-3.1-70B-BF16",
}
CONFIG_LABELS = {
    "no-offload":         "no-offload",
    "native-offload-20k": "native-offload-20k",
}

# Ordered rate values per profile -- equal x-axis spacing
RATES_STANDARD = [1, 50, 100, 150, 300, 400, 500, 650]
RATES_LONGCTX  = [1, 5, 10, 20, 50, 100, 200, 300]

RPOS_STANDARD = {r: i for i, r in enumerate(RATES_STANDARD)}
RPOS_LONGCTX  = {r: i for i, r in enumerate(RATES_LONGCTX)}

# GPU KV cache block counts from vLLM v1 startup logs.
# vLLM v1 reports "GPU KV cache size: N tokens"; divide by block_size to get blocks.
# Dense Llama models: block_size=16; gpt-oss-120b MoE: block_size=8.
BLOCK_COUNTS = {
    "Meta-Llama-3.1-70B-Instruct-FP8": {
        "gpu":   26842,   # 429,472 tokens / 16 at gpu_util=0.75
        "cpu":   20000,
        "label": "FP8, gpu_util=0.75",
    },
    "gpt-oss-120b": {
        "gpu":   181691,  # 1,453,520 tokens / 8 at gpu_util=0.65
        "cpu":   20000,
        "label": "MoE MXFP4, gpu_util=0.65",
    },
    "Llama-3.1-70B-Instruct": {
        "gpu":   22376,   # 358,016 tokens / 16 at gpu_util=0.90
        "cpu":   20000,
        "label": "BF16, gpu_util=0.90",
    },
}

# Display order: dense FP8, dense BF16, MoE
MODEL_ORDER = [
    "Meta-Llama-3.1-70B-Instruct-FP8",
    "Llama-3.1-70B-Instruct",
    "gpt-oss-120b",
]


def rpos(series, profile="standard"):
    rmap = RPOS_LONGCTX if profile == "longctx" else RPOS_STANDARD
    return series.map(rmap)


def set_rate_xaxis(ax, profile="standard"):
    rates = RATES_LONGCTX if profile == "longctx" else RATES_STANDARD
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(rates)
    ax.set_xlim(-0.5, len(rates) - 0.5)


# -- Data loading -------------------------------------------------------------
def load_guidellm_json(path):
    result = subprocess.run(
        ["zstd", "-d", "-q", "-c", str(path)],
        capture_output=True, check=True
    )
    data = json.loads(result.stdout)
    bm = data["benchmarks"][0]
    m  = bm["metrics"]
    return {
        "duration_s":         bm["duration"],
        "start_time":         bm["start_time"],
        "end_time":           bm["end_time"],
        "completed":          m["request_totals"]["successful"],
        "errors":             m["request_totals"]["errored"],
        "gen_tok_s_mean":     m["output_tokens_per_second"]["successful"]["mean"],
        "gen_tok_s_p50":      m["output_tokens_per_second"]["successful"]["percentiles"]["p50"],
        "ttft_ms_p50":        m["time_to_first_token_ms"]["successful"]["percentiles"]["p50"],
        "ttft_ms_p90":        m["time_to_first_token_ms"]["successful"]["percentiles"]["p90"],
        "ttft_ms_mean":       m["time_to_first_token_ms"]["successful"]["mean"],
        "itl_ms_p50":         m["inter_token_latency_ms"]["successful"]["percentiles"]["p50"],
        "itl_ms_mean":        m["inter_token_latency_ms"]["successful"]["mean"],
        "tpot_ms_p50":        m["time_per_output_token_ms"]["successful"]["percentiles"]["p50"],
        "tpot_ms_mean":       m["time_per_output_token_ms"]["successful"]["mean"],
        "req_latency_s_mean": m["request_latency"]["successful"]["mean"],
        "req_s_mean":         m["requests_per_second"]["successful"]["mean"],
    }


def parse_run_dir(name):
    m = re.match(
        r"1x8xH200_(rhoai-3\.3(?:-[\w]+)*)_(.+?)_(no-offload|native-offload-20k)"
        r"_replica(\d+)_rate(\d+)$",
        name,
    )
    if not m:
        return None
    software = m.group(1)
    profile = (
        "longctx"  if software == "rhoai-3.3-longctx"  else
        "kvstress" if software == "rhoai-3.3-kv-stress" else
        "standard"
    )
    return {
        "software": software,
        "profile":  profile,
        "model":    m.group(2),
        "config":   m.group(3),
        "replicas": int(m.group(4)),
        "rate":     int(m.group(5)),
    }


def load_benchmark_config(run_dir):
    cfg_path = run_dir / "benchmark-config.txt"
    gpu_util = None
    if cfg_path.exists():
        for line in cfg_path.read_text().splitlines():
            if line.startswith("GPU Memory Utilization:"):
                try:
                    gpu_util = float(line.split(":")[1].strip())
                except ValueError:
                    pass
    return {"gpu_util": gpu_util}


def load_all_results():
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
        cfg = load_benchmark_config(run_dir)
        rows.append({**meta, **metrics, **cfg, "run_dir": str(run_dir)})
    df = pd.DataFrame(rows)
    df["model_label"]  = df["model"].map(MODEL_LABELS)
    df["config_label"] = df["config"].map(CONFIG_LABELS)
    return df


# -- PCP KV cache metrics extraction ------------------------------------------
def _extract_archive(run_dir):
    """Decompress PCP archive to a temp dir, return arch path or None."""
    archive_base = Path(run_dir) / "pcp-archives"
    node_dirs = list(archive_base.iterdir()) if archive_base.exists() else []
    if not node_dirs:
        return None, None
    node_dir = node_dirs[0]
    zst_files = list(node_dir.glob("*.zst"))
    if not zst_files:
        return None, None
    tmpdir = tempfile.mkdtemp()
    for zf in zst_files:
        out = Path(tmpdir) / zf.stem
        subprocess.run(
            ["zstd", "-d", "-q", "-c", str(zf)],
            stdout=open(out, "wb"), stderr=subprocess.DEVNULL, check=False
        )
    meta_files = list(Path(tmpdir).glob("*.meta"))
    if not meta_files:
        import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
        return None, None
    return str(meta_files[0]).replace(".meta", ""), tmpdir


def _pmrep_series(arch, metric):
    """Return list of float values sampled at 10s intervals from arch."""
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


def _counter_delta(vals):
    """Compute delta of a monotonic counter, stripping trailing zeros (pod shutdown)."""
    vs = [v for v in vals if not np.isnan(v)]
    while vs and vs[-1] == 0:
        vs.pop()
    if len(vs) < 2:
        return np.nan
    return vs[-1] - vs[0]


def extract_pcp_kv_metrics(run_dir):
    arch, tmpdir = _extract_archive(run_dir)
    if arch is None:
        return pd.DataFrame()
    try:
        kv   = _pmrep_series(arch, "openmetrics.vllm.vllm.kv_cache_usage_perc")
        phit = _pmrep_series(arch, "openmetrics.vllm.vllm.prefix_cache_hits_total")
        pq   = _pmrep_series(arch, "openmetrics.vllm.vllm.prefix_cache_queries_total")
        records = [
            {"kv_usage": kv[i] if i < len(kv) else np.nan,
             "pfx_hits": phit[i] if i < len(phit) else np.nan,
             "pfx_q":    pq[i]   if i < len(pq)   else np.nan}
            for i in range(len(kv))
        ]
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()
    finally:
        import shutil; shutil.rmtree(tmpdir, ignore_errors=True)




# -- Figures ------------------------------------------------------------------

def fig_throughput_curves(df):
    """3x3 grid: 3 models x 3 replica counts, standard profile."""
    std = df[df["profile"] == "standard"]
    models   = [m for m in MODEL_ORDER if m in std["model"].unique()]
    replicas = [1, 2, 4]

    fig, axes = plt.subplots(len(replicas), len(models),
                             figsize=(14, 13), sharey=False)
    fig.suptitle(
        "Output Throughput vs Concurrency -- RHOAI 3.3 Short-Context Workload (1x8xH200)",
        fontsize=13, y=1.01
    )

    for ri, rep in enumerate(replicas):
        for mi, model in enumerate(models):
            ax = axes[ri][mi]
            sub = std[(std["model"] == model) & (std["replicas"] == rep)]
            for config, lbl, col, ls in [
                ("no-offload",         "no-offload",         COL_NO_OFFLOAD, "-"),
                ("native-offload-20k", "native-offload-20k", COL_OFFLOAD,    "--"),
            ]:
                d = sub[sub["config"] == config].sort_values("rate")
                if d.empty:
                    continue
                ax.plot(rpos(d["rate"]), d["gen_tok_s_mean"], marker="o", ms=5,
                        color=col, ls=ls, lw=2, label=lbl)
            ax.set_title(
                f"{MODEL_LABELS[model]}  x  {rep} replica{'s' if rep > 1 else ''}",
                fontsize=10
            )
            ax.set_xlabel("Concurrency")
            ax.set_ylabel("Output throughput (tok/s)")
            set_rate_xaxis(ax)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("throughput_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  throughput_curves.png")


def fig_offload_impact_heatmap(df):
    """Throughput delta heatmap -- standard profile, all models and replicas."""
    std = df[df["profile"] == "standard"]
    rates = sorted(std["rate"].unique())
    rows_out = []

    for model in MODEL_ORDER:
        for rep in [1, 2, 4]:
            base    = std[(std["model"]==model) & (std["config"]=="no-offload") & (std["replicas"]==rep)]
            offload = std[(std["model"]==model) & (std["config"]=="native-offload-20k") & (std["replicas"]==rep)]
            for rate in rates:
                b = base[base["rate"]==rate]["gen_tok_s_mean"].values
                o = offload[offload["rate"]==rate]["gen_tok_s_mean"].values
                if len(b) and len(o):
                    rows_out.append({
                        "label": f"{MODEL_LABELS[model]}\nr={rep}",
                        "rate":  rate,
                        "delta": (o[0] - b[0]) / b[0] * 100,
                    })

    if not rows_out:
        return
    pivot = pd.DataFrame(rows_out).pivot(index="label", columns="rate", values="delta")
    label_order = [
        f"{MODEL_LABELS[m]}\nr={r}"
        for m in MODEL_ORDER for r in [1, 2, 4]
    ]
    pivot = pivot.reindex([l for l in label_order if l in pivot.index])

    fig, ax = plt.subplots(figsize=(14, 7))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".1f", annot_kws={"size": 9},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Throughput delta (%)", "shrink": 0.7},
    )
    ax.set_title(
        "Native CPU Offload Throughput Impact -- Short-Context Workload\n"
        "(native-offload-20k vs no-offload; positive = higher throughput with offload)",
        fontsize=12
    )
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig("offload_impact_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  offload_impact_heatmap.png")


def fig_kvstress_heatmap(df):
    """Throughput delta heatmap -- kv-stress profile."""
    kvs = df[df["profile"] == "kvstress"]
    if kvs.empty:
        return
    rates = sorted(kvs["rate"].unique())
    rows_out = []
    for model in MODEL_ORDER:
        for rep in [1, 2, 4]:
            base    = kvs[(kvs["model"]==model) & (kvs["config"]=="no-offload") & (kvs["replicas"]==rep)]
            offload = kvs[(kvs["model"]==model) & (kvs["config"]=="native-offload-20k") & (kvs["replicas"]==rep)]
            for rate in rates:
                b = base[base["rate"]==rate]["gen_tok_s_mean"].values
                o = offload[offload["rate"]==rate]["gen_tok_s_mean"].values
                if len(b) and len(o):
                    rows_out.append({
                        "label": f"{MODEL_LABELS[model]}\nr={rep}",
                        "rate":  rate,
                        "delta": (o[0] - b[0]) / b[0] * 100,
                    })
    if not rows_out:
        return
    pivot = pd.DataFrame(rows_out).pivot(index="label", columns="rate", values="delta")
    label_order = [
        f"{MODEL_LABELS[m]}\nr={r}"
        for m in MODEL_ORDER for r in [1, 2, 4]
    ]
    pivot = pivot.reindex([l for l in label_order if l in pivot.index])

    fig, ax = plt.subplots(figsize=(14, 7))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".1f", annot_kws={"size": 9},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Throughput delta (%)", "shrink": 0.7},
    )
    ax.set_title(
        "Native CPU Offload Throughput Impact -- Long-Output Workload (output=512)\n"
        "(native-offload-20k vs no-offload; positive = higher throughput with offload)",
        fontsize=12
    )
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig("kvstress_impact_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  kvstress_impact_heatmap.png")


def fig_latency_curves(df):
    """TTFT mean and TPOT p50 vs concurrency -- standard, replicas=1."""
    std = df[(df["profile"] == "standard") & (df["replicas"] == 1)]
    models = [m for m in MODEL_ORDER if m in std["model"].unique()]

    fig, axes = plt.subplots(2, len(models), figsize=(14, 10))
    fig.suptitle(
        "Latency vs Concurrency (replicas=1) -- RHOAI 3.3 Short-Context Workload",
        fontsize=13, y=1.01
    )
    metrics_labels = [
        ("ttft_ms_mean", "TTFT mean (ms)"),
        ("tpot_ms_p50",  "TPOT p50 (ms)"),
    ]
    for ki, (metric, ylabel) in enumerate(metrics_labels):
        for mi, model in enumerate(models):
            ax = axes[ki][mi]
            sub = std[std["model"] == model]
            for config, lbl, col, ls in [
                ("no-offload",         "no-offload",         COL_NO_OFFLOAD, "-"),
                ("native-offload-20k", "native-offload-20k", COL_OFFLOAD,    "--"),
            ]:
                d = sub[sub["config"] == config].sort_values("rate")
                if d.empty:
                    continue
                ax.plot(rpos(d["rate"]), d[metric], marker="o", ms=5,
                        color=col, ls=ls, lw=2, label=lbl)
            ax.set_title(MODEL_LABELS[model], fontsize=10)
            ax.set_xlabel("Concurrency")
            ax.set_ylabel(ylabel)
            set_rate_xaxis(ax)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("latency_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  latency_curves.png")


def fig_replica_scaling(df):
    """Scaling efficiency -- replicas=2,4 vs N x replicas=1, standard profile."""
    std = df[df["profile"] == "standard"]
    rates = sorted(std["rate"].unique())
    models = [m for m in MODEL_ORDER if m in std["model"].unique()]

    fig, axes = plt.subplots(1, len(models), figsize=(14, 6))
    fig.suptitle(
        "Replica Scaling Efficiency -- RHOAI 3.3 Short-Context Workload",
        fontsize=13
    )

    for mi, model in enumerate(models):
        ax = axes[mi]
        for config, col, ls in [
            ("no-offload",         COL_NO_OFFLOAD, "-"),
            ("native-offload-20k", COL_OFFLOAD,    "--"),
        ]:
            r1 = std[(std["model"]==model) & (std["config"]==config) & (std["replicas"]==1)]
            for rep, marker, ls2, lbl_sfx in [
                (2, "o", ls,  "r=2/2xr1"),
                (4, "s", ":", "r=4/4xr1"),
            ]:
                rN = std[(std["model"]==model) & (std["config"]==config) & (std["replicas"]==rep)]
                effs, valid_r = [], []
                for rate in rates:
                    t1 = r1[r1["rate"]==rate]["gen_tok_s_mean"].values
                    tN = rN[rN["rate"]==rate]["gen_tok_s_mean"].values
                    if len(t1) and len(tN) and t1[0] > 0:
                        effs.append(tN[0] / (rep * t1[0]) * 100)
                        valid_r.append(rate)
                if effs:
                    lbl = f"{CONFIG_LABELS[config]} {lbl_sfx}"
                    ax.plot(
                        [RPOS_STANDARD[r] for r in valid_r], effs,
                        marker=marker, ms=5, color=col, ls=ls2, lw=2, label=lbl
                    )

        ax.axhline(100, color="gray", ls=":", lw=1.5, label="ideal (100%)")
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Scaling efficiency (%)")
        set_rate_xaxis(ax)
        ax.set_ylim(0, 165)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("replica_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  replica_scaling.png")


def fig_kv_cache_pressure(df):
    """GPU KV cache usage from PCP archives -- standard, replicas=1."""
    print("  Extracting PCP KV cache metrics (this may take a minute)...")
    sub = df[(df["profile"] == "standard") & (df["replicas"] == 1)]
    records = []
    for _, row in sub.iterrows():
        pcp_df = extract_pcp_kv_metrics(row["run_dir"])
        if pcp_df.empty or pcp_df["kv_usage"].isna().all():
            continue
        avg_kv = pcp_df["kv_usage"].dropna().mean() * 100
        max_kv = pcp_df["kv_usage"].dropna().max() * 100
        records.append({
            "model":  row["model"],
            "config": row["config"],
            "rate":   row["rate"],
            "avg_kv": avg_kv,
            "max_kv": max_kv,
        })

    if not records:
        print("  WARNING: no PCP KV cache data, skipping fig")
        return

    pcp = pd.DataFrame(records)
    models = [m for m in MODEL_ORDER if m in pcp["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(14, 6))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle("GPU KV Cache Usage vs Concurrency (replicas=1) -- from PCP Archives",
                 fontsize=12)

    for mi, model in enumerate(models):
        ax = axes[mi]
        sub2 = pcp[pcp["model"] == model]
        for config, lbl, col, ls in [
            ("no-offload",         "no-offload",         COL_NO_OFFLOAD, "-"),
            ("native-offload-20k", "native-offload-20k", COL_OFFLOAD,    "--"),
        ]:
            d = sub2[sub2["config"] == config].sort_values("rate")
            if d.empty:
                continue
            ax.plot(rpos(d["rate"]), d["avg_kv"], marker="o", ms=5,
                    color=col, ls=ls, lw=2, label=f"{lbl} (avg)")
            ax.fill_between(rpos(d["rate"]), d["avg_kv"], d["max_kv"],
                            alpha=0.12, color=col)

        blocks = BLOCK_COUNTS.get(model, {})
        gpu_b  = blocks.get("gpu", 0)
        cpu_b  = blocks.get("cpu", 0)
        extra  = (f"GPU blocks: {gpu_b:,}\n"
                  f"CPU blocks: {cpu_b:,}\n"
                  f"CPU/GPU ratio: {cpu_b/max(gpu_b,1)*100:.0f}%")
        ax.text(0.03, 0.97, extra, transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("GPU KV cache usage (%)")
        set_rate_xaxis(ax)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("kv_cache_pressure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  kv_cache_pressure.png")


def fig_ttft_itl_comparison(df):
    """Mean latency bar chart averaged over concurrency >= 50 -- standard."""
    std = df[(df["profile"] == "standard") & (df["rate"] >= 50)]
    summary = std.groupby(["model", "config", "replicas"]).agg(
        ttft=("ttft_ms_mean", "mean"),
        itl=("tpot_ms_mean", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Mean Latency (concurrency >= 50, short-context workload) by Model, Config, Replica Count",
        fontsize=12
    )
    for ki, (metric, ylabel, title_sfx) in enumerate([
        ("ttft", "TTFT mean (ms)",  "Time to First Token (mean)"),
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
        ax.tick_params(axis="x", rotation=40)
        ax.legend(title="Config", fontsize=9)
        ax.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig("latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  latency_comparison.png")




def fig_longctx_throughput(df):
    """Long-context throughput curves -- 3 models x 3 replica counts."""
    lctx = df[df["profile"] == "longctx"]
    models   = [m for m in MODEL_ORDER if m in lctx["model"].unique()]
    replicas = [1, 2, 4]

    fig, axes = plt.subplots(len(replicas), len(models),
                             figsize=(14, 13), sharey=False)
    fig.suptitle(
        "Long-Context Throughput (prompt=4096 tokens) -- RHOAI 3.3 (1x8xH200)",
        fontsize=13, y=1.01
    )

    for ri, rep in enumerate(replicas):
        for mi, model in enumerate(models):
            ax = axes[ri][mi]
            sub = lctx[(lctx["model"] == model) & (lctx["replicas"] == rep)]
            for config, lbl, col, ls in [
                ("no-offload",         "no-offload",         COL_NO_OFFLOAD, "-"),
                ("native-offload-20k", "native-offload-20k", COL_OFFLOAD,    "--"),
            ]:
                d = sub[sub["config"] == config].sort_values("rate")
                if d.empty:
                    continue
                util_vals = d["gpu_util"].dropna().unique()
                util_note = f" (util={util_vals[0]:.2f})" if len(util_vals) == 1 else ""
                ax.plot(rpos(d["rate"], "longctx"), d["gen_tok_s_mean"],
                        marker="o", ms=5, color=col, ls=ls, lw=2,
                        label=lbl + util_note)
            ax.set_title(
                f"{MODEL_LABELS[model]}  x  {rep} replica{'s' if rep > 1 else ''}",
                fontsize=10
            )
            ax.set_xlabel("Concurrency")
            ax.set_ylabel("Output throughput (tok/s)")
            set_rate_xaxis(ax, "longctx")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("longctx_throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  longctx_throughput.png")


def fig_longctx_offload_delta(df):
    """Long-context offload delta by model and replica count.

    For Llama-3.1-70B-BF16, replica=2 is omitted: the no-offload runs used
    gpu_util=0.60 while native-offload used 0.90, making them incomparable.
    Only replica pairs with matching utilization (within 0.05) are shown.
    """
    lctx  = df[df["profile"] == "longctx"]
    rates = sorted(lctx["rate"].unique())
    models = [m for m in MODEL_ORDER if m in lctx["model"].unique()]

    fig, axes = plt.subplots(1, len(models), figsize=(14, 6), sharey=True)
    if len(models) == 1:
        axes = [axes]
    fig.suptitle(
        "Long-Context Offload Throughput Delta (native-offload-20k vs no-offload)",
        fontsize=13
    )

    rep_colors = {1: COL_REPLICA1, 2: COL_REPLICA2, 4: COL_REPLICA4}

    for mi, model in enumerate(models):
        ax = axes[mi]
        for rep in [1, 2, 4]:
            base    = lctx[(lctx["model"]==model) & (lctx["config"]=="no-offload") & (lctx["replicas"]==rep)]
            offload = lctx[(lctx["model"]==model) & (lctx["config"]=="native-offload-20k") & (lctx["replicas"]==rep)]
            if base.empty or offload.empty:
                continue
            bu = base["gpu_util"].dropna().unique()
            ou = offload["gpu_util"].dropna().unique()
            if len(bu) == 1 and len(ou) == 1 and abs(bu[0] - ou[0]) >= 0.05:
                print(f"    Skipping {MODEL_LABELS[model]} longctx r={rep}: "
                      f"mismatched gpu_util (no-offload={bu[0]:.2f}, offload={ou[0]:.2f})")
                continue

            deltas, valid_r = [], []
            util_used = None
            for rate in rates:
                b_row = base[base["rate"]==rate]
                o_row = offload[offload["rate"]==rate]
                if b_row.empty or o_row.empty:
                    continue
                b_util = b_row["gpu_util"].iloc[0]
                o_util = o_row["gpu_util"].iloc[0]
                # Skip rate if gpu_util differs between configs
                if (not np.isnan(b_util) and not np.isnan(o_util) and
                        abs(b_util - o_util) >= 0.05):
                    continue
                b_val = b_row["gen_tok_s_mean"].values[0]
                o_val = o_row["gen_tok_s_mean"].values[0]
                if b_val > 0:
                    deltas.append((o_val - b_val) / b_val * 100)
                    valid_r.append(rate)
                    if util_used is None and not np.isnan(o_util):
                        util_used = o_util
            if deltas:
                col = rep_colors.get(rep, "gray")
                util_note = f" (util={util_used:.2f})" if util_used is not None else ""
                ax.plot([RPOS_LONGCTX[r] for r in valid_r], deltas,
                        marker="o", ms=5, color=col, lw=2,
                        label=f"r={rep}{util_note}")

        ax.axhline(0, color="black", ls="--", lw=1.0, alpha=0.5)
        ax.fill_between(range(len(RATES_LONGCTX)), 0, 30, alpha=0.05, color="green",
                        label="offload benefit")
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Throughput delta (%)")
        set_rate_xaxis(ax, "longctx")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("longctx_offload_delta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  longctx_offload_delta.png")


def fig_longctx_latency(df):
    """Latency delta plot for long-context -- FP8 configs with positive offload impact.

    Shows % change in TTFT p90 and TPOT p50 (native-offload-20k vs no-offload)
    for FP8 at all three replica counts. Negative values indicate reduced latency
    with offload. MoE is excluded: corrected data shows no positive offload impact
    for gpt-oss-120b under long-context conditions.
    """
    lctx = df[df["profile"] == "longctx"]
    rates = sorted(lctx["rate"].unique())

    panels = [
        ("Meta-Llama-3.1-70B-Instruct-FP8", 1, "Llama-3.1-70B-FP8  r=1"),
        ("Meta-Llama-3.1-70B-Instruct-FP8", 2, "Llama-3.1-70B-FP8  r=2"),
        ("Meta-Llama-3.1-70B-Instruct-FP8", 4, "Llama-3.1-70B-FP8  r=4"),
    ]
    metrics = [
        ("ttft_ms_p90", "TTFT p90 change (%)"),
        ("tpot_ms_p50", "TPOT p50 change (%)"),
    ]
    metric_colors = [PALETTE[0], PALETTE[1]]

    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(11, 8), sharey=False)
    fig.suptitle(
        "Long-Context Latency Delta -- Llama-3.1-70B-FP8 (native-offload-20k vs no-offload)\n"
        "Negative = latency reduced with offload",
        fontsize=13, y=1.02
    )

    for ki, (metric, ylabel) in enumerate(metrics):
        for pi, (model, rep, title) in enumerate(panels):
            ax = axes[ki][pi]
            base    = lctx[(lctx["model"]==model) & (lctx["config"]=="no-offload")         & (lctx["replicas"]==rep)]
            offload = lctx[(lctx["model"]==model) & (lctx["config"]=="native-offload-20k") & (lctx["replicas"]==rep)]
            deltas, valid_r = [], []
            for rate in rates:
                b = base[base["rate"]==rate][metric].values
                o = offload[offload["rate"]==rate][metric].values
                if len(b) and len(o) and b[0] > 0:
                    deltas.append((o[0] - b[0]) / b[0] * 100)
                    valid_r.append(rate)
            if deltas:
                ax.plot([RPOS_LONGCTX[r] for r in valid_r], deltas,
                        marker="o", ms=5, color=metric_colors[ki], lw=2)
            ax.axhline(0, color="black", ls="--", lw=1.0, alpha=0.5)
            ax.fill_between(range(len(RATES_LONGCTX)), -50, 0, alpha=0.05,
                            color="green", label="latency reduced")
            if ki == 0:
                ax.set_title(title, fontsize=10)
            ax.set_xlabel("Concurrency")
            ax.set_ylabel(ylabel)
            set_rate_xaxis(ax, "longctx")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig("longctx_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  longctx_latency.png")


def save_summary_csv(df):
    cols = ["profile", "model", "config", "replicas", "rate", "gpu_util",
            "gen_tok_s_mean", "ttft_ms_p50", "ttft_ms_p90", "ttft_ms_mean",
            "itl_ms_p50", "tpot_ms_p50", "tpot_ms_mean", "req_s_mean",
            "completed", "errors", "duration_s"]
    df[cols].sort_values(["profile", "model", "replicas", "config", "rate"]).to_csv(
        "results_summary.csv", index=False
    )
    print("  results_summary.csv")


def print_peak_summary(df):
    print("\n=== Peak Throughput Summary ===")
    for profile in ["standard", "kvstress", "longctx"]:
        sub = df[df["profile"] == profile]
        if sub.empty:
            continue
        print(f"\n-- {profile} --")
        for model in MODEL_ORDER:
            if model not in sub["model"].unique():
                continue
            print(f"  {MODEL_LABELS[model]}:")
            for rep in sorted(sub["replicas"].unique()):
                base    = sub[(sub["model"]==model) & (sub["config"]=="no-offload") & (sub["replicas"]==rep)]
                offload = sub[(sub["model"]==model) & (sub["config"]=="native-offload-20k") & (sub["replicas"]==rep)]
                if base.empty or offload.empty:
                    continue
                if profile == "longctx":
                    bu = base["gpu_util"].dropna().unique()
                    ou = offload["gpu_util"].dropna().unique()
                    if len(bu) == 1 and len(ou) == 1 and abs(bu[0] - ou[0]) >= 0.05:
                        print(f"    r={rep}: SKIP mismatched gpu_util "
                              f"(no-offload={bu[0]:.2f} offload={ou[0]:.2f})")
                        continue
                peak_b = base["gen_tok_s_mean"].max()
                peak_o = offload["gen_tok_s_mean"].max()
                rate_b = base.loc[base["gen_tok_s_mean"].idxmax(), "rate"]
                rate_o = offload.loc[offload["gen_tok_s_mean"].idxmax(), "rate"]
                delta  = (peak_o - peak_b) / peak_b * 100
                print(f"    r={rep}: no-offload {peak_b:.1f} @{rate_b}, "
                      f"offload {peak_o:.1f} @{rate_o}  ({delta:+.1f}%)")


# -- Main ---------------------------------------------------------------------
def main():
    os.chdir(Path(__file__).parent.parent)
    print("Loading guidellm results...")
    df = load_all_results()
    print(f"  Loaded {len(df)} runs across {df['model'].nunique()} models")
    print(f"  Models:   {sorted(df['model'].unique())}")
    print(f"  Profiles: {sorted(df['profile'].unique())}")
    print(f"  Configs:  {sorted(df['config'].unique())}")
    print(f"  Replicas: {sorted(df['replicas'].unique())}")
    print(f"  Errors:   {int(df['errors'].sum())} total across all runs")
    print()

    print("Generating figures...")
    fig_throughput_curves(df)
    fig_offload_impact_heatmap(df)
    fig_kvstress_heatmap(df)
    fig_latency_curves(df)
    fig_replica_scaling(df)
    fig_kv_cache_pressure(df)
    fig_ttft_itl_comparison(df)
    fig_longctx_throughput(df)
    fig_longctx_offload_delta(df)
    fig_longctx_latency(df)
    save_summary_csv(df)

    print_peak_summary(df)


if __name__ == "__main__":
    main()
