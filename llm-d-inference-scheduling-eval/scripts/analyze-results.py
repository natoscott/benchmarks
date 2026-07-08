#!/usr/bin/env python3
"""Analyze EPP scheduling evaluation results.

Compares prior-default vs optimized-baseline EPP configurations across
models and workload profiles. Produces comparison tables, delta analysis,
and visualizations.

Usage:
    python3 scripts/analyze-results.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
ANALYSIS_DIR = REPO_ROOT / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

# Project visualization standards
sns.set_palette("muted")
plt.rcParams.update({"font.size": 11})
FIG_SIZE = (14, 8)

MODELS = {
    "Qwen3-30B-A3B": {"label": "Qwen3-30B-A3B", "tp": 1, "replicas": 8},
    "Llama-3.3-70B-FP8": {"label": "Llama-3.3-70B-FP8", "tp": 2, "replicas": 4},
    "gpt-oss-120b": {"label": "gpt-oss-120b", "tp": 4, "replicas": 2},
}

PROFILES = ["multi-turn", "heavy-heterogeneous", "prefix-cache-stress"]
CONFIGS = ["prior-default", "optimized-baseline"]


def load_run(result_dir: Path) -> list[dict]:
    """Load all guidellm JSON results from a run directory."""
    benchmarks = []
    for f in sorted(result_dir.glob("*.json.zst")):
        import subprocess
        raw = subprocess.run(
            ["zstd", "-d", "-c", str(f)],
            capture_output=True, check=True
        ).stdout
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  WARNING: corrupt JSON in {f.name}, skipping", file=sys.stderr)
            continue
        for b in data.get("benchmarks", []):
            strategy = b.get("config", {}).get("strategy", {})
            streams = strategy.get("streams", strategy.get("rate", 0))
            kind = strategy.get("type_", "unknown")
            metrics = b.get("metrics", {})
            benchmarks.append({
                "kind": kind,
                "streams": streams,
                "rate": strategy.get("rate"),
                "duration": b.get("duration", 0),
                "requests_total": metrics.get("request_totals", {}).get("successful", 0),
                "requests_errored": metrics.get("request_totals", {}).get("errored", 0),
                "ttft_p50": metrics.get("time_to_first_token_ms", {}).get("successful", {}).get("percentiles", {}).get("p50", None),
                "ttft_p95": metrics.get("time_to_first_token_ms", {}).get("successful", {}).get("percentiles", {}).get("p95", None),
                "ttft_p99": metrics.get("time_to_first_token_ms", {}).get("successful", {}).get("percentiles", {}).get("p99", None),
                "ttft_mean": metrics.get("time_to_first_token_ms", {}).get("successful", {}).get("mean", None),
                "itl_p50": metrics.get("inter_token_latency_ms", {}).get("successful", {}).get("percentiles", {}).get("p50", None),
                "itl_p95": metrics.get("inter_token_latency_ms", {}).get("successful", {}).get("percentiles", {}).get("p95", None),
                "itl_p99": metrics.get("inter_token_latency_ms", {}).get("successful", {}).get("percentiles", {}).get("p99", None),
                "tpot_p50": metrics.get("time_per_output_token_ms", {}).get("successful", {}).get("percentiles", {}).get("p50", None),
                "tpot_p95": metrics.get("time_per_output_token_ms", {}).get("successful", {}).get("percentiles", {}).get("p95", None),
                "throughput_input_tps": metrics.get("prompt_tokens_per_second", {}).get("successful", {}).get("mean", None),
                "throughput_output_tps": metrics.get("output_tokens_per_second", {}).get("successful", {}).get("mean", None),
                "throughput_rps": metrics.get("requests_per_second", {}).get("successful", {}).get("mean", None),
                "concurrency_mean": metrics.get("request_concurrency", {}).get("successful", {}).get("mean", None),
                "request_latency_p50": metrics.get("request_latency", {}).get("successful", {}).get("percentiles", {}).get("p50", None),
                "request_latency_p99": metrics.get("request_latency", {}).get("successful", {}).get("percentiles", {}).get("p99", None),
            })
    return benchmarks


def build_dataframe() -> pd.DataFrame:
    """Build a DataFrame from all result directories."""
    rows = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or not (d / "benchmark-config.txt").exists():
            continue
        parts = d.name.split("_")
        # Parse: HARDWARE_SOFTWARE_MODEL_PROFILE_CONFIG_replicaN
        hardware = parts[0]
        software = parts[1]
        # Model name may contain hyphens, profile and config also
        # Read from benchmark-config.txt for accuracy
        config_txt = (d / "benchmark-config.txt").read_text()
        cfg = {}
        for line in config_txt.strip().split("\n"):
            if ": " in line:
                k, v = line.split(": ", 1)
                cfg[k.strip()] = v.strip()

        model_name = cfg.get("Model Name", "unknown")
        profile = cfg.get("Profile", "unknown")
        epp_config = cfg.get("EPP Config", "unknown")
        replicas = int(cfg.get("Replicas", 1))

        benchmarks = load_run(d)
        for b in benchmarks:
            b["model"] = model_name
            b["profile"] = profile
            b["epp_config"] = epp_config
            b["replicas"] = replicas
            b["hardware"] = hardware
            b["software"] = software
            rows.append(b)

    return pd.DataFrame(rows)


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage deltas between optimized-baseline and prior-default."""
    metrics = [
        "throughput_output_tps", "throughput_rps", "ttft_p50", "ttft_p95",
        "ttft_p99", "itl_p50", "itl_p95", "tpot_p50", "request_latency_p50",
        "request_latency_p99",
    ]

    rows = []
    for (model, profile, streams), group in df.groupby(["model", "profile", "streams"]):
        prior = group[group["epp_config"] == "prior-default"]
        optimized = group[group["epp_config"] == "optimized-baseline"]
        if prior.empty or optimized.empty:
            continue
        row = {"model": model, "profile": profile, "streams": streams}
        for m in metrics:
            pval = prior[m].iloc[0]
            oval = optimized[m].iloc[0]
            if pval is not None and oval is not None and pval != 0:
                delta_pct = ((oval - pval) / abs(pval)) * 100
                row[f"{m}_prior"] = pval
                row[f"{m}_optimized"] = oval
                row[f"{m}_delta_pct"] = delta_pct
            else:
                row[f"{m}_prior"] = pval
                row[f"{m}_optimized"] = oval
                row[f"{m}_delta_pct"] = None
        rows.append(row)

    return pd.DataFrame(rows)


def plot_ttft_comparison(df: pd.DataFrame, deltas: pd.DataFrame):
    """TTFT p50 and p99 comparison across concurrency levels for multi-turn."""
    mt = df[df["profile"] == "multi-turn"].copy()
    if mt.empty:
        return

    for model in mt["model"].unique():
        mdf = mt[mt["model"] == model].sort_values("streams")

        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

        for ax, metric, label in [
            (axes[0], "ttft_p50", "TTFT p50 (ms)"),
            (axes[1], "ttft_p99", "TTFT p99 (ms)"),
        ]:
            for config in CONFIGS:
                cdf = mdf[mdf["epp_config"] == config].sort_values("streams")
                if cdf.empty:
                    continue
                ax.plot(cdf["streams"], cdf[metric], marker="o",
                        label=config.replace("-", " ").title())
            ax.set_xlabel("Concurrent Streams")
            ax.set_ylabel(label)
            ax.set_title(f"{model}: {label}")
            ax.legend()
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"ttft_comparison_{model.replace('/', '_').replace('.', '_')}.png"
        fig.savefig(ANALYSIS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_throughput_comparison(df: pd.DataFrame):
    """Output tokens/sec comparison across concurrency levels."""
    mt = df[df["profile"] == "multi-turn"].copy()
    if mt.empty:
        return

    for model in mt["model"].unique():
        mdf = mt[mt["model"] == model].sort_values("streams")

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for config in CONFIGS:
            cdf = mdf[mdf["epp_config"] == config].sort_values("streams")
            if cdf.empty:
                continue
            ax.plot(cdf["streams"], cdf["throughput_output_tps"], marker="o",
                    label=config.replace("-", " ").title())
        ax.set_xlabel("Concurrent Streams")
        ax.set_ylabel("Output Tokens/sec")
        ax.set_title(f"{model}: Output Throughput vs Concurrency")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"throughput_comparison_{model.replace('/', '_').replace('.', '_')}.png"
        fig.savefig(ANALYSIS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_itl_comparison(df: pd.DataFrame):
    """ITL p50 and p95 for multi-turn profile."""
    mt = df[df["profile"] == "multi-turn"].copy()
    if mt.empty:
        return

    for model in mt["model"].unique():
        mdf = mt[mt["model"] == model].sort_values("streams")

        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

        for ax, metric, label in [
            (axes[0], "itl_p50", "ITL p50 (ms)"),
            (axes[1], "itl_p95", "ITL p95 (ms)"),
        ]:
            for config in CONFIGS:
                cdf = mdf[mdf["epp_config"] == config].sort_values("streams")
                if cdf.empty:
                    continue
                ax.plot(cdf["streams"], cdf[metric], marker="o",
                        label=config.replace("-", " ").title())
            ax.set_xlabel("Concurrent Streams")
            ax.set_ylabel(label)
            ax.set_title(f"{model}: {label}")
            ax.legend()
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"itl_comparison_{model.replace('/', '_').replace('.', '_')}.png"
        fig.savefig(ANALYSIS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_prefix_cache_rate_sweep(df: pd.DataFrame):
    """Throughput across Poisson arrival rates for prefix-cache-stress."""
    pcs = df[df["profile"] == "prefix-cache-stress"].copy()
    if pcs.empty:
        return

    for model in pcs["model"].unique():
        mdf = pcs[pcs["model"] == model].sort_values("streams")

        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

        # Throughput
        ax = axes[0]
        for config in CONFIGS:
            cdf = mdf[mdf["epp_config"] == config].sort_values("streams")
            if cdf.empty:
                continue
            ax.plot(cdf["streams"], cdf["throughput_output_tps"], marker=".",
                    label=config.replace("-", " ").title(), markersize=4)
        ax.set_xlabel("Poisson Arrival Rate (req/s)")
        ax.set_ylabel("Output Tokens/sec")
        ax.set_title(f"{model}: Throughput vs Arrival Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # TTFT p50
        ax = axes[1]
        for config in CONFIGS:
            cdf = mdf[mdf["epp_config"] == config].sort_values("streams")
            if cdf.empty:
                continue
            ax.plot(cdf["streams"], cdf["ttft_p50"], marker=".",
                    label=config.replace("-", " ").title(), markersize=4)
        ax.set_xlabel("Poisson Arrival Rate (req/s)")
        ax.set_ylabel("TTFT p50 (ms)")
        ax.set_title(f"{model}: TTFT p50 vs Arrival Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"prefix_cache_sweep_{model.replace('/', '_').replace('.', '_')}.png"
        fig.savefig(ANALYSIS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_heavy_hetero_comparison(df: pd.DataFrame):
    """Bar chart comparing configs for heavy-heterogeneous profile."""
    hh = df[df["profile"] == "heavy-heterogeneous"].copy()
    if hh.empty:
        return

    for model in hh["model"].unique():
        mdf = hh[hh["model"] == model].sort_values("streams")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = [
            ("throughput_output_tps", "Output Tokens/sec"),
            ("ttft_p50", "TTFT p50 (ms)"),
            ("itl_p50", "ITL p50 (ms)"),
        ]

        for ax, (metric, ylabel) in zip(axes, metrics):
            for config in CONFIGS:
                cdf = mdf[mdf["epp_config"] == config].sort_values("streams")
                if cdf.empty:
                    continue
                ax.plot(cdf["streams"], cdf[metric], marker="o",
                        label=config.replace("-", " ").title())
            ax.set_xlabel("Concurrent Streams")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{model}: {ylabel}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"heavy_hetero_{model.replace('/', '_').replace('.', '_')}.png"
        fig.savefig(ANALYSIS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_summary_heatmap(deltas: pd.DataFrame):
    """Heatmap of throughput deltas (optimized vs prior) across all runs."""
    mt_deltas = deltas[deltas["profile"] == "multi-turn"].copy()
    if mt_deltas.empty:
        return

    pivot = mt_deltas.pivot_table(
        index="model", columns="streams",
        values="throughput_output_tps_delta_pct"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        ax=ax, cbar_kws={"label": "Δ Output Tokens/sec (%)"}
    )
    ax.set_title("Multi-Turn: Throughput Change (Optimized Baseline vs Prior Default)")
    ax.set_xlabel("Concurrent Streams")
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "summary_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary_heatmap.png")


def generate_markdown_tables(df: pd.DataFrame, deltas: pd.DataFrame) -> str:
    """Generate markdown tables for the report."""
    sections = []

    # Per-model, per-profile comparison tables
    for model in df["model"].unique():
        for profile in df[df["model"] == model]["profile"].unique():
            subset = df[(df["model"] == model) & (df["profile"] == profile)]
            if subset.empty:
                continue

            delta_subset = deltas[(deltas["model"] == model) & (deltas["profile"] == profile)]

            level_col = "Streams" if profile != "prefix-cache-stress" else "Rate"
            header = f"### {model} — {profile}\n\n"
            header += f"| {level_col} | Config | Throughput (out tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p95 (ms) | Requests |\n"
            header += "|---|---|---|---|---|---|---|---|\n"

            rows = []
            for _, row in subset.sort_values(["streams", "epp_config"]).iterrows():
                level = int(row["streams"]) if pd.notna(row["streams"]) else "?"
                rows.append(
                    f"| {level} | {row['epp_config']} | "
                    f"{row['throughput_output_tps']:.1f}" if pd.notna(row['throughput_output_tps']) else "N/A"
                    + f" | {row['ttft_p50']:.1f}" if pd.notna(row.get('ttft_p50')) else " | N/A"
                    + f" | {row['ttft_p99']:.1f}" if pd.notna(row.get('ttft_p99')) else " | N/A"
                    + f" | {row['itl_p50']:.1f}" if pd.notna(row.get('itl_p50')) else " | N/A"
                    + f" | {row['itl_p95']:.1f}" if pd.notna(row.get('itl_p95')) else " | N/A"
                    + f" | {int(row['requests_total'])} |"
                )

            # Delta summary
            if not delta_subset.empty:
                header += "\n".join(rows)
                header += "\n\n**Deltas (optimized-baseline vs prior-default):**\n\n"
                header += f"| {level_col} | Throughput Δ | TTFT p50 Δ | TTFT p99 Δ | ITL p50 Δ |\n"
                header += "|---|---|---|---|---|\n"
                for _, dr in delta_subset.sort_values("streams").iterrows():
                    level = int(dr["streams"])

                    def fmt_delta(val):
                        if pd.isna(val):
                            return "N/A"
                        sign = "+" if val > 0 else ""
                        return f"{sign}{val:.1f}%"

                    header += (
                        f"| {level} | {fmt_delta(dr.get('throughput_output_tps_delta_pct'))} | "
                        f"{fmt_delta(dr.get('ttft_p50_delta_pct'))} | "
                        f"{fmt_delta(dr.get('ttft_p99_delta_pct'))} | "
                        f"{fmt_delta(dr.get('itl_p50_delta_pct'))} |\n"
                    )
            else:
                header += "\n".join(rows)

            sections.append(header)

    return "\n\n".join(sections)


def save_csv(df: pd.DataFrame, deltas: pd.DataFrame):
    """Save raw data and deltas as CSV."""
    df.to_csv(ANALYSIS_DIR / "all_results.csv", index=False)
    deltas.to_csv(ANALYSIS_DIR / "deltas.csv", index=False)
    print("  Saved all_results.csv, deltas.csv")


def main():
    print("Loading results...")
    df = build_dataframe()
    print(f"  Loaded {len(df)} benchmark data points from {df[['model','profile','epp_config']].drop_duplicates().shape[0]} runs")

    print("\nComputing deltas...")
    deltas = compute_deltas(df)
    print(f"  {len(deltas)} comparison points")

    print("\nGenerating visualizations...")
    plot_ttft_comparison(df, deltas)
    plot_throughput_comparison(df)
    plot_itl_comparison(df)
    plot_prefix_cache_rate_sweep(df)
    plot_heavy_hetero_comparison(df)
    plot_summary_heatmap(deltas)

    print("\nSaving CSV data...")
    save_csv(df, deltas)

    print("\nDone. Output in", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
