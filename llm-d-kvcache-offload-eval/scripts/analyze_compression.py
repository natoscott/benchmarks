#!/usr/bin/env python3
"""
analyze_compression.py — Extract and visualise compression research results.

Reads three data sources and produces publication-quality figures:
  1. Offline characterisation Parquet (from benchmark_compressors.py)
  2. Online benchmark guidellm JSON results (from run-valkey-compression.sh)
  3. PCP Parquet archives (from pcp2arrow) — compression Prometheus metrics,
     CPU utilisation, PCIe bandwidth

Usage
-----
    python scripts/analyze_compression.py \\
        --characterisation results/compression_characterisation.parquet \\
        --results-dir results/ \\
        --pcp-dir results/ \\
        --output-dir analysis/

Figures produced
----------------
  compression_ratio_by_algo.png      - Ratio × algorithm, all models
  compression_ratio_by_layer.png     - Ratio × layer group (early/mid/late)
  compression_speed_tradeoff.png     - Ratio vs compress MB/s scatter (Pareto)
  online_throughput_by_algo.png      - guidellm tokens/sec × algorithm
  online_ttft_by_algo.png            - TTFT P95 × algorithm × concurrency
  online_itl_by_algo.png             - ITL P95 × algorithm
  cpu_overhead_by_algo.png           - CPU utilisation during compression
  effective_bandwidth_by_algo.png    - Effective PCIe bandwidth (data/time)
  compression_benefit_analysis.png   - Transfer time saved vs CPU overhead added
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Visualization standards (from repo CLAUDE.md)
PALETTE   = "muted"
FIG_W     = 14
FIG_H     = 8
FONT_SIZE = 11
matplotlib.rcParams.update({"font.size": FONT_SIZE})

# Algorithm display order (from fastest/lowest-ratio to slowest/highest-ratio)
ALGO_ORDER = [
    "none",
    "lz4",
    "zstd-1",
    "zstd-3",
    "zstd-10",
    "zstd-19",
    "blosc2-shuffle-zstd-3",
    "blosc2-bitshuffle-zstd-3",
    "lzma-6",
]

# Map benchmark PARAMETERS field to algorithm name
PARAMS_TO_ALGO = {
    "valkey-dram-none":              "none",
    "valkey-dram-lz4":               "lz4",
    "valkey-dram-zstd1":             "zstd-1",
    "valkey-dram-zstd3":             "zstd-3",
    "valkey-dram-zstd10":            "zstd-10",
    "valkey-dram-blosc2-shuffle":    "blosc2-shuffle-zstd-3",
    "valkey-dram-blosc2-bitshuffle": "blosc2-bitshuffle-zstd-3",
    "valkey-dram-lzma":              "lzma-6",
    # Baselines from v0.5.1 for direct comparison
    "no-offload":                    "no-offload (baseline)",
    "native-offload-20k":            "native-offload-20k",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_characterisation(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        logger.warning("Characterisation Parquet not found: %s", path)
        return None
    df = pd.read_parquet(path)
    logger.info("Loaded characterisation: %d rows", len(df))
    return df


def _parse_run_id(run_id: str) -> dict:
    """
    Parse a benchmark run directory name into its components.
    Format: {hardware}_{software}_{model}_{parameters}_replica{n}_rate{r}
    """
    pattern = r"^(.+?)_(.+?)_(.+?)_(.+?)_replica(\d+)_rate(\d+)$"
    m = re.match(pattern, run_id)
    if not m:
        return {}
    return {
        "hardware":   m.group(1),
        "software":   m.group(2),
        "model_name": m.group(3),
        "parameters": m.group(4),
        "replicas":   int(m.group(5)),
        "rate":       int(m.group(6)),
    }


def load_guidellm_results(results_dir: str) -> pd.DataFrame:
    """
    Load guidellm JSON results from all benchmark run directories.
    Extracts per-run throughput, TTFT, ITL statistics.
    """
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(results_dir, "*_rate*"))):
        run_id = os.path.basename(run_dir)
        meta = _parse_run_id(run_id)
        if not meta:
            continue

        # Load guidellm results
        json_gz = os.path.join(run_dir, "guidellm-results.json.zst")
        json_plain = os.path.join(run_dir, "guidellm-results.json")
        json_path = json_gz if os.path.exists(json_gz) else json_plain
        if not os.path.exists(json_path):
            continue

        try:
            if json_path.endswith(".zst"):
                import zstandard
                with open(json_path, "rb") as f:
                    dctx = zstandard.ZstdDecompressor()
                    data = json.loads(dctx.decompress(f.read()))
            else:
                with open(json_path) as f:
                    data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", json_path, e)
            continue

        # Extract benchmark results (structure varies by guidellm version)
        benchmarks = data if isinstance(data, list) else data.get("benchmarks", [data])
        for bench in benchmarks:
            stats = bench.get("request_concurrency_stats", bench)
            row = {
                "run_id":        run_id,
                "algorithm":     PARAMS_TO_ALGO.get(meta["parameters"], meta["parameters"]),
                "parameters":    meta["parameters"],
                "model_name":    meta["model_name"],
                "rate":          meta["rate"],
                "hardware":      meta["hardware"],
                # Throughput
                "output_tok_per_sec": stats.get("output_tokens_per_second", None),
                "request_per_sec":    stats.get("request_per_second", None),
                # TTFT
                "ttft_mean_ms":   _ms(stats.get("time_to_first_token_ms", {}).get("mean")),
                "ttft_p50_ms":    _ms(stats.get("time_to_first_token_ms", {}).get("median")),
                "ttft_p95_ms":    _ms(stats.get("time_to_first_token_ms", {}).get("p95")),
                "ttft_p99_ms":    _ms(stats.get("time_to_first_token_ms", {}).get("p99")),
                # ITL
                "itl_mean_ms":    _ms(stats.get("inter_token_latency_ms", {}).get("mean")),
                "itl_p50_ms":     _ms(stats.get("inter_token_latency_ms", {}).get("median")),
                "itl_p95_ms":     _ms(stats.get("inter_token_latency_ms", {}).get("p95")),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Loaded guidellm results: %d runs", len(df))
    return df


def _ms(val):
    """Return float ms or None."""
    if val is None:
        return None
    return float(val)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def fig_compression_ratio_by_algo(char_df: pd.DataFrame, out_dir: str) -> None:
    """
    Phase 1 key result: compression ratio by algorithm, faceted by model.
    This is the primary figure answering "does lossless compression work?"
    """
    fig, axes = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    pal = sns.color_palette(PALETTE, n_colors=len(ALGO_ORDER))

    models = char_df["model"].unique()
    present_algos = [a for a in ALGO_ORDER if a in char_df["algorithm"].unique()]

    plot_df = (
        char_df[char_df["algorithm"].isin(present_algos)]
        .groupby(["algorithm", "model"])["ratio"]
        .agg(["mean", "std"])
        .reset_index()
    )

    sns.barplot(
        data=plot_df,
        x="algorithm", y="mean",
        hue="model", order=present_algos,
        palette=PALETTE, ax=axes,
        errorbar=None,
    )

    # Add reference line at ratio=1.0 (no compression)
    axes.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
                 label="No compression (ratio=1)")
    axes.set_xlabel("Compression Algorithm")
    axes.set_ylabel("Compression Ratio (×)\n(higher = more compressed)")
    axes.set_title(
        "KV Cache Lossless Compression Ratio by Algorithm and Model\n"
        "(Phase 1 offline characterisation — FP16/BF16 tensors)"
    )
    axes.tick_params(axis="x", rotation=30)
    axes.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    path = os.path.join(out_dir, "compression_ratio_by_algo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", path)


def fig_compression_speed_tradeoff(char_df: pd.DataFrame, out_dir: str) -> None:
    """
    Pareto scatter: compression ratio vs compression throughput.
    Key design aid: shows the optimal algorithms for different scenarios
    (network-bound = high ratio, latency-bound = high MB/s).
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H // 2 + 2))
    pal = sns.color_palette(PALETTE, n_colors=len(ALGO_ORDER))

    summary = (
        char_df.groupby("algorithm")[
            ["ratio", "compress_mbs", "decompress_mbs"]
        ].mean().reset_index()
    )

    for ax, (x_col, x_label) in zip(
        axes,
        [("compress_mbs", "Compression Throughput (MB/s)"),
         ("decompress_mbs", "Decompression Throughput (MB/s)")]
    ):
        for i, row in summary.iterrows():
            alg = row["algorithm"]
            color = pal[ALGO_ORDER.index(alg)] if alg in ALGO_ORDER else "grey"
            ax.scatter(row[x_col], row["ratio"], color=color, s=80, zorder=3)
            ax.annotate(alg, (row[x_col], row["ratio"]),
                        textcoords="offset points", xytext=(5, 2),
                        fontsize=8)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean Compression Ratio (×)")
        ax.set_title(f"Ratio vs {x_label.split('(')[0].strip()}")

    fig.suptitle("Compression Speed–Ratio Trade-off\n"
                 "Top-right = best; GPU←CPU path needs high decompress MB/s",
                 y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "compression_speed_tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", path)


def fig_online_throughput(bench_df: pd.DataFrame, out_dir: str) -> None:
    """
    Phase 2/3 key result: output token throughput by algorithm × concurrency.
    Shows whether compression overhead reduces serving throughput.
    """
    models = bench_df["model_name"].unique()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(FIG_W, FIG_H), sharey=False)
    if n == 1:
        axes = [axes]

    present_algos = [a for a in ALGO_ORDER
                     if a in bench_df["algorithm"].unique()]
    pal = sns.color_palette(PALETTE, n_colors=len(present_algos))
    color_map = dict(zip(present_algos, pal))

    for ax, model in zip(axes, models):
        mdf = bench_df[bench_df["model_name"] == model]
        for algo in present_algos:
            adf = mdf[mdf["algorithm"] == algo].sort_values("rate")
            if adf.empty:
                continue
            ax.plot(adf["rate"], adf["output_tok_per_sec"],
                    marker="o", label=algo, color=color_map[algo])
        ax.set_title(model)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Output Tokens/sec")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Output Throughput vs Concurrency by Compression Algorithm\n"
        "(lower = compression overhead dominates)"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "online_throughput_by_algo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", path)


def fig_ttft_comparison(bench_df: pd.DataFrame, out_dir: str) -> None:
    """
    TTFT P95 by algorithm — shows whether compression adds to prefill latency.
    """
    # Peak throughput rate per model × algorithm
    idx = bench_df.groupby(["model_name", "algorithm"])["output_tok_per_sec"].idxmax()
    peak_df = bench_df.loc[idx.dropna()]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    present_algos = [a for a in ALGO_ORDER if a in peak_df["algorithm"].unique()]

    sns.barplot(
        data=peak_df[peak_df["algorithm"].isin(present_algos)],
        x="algorithm", y="ttft_p95_ms",
        hue="model_name", order=present_algos,
        palette=PALETTE, ax=ax,
    )
    ax.set_xlabel("Compression Algorithm")
    ax.set_ylabel("TTFT P95 (ms)")
    ax.set_title(
        "Time-to-First-Token P95 at Peak Throughput Rate\n"
        "(at the concurrency level giving highest throughput)"
    )
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    path = os.path.join(out_dir, "online_ttft_by_algo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", path)


def print_summary_table(char_df: pd.DataFrame | None, bench_df: pd.DataFrame) -> None:
    """Print a results table suitable for direct inclusion in a paper."""
    print("\n" + "=" * 80)
    print("COMPRESSION CHARACTERISATION SUMMARY (Phase 1 Offline)")
    print("=" * 80)
    if char_df is not None and not char_df.empty:
        summary = (
            char_df.groupby("algorithm")[["ratio", "compress_mbs", "decompress_mbs"]]
            .mean().round(3)
            .sort_values("ratio", ascending=False)
        )
        print(summary.to_string())
        print(f"\nBest ratio:      {summary['ratio'].idxmax()} "
              f"({summary['ratio'].max():.3f}×)")
        print(f"Fastest compress: {summary['compress_mbs'].idxmax()} "
              f"({summary['compress_mbs'].max():.0f} MB/s)")
    else:
        print("(no characterisation data)")

    print("\n" + "=" * 80)
    print("ONLINE BENCHMARK SUMMARY (Phase 3 guidellm)")
    print("=" * 80)
    if not bench_df.empty:
        peak = (
            bench_df.groupby(["model_name", "algorithm"])["output_tok_per_sec"]
            .max().unstack("algorithm").round(1)
        )
        print(peak.to_string())

        # Show overhead vs no-offload
        if "none" in bench_df["algorithm"].unique():
            baseline = bench_df[bench_df["algorithm"] == "none"].groupby(
                "model_name"
            )["output_tok_per_sec"].max()
            print("\n--- Compression overhead vs no-compression baseline ---")
            for algo in bench_df["algorithm"].unique():
                if algo == "none":
                    continue
                algo_peak = bench_df[bench_df["algorithm"] == algo].groupby(
                    "model_name"
                )["output_tok_per_sec"].max()
                delta = ((algo_peak - baseline) / baseline * 100).mean()
                print(f"  {algo:<35}: {delta:+.1f}% mean throughput change")
    else:
        print("(no online benchmark data — run guidellm benchmarks first)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--characterisation", default=None,
                        help="Parquet from benchmark_compressors.py")
    parser.add_argument("--results-dir", default="results/",
                        help="Directory containing benchmark run subdirectories")
    parser.add_argument("--output-dir", default="analysis/",
                        help="Directory for output figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    char_df = load_characterisation(args.characterisation)
    bench_df = load_guidellm_results(args.results_dir)

    if char_df is not None and not char_df.empty:
        fig_compression_ratio_by_algo(char_df, args.output_dir)
        fig_compression_speed_tradeoff(char_df, args.output_dir)
        logger.info("Phase 1 characterisation figures written.")
    else:
        logger.info("No characterisation data — skipping Phase 1 figures.")

    if not bench_df.empty:
        fig_online_throughput(bench_df, args.output_dir)
        fig_ttft_comparison(bench_df, args.output_dir)
        logger.info("Phase 2/3 online figures written.")
    else:
        logger.info("No online benchmark data found in %s.", args.results_dir)

    print_summary_table(char_df, bench_df)


if __name__ == "__main__":
    main()
