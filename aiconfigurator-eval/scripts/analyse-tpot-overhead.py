#!/usr/bin/env python3
"""
Generate TPOT overhead characterisation plots for the aiconfigurator-eval report.

Compares measured ITL (guidellm inter_token_latency_ms) against AIC predicted TPOT
across a grid of (ISL, batch_size) combinations collected by run-overhead-sweep.sh.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "aiconfigurator" / "src"))
from aiconfigurator.cli.api import cli_estimate

sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11

RESULTS_DIR = Path(__file__).parent.parent / "results" / "overhead-sweep-20260526"
OUTPUT_DIR = Path(__file__).parent.parent
MODEL = "Qwen/Qwen3-8B"
SYSTEM = "h200_sxm"
BACKEND = "vllm"
VERSION = "0.18.0"


def load_results() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json.zst")):
        m = re.search(r"isl(\d+)-rate(\d+)", path.name)
        if not m:
            continue
        isl, rate = int(m.group(1)), int(m.group(2))
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            d = json.loads(dctx.decompress(f.read()))
        met = d["benchmarks"][0]["metrics"]
        itl = met["inter_token_latency_ms"]["successful"]
        conc = met["request_concurrency"]["successful"]
        rows.append({
            "isl": isl, "b": rate,
            "itl_mean": itl["mean"], "itl_std": itl["std_dev"],
            "conc_mode": conc["mode"],
        })

    df = pd.DataFrame(rows).sort_values(["isl", "b"]).reset_index(drop=True)

    print("Querying AIC predictions...")
    tpot_vals = []
    for _, row in df.iterrows():
        try:
            r = cli_estimate(model_path=MODEL, system_name=SYSTEM,
                             backend_name=BACKEND, backend_version=VERSION,
                             isl=int(row.isl), osl=128, batch_size=int(row.b),
                             mode="agg")
            tpot_vals.append(r.tpot)
        except Exception as e:
            print(f"  WARN ISL={row.isl} b={row.b}: {e}", file=sys.stderr)
            tpot_vals.append(None)

    df["tpot_aic"] = tpot_vals
    df["overhead_ms"] = df["itl_mean"] - df["tpot_aic"]
    df["error_pct"] = (df["overhead_ms"] / df["tpot_aic"]) * 100
    return df


def plot_itl_vs_b(df: pd.DataFrame) -> None:
    """Line plot: measured ITL and AIC TPOT vs batch size, one panel per ISL."""
    isl_values = sorted(df["isl"].unique())
    palette = sns.color_palette("muted", n_colors=len(isl_values))

    fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for ax, isl, color in zip(axes, isl_values, palette):
        sub = df[df["isl"] == isl].sort_values("b")
        ax.plot(sub["b"], sub["itl_mean"], "o-", color=color, label="Measured ITL", linewidth=2)
        ax.fill_between(sub["b"],
                        sub["itl_mean"] - sub["itl_std"],
                        sub["itl_mean"] + sub["itl_std"],
                        alpha=0.15, color=color)
        ax.plot(sub["b"], sub["tpot_aic"], "s--", color="black", label="AIC TPOT",
                linewidth=1.5, markersize=5, alpha=0.7)
        ax.set_title(f"ISL={isl}", fontsize=10)
        ax.set_xlabel("Concurrency (b)", fontsize=9)
        ax.set_ylabel("Latency (ms)", fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(16))
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Measured ITL vs AIC TPOT — {MODEL} on {SYSTEM} vLLM {VERSION}\n"
        f"(OSL=128, shaded band = ±1 std dev of measured ITL)",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "tpot_itl_vs_b_by_isl.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_error_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: (ITL - AIC TPOT) in ms across (ISL, b) grid."""
    pivot = df.pivot_table(index="isl", columns="b", values="overhead_ms")
    pivot.index = [int(x) for x in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    sns.heatmap(
        pivot, ax=ax, cmap="RdBu_r", center=0,
        vmin=-vmax, vmax=vmax,
        annot=True, fmt=".1f",
        cbar_kws={"label": "ITL − AIC TPOT (ms)"},
    )
    ax.set_title(
        f"TPOT prediction error (ms): ITL_measured − TPOT_AIC\n"
        f"{MODEL} on {SYSTEM} vLLM {VERSION}, OSL=128\n"
        f"Blue = AIC over-predicts, Red = AIC under-predicts",
        fontsize=11,
    )
    ax.set_xlabel("Concurrency (b)", fontsize=11)
    ax.set_ylabel("ISL (tokens)", fontsize=11)
    fig.tight_layout()
    out = OUTPUT_DIR / "tpot_error_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_error_pct_by_b(df: pd.DataFrame) -> None:
    """Line plot: error % vs ISL, one line per batch size."""
    b_values = sorted(df["b"].unique())
    palette = sns.color_palette("muted", n_colors=len(b_values))

    fig, ax = plt.subplots(figsize=(14, 8))
    for b, color in zip(b_values, palette):
        sub = df[df["b"] == b].sort_values("isl")
        ax.plot(sub["isl"], sub["error_pct"], "o-", color=color,
                label=f"b={b}", linewidth=2, markersize=6)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.axhline(5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(-5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ISL (tokens)", fontsize=11)
    ax.set_ylabel("Prediction error (%)\n(ITL − AIC TPOT) / AIC TPOT × 100", fontsize=11)
    ax.set_title(
        f"AIC TPOT prediction error vs ISL by concurrency level\n"
        f"{MODEL} on {SYSTEM} vLLM {VERSION}, OSL=128\n"
        f"Dashed line = 0% (perfect), dotted lines = ±5%",
        fontsize=11,
    )
    ax.legend(title="Concurrency (b)", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "tpot_error_pct_vs_isl.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(df: pd.DataFrame) -> None:
    oh = df["overhead_ms"].dropna()
    pct = df["error_pct"].dropna()
    print("\n=== Summary statistics ===")
    print(f"Points:           {len(oh)}")
    print(f"Mean error:       {oh.mean():+.2f} ms")
    print(f"Mean abs error:   {oh.abs().mean():.2f} ms")
    print(f"Std dev error:    {oh.std():.2f} ms")
    print(f"Max under-pred:   {oh.max():+.2f} ms  (ISL={df.loc[oh.idxmax(),'isl']}, b={df.loc[oh.idxmax(),'b']})")
    print(f"Max over-pred:    {oh.min():+.2f} ms  (ISL={df.loc[oh.idxmin(),'isl']}, b={df.loc[oh.idxmin(),'b']})")
    print(f"Within ±5%:       {(pct.abs() <= 5).sum()} / {len(pct)} points")
    print(f"Within ±10%:      {(pct.abs() <= 10).sum()} / {len(pct)} points")


if __name__ == "__main__":
    df = load_results()
    print_summary(df)
    plot_itl_vs_b(df)
    plot_error_heatmap(df)
    plot_error_pct_by_b(df)
    # Save CSV for reference
    csv_path = RESULTS_DIR / "tpot_overhead_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
