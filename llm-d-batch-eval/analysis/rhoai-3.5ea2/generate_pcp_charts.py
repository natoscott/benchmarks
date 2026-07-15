#!/usr/bin/env python3
"""Generate PCP time-series charts for EA2 batch eval results."""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

# Visualization standards
sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11

RESULTS_DIR = Path("/home/nathans/git/benchmarks/llm-d-batch-eval/results")
OUTPUT_DIR = Path("/home/nathans/git/benchmarks/llm-d-batch-eval/analysis/rhoai-3.5ea2")


def parse_pmval_output(text):
    """Parse pmval output into (timestamps, values) lists.

    Skips 'No values available' lines and header lines.
    Returns only rows with numeric values.
    """
    timestamps = []
    values = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or "No values" in line:
            continue
        # Match lines like: "16:35:42.301             100.0"
        m = re.match(r"^(\d{2}:\d{2}:\d{2}\.\d+)\s+([\d.]+)", line)
        if m:
            time_str = m.group(1)
            val = float(m.group(2))
            # Parse time (date doesn't matter for relative plotting)
            t = datetime.strptime(time_str, "%H:%M:%S.%f")
            timestamps.append(t)
            values.append(val)
    return timestamps, values


def extract_metric(archive_path, metric, interval=10):
    """Run pmval and return parsed (timestamps, values)."""
    cmd = [
        "pmval", "-a", str(archive_path),
        "-t", str(interval),
        metric,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_pmval_output(result.stdout)


def find_archive(result_dir):
    """Find the PCP archive base path (without extension) in a result directory."""
    pcp_dir = result_dir / "pcp-archives"
    # Find the node subdirectory
    node_dirs = [d for d in pcp_dir.iterdir() if d.is_dir()]
    if not node_dirs:
        raise FileNotFoundError(f"No node directories in {pcp_dir}")
    node_dir = node_dirs[0]
    # Find .0 file (decompressed archive)
    archives = list(node_dir.glob("*.0"))
    if not archives:
        raise FileNotFoundError(f"No decompressed .0 archive in {node_dir}")
    # Return path without the .0 extension
    archive = archives[0]
    return str(archive).rsplit(".0", 1)[0]


def to_elapsed_minutes(timestamps):
    """Convert timestamps to elapsed minutes from first timestamp."""
    if not timestamps:
        return []
    t0 = timestamps[0]
    return [(t - t0).total_seconds() / 60.0 for t in timestamps]


def generate_inflight_comparison():
    """Chart 1: Dual-panel inflight requests comparison, Qwen3-8B r=1."""
    palette = sns.color_palette("muted")

    # Extract data
    ungated_dir = RESULTS_DIR / "2x8xH200_rhoai-3.5ea2_Qwen3-8B_ungated_replica1"
    aimd_dir = RESULTS_DIR / "2x8xH200_rhoai-3.5ea2_Qwen3-8B_aimd_replica1"

    ungated_archive = find_archive(ungated_dir)
    aimd_archive = find_archive(aimd_dir)

    metric = "openmetrics.batch_processor.model_inflight_requests"

    ts_ungated, vals_ungated = extract_metric(ungated_archive, metric)
    ts_aimd, vals_aimd = extract_metric(aimd_archive, metric)

    elapsed_ungated = to_elapsed_minutes(ts_ungated)
    elapsed_aimd = to_elapsed_minutes(ts_aimd)

    # Dual-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    # Left panel: ungated
    ax1.plot(elapsed_ungated, vals_ungated, color=palette[3], linewidth=2,
             label="Ungated")
    ax1.fill_between(elapsed_ungated, vals_ungated, alpha=0.15, color=palette[3])
    ax1.axhline(y=100, color=palette[3], linestyle="--", alpha=0.5, linewidth=1,
                label="Default limit (100)")
    ax1.set_xlabel("Elapsed Time (minutes)")
    ax1.set_ylabel("In-Flight Requests")
    ax1.set_title("Ungated (no AIMD)")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_ylim(-5, 115)
    ax1.set_xlim(left=0)

    # Right panel: AIMD
    ax2.plot(elapsed_aimd, vals_aimd, color=palette[0], linewidth=2,
             label="AIMD-gated")
    ax2.fill_between(elapsed_aimd, vals_aimd, alpha=0.15, color=palette[0])
    ax2.axhline(y=20, color=palette[0], linestyle="--", alpha=0.5, linewidth=1,
                label="AIMD limit (20)")
    ax2.set_xlabel("Elapsed Time (minutes)")
    ax2.set_title("AIMD Flow Control")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_xlim(left=0)

    fig.suptitle(
        "Batch Processor In-Flight Requests: Qwen3-8B, r=1\n"
        "EA2 perEndpoint enforcement limits AIMD-gated concurrency to 20 vs 100 ungated",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = OUTPUT_DIR / "pcp_inflight_comparison_Qwen3-8B_r1.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_aimd_concurrency():
    """Chart 2: AIMD concurrency limit time-series, FP8-70B r=1."""
    palette = sns.color_palette("muted")

    aimd_dir = RESULTS_DIR / "2x8xH200_rhoai-3.5ea2_Meta-Llama-3.1-70B-Instruct-FP8_aimd_replica1"
    archive = find_archive(aimd_dir)

    metric = "openmetrics.batch_processor.batch_processor_aimd_concurrency_limit"

    ts, vals = extract_metric(archive, metric)
    elapsed = to_elapsed_minutes(ts)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(elapsed, vals, color=palette[0], linewidth=2.5, label="AIMD concurrency limit")
    ax.fill_between(elapsed, vals, alpha=0.12, color=palette[0])

    # Reference line at 20
    ax.axhline(y=20, color=palette[2], linestyle="--", alpha=0.7, linewidth=1.5,
               label="Configured limit (20)")

    # Annotate statistics
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    min_val = np.min(vals)
    max_val = np.max(vals)

    stats_text = (
        f"Mean: {mean_val:.1f}\n"
        f"Std:  {std_val:.1f}\n"
        f"Min:  {min_val:.0f}\n"
        f"Max:  {max_val:.0f}"
    )
    ax.text(
        0.02, 0.97, stats_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    ax.set_xlabel("Elapsed Time (minutes)")
    ax.set_ylabel("AIMD Concurrency Limit")
    ax.set_title(
        "Batch Processor AIMD Concurrency Limit: Meta-Llama-3.1-70B-Instruct-FP8, r=1\n"
        "Limit remains flat at 20 throughout the run (no adaptation triggered)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(-2, 30)
    ax.set_xlim(left=0)

    fig.tight_layout()

    out_path = OUTPUT_DIR / "pcp_aimd_concurrency_FP8-70B_r1.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_inflight_comparison()
    generate_aimd_concurrency()
    print("Done.")
