#!/usr/bin/env python3
"""Preliminary analysis of llm-d batch gateway benchmark results.

Extracts guidellm metrics from all completed runs, produces comparison
tables and visualizations across scenarios and replica counts.

Usage:
    python3 scripts/analyze-results.py
"""

import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11

RESULTS_DIR = Path(__file__).parent.parent / "results"
ANALYSIS_DIR = Path(__file__).parent.parent / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

SCENARIOS = ["interactive-only", "ungated", "aimd", "aimd-flow-control"]


@dataclass
class RunMetrics:
    model: str
    scenario: str
    replicas: int
    phase: str  # burst or idle
    cycle: int
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    itl_p50: float = 0.0
    itl_p95: float = 0.0
    itl_p99: float = 0.0
    tpot_p50: float = 0.0
    tpot_p95: float = 0.0
    tpot_p99: float = 0.0
    req_latency_median: float = 0.0
    req_latency_p95: float = 0.0
    req_latency_p99: float = 0.0
    rps_mean: float = 0.0
    completed: int = 0
    errored: int = 0
    duration: float = 0.0


def extract_guidellm_metrics(json_path: Path) -> RunMetrics | None:
    """Extract metrics from a single guidellm JSON file."""
    try:
        proc = subprocess.run(
            ["zstd", "-d", "-c", str(json_path)],
            capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
    except Exception:
        return None

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        return None

    bench = benchmarks[0]
    m = bench.get("metrics", {})
    rt = m.get("request_totals", {})

    def pct(metric_key, sub="successful"):
        sub_data = m.get(metric_key, {}).get(sub, {})
        pcts = sub_data.get("percentiles", {})
        return (
            pcts.get("p50", 0.0) or 0.0,
            pcts.get("p95", 0.0) or 0.0,
            pcts.get("p99", 0.0) or 0.0,
        )

    def stat(metric_key, field, sub="successful"):
        return m.get(metric_key, {}).get(sub, {}).get(field, 0.0) or 0.0

    ttft = pct("time_to_first_token_ms")
    itl = pct("inter_token_latency_ms")
    tpot = pct("time_per_output_token_ms")
    rl = pct("request_latency")

    # Parse phase and cycle from filename: burst-2.json.zst -> (burst, 2)
    stem = json_path.name.replace(".json.zst", "")
    parts = stem.rsplit("-", 1)
    phase = parts[0] if len(parts) == 2 else stem
    cycle = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

    return RunMetrics(
        model="", scenario="", replicas=0,
        phase=phase, cycle=cycle,
        ttft_p50=ttft[0], ttft_p95=ttft[1], ttft_p99=ttft[2],
        itl_p50=itl[0], itl_p95=itl[1], itl_p99=itl[2],
        tpot_p50=tpot[0], tpot_p95=tpot[1], tpot_p99=tpot[2],
        req_latency_median=stat("request_latency", "median"),
        req_latency_p95=rl[1], req_latency_p99=rl[2],
        rps_mean=stat("requests_per_second", "mean"),
        completed=int(rt.get("successful", 0)),
        errored=int(rt.get("errored", 0)),
        duration=bench.get("duration", 0.0) or 0.0,
    )


def parse_run_id(dirname: str):
    """Parse run directory name into components."""
    # 2x8xH200_rhoai-3.5ea1_Qwen3-8B_interactive-only_replica1
    parts = dirname.split("_")
    hardware = parts[0]
    software = parts[1]
    model = parts[2]
    replica_part = parts[-1]  # replica1
    replicas = int(replica_part.replace("replica", ""))
    scenario = "_".join(parts[3:-1])
    return model, scenario, replicas


def load_all_results():
    """Load all benchmark results into a list of RunMetrics."""
    all_metrics = []

    for result_dir in sorted(RESULTS_DIR.iterdir()):
        if not result_dir.is_dir():
            continue
        config = result_dir / "benchmark-config.txt"
        if not config.exists():
            continue

        model, scenario, replicas = parse_run_id(result_dir.name)

        for json_file in sorted(result_dir.glob("*.json.zst")):
            # Skip non-guidellm files
            stem = json_file.name.replace(".json.zst", "")
            if not (stem.startswith("burst-") or stem.startswith("idle-")):
                continue
            if "warmup" in stem:
                continue
            rm = extract_guidellm_metrics(json_file)
            if rm and rm.completed > 0:
                rm.model = model
                rm.scenario = scenario
                rm.replicas = replicas
                all_metrics.append(rm)

    return all_metrics


def build_dataframe(metrics: list[RunMetrics]) -> pd.DataFrame:
    """Convert metrics list to a DataFrame."""
    records = []
    for m in metrics:
        records.append({
            "model": m.model,
            "scenario": m.scenario,
            "replicas": m.replicas,
            "phase": m.phase,
            "cycle": m.cycle,
            "ttft_p50": m.ttft_p50,
            "ttft_p95": m.ttft_p95,
            "ttft_p99": m.ttft_p99,
            "itl_p50": m.itl_p50,
            "itl_p95": m.itl_p95,
            "itl_p99": m.itl_p99,
            "tpot_p50": m.tpot_p50,
            "tpot_p95": m.tpot_p95,
            "tpot_p99": m.tpot_p99,
            "req_latency_median_s": m.req_latency_median,
            "req_latency_p95_s": m.req_latency_p95,
            "req_latency_p99_s": m.req_latency_p99,
            "rps": m.rps_mean,
            "completed": m.completed,
            "errored": m.errored,
            "duration_s": m.duration,
        })
    return pd.DataFrame(records)


def print_summary_table(df: pd.DataFrame):
    """Print scenario comparison tables per model."""
    burst = df[df["phase"] == "burst"].copy()
    if burst.empty:
        print("No burst phase data found")
        return

    for model in sorted(burst["model"].unique()):
        print(f"\n{'='*80}")
        print(f"  {model} — Burst Phase Latency (ms)")
        print(f"{'='*80}")

        mdf = burst[burst["model"] == model]
        agg = mdf.groupby(["scenario", "replicas"]).agg({
            "ttft_p50": "mean", "ttft_p95": "mean", "ttft_p99": "mean",
            "itl_p50": "mean", "itl_p95": "mean", "itl_p99": "mean",
            "tpot_p50": "mean", "tpot_p95": "mean", "tpot_p99": "mean",
            "rps": "mean",
            "completed": "sum", "errored": "sum",
        }).round(1)

        # Reorder scenarios
        scenario_order = [s for s in SCENARIOS if s in agg.index.get_level_values("scenario")]
        agg = agg.reindex(scenario_order, level="scenario")

        print(f"\n{'Scenario':<22s} {'R':>2s}  {'TTFT p50':>9s} {'p95':>7s} {'p99':>7s}"
              f"  {'ITL p50':>8s} {'p95':>7s} {'p99':>7s}"
              f"  {'TPOT p50':>9s} {'p95':>7s} {'p99':>7s}"
              f"  {'RPS':>6s} {'OK':>6s} {'Err':>4s}")
        print("-" * 120)

        for (scenario, replicas), row in agg.iterrows():
            print(f"{scenario:<22s} {replicas:>2d}"
                  f"  {row['ttft_p50']:>9.1f} {row['ttft_p95']:>7.1f} {row['ttft_p99']:>7.1f}"
                  f"  {row['itl_p50']:>8.1f} {row['itl_p95']:>7.1f} {row['itl_p99']:>7.1f}"
                  f"  {row['tpot_p50']:>9.1f} {row['tpot_p95']:>7.1f} {row['tpot_p99']:>7.1f}"
                  f"  {row['rps']:>6.1f} {int(row['completed']):>6d} {int(row['errored']):>4d}")

        # Compute overhead vs interactive-only baseline
        baseline = mdf[mdf["scenario"] == "interactive-only"]
        if not baseline.empty:
            print(f"\n  Overhead vs interactive-only baseline (TTFT p99):")
            for scenario in scenario_order:
                if scenario == "interactive-only":
                    continue
                sdf = mdf[mdf["scenario"] == scenario]
                for r in sorted(sdf["replicas"].unique()):
                    base_val = baseline[baseline["replicas"] == r]["ttft_p99"].mean()
                    test_val = sdf[sdf["replicas"] == r]["ttft_p99"].mean()
                    if base_val > 0:
                        delta_pct = ((test_val - base_val) / base_val) * 100
                        print(f"    {scenario} r={r}: {delta_pct:+.1f}% "
                              f"({base_val:.1f} -> {test_val:.1f} ms)")


def plot_ttft_comparison(df: pd.DataFrame):
    """Bar chart: TTFT p99 during burst across scenarios."""
    burst = df[df["phase"] == "burst"].copy()
    if burst.empty:
        return

    for model in sorted(burst["model"].unique()):
        mdf = burst[burst["model"] == model]
        agg = mdf.groupby(["scenario", "replicas"]).agg(
            ttft_p99=("ttft_p99", "mean")
        ).reset_index()

        scenario_order = [s for s in SCENARIOS if s in agg["scenario"].values]
        agg["scenario"] = pd.Categorical(agg["scenario"], categories=scenario_order, ordered=True)
        agg = agg.sort_values(["scenario", "replicas"])

        fig, ax = plt.subplots(figsize=(14, 8))
        agg["label"] = agg.apply(lambda r: f"{r['scenario']}\nr={int(r['replicas'])}", axis=1)

        bars = ax.bar(range(len(agg)), agg["ttft_p99"], color=sns.color_palette("muted", len(agg)))
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg["label"], fontsize=9)
        ax.set_ylabel("TTFT p99 (ms)")
        ax.set_title(f"{model} — TTFT p99 During Burst (lower is better)")

        for bar, val in zip(bars, agg["ttft_p99"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        fname = ANALYSIS_DIR / f"ttft_p99_burst_{model}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def plot_latency_by_replica(df: pd.DataFrame):
    """Line chart: latency vs replica count per scenario."""
    burst = df[df["phase"] == "burst"].copy()
    if burst.empty:
        return

    for model in sorted(burst["model"].unique()):
        mdf = burst[burst["model"] == model]
        agg = mdf.groupby(["scenario", "replicas"]).agg(
            ttft_p99=("ttft_p99", "mean"),
            tpot_p99=("tpot_p99", "mean"),
        ).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for scenario in SCENARIOS:
            sdf = agg[agg["scenario"] == scenario].sort_values("replicas")
            if sdf.empty:
                continue
            axes[0].plot(sdf["replicas"], sdf["ttft_p99"], marker="o", label=scenario)
            axes[1].plot(sdf["replicas"], sdf["tpot_p99"], marker="o", label=scenario)

        axes[0].set_xlabel("Replicas")
        axes[0].set_ylabel("TTFT p99 (ms)")
        axes[0].set_title(f"{model} — TTFT p99 vs Replicas")
        axes[0].legend(fontsize=8)

        axes[1].set_xlabel("Replicas")
        axes[1].set_ylabel("TPOT p99 (ms)")
        axes[1].set_title(f"{model} — TPOT p99 vs Replicas")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        fname = ANALYSIS_DIR / f"latency_vs_replicas_{model}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def plot_throughput_comparison(df: pd.DataFrame):
    """Bar chart: throughput (RPS) during burst."""
    burst = df[df["phase"] == "burst"].copy()
    if burst.empty:
        return

    for model in sorted(burst["model"].unique()):
        mdf = burst[burst["model"] == model]
        agg = mdf.groupby(["scenario", "replicas"]).agg(
            rps=("rps", "mean")
        ).reset_index()

        scenario_order = [s for s in SCENARIOS if s in agg["scenario"].values]
        agg["scenario"] = pd.Categorical(agg["scenario"], categories=scenario_order, ordered=True)
        agg = agg.sort_values(["scenario", "replicas"])

        fig, ax = plt.subplots(figsize=(14, 8))
        agg["label"] = agg.apply(lambda r: f"{r['scenario']}\nr={int(r['replicas'])}", axis=1)

        bars = ax.bar(range(len(agg)), agg["rps"], color=sns.color_palette("muted", len(agg)))
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg["label"], fontsize=9)
        ax.set_ylabel("Requests/sec")
        ax.set_title(f"{model} — Interactive Throughput During Burst")

        for bar, val in zip(bars, agg["rps"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        fname = ANALYSIS_DIR / f"throughput_burst_{model}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def plot_idle_vs_burst(df: pd.DataFrame):
    """Grouped bar: idle vs burst TTFT p99 per scenario (shows batch impact)."""
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        agg = mdf.groupby(["scenario", "replicas", "phase"]).agg(
            ttft_p99=("ttft_p99", "mean")
        ).reset_index()

        # Pick replica=1 for cleaner comparison
        r1 = agg[agg["replicas"] == agg["replicas"].min()]
        if r1.empty:
            continue

        scenario_order = [s for s in SCENARIOS if s in r1["scenario"].values]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(scenario_order))
        width = 0.35

        for i, phase in enumerate(["idle", "burst"]):
            vals = []
            for s in scenario_order:
                v = r1[(r1["scenario"] == s) & (r1["phase"] == phase)]["ttft_p99"]
                vals.append(v.mean() if not v.empty else 0)
            offset = (i - 0.5) * width
            bars = ax.bar([xi + offset for xi in x], vals, width, label=phase.capitalize())

        ax.set_xticks(x)
        ax.set_xticklabels(scenario_order, fontsize=9)
        ax.set_ylabel("TTFT p99 (ms)")
        r_val = int(agg["replicas"].min())
        ax.set_title(f"{model} r={r_val} — Idle vs Burst TTFT p99")
        ax.legend()

        plt.tight_layout()
        fname = ANALYSIS_DIR / f"idle_vs_burst_{model}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def plot_batch_timelines():
    """Plot batch completion timelines from batch-timeline.json files."""
    timelines = {}
    for result_dir in sorted(RESULTS_DIR.iterdir()):
        tl_file = result_dir / "batch-timeline.json"
        if not tl_file.exists():
            continue
        try:
            with open(tl_file) as f:
                tl = json.load(f)
            if not tl:
                continue
            model, scenario, replicas = parse_run_id(result_dir.name)
            label = f"{scenario} r={replicas}"
            elapsed = [entry["elapsed"] for entry in tl]
            completed = []
            for entry in tl:
                total = sum(j.get("completed", 0) for j in entry.get("jobs", []))
                completed.append(total)
            timelines[f"{model}/{label}"] = (elapsed, completed, model)
        except Exception:
            continue

    if not timelines:
        print("  No batch timelines found (will be available after re-run)")
        return

    models = sorted(set(v[2] for v in timelines.values()))
    for model in models:
        fig, ax = plt.subplots(figsize=(14, 6))
        for key, (elapsed, completed, m) in sorted(timelines.items()):
            if m != model:
                continue
            label = key.split("/", 1)[1]
            ax.plot(elapsed, completed, marker=".", markersize=3, label=label)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Batch Requests Completed")
        ax.set_title(f"{model} — Batch Completion Timeline")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = ANALYSIS_DIR / f"batch_timeline_{model}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def plot_error_rates(df: pd.DataFrame):
    """Bar chart: error rates across scenarios."""
    burst = df[df["phase"] == "burst"].copy()
    if burst.empty:
        return

    burst["error_rate"] = burst["errored"] / (burst["completed"] + burst["errored"]).replace(0, 1) * 100

    for model in sorted(burst["model"].unique()):
        mdf = burst[burst["model"] == model]
        agg = mdf.groupby(["scenario", "replicas"]).agg(
            error_rate=("error_rate", "mean"),
            total_errors=("errored", "sum"),
            total_completed=("completed", "sum"),
        ).reset_index()

        if agg["total_errors"].sum() == 0:
            print(f"  {model}: 0 errors across all scenarios — skipping error rate chart")
            continue

        scenario_order = [s for s in SCENARIOS if s in agg["scenario"].values]
        agg["scenario"] = pd.Categorical(agg["scenario"], categories=scenario_order, ordered=True)
        agg = agg.sort_values(["scenario", "replicas"])

        fig, ax = plt.subplots(figsize=(14, 6))
        agg["label"] = agg.apply(lambda r: f"{r['scenario']}\nr={int(r['replicas'])}", axis=1)
        bars = ax.bar(range(len(agg)), agg["error_rate"], color=sns.color_palette("muted", len(agg)))
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg["label"], fontsize=9)
        ax.set_ylabel("Error Rate (%)")
        ax.set_title(f"{model} — Error Rate During Burst")
        plt.tight_layout()
        fname = ANALYSIS_DIR / f"error_rate_{model}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def load_batch_final_status():
    """Load and summarize batch job final status across runs."""
    records = []
    for result_dir in sorted(RESULTS_DIR.iterdir()):
        status_file = result_dir / "batch-final-status.json"
        if not status_file.exists():
            continue
        try:
            with open(status_file) as f:
                data = json.load(f)
            model, scenario, replicas = parse_run_id(result_dir.name)
            for batch in data.get("data", []):
                rc = batch.get("request_counts", {})
                records.append({
                    "model": model,
                    "scenario": scenario,
                    "replicas": replicas,
                    "batch_id": batch.get("id", ""),
                    "status": batch.get("status", "unknown"),
                    "completed": rc.get("completed", 0),
                    "failed": rc.get("failed", 0),
                    "total": rc.get("total", 0),
                })
        except Exception:
            continue

    if records:
        bdf = pd.DataFrame(records)
        print("\n  Batch Job Summary:")
        for model in sorted(bdf["model"].unique()):
            mdf = bdf[bdf["model"] == model]
            for scenario in SCENARIOS:
                sdf = mdf[mdf["scenario"] == scenario]
                if sdf.empty:
                    continue
                total_completed = sdf["completed"].sum()
                total_failed = sdf["failed"].sum()
                total_expected = sdf["total"].sum()
                statuses = sdf["status"].value_counts().to_dict()
                print(f"    {model} {scenario}: "
                      f"{total_completed}/{total_expected} completed, "
                      f"{total_failed} failed, statuses={statuses}")
    else:
        print("\n  No batch final status files found (will be available after re-run)")


def main():
    print("Loading results...")
    metrics = load_all_results()
    print(f"  Loaded {len(metrics)} phase measurements from {RESULTS_DIR}")

    if not metrics:
        print("ERROR: No results found")
        sys.exit(1)

    df = build_dataframe(metrics)

    models = sorted(df["model"].unique())
    scenarios = sorted(df["scenario"].unique())
    replicas = sorted(df["replicas"].unique())
    print(f"  Models: {models}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Replicas: {replicas}")
    print(f"  Phases: {sorted(df['phase'].unique())}")

    # Save raw data
    csv_path = ANALYSIS_DIR / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Raw data: {csv_path}")

    # Summary tables
    print_summary_table(df)

    # Batch job summary
    load_batch_final_status()

    # Visualizations
    print("\nGenerating visualizations...")
    plot_ttft_comparison(df)
    plot_latency_by_replica(df)
    plot_throughput_comparison(df)
    plot_idle_vs_burst(df)
    plot_error_rates(df)
    plot_batch_timelines()

    print(f"\nAnalysis complete. Output in {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
