#!/usr/bin/env python3
"""
Validate the Kingman G/G/1 TTFT queuing model against Poisson-arrival benchmark data.

For each run (a specific Poisson arrival rate λ), computes:
  - Measured TTFT from guidellm results
  - Predicted TTFT from Kingman model: T_prefill × (1 + ρ/(1-ρ) × (ca²+cs²)/2)
    where ρ = λ × T_prefill, T_prefill from AIC silicon model

Outputs a table comparing measured vs predicted, and fits ca² from the data.

Usage:
    PYTHONPATH=/path/to/aiconfigurator/src \
    python3 scripts/analyse-poisson-ttft.py \
        --results-dir results/poisson-ttft-sweep-20260528 \
        --model Qwen/Qwen3-8B \
        --system h200_sxm \
        --backend vllm \
        --backend-version 0.18.0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd


def load_result(path: Path) -> dict | None:
    try:
        if path.suffix == ".zst":
            return json.loads(zstd.ZstdDecompressor().decompress(path.read_bytes()))
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  WARNING: failed to load {path.name}: {e}", file=sys.stderr)
        return None


def extract_metrics(data: dict) -> dict | None:
    """Extract metrics from the benchmark with the most successful requests (main run, not warmup)."""
    try:
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            return None
        # Pick the benchmark with the most successful requests to skip warmup/cooldown phases
        def n_requests(b):
            try:
                return b["metrics"].get("request_totals", {}).get("successful", 0)
            except Exception:
                return 0
        bm = max(benchmarks, key=n_requests)
        m = bm["metrics"]
        s = lambda k: m[k]["successful"]
        return {
            "ttft_mean":  s("time_to_first_token_ms")["mean"],
            "ttft_p50":   s("time_to_first_token_ms")["median"],
            "ttft_p90":   s("time_to_first_token_ms").get("percentiles", {}).get("p90"),
            "itl_mean":   s("inter_token_latency_ms")["mean"],
            "concurrency":s("request_concurrency")["mean"],
            "rps":        s("requests_per_second")["mean"],
            "n_requests": m.get("request_totals", {}).get("successful", 0),
        }
    except Exception as e:
        print(f"  WARNING: metric extraction failed: {e}", file=sys.stderr)
        return None


def get_aic_prefill_time(model, system, backend, version, isl, osl, tp_size) -> float | None:
    """Return AIC's silicon prefill time per request (ms) at b=1, no queuing."""
    try:
        from aiconfigurator.cli.api import cli_estimate
        # At b=1, our (b+1)/2 queuing factor = 1.0, so TTFT ≈ pure prefill time
        r = cli_estimate(
            model_path=model, system_name=system, backend_name=backend,
            backend_version=version, isl=isl, osl=osl, batch_size=1,
            tp_size=tp_size, mode="agg",
        )
        # TTFT at b=1 includes queuing factor of 1.0, so it IS the prefill time
        return r.ttft
    except Exception as e:
        print(f"  WARNING: AIC estimate failed: {e}", file=sys.stderr)
        return None


def kingman_ttft(lam_rps: float, t_prefill_ms: float, ca2: float = 1.0, cs2: float = 0.0) -> float:
    """Kingman G/G/1 TTFT prediction.

    Args:
        lam_rps: arrival rate (req/s)
        t_prefill_ms: per-request prefill service time (ms)
        ca2: squared coefficient of variation of inter-arrival times (1.0 = Poisson)
        cs2: squared coefficient of variation of service times (0.0 = deterministic)

    Returns:
        Predicted mean TTFT (ms)
    """
    rho = lam_rps * t_prefill_ms / 1000.0
    if rho >= 1.0:
        return float("inf")
    wait_ms = (rho / (1.0 - rho)) * (ca2 + cs2) / 2.0 * t_prefill_ms
    return t_prefill_ms + wait_ms


def fit_ca2(lambdas: list[float], ttft_measured: list[float], t_prefill: float) -> float:
    """Fit ca² from measured TTFT data via least-squares on Kingman formula."""
    # TTFT = T + ρ/(1-ρ) × (ca²+0)/2 × T → ca² = 2(TTFT/T - 1) × (1-ρ)/ρ
    ca2_vals = []
    for lam, ttft_m in zip(lambdas, ttft_measured):
        rho = lam * t_prefill / 1000.0
        if rho <= 0 or rho >= 1:
            continue
        ca2_est = 2.0 * (ttft_m / t_prefill - 1.0) * (1.0 - rho) / rho
        if ca2_est > 0:
            ca2_vals.append(ca2_est)
    return float(np.median(ca2_vals)) if ca2_vals else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True, help="Directory with .json.zst result files")
    parser.add_argument("--file-pattern", default=None,
                        help="Glob prefix to filter result files (e.g. 'qwen3-8b-poisson')")
    parser.add_argument("--model",           default="Qwen/Qwen3-8B")
    parser.add_argument("--system",          default="h200_sxm")
    parser.add_argument("--backend",         default="vllm")
    parser.add_argument("--backend-version", default="0.18.0")
    parser.add_argument("--isl",   type=int, default=9000)
    parser.add_argument("--osl",   type=int, default=30)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--aic-src", help="Path to aiconfigurator/src to add to sys.path")
    args = parser.parse_args()

    if args.aic_src:
        sys.path.insert(0, args.aic_src)

    results_dir = Path(args.results_dir)
    pattern = f"{args.file_pattern}*.json.zst" if args.file_pattern else "*.json.zst"
    files = sorted(results_dir.glob(pattern))
    if not args.file_pattern:
        files += sorted(results_dir.glob("*.json"))
    if not files:
        print(f"ERROR: no result files found in {results_dir}")
        sys.exit(1)

    # Load all results and extract the configured rate from filename
    rows = []
    for path in files:
        # Parse rate from filename: *-rateN.json.zst or *-rateNpM.json.zst
        stem = path.name.replace(".json.zst", "").replace(".json", "")
        parts = stem.split("-rate")
        if len(parts) < 2:
            continue
        rate_str = parts[-1].replace("p", ".")
        try:
            lam = float(rate_str)
        except ValueError:
            continue

        data = load_result(path)
        if data is None:
            continue
        m = extract_metrics(data)
        if m is None:
            continue
        m["lambda"] = lam
        m["filename"] = path.name
        rows.append(m)

    rows.sort(key=lambda r: r["lambda"])

    if not rows:
        print("ERROR: no valid result files parsed")
        sys.exit(1)

    # Get AIC prefill time
    print(f"\nQuerying AIC for silicon prefill time at b=1 "
          f"({args.model}, {args.system}, {args.backend} {args.backend_version})...")
    t_prefill = get_aic_prefill_time(
        args.model, args.system, args.backend, args.backend_version,
        args.isl, args.osl, args.tp_size,
    )
    if t_prefill is None:
        print("ERROR: could not get AIC prefill time. Check PYTHONPATH and AIC installation.")
        sys.exit(1)
    print(f"  T_prefill (AIC b=1 TTFT) = {t_prefill:.1f} ms")

    # Fit ca² from data
    lambdas   = [r["lambda"] for r in rows]
    ttft_meas = [r["ttft_mean"] for r in rows]
    ca2_fit = fit_ca2(lambdas, ttft_meas, t_prefill)
    print(f"  Fitted ca² = {ca2_fit:.3f}  (1.0 = pure Poisson; <1 = more regular)")

    # Table
    print(f"\n  {'λ(req/s)':>9}  {'ρ':>5}  {'TTFT_meas':>10}  {'TTFT_p50':>9}  "
          f"{'TTFT_p90':>9}  {'King_ca1':>9}  {'King_fit':>9}  "
          f"{'err_ca1':>8}  {'err_fit':>8}  {'n_req':>6}  {'conc':>5}")
    print("  " + "-"*110)

    for r in rows:
        lam = r["lambda"]
        rho = lam * t_prefill / 1000.0
        ttft_k1  = kingman_ttft(lam, t_prefill, ca2=1.0)
        ttft_kfit= kingman_ttft(lam, t_prefill, ca2=ca2_fit)
        err_k1   = (ttft_k1   - r["ttft_mean"]) / r["ttft_mean"] * 100 if r["ttft_mean"] else float("nan")
        err_kfit = (ttft_kfit - r["ttft_mean"]) / r["ttft_mean"] * 100 if r["ttft_mean"] else float("nan")
        rho_str  = f"{rho:.3f}" if rho < 1.0 else "≥1.00"
        k1_str   = f"{ttft_k1:.1f}" if ttft_k1 != float("inf") else "∞"
        kfit_str = f"{ttft_kfit:.1f}" if ttft_kfit != float("inf") else "∞"

        print(f"  {lam:>9.1f}  {rho_str:>5}  {r['ttft_mean']:>10.1f}  "
              f"{r['ttft_p50']:>9.1f}  "
              f"{r['ttft_p90'] or 0:>9.1f}  "
              f"{k1_str:>9}  {kfit_str:>9}  "
              f"{err_k1:>+7.1f}%  {err_kfit:>+7.1f}%  "
              f"{r['n_requests']:>6}  {r['concurrency']:>5.1f}")

    print(f"\n  King_ca1 = Kingman with ca²=1 (pure Poisson)")
    print(f"  King_fit = Kingman with fitted ca²={ca2_fit:.3f}")
    print(f"  ρ = λ × T_prefill / 1000 (utilization of prefill server)")
    print(f"\n  T_prefill = {t_prefill:.1f} ms  (AIC silicon prediction at b=1)")
    print(f"  Fitted ca² = {ca2_fit:.3f}")
    if abs(ca2_fit - 1.0) < 0.15:
        print(f"  → Arrival process is approximately Poisson (ca²≈1) ✓")
    elif ca2_fit < 1.0:
        print(f"  → Arrivals are more regular than Poisson (ca²<1) — sub-Poisson")
    else:
        print(f"  → Arrivals are more bursty than Poisson (ca²>1) — super-Poisson")


if __name__ == "__main__":
    main()
