#!/usr/bin/env python3
"""
Decode-step overhead model fitting from guidellm overhead sweep results.

Reads guidellm results collected by run-overhead-sweep.sh and fits:

    overhead(b, ISL) = alpha * b + beta * (b * ISL) + gamma

where:
    overhead = ITL_measured(b, ISL) - TPOT_predicted_by_AIC(b, ISL)
    alpha    = per-sequence scheduling cost (ms per concurrent request)
    beta     = memory management cost proportional to KV cache working set
    gamma    = fixed per-step constant (CUDA sync, Python dispatch)

Usage:
    python3 scripts/analyse-overhead-sweep.py \
        --results-dir results/overhead-sweep-20260525 \
        --model Qwen/Qwen3-8B \
        --system h200_sxm \
        --backend vllm \
        --backend-version 0.18.0

Outputs:
    - Fitted coefficients (alpha, beta, gamma) with 95% CIs
    - Residual plot: overhead vs b for each ISL
    - Heatmap: measured ITL vs AIC TPOT across (b, ISL) grid
    - CSV: raw (b, ISL, ITL_measured, TPOT_AIC, overhead) data
"""

import argparse
import json
import sys
import zstandard as zstd
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_result(path: Path) -> dict | None:
    """Load a guidellm results file (plain JSON or .zst compressed)."""
    try:
        if path.suffix == ".zst":
            dctx = zstd.ZstdDecompressor()
            with open(path, "rb") as f:
                data = json.loads(dctx.decompress(f.read()))
        else:
            with open(path) as f:
                data = json.load(f)
        return data
    except Exception as e:
        print(f"  WARNING: failed to load {path.name}: {e}", file=sys.stderr)
        return None


def extract_metrics(data: dict, filename: str) -> dict | None:
    """Extract ITL, concurrency, ISL, OSL from a guidellm result."""
    try:
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            return None
        b = benchmarks[0]
        m = b.get("metrics", {})

        itl = m.get("inter_token_latency_ms", {}).get("successful", {})
        conc = m.get("request_concurrency", {}).get("successful", {})
        isl_m = m.get("prompt_token_count", {}).get("successful", {})
        osl_m = m.get("output_token_count", {}).get("successful", {})
        n_req = m.get("request_totals", {}).get("successful", 0)

        return {
            "filename": filename,
            "itl_mean": itl.get("mean"),
            "itl_median": itl.get("median"),
            "itl_p90": itl.get("percentiles", {}).get("p90"),
            "itl_std": itl.get("std_dev"),
            "itl_count": itl.get("count", 0),
            "concurrency_mean": conc.get("mean"),
            "concurrency_median": conc.get("median"),
            "concurrency_mode": conc.get("mode"),
            "isl_mean": isl_m.get("mean"),
            "osl_mean": osl_m.get("mean"),
            "n_requests": n_req,
        }
    except Exception as e:
        print(f"  WARNING: failed to extract metrics from {filename}: {e}", file=sys.stderr)
        return None


def get_aic_tpot(model: str, system: str, backend: str, version: str,
                 isl: int, osl: int, concurrency: int) -> float | None:
    """Query AIC for predicted TPOT at given (ISL, OSL, concurrency)."""
    try:
        from aiconfigurator.cli.api import cli_estimate
        result = cli_estimate(
            model_path=model,
            system_name=system,
            backend_name=backend,
            backend_version=version,
            isl=isl,
            osl=osl,
            batch_size=concurrency,
            mode="agg",
        )
        return result.tpot
    except Exception as e:
        print(f"  WARNING: AIC estimate failed (ISL={isl}, b={concurrency}): {e}", file=sys.stderr)
        return None


def fit_overhead_model(df: pd.DataFrame) -> dict:
    """Fit overhead(b, ISL) = alpha*b + beta*(b*ISL) + gamma via OLS."""
    df = df.dropna(subset=["overhead_ms", "b", "isl"])

    X = np.column_stack([
        df["b"].values,            # alpha coefficient
        df["b"].values * df["isl"].values,  # beta coefficient
        np.ones(len(df)),          # gamma (intercept)
    ])
    y = df["overhead_ms"].values

    result = np.linalg.lstsq(X, y, rcond=None)
    coeffs = result[0]

    # Bootstrap 95% CIs
    n_boot = 2000
    rng = np.random.default_rng(42)
    boot_coeffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        try:
            c = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
            boot_coeffs.append(c)
        except Exception:
            pass
    boot_coeffs = np.array(boot_coeffs)
    ci_lo = np.percentile(boot_coeffs, 2.5, axis=0)
    ci_hi = np.percentile(boot_coeffs, 97.5, axis=0)

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": coeffs[0], "alpha_ci": (ci_lo[0], ci_hi[0]),
        "beta": coeffs[1],  "beta_ci":  (ci_lo[1], ci_hi[1]),
        "gamma": coeffs[2], "gamma_ci": (ci_lo[2], ci_hi[2]),
        "r2": r2,
        "n": len(df),
    }


def plot_results(df: pd.DataFrame, fit: dict, output_dir: Path) -> None:
    palette = sns.color_palette("muted")

    # --- 1. Overhead vs b, one line per ISL ---
    fig, ax = plt.subplots(figsize=(14, 8))
    isl_values = sorted(df["isl"].dropna().unique())
    b_range = np.linspace(df["b"].min(), df["b"].max(), 100)

    for i, isl in enumerate(isl_values):
        sub = df[df["isl"] == isl].sort_values("b")
        ax.scatter(sub["b"], sub["overhead_ms"], color=palette[i % len(palette)],
                   s=60, zorder=3, label=f"ISL={int(isl)}")
        predicted = fit["alpha"] * b_range + fit["beta"] * b_range * isl + fit["gamma"]
        ax.plot(b_range, predicted, color=palette[i % len(palette)], alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Concurrency (b)", fontsize=11)
    ax.set_ylabel("Overhead (ms)  [ITL_measured − TPOT_AIC]", fontsize=11)
    ax.set_title("Decode-step overhead vs concurrency\n"
                 f"overhead = {fit['alpha']:.3f}·b + {fit['beta']:.6f}·b·ISL + {fit['gamma']:.3f}  "
                 f"(R²={fit['r2']:.3f})", fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "overhead_vs_concurrency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # --- 2. Heatmap: measured ITL ---
    pivot_itl = df.pivot_table(index="isl", columns="b", values="itl_mean", aggfunc="mean")
    pivot_itl.index = [int(x) for x in pivot_itl.index]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot_itl, ax=ax, cmap="magma", annot=True, fmt=".1f",
                cbar_kws={"label": "ITL mean (ms)"})
    ax.set_title("Measured ITL (ms) — guidellm inter_token_latency_ms", fontsize=11)
    ax.set_xlabel("Concurrency (b)", fontsize=11)
    ax.set_ylabel("ISL (tokens)", fontsize=11)
    fig.tight_layout()
    out = output_dir / "itl_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # --- 3. Heatmap: overhead ---
    pivot_oh = df.pivot_table(index="isl", columns="b", values="overhead_ms", aggfunc="mean")
    pivot_oh.index = [int(x) for x in pivot_oh.index]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot_oh, ax=ax, cmap="magma", annot=True, fmt=".1f",
                cbar_kws={"label": "Overhead (ms)"})
    ax.set_title("Decode-step overhead (ms)  [ITL_measured − TPOT_AIC]", fontsize=11)
    ax.set_xlabel("Concurrency (b)", fontsize=11)
    ax.set_ylabel("ISL (tokens)", fontsize=11)
    fig.tight_layout()
    out = output_dir / "overhead_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing overhead sweep .json.zst files")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model path (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--system", required=True,
                        help="AIC system name (e.g. h200_sxm)")
    parser.add_argument("--backend", default="vllm",
                        help="AIC backend name (default: vllm)")
    parser.add_argument("--backend-version", default=None,
                        help="AIC backend version (default: latest)")
    parser.add_argument("--no-aic", action="store_true",
                        help="Skip AIC predictions (plot raw ITL only)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(results_dir.glob("*.json.zst")) + sorted(results_dir.glob("*.json"))
    if not files:
        print(f"ERROR: no .json or .json.zst files in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(files)} result files from {results_dir}...")
    rows = []
    for path in files:
        data = load_result(path)
        if data is None:
            continue
        m = extract_metrics(data, path.name)
        if m is None:
            continue
        rows.append(m)

    if not rows:
        print("ERROR: no valid results loaded", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df["b"] = df["concurrency_mode"].fillna(df["concurrency_median"])
    df["isl"] = df["isl_mean"].round(-1)  # round to nearest 10 for grouping

    print(f"Loaded {len(df)} runs. ISL range: {df['isl'].min():.0f}–{df['isl'].max():.0f}, "
          f"b range: {df['b'].min():.0f}–{df['b'].max():.0f}")

    # --- get AIC TPOT predictions ---
    if not args.no_aic:
        print("Querying AIC for TPOT predictions...")
        tpot_values = []
        for _, row in df.iterrows():
            tpot = get_aic_tpot(
                model=args.model,
                system=args.system,
                backend=args.backend,
                version=args.backend_version,
                isl=int(row["isl"]),
                osl=int(row["osl_mean"]) if row["osl_mean"] else 128,
                concurrency=int(row["b"]),
            )
            tpot_values.append(tpot)
            status = f"{tpot:.2f}ms" if tpot else "FAILED"
            print(f"  ISL={int(row['isl'])} b={int(row['b'])}: AIC TPOT={status}")
        df["tpot_aic_ms"] = tpot_values
        df["overhead_ms"] = df["itl_mean"] - df["tpot_aic_ms"]
    else:
        df["tpot_aic_ms"] = None
        df["overhead_ms"] = None

    # --- save raw data CSV ---
    csv_path = results_dir / "overhead_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRaw data saved: {csv_path}")

    # --- print summary table ---
    print("\n=== Raw data (ITL_measured vs TPOT_AIC) ===")
    cols = ["isl", "b", "itl_mean", "itl_std", "itl_count", "tpot_aic_ms", "overhead_ms"]
    available = [c for c in cols if c in df.columns]
    print(df[available].sort_values(["isl", "b"]).to_string(index=False, float_format="{:.2f}".format))

    # --- fit model ---
    if not args.no_aic and df["overhead_ms"].notna().sum() >= 4:
        print("\n=== Overhead model fit ===")
        print("Model: overhead(b, ISL) = alpha*b + beta*(b*ISL) + gamma")
        fit = fit_overhead_model(df)
        print(f"  alpha (per-sequence scheduling):  {fit['alpha']:.4f} ms  "
              f"95% CI [{fit['alpha_ci'][0]:.4f}, {fit['alpha_ci'][1]:.4f}]")
        print(f"  beta  (KV cache memory pressure): {fit['beta']:.8f} ms/token  "
              f"95% CI [{fit['beta_ci'][0]:.8f}, {fit['beta_ci'][1]:.8f}]")
        print(f"  gamma (fixed per-step constant):  {fit['gamma']:.4f} ms  "
              f"95% CI [{fit['gamma_ci'][0]:.4f}, {fit['gamma_ci'][1]:.4f}]")
        print(f"  R²: {fit['r2']:.4f}  (n={fit['n']} points)")

        if fit["r2"] < 0.8:
            print("  WARNING: low R² — model may be missing a term or data quality is poor")

        plot_results(df, fit, results_dir)
    else:
        print("\nSkipping model fit (no AIC predictions or insufficient data).")
        if not args.no_aic:
            fit = {"alpha": 0, "beta": 0, "gamma": 0, "r2": 0, "n": 0}
            plot_results(df, fit, results_dir)


if __name__ == "__main__":
    main()
