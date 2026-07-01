#!/usr/bin/env python3
"""
Validate the fix/throughput-queueing-model branch predictions against measured data.

Two checks:
  1. TTFT accuracy at all batch sizes (b=1..64) for Qwen3-8B ISL=9000 — does
     TTFT error corrupt the Little's Law throughput cap?
  2. Cross-model validation on Qwen3-32B-FP8 ISL=9000.

Context: the branch was calibrated without PRs #1147 and #1151. Points where
b >= osl (b >= 30 here) are affected by the absent #1147 fix (num_mix_gen_tokens).
Points where the -3 num_mix_steps_for_tpot_calc correction still applies affect
TPOT (and hence request_latency) at all b — that's the absent #1151 fix.

Run from the fix/throughput-queueing-model branch with AIC installed:
    cd /path/to/aiconfigurator
    git checkout fix/throughput-queueing-model
    pip install -e . -q
    python3 /path/to/this/script

Or pass --aic-src to add a source tree to sys.path:
    python3 validate-throughput-branch.py --aic-src /path/to/aiconfigurator/src
"""

import argparse
import json
import sys
from pathlib import Path

import zstandard as zstd

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = [
    {
        "label": "Qwen3-8B",
        "model": "Qwen/Qwen3-8B",
        "results_dir": RESULTS_DIR / "overhead-sweep-isl9000",
        "pattern": "qwen3-8b-overhead-isl9000-rate*.json.zst",
        "system": "h200_sxm",
        "backend": "vllm",
        "version": "0.18.0",
        "isl": 9000,
        "osl": 30,
        "tp_size": 1,
    },
    {
        # AIC predictions use BF16 (Qwen/Qwen3-32B); measured data is FP8.
        # FP8 runs ~30-40% faster than BF16 for the same architecture, so TPOT
        # predictions will be over-estimated — but the overhead scaling trend
        # (does correction(b) grow correctly with b?) is still checkable.
        # Full FP8 cross-validation requires PR #1142 silicon data (FP8 context
        # attention rows) to be merged first.
        "label": "Qwen3-32B (BF16 AIC vs FP8 measured — quant mismatch noted)",
        "model": "Qwen/Qwen3-32B",
        "results_dir": RESULTS_DIR / "Qwen3-32B-FP8-9k-30-rate-sweep",
        "pattern": "guidellm-qwen3-32b-fp8-9k30-rate*.json.zst",
        "system": "h200_sxm",
        "backend": "vllm",
        "version": "0.18.0",
        "isl": 9000,
        "osl": 30,
        "tp_size": 4,
    },
]


def load_result(path: Path) -> dict | None:
    try:
        if path.suffix == ".zst":
            dctx = zstd.ZstdDecompressor()
            with open(path, "rb") as f:
                return json.loads(dctx.decompress(f.read()))
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: failed to load {path.name}: {e}", file=sys.stderr)
        return None


def extract_metrics(data: dict) -> dict | None:
    try:
        bm = data["benchmarks"][0]
        m = bm["metrics"]

        def s(key):
            return m.get(key, {}).get("successful", {})

        return {
            "concurrency": s("request_concurrency").get("mean"),
            "ttft_ms": s("time_to_first_token_ms").get("mean"),
            "itl_ms": s("inter_token_latency_ms").get("mean"),
            "req_s": s("requests_per_second").get("mean"),
            "tokens_s": s("output_tokens_per_second").get("mean"),
            "isl": s("prompt_token_count").get("mean"),
            "osl": s("output_token_count").get("mean"),
        }
    except Exception as e:
        print(f"  WARNING: metric extraction failed: {e}", file=sys.stderr)
        return None


def run_aic(model_cfg: dict, concurrency: int) -> dict | None:
    try:
        from aiconfigurator.cli.api import cli_estimate
        result = cli_estimate(
            model_path=model_cfg["model"],
            system_name=model_cfg["system"],
            backend_name=model_cfg["backend"],
            backend_version=model_cfg["version"],
            isl=model_cfg["isl"],
            osl=model_cfg["osl"],
            batch_size=concurrency,
            tp_size=model_cfg["tp_size"],
            mode="agg",
        )
        return {
            "ttft_ms": result.ttft,
            "tpot_ms": result.tpot,
            "req_s": result.raw.get("request_rate", 0.0),
            "tokens_s": result.tokens_per_second,
            "request_latency_ms": result.request_latency,
        }
    except Exception as e:
        print(f"  WARNING: AIC estimate failed (b={concurrency}): {e}", file=sys.stderr)
        return None


def analyse_model(model_cfg: dict) -> None:
    label = model_cfg["label"]
    osl = model_cfg["osl"]
    results_dir = model_cfg["results_dir"]

    print(f"\n{'='*72}")
    print(f"  {label}  ISL={model_cfg['isl']} OSL={osl}  "
          f"system={model_cfg['system']}  vLLM {model_cfg['version']}")
    print(f"{'='*72}")

    # --- Load all result files ---
    files = sorted(results_dir.glob(model_cfg["pattern"]))
    if not files:
        print(f"  ERROR: no files found in {results_dir} matching {model_cfg['pattern']}")
        return

    rows = []
    for path in files:
        data = load_result(path)
        if data is None:
            continue
        m = extract_metrics(data)
        if m is None or m["concurrency"] is None:
            continue
        m["filename"] = path.name
        rows.append(m)

    rows.sort(key=lambda r: r["concurrency"])

    # --- Run AIC at each measured concurrency ---
    print(f"\n  {'b':>4}  {'TTFT meas':>10}  {'TTFT AIC':>9}  {'TTFT err':>9}  "
          f"{'ITL meas':>9}  {'TPOT AIC':>9}  {'TPOT err':>9}  {'overhead':>9}  "
          f"{'req/s meas':>10}  {'req/s AIC':>9}  {'tput err':>9}  "
          f"{'b>=osl?':>7}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*9}  {'-'*9}  "
          f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  "
          f"{'-'*10}  {'-'*9}  {'-'*9}  "
          f"{'-'*7}")

    overhead_points = []  # (b, overhead_ratio) for curve fitting

    for r in rows:
        b = round(r["concurrency"])
        aic = run_aic(model_cfg, b)
        if aic is None:
            print(f"  {b:>4}  {'(AIC failed)':>10}")
            continue

        ttft_err = (aic["ttft_ms"] - r["ttft_ms"]) / r["ttft_ms"] * 100 if r["ttft_ms"] else float("nan")
        tpot_err = (aic["tpot_ms"] - r["itl_ms"]) / r["itl_ms"] * 100 if r["itl_ms"] else float("nan")
        req_s_err = (aic["req_s"] - r["req_s"]) / r["req_s"] * 100 if r["req_s"] else float("nan")
        # overhead ratio = measured ITL / silicon TPOT (correction disabled)
        overhead = r["itl_ms"] / aic["tpot_ms"] if aic["tpot_ms"] else float("nan")
        if not (float("nan") == overhead) and b >= 1:
            overhead_points.append((b, overhead))

        b_ge_osl = "YES *" if b >= osl else "no"

        rl_req_s = b * 1000.0 / aic["request_latency_ms"] if aic["request_latency_ms"] > 0 else float("inf")
        cap_note = "LL-cap" if aic["req_s"] >= rl_req_s * 0.999 else "step-lat"

        print(f"  {b:>4}  {r['ttft_ms']:>10.1f}  {aic['ttft_ms']:>9.1f}  {ttft_err:>+8.1f}%  "
              f"{r['itl_ms']:>9.1f}  {aic['tpot_ms']:>9.1f}  {tpot_err:>+8.1f}%  {overhead:>8.3f}x  "
              f"{r['req_s']:>10.2f}  {aic['req_s']:>9.2f}  {req_s_err:>+8.1f}%  "
              f"{b_ge_osl:>7}")

    print()
    print("  * b >= osl: num_mix_gen_tokens fix (PR #1147) applies at these points.")
    print("  overhead = measured ITL / silicon TPOT (correction disabled — baseline run).")

    # --- Fit correction(b) = 1 + (b-1)^exp * scale to overhead ratios ---
    if len(overhead_points) >= 3:
        import numpy as np
        from scipy.optimize import curve_fit

        bs = np.array([p[0] for p in overhead_points], dtype=float)
        oh = np.array([p[1] for p in overhead_points], dtype=float)

        def model(b, scale, exp):
            return 1.0 + np.where(b > 1, (b - 1) ** exp * scale, 0.0)

        try:
            popt, _ = curve_fit(model, bs, oh, p0=[1.0, 0.5], bounds=([0, 0.1], [20, 2.0]))
            scale_fit, exp_fit = popt
            residuals = oh - model(bs, *popt)
            mae = float(np.mean(np.abs(residuals / oh)) * 100)
            print(f"\n  Fitted overhead model: correction(b) = 1 + (b-1)^{exp_fit:.3f} * {scale_fit:.3f}")
            print(f"  Mean absolute error: {mae:.1f}%")
            print(f"\n  Suggested vllm_backend.py values:")
            print(f"    return 1.0 + ((b - 1) ** {exp_fit:.3f}) * {scale_fit:.3f}")
        except Exception as e:
            print(f"\n  Curve fit failed: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aic-src", metavar="PATH",
                        help="Path to aiconfigurator/src to add to sys.path")
    parser.add_argument("--model", choices=["8B", "32B-FP8", "both"], default="both",
                        help="Which model to validate (default: both)")
    args = parser.parse_args()

    if args.aic_src:
        sys.path.insert(0, args.aic_src)

    models = MODELS
    if args.model == "8B":
        models = [MODELS[0]]
    elif args.model == "32B-FP8":
        models = [MODELS[1]]

    for model_cfg in models:
        analyse_model(model_cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
