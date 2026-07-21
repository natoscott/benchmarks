#!/usr/bin/env python3
"""Extract key metrics from PP validation guidellm v0.7 results and compare against AIC predictions."""

import json
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "pp-validation"
PREDICTIONS_FILE = RESULTS_DIR / "aic-predictions.json"


def load_json_zst(path: Path) -> dict:
    data = subprocess.run(["zstd", "-d", "-c", str(path)], capture_output=True, check=True).stdout
    return json.loads(data)


def extract_metrics(data: dict) -> dict | None:
    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        return None
    b = benchmarks[0]

    successful = b.get("requests", {}).get("successful", [])
    if not successful:
        return None

    m = b["metrics"]
    s_otps = m["output_tokens_per_second"]["successful"]
    s_ttft = m["time_to_first_token_ms"]["successful"]
    s_itl = m["inter_token_latency_ms"]["successful"]
    s_rps = m["requests_per_second"]["successful"]

    streams = b["config"]["profile"].get("streams", [0])
    concurrency = streams[0] if isinstance(streams, list) else streams

    return {
        "concurrency": concurrency,
        "successful": len(successful),
        "output_tok_per_sec": s_otps["mean"],
        "req_per_sec": s_rps["mean"],
        "ttft_p50_ms": s_ttft["median"],
        "itl_p50_ms": s_itl["median"],
    }


def main():
    predictions = json.loads(PREDICTIONS_FILE.read_text()) if PREDICTIONS_FILE.exists() else {}

    configs = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        config_name = d.name

        results = []
        for f in sorted(d.glob("*.json.zst")):
            try:
                data = load_json_zst(f)
                m = extract_metrics(data)
                if m:
                    results.append(m)
            except Exception as e:
                print(f"WARN: {f.name}: {e}", file=sys.stderr)

        if not results:
            continue

        # Pick peak throughput (highest output_tok_per_sec)
        peak = max(results, key=lambda r: r["output_tok_per_sec"])

        pred = predictions.get(config_name, {})
        pred_tps = pred.get("tokens_per_s_per_gpu", 0)
        pred_ttft = pred.get("ttft_ms", 0)

        # Per-GPU throughput (8 GPUs per config)
        actual_tps_per_gpu = peak["output_tok_per_sec"] / 8

        error_pct = round((actual_tps_per_gpu - pred_tps) / pred_tps * 100, 1) if pred_tps else None

        configs.append({
            "config": config_name,
            "actual_tok_s_gpu": round(actual_tps_per_gpu, 1),
            "predicted_tok_s_gpu": pred_tps,
            "error_pct": error_pct,
            "actual_ttft_p50": round(peak["ttft_p50_ms"], 1),
            "predicted_ttft": pred_ttft,
            "peak_concurrency": peak["concurrency"],
            "peak_output_tok_s": round(peak["output_tok_per_sec"], 1),
            "all_results": results,
        })

    # Print summary table
    print(f"{'Config':<25} {'Actual':>10} {'Predicted':>10} {'Error':>8} {'TTFT_act':>10} {'TTFT_pred':>10} {'Peak_conc':>10}")
    print("-" * 95)

    current_model = ""
    for c in configs:
        model = c["config"].rsplit("-tp", 1)[0]
        if model != current_model:
            if current_model:
                print()
            current_model = model

        err = f"{c['error_pct']:+.1f}%" if c["error_pct"] is not None else "N/A"
        print(
            f"{c['config']:<25} {c['actual_tok_s_gpu']:>10.1f} {c['predicted_tok_s_gpu']:>10.1f} {err:>8} "
            f"{c['actual_ttft_p50']:>10.1f} {c['predicted_ttft']:>10.1f} {c['peak_concurrency']:>10}"
        )

    # Save full results
    output = RESULTS_DIR / "pp-validation-summary.json"
    with open(output, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"\nFull results saved to: {output}")


if __name__ == "__main__":
    main()
