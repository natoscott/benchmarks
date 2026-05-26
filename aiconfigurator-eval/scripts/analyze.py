#!/usr/bin/env python3
"""Generate figures for the aiconfigurator evaluation report."""

import json
import zstandard as zstd
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RESULTS = pathlib.Path("results")
FIGURES = pathlib.Path("figures")
FIGURES.mkdir(exist_ok=True)

sns.set_style("whitegrid")
sns.set_palette("muted")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11

PALETTE = sns.color_palette("muted")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    if p.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(p.read_bytes(), max_output_size=200 * 1024 * 1024)
        data = json.loads(raw)
    else:
        data = json.loads(p.read_text())
    bm = data["benchmarks"][0]
    m = bm["metrics"]
    def mean(key):
        v = m.get(key, {})
        if isinstance(v, dict):
            s = v.get("successful", v)
            if isinstance(s, dict):
                return s.get("mean")
        return None
    return {
        "rps":  mean("requests_per_second"),
        "ttft": mean("time_to_first_token_ms"),
        "tpot": mean("time_per_output_token_ms"),   # total_latency/output_tokens, includes TTFT
        "itl":  mean("inter_token_latency_ms"),     # decode interval only; matches AIC TPOT model
        "n":    m.get("request_totals", {}).get("successful"),
    }


def load_sweep(directory, prefix, concurrencies):
    rows = []
    for c in concurrencies:
        path = RESULTS / directory / f"{prefix}-rate{c}.json.zst"
        m = load_metrics(path)
        if m and m["rps"] is not None:
            rows.append({"conc": c, **m})
    return rows


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "label":    "Qwen3-8B agg (TP=1×8)",
        "short":    "8B agg",
        "dir":      "Qwen3-8B-9k-30-rate-sweep",
        "prefix":   "guidellm-qwen3-8b-9k30",
        "concs":    [1, 2, 4, 8, 16, 24, 32, 40],
        "aic_rps":  30.4,  # vLLM 0.18.0 silicon data
        "aic_ttft": 484,
        "aic_tpot": 23.2,
        "color":    PALETTE[0],
        "ls":       "-",
        "marker":   "o",
    },
    {
        "label":    "Qwen3-8B disagg (7P+1D)",
        "short":    "8B disagg",
        "dir":      "Qwen3-8B-9k-30-disagg-rate-sweep",
        "prefix":   "guidellm-qwen3-8b-disagg-9k30",
        "concs":    [1, 2, 4, 8, 16],
        "aic_rps":  25.0,  # vLLM 0.18.0 silicon data
        "aic_ttft": 453,
        "aic_tpot": 7.7,
        "color":    PALETTE[3],
        "ls":       "-",
        "marker":   "^",
    },
    {
        "label":    "Qwen3-32B-FP8 agg (TP=1×8)",
        "short":    "32B-FP8 agg TP=1",
        "dir":      "Qwen3-32B-FP8-9k-30-rate-sweep",
        "prefix":   "guidellm-qwen3-32b-fp8-9k30",
        "concs":    [1, 2, 4, 8, 16],
        "aic_rps":  None,  # not in SLA-compliant pareto with 0.18.0 data
        "aic_ttft": None,
        "aic_tpot": None,
        "color":    PALETTE[1],
        "ls":       "-",
        "marker":   "o",
    },
    {
        "label":    "Qwen3-32B-FP8 agg (TP=4×2) — AIC top-1",
        "short":    "32B-FP8 agg TP=4",
        "dir":      "Qwen3-32B-FP8-9k-30-tp4-rate-sweep",
        "prefix":   "guidellm-qwen3-32b-fp8-tp4-9k30",
        "concs":    [1, 2, 4, 6, 8, 12, 16],
        "aic_rps":  5.4,   # vLLM 0.18.0 silicon data, AIC top-1
        "aic_ttft": 489,
        "aic_tpot": 28.8,
        "color":    PALETTE[2],
        "ls":       "-",
        "marker":   "s",
    },
    {
        "label":    "Qwen3-32B-FP8 disagg (7P+1D)",
        "short":    "32B-FP8 disagg",
        "dir":      "Qwen3-32B-FP8-9k-30-disagg-rate-sweep",
        "prefix":   "guidellm-qwen3-32b-fp8-disagg-9k30",
        "concs":    [1, 2, 4, 8, 16],
        "aic_rps":  3.4,   # vLLM 0.18.0 silicon data
        "aic_ttft": 470,
        "aic_tpot": 23.8,
        "color":    PALETTE[4],
        "ls":       "-",
        "marker":   "^",
    },
]

# Load all data
for cfg in CONFIGS:
    cfg["data"] = load_sweep(cfg["dir"], cfg["prefix"], cfg["concs"])

# ---------------------------------------------------------------------------
# Fig 1 — Throughput vs concurrency
# ---------------------------------------------------------------------------

fig, ax = plt.subplots()

for cfg in CONFIGS:
    xs = [r["conc"] for r in cfg["data"]]
    ys = [r["rps"]  for r in cfg["data"]]
    ax.plot(xs, ys, marker=cfg["marker"], color=cfg["color"], ls=cfg["ls"],
            linewidth=2, label=cfg["label"])
    # AIC predicted peak as horizontal dotted line (same colour, lighter)
    if cfg["aic_rps"] is not None:
        ax.axhline(cfg["aic_rps"], color=cfg["color"], ls=":", linewidth=1.2, alpha=0.6)

ax.axvline(16, color="grey", ls=":", linewidth=1, alpha=0.5,
           label="Original sweep ceiling (conc=16)")
ax.set_xlabel("Max concurrent requests")
ax.set_ylabel("Observed throughput (req/s)")
ax.set_title("Throughput vs Concurrency\n(dotted horizontals = AIC predicted peak)")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(FIGURES / "fig1-throughput.png", dpi=150)
plt.close()
print("fig1-throughput.png")

# ---------------------------------------------------------------------------
# Fig 2 — TTFT vs throughput (latency-throughput curve)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots()

for cfg in CONFIGS:
    xs = [r["rps"]  for r in cfg["data"]]
    ys = [r["ttft"] for r in cfg["data"]]
    ax.plot(xs, ys, marker=cfg["marker"], color=cfg["color"], ls=cfg["ls"],
            linewidth=2, label=cfg["label"])

ax.axhline(500, color="red", ls="--", linewidth=1.5, label="TTFT SLA = 500 ms")
ax.set_xlabel("Observed throughput (req/s)")
ax.set_ylabel("TTFT mean (ms)")
ax.set_title("TTFT vs Throughput")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "fig2-latency.png", dpi=150)
plt.close()
print("fig2-latency.png")

# ---------------------------------------------------------------------------
# Fig 3 — TTFT vs concurrency
# ---------------------------------------------------------------------------

fig, ax = plt.subplots()

for cfg in CONFIGS:
    xs = [r["conc"] for r in cfg["data"]]
    ys = [r["ttft"] for r in cfg["data"]]
    ax.plot(xs, ys, marker=cfg["marker"], color=cfg["color"], ls=cfg["ls"],
            linewidth=2, label=cfg["label"])
    if cfg["aic_ttft"] is not None:
        ax.axhline(cfg["aic_ttft"], color=cfg["color"], ls=":", linewidth=1.2, alpha=0.6)

ax.axhline(500, color="red", ls="--", linewidth=1.5, label="TTFT SLA = 500 ms")
ax.set_xlabel("Max concurrent requests")
ax.set_ylabel("TTFT mean (ms)")
ax.set_title("TTFT vs Concurrency\n(dotted horizontals = AIC predicted TTFT)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "fig3-ttft.png", dpi=150)
plt.close()
print("fig3-ttft.png")

# ---------------------------------------------------------------------------
# Fig 4 — ITL (inter-token latency) vs concurrency
# ITL = decode interval only, matching AIC's TPOT model.
# guidellm time_per_output_token_ms = total_latency/output_tokens (includes TTFT)
# and is NOT plotted here.
# ---------------------------------------------------------------------------

fig, ax = plt.subplots()

for cfg in CONFIGS:
    xs = [r["conc"] for r in cfg["data"]]
    ys = [r["itl"]  for r in cfg["data"] if r.get("itl") is not None]
    xs = [r["conc"] for r in cfg["data"] if r.get("itl") is not None]
    if not ys:
        continue
    ax.plot(xs, ys, marker=cfg["marker"], color=cfg["color"], ls=cfg["ls"],
            linewidth=2, label=cfg["label"])
    if cfg["aic_tpot"] is not None:
        ax.axhline(cfg["aic_tpot"], color=cfg["color"], ls=":", linewidth=1.2, alpha=0.6)

ax.axhline(30, color="red", ls="--", linewidth=1.5, label="TPOT SLA = 30 ms/tok")
ax.set_xlabel("Max concurrent requests")
ax.set_ylabel("Inter-token latency / ITL mean (ms/token)")
ax.set_title("ITL vs Concurrency\n(dotted horizontals = AIC predicted TPOT; dashed red = SLA)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "fig4-tpot.png", dpi=150)
plt.close()
print("fig4-tpot.png")

# ---------------------------------------------------------------------------
# Fig 5 — AIC predicted vs observed peak throughput
# ---------------------------------------------------------------------------

# Use the max SLA-compliant operating point for "observed" where applicable.
# For configs where TTFT always exceeds SLA, use the peak observed throughput.
#
# SLA-compliant peaks (TTFT <= 500ms):
#   8B agg:        conc=16 → 13.88 req/s (TTFT=487ms)
#   8B disagg:     conc=2  →  2.97 req/s (TTFT=381ms); conc=4 borderline (525ms)
#   32B-FP8 agg TP=1: TTFT >=737ms at all levels → no SLA-compliant point → peak=5.14
#   32B-FP8 agg TP=4: TTFT >=729ms at all levels → no SLA-compliant point → peak=2.23
#   32B-FP8 disagg: TTFT >=731ms → no SLA-compliant point → peak=1.27

fig5_data = [
    {"label": "8B agg",           "aic": 30.4, "obs": 13.88, "note": "conc=16, TTFT=487ms"},
    {"label": "8B disagg",        "aic": 25.0, "obs":  2.97, "note": "conc=2, TTFT=381ms"},
    {"label": "32B-FP8\nagg TP=4","aic":  5.4, "obs":  2.23, "note": "conc=16, TTFT>SLA"},
    {"label": "32B-FP8\ndisagg",  "aic":  3.4, "obs":  1.27, "note": "conc=8,  TTFT>SLA"},
]

labels  = [d["label"] for d in fig5_data]
aic_vals = [d["aic"]  for d in fig5_data]
obs_vals = [d["obs"]  for d in fig5_data]

x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots()
bars_aic = ax.bar(x - w/2, aic_vals, w, label="AIC predicted", color=PALETTE[4], alpha=0.85)
bars_obs = ax.bar(x + w/2, obs_vals, w, label="Observed peak",  color=PALETTE[0], alpha=0.85)

# Annotate with ratios
for i, d in enumerate(fig5_data):
    ratio = d["obs"] / d["aic"]
    ax.text(x[i] + w/2, d["obs"] + 0.2, f"{ratio:.2f}×",
            ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Throughput (req/s)")
ax.set_title("AIC Predicted vs Observed Peak Throughput\n(ratio = observed / predicted; 8B agg and disagg are SLA-constrained peaks)")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "fig5-aic-vs-observed.png", dpi=150)
plt.close()
print("fig5-aic-vs-observed.png")

print("All figures written to", FIGURES)
