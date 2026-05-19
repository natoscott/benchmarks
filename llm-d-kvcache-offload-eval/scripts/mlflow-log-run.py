#!/usr/bin/env python3
"""Log a single benchmark result directory to MLflow.

Usage:
    python3 scripts/mlflow-log-run.py <OUTPUT_DIR>

Configuration (in order of precedence):
    1. Environment variables
    2. mlflow.conf in the repo root (gitignored; copy from mlflow.conf.example)

Keys (env var names / mlflow.conf keys):
    MLFLOW_TRACKING_URI          MLflow server URL (required)
    MLFLOW_TRACKING_USERNAME     MLflow username (required)
    MLFLOW_TRACKING_PASSWORD     MLflow password (required)
    MLFLOW_TRACKING_INSECURE_TLS set to "true" for self-signed certs (default: true)

MLflow structure:
    Experiment: {HARDWARE}/{SOFTWARE}   e.g. "1x2xL40S/upstream-llm-d-0.7.0"
    Run name:   {MODEL_NAME}_{PARAMETERS}_r{replicas}_rate{rate}
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict

# Suppress InsecureRequestWarning — self-signed cert is expected on this server
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

_DEFAULT_CONF_FILE = REPO_ROOT / "mlflow.conf"

# MLflow log_batch API limits (server-enforced)
MLFLOW_PARAM_BATCH_SIZE  = 100   # max params per log_batch call
MLFLOW_METRIC_BATCH_SIZE = 1000  # max metrics per log_batch call


def _load_conf() -> dict:
    """Parse mlflow.conf (KEY=VALUE, # comments) into a dict.

    Conf file location: MLFLOW_CONF env var, or mlflow.conf in the repo root.
    """
    conf_path = Path(os.environ.get("MLFLOW_CONF", "") or _DEFAULT_CONF_FILE)
    if not conf_path.exists():
        return {}
    conf = {}
    for line in conf_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            conf[key.strip()] = val.strip()
    return conf

# Guidellm: (mlflow_prefix, metric_category, subcategory)
# We always pull from the "successful" subcategory.
GUIDELLM_STAT_METRICS = [
    ("rps",          "requests_per_second",        "successful"),
    ("latency_s",    "request_latency",             "successful"),
    ("ttft_ms",      "time_to_first_token_ms",      "successful"),
    ("tpot_ms",      "time_per_output_token_ms",    "successful"),
    ("itl_ms",       "inter_token_latency_ms",      "successful"),
    ("output_tps",   "output_tokens_per_second",    "successful"),
    ("total_tps",    "tokens_per_second",           "successful"),
    ("prompt_tps",   "prompt_tokens_per_second",    "successful"),
    ("concurrency",  "request_concurrency",         "successful"),
    ("prompt_tok",   "prompt_token_count",          "successful"),
    ("output_tok",   "output_token_count",          "successful"),
]

GUIDELLM_STAT_FIELDS = ["mean", "median", "std_dev", "min", "max",
                         "p50", "p75", "p90", "p95", "p99"]

# Metric names to skip — metadata / labels / histograms we derive separately
_PCP_SKIP_SUFFIXES = ("_bucket", "_created", "_info", "cache_config_info",
                      "python_info", "process_start_time_seconds")

# How many PCP metrics to fetch per pmrep call (avoid ARG_MAX limits)
_PCP_PMREP_BATCH = 150


# ---------------------------------------------------------------------------
# MLflow connection setup
# ---------------------------------------------------------------------------

def setup_mlflow():
    """Configure the MLflow client and return the mlflow module."""
    import mlflow as _mlflow

    conf = _load_conf()
    def _get(key):
        return os.environ.get(key) or conf.get(key, "")

    uri       = _get("MLFLOW_TRACKING_URI")
    username  = _get("MLFLOW_TRACKING_USERNAME")
    password  = _get("MLFLOW_TRACKING_PASSWORD")
    workspace = _get("MLFLOW_WORKSPACE")
    insecure  = (_get("MLFLOW_TRACKING_INSECURE_TLS") or "true").lower() == "true"

    missing = [k for k, v in [("MLFLOW_TRACKING_URI", uri),
                               ("MLFLOW_TRACKING_USERNAME", username),
                               ("MLFLOW_TRACKING_PASSWORD", password)] if not v]
    if missing:
        print(f"Error: missing config: {', '.join(missing)}")
        print(f"  Set via environment variables or mlflow.conf (MLFLOW_CONF env var)")
        print(f"  See mlflow.conf.example for format")
        sys.exit(1)

    os.environ["MLFLOW_TRACKING_USERNAME"]  = username
    os.environ["MLFLOW_TRACKING_PASSWORD"]  = password
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = str(insecure).lower()

    _mlflow.set_tracking_uri(uri)
    if workspace:
        _mlflow.set_workspace(workspace)

    return _mlflow



# ---------------------------------------------------------------------------
# Guidellm metrics extraction
# ---------------------------------------------------------------------------

def extract_guidellm_metrics(output_dir: Path):
    """Return (params dict, metrics dict) from guidellm-results.json.zst."""
    result_file = output_dir / "guidellm-results.json.zst"
    if not result_file.exists():
        print(f"  Warning: {result_file} not found, skipping guidellm metrics")
        return {}, {}

    proc = subprocess.run(
        ["zstd", "-d", "-c", str(result_file)],
        capture_output=True, timeout=60
    )
    if proc.returncode != 0:
        print(f"  Warning: failed to decompress {result_file}")
        return {}, {}

    data = json.loads(proc.stdout)
    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        print("  Warning: no benchmarks in guidellm JSON")
        return {}, {}

    bench = benchmarks[0]
    cfg = bench.get("config", {})
    sched = bench.get("scheduler_metrics", {})
    metric_data = bench.get("metrics", {})

    # --- params from guidellm config ---
    params = {}
    strategy = cfg.get("strategy", {})
    params["guidellm.rate_type"] = strategy.get("type_", "")
    params["guidellm.worker_count"] = strategy.get("worker_count", "")
    constraints = cfg.get("constraints", {})
    max_dur = constraints.get("max_seconds", {})
    params["guidellm.max_seconds"] = max_dur.get("max_duration", "")
    backend = cfg.get("backend", {})
    params["guidellm.target"] = backend.get("target", "")
    params["guidellm.timeout"] = backend.get("timeout", "")

    req_cfg = cfg.get("requests", {})
    params["guidellm.data"] = req_cfg.get("data", "")
    params["guidellm.random_seed"] = req_cfg.get("random_seed", "")

    # --- request totals ---
    metrics = {}
    totals = metric_data.get("request_totals", {})
    metrics["requests.successful"] = float(totals.get("successful", 0))
    metrics["requests.errored"] = float(totals.get("errored", 0))
    metrics["requests.incomplete"] = float(totals.get("incomplete", 0))
    metrics["requests.total"] = float(totals.get("total", 0))

    # --- scheduler metrics ---
    for k, v in sched.items():
        if isinstance(v, (int, float)) and k not in ("start_time", "end_time",
                                                       "request_start_time",
                                                       "measure_start_time",
                                                       "measure_end_time",
                                                       "request_end_time"):
            metrics[f"sched.{k}"] = float(v)
        elif k == "requests_made" and isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (int, float)):
                    metrics[f"sched.requests_made.{sk}"] = float(sv)

    # --- stat metrics (mean, percentiles, etc.) ---
    for prefix, category, sub in GUIDELLM_STAT_METRICS:
        cat_data = metric_data.get(category, {})
        sub_data = cat_data.get(sub, {})
        if not sub_data:
            continue
        for field in GUIDELLM_STAT_FIELDS:
            val = sub_data.get(field)
            if val is not None:
                metrics[f"{prefix}.{field}"] = float(val)
        # percentiles sub-dict
        percs = sub_data.get("percentiles", {})
        for pk, pv in percs.items():
            if pv is not None:
                metrics[f"{prefix}.{pk}"] = float(pv)

    return params, metrics


# ---------------------------------------------------------------------------
# PCP metrics extraction — auto-discovers all metrics in the archive
# ---------------------------------------------------------------------------

def _pcp_list_metrics(archive_base: str) -> list[str]:
    """Return all metric names present in a PCP archive."""
    proc = subprocess.run(
        ["pminfo", "-a", archive_base],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        return []
    names = []
    for line in proc.stdout.splitlines():
        name = line.strip()
        if not name:
            continue
        if any(name.endswith(s) for s in _PCP_SKIP_SUFFIXES):
            continue
        names.append(name)
    return names


def _pcp_get_semantics(archive_base: str, names: list[str]) -> dict[str, str]:
    """Return {name: 'counter'|'gauge'} using pminfo -d for descriptor info."""
    proc = subprocess.run(
        ["pminfo", "-d", "-a", archive_base] + names,
        capture_output=True, text=True, timeout=120,
    )
    semantics = {}
    current = None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("Data Type:") and not stripped.startswith("Semantics:"):
            current = stripped.split()[0]
        elif stripped.startswith("Semantics:") and current:
            sem = stripped.split(":", 1)[1].strip().split()[0].lower()
            semantics[current] = "counter" if sem == "counter" else "gauge"
    # Fall back to suffix heuristic for anything pminfo didn't describe
    for name in names:
        if name not in semantics:
            base = name.split(".")[-1]
            semantics[name] = "counter" if any(
                base.endswith(s) for s in ("_total", "_sum", "_count")
            ) else "gauge"
    return semantics


def _pcp_run_pmrep(archive_base: str, names: list[str]) -> dict[str, list[float]]:
    """Run pmrep and return {name: [sample_values...]} summing across instances."""
    result: dict[str, list[float]] = defaultdict(list)
    # Batch to stay under ARG_MAX
    for i in range(0, len(names), _PCP_PMREP_BATCH):
        batch = names[i:i + _PCP_PMREP_BATCH]
        try:
            proc = subprocess.run(
                ["pmrep", "-a", archive_base, "-o", "csv", "-z", "-t", "10s"] + batch,
                capture_output=True, text=True, timeout=300,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        reader = csv.reader(lines)
        raw_headers = next(reader)

        # Map metric name → column indices (sums across instances)
        col_map: dict[str, list[int]] = defaultdict(list)
        for ci, ch in enumerate(raw_headers):
            if ci == 0:
                continue
            col_raw = ch.strip()
            bracket = col_raw.find("[")
            col_base = col_raw[:bracket] if bracket != -1 else col_raw.split()[0]
            # Strip trailing "-N" instance suffix
            parts = col_base.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                col_base = parts[0]
            if col_base in batch:
                col_map[col_base].append(ci)

        for row in reader:
            ts_str = row[0].strip() if row else ""
            try:
                from datetime import datetime, timezone as _tz
                ts_ms = int(datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                            .replace(tzinfo=_tz.utc).timestamp() * 1000)
            except (ValueError, IndexError):
                ts_ms = None

            for name, cols in col_map.items():
                total = 0.0
                valid = False
                for c in cols:
                    if c < len(row) and row[c].strip():
                        try:
                            total += float(row[c])
                            valid = True
                        except (ValueError, TypeError):
                            pass
                if valid and ts_ms is not None:
                    result[name].append((ts_ms, total))

    return dict(result)


# Maps key PCP metric names to MLflow's well-known system/ metric names so the
# System metrics tab shows them alongside any auto-collected system metrics.
_SYSTEM_MAP = {
    "openmetrics.dcgm.DCGM_FI_DEV_GPU_UTIL":      "system/gpu_utilization_percentage",
    "openmetrics.dcgm.DCGM_FI_DEV_FB_USED":       "system/gpu_memory_usage_megabytes",
    "openmetrics.dcgm.DCGM_FI_DEV_POWER_USAGE":   "system/gpu_power_watts",
    "openmetrics.dcgm.DCGM_FI_DEV_GPU_TEMP":      "system/gpu_temp_celsius",
    "openmetrics.vllm.vllm.gpu_cache_usage_perc": "system/vllm_gpu_kvcache_pct",
    "openmetrics.vllm.vllm.kv_cache_usage_perc":  "system/vllm_cpu_kvcache_pct",
    "openmetrics.vllm.vllm.num_requests_running":  "system/vllm_requests_running",
    "openmetrics.vllm.vllm.num_requests_waiting":  "system/vllm_requests_waiting",
    "kernel.all.cpu.user":                          "system/cpu_user_milliseconds",
    "mem.util.used":                                "system/memory_used_kb",
}

# PCP time series point: (metric_key, value, timestamp_ms, step)
PcpPoint = tuple[str, float, int, int]


def extract_pcp_metrics(output_dir: Path) -> list[PcpPoint]:
    """Auto-discover PCP metrics and return full time series for MLflow.

    Returns a list of (key, value, timestamp_ms, step) tuples — one per sample
    per metric. Gauge metrics are logged as system/<pcp_name> time series.
    Counter metrics are logged as system/<pcp_name>.rate (per-interval delta,
    which shows throughput as a line chart rather than a cumulative staircase).
    _SYSTEM_MAP aliases also emit each sample under the well-known system/ name.
    All-zero series are omitted.
    """
    pcp_dir = output_dir / "pcp-archives"
    if not pcp_dir.exists():
        return []

    zst_files = list(pcp_dir.rglob("*.zst"))
    if not zst_files:
        return []

    with tempfile.TemporaryDirectory(prefix="pcp-mlflow-") as tmpdir:
        for zst_path in zst_files:
            dest = (Path(tmpdir) / zst_path.relative_to(pcp_dir)).with_suffix("")
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["zstd", "-d", "-q", "-o", str(dest), str(zst_path)],
                capture_output=True, timeout=60,
            )

        host_dirs = sorted(d for d in Path(tmpdir).iterdir() if d.is_dir())
        if not host_dirs:
            host_dirs = [Path(tmpdir)]

        points: list[PcpPoint] = []

        # Accumulate _SYSTEM_MAP metrics summed across hosts, keyed by
        # timestamp (rounded to 10s to handle minor clock skew between hosts)
        # {pcp_name: {ts_10s: summed_value}}
        alias_by_ts: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

        for host_dir in host_dirs:
            hostname = host_dir.name if host_dir != Path(tmpdir) else "host"

            raw: dict[str, list[tuple[int, float]]] = defaultdict(list)
            all_names: list[str] = []

            for meta_file in sorted(host_dir.rglob("*.meta")):
                archive_base = str(meta_file)[:-len(".meta")]
                names = _pcp_list_metrics(archive_base)
                if not all_names:
                    all_names = names
                for name, samples in _pcp_run_pmrep(archive_base, names).items():
                    raw[name].extend(samples)

            if not all_names:
                continue

            semantics = _pcp_get_semantics(
                str(sorted(host_dir.rglob("*.meta"))[0])[:-len(".meta")],
                all_names,
            )

            for name, samples in raw.items():
                if not samples or all(v == 0.0 for _, v in samples):
                    continue

                host_key = f"system/{hostname}/{name}"
                is_counter = semantics.get(name) == "counter"

                if is_counter:
                    rate_key = f"{host_key}.rate"
                    prev_val = None
                    for step, (ts_ms, val) in enumerate(samples):
                        if prev_val is not None:
                            rate = val - prev_val
                            if rate >= 0:
                                points.append((rate_key, rate, ts_ms, step))
                                # Accumulate counter rates into alias sum
                                if name in _SYSTEM_MAP:
                                    ts_10s = round(ts_ms / 10000) * 10000
                                    alias_by_ts[name][ts_10s] += rate
                        prev_val = val
                else:
                    for step, (ts_ms, val) in enumerate(samples):
                        points.append((host_key, val, ts_ms, step))
                        # Accumulate gauge values into alias sum
                        if name in _SYSTEM_MAP:
                            ts_10s = round(ts_ms / 10000) * 10000
                            alias_by_ts[name][ts_10s] += val

        # Emit _SYSTEM_MAP aliases as summed-across-hosts time series
        # under their canonical well-known system/ names (no hostname)
        for pcp_name, well_known_key in _SYSTEM_MAP.items():
            ts_data = alias_by_ts.get(pcp_name)
            if not ts_data:
                continue
            is_counter = well_known_key.endswith(".rate") or any(
                pcp_name.endswith(s) for s in ("_total", "_sum", "_count"))
            emit_key = f"{well_known_key}.rate" if is_counter else well_known_key
            for step, ts_10s in enumerate(sorted(ts_data)):
                val = ts_data[ts_10s]
                if val != 0.0:
                    points.append((emit_key, val, ts_10s, step))

    return points


# ---------------------------------------------------------------------------
# Config file parsing
# ---------------------------------------------------------------------------

def parse_benchmark_config(output_dir: Path) -> dict:
    """Parse benchmark-config.txt into a params dict."""
    config_file = output_dir / "benchmark-config.txt"
    if not config_file.exists():
        return {}

    params = {}
    for line in config_file.read_text().splitlines():
        if ": " in line:
            key, _, val = line.partition(": ")
            key = key.strip().lower().replace(" ", "_")
            params[key] = val.strip()
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Benchmark result directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory")
        sys.exit(1)

    run_id = output_dir.name  # e.g. 1x2xL40S_upstream-llm-d-0.7.0_Qwen3-8B_no-offload_replica1_rate1
    print(f"MLflow: logging run {run_id}")

    # --- Parse run ID ---
    parts = run_id.split("_")
    hardware = parts[0] if len(parts) > 0 else "unknown"
    software = parts[1] if len(parts) > 1 else "unknown"
    model_name = parts[2] if len(parts) > 2 else "unknown"
    parameters = parts[3] if len(parts) > 3 else "unknown"
    # replica and rate may contain embedded keyword
    replica_part = next((p for p in parts if p.startswith("replica")), "replica1")
    rate_part = next((p for p in parts if p.startswith("rate")), "rate0")
    replicas = replica_part.replace("replica", "")
    rate = rate_part.replace("rate", "")

    experiment_name = f"{hardware}/{software}"
    run_name = f"{model_name}_{parameters}_r{replicas}_rate{rate}"

    # --- Collect data ---
    print("  Reading benchmark-config.txt...")
    config_params = parse_benchmark_config(output_dir)

    print("  Reading guidellm results...")
    guidellm_params, guidellm_metrics = extract_guidellm_metrics(output_dir)

    print("  Extracting PCP metrics...")
    pcp_metrics = extract_pcp_metrics(output_dir)
    if pcp_metrics:
        print(f"  PCP: extracted {len(pcp_metrics)} metrics")
    else:
        print("  PCP: no metrics extracted")

    # --- Build MLflow payload ---
    all_params = {
        "hardware": hardware,
        "software": software,
        "model_name": model_name,
        "parameters": parameters,
        "replicas": replicas,
        "rate": rate,
    }
    all_params.update(config_params)
    all_params.update(guidellm_params)

    # guidellm metrics are single-value summaries; pcp_metrics are time series points
    all_metrics = dict(guidellm_metrics)

    tags = {
        "hardware": hardware,
        "software": software,
        "model_name": model_name,
        "parameters": parameters,
        "replicas": replicas,
        "rate": rate,
        "run_id_dir": run_id,
    }

    # MLflow param keys: alphanumerics, _ - . space : / only; values ≤ 500 chars
    import re as _re
    def _sanitize_key(k):
        return _re.sub(r"[^\w\-\.\s:/]", "_", k)

    all_params = {_sanitize_key(k): str(v)[:500] for k, v in all_params.items()}

    # --- Connect and log ---
    mlflow = setup_mlflow()
    conf   = _load_conf()
    def _get(key): return os.environ.get(key) or conf.get(key, "")

    uri       = _get("MLFLOW_TRACKING_URI")
    workspace = _get("MLFLOW_WORKSPACE")

    client = mlflow.tracking.MlflowClient()

    # Get or create experiment, restoring it if previously deleted
    exp = client.get_experiment_by_name(experiment_name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(experiment_name)
        print(f"  Created experiment: {experiment_name} (id={exp_id})")
    else:
        exp_id = exp.experiment_id
        print(f"  Using experiment: {experiment_name} (id={exp_id})")

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags) as run:
        mlflow_run_id  = run.info.run_id
        artifact_uri   = run.info.artifact_uri
        print(f"  Run ID: {mlflow_run_id}")

        # Params in batches of MLFLOW_PARAM_BATCH_SIZE
        param_items = list(all_params.items())
        for i in range(0, len(param_items), MLFLOW_PARAM_BATCH_SIZE):
            mlflow.log_params(dict(param_items[i:i+MLFLOW_PARAM_BATCH_SIZE]))

        # guidellm summary metrics (single values) via log_metrics
        metric_items = list(all_metrics.items())
        for i in range(0, len(metric_items), MLFLOW_METRIC_BATCH_SIZE):
            mlflow.log_metrics(dict(metric_items[i:i+MLFLOW_METRIC_BATCH_SIZE]))

        # PCP time series via log_batch with Metric objects (key, value, ts_ms, step)
        from mlflow.entities import Metric as MlflowMetric
        pcp_mlflow = [MlflowMetric(key=k, value=v, timestamp=ts, step=s)
                      for k, v, ts, s in pcp_metrics]
        for i in range(0, len(pcp_mlflow), MLFLOW_METRIC_BATCH_SIZE):
            client.log_batch(mlflow_run_id, metrics=pcp_mlflow[i:i+MLFLOW_METRIC_BATCH_SIZE])

        print(f"  Logged {len(all_params)} params, "
              f"{len(all_metrics)} guidellm metrics, "
              f"{len(pcp_metrics)} PCP time series points "
              f"({len(set(k for k,*_ in pcp_metrics))} metrics)")

        # Upload artifacts via the standard MLflow client (best-effort).
        # Requires S3 write access on the server — will work once that is configured.
        print("  Uploading artifacts...")
        try:
            for name in ["guidellm-results.json.zst", "benchmark-config.txt",
                         "vllm-startup.log.zst"]:
                path = output_dir / name
                if path.exists():
                    size_mb = path.stat().st_size / 1_048_576
                    print(f"    {name} ({size_mb:.1f} MB)...")
                    mlflow.log_artifact(str(path))

            pcp_dir = output_dir / "pcp-archives"
            if pcp_dir.exists():
                print(f"    pcp-archives/...")
                mlflow.log_artifacts(str(pcp_dir), artifact_path="pcp-archives")

            print("  Artifacts uploaded")
        except Exception as e:
            print(f"  Warning: artifact upload failed — {e}")

    ws_qs = f"?workspace={workspace}" if workspace else ""
    print(f"  Run:        {uri}/#/experiments/{exp_id}/runs/{mlflow_run_id}{ws_qs}")
    print(f"  Experiment: {uri}/#/experiments/{exp_id}{ws_qs}")


if __name__ == "__main__":
    main()
