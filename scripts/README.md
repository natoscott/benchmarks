# Shared Scripts

Utility scripts shared across benchmark scenarios in this repository.

Scripts in each scenario's own `scripts/` directory reference these as
`${SCRIPT_DIR}/../../scripts/<name>`.

## Scripts

| Script | Purpose |
|--------|---------|
| `strip-guidellm-request-content.py` | Strip prompt/response text from `guidellm-results.json.zst` files, preserving all metrics. Supports single-file and batch modes. Typically reduces file size by ~86%. |
| `transfer-large-file-chunked.sh` | Base64-chunked file transfer via `kubectl exec`. Splits a remote file into chunks, base64-encodes each for transfer, and reassembles locally with size verification and retry logic. Avoids binary stream truncation that affects `kubectl cp` for large files. |
| `mlflow-log-run.py` | Log a single benchmark result directory to MLflow. Extracts guidellm metrics, PCP time series, and benchmark config. Requires `mlflow.conf` (see `mlflow.conf.example` in repo root) or `MLFLOW_TRACKING_URI` env var. |
| `mlflow-import-all.sh` | Bulk-import all benchmark result directories to MLflow. Skips previously imported runs (tracked via `.mlflow-imported` sentinel file). |
