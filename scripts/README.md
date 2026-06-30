# Shared Scripts

Utility scripts shared across benchmark scenarios in this repository.

Scripts in each scenario's own `scripts/` directory reference these as
`${SCRIPT_DIR}/../../scripts/<name>`.

## Scripts

| Script | Purpose |
|--------|---------|
| `strip-guidellm-request-content.py` | Strip prompt/response text from `guidellm-results.json.zst` files, preserving all metrics. Supports single-file and batch modes. Typically reduces file size by ~86%. |
| `transfer-large-file-chunked.sh` | Base64-chunked file transfer via `kubectl exec`. Splits a remote file into chunks, base64-encodes each for transfer, and reassembles locally with size verification and retry logic. Avoids binary stream truncation that affects `kubectl cp` for large files. |
