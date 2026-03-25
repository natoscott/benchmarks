# Shared Scripts

Utility scripts shared across benchmark scenarios in this repository.

Scripts in each scenario's own `scripts/` directory reference these as
`${SCRIPT_DIR}/../../scripts/<name>`.

## Scripts

| Script | Purpose |
|--------|---------|
| `transfer-large-file-chunked.sh` | Base64-chunked file transfer via `kubectl exec`. Avoids binary stream truncation that affects `kubectl cp` for large files. |
