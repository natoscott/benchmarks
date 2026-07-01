#!/bin/bash
# Bulk-import all benchmark result directories to MLflow.
# Skips directories that have already been imported (tracked via a sentinel file).
#
# Usage: bash scripts/mlflow-import-all.sh [PATTERN]
#   PATTERN: glob pattern to filter result dirs (default: all)
#   e.g. bash scripts/mlflow-import-all.sh '*upstream-llm-d-0.5.1*'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
PATTERN="${1:-*}"
SENTINEL=".mlflow-imported"

pass=0
fail=0
skip=0

for result_dir in "${RESULTS_DIR}"/${PATTERN}; do
    [ -d "${result_dir}" ] || continue

    # Skip if already imported
    if [ -f "${result_dir}/${SENTINEL}" ]; then
        skip=$((skip + 1))
        continue
    fi

    # Must have guidellm results to be worth importing
    if [ ! -f "${result_dir}/guidellm-results.json.zst" ]; then
        skip=$((skip + 1))
        continue
    fi

    echo "Importing: $(basename "${result_dir}")"
    if python3 "${SCRIPT_DIR}/mlflow-log-run.py" "${result_dir}" 2>&1; then
        touch "${result_dir}/${SENTINEL}"
        pass=$((pass + 1))
    else
        echo "  FAILED: $(basename "${result_dir}")"
        fail=$((fail + 1))
    fi
done

echo ""
echo "Done: ${pass} imported, ${fail} failed, ${skip} skipped"
