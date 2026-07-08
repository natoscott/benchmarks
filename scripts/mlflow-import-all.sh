#!/bin/bash
# Bulk-import all benchmark result directories to MLflow.
# Skips directories that have already been imported (tracked via a sentinel file).
#
# Usage: bash scripts/mlflow-import-all.sh [PATTERN] [RESULTS_DIR]
#   PATTERN:     glob pattern to filter result dirs (default: all)
#   RESULTS_DIR: path to results directory (default: ../results relative to script)
#
# Examples:
#   bash scripts/mlflow-import-all.sh '*upstream-llm-d-0.5.1*'
#   bash scripts/mlflow-import-all.sh '*' llm-d-batch-eval/results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN="${1:-*}"
RESULTS_DIR="${2:-${SCRIPT_DIR}/../results}"
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

    # Must have guidellm results (single file or per-phase) to be worth importing
    if [ ! -f "${result_dir}/guidellm-results.json.zst" ]; then
        phase_count=$(ls "${result_dir}"/burst-*.json.zst 2>/dev/null | wc -l)
        if [ "${phase_count}" -eq 0 ]; then
            skip=$((skip + 1))
            continue
        fi
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
