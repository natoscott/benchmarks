#!/usr/bin/env bash
# PP validation benchmark suite.
#
# Deploys each model with 4 TP/PP configurations (TP=8/PP=1, TP=4/PP=2,
# TP=2/PP=4, TP=1/PP=8), runs guidellm at multiple concurrency levels,
# and collects throughput/latency results for comparison against AIC
# analytical model predictions.
#
# All configs use 8 GPUs / 1 replica on H200 SXM with vLLM.
# Workload: ISL=4000, OSL=1000 (matching the AIC prediction parameters).
#
# Usage:
#   bash scripts/run-pp-validation.sh 2>&1 | tee /tmp/pp-validation.log
#   tail -F /tmp/pp-validation.log
#
# To resume from a specific model/config (skips already-collected results):
#   RESUME_FROM=qwen3-14b-tp2pp4 bash scripts/run-pp-validation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST_DIR="${SCRIPT_DIR}/../manifests/pp-validation"
RESULTS_DIR="${SCRIPT_DIR}/../results/pp-validation"
PROMPT_TOKENS=4000
OUTPUT_TOKENS=1000
MAX_SECONDS=120
RANDOM_SEED=889
RESUME_FROM="${RESUME_FROM:-}"

# Concurrency levels to sweep — covers light to saturated load.
CONC_SMALL="1 2 4 8 16 32 64"     # 0.6B, 8B
CONC_LARGE="1 2 4 8 16 32"        # 14B, 32B

models=(
    "qwen3-06b:Qwen/Qwen3-0.6B:$CONC_SMALL"
    "qwen3-8b:Qwen/Qwen3-8B:$CONC_SMALL"
    "qwen3-14b:Qwen/Qwen3-14B:$CONC_LARGE"
    "qwen3-32b-fp8:Qwen/Qwen3-32B-FP8:$CONC_LARGE"
)

configs=("tp8pp1" "tp4pp2" "tp2pp4" "tp1pp8")

mkdir -p "$RESULTS_DIR"

# Track overall progress
total=$((${#models[@]} * ${#configs[@]}))
current=0
skipping=true

echo "=========================================="
echo "PP Validation Benchmark Suite"
echo "Models:  ${#models[@]}"
echo "Configs: ${#configs[@]}"
echo "Total:   $total deployments"
echo "ISL=$PROMPT_TOKENS OSL=$OUTPUT_TOKENS"
echo "=========================================="
echo ""

for model_spec in "${models[@]}"; do
    IFS=: read -r model_short hf_model_name concurrencies <<< "$model_spec"

    for config in "${configs[@]}"; do
        svc_name="${model_short}-${config}"
        current=$((current + 1))

        # Resume support: skip until we reach RESUME_FROM
        if [ -n "$RESUME_FROM" ] && [ "$skipping" = true ]; then
            if [ "$svc_name" = "$RESUME_FROM" ]; then
                skipping=false
            else
                echo "[$current/$total] Skipping $svc_name (resuming from $RESUME_FROM)"
                continue
            fi
        fi

        manifest="${MANIFEST_DIR}/${svc_name}.yaml"
        output_dir="${RESULTS_DIR}/${svc_name}"

        # Skip if results already exist
        if [ -d "$output_dir" ] && ls "$output_dir"/*.json.zst >/dev/null 2>&1; then
            echo "[$current/$total] Skipping $svc_name (results exist in $output_dir)"
            continue
        fi

        echo ""
        echo "=========================================="
        echo "[$current/$total] $svc_name"
        echo "=========================================="

        LLMISVC_MANIFEST="$manifest" \
        LLMISVC_NAME="$svc_name" \
        MODEL_NAME="$hf_model_name" \
        CONCURRENCIES="$concurrencies" \
        OUTPUT_DIR="$output_dir" \
        OUTPUT_PREFIX="pp-val-${svc_name}" \
        MAX_SECONDS="$MAX_SECONDS" \
        RANDOM_SEED="$RANDOM_SEED" \
        PROMPT_TOKENS="$PROMPT_TOKENS" \
        OUTPUT_TOKENS="$OUTPUT_TOKENS" \
            bash "${SCRIPT_DIR}/run-pp-rate-sweep.sh"

        echo "[$current/$total] $svc_name complete."
        echo ""
    done
done

echo ""
echo "=========================================="
echo "PP Validation Suite Complete"
echo "Results in: $RESULTS_DIR/"
echo ""
for d in "$RESULTS_DIR"/*/; do
    [ -d "$d" ] || continue
    count=$(ls "$d"/*.json.zst 2>/dev/null | wc -l)
    echo "  $(basename "$d"): $count result files"
done
echo "=========================================="
