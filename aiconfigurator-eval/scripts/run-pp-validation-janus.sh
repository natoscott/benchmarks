#!/usr/bin/env bash
# PP validation on janus cluster — MoE + dense large models.
# Uses existing run-pp-rate-sweep.sh with janus-specific config.
#
# Usage:
#   bash scripts/run-pp-validation-janus.sh 2>&1 | tee /tmp/pp-validation-janus.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST_DIR="${SCRIPT_DIR}/../manifests/pp-validation-janus"
RESULTS_DIR="${SCRIPT_DIR}/../results/pp-validation-janus"

export KUBECONFIG="${KUBECONFIG:?set KUBECONFIG to the cluster kubeconfig path}"
export NS="llm-d-nathans-epp-eval"

PROMPT_TOKENS=4000
OUTPUT_TOKENS=1000
MAX_SECONDS=120
RANDOM_SEED=889

CONC_MOE="1 2 4 8 16 32"
CONC_DENSE="1 2 4 8 16 32"

models=(
    "qwen3-30b-moe:Qwen/Qwen3-30B-A3B-Instruct-2507:$CONC_MOE"
    "llama-70b-fp8:RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic:$CONC_DENSE"
)

configs=("tp8pp1" "tp4pp2" "tp2pp4" "tp1pp8")

mkdir -p "$RESULTS_DIR"

total=$((${#models[@]} * ${#configs[@]}))
current=0

echo "=========================================="
echo "PP Validation — Janus Cluster"
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

        manifest="${MANIFEST_DIR}/${svc_name}.yaml"
        output_dir="${RESULTS_DIR}/${svc_name}"

        # Skip if results already exist
        if [ -d "$output_dir" ] && ls "$output_dir"/*.json.zst >/dev/null 2>&1; then
            echo "[$current/$total] Skipping $svc_name (results exist)"
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
    done
done

echo ""
echo "=========================================="
echo "PP Validation Janus Complete"
echo "Results in: $RESULTS_DIR/"
for d in "$RESULTS_DIR"/*/; do
    [ -d "$d" ] || continue
    count=$(ls "$d"/*.json.zst 2>/dev/null | wc -l | tr -d ' ')
    echo "  $(basename "$d"): $count result files"
done
echo "=========================================="
