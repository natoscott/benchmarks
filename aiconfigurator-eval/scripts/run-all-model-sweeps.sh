#!/usr/bin/env bash
# Full benchmark suite: Poisson TTFT sweep + ISL=9000 overhead sweep for all models.
#
# Runs all four models sequentially (each uses all 8 GPUs).
# Each model runs:
#   1. Poisson TTFT sweep  — validates Kingman G/G/1 queuing model
#   2. Overhead sweep      — recalibrates decode overhead correction model
#
# Usage:
#   bash scripts/run-all-model-sweeps.sh 2>&1 | tee /tmp/all-model-sweeps.log
#   tail -F /tmp/all-model-sweeps.log

set -euo pipefail

KUBECONFIG="${KUBECONFIG:-$HOME/psap/kubeconfig-psap-fire-athena}"
export KUBECONFIG

NS="${NS:-aiconfigurator}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DATE="$(date +%Y%m%d)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 300s per run — enough for stable statistics (600 requests at λ=2 req/s).
# Override with MAX_SECONDS env var if longer runs are needed.
export MAX_SECONDS="${MAX_SECONDS:-300}"

run_sweep() {
    local sweep_type="$1"   # poisson or overhead
    local manifest="$2"
    local name="$3"
    local target="$4"
    local expected_pods="$5"
    local rates="$6"
    local model_tag="$7"
    local isl="${8:-9000}"
    local osl="${9:-30}"

    log ">>> Starting $sweep_type sweep for $name (rates: $rates)"

    if [ "$sweep_type" = "poisson" ]; then
        LLMISVC_MANIFEST="$manifest" \
        LLMISVC_NAME="$name" \
        TARGET="$target" \
        EXPECTED_PODS="$expected_pods" \
        RATES="$rates" \
        MODEL_TAG="$model_tag" \
        ISL="$isl" \
        OSL="$osl" \
        OUTPUT_DIR="results/poisson-ttft-sweep-${OUTPUT_DATE}" \
        LOG_FILE="/tmp/poisson-${model_tag}.log" \
        bash "$SCRIPTS_DIR/run-poisson-ttft-sweep.sh"
    else
        LLMISVC_MANIFEST="$manifest" \
        LLMISVC_NAME="$name" \
        TARGET="$target" \
        EXPECTED_PODS="$expected_pods" \
        ISL_VALUES="$isl" \
        RATES="$rates" \
        OUTPUT_TOKENS="$osl" \
        MODEL_TAG="$model_tag" \
        OUTPUT_DIR="results/overhead-sweep-isl${isl}-${OUTPUT_DATE}" \
        LOG_FILE="/tmp/overhead-${model_tag}.log" \
        bash "$SCRIPTS_DIR/run-overhead-sweep.sh"
    fi

    log "<<< Finished $sweep_type sweep for $name"
}

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
#   name | manifest | target svc | expected_pods | poisson_rates | overhead_rates
# ---------------------------------------------------------------------------
#
# Saturation estimates (AIC, ISL=9000, OSL=30):
#   0.6B  ~24 req/s  → Poisson ρ 0.1-0.8 = 2..20 req/s
#   8B    ~4  req/s  → Poisson ρ 0.1-0.9 = 0.5..3.5 req/s
#   14B   ~2.3 req/s → Poisson ρ 0.1-0.9 = 0.3..2.0 req/s
#   32B   ~2  req/s  → Poisson ρ 0.1-0.9 = 0.3..1.5 req/s

BASE_URL="svc.cluster.local:8000"

declare -A MODELS
# Format: "manifest|name|target|pods|poisson_rates|overhead_rates"
MODELS[qwen3-0.6b]="manifests/llm-inference-service-qwen3-0.6b.yaml|qwen3-0p6b|https://qwen3-0p6b-kserve-workload-svc.${NS}.${BASE_URL}|8|2 5 8 12 16 20|2 8 16 32 64"
MODELS[qwen3-8b]="manifests/llm-inference-service-qwen3-8b.yaml|qwen3-8b|https://qwen3-8b-kserve-workload-svc.${NS}.${BASE_URL}|8|0.5 1.0 1.5 2.0 2.5 3.0 3.5|4 8 16 32 64"
MODELS[qwen3-14b]="manifests/llm-inference-service-qwen3-14b.yaml|qwen3-14b|https://qwen3-14b-kserve-workload-svc.${NS}.${BASE_URL}|8|0.3 0.5 0.8 1.0 1.3 1.5 2.0|4 8 16 32"
MODELS[qwen3-32b-fp8]="manifests/llm-inference-service-qwen3-32b-fp8-tp4.yaml|qwen3-32b-fp8-tp4|https://qwen3-32b-fp8-tp4-kserve-workload-svc.${NS}.${BASE_URL}|2|0.3 0.5 0.8 1.0 1.3 1.5 2.0|1 2 4 8 16"

# Run order: smallest to largest (faster models first)
RUN_ORDER="${RUN_ORDER:-qwen3-0.6b qwen3-8b qwen3-14b qwen3-32b-fp8}"

log "=========================================================="
log "Full model benchmark suite"
log "Models: $RUN_ORDER"
log "Output date tag: $OUTPUT_DATE"
log "=========================================================="

total_start=$SECONDS

for key in $RUN_ORDER; do
    IFS='|' read -r manifest name target pods poisson_rates overhead_rates <<< "${MODELS[$key]}"

    log ""
    log "########## $key ##########"
    model_start=$SECONDS

    # 1. Poisson TTFT sweep
    run_sweep poisson "$manifest" "$name" "$target" "$pods" "$poisson_rates" "$key"

    # Small pause between runs so the model server fully shuts down
    log "Waiting 30s between sweeps..."
    sleep 30

    # 2. Overhead sweep (ISL=9000 fixed concurrency, for silicon baseline calibration)
    run_sweep overhead "$manifest" "$name" "$target" "$pods" "$overhead_rates" "$key"

    model_elapsed=$(( SECONDS - model_start ))
    log "Completed $key in $(( model_elapsed / 60 ))m $(( model_elapsed % 60 ))s"
    log ""
done

total_elapsed=$(( SECONDS - total_start ))
log "=========================================================="
log "All sweeps complete in $(( total_elapsed / 60 ))m $(( total_elapsed % 60 ))s"
log ""
log "Analyse results:"
log "  python3 scripts/analyse-poisson-ttft.py --results-dir results/poisson-ttft-sweep-${OUTPUT_DATE} --model Qwen/Qwen3-8B"
log "  python3 scripts/analyse-overhead-sweep.py --results-dir results/overhead-sweep-isl9000-${OUTPUT_DATE} --model Qwen/Qwen3-8B"
log "=========================================================="
