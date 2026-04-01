#!/bin/bash
# Re-run all gpt-oss-120b benchmarks with fixed guidellm image.
# guidellm deployment must already be patched to the nightly build before running.
#
# Covers all three profiles:
#   rhoai-3.3          (short-context: prompt=512, output=128)
#   rhoai-3.3-kv-stress (long-output:  prompt=512, output=512)
#   rhoai-3.3-longctx  (long-context:  prompt=4096, output=256)
#
# 2 configs × 3 replicas × 8 rates × 3 profiles = 144 runs
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-./kubeconfig-psap-fire-athena}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export HARDWARE="${HARDWARE:-1x8xH200}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export MAX_SECONDS="${MAX_SECONDS:-120}"
export TURNS="${TURNS:-5}"
export RATE_TYPE="${RATE_TYPE:-concurrent}"
export RANDOM_SEED="${RANDOM_SEED:-889}"

export MODEL="openai/gpt-oss-120b"
export MODEL_NAME="gpt-oss-120b"
export LLM_SERVICE_NAME="gpt-oss-120b"

export RUNS="no-offload native-offload-20k"
export REPLICAS_LIST="1 2 4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "gpt-oss-120b Re-run (guidellm bugfix)"
echo "guidellm image: $(kubectl --kubeconfig=${KUBECONFIG} -n ${NAMESPACE} \
    get deployment guidellm -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null)"
echo "=========================================="

# ── short-context (rhoai-3.3) ─────────────────────────────────────────────────
export SOFTWARE="rhoai-3.3"
export GPU_MEMORY_UTILIZATION="0.65"
export PROMPT_TOKENS="512"
export OUTPUT_TOKENS="128"
export PREFIX_TOKENS="10000"
RATE_LIST="1,50,100,150,300,400,500,650"
IFS=',' read -ra RATES <<< "${RATE_LIST}"

echo ""
echo "── short-context (${SOFTWARE}) ──────────────────"
for replicas in ${REPLICAS_LIST}; do
    export CURRENT_REPLICAS="${replicas}"
    for run in ${RUNS}; do
        export PARAMETERS="${run}"
        export NUM_CPU_BLOCKS="20000"
        for rate in "${RATES[@]}"; do
            export RATE="${rate}"
            export PREFIX_COUNT=$(( 2 * rate ))
            echo "▶ ${SOFTWARE} | gpt-oss-120b | ${run} | replicas=${replicas} | rate=${rate}"
            bash "${SCRIPT_DIR}/run-benchmark.sh"
        done
    done
done

# ── long-output (rhoai-3.3-kv-stress) ────────────────────────────────────────
export SOFTWARE="rhoai-3.3-kv-stress"
export GPU_MEMORY_UTILIZATION="0.65"
export PROMPT_TOKENS="512"
export OUTPUT_TOKENS="512"
export PREFIX_TOKENS="10000"
RATE_LIST="1,50,100,150,300,400,500,650"
IFS=',' read -ra RATES <<< "${RATE_LIST}"

echo ""
echo "── long-output (${SOFTWARE}) ────────────────────"
for replicas in ${REPLICAS_LIST}; do
    export CURRENT_REPLICAS="${replicas}"
    for run in ${RUNS}; do
        export PARAMETERS="${run}"
        export NUM_CPU_BLOCKS="20000"
        for rate in "${RATES[@]}"; do
            export RATE="${rate}"
            export PREFIX_COUNT=$(( 2 * rate ))
            echo "▶ ${SOFTWARE} | gpt-oss-120b | ${run} | replicas=${replicas} | rate=${rate}"
            bash "${SCRIPT_DIR}/run-benchmark.sh"
        done
    done
done

# ── long-context (rhoai-3.3-longctx) ─────────────────────────────────────────
export SOFTWARE="rhoai-3.3-longctx"
export GPU_MEMORY_UTILIZATION="0.50"
export PROMPT_TOKENS="4096"
export OUTPUT_TOKENS="256"
export PREFIX_TOKENS="10000"
RATE_LIST="1,5,10,20,50,100,200,300"
IFS=',' read -ra RATES <<< "${RATE_LIST}"

echo ""
echo "── long-context (${SOFTWARE}) ───────────────────"
for replicas in ${REPLICAS_LIST}; do
    export CURRENT_REPLICAS="${replicas}"
    for run in ${RUNS}; do
        export PARAMETERS="${run}"
        export NUM_CPU_BLOCKS="20000"
        for rate in "${RATES[@]}"; do
            export RATE="${rate}"
            export PREFIX_COUNT=$(( 2 * rate ))
            echo "▶ ${SOFTWARE} | gpt-oss-120b | ${run} | replicas=${replicas} | rate=${rate}"
            bash "${SCRIPT_DIR}/run-benchmark.sh"
        done
    done
done

echo ""
echo "=========================================="
echo "All gpt-oss-120b re-runs complete (144 runs)."
echo "Results in: results/"
echo "Remember to revert guidellm image when done:"
echo "  kubectl --kubeconfig=${KUBECONFIG} -n ${NAMESPACE} patch deployment guidellm \\"
echo "    --type=json -p '[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/image\",\"value\":\"ghcr.io/vllm-project/guidellm:latest\"}]'"
echo "=========================================="
