#!/bin/bash
set -e

# Benchmarks for llm-d v0.6.0 (vLLM 0.17.1) — filesystem and CPU+filesystem offload.
# Uses llmd_fs_connector-0.18.0 wheel (llm-d-kv-cache v0.7.1).
# Wheel must be pre-staged on model-storage-pvc at /data/llmd_fs_connector-0.18.0-cp312-cp312-linux_x86_64.whl
# Mirrors run-v0.5.1-filesystem-offload.sh for version comparison.

RUNS="${RUNS:-fs-offload cpu+fs-offload-20k}"
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ}"
REPLICAS="${REPLICAS:-1}"

export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export RATE_LIST="${RATE:-1,50,100,150,300,400,500,650}"
export MAX_SECONDS=120
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="${SOFTWARE:-upstream-llm-d-0.6.0}"
export TURNS=5
export INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-llm-d-model-server}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPUS_PER_REPLICA="${TENSOR_PARALLEL_SIZE}"

# llmd_fs_connector wheel — v0.18.0 built with -static-libstdc++ (llm-d-kv-cache PR#498).
# Statically links libstdc++ to avoid GLIBCXX_3.4.30 runtime dependency missing in
# llm-d-cuda:v0.6.0. No LD_PRELOAD workaround required.
# Staged on model-storage-pvc as /data/llmd_fs_connector-0.18-cp312-cp312-linux_x86_64.whl
# (copied from the -static- named artifact with valid wheel filename for pip).
FS_WHEEL_PATH="/data/llmd_fs_connector-0.18-cp312-cp312-linux_x86_64.whl"
FS_PACKAGES_DIR="/tmp/llmd_packages"

# ---------------------------------------------------------------------------
# One-time setup: ensure kvcache-storage-pvc is mounted
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Checking kvcache volume mount"
echo "=========================================="
EXISTING_VOLUMES=$(kubectl --kubeconfig="${KUBECONFIG}" get deployment "${INFERENCE_DEPLOYMENT}" \
    -n "${NAMESPACE}" -o jsonpath='{.spec.template.spec.volumes[*].name}')
if echo "${EXISTING_VOLUMES}" | grep -qw kvcache; then
    echo "  kvcache volume already mounted"
else
    echo "  Adding kvcache-storage-pvc volume mount at /kvcache..."
    kubectl --kubeconfig="${KUBECONFIG}" patch deployment "${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" --type='json' -p='[
        {"op":"add","path":"/spec/template/spec/volumes/-",
         "value":{"name":"kvcache","persistentVolumeClaim":{"claimName":"kvcache-storage-pvc"}}},
        {"op":"add","path":"/spec/template/spec/containers/0/volumeMounts/-",
         "value":{"name":"kvcache","mountPath":"/kvcache"}}
    ]'
fi
echo ""

IFS=',' read -ra RATES <<< "$RATE_LIST"

for replicas in ${REPLICAS}; do
    export CURRENT_REPLICAS="${replicas}"

    echo "=========================================="
    echo "Setting replica count to: ${replicas}"
    echo "=========================================="
    kubectl --kubeconfig="${KUBECONFIG}" scale deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" --replicas="${replicas}"
    kubectl --kubeconfig="${KUBECONFIG}" rollout status deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" --timeout=300s
    sleep 30

    for model in ${MODELS}; do
        export MODEL="${model}"
        export MODEL_NAME="${model##*/}"

        case "${MODEL_NAME}" in
            "Qwen3-0.6B")    CPU_BYTES_20K=72842645340 ;;
            "Qwen3-8B")      CPU_BYTES_20K=57616986275 ;;
            "Qwen3-14B")     CPU_BYTES_20K=44195213475 ;;
            "Qwen3-32B-AWQ") CPU_BYTES_20K=54546084659 ;;
            *) echo "Unknown model: ${MODEL_NAME}"; exit 1 ;;
        esac

        for run in ${RUNS}; do
            export PARAMETERS="${run}"

            case "${run}" in
                "fs-offload")
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.6.0"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"SharedStorageOffloadingSpec\",\"shared_storage_path\":\"/kvcache/kv-cache/\",\"block_size\":256,\"threads_per_gpu\":128,\"spec_module_path\":\"llmd_fs_backend.spec\"}}' --distributed-executor-backend mp"
                    export VLLM_ENV_VARS="PYTHONHASHSEED=42 PYTHONPATH=${FS_PACKAGES_DIR}"
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE=""
                    export VLLM_PRE_CMD="pip3.12 install --quiet --target ${FS_PACKAGES_DIR} ${FS_WHEEL_PATH} && mkdir -p /kvcache/kv-cache"
                    ;;
                "cpu+fs-offload-20k")
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.6.0"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${CPU_BYTES_20K}}},{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"SharedStorageOffloadingSpec\",\"shared_storage_path\":\"/kvcache/kv-cache/\",\"block_size\":256,\"threads_per_gpu\":128,\"spec_module_path\":\"llmd_fs_backend.spec\"}}]}}' --distributed-executor-backend mp"
                    export VLLM_ENV_VARS="PYTHONHASHSEED=42 PYTHONPATH=${FS_PACKAGES_DIR}"
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE=""
                    export VLLM_PRE_CMD="pip3.12 install --quiet --target ${FS_PACKAGES_DIR} ${FS_WHEEL_PATH} && mkdir -p /kvcache/kv-cache"
                    ;;
                *)
                    echo "Unknown configuration: ${run}"
                    exit 1
                    ;;
            esac

            for rate in "${RATES[@]}"; do
                export RATE="${rate}"
                export PREFIX_COUNT=$((2 * rate))
                export SAMPLE_REQUESTS=$((2 * rate * TURNS))

                RUN_ID="${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PARAMETERS}_replica${replicas}_rate${rate}"
                OUTPUT_DIR="./results/${RUN_ID}"

                if [ -f "${OUTPUT_DIR}/guidellm-results.json.zst" ]; then
                    echo "SKIPPING (already complete): ${RUN_ID}"
                    continue
                fi

                echo ""
                echo "=========================================="
                echo "Starting: ${MODEL_NAME} / ${run} / rate=${rate}"
                echo "=========================================="

                bash "$(dirname "$0")/run-benchmark.sh"

                # Clean kvcache between runs to prevent cross-run contamination
                INFER_POD=$(kubectl --kubeconfig="${KUBECONFIG}" get pods -n "${NAMESPACE}" \
                    -l llm-d.ai/inference-serving=true --field-selector=status.phase=Running \
                    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
                if [ -n "${INFER_POD}" ]; then
                    kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${INFER_POD}" -- \
                        sh -c "rm -rf /kvcache/kv-cache/* 2>/dev/null; echo 'kvcache cleared'" || true
                fi
                sleep 10
            done
        done
    done
done

echo ""
echo "=========================================="
echo "v0.6.0 filesystem offload benchmarks complete!"
echo "Results in: results/${HARDWARE}_${SOFTWARE}_*"
echo "=========================================="
