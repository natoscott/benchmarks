#!/bin/bash
set -e

# Benchmarks for llm-d v0.5.1 filesystem KV cache offload via llmd_fs_connector.
#
# Uses vLLM's OffloadingConnector with SharedStorageOffloadingSpec from the
# llmd_fs_backend wheel (https://github.com/llm-d/llm-d-kv-cache).  The wheel
# is pip-installed into the vLLM venv at pod startup; no image change required
# since ghcr.io/llm-d/llm-d-cuda:v0.5.1 ships vLLM 0.15.1 + Python 3.12.
#
# Storage: kvcache-storage-pvc (256Gi IBM VPC block) mounted at /kvcache,
# presenting as a local POSIX filesystem inside the pod.
#
# Primary purpose: establish baseline overhead of the filesystem offload path
# vs CPU-only offload and no-offload on a single-node setup.
#
# Configs:
#   fs-offload  - GPU → filesystem (SharedStorageOffloadingSpec via llmd_fs_connector)

RUNS="${RUNS:-cpu+fs-offload-20k}"
MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B-AWQ}"
REPLICAS="${REPLICAS:-1}"

export KUBECONFIG="${KUBECONFIG:-./kubeconfig}"
export NAMESPACE="${NAMESPACE:-llm-d-pfc-cpu}"
export RATE_LIST="${RATE:-1,50,100,150,300,400,500,650}"
export MAX_SECONDS=120
export HARDWARE="${HARDWARE:-1x2xL40S}"
export SOFTWARE="${SOFTWARE:-upstream-llm-d-0.5.1}"
export TURNS=5
export INFERENCE_DEPLOYMENT="${INFERENCE_DEPLOYMENT:-llm-d-model-server}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPUS_PER_REPLICA="${TENSOR_PARALLEL_SIZE}"

# llmd_fs_connector wheel — version-matched to vLLM 0.15.1 (shipped in llm-d-cuda:v0.5.1).
# Wheel is pre-copied onto model-storage-pvc at /data/llmd_fs_connector.whl so there
# is no network dependency at pod startup. Uses --target to install into the vLLM venv
# site-packages where /opt/vllm/bin/python3 will find it.
FS_WHEEL_PATH="/data/llmd_fs_connector-0.15.1-cp312-cp312-linux_x86_64.whl"
FS_PACKAGES_DIR="/tmp/llmd_packages"
# Install to /tmp (writable) and expose via PYTHONPATH (added to VLLM_ENV_VARS below)
export VLLM_PRE_CMD="pip3.12 install --quiet --target ${FS_PACKAGES_DIR} ${FS_WHEEL_PATH} && mkdir -p /kvcache/kv-cache"

# ---------------------------------------------------------------------------
# One-time setup: ensure kvcache-storage-pvc is mounted in the inference server.
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Setting up kvcache volume mount"
echo "=========================================="
EXISTING_VOLUMES=$(kubectl --kubeconfig="${KUBECONFIG}" get deployment "${INFERENCE_DEPLOYMENT}" \
    -n "${NAMESPACE}" -o jsonpath='{.spec.template.spec.volumes[*].name}')
if echo "${EXISTING_VOLUMES}" | grep -qw kvcache; then
    echo "  kvcache volume already mounted, skipping patch"
else
    echo "  Adding kvcache-storage-pvc volume mount at /kvcache..."
    kubectl --kubeconfig="${KUBECONFIG}" patch deployment "${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" --type='json' -p='[
        {"op":"add","path":"/spec/template/spec/volumes/-",
         "value":{"name":"kvcache","persistentVolumeClaim":{"claimName":"kvcache-storage-pvc"}}},
        {"op":"add","path":"/spec/template/spec/containers/0/volumeMounts/-",
         "value":{"name":"kvcache","mountPath":"/kvcache"}}
    ]'
    echo "  Volume mount added"
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
    echo "Waiting for deployment to scale to ${replicas} replicas..."
    kubectl --kubeconfig="${KUBECONFIG}" rollout status deployment/"${INFERENCE_DEPLOYMENT}" \
        -n "${NAMESPACE}" --timeout=300s
    echo "Waiting 30 seconds for all replicas to stabilize..."
    sleep 30

    for model in ${MODELS}; do
        export MODEL="$model"
        export MODEL_NAME="${model##*/}"

        for run in ${RUNS}; do
            export PARAMETERS="$run"

            # Per-model CPU byte allocations (20k blocks, same as native-offload-20k)
            case "$MODEL_NAME" in
                "Qwen3-0.6B")  CPU_BYTES_20K=72842645340 ;;
                "Qwen3-8B")    CPU_BYTES_20K=57616986275 ;;
                "Qwen3-14B")   CPU_BYTES_20K=44195213475 ;;
                "Qwen3-32B-AWQ") CPU_BYTES_20K=54546084659 ;;
                *) echo "Unknown model: $MODEL_NAME"; exit 1 ;;
            esac

            NSIGHT_LIBSTDCPP="/opt/nvidia/nsight-compute/2025.2.1/host/linux-desktop-glibc_2_11_3-x64/libstdc++.so.6"

            case "$run" in
                "fs-offload")
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.5.1"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"SharedStorageOffloadingSpec\",\"shared_storage_path\":\"/kvcache/kv-cache/\",\"block_size\":256,\"threads_per_gpu\":64,\"spec_module_path\":\"llmd_fs_backend.spec\"}}' --distributed-executor-backend mp"
                    export VLLM_ENV_VARS="PYTHONHASHSEED=42 PYTHONPATH=${FS_PACKAGES_DIR} LD_PRELOAD=${NSIGHT_LIBSTDCPP}"
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE=""
                    ;;
                "cpu+fs-offload-20k")
                    # Hierarchical GPU→CPU→filesystem via MultiConnector.
                    # Load: CPU first (fast hit), filesystem fallback.
                    # Save: both simultaneously.
                    # CPU allocation matches native-offload-20k for direct comparison.
                    export CONTAINER_IMAGE="ghcr.io/llm-d/llm-d-cuda:v0.5.1"
                    export VLLM_EXTRA_ARGS="--kv-transfer-config '{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${CPU_BYTES_20K}}},{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"SharedStorageOffloadingSpec\",\"shared_storage_path\":\"/kvcache/kv-cache/\",\"block_size\":256,\"threads_per_gpu\":64,\"spec_module_path\":\"llmd_fs_backend.spec\"}}]}}' --distributed-executor-backend mp"
                    export VLLM_ENV_VARS="PYTHONHASHSEED=42 PYTHONPATH=${FS_PACKAGES_DIR} LD_PRELOAD=${NSIGHT_LIBSTDCPP}"
                    export EPP_BACKEND_CONFIG="in-memory"
                    export USE_LMCACHE_IMAGE=""
                    ;;
                *)
                    echo "Unknown configuration: $run"
                    exit 1
                    ;;
            esac

            for rate in "${RATES[@]}"; do
                export RATE="$rate"
                export PREFIX_COUNT=$((2 * rate))
                export SAMPLE_REQUESTS=$((2 * rate * TURNS))

                RUN_ID="${HARDWARE}_${SOFTWARE}_${MODEL_NAME}_${PARAMETERS}_replica${replicas}_rate${rate}"
                OUTPUT_DIR="./results/${RUN_ID}"

                if [ -f "${OUTPUT_DIR}/guidellm-results.json.zst" ]; then
                    echo ""
                    echo "SKIPPING (already complete): ${RUN_ID}"
                    continue
                fi

                echo ""
                echo "=========================================="
                echo "Starting benchmark run:"
                echo "  Model:         ${MODEL}"
                echo "  Configuration: ${PARAMETERS}"
                echo "  Storage path:  /kvcache/kv-cache/"
                echo "  Replicas:      ${replicas}"
                echo "  Rate:          ${rate}"
                echo "  Prefix Count:  ${PREFIX_COUNT}"
                echo "  Sample Req:    ${SAMPLE_REQUESTS}"
                echo "=========================================="

                bash "$(dirname "$0")/run-benchmark.sh"

                echo "Waiting 10 seconds before next run..."
                # Clean up kvcache storage after each run so that:
                # 1. Each run starts with a clean cache (no cross-run contamination)
                # 2. Accumulated files don't cause slow fsGroup chown on next pod restart
                # The model server pod has kvcache-storage-pvc mounted at /kvcache.
                echo "Cleaning up kvcache storage..."
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
echo "All filesystem offload benchmarks complete!"
echo "Results saved in: results/"
echo "=========================================="
