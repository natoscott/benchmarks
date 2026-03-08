#!/bin/bash
# Chunked file transfer to avoid kubectl connection timeouts on large files
# This script splits a file in the pod, transfers chunks, and reassembles locally

set -euo pipefail

KUBECONFIG="${1:?KUBECONFIG required}"
NAMESPACE="${2:?NAMESPACE required}"
POD="${3:?POD required}"
REMOTE_FILE="${4:?REMOTE_FILE required}"
LOCAL_FILE="${5:?LOCAL_FILE required}"
CHUNK_SIZE_MB="${6:-10}"  # Default 10MB chunks

echo "Chunked file transfer:"
echo "  Remote: ${REMOTE_FILE}"
echo "  Local: ${LOCAL_FILE}"
echo "  Chunk size: ${CHUNK_SIZE_MB}MB"

# Get file size in pod
FILE_SIZE=$(kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${POD}" -- stat -c '%s' "${REMOTE_FILE}")
echo "  File size: $((FILE_SIZE / 1024 / 1024))MB (${FILE_SIZE} bytes)"

# Calculate number of chunks needed
CHUNK_SIZE_BYTES=$((CHUNK_SIZE_MB * 1024 * 1024))
NUM_CHUNKS=$(( (FILE_SIZE + CHUNK_SIZE_BYTES - 1) / CHUNK_SIZE_BYTES ))
echo "  Chunks needed: ${NUM_CHUNKS}"

# Create temp directory for chunks
CHUNK_DIR=$(mktemp -d)
trap "rm -rf '${CHUNK_DIR}'" EXIT

# Split file in pod into chunks
echo "Splitting file in pod..."
kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${POD}" -- \
    sh -c "split -b ${CHUNK_SIZE_BYTES} -d -a 4 '${REMOTE_FILE}' '${REMOTE_FILE}.chunk.'"

# Get list of chunks created
CHUNKS=$(kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${POD}" -- \
    sh -c "ls '${REMOTE_FILE}'.chunk.* 2>/dev/null" | sort)

if [ -z "${CHUNKS}" ]; then
    echo "ERROR: No chunks created"
    exit 1
fi

# Transfer each chunk
CHUNK_NUM=0
TOTAL_CHUNKS=$(echo "${CHUNKS}" | wc -l)
echo "Transferring ${TOTAL_CHUNKS} chunks..."

for CHUNK in ${CHUNKS}; do
    CHUNK_NUM=$((CHUNK_NUM + 1))
    CHUNK_NAME=$(basename "${CHUNK}")
    echo "  Chunk ${CHUNK_NUM}/${TOTAL_CHUNKS}: ${CHUNK_NAME}"

    # Transfer with retry logic
    SUCCESS=0
    for attempt in 1 2 3; do
        if timeout --signal=KILL 30 sh -c \
            "kubectl --kubeconfig='${KUBECONFIG}' exec -n '${NAMESPACE}' '${POD}' -- cat '${CHUNK}' > '${CHUNK_DIR}/${CHUNK_NAME}'" 2>&1; then
            SUCCESS=1
            break
        else
            echo "    Retry $attempt/3..."
            sleep 1
        fi
    done

    if [ ${SUCCESS} -eq 0 ]; then
        echo "ERROR: Failed to transfer chunk ${CHUNK_NAME}"
        exit 1
    fi
done

# Reassemble chunks locally
echo "Reassembling file..."
cat "${CHUNK_DIR}"/*.chunk.* > "${LOCAL_FILE}"

# Verify file size matches
LOCAL_SIZE=$(stat -c '%s' "${LOCAL_FILE}")
if [ "${LOCAL_SIZE}" -ne "${FILE_SIZE}" ]; then
    echo "ERROR: Size mismatch! Remote: ${FILE_SIZE}, Local: ${LOCAL_SIZE}"
    exit 1
fi

echo "File transferred successfully (${LOCAL_SIZE} bytes)"

# Clean up chunks in pod
echo "Cleaning up chunks in pod..."
kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${POD}" -- \
    sh -c "rm -f '${REMOTE_FILE}'.chunk.*" 2>/dev/null || true

echo "Transfer complete!"
