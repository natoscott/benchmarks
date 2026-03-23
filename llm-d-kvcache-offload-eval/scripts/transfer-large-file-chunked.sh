#!/bin/bash
# Chunked file transfer using base64 encoding for reliability over kubectl exec.
# kubectl exec truncates large binary streams unpredictably; base64 (ASCII) is safe.

set -euo pipefail

KUBECONFIG="${1:?KUBECONFIG required}"
NAMESPACE="${2:?NAMESPACE required}"
POD="${3:?POD required}"
REMOTE_FILE="${4:?REMOTE_FILE required}"
LOCAL_FILE="${5:?LOCAL_FILE required}"
CHUNK_SIZE_MB="${6:-5}"  # Default 5MB chunks

echo "Chunked file transfer (base64):"
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

# Get list of chunks with their sizes
CHUNK_INFO=$(kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${POD}" -- \
    sh -c "stat -c '%n %s' '${REMOTE_FILE}'.chunk.* 2>/dev/null | sort")

if [ -z "${CHUNK_INFO}" ]; then
    echo "ERROR: No chunks created"
    exit 1
fi

TOTAL_CHUNKS=$(echo "${CHUNK_INFO}" | wc -l)
echo "Transferring ${TOTAL_CHUNKS} chunks via base64..."

CHUNK_NUM=0
while IFS=' ' read -r CHUNK EXPECTED_CHUNK_SIZE; do
    CHUNK_NUM=$((CHUNK_NUM + 1))
    CHUNK_NAME=$(basename "${CHUNK}")
    echo "  Chunk ${CHUNK_NUM}/${TOTAL_CHUNKS}: ${CHUNK_NAME} (${EXPECTED_CHUNK_SIZE} bytes)"

    # Transfer with retry: base64-encode in pod, decode locally.
    # base64 produces ASCII text, avoiding binary stream truncation in kubectl exec.
    SUCCESS=0
    for attempt in 1 2 3; do
        kubectl --kubeconfig="${KUBECONFIG}" exec -n "${NAMESPACE}" "${POD}" -- \
            base64 "${CHUNK}" | base64 -d > "${CHUNK_DIR}/${CHUNK_NAME}" 2>/dev/null || true

        LOCAL_CHUNK_SIZE=$(stat -c '%s' "${CHUNK_DIR}/${CHUNK_NAME}" 2>/dev/null || echo 0)
        if [ "${LOCAL_CHUNK_SIZE}" -eq "${EXPECTED_CHUNK_SIZE}" ]; then
            SUCCESS=1
            break
        else
            echo "    Size mismatch (got ${LOCAL_CHUNK_SIZE}, expected ${EXPECTED_CHUNK_SIZE}), retry ${attempt}/3..."
            sleep 2
        fi
    done

    if [ ${SUCCESS} -eq 0 ]; then
        echo "ERROR: Failed to transfer chunk ${CHUNK_NAME} after 3 attempts"
        exit 1
    fi
done <<< "${CHUNK_INFO}"

# Reassemble chunks locally
echo "Reassembling file..."
cat "${CHUNK_DIR}"/*.chunk.* > "${LOCAL_FILE}"

# Verify final file size
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
