#!/bin/bash
# Capture all KV-cache configurations one by one

set -e
chmod +x scripts/capture-one-config.sh

echo "=========================================="
echo "Capturing KV-cache logs for all configs"
echo "=========================================="
echo ""

# Define configuration to capture (using printf for proper escaping)
# 0.6B configs
printf "=== 0.6B native-offload-10k ===\n"
bash scripts/capture-one-config.sh "Qwen/Qwen3-0.6B" "native-offload-10k" \
  "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
sleep 3

# 8B configs
printf "\n=== 8B no-offload ===\n"
bash scripts/capture-one-config.sh "Qwen/Qwen3-8B" "no-offload" ""
sleep 3

printf "\n=== 8B native-offload-10k ===\n"
bash scripts/capture-one-config.sh "Qwen/Qwen3-8B" "native-offload-10k" \
  "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
sleep 3

# 14B configs
printf "\n=== 14B no-offload ===\n"
bash scripts/capture-one-config.sh "Qwen/Qwen3-14B" "no-offload" ""
sleep 3

printf "\n=== 14B native-offload-10k ===\n"
bash scripts/capture-one-config.sh "Qwen/Qwen3-14B" "native-offload-10k" \
  "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
sleep 3

printf "\n=== 14B native-offload-20k ===\n"
bash scripts/capture-one-config.sh "Qwen/Qwen3-14B" "native-offload-20k" \
  "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":20000}}'"
sleep 3

# 32B-AWQ configs
printf "\n=== 32B-AWQ no-offload ===\n"
bash scripts/capture-one-config.sh "Qwen/Qwen3-32B-AWQ" "no-offload" ""
sleep 3

printf "\n=== 32B-AWQ native-offload-10k ===\n"
bash scripts/capture-one-config.sh "Qwen/Qwen3-32B-AWQ" "native-offload-10k" \
  "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":10000}}'"
sleep 3

printf "\n=== 32B-AWQ native-offload-20k ===\n"
bash scripts/capture-one-config.sh "Qwen/Qwen3-32B-AWQ" "native-offload-20k" \
  "--kv-transfer-config '{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":20000}}'"

echo ""
echo "=========================================="
echo "All configurations captured!"
echo "=========================================="
echo ""
ls -lh vllm-startup-logs/
