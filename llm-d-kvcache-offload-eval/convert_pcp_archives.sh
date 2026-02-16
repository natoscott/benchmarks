#!/bin/bash
# Convert all PCP archives to Parquet format for analysis

for result_dir in results/1x2xL40S_upstream-llm-d-0.4.0_Qwen3-*_{no-offload,native-offload}; do
  if [ ! -d "$result_dir" ]; then
    continue
  fi

  model=$(basename "$result_dir" | cut -d_ -f4)
  config=$(basename "$result_dir" | cut -d_ -f5)
  archive=$(find "$result_dir/pcp-archives" -name "*.0" -type f 2>/dev/null | head -1)

  if [ -z "$archive" ]; then
    echo "Warning: No PCP archive found in $result_dir"
    continue
  fi

  archive_base="${archive%.0}"

  echo "Converting: $model $config"
  pcp2arrow -a "$archive_base" -o "analysis/${model}_${config}.parquet" 2>&1 | grep -E "ERROR|Written" || echo "  Converted successfully"
done

echo ""
echo "Conversion complete. Created files:"
ls -lh analysis/*.parquet 2>/dev/null | awk '{print $9, $5}'
