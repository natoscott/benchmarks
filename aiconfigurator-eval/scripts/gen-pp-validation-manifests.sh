#!/bin/bash
# Generate LLMInferenceService manifests for PP validation.
# 4 models x 4 TP/PP configs = 16 manifests, all using 8 GPUs / 1 replica.

OUTDIR="$(dirname "$0")/../manifests/pp-validation"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

gen() {
    local model_name="$1" pvc_path="$2" short_name="$3" mem="$4"
    local model_uri="pvc://model-storage-pvc/${pvc_path}"

    for tp in 8 4 2 1; do
        pp=$((8 / tp))
        svc_name="${short_name}-tp${tp}pp${pp}"
        filename="${OUTDIR}/${svc_name}.yaml"

        cat > "$filename" <<YAML
# PP validation: ${model_name} TP=${tp}/PP=${pp} (1 replica x 8 GPUs)
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: ${svc_name}
  namespace: aiconfigurator
spec:
  baseRefs:
  - name: v3-5-0-ea-1-kserve-config-llm-template
  - name: v3-5-0-ea-1-kserve-config-llm-scheduler
  - name: v3-5-0-ea-1-kserve-config-llm-router-route
  model:
    name: "${model_name}"
    uri: "${model_uri}"
  parallelism:
    tensor: ${tp}
    pipeline: ${pp}
  replicas: 1
  template:
    containers:
    - name: main
      env:
      - name: HUGGING_FACE_HUB_TOKEN
        valueFrom:
          secretKeyRef:
            name: llm-d-hf-token
            key: HF_TOKEN
      - name: VLLM_ADDITIONAL_ARGS
        value: >-
          --tensor-parallel-size ${tp}
          --pipeline-parallel-size ${pp}
          --gpu-memory-utilization 0.90
          --max-model-len 8000
          --max-num-seqs 256
      resources:
        requests:
          nvidia.com/gpu: "8"
          memory: ${mem}
        limits:
          nvidia.com/gpu: "8"
YAML
        echo "Created: ${svc_name}.yaml"
    done
}

gen "Qwen/Qwen3-0.6B"      "Qwen__Qwen3-0.6B"      "qwen3-06b"       "20Gi"
gen "Qwen/Qwen3-8B"        "Qwen__Qwen3-8B"        "qwen3-8b"        "40Gi"
gen "Qwen/Qwen3-14B"       "Qwen__Qwen3-14B"       "qwen3-14b"       "60Gi"
gen "Qwen/Qwen3-32B-FP8"   "Qwen__Qwen3-32B-FP8"   "qwen3-32b-fp8"   "80Gi"
