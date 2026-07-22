#!/bin/bash
# Generate LLMInferenceService manifests for PP validation on janus cluster.
# Priority models: Qwen3-30B-A3B (MoE) + Llama-3.3-70B-FP8 (dense large).

OUTDIR="$(dirname "$0")/../manifests/pp-validation-janus"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

NS="llm-d-nathans-epp-eval"

gen() {
    local model_name="$1" pvc_subpath="$2" short_name="$3" mem="$4"

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
  namespace: ${NS}
spec:
  baseRefs:
  - name: v3-5-0-ea-1-kserve-config-llm-template
  - name: epp-scheduler-prior-default
  - name: v3-5-0-ea-1-kserve-config-llm-router-route
  model:
    name: "${model_name}"
    uri: "pvc://model-storage-nfs/${pvc_subpath}"
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

# Priority 1: MoE model — AIC predicts PP2=1.70x
gen "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    "hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/main" \
    "qwen3-30b-moe" "60Gi"

# Priority 2: Dense large — AIC predicts PP2~1.15x
gen "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" \
    "hub/models--RedHatAI--Llama-3.3-70B-Instruct-FP8-dynamic/snapshots/main" \
    "llama-70b-fp8" "120Gi"
