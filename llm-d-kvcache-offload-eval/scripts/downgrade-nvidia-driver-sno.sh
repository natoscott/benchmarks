#!/bin/bash
# Script to downgrade NVIDIA driver on SNO without draining
# This runs in a privileged pod with host access

set -e

TARGET_VERSION="550.90.07"

echo "=== NVIDIA Driver Downgrade for SNO ==="
echo "Target version: $TARGET_VERSION"

# Step 1: Stop all GPU workloads
echo ""
echo "Step 1: Scaling down GPU workloads..."
kubectl scale deployment -n llm-d-pfc-cpu llm-d-model-server --replicas=0
kubectl scale deployment -n llm-d-pfc-cpu llm-d-infpool-epp --replicas=0

# Wait for pods to terminate
echo "Waiting for pods to terminate..."
sleep 30

# Step 2: Unload nvidia kernel modules
echo ""
echo "Step 2: Unloading NVIDIA kernel modules..."
kubectl debug node/$(kubectl get nodes -o name | cut -d/ -f2) -it --image=registry.access.redhat.com/ubi9/ubi -- bash -c "
  modprobe -r nvidia_uvm || true
  modprobe -r nvidia_modeset || true
  modprobe -r nvidia_drm || true
  modprobe -r nvidia || true
"

echo ""
echo "=== Manual Steps Required ==="
echo ""
echo "The driver cannot be automatically downgraded on SNO."
echo "You need to manually unload modules and reinstall the driver on the node."
echo ""
echo "SSH to the node and run:"
echo "  sudo rmmod nvidia_uvm nvidia_modeset nvidia_drm nvidia"
echo "  sudo dnf remove -y nvidia-driver-cuda"
echo "  sudo dnf install -y nvidia-driver-cuda-$TARGET_VERSION"
echo "  sudo modprobe nvidia"
echo ""
echo "Or, consider reprovisioning the cluster with a custom MachineConfig"
echo "that installs the correct driver version during boot."
