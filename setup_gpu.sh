#!/bin/bash
# Script to diagnose and fix NVIDIA GPU issues

echo "NVIDIA GPU Setup Script"
echo "======================="
echo ""

echo "1. Checking NVIDIA driver..."
nvidia-smi

echo ""
echo "2. Checking NVIDIA kernel modules..."
lsmod | grep nvidia

echo ""
echo "3. Checking CUDA libraries..."
ldconfig -p | grep -E "(libcuda|libcudart)"

echo ""
echo "4. Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "5. Testing PyTorch CUDA..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "If CUDA is not available, try these fixes:"
echo "1. Reboot the system: sudo reboot"
echo "2. Reload NVIDIA modules: sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm"
echo "3. Install persistence daemon: sudo apt install nvidia-persistenced && sudo systemctl enable nvidia-persistenced"
echo "4. Check for conflicting drivers: sudo lspci -k | grep -A 2 -E '(VGA|3D)'"
