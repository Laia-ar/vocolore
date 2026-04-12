#!/bin/bash
# Fix CUDA initialization issue

echo "Fixing CUDA..."

# Unset problematic environment variables
unset CUDA_VISIBLE_DEVICES

# Set library path to use PyTorch's bundled CUDA
export LD_LIBRARY_PATH="/home/didi/code/vocolore/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Test CUDA
cd /home/didi/code/vocolore
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    x = torch.tensor([1.0]).cuda()
    print(f'Test tensor: {x}')
    print('SUCCESS!')
else:
    print('FAILED - CUDA not available')
"
