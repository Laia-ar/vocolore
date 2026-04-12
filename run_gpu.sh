#!/bin/bash
# Run the transcription system with GPU support

cd /home/didi/code/vocolore

# Set up environment
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# Run the script
exec uv run python wifi_transcribe.py "$@"
