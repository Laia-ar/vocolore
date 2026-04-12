#!/usr/bin/env python3
"""Test script to diagnose Whisper GPU loading issues."""

import os
import sys

print("=" * 60)
print("Whisper GPU Test Script")
print("=" * 60)

# Check environment
print("\n1. Environment Variables:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

# Test PyTorch CUDA
print("\n2. PyTorch CUDA Test:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
    print(f"   Device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Try to allocate a tensor
        print("\n3. Tensor Allocation Test:")
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"   Successfully allocated tensor on GPU: {x}")
            del x
            torch.cuda.empty_cache()
            print("   Cleared CUDA cache")
        except Exception as e:
            print(f"   FAILED to allocate tensor: {e}")
    else:
        print("   WARNING: CUDA not available!")
        
except ImportError as e:
    print(f"   ERROR: Could not import torch: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ERROR: {e}")

# Test Whisper
print("\n4. Whisper Model Loading Test:")
try:
    from faster_whisper import WhisperModel
    
    model_size = "tiny"
    device = "cuda"
    compute_type = "int8_float16"
    
    print(f"   Model: {model_size}")
    print(f"   Device: {device}")
    print(f"   Compute type: {compute_type}")
    print("   Loading model...")
    
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("   SUCCESS: Model loaded on GPU!")
    
    # Test transcription
    print("\n5. Transcription Test:")
    import numpy as np
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
    segments, info = model.transcribe(test_audio, beam_size=1, language="es")
    print(f"   Detected language: {info.language}")
    print("   SUCCESS: Transcription works!")
    
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
