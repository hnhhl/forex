#!/usr/bin/env python3
import torch
import sys

print("="*50)
print("CUDA AVAILABILITY TEST")
print("="*50)

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Test GPU tensor
    try:
        device = torch.device('cuda')
        test_tensor = torch.randn(1000, 1000).to(device)
        result = torch.matmul(test_tensor, test_tensor.T)
        print("✅ GPU Tensor Test: SUCCESS")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    except Exception as e:
        print(f"❌ GPU Tensor Test: FAILED - {e}")
else:
    print("❌ CUDA NOT AVAILABLE!")
    print("Possible reasons:")
    print("1. PyTorch CPU-only version installed")
    print("2. CUDA drivers not installed")
    print("3. CUDA version mismatch")

print("="*50) 