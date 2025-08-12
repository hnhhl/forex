#!/usr/bin/env python3
import os
import sys

print("🔍 TESTING GPU SETUP...")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        
        # Test GPU tensor
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"✅ GPU tensor test: {z.shape}")
        print(f"✅ GPU memory used: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        
        print("🔥 GPU IS WORKING!")
        
    else:
        print("❌ CUDA not available!")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ PyTorch import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ GPU test error: {e}")
    sys.exit(1) 