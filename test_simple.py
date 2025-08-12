#!/usr/bin/env python3
print("=== PYTHON TEST START ===")

try:
    print("1. Basic imports...")
    import os
    import sys
    import pandas as pd
    import numpy as np
    print("   ✅ Basic imports OK")
    
    print("2. PyTorch import...")
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    print("3. Data loading test...")
    df = pd.read_csv("data/working_free_data/XAUUSD_M1_realistic.csv", nrows=1000)
    print(f"   ✅ Data loaded: {len(df)} rows")
    print(f"   ✅ Columns: {list(df.columns)}")
    
    print("4. GPU tensor test...")
    if torch.cuda.is_available():
        x = torch.randn(100, 10).cuda()
        print(f"   ✅ GPU tensor: {x.device}")
        print(f"   ✅ GPU memory used: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
    
    print("=== ALL TESTS PASSED ===")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc() 