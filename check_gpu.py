#!/usr/bin/env python3
import sys
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Kiểm tra GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            
        # Kiểm tra GPU có sử dụng được không
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print("✅ GPU computation test: SUCCESS")
                print(f"Result: {c.numpy()}")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
    else:
        print("❌ No GPU detected")
        
    # Kiểm tra CUDA
    print(f"CUDA built with TensorFlow: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    
except ImportError:
    print("❌ TensorFlow not installed")
except Exception as e:
    print(f"❌ Error: {e}")

# Kiểm tra các thư viện khác
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("❌ NumPy not installed")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError:
    print("❌ Pandas not installed") 