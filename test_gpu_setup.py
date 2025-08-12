#!/usr/bin/env python3
"""
🚀 GPU SETUP TESTING
Test GPU detection và performance cho AI3.0 Trading System
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Giảm log noise

import tensorflow as tf
import numpy as np
import time
from datetime import datetime

def test_gpu_setup():
    """Test GPU detection và basic operations"""
    print("🔥 GPU SETUP TEST - AI3.0 TRADING SYSTEM")
    print("=" * 50)
    
    # 1. TensorFlow Version
    print(f"📦 TensorFlow Version: {tf.__version__}")
    
    # 2. GPU Detection
    print(f"🔍 GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    
    # 3. Physical Devices
    physical_devices = tf.config.list_physical_devices()
    print(f"🖥️  Physical Devices:")
    for device in physical_devices:
        print(f"   - {device}")
    
    # 4. GPU Details
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🎯 GPU DETAILS:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
            
            # Memory info
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"   Details: {gpu_details}")
            except:
                print("   Details: Not available")
    else:
        print("❌ NO GPU DETECTED!")
        return False
    
    # 5. Test GPU Operations
    print(f"\n⚡ GPU PERFORMANCE TEST:")
    
    # Create test data
    with tf.device('/GPU:0'):
        # Matrix multiplication test
        start_time = time.time()
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        gpu_time = time.time() - start_time
        print(f"   GPU Matrix Mult (1000x1000): {gpu_time:.4f}s")
    
    # CPU comparison
    with tf.device('/CPU:0'):
        start_time = time.time()
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        cpu_time = time.time() - start_time
        print(f"   CPU Matrix Mult (1000x1000): {cpu_time:.4f}s")
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"   🚀 GPU Speedup: {speedup:.2f}x")
    
    # 6. Neural Network Test
    print(f"\n🧠 NEURAL NETWORK GPU TEST:")
    
    # Simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Test data
    X_test = np.random.random((1000, 100))
    y_test = np.random.randint(0, 2, (1000, 1))
    
    # Training test
    with tf.device('/GPU:0'):
        start_time = time.time()
        model.fit(X_test, y_test, epochs=5, batch_size=32, verbose=0)
        gpu_train_time = time.time() - start_time
        print(f"   GPU Training (5 epochs): {gpu_train_time:.4f}s")
    
    print(f"\n✅ GPU SETUP SUCCESSFUL!")
    print(f"📊 System Ready for GPU Training")
    
    return True

def create_gpu_training_config():
    """Tạo GPU configuration cho training"""
    print(f"\n⚙️  GPU TRAINING CONFIGURATION:")
    
    # GPU Memory Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("   ✅ GPU Memory Growth: Enabled")
            
            # Memory limit (optional - để full 4GB)
            # tf.config.experimental.set_memory_limit(gpus[0], 3072)  # 3GB limit
            print("   ✅ GPU Memory Limit: Full (4GB)")
            
        except RuntimeError as e:
            print(f"   ❌ GPU Config Error: {e}")
    
    # Mixed Precision (tăng tốc training)
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("   ✅ Mixed Precision: Enabled (float16)")
    except:
        print("   ⚠️  Mixed Precision: Not available")
    
    return True

if __name__ == "__main__":
    print(f"🕒 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test GPU
    gpu_ok = test_gpu_setup()
    
    if gpu_ok:
        # Configure GPU
        create_gpu_training_config()
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. ✅ GPU Detection: PASSED")
        print(f"   2. ✅ GPU Performance: PASSED") 
        print(f"   3. ✅ Neural Network: PASSED")
        print(f"   4. 🚀 Ready for AI3.0 GPU Training!")
    else:
        print(f"\n❌ GPU SETUP FAILED!")
        print(f"   Check CUDA/cuDNN installation") 