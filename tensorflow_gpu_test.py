#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 TensorFlow 2.18 GPU Test
"""

import os
import sys
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🔥 TENSORFLOW 2.18 GPU TEST")
print("=" * 50)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # GPU detection
    print("\n📱 GPU DETECTION:")
    physical_devices = tf.config.list_physical_devices()
    print(f"🖥️  All devices: {len(physical_devices)}")
    for device in physical_devices:
        print(f"   - {device}")
    
    # Specific GPU check
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"\n🎮 GPU devices: {len(gpu_devices)}")
    for i, gpu in enumerate(gpu_devices):
        print(f"   GPU {i}: {gpu}")
        
        # GPU memory info
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"      Details: {gpu_details}")
        except Exception as e:
            print(f"      Details: Unable to get details - {e}")
    
    # Test GPU operation
    if gpu_devices:
        print("\n⚡ TESTING GPU OPERATIONS:")
        with tf.device('/GPU:0'):
            # Simple matrix multiplication
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]], name='a')
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]], name='b')
            c = tf.matmul(a, b)
            print(f"✅ Matrix multiplication result: {c.numpy()}")
            
            # Create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                tf.keras.layers.Dense(5, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy')
            print(f"✅ Simple model created: {model.count_params()} parameters")
            
            # Test with sample data
            import numpy as np
            X_test = np.random.random((100, 5))
            y_test = np.random.randint(0, 2, (100, 1))
            
            print("🏋️ Testing model training on GPU...")
            history = model.fit(X_test, y_test, epochs=5, batch_size=32, verbose=0)
            print(f"✅ Training completed! Final loss: {history.history['loss'][-1]:.4f}")
    else:
        print("❌ No GPU detected!")
    
    # Memory growth config
    if gpu_devices:
        print("\n💾 CONFIGURING GPU MEMORY GROWTH:")
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled for all GPUs")
        except RuntimeError as e:
            print(f"⚠️  Memory growth config failed: {e}")
    
    print("\n🎯 CUDA & cuDNN INFO:")
    print(f"CUDA built version: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    
    # Advanced GPU info
    if tf.test.is_gpu_available():
        print(f"GPU support: ✅ ENABLED")
        try:
            print(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
            print(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")
        except:
            print("CUDA/cuDNN version info not available")
    else:
        print(f"GPU support: ❌ DISABLED")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🏁 Test completed!")
print(f"📅 Time: {datetime.datetime.now()}") 