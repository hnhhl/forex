#!/usr/bin/env python3
"""
🚀 GPU INTEGRATION FIX - AI3.0 TRADING SYSTEM
Tích hợp GPU training vào core system và fix TensorFlow conflicts
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
import json

def fix_tensorflow_conflicts():
    """Fix TensorFlow version conflicts"""
    print("🔧 FIXING TENSORFLOW CONFLICTS...")
    print("=" * 50)
    
    try:
        # Uninstall conflicting TensorFlow CPU version
        print("1. Removing TensorFlow CPU 2.19.0...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'uninstall', 'tensorflow', '-y'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ TensorFlow CPU removed")
        else:
            print(f"   ⚠️  Warning: {result.stderr}")
        
        # Keep only TensorFlow-GPU 2.10.0
        print("2. Verifying TensorFlow-GPU 2.10.0...")
        result = subprocess.run([
            sys.executable, '-c', 'import tensorflow as tf; print(f"TF Version: {tf.__version__}"); print(f"GPU Available: {len(tf.config.list_physical_devices(\"GPU\"))}")'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ TensorFlow-GPU verified")
            print(f"   {result.stdout.strip()}")
        else:
            print(f"   ❌ Error: {result.stderr}")
            return False
        
        print("✅ TensorFlow conflicts resolved!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing TensorFlow: {e}")
        return False

def create_gpu_neural_system():
    """Tạo GPU-optimized Neural Network System"""
    print("\n🧠 CREATING GPU-OPTIMIZED NEURAL SYSTEM...")
    print("=" * 50)
    
    gpu_neural_code = '''"""
GPU-Optimized Neural Network System for AI3.0
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class GPUNeuralNetworkSystem:
    """GPU-optimized Neural Network System"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.is_gpu_available = False
        self.setup_gpu()
    
    def setup_gpu(self):
        """Setup GPU configuration"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Mixed precision for speed
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                self.is_gpu_available = True
                self.logger.info("GPU setup completed successfully")
            else:
                self.logger.warning("No GPU available, using CPU")
                
        except Exception as e:
            self.logger.error(f"GPU setup failed: {e}")
    
    def create_models(self):
        """Create GPU-optimized models"""
        if not self.is_gpu_available:
            return False
        
        with tf.device('/GPU:0'):
            # LSTM Model
            self.models['lstm'] = tf.keras.Sequential([
                tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(60, 5)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
            ])
            
            # CNN Model
            self.models['cnn'] = tf.keras.Sequential([
                tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(60, 5)),
                tf.keras.layers.Conv1D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dense(50, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
            ])
            
            # Dense Model
            self.models['dense'] = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(60, 5)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
            ])
            
            # Compile models
            for name, model in self.models.items():
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
        
        return True
    
    def train_gpu(self, X_train, y_train, X_val, y_val):
        """Train models on GPU"""
        if not self.is_gpu_available:
            return {}
        
        results = {}
        
        with tf.device('/GPU:0'):
            for name, model in self.models.items():
                print(f"Training {name} on GPU...")
                
                history = model.fit(
                    X_train, y_train,
                    epochs=30,
                    batch_size=64,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10),
                        tf.keras.callbacks.ReduceLROnPlateau(patience=5)
                    ]
                )
                
                # Save model
                model.save(f'trained_models/gpu_{name}_model.keras')
                
                # Store results
                results[name] = {
                    'final_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                    'epochs': len(history.history['loss'])
                }
        
        return results
    
    def predict_gpu(self, X):
        """Make predictions using GPU"""
        if not self.is_gpu_available or not self.models:
            return {}
        
        predictions = {}
        
        with tf.device('/GPU:0'):
            for name, model in self.models.items():
                pred = model.predict(X, batch_size=64)
                predictions[name] = pred
        
        return predictions
'''
    
    # Save GPU Neural System
    with open('src/core/gpu_neural_system.py', 'w') as f:
        f.write(gpu_neural_code)
    
    print("✅ GPU Neural System created!")
    return True

def integrate_gpu_into_ultimate_system():
    """Tích hợp GPU system vào Ultimate XAU System"""
    print("\n🔗 INTEGRATING GPU INTO ULTIMATE SYSTEM...")
    print("=" * 50)
    
    # Read current ultimate system
    ultimate_file = 'src/core/ultimate_xau_system.py'
    
    if not os.path.exists(ultimate_file):
        print("❌ Ultimate XAU System not found!")
        return False
    
    with open(ultimate_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add GPU import
    gpu_import = '''
# GPU Neural System Import
try:
    from .gpu_neural_system import GPUNeuralNetworkSystem
    GPU_NEURAL_AVAILABLE = True
except ImportError:
    GPU_NEURAL_AVAILABLE = False
'''
    
    # Find import section and add GPU import
    import_pos = content.find('import logging')
    if import_pos != -1:
        content = content[:import_pos] + gpu_import + '\n' + content[import_pos:]
    
    # Add GPU system registration
    gpu_registration = '''
        # GPU Neural Network System (if available)
        if GPU_NEURAL_AVAILABLE:
            gpu_neural = GPUNeuralNetworkSystem(self.config)
            self.system_manager.register_system(gpu_neural)
            print("   🚀 GPU Neural Network System registered")
'''
    
    # Find neural system registration and add GPU
    neural_pos = content.find('neural_network = NeuralNetworkSystem(self.config)')
    if neural_pos != -1:
        end_pos = content.find('\n', neural_pos)
        content = content[:end_pos] + '\n' + gpu_registration + content[end_pos:]
    
    # Save modified file
    with open(ultimate_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ GPU integration completed!")
    return True

def main():
    """Main execution"""
    print("🚀 GPU INTEGRATION FIX - AI3.0 SYSTEM")
    print("=" * 70)
    print(f"🕒 Start Time: {datetime.now()}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tensorflow_fix': False,
        'gpu_system_created': False,
        'integration_completed': False
    }
    
    # Step 1: Fix TensorFlow conflicts
    results['tensorflow_fix'] = fix_tensorflow_conflicts()
    
    # Step 2: Create GPU Neural System
    if results['tensorflow_fix']:
        results['gpu_system_created'] = create_gpu_neural_system()
    
    # Step 3: Integrate into Ultimate System
    if results['gpu_system_created']:
        results['integration_completed'] = integrate_gpu_into_ultimate_system()
    
    # Save results
    with open('gpu_integration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   TensorFlow Fix: {'✅' if results['tensorflow_fix'] else '❌'}")
    print(f"   GPU System: {'✅' if results['gpu_system_created'] else '❌'}")
    print(f"   Integration: {'✅' if results['integration_completed'] else '❌'}")
    
    if all(results.values()):
        print(f"\n🎉 GPU INTEGRATION COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  Some issues remain, check logs")
    
    print(f"🕒 End Time: {datetime.now()}")

if __name__ == "__main__":
    main() 