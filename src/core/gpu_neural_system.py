"""
GPU-Optimized Neural Network System for AI3.0
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class GPUNeuralNetworkSystem:
    """GPU-optimized Neural Network System"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.is_gpu_available = False
        self.is_active = False
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
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    self.logger.info("Mixed precision enabled")
                except:
                    self.logger.warning("Mixed precision not available")
                
                self.is_gpu_available = True
                self.logger.info(f"GPU setup completed: {len(gpus)} GPU(s)")
                print(f"   🚀 GPU Neural System: {len(gpus)} GPU(s) ready")
            else:
                self.logger.warning("No GPU available, system disabled")
                print("   ⚠️  GPU Neural System: No GPU found, disabled")
                
        except Exception as e:
            self.logger.error(f"GPU setup failed: {e}")
            print(f"   ❌ GPU Neural System: Setup failed - {e}")
    
    def initialize(self) -> bool:
        """Initialize GPU Neural System"""
        if not self.is_gpu_available:
            return False
        
        try:
            self.create_models()
            self.is_active = True
            return True
        except Exception as e:
            self.logger.error(f"GPU Neural System initialization failed: {e}")
            return False
    
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
        
        self.logger.info(f"Created {len(self.models)} GPU models")
        return True
    
    def train_gpu(self, X_train, y_train, X_val, y_val):
        """Train models on GPU"""
        if not self.is_gpu_available or not self.models:
            return {}
        
        results = {}
        
        with tf.device('/GPU:0'):
            for name, model in self.models.items():
                print(f"   🔥 Training {name.upper()} on GPU...")
                
                start_time = datetime.now()
                
                history = model.fit(
                    X_train, y_train,
                    epochs=30,
                    batch_size=64,
                    validation_data=(X_val, y_val),
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                    ]
                )
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Save model
                model_path = f'trained_models/gpu_{name}_model.keras'
                model.save(model_path)
                
                # Evaluate
                train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
                val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                
                # Store results
                results[name] = {
                    'training_time': training_time,
                    'final_loss': float(train_loss),
                    'final_accuracy': float(train_acc),
                    'final_val_loss': float(val_loss),
                    'final_val_accuracy': float(val_acc),
                    'epochs': len(history.history['loss']),
                    'model_path': model_path
                }
                
                print(f"      ✅ {name.upper()}: {training_time:.1f}s, Acc: {train_acc:.4f}")
        
        return results
    
    def predict_gpu(self, X):
        """Make predictions using GPU"""
        if not self.is_gpu_available or not self.models:
            return {}
        
        predictions = {}
        
        with tf.device('/GPU:0'):
            for name, model in self.models.items():
                pred = model.predict(X, batch_size=64, verbose=0)
                predictions[name] = pred
        
        return predictions
    
    def get_ensemble_prediction(self, X):
        """Get ensemble prediction from all GPU models"""
        if not self.is_gpu_available or not self.models:
            return None, 0.0
        
        predictions = self.predict_gpu(X)
        
        if not predictions:
            return None, 0.0
        
        # Weighted average (equal weights for now)
        weights = {name: 1.0 for name in predictions.keys()}
        total_weight = sum(weights.values())
        
        ensemble_pred = 0.0
        for name, pred in predictions.items():
            weight = weights[name] / total_weight
            ensemble_pred += weight * pred[0][0]  # Assuming single prediction
        
        # Calculate confidence based on agreement
        pred_values = [pred[0][0] for pred in predictions.values()]
        confidence = 1.0 - (np.std(pred_values) * 2)  # Simple confidence measure
        confidence = max(0.0, min(1.0, confidence))
        
        return ensemble_pred, confidence
    
    def cleanup(self) -> bool:
        """Cleanup GPU resources"""
        try:
            if self.is_gpu_available:
                tf.keras.backend.clear_session()
                
                # Clear GPU memory
                if hasattr(tf.config.experimental, 'reset_memory_stats'):
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    for gpu in gpus:
                        try:
                            tf.config.experimental.reset_memory_stats(gpu)
                        except:
                            pass
            
            self.models.clear()
            self.is_active = False
            
            return True
            
        except Exception as e:
            self.logger.error(f"GPU cleanup failed: {e}")
            return False 