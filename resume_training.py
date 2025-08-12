# -*- coding: utf-8 -*-
"""Resume Training System - Khôi phục training khi bị ngắt"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import json
import glob
import warnings
warnings.filterwarnings('ignore')

class ResumeTrainingSystem:
    """System để resume training từ checkpoint"""
    
    def __init__(self):
        self.checkpoint_dir = 'training_checkpoints'
        self.models_dir = 'trained_models'
        self.results_dir = 'training_results'
        self.ensure_directories()
    
    def ensure_directories(self):
        """Tạo các thư mục cần thiết"""
        for directory in [self.checkpoint_dir, self.models_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def find_latest_checkpoint(self):
        """Tìm checkpoint mới nhất"""
        try:
            # Tìm các checkpoint files
            checkpoint_pattern = f"{self.checkpoint_dir}/checkpoint_*.json"
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            if not checkpoint_files:
                print("📍 No previous checkpoints found - starting fresh training")
                return None
            
            # Sắp xếp theo thời gian tạo
            checkpoint_files.sort(key=os.path.getctime, reverse=True)
            latest_checkpoint = checkpoint_files[0]
            
            print(f"🔄 Found latest checkpoint: {latest_checkpoint}")
            
            # Load checkpoint info
            with open(latest_checkpoint, 'r') as f:
                checkpoint_info = json.load(f)
            
            return checkpoint_info
            
        except Exception as e:
            print(f"❌ Error finding checkpoint: {e}")
            return None
    
    def save_checkpoint(self, epoch, step, model_states, training_info):
        """Lưu checkpoint"""
        try:
            checkpoint_info = {
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch,
                'step': step,
                'model_states': model_states,
                'training_info': training_info,
                'data_info': {
                    'total_sequences': training_info.get('total_sequences', 0),
                    'features_shape': training_info.get('features_shape', []),
                    'scaler_file': f"{self.checkpoint_dir}/scaler_epoch_{epoch}.pkl"
                }
            }
            
            # Save checkpoint info
            checkpoint_file = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}_step_{step}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            
            print(f"💾 Checkpoint saved: {checkpoint_file}")
            return checkpoint_file
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_info):
        """Load checkpoint và khôi phục training state"""
        try:
            print(f"🔄 Loading checkpoint from epoch {checkpoint_info['epoch']}, step {checkpoint_info['step']}")
            
            # Load models nếu có
            loaded_models = {}
            for model_name, model_file in checkpoint_info['model_states'].items():
                if os.path.exists(model_file):
                    try:
                        model = load_model(model_file)
                        loaded_models[model_name] = model
                        print(f"✅ Loaded {model_name} from {model_file}")
                    except Exception as e:
                        print(f"⚠️ Could not load {model_name}: {e}")
                        loaded_models[model_name] = None
                else:
                    print(f"⚠️ Model file not found: {model_file}")
                    loaded_models[model_name] = None
            
            return {
                'epoch': checkpoint_info['epoch'],
                'step': checkpoint_info['step'],
                'models': loaded_models,
                'training_info': checkpoint_info['training_info']
            }
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return None
    
    def resume_full_data_training(self):
        """Resume full data training từ checkpoint"""
        print("🔄 RESUME FULL DATA TRAINING")
        print("="*60)
        
        # Check for existing checkpoint
        checkpoint_info = self.find_latest_checkpoint()
        
        # Load data
        print(f"📁 Loading dataset...")
        try:
            data_path = 'data/working_free_data/XAUUSD_M1_realistic.csv'
            data = pd.read_csv(data_path)
            print(f"✅ Loaded {len(data):,} records")
            
            # Prepare features (same as before)
            features = self.prepare_features(data)
            X, y = self.create_sequences(features)
            
            print(f"✅ Prepared {len(X):,} training sequences")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Resume or start training
        if checkpoint_info:
            print(f"🔄 Resuming from checkpoint...")
            checkpoint_data = self.load_checkpoint(checkpoint_info)
            
            if checkpoint_data:
                start_epoch = checkpoint_data['epoch']
                models = checkpoint_data['models']
                
                # Continue training from checkpoint
                self.continue_training(models, X_train, y_train, X_val, y_val, start_epoch)
            else:
                print(f"⚠️ Could not load checkpoint, starting fresh")
                self.start_fresh_training(X_train, y_train, X_val, y_val)
        else:
            print(f"🆕 Starting fresh training...")
            self.start_fresh_training(X_train, y_train, X_val, y_val)
    
    def prepare_features(self, data):
        """Chuẩn bị features (same as ultimate_training_full_data.py)"""
        # Create datetime column
        if 'Date' in data.columns and 'Time' in data.columns:
            data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
        
        # Standardize column names
        column_mapping = {}
        for col in data.columns:
            if col.lower() in ['open', 'high', 'low', 'close']:
                column_mapping[col] = col.lower()
        
        data = data.rename(columns=column_mapping)
        
        # Handle volume
        if 'Volume' in data.columns:
            data['volume'] = data['Volume']
        else:
            data['volume'] = (data['High'] - data['Low']) * 1000
        
        # Select features
        feature_cols = []
        for required_col in ['open', 'high', 'low', 'close']:
            found_col = None
            for col in data.columns:
                if col.lower() == required_col or col.title() == required_col.title():
                    found_col = col
                    break
            if found_col:
                feature_cols.append(found_col)
        
        feature_cols.append('volume')
        features = data[feature_cols].copy()
        features = features.dropna()
        
        return features
    
    def create_sequences(self, features, sequence_length=60):
        """Tạo sequences cho training"""
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            current_close = features_scaled[i-1, 3]
            next_close = features_scaled[i, 3]
            y.append(1 if next_close > current_close else 0)
        
        return np.array(X), np.array(y)
    
    def start_fresh_training(self, X_train, y_train, X_val, y_val):
        """Bắt đầu training từ đầu với checkpoint system"""
        print(f"🆕 Starting fresh training with checkpoint system...")
        
        models_to_train = {
            'LSTM': self.create_lstm_model,
            'CNN': self.create_cnn_model,
            'Dense': self.create_dense_model
        }
        
        for model_name, model_func in models_to_train.items():
            print(f"\n🔥 Training {model_name} with checkpoints...")
            
            model = model_func(input_shape=(60, 5))
            
            # Enhanced callbacks with checkpointing
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7, monitor='val_loss'),
                ModelCheckpoint(
                    f'{self.models_dir}/{model_name.lower()}_checkpoint_{{epoch:02d}}.keras',
                    save_best_only=False,  # Save every epoch for resume capability
                    monitor='val_loss',
                    save_freq='epoch'
                ),
                CheckpointCallback(self, model_name)  # Custom callback
            ]
            
            # Train with checkpointing
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=200,
                    batch_size=64,
                    callbacks=callbacks,
                    verbose=1
                )
                
                print(f"✅ {model_name} training completed")
                
            except KeyboardInterrupt:
                print(f"⚠️ Training interrupted for {model_name}")
                # Save current state
                model.save(f'{self.models_dir}/{model_name.lower()}_interrupted.keras')
                print(f"💾 Model saved before interruption")
                break
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue
    
    def continue_training(self, models, X_train, y_train, X_val, y_val, start_epoch):
        """Tiếp tục training từ checkpoint"""
        print(f"▶️ Continuing training from epoch {start_epoch}...")
        
        for model_name, model in models.items():
            if model is None:
                print(f"⚠️ Skipping {model_name} - model not loaded")
                continue
            
            print(f"\n🔄 Resuming {model_name} training...")
            
            # Calculate remaining epochs
            remaining_epochs = max(0, 200 - start_epoch)
            
            if remaining_epochs <= 0:
                print(f"✅ {model_name} training already completed")
                continue
            
            # Enhanced callbacks
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7, monitor='val_loss'),
                ModelCheckpoint(
                    f'{self.models_dir}/{model_name.lower()}_resumed_{{epoch:02d}}.keras',
                    save_best_only=False,
                    monitor='val_loss',
                    save_freq='epoch'
                ),
                CheckpointCallback(self, model_name, start_epoch)
            ]
            
            try:
                # Resume training
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=remaining_epochs,
                    initial_epoch=start_epoch,  # Important: start from checkpoint epoch
                    batch_size=64,
                    callbacks=callbacks,
                    verbose=1
                )
                
                print(f"✅ {model_name} resumed training completed")
                
            except KeyboardInterrupt:
                print(f"⚠️ Training interrupted again for {model_name}")
                model.save(f'{self.models_dir}/{model_name.lower()}_interrupted_again.keras')
                break
            except Exception as e:
                print(f"❌ Error resuming {model_name}: {e}")
                continue
    
    def create_lstm_model(self, input_shape):
        """Create LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def create_cnn_model(self, input_shape):
        """Create CNN model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def create_dense_model(self, input_shape):
        """Create Dense model"""
        model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

class CheckpointCallback(tf.keras.callbacks.Callback):
    """Custom callback để save checkpoint mỗi epoch"""
    
    def __init__(self, resume_system, model_name, start_epoch=0):
        super().__init__()
        self.resume_system = resume_system
        self.model_name = model_name
        self.start_epoch = start_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint sau mỗi epoch"""
        try:
            actual_epoch = epoch + self.start_epoch
            
            # Save model
            model_file = f"{self.resume_system.models_dir}/{self.model_name.lower()}_epoch_{actual_epoch:02d}.keras"
            self.model.save(model_file)
            
            # Save checkpoint info
            model_states = {self.model_name: model_file}
            training_info = {
                'model_name': self.model_name,
                'total_sequences': 'unknown',
                'features_shape': [60, 5],
                'logs': logs or {}
            }
            
            self.resume_system.save_checkpoint(
                epoch=actual_epoch,
                step=0,  # We don't track steps in this callback
                model_states=model_states,
                training_info=training_info
            )
            
        except Exception as e:
            print(f"⚠️ Error in checkpoint callback: {e}")

def main():
    """Main function"""
    print("🔄 RESUME TRAINING SYSTEM")
    print("="*60)
    
    resume_system = ResumeTrainingSystem()
    
    # Check for interrupted training
    checkpoint = resume_system.find_latest_checkpoint()
    
    if checkpoint:
        print(f"🔍 Found interrupted training:")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Step: {checkpoint['step']}")
        print(f"   Timestamp: {checkpoint['timestamp']}")
        
        choice = input("\n❓ Do you want to resume from checkpoint? (y/n): ").lower().strip()
        
        if choice == 'y':
            resume_system.resume_full_data_training()
        else:
            print("🆕 Starting fresh training...")
            resume_system.start_fresh_training(*resume_system.load_and_prepare_data())
    else:
        print("🆕 No checkpoint found, starting fresh training...")
        resume_system.resume_full_data_training()

if __name__ == "__main__":
    main() 