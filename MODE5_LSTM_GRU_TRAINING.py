#!/usr/bin/env python3
"""
🔮 MODE 5.1: LSTM/GRU MODELS TRAINING
Ultimate XAU Super System V4.0

Long Short-Term Memory và Gated Recurrent Unit Networks
cho Time Series Prediction nâng cao
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import MetaTrader5 as mt5
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMGRUTrainingSystem:
    """Advanced LSTM/GRU Training System"""
    
    def __init__(self):
        self.sequence_length = 60  # 60 bars lookback
        self.features = 67  # 67 technical indicators
        self.scaler = MinMaxScaler()
        
    def explain_lstm_gru(self):
        """Giải thích LSTM/GRU"""
        print("🔮 LSTM/GRU MODELS TRAINING")
        print("=" * 60)
        print("📚 KHÁI NIỆM:")
        print("  • LSTM (Long Short-Term Memory):")
        print("    - Nhớ thông tin dài hạn trong time series")
        print("    - Xử lý vanishing gradient problem")
        print("    - 3 gates: Forget, Input, Output")
        print("    - Ideal cho forex prediction")
        print()
        print("  • GRU (Gated Recurrent Unit):")
        print("    - Simplified version của LSTM")
        print("    - 2 gates: Reset, Update")
        print("    - Faster training, ít parameters hơn")
        print("    - Good balance performance/speed")
        print()
        print("🎯 ỨNG DỤNG CHO XAU/USDc:")
        print("  • Sequence Learning: 60 bars → 1 prediction")
        print("  • Pattern Recognition: Support/Resistance levels")
        print("  • Trend Continuation: Multi-step predictions")
        print("  • Market Memory: Remember past market conditions")
        print()
        
    def create_lstm_model(self, input_shape):
        """Tạo LSTM model architecture"""
        model = Sequential([
            # LSTM Layer 1
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            # LSTM Layer 2
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # LSTM Layer 3
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense Layers
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(3, activation='softmax')  # BUY/SELL/HOLD
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def create_gru_model(self, input_shape):
        """Tạo GRU model architecture"""
        model = Sequential([
            # GRU Layer 1
            GRU(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            # GRU Layer 2
            GRU(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # GRU Layer 3
            GRU(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense Layers
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(3, activation='softmax')  # BUY/SELL/HOLD
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def prepare_sequence_data(self, data):
        """Chuẩn bị data cho sequence learning"""
        print("📊 Preparing sequence data...")
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            # 60 bars history → 1 prediction
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, -1])  # Target column
            
        X = np.array(X)
        y = np.array(y)
        
        print(f"  ✅ Sequences created: {X.shape}")
        print(f"  ✅ Input shape: (samples, timesteps, features)")
        print(f"  ✅ X shape: {X.shape}")
        print(f"  ✅ y shape: {y.shape}")
        
        return X, y
        
    def demo_training(self):
        """Demo LSTM/GRU training process"""
        print("🚀 DEMO LSTM/GRU TRAINING")
        print("=" * 50)
        
        # Simulate data (in real system, load from MT5)
        samples = 1000
        sequence_length = self.sequence_length
        features = self.features
        
        # Create mock data
        X = np.random.random((samples, sequence_length, features))
        y = np.random.randint(0, 3, (samples, 3))  # One-hot encoded
        
        print(f"📊 Training Data:")
        print(f"  • Samples: {samples}")
        print(f"  • Sequence Length: {sequence_length} bars")
        print(f"  • Features: {features} indicators")
        print(f"  • Input Shape: {X.shape}")
        print()
        
        # Create models
        print("🔮 Creating LSTM Model...")
        lstm_model = self.create_lstm_model((sequence_length, features))
        
        print("🔄 Creating GRU Model...")
        gru_model = self.create_gru_model((sequence_length, features))
        
        # Model summaries
        print("\n📋 LSTM Model Architecture:")
        lstm_model.summary()
        
        print("\n📋 GRU Model Architecture:")
        gru_model.summary()
        
        # Training simulation
        print("\n🎯 Training Results (Simulated):")
        print("  LSTM Model:")
        print("    • Epoch 1/50: loss: 0.8234 - accuracy: 0.6543")
        print("    • Epoch 25/50: loss: 0.4567 - accuracy: 0.7834")
        print("    • Epoch 50/50: loss: 0.2341 - accuracy: 0.8756")
        print("    • Final Validation Accuracy: 87.56%")
        print()
        print("  GRU Model:")
        print("    • Epoch 1/50: loss: 0.7854 - accuracy: 0.6721")
        print("    • Epoch 25/50: loss: 0.4123 - accuracy: 0.8012")
        print("    • Epoch 50/50: loss: 0.2156 - accuracy: 0.8943")
        print("    • Final Validation Accuracy: 89.43%")
        print()
        
        return lstm_model, gru_model

if __name__ == "__main__":
    system = LSTMGRUTrainingSystem()
    system.explain_lstm_gru()
    system.demo_training() 