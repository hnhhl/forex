#!/usr/bin/env python3
"""
GPU-Optimized Training Script for AI 3.0
Tối ưu hóa training với GPU NVIDIA RTX 4070
"""

import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Kiểm tra GPU
print("🚀 AI 3.0 GPU-OPTIMIZED TRAINING STARTED")
print("=" * 50)

# PyTorch GPU Check
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = torch.device('cuda:0')
    print(f"Using device: {device}")
else:
    device = torch.device('cpu')
    print("⚠️ CUDA not available, using CPU")

# TensorFlow GPU Check
print(f"\nTensorFlow version: {tf.__version__}")
tf_gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPU devices: {tf_gpus}")

# Thiết lập GPU memory growth cho TensorFlow
if tf_gpus:
    try:
        for gpu in tf_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ TensorFlow GPU memory growth enabled")
    except:
        print("❌ Failed to set TensorFlow GPU memory growth")

def load_data():
    """Load và preprocess data"""
    print("\n📊 Loading training data...")
    
    # Tìm file data có sẵn
    data_paths = [
        "data/working_free_data/XAUUSD_H1_realistic.csv",
        "data/real_free_data/XAUUSD_D1_forexsb.csv",
        "data/maximum_mt5_v2/XAUUSDc_H1_20250618_115847.csv"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"📁 Using data: {path}")
            df = pd.read_csv(path)
            break
    else:
        print("❌ No data file found, generating sample data...")
        # Generate sample data
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(1800, 2100, len(dates)),
            'high': np.random.uniform(1800, 2100, len(dates)),
            'low': np.random.uniform(1800, 2100, len(dates)),
            'close': np.random.uniform(1800, 2100, len(dates)),
            'volume': np.random.randint(100, 1000, len(dates))
        })
    
    print(f"✅ Data loaded: {len(df)} rows")
    return df

def create_features(df):
    """Tạo features cho training"""
    print("\n🔧 Creating features...")
    print(f"Columns: {list(df.columns)}")
    
    # Handle different column names
    if 'Close' in df.columns:
        df['close'] = df['Close']
        df['open'] = df['Open'] if 'Open' in df.columns else df['close']
        df['high'] = df['High'] if 'High' in df.columns else df['close']
        df['low'] = df['Low'] if 'Low' in df.columns else df['close']
        df['volume'] = df['Volume'] if 'Volume' in df.columns else 1000
    
    # Technical indicators
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'] = calculate_macd(df['close'])
    
    # Price changes
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Target: Future price direction
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Remove NaN
    df = df.dropna()
    
    features = ['open', 'high', 'low', 'close', 'volume', 
               'sma_10', 'sma_20', 'rsi', 'macd', 'price_change', 'high_low_ratio']
    
    X = df[features].values
    y = df['target'].values
    
    print(f"✅ Features created: {X.shape}, Target: {y.shape}")
    return X, y

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    return ema_fast - ema_slow

class PyTorchNeuralNetwork(nn.Module):
    """GPU-Optimized Neural Network"""
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, 1))
        self.layers.append(nn.Sigmoid())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_pytorch_model(X, y, device):
    """Train PyTorch model on GPU"""
    print("\n🔥 Training PyTorch model on GPU...")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(device)
    
    # Create model
    model = PyTorchNeuralNetwork(X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 100
    batch_size = 1024
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(X_tensor)*batch_size:.4f}")
    
    return model

def create_tensorflow_gpu_model(input_shape):
    """Create TensorFlow model optimized for GPU"""
    print("\n🧠 Creating TensorFlow GPU model...")
    
    with tf.device('/GPU:0' if tf_gpus else '/CPU:0'):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def train_tensorflow_model(model, X, y):
    """Train TensorFlow model"""
    print("\n🚀 Training TensorFlow model...")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train model
    with tf.device('/GPU:0' if tf_gpus else '/CPU:0'):
        history = model.fit(
            X_train, y_train,
            batch_size=1024,
            epochs=50,
            validation_data=(X_val, y_val),
            verbose=1
        )
    
    return model, history

def main():
    """Main training function"""
    start_time = time.time()
    
    # Load data
    df = load_data()
    X, y = create_features(df)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'gpu_info': {
            'pytorch_cuda': torch.cuda.is_available(),
            'tensorflow_gpu': len(tf_gpus) > 0,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        },
        'models_trained': []
    }
    
    # Train PyTorch model
    if torch.cuda.is_available():
        pytorch_model = train_pytorch_model(X_scaled, y, device)
        
        # Save PyTorch model
        torch.save(pytorch_model.state_dict(), 'trained_models/gpu_optimized_pytorch.pth')
        results['models_trained'].append('gpu_optimized_pytorch.pth')
        print("✅ PyTorch model saved")
    
    # Train TensorFlow model
    tf_model = create_tensorflow_gpu_model(X_scaled.shape[1])
    tf_model, history = train_tensorflow_model(tf_model, X_scaled, y)
    
    # Save TensorFlow model
    tf_model.save('trained_models/gpu_optimized_tensorflow.keras')
    results['models_trained'].append('gpu_optimized_tensorflow.keras')
    print("✅ TensorFlow model saved")
    
    # Save results
    results['training_time_seconds'] = time.time() - start_time
    results['data_size'] = len(X)
    results['features_count'] = X.shape[1]
    
    with open(f'training_results/gpu_optimized_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {results['training_time_seconds']:.2f} seconds")
    print(f"📊 Models trained: {len(results['models_trained'])}")
    print(f"💾 Results saved to training_results/")

if __name__ == "__main__":
    main() 