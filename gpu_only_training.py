#!/usr/bin/env python3
"""
GPU ONLY Training - Chi chay GPU models
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Force GPU environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import json
from pathlib import Path

def create_gpu_models(input_size):
    """Tạo nhiều GPU models khác nhau"""
    
    # Model 1: Dense
    class DenseModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Model 2: LSTM
    class LSTMModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
            self.dropout1 = nn.Dropout(0.3)
            self.lstm2 = nn.LSTM(128, 64, batch_first=True)
            self.dropout2 = nn.Dropout(0.2)
            self.fc = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = x.unsqueeze(1)
            x, _ = self.lstm1(x)
            x = self.dropout1(x)
            x, _ = self.lstm2(x)
            x = self.dropout2(x[:, -1, :])
            x = self.sigmoid(self.fc(x))
            return x
    
    # Model 3: CNN
    class CNNModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.global_pool = nn.AdaptiveMaxPool1d(1)
            self.fc1 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = x.unsqueeze(1)
            x = torch.relu(self.conv1(x))
            x = self.pool1(x)
            x = torch.relu(self.conv2(x))
            x = self.global_pool(x).squeeze(-1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.sigmoid(self.fc2(x))
            return x
    
    return [
        ("Dense", DenseModel(input_size)),
        ("LSTM", LSTMModel(input_size)),
        ("CNN", CNNModel(input_size))
    ]

def train_gpu_model(model, name, X_train, y_train, X_val, y_val, epochs=20):
    """Train single GPU model"""
    print(f"   🔥 Training {name} on GPU...")
    
    model = model.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 1024  # Large batch for GPU
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size].unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_acc = ((val_outputs > 0.5).float().squeeze() == y_val).float().mean().item()
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        if epoch % 5 == 0:
            print(f"     Epoch {epoch}: Loss={total_loss/len(X_train)*batch_size:.4f}, Val_Acc={val_acc:.4f}")
    
    print(f"     ✅ {name} Final Accuracy: {best_acc:.4f}")
    
    # Save model
    model_path = f"trained_models/gpu_{name.lower()}_{datetime.now().strftime('%H%M%S')}.pth"
    torch.save(model.state_dict(), model_path)
    
    return best_acc

def gpu_only_training():
    print("="*80)
    print("🔥 GPU ONLY TRAINING - ULTIMATE XAU SYSTEM")
    print("="*80)
    
    # GPU Check
    if not torch.cuda.is_available():
        print("❌ GPU not available!")
        return False
    
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    # Load data
    print("\n📊 LOADING XAU/USD DATA:")
    try:
        df = pd.read_csv("data/working_free_data/XAUUSD_M1_realistic.csv")
        print(f"   ✅ Loaded: {len(df):,} records")
        
        # Feature engineering
        df['price_change'] = df['Close'] - df['Open']
        df['volatility'] = df['High'] - df['Low']
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['volume_ma'] = df['Volume'].rolling(10).mean()
        df['price_ma'] = df['Close'].rolling(10).mean()
        
        # Technical indicators
        df['rsi'] = df['Close'].rolling(14).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / x.diff().clip(upper=0).abs().mean()))))
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'price_change', 'volatility', 'price_position', 'volume_ma', 'price_ma', 'rsi']
        X = df[features].fillna(0).values
        y = (df['Close'].shift(-1) > df['Close']).astype(int).fillna(0).values
        
        X, y = X[:-1], y[:-1]  # Remove last row
        
        print(f"   ✅ Features: {X.shape[1]}")
        print(f"   ✅ Samples: {X.shape[0]:,}")
        
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False
    
    # Prepare GPU data
    print("\n🔧 PREPARING GPU DATA:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Convert to GPU tensors
    X_train_gpu = torch.FloatTensor(X_train).cuda()
    y_train_gpu = torch.FloatTensor(y_train).cuda()
    X_val_gpu = torch.FloatTensor(X_val).cuda()
    y_val_gpu = torch.FloatTensor(y_val).cuda()
    
    print(f"   ✅ Train: {X_train_gpu.shape[0]:,} samples")
    print(f"   ✅ Validation: {X_val_gpu.shape[0]:,} samples")
    print(f"   ✅ GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
    
    # Create and train models
    print("\n🤖 TRAINING GPU MODELS:")
    models = create_gpu_models(X_train_gpu.shape[1])
    
    results = []
    for name, model in models:
        start_time = time.time()
        accuracy = train_gpu_model(model, name, X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu)
        training_time = time.time() - start_time
        
        results.append({
            'model': name,
            'accuracy': accuracy,
            'training_time': training_time,
            'gpu_memory': torch.cuda.memory_allocated(0) / 1024**3
        })
        
        print(f"   ⏱️ {name} Training Time: {training_time:.1f}s")
        print(f"   🔥 GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        print()
    
    # Save results
    results_file = f"training_results/gpu_only_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'gpu_info': {
                'name': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory // 1024**3
            },
            'data_info': {
                'total_samples': len(X),
                'features': X.shape[1],
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            },
            'results': results
        }, indent=2)
    
    print("🎉 GPU TRAINING COMPLETED:")
    print(f"   ✅ Models trained: {len(results)}")
    print(f"   ✅ Results saved: {results_file}")
    print(f"   ✅ Final GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
    
    return True

if __name__ == "__main__":
    success = gpu_only_training()
    
    if success:
        print("\n🚀 GPU-only training completed successfully!")
    else:
        print("\n❌ GPU-only training failed!")
        sys.exit(1) 