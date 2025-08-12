#!/usr/bin/env python3
"""
Pure GPU Training Script - Chỉ sử dụng PyTorch GPU
Đảm bảo GPU được sử dụng 100%
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
import os

print("🔥 PURE GPU TRAINING - PYTORCH ONLY")
print("=" * 50)

# Force GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Device: {device}")
print(f"🎮 GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def load_and_process_data():
    """Load data with proper column handling"""
    print("\n📊 Loading data...")
    
    try:
        df = pd.read_csv("data/working_free_data/XAUUSD_H1_realistic.csv")
        print(f"✅ Data loaded: {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Create proper features
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['price_change'] = df['Close'].pct_change()
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Clean data
        df = df.dropna()
        
        # Features and target
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'sma_10', 'sma_20', 'price_change']
        X = df[features].values.astype(np.float32)
        y = df['target'].values.astype(np.float32)
        
        print(f"✅ Features: {X.shape}, Target: {y.shape}")
        return X, y
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔄 Generating synthetic data...")
        
        # Synthetic data
        n_samples = 10000
        n_features = 8
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (np.random.random(n_samples) > 0.5).astype(np.float32)
        return X, y

class GPUNeuralNetwork(nn.Module):
    """Large Neural Network for maximum GPU usage"""
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),  # Larger layers for more GPU usage
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_gpu_intensive():
    """Train multiple models simultaneously for max GPU usage"""
    print("\n🚀 Loading data...")
    X, y = load_and_process_data()
    
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to tensors and move to GPU
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(device)
    
    print(f"📊 Data on GPU: {X_tensor.device}, Shape: {X_tensor.shape}")
    
    # Train multiple models for maximum GPU usage
    models = []
    optimizers = []
    
    print("\n🔥 Creating multiple models for intensive GPU training...")
    for i in range(3):  # Train 3 models simultaneously
        model = GPUNeuralNetwork(X_tensor.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        models.append(model)
        optimizers.append(optimizer)
        print(f"✅ Model {i+1} created on GPU")
    
    criterion = nn.BCELoss()
    
    # Training parameters
    epochs = 200
    batch_size = 2048  # Large batch size for GPU
    
    print(f"\n🎯 Training {len(models)} models for {epochs} epochs")
    print(f"📦 Batch size: {batch_size}")
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_start in range(0, len(X_tensor), batch_size):
            batch_end = min(batch_start + batch_size, len(X_tensor))
            
            batch_X = X_tensor[batch_start:batch_end]
            batch_y = y_tensor[batch_start:batch_end]
            
            # Train all models on the same batch
            for i, (model, optimizer) in enumerate(zip(models, optimizers)):
                model.train()
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        if epoch % 20 == 0:
            avg_loss = total_loss / (len(models) * (len(X_tensor) // batch_size))
            print(f"🔥 Epoch {epoch}/{epochs} - Average Loss: {avg_loss:.4f}")
            
            # Force GPU activity
            with torch.no_grad():
                for model in models:
                    _ = model(X_tensor[:1000])  # Test predictions
    
    print("\n💾 Saving models...")
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'trained_models/gpu_intensive_model_{i+1}.pth')
        print(f"✅ Model {i+1} saved")
    
    return models

def gpu_stress_test():
    """Run continuous GPU stress test"""
    print("\n🔥 GPU STRESS TEST - Continuous Training")
    
    while True:
        try:
            # Generate random data for continuous training
            X = torch.randn(5000, 64, device=device)  # Large tensors
            y = torch.randint(0, 2, (5000, 1), device=device, dtype=torch.float32)
            
            # Create and train model
            model = GPUNeuralNetwork(64).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            # Intensive training
            for _ in range(100):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            print(f"🚀 GPU Stress Test - Loss: {loss.item():.4f} - GPU Temp: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping GPU stress test")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)

def main():
    """Main function"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Cannot run GPU training.")
        return
    
    print(f"🎯 Starting intensive GPU training...")
    
    # Option 1: Train real models
    models = train_gpu_intensive()
    
    # Option 2: Run stress test (uncomment if needed)
    # gpu_stress_test()
    
    print("\n🎉 GPU Training completed!")
    print("📊 Check nvidia-smi to see GPU usage")

if __name__ == "__main__":
    main() 