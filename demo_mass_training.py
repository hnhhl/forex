#!/usr/bin/env python3
"""
Demo Mass Training System AI3.0
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

def load_xauusd_data():
    """Load XAUUSD data"""
    print("📊 Loading XAUUSD data...")
    
    path = "data/working_free_data/XAUUSD_H1_realistic.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # Feature engineering
        df['returns'] = df['Close'].pct_change()
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['volatility'] = df['returns'].rolling(10).std()
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df = df.dropna()
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'ma_5', 'ma_20', 'volatility']
        X = df[features].values
        y = df['target'].values
        
        print(f"   ✅ Loaded {len(X)} samples")
        return X, y
    
    # Synthetic data
    print("   🔄 Using synthetic data")
    np.random.seed(42)
    X = np.random.randn(1000, 9)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def demo_training():
    """Demo training với một vài models"""
    print("\n🏋️ DEMO TRAINING")
    print("-" * 20)
    
    # Load data
    X, y = load_xauusd_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Train models
    models = {}
    
    # Random Forest
    print("\n🌲 Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    start = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - start
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    models['RandomForest'] = {'accuracy': rf_acc, 'time': rf_time}
    print(f"   Accuracy: {rf_acc:.4f} ({rf_time:.2f}s)")
    
    # Logistic Regression
    print("\n📈 Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    start = time.time()
    lr.fit(X_train_scaled, y_train)
    lr_time = time.time() - start
    lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
    models['LogisticRegression'] = {'accuracy': lr_acc, 'time': lr_time}
    print(f"   Accuracy: {lr_acc:.4f} ({lr_time:.2f}s)")
    
    return models

def main():
    print("🚀 MASS TRAINING DEMO")
    print("=" * 30)
    
    models = demo_training()
    
    print("\n📊 RESULTS:")
    for name, result in models.items():
        print(f"   {name}: {result['accuracy']:.4f}")
    
    print("\n🎯 MASS TRAINING SYSTEM CAN:")
    print("   • Train 50+ models simultaneously")
    print("   • Parallel processing")
    print("   • Auto ensemble creation")
    print("   • Real-time monitoring")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main() 