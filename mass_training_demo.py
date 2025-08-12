#!/usr/bin/env python3
"""
Mass Training Demo - Sử dụng dữ liệu thực từ XAUUSD
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load và prepare dữ liệu XAUUSD"""
    print("📊 Loading XAUUSD data...")
    
    # Try to find real data
    data_paths = [
        "data/working_free_data/XAUUSD_H1_realistic.csv",
        "data/working_free_data/XAUUSD_D1_realistic.csv"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"   Found: {path}")
            df = pd.read_csv(path)
            
            # Basic feature engineering
            df['returns'] = df['Close'].pct_change()
            df['ma_5'] = df['Close'].rolling(5).mean()
            df['ma_20'] = df['Close'].rolling(20).mean()
            df['ma_50'] = df['Close'].rolling(50).mean()
            
            # Volatility
            df['volatility'] = df['returns'].rolling(10).std()
            
            # Price ratios
            df['ma_ratio_5'] = df['Close'] / df['ma_5']
            df['ma_ratio_20'] = df['Close'] / df['ma_20']
            
            # Target: 1 if next close > current close
            df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            # Remove NaN
            df = df.dropna()
            
            # Features
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'returns', 'ma_5', 'ma_20', 'ma_50',
                'volatility', 'ma_ratio_5', 'ma_ratio_20'
            ]
            
            X = df[feature_cols].values
            y = df['target'].values
            
            print(f"   ✅ Loaded {len(X)} samples, {X.shape[1]} features")
            print(f"   🎯 Target distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")
            
            return X, y, path
    
    # Generate synthetic data if no real data found
    print("   🔄 No real data found, generating synthetic...")
    np.random.seed(42)
    n_samples = 5000
    n_features = 12
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    print(f"   ✅ Generated {n_samples} synthetic samples")
    return X, y, "synthetic"

def demo_simple_training(X, y):
    """Demo training với một vài models đơn giản"""
    print("\n🏋️ DEMO SIMPLE TRAINING")
    print("-" * 30)
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   📊 Training: {X_train.shape[0]} samples")
        print(f"   📊 Testing: {X_test.shape[0]} samples")
        
        # Train models
        models = {}
        results = {}
        
        # Random Forest
        print("\n   🌲 Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        start_time = time.time()
        rf.fit(X_train, y_train)
        rf_time = time.time() - start_time
        
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        models['random_forest'] = rf
        results['random_forest'] = {'accuracy': rf_acc, 'time': rf_time}
        
        print(f"      ✅ Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
        print(f"      ⏱️ Time: {rf_time:.2f}s")
        
        # Logistic Regression
        print("\n   📈 Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        start_time = time.time()
        lr.fit(X_train_scaled, y_train)
        lr_time = time.time() - start_time
        
        lr_pred = lr.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)
        
        models['logistic_regression'] = lr
        results['logistic_regression'] = {'accuracy': lr_acc, 'time': lr_time}
        
        print(f"      ✅ Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")
        print(f"      ⏱️ Time: {lr_time:.2f}s")
        
        # Simple Ensemble
        print("\n   🤝 Creating Simple Ensemble...")
        ensemble_pred = []
        for i in range(len(y_test)):
            votes = [rf_pred[i], lr_pred[i]]
            ensemble_pred.append(max(set(votes), key=votes.count))
        
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        results['ensemble'] = {'accuracy': ensemble_acc, 'time': rf_time + lr_time}
        
        print(f"      ✅ Ensemble Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
        
        return models, results
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Please install: pip install scikit-learn")
        return None, None

def demo_neural_network(X, y):
    """Demo neural network nếu TensorFlow available"""
    print("\n🧠 DEMO NEURAL NETWORK")
    print("-" * 25)
    
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   📊 Training samples: {X_train.shape[0]}")
        print(f"   🎯 Features: {X_train.shape[1]}")
        
        # Create simple neural network
        print("\n   🔧 Building Neural Network...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("   🏋️ Training Neural Network...")
        start_time = time.time()
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=30,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True
                )
            ]
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
        
        print(f"      ✅ Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"      ✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"      ⏱️ Training Time: {training_time:.2f}s")
        print(f"      📈 Epochs: {len(history.history['loss'])}")
        
        return model, test_acc, training_time
        
    except ImportError as e:
        print(f"   ❌ TensorFlow not available: {e}")
        print("   💡 Install with: pip install tensorflow")
        return None, 0, 0

def show_summary(data_source, simple_results, neural_results):
    """Show final summary"""
    print("\n" + "="*60)
    print("📊 MASS TRAINING DEMO SUMMARY")
    print("="*60)
    
    print(f"📁 Data Source: {data_source}")
    
    if simple_results:
        print("\n🔧 Traditional ML Results:")
        for model_name, result in simple_results.items():
            acc = result['accuracy']
            time_taken = result['time']
            print(f"   {model_name:20}: {acc:.4f} ({acc*100:.2f}%) - {time_taken:.2f}s")
    
    if neural_results[0] is not None:
        print(f"\n🧠 Neural Network Result:")
        print(f"   neural_network      : {neural_results[1]:.4f} ({neural_results[1]*100:.2f}%) - {neural_results[2]:.2f}s")
    
    print("\n🚀 MASS TRAINING SYSTEM CAPABILITIES:")
    print("   • Training 50+ models simultaneously")
    print("   • Parallel processing optimization")
    print("   • Auto resource management")
    print("   • Advanced ensemble creation")
    print("   • Real-time monitoring")
    
    print("\n💡 NEXT STEPS:")
    print("   1. Run: python MASS_TRAINING_SYSTEM_AI30.py")
    print("   2. Choose training mode (quick/full/production)")
    print("   3. Monitor results in training_results/")
    
    print("\n✅ Demo completed successfully!")

def main():
    """Main demo function"""
    print("🚀 MASS TRAINING SYSTEM AI3.0 - DEMO")
    print("="*60)
    
    # Load data
    X, y, data_source = load_and_prepare_data()
    
    # Demo simple training
    simple_models, simple_results = demo_simple_training(X, y)
    
    # Demo neural network
    neural_model, neural_acc, neural_time = demo_neural_network(X, y)
    
    # Show summary
    show_summary(data_source, simple_results, (neural_model, neural_acc, neural_time))

if __name__ == "__main__":
    main() 