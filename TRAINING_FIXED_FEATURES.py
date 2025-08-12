#!/usr/bin/env python3
"""
🔥 TRAINING WITH FIXED FEATURES (NO INHOMOGENEOUS SHAPE ERROR)
======================================================================
🎯 Fixed version với features có độ dài cố định
🗳️ AI2.0 Voting System
📊 Dữ liệu 3 năm realistic
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class FixedFeaturesTraining:
    def __init__(self):
        self.data_dir = "data/working_free_data"
        self.output_dir = "fixed_features_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.multi_timeframe_data = {}
        self.scaler = StandardScaler()
        
        # FIXED FEATURE DIMENSIONS
        self.FEATURES_PER_TF = {
            'M1': 8,   # Simplified M1 features
            'M5': 4,   # Higher TF features
            'M15': 4,
            'M30': 4,
            'H1': 4
        }
        self.TOTAL_FEATURES = sum(self.FEATURES_PER_TF.values())  # 24 total
        
    def load_data(self):
        """Load dữ liệu với sampling để tăng tốc"""
        print("🔥 LOADING DATA (SAMPLED FOR SPEED)")
        print("=" * 50)
        
        timeframes = ['M1', 'M5', 'M15', 'M30', 'H1']
        
        for tf in timeframes:
            csv_file = f"{self.data_dir}/XAUUSD_{tf}_realistic.csv"
            
            if os.path.exists(csv_file):
                print(f"📊 Loading {tf}...")
                
                df = pd.read_csv(csv_file)
                
                # SAMPLING để tăng tốc: Lấy mỗi 10 records
                if tf == 'M1':
                    df = df.iloc[::10].reset_index(drop=True)  # Sample M1 data
                
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
                
                self.multi_timeframe_data[tf] = df.sort_values('datetime').reset_index(drop=True)
                
                print(f"   ✅ {len(df):,} records (sampled)")
        
        return len(self.multi_timeframe_data) > 0
    
    def create_fixed_features(self):
        """Tạo features với độ dài cố định"""
        print(f"\n🔄 CREATING FIXED FEATURES ({self.TOTAL_FEATURES} per sample)")
        print("=" * 60)
        
        base_data = self.multi_timeframe_data['M1']
        step_size = 60  # Mỗi 1 giờ
        lookback = 30
        
        X = []
        y = []
        
        print(f"📊 Processing {len(base_data):,} M1 records...")
        
        for i in range(lookback, len(base_data) - 15, step_size):  # -15 for future label
            if i % 1000 == 0:
                progress = (i / len(base_data)) * 100
                print(f"   🔄 Progress: {progress:.1f}% ({i:,}/{len(base_data):,})")
            
            try:
                current_time = base_data.iloc[i]['datetime']
                
                # Create FIXED features
                features = self.create_sample_features(current_time, i)
                
                if features is not None and len(features) == self.TOTAL_FEATURES:
                    # Create label
                    label = self.create_simple_label(i)
                    
                    if label is not None:
                        X.append(features)
                        y.append(label)
                        
            except Exception as e:
                continue
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        print(f"\n✅ FEATURES CREATED:")
        print(f"   📊 Samples: {len(X):,}")
        print(f"   🎯 Features: {X.shape[1]}")
        print(f"   📈 Labels: {np.unique(y, return_counts=True)}")
        
        return X, y
    
    def create_sample_features(self, current_time, base_idx):
        """Tạo một sample features với độ dài cố định"""
        features = []
        
        try:
            # M1 features (8 features)
            m1_data = self.multi_timeframe_data['M1']
            recent_m1 = m1_data.iloc[max(0, base_idx-10):base_idx+1]
            
            if len(recent_m1) >= 3:
                current_close = recent_m1['close'].iloc[-1]
                current_open = recent_m1['open'].iloc[-1]
                
                # Price features (4)
                sma_3 = recent_m1['close'].rolling(3).mean().iloc[-1]
                sma_5 = recent_m1['close'].rolling(5).mean().iloc[-1] if len(recent_m1) >= 5 else sma_3
                
                # Momentum (2)
                price_change = (current_close - recent_m1['close'].iloc[0]) / recent_m1['close'].iloc[0] * 100
                returns = recent_m1['close'].pct_change().dropna()
                volatility = returns.std() * 100 if len(returns) > 0 else 0
                
                # Volume (2)
                avg_volume = recent_m1['volume'].mean()
                volume_ratio = recent_m1['volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
                
                m1_features = [current_close, current_open, sma_3, sma_5, 
                              price_change, volatility, avg_volume, volume_ratio]
                features.extend(m1_features)  # 8 features
            else:
                features.extend([0.0] * 8)  # Default M1 features
            
            # Higher timeframe features (4 each)
            for tf_name in ['M5', 'M15', 'M30', 'H1']:
                tf_features = self.get_tf_simple_features(tf_name, current_time)
                features.extend(tf_features)  # 4 features each
            
            # Ensure exactly TOTAL_FEATURES
            if len(features) == self.TOTAL_FEATURES:
                return features
            else:
                # Pad or truncate
                if len(features) < self.TOTAL_FEATURES:
                    features.extend([0.0] * (self.TOTAL_FEATURES - len(features)))
                else:
                    features = features[:self.TOTAL_FEATURES]
                return features
                
        except Exception as e:
            return [0.0] * self.TOTAL_FEATURES
    
    def get_tf_simple_features(self, tf_name, current_time):
        """Lấy 4 features đơn giản từ timeframe"""
        try:
            tf_data = self.multi_timeframe_data[tf_name]
            
            # Tìm record gần nhất
            time_diffs = (tf_data['datetime'] - current_time).abs()
            closest_idx = time_diffs.idxmin()
            
            if closest_idx >= 3:
                recent_data = tf_data.iloc[closest_idx-3:closest_idx+1]
                
                current_close = recent_data['close'].iloc[-1]
                sma_2 = recent_data['close'].rolling(2).mean().iloc[-1]
                trend = (current_close - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0] * 100
                volume_avg = recent_data['volume'].mean()
                
                return [current_close, sma_2, trend, volume_avg]
            else:
                return [0.0, 0.0, 0.0, 0.0]
                
        except Exception as e:
            return [0.0, 0.0, 0.0, 0.0]
    
    def create_simple_label(self, base_idx):
        """Tạo label đơn giản với AI2.0 voting"""
        try:
            m1_data = self.multi_timeframe_data['M1']
            
            current_price = m1_data.iloc[base_idx]['close']
            future_price = m1_data.iloc[base_idx + 15]['close']  # 15 minutes ahead
            
            price_change_pct = (future_price - current_price) / current_price * 100
            
            # AI2.0 Voting với 3 factors
            votes = []
            
            # Vote 1: Price direction
            if price_change_pct > 0.1:
                votes.append('BUY')
            elif price_change_pct < -0.1:
                votes.append('SELL')
            else:
                votes.append('HOLD')
            
            # Vote 2: Magnitude
            if abs(price_change_pct) > 0.2:
                if price_change_pct > 0:
                    votes.append('BUY')
                else:
                    votes.append('SELL')
            else:
                votes.append('HOLD')
            
            # Vote 3: Conservative
            if abs(price_change_pct) > 0.05:
                if price_change_pct > 0:
                    votes.append('BUY')
                else:
                    votes.append('SELL')
            else:
                votes.append('HOLD')
            
            # Count votes
            buy_votes = votes.count('BUY')
            sell_votes = votes.count('SELL')
            hold_votes = votes.count('HOLD')
            
            # Majority wins
            if buy_votes > sell_votes and buy_votes > hold_votes:
                return 2  # BUY
            elif sell_votes > buy_votes and sell_votes > hold_votes:
                return 0  # SELL
            else:
                return 1  # HOLD
                
        except Exception as e:
            return 1  # Default HOLD
    
    def train_model(self, X, y):
        """Train model nhanh"""
        print(f"\n🧠 TRAINING MODEL")
        print("=" * 30)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to categorical
        y_train_cat = keras.utils.to_categorical(y_train, 3)
        y_test_cat = keras.utils.to_categorical(y_test, 3)
        
        # Simple model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.TOTAL_FEATURES,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train
        history = model.fit(
            X_train_scaled, y_train_cat,
            validation_data=(X_test_scaled, y_test_cat),
            epochs=20,  # Reduced epochs
            batch_size=64,
            verbose=1
        )
        
        # Evaluate
        train_acc = model.evaluate(X_train_scaled, y_train_cat, verbose=0)[1]
        test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)[1]
        
        # Predictions
        y_pred = model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Report
        report = classification_report(y_test, y_pred_classes, 
                                     target_names=['SELL', 'HOLD', 'BUY'], 
                                     output_dict=True)
        
        print(f"\n✅ RESULTS:")
        print(f"   🎯 Train Accuracy: {train_acc:.4f}")
        print(f"   🎯 Test Accuracy: {test_acc:.4f}")
        
        # Trading simulation
        trades = 0
        for pred in y_pred_classes:
            if pred != 1:  # Not HOLD
                trades += 1
        
        trade_rate = trades / len(y_pred_classes) * 100
        print(f"   📈 Trading Rate: {trade_rate:.1f}% ({trades}/{len(y_pred_classes)})")
        
        return {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classification_report': report,
            'trading_rate': trade_rate,
            'total_trades': trades
        }
    
    def run_training(self):
        """Chạy training nhanh"""
        print("🔥 FIXED FEATURES TRAINING")
        print("=" * 40)
        
        # Load data
        if not self.load_data():
            return None
        
        # Create features
        X, y = self.create_fixed_features()
        
        if len(X) == 0:
            return None
        
        # Train
        results = self.train_model(X, y)
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"{self.output_dir}/results_{timestamp}.json"
        
        results_data = {
            'timestamp': timestamp,
            'train_accuracy': float(results['train_accuracy']),
            'test_accuracy': float(results['test_accuracy']),
            'trading_rate': float(results['trading_rate']),
            'total_trades': int(results['total_trades']),
            'total_samples': len(X),
            'features_count': self.TOTAL_FEATURES
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n🎉 COMPLETED!")
        print(f"📁 Results: {results_file}")
        
        return results_file

def main():
    trainer = FixedFeaturesTraining()
    return trainer.run_training()

if __name__ == "__main__":
    main() 