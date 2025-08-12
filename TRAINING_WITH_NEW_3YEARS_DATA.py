#!/usr/bin/env python3
"""
🔥 TRAINING WITH NEW 3 YEARS DATA + AI2.0 VOTING SYSTEM
======================================================================
🎯 Sử dụng dữ liệu realistic 3 năm (2022-2024)
🗳️ AI2.0 Voting System (thay thế hard thresholds)
📊 1.1M+ M1 records training
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

class New3YearsTraining:
    def __init__(self):
        self.data_dir = "data/working_free_data"
        self.output_dir = "new_3years_training_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.multi_timeframe_data = {}
        self.scaler = StandardScaler()
        
    def load_3years_data(self):
        """Load dữ liệu 3 năm realistic"""
        print("🔥 LOADING 3 YEARS REALISTIC DATA (2022-2024)")
        print("=" * 60)
        
        timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        
        for tf in timeframes:
            csv_file = f"{self.data_dir}/XAUUSD_{tf}_realistic.csv"
            
            if os.path.exists(csv_file):
                print(f"\n📊 Loading {tf}...")
                
                df = pd.read_csv(csv_file)
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                
                # Rename columns
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
                
                df = df.sort_values('datetime').reset_index(drop=True)
                self.multi_timeframe_data[tf] = df
                
                print(f"   ✅ {len(df):,} records")
                print(f"   📅 {df['datetime'].min()} → {df['datetime'].max()}")
                print(f"   💰 ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            else:
                print(f"   ❌ File not found: {csv_file}")
        
        total_records = sum(len(df) for df in self.multi_timeframe_data.values())
        print(f"\n✅ LOADED {len(self.multi_timeframe_data)} timeframes")
        print(f"📊 Total records: {total_records:,}")
        
        return len(self.multi_timeframe_data) > 0
    
    def create_features_and_labels(self):
        """Tạo features và labels từ dữ liệu 3 năm với AI2.0 logic"""
        print(f"\n🔄 CREATING FEATURES & LABELS (AI2.0 VOTING LOGIC)")
        print("=" * 60)
        
        # Sử dụng M1 làm base
        base_data = self.multi_timeframe_data['M1']
        print(f"🎯 Base timeframe: M1 ({len(base_data):,} records)")
        
        # Sampling: Mỗi 30 phút (30 M1 candles) để có đủ data nhưng không quá nhiều
        step_size = 30
        sequence_length = 60  # 1 hour lookback
        
        X = []
        y = []
        
        print(f"📊 Processing with step_size={step_size} minutes...")
        
        # FIXED: Tính toán expected feature length trước
        expected_feature_length = self.calculate_expected_feature_length()
        print(f"🎯 Expected feature length: {expected_feature_length}")
        
        # Tạo sequences
        for i in range(sequence_length, len(base_data) - 1, step_size):
            if i % 10000 == 0:
                progress = (i / len(base_data)) * 100
                print(f"   🔄 Progress: {progress:.1f}% ({i:,}/{len(base_data):,})")
            
            try:
                # Current time
                current_time = base_data.iloc[i]['datetime']
                
                # Create multi-timeframe features với fixed length
                features = self.create_multi_tf_features_fixed(current_time, i, expected_feature_length)
                
                if features is not None and len(features) == expected_feature_length:
                    # Create AI2.0 voting-based label
                    label = self.create_ai2_voting_label(i)
                    
                    if label is not None:
                        X.append(features)
                        y.append(label)
                        
            except Exception as e:
                continue
        
        # FIXED: Convert với error handling
        try:
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int32)
        except Exception as e:
            print(f"   ⚠️  Error converting to arrays: {e}")
            # Fallback: Filter ra các features có độ dài sai
            valid_X = []
            valid_y = []
            
            for i, (feat, label) in enumerate(zip(X, y)):
                if len(feat) == expected_feature_length:
                    valid_X.append(feat)
                    valid_y.append(label)
            
            X = np.array(valid_X, dtype=np.float32)
            y = np.array(valid_y, dtype=np.int32)
        
        print(f"\n✅ FEATURES & LABELS CREATED:")
        print(f"   📊 Total samples: {len(X):,}")
        print(f"   🎯 Features per sample: {X.shape[1] if len(X) > 0 else 0}")
        
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            print(f"   📈 Labels distribution:")
            for label, count in zip(unique, counts):
                label_name = ['SELL', 'HOLD', 'BUY'][int(label)]
                percentage = (count / len(y)) * 100
                print(f"      {label_name}: {count:,} ({percentage:.1f}%)")
        
        return X, y
    
    def calculate_expected_feature_length(self):
        """Tính toán độ dài features mong đợi"""
        # M1 features: 12
        m1_features = 12
        
        # Higher timeframe features: 4 features each
        higher_tf_features = 4 * 4  # M5, M15, M30, H1
        
        total_expected = m1_features + higher_tf_features
        return total_expected
    
    def create_multi_tf_features_fixed(self, current_time, base_idx, expected_length):
        """Tạo features từ multiple timeframes với fixed length"""
        try:
            features = []
            
            # Features từ M1 (base) - ALWAYS 12 features
            m1_data = self.multi_timeframe_data['M1']
            lookback = min(20, base_idx)
            
            if lookback > 5:
                recent_m1 = m1_data.iloc[base_idx-lookback:base_idx+1]
                
                # Basic OHLC features (4)
                current_close = recent_m1['close'].iloc[-1]
                current_open = recent_m1['open'].iloc[-1]
                current_high = recent_m1['high'].iloc[-1]
                current_low = recent_m1['low'].iloc[-1]
                
                # Moving averages (3)
                sma_5 = recent_m1['close'].rolling(5).mean().iloc[-1]
                sma_10 = recent_m1['close'].rolling(10).mean().iloc[-1]
                sma_20 = recent_m1['close'].rolling(20).mean().iloc[-1] if len(recent_m1) >= 20 else sma_10
                
                # Price momentum (1)
                price_change = (current_close - recent_m1['close'].iloc[0]) / recent_m1['close'].iloc[0] * 100
                
                # Volatility (1)
                returns = recent_m1['close'].pct_change().dropna()
                volatility = returns.std() * 100 if len(returns) > 1 else 0
                
                # Volume features (2)
                avg_volume = recent_m1['volume'].mean()
                volume_ratio = recent_m1['volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
                
                # Price position (1)
                recent_high = recent_m1['high'].max()
                recent_low = recent_m1['low'].min()
                price_position = (current_close - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
                
                # M1 features - EXACTLY 12
                m1_features = [
                    current_close, current_open, current_high, current_low,  # 4
                    sma_5, sma_10, sma_20,  # 3
                    price_change, volatility,  # 2
                    avg_volume, volume_ratio,  # 2
                    price_position  # 1
                ]  # Total: 12
                
                features.extend(m1_features)
                
                # Features từ higher timeframes - EXACTLY 4 features each
                for tf_name in ['M5', 'M15', 'M30', 'H1']:
                    if tf_name in self.multi_timeframe_data:
                        tf_features = self.get_tf_features_fixed(tf_name, current_time)
                        features.extend(tf_features)  # Always 4 features
                
                # Đảm bảo đúng độ dài
                if len(features) == expected_length:
                    return np.array(features, dtype=np.float32)
                else:
                    # Pad hoặc truncate nếu cần
                    if len(features) < expected_length:
                        features.extend([0.0] * (expected_length - len(features)))
                    else:
                        features = features[:expected_length]
                    
                    return np.array(features, dtype=np.float32)
            else:
                # Return zero features nếu không đủ data
                return np.zeros(expected_length, dtype=np.float32)
                
        except Exception as e:
            # Return zero features nếu có lỗi
            return np.zeros(expected_length, dtype=np.float32)
    
    def get_tf_features_fixed(self, tf_name, current_time):
        """Lấy EXACTLY 4 features từ một timeframe cụ thể"""
        try:
            tf_data = self.multi_timeframe_data[tf_name]
            
            # Tìm record gần nhất
            time_diffs = (tf_data['datetime'] - current_time).abs()
            closest_idx = time_diffs.idxmin()
            
            lookback = min(10, closest_idx)
            if lookback > 2:
                recent_data = tf_data.iloc[closest_idx-lookback:closest_idx+1]
                
                # EXACTLY 4 features
                current_close = recent_data['close'].iloc[-1]
                sma_3 = recent_data['close'].rolling(3).mean().iloc[-1]
                sma_5 = recent_data['close'].rolling(5).mean().iloc[-1]
                trend = (current_close - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0] * 100
                
                return [current_close, sma_3, sma_5, trend]
            else:
                # Return default 4 features
                return [0.0, 0.0, 0.0, 0.0]
                
        except Exception as e:
            # Return default 4 features
            return [0.0, 0.0, 0.0, 0.0]
    
    def create_ai2_voting_label(self, base_idx):
        """Tạo label sử dụng AI2.0 voting logic thay vì hard thresholds"""
        try:
            m1_data = self.multi_timeframe_data['M1']
            
            # Current và future prices
            current_price = m1_data.iloc[base_idx]['close']
            
            # Look ahead 15 minutes (15 M1 candles)
            future_idx = min(base_idx + 15, len(m1_data) - 1)
            future_price = m1_data.iloc[future_idx]['close']
            
            # Price change
            price_change_pct = (future_price - current_price) / current_price * 100
            
            # AI2.0 VOTING SYSTEM: Multiple factors vote
            votes = []
            
            # Vote 1: Price momentum
            if abs(price_change_pct) > 0.05:  # >0.05% change
                if price_change_pct > 0:
                    votes.append('BUY')
                else:
                    votes.append('SELL')
            else:
                votes.append('HOLD')
            
            # Vote 2: Volatility-adjusted threshold
            recent_data = m1_data.iloc[max(0, base_idx-20):base_idx+1]
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * 100 if len(returns) > 1 else 0.5
            
            vol_threshold = max(0.1, volatility * 0.3)
            if price_change_pct > vol_threshold:
                votes.append('BUY')
            elif price_change_pct < -vol_threshold:
                votes.append('SELL')
            else:
                votes.append('HOLD')
            
            # Vote 3: Volume confirmation
            if base_idx > 0:
                current_volume = m1_data.iloc[base_idx]['volume']
                avg_volume = recent_data['volume'].mean()
                
                if current_volume > avg_volume * 1.2:  # High volume
                    if price_change_pct > 0.02:
                        votes.append('BUY')
                    elif price_change_pct < -0.02:
                        votes.append('SELL')
                    else:
                        votes.append('HOLD')
                else:
                    votes.append('HOLD')
            else:
                votes.append('HOLD')
            
            # MAJORITY VOTING
            buy_votes = votes.count('BUY')
            sell_votes = votes.count('SELL')
            hold_votes = votes.count('HOLD')
            
            if buy_votes > sell_votes and buy_votes > hold_votes:
                return 2  # BUY
            elif sell_votes > buy_votes and sell_votes > hold_votes:
                return 0  # SELL
            else:
                return 1  # HOLD
                
        except Exception as e:
            return None
    
    def train_neural_network(self, X, y):
        """Train neural network với dữ liệu 3 năm"""
        print(f"\n🧠 TRAINING NEURAL NETWORK (3 YEARS DATA)")
        print("=" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"📊 Test samples: {len(X_test):,}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(y_train, 3)
        y_test_cat = keras.utils.to_categorical(y_test, 3)
        
        # Build model
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"🏗️  Model architecture:")
        print(f"   Input features: {X_train.shape[1]}")
        print(f"   Hidden layers: 128 → 64 → 32")
        print(f"   Output classes: 3 (SELL, HOLD, BUY)")
        
        # Train
        print(f"\n🚀 Training...")
        history = model.fit(
            X_train_scaled, y_train_cat,
            validation_data=(X_test_scaled, y_test_cat),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train_scaled, y_train_cat, verbose=0)
        test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        
        print(f"\n✅ TRAINING COMPLETED:")
        print(f"   🎯 Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Detailed evaluation
        y_pred = model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        report = classification_report(
            y_test, y_pred_classes, 
            target_names=['SELL', 'HOLD', 'BUY'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        print(f"\n📊 DETAILED RESULTS:")
        print(f"   SELL: Precision={report['SELL']['precision']:.3f}, Recall={report['SELL']['recall']:.3f}")
        print(f"   HOLD: Precision={report['HOLD']['precision']:.3f}, Recall={report['HOLD']['recall']:.3f}")
        print(f"   BUY:  Precision={report['BUY']['precision']:.3f}, Recall={report['BUY']['recall']:.3f}")
        
        # Trading simulation
        trading_results = self.simulate_trading(y_test, y_pred_classes)
        
        return {
            'model': model,
            'history': history,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'trading_simulation': trading_results,
            'data_info': {
                'total_samples': len(X),
                'features_count': X.shape[1],
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        }
    
    def simulate_trading(self, y_true, y_pred):
        """Simulate trading để test performance"""
        balance = 10000
        position = 0
        trades = []
        
        for i in range(len(y_true)):
            if y_pred[i] == 2 and position <= 0:  # BUY signal
                trades.append({'action': 'BUY', 'balance': balance})
                position = 1
            elif y_pred[i] == 0 and position >= 0:  # SELL signal
                trades.append({'action': 'SELL', 'balance': balance})
                position = -1
        
        win_rate = 0 if len(trades) == 0 else len([t for t in trades if 'profit' in t and t['profit'] > 0]) / len(trades)
        
        return {
            'initial_balance': 10000,
            'final_balance': balance,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'trades_sample': trades[:10]  # First 10 trades
        }
    
    def save_results(self, results):
        """Save training results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_file = f"{self.output_dir}/model_3years_{timestamp}.keras"
        results['model'].save(model_file)
        
        # Save results
        results_data = {
            'timestamp': timestamp,
            'approach': '3_years_realistic_data_ai2_voting',
            'description': 'Training with 3 years realistic data + AI2.0 voting system',
            'train_accuracy': float(results['train_accuracy']),
            'test_accuracy': float(results['test_accuracy']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'],
            'trading_simulation': results['trading_simulation'],
            'data_info': results['data_info'],
            'model_file': model_file
        }
        
        results_file = f"{self.output_dir}/results_3years_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n💾 RESULTS SAVED:")
        print(f"   📁 Model: {model_file}")
        print(f"   📁 Results: {results_file}")
        
        return results_file
    
    def run_training(self):
        """Chạy toàn bộ quá trình training"""
        print("🔥 NEW 3 YEARS TRAINING WITH AI2.0 VOTING SYSTEM")
        print("=" * 70)
        
        # 1. Load 3 years data
        if not self.load_3years_data():
            print("❌ Không thể load dữ liệu!")
            return None
        
        # 2. Create features & labels
        X, y = self.create_features_and_labels()
        
        if len(X) == 0:
            print("❌ Không có training data!")
            return None
        
        # 3. Train model
        results = self.train_neural_network(X, y)
        
        # 4. Save results
        results_file = self.save_results(results)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"📊 Final Summary:")
        print(f"   🎯 Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"   📈 Total Samples: {results['data_info']['total_samples']:,}")
        print(f"   🔄 Total Trades: {results['trading_simulation']['total_trades']}")
        print(f"   📁 Results: {results_file}")
        
        return results_file

def main():
    trainer = New3YearsTraining()
    results_file = trainer.run_training()
    return results_file

if __name__ == "__main__":
    main() 