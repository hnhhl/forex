#!/usr/bin/env python3
"""
🔗 TRUE MULTI-TIMEFRAME SYSTEM
Ultimate XAU System - CÁI NHÌN TỔNG QUAN THỊ TRƯỜNG

Tích hợp TẤT CẢ timeframes trong 1 model duy nhất
để có perspective hoàn chỉnh về thị trường
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class TrueMultiTimeframeSystem:
    """
    TRUE Multi-Timeframe System
    - 1 model nhìn TẤT CẢ timeframes cùng lúc
    - Hiểu context và relationship giữa các TF
    - Đưa ra entry signal + best timeframe
    """
    
    def __init__(self):
        self.symbol = "XAUUSDc"
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        # Features per timeframe (compressed)
        self.features_per_tf = 20  # Reduced from 67
        self.total_features = len(self.timeframes) * self.features_per_tf  # 7 * 20 = 140
        
        self.scaler = StandardScaler()
        self.model = None
        
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize():
            print("❌ Cannot connect to MT5")
            return False
        print("✅ Connected to MT5")
        return True
    
    def get_timeframe_data(self, timeframe_mt5, bars=1000):
        """Get data for specific timeframe"""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe_mt5, 0, bars)
        if rates is None:
            return None
        return pd.DataFrame(rates)
    
    def calculate_compressed_features(self, df, tf_name):
        """Calculate 20 most important features per timeframe"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price features (8)
            features[f'{tf_name}_sma_20'] = df['close'].rolling(20).mean()
            features[f'{tf_name}_ema_12'] = df['close'].ewm(span=12).mean()
            features[f'{tf_name}_price_sma_ratio'] = df['close'] / features[f'{tf_name}_sma_20']
            features[f'{tf_name}_price_change'] = df['close'].pct_change()
            features[f'{tf_name}_high_low_range'] = (df['high'] - df['low']) / df['close']
            features[f'{tf_name}_open_close_ratio'] = (df['close'] - df['open']) / df['open']
            features[f'{tf_name}_true_range'] = np.maximum(df['high'] - df['low'], 
                                                         np.maximum(abs(df['high'] - df['close'].shift()), 
                                                                  abs(df['low'] - df['close'].shift())))
            features[f'{tf_name}_gap'] = df['open'] - df['close'].shift()
            
            # Momentum features (6)
            features[f'{tf_name}_rsi'] = self.calculate_rsi(df['close'], 14)
            features[f'{tf_name}_macd'] = features[f'{tf_name}_ema_12'] - df['close'].ewm(span=26).mean()
            features[f'{tf_name}_macd_signal'] = features[f'{tf_name}_macd'].ewm(span=9).mean()
            features[f'{tf_name}_stoch'] = self.calculate_stochastic(df, 14)
            features[f'{tf_name}_williams_r'] = self.calculate_williams_r(df, 14)
            features[f'{tf_name}_momentum'] = df['close'] / df['close'].shift(10) - 1
            
            # Volatility features (4)
            features[f'{tf_name}_atr'] = features[f'{tf_name}_true_range'].rolling(14).mean()
            features[f'{tf_name}_volatility'] = df['close'].rolling(20).std()
            features[f'{tf_name}_bb_position'] = self.calculate_bb_position(df)
            features[f'{tf_name}_atr_ratio'] = features[f'{tf_name}_atr'] / df['close']
            
            # Volume features (2)
            features[f'{tf_name}_volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
            features[f'{tf_name}_volume_change'] = df['tick_volume'].pct_change()
            
            # Fill NaN
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            print(f"❌ Feature calculation error for {tf_name}: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices, period):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def calculate_stochastic(self, df, period):
        """Calculate Stochastic %K"""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        return 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
    
    def calculate_williams_r(self, df, period):
        """Calculate Williams %R"""
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        return -100 * (high_max - df['close']) / (high_max - low_min + 1e-8)
    
    def calculate_bb_position(self, df):
        """Calculate Bollinger Band position"""
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        return (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    
    def collect_multi_timeframe_data(self):
        """Thu thập data từ TẤT CẢ timeframes và align"""
        print("📊 Collecting multi-timeframe data...")
        
        all_data = {}
        
        # Collect data from all timeframes
        for tf_name, tf_mt5 in self.timeframes.items():
            print(f"  📈 Processing {tf_name}...")
            df = self.get_timeframe_data(tf_mt5, 2000)
            
            if df is not None and len(df) >= 100:
                # Calculate features
                features = self.calculate_compressed_features(df, tf_name)
                if not features.empty:
                    all_data[tf_name] = {
                        'features': features,
                        'prices': df['close'],
                        'time': pd.to_datetime(df['time'], unit='s')
                    }
                    print(f"    ✅ {tf_name}: {len(features)} samples, {features.shape[1]} features")
            else:
                print(f"    ❌ {tf_name}: Insufficient data")
        
        if len(all_data) < 3:
            print("❌ Need at least 3 timeframes")
            return None, None
        
        # Align all timeframes to common time range
        print("🔗 Aligning timeframes...")
        return self.align_multi_timeframe_data(all_data)
    
    def align_multi_timeframe_data(self, all_data):
        """Align tất cả timeframes về chung timeline"""
        
        # Find common time range
        start_times = [data['time'].min() for data in all_data.values()]
        end_times = [data['time'].max() for data in all_data.values()]
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        print(f"  📅 Common time range: {common_start} to {common_end}")
        
        # Use M15 as base timeframe for alignment
        if 'M15' not in all_data:
            print("❌ M15 timeframe required as base")
            return None, None
        
        base_data = all_data['M15']
        base_time = base_data['time']
        base_prices = base_data['prices']
        
        # Filter base data to common time range
        mask = (base_time >= common_start) & (base_time <= common_end)
        base_time_filtered = base_time[mask]
        base_prices_filtered = base_prices[mask]
        
        if len(base_time_filtered) < 500:
            print(f"❌ Insufficient aligned data: {len(base_time_filtered)}")
            return None, None
        
        # Create combined feature matrix
        combined_features = []
        
        for tf_name in ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']:
            if tf_name in all_data:
                tf_data = all_data[tf_name]
                tf_features = tf_data['features']
                tf_time = tf_data['time']
                
                # Resample to M15 timeline
                aligned_features = self.resample_to_base_timeframe(
                    tf_features, tf_time, base_time_filtered
                )
                
                combined_features.append(aligned_features)
                print(f"    ✅ {tf_name}: {aligned_features.shape[1]} features aligned")
            else:
                # Fill missing timeframe with zeros
                zero_features = np.zeros((len(base_time_filtered), self.features_per_tf))
                combined_features.append(zero_features)
                print(f"    ⚠️ {tf_name}: Missing, filled with zeros")
        
        # Combine all features
        X = np.concatenate(combined_features, axis=1)
        
        # Create labels based on M15 price movement
        y = self.create_multi_timeframe_labels(base_prices_filtered)
        
        print(f"✅ Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Labels distribution: {np.bincount(y)}")
        
        return X, y
    
    def resample_to_base_timeframe(self, features, time_index, base_time):
        """Resample features to base timeframe"""
        
        # Create DataFrame with time index
        df = pd.DataFrame(features.values, index=time_index, columns=features.columns)
        
        # Resample to base timeframe using forward fill
        base_df = pd.DataFrame(index=base_time)
        resampled = base_df.join(df, how='left').fillna(method='ffill').fillna(0)
        
        return resampled.values
    
    def create_multi_timeframe_labels(self, prices):
        """Create labels với multiple prediction horizons"""
        
        labels = []
        
        for i in range(len(prices) - 4):  # Need 4 bars ahead
            current_price = prices.iloc[i]
            future_price = prices.iloc[i + 4]  # 4 M15 bars = 1 hour ahead
            
            if pd.notna(current_price) and pd.notna(future_price):
                pct_change = (future_price - current_price) / current_price
                
                # Multi-class classification
                if pct_change > 0.002:  # > 0.2% = Strong BUY
                    labels.append(2)
                elif pct_change < -0.002:  # < -0.2% = Strong SELL
                    labels.append(0)
                else:  # HOLD/NEUTRAL
                    labels.append(1)
            else:
                labels.append(1)  # Default HOLD
        
        return np.array(labels)
    
    def create_multi_timeframe_model(self, input_dim):
        """Create model nhìn tất cả timeframes cùng lúc"""
        
        print("🏗️ Creating True Multi-Timeframe Model...")
        
        # Input layer
        main_input = Input(shape=(input_dim,), name='multi_tf_input')
        
        # Separate timeframe branches
        tf_branches = []
        
        for i, tf_name in enumerate(['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']):
            start_idx = i * self.features_per_tf
            end_idx = (i + 1) * self.features_per_tf
            
            # Extract timeframe-specific features
            tf_slice = tf.keras.layers.Lambda(
                lambda x, start=start_idx, end=end_idx: x[:, start:end],
                name=f'{tf_name}_slice'
            )(main_input)
            
            # Process each timeframe
            tf_processed = Dense(32, activation='relu', name=f'{tf_name}_dense1')(tf_slice)
            tf_processed = BatchNormalization(name=f'{tf_name}_bn1')(tf_processed)
            tf_processed = Dropout(0.2, name=f'{tf_name}_dropout1')(tf_processed)
            tf_processed = Dense(16, activation='relu', name=f'{tf_name}_dense2')(tf_processed)
            
            tf_branches.append(tf_processed)
        
        # Hierarchical combination
        # Short-term: M1, M5, M15
        short_term = Concatenate(name='short_term_concat')(tf_branches[0:3])
        short_term = Dense(32, activation='relu', name='short_term_dense')(short_term)
        
        # Medium-term: M30, H1
        medium_term = Concatenate(name='medium_term_concat')(tf_branches[3:5])
        medium_term = Dense(24, activation='relu', name='medium_term_dense')(medium_term)
        
        # Long-term: H4, D1
        long_term = Concatenate(name='long_term_concat')(tf_branches[5:7])
        long_term = Dense(16, activation='relu', name='long_term_dense')(long_term)
        
        # Final integration với attention
        all_timeframes = Concatenate(name='all_tf_concat')([short_term, medium_term, long_term])
        
        # Multi-head attention for timeframe relationships
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=18, name='tf_attention'
        )(tf.expand_dims(all_timeframes, 1), tf.expand_dims(all_timeframes, 1))
        
        attention_output = tf.squeeze(attention_output, 1)
        
        # Final prediction layers
        x = Dense(64, activation='relu', name='final_dense1')(attention_output)
        x = BatchNormalization(name='final_bn')(x)
        x = Dropout(0.3, name='final_dropout')(x)
        x = Dense(32, activation='relu', name='final_dense2')(x)
        
        # Multi-output
        signal_output = Dense(3, activation='softmax', name='signal_prediction')(x)  # BUY/HOLD/SELL
        timeframe_output = Dense(7, activation='softmax', name='best_timeframe')(x)  # Best TF for entry
        
        # Create model
        model = Model(
            inputs=main_input,
            outputs=[signal_output, timeframe_output],
            name='TrueMultiTimeframeModel'
        )
        
        # Compile with multiple losses
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'signal_prediction': 'sparse_categorical_crossentropy',
                'best_timeframe': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'signal_prediction': 0.7,  # Main prediction
                'best_timeframe': 0.3      # Secondary: best timeframe
            },
            metrics=['accuracy']
        )
        
        return model
    
    def train_true_multi_timeframe_system(self):
        """Train the TRUE multi-timeframe system"""
        
        print("🚀 TRAINING TRUE MULTI-TIMEFRAME SYSTEM")
        print("=" * 60)
        
        if not self.connect_mt5():
            return False
        
        try:
            # Collect and align multi-timeframe data
            X, y_signal = self.collect_multi_timeframe_data()
            
            if X is None:
                print("❌ Failed to collect multi-timeframe data")
                return False
            
            # Create best timeframe labels (simplified: prefer M15 for entries)
            y_timeframe = np.full(len(y_signal), 2)  # Index 2 = M15
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_signal_train, y_signal_test, y_tf_train, y_tf_test = train_test_split(
                X_scaled, y_signal, y_timeframe, test_size=0.2, random_state=42, stratify=y_signal
            )
            
            print(f"📊 Training data: {X_train.shape[0]} samples")
            print(f"📊 Test data: {X_test.shape[0]} samples")
            print(f"📊 Features: {X_train.shape[1]} (7 timeframes × {self.features_per_tf} each)")
            
            # Create model
            self.model = self.create_multi_timeframe_model(X_train.shape[1])
            
            print("\n📋 Model Architecture:")
            self.model.summary()
            
            # Training callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8),
                tf.keras.callbacks.ModelCheckpoint(
                    'training/xauusdc/models/true_multi_timeframe_model.h5',
                    save_best_only=True
                )
            ]
            
            # Train model
            print("\n🔥 Starting training...")
            history = self.model.fit(
                X_train,
                {
                    'signal_prediction': y_signal_train,
                    'best_timeframe': y_tf_train
                },
                validation_data=(
                    X_test,
                    {
                        'signal_prediction': y_signal_test,
                        'best_timeframe': y_tf_test
                    }
                ),
                batch_size=64,
                epochs=100,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            print("\n📊 Evaluating model...")
            test_results = self.model.evaluate(
                X_test,
                {
                    'signal_prediction': y_signal_test,
                    'best_timeframe': y_tf_test
                },
                verbose=0
            )
            
            signal_accuracy = test_results[3]  # signal_prediction_accuracy
            timeframe_accuracy = test_results[4]  # best_timeframe_accuracy
            
            print(f"\n🎯 FINAL RESULTS:")
            print(f"• Signal Prediction Accuracy: {signal_accuracy:.1%}")
            print(f"• Best Timeframe Accuracy: {timeframe_accuracy:.1%}")
            
            # Save scaler
            import joblib
            joblib.dump(self.scaler, 'training/xauusdc/models/true_mtf_scaler.pkl')
            
            print(f"\n✅ Model saved: true_multi_timeframe_model.h5")
            print(f"✅ Scaler saved: true_mtf_scaler.pkl")
            
            return True
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
        
        finally:
            mt5.shutdown()
    
    def get_live_prediction(self):
        """Get live prediction với cái nhìn tổng quan"""
        
        if self.model is None:
            print("❌ Model not loaded")
            return None
        
        if not self.connect_mt5():
            return None
        
        try:
            print("🔍 Getting live multi-timeframe analysis...")
            
            # Collect current market data
            current_data = {}
            
            for tf_name, tf_mt5 in self.timeframes.items():
                df = self.get_timeframe_data(tf_mt5, 100)
                if df is not None and len(df) >= 50:
                    features = self.calculate_compressed_features(df, tf_name)
                    if not features.empty:
                        current_data[tf_name] = features.iloc[-1:].values  # Latest row
                        print(f"  ✅ {tf_name}: Current data collected")
            
            if len(current_data) < 3:
                print("❌ Insufficient current data")
                return None
            
            # Prepare input vector
            input_vector = []
            for tf_name in ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']:
                if tf_name in current_data:
                    input_vector.append(current_data[tf_name][0])
                else:
                    input_vector.append(np.zeros(self.features_per_tf))
            
            X_current = np.concatenate(input_vector).reshape(1, -1)
            X_current_scaled = self.scaler.transform(X_current)
            
            # Get prediction
            predictions = self.model.predict(X_current_scaled, verbose=0)
            signal_probs = predictions[0][0]
            timeframe_probs = predictions[1][0]
            
            # Interpret results
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            tf_map = {0: 'M1', 1: 'M5', 2: 'M15', 3: 'M30', 4: 'H1', 5: 'H4', 6: 'D1'}
            
            predicted_signal = signal_map[np.argmax(signal_probs)]
            signal_confidence = np.max(signal_probs)
            
            best_timeframe = tf_map[np.argmax(timeframe_probs)]
            tf_confidence = np.max(timeframe_probs)
            
            result = {
                'signal': predicted_signal,
                'signal_confidence': float(signal_confidence),
                'best_timeframe': best_timeframe,
                'timeframe_confidence': float(tf_confidence),
                'signal_probabilities': signal_probs.tolist(),
                'timeframe_probabilities': timeframe_probs.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Live prediction error: {e}")
            return None
        
        finally:
            mt5.shutdown()

def main():
    """Main function"""
    system = TrueMultiTimeframeSystem()
    
    print("🔗 TRUE MULTI-TIMEFRAME SYSTEM")
    print("Tích hợp TẤT CẢ timeframes trong 1 model duy nhất")
    print("Cái nhìn tổng quan về thị trường → Entry signal + Best timeframe")
    print("=" * 70)
    
    # Train the system
    success = system.train_true_multi_timeframe_system()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("System now has COMPLETE market overview across all timeframes")
    else:
        print("\n❌ Training failed")

if __name__ == "__main__":
    main() 