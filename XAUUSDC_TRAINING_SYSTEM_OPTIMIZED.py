#!/usr/bin/env python3
"""
XAU/USDc Multi-Timeframe Training System - Optimized Version
Ultimate XAU Super System V4.0

Training chuyên sâu cho XAU/USDc trên tất cả timeframes:
M1, M5, M15, M30, H1, H4, D1

Mục đích: Nâng cao khả năng học tập và phản xạ với đa dạng thị trường
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import logging
import json
import pickle
from typing import Dict, List, Tuple, Optional
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XAUUSDcTrainingSystem:
    """Hệ thống training XAU/USDc tối ưu"""
    
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
        
        self.models = {}
        self.scalers = {}
        self.training_data = {}
        self.performance_metrics = {}
        
        # Tạo thư mục
        for folder in ['training/xauusdc/models', 'training/xauusdc/data', 'training/xauusdc/results']:
            os.makedirs(folder, exist_ok=True)
        
    def connect_mt5(self) -> bool:
        """Kết nối MT5"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
                
            # Kiểm tra symbol - thử cả XAUUSDc và XAUUSD
            for symbol_test in ["XAUUSDc", "XAUUSD", "GOLD"]:
                symbol_info = mt5.symbol_info(symbol_test)
                if symbol_info is not None:
                    self.symbol = symbol_test
                    if not symbol_info.visible:
                        mt5.symbol_select(symbol_test, True)
                    logger.info(f"Using symbol: {self.symbol}")
                    return True
                    
            logger.error("No gold symbol found in MT5")
            return False
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
            
    def collect_data(self, timeframe_key: str, bars: int = 5000) -> pd.DataFrame:
        """Thu thập dữ liệu từ MT5"""
        try:
            timeframe = self.timeframes[timeframe_key]
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No data for {timeframe_key}")
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            logger.info(f"Collected {len(df)} bars for {timeframe_key}")
            return df
            
        except Exception as e:
            logger.error(f"Data collection error: {e}")
            return pd.DataFrame()
            
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo technical features không cần talib"""
        try:
            if len(df) < 200:
                return df
                
            # Basic price features
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            df['oc_ratio'] = (df['close'] - df['open']) / df['open']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            
            # Moving averages
            for period in [5, 10, 20, 50, 100]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
                
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # RSI
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
                
            df['rsi_14'] = calculate_rsi(df['close'])
            df['rsi_21'] = calculate_rsi(df['close'], 21)
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Stochastic
            def calculate_stochastic(df, k_period=14, d_period=3):
                low_min = df['low'].rolling(window=k_period).min()
                high_max = df['high'].rolling(window=k_period).max()
                k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
                d_percent = k_percent.rolling(window=d_period).mean()
                return k_percent, d_percent
                
            df['stoch_k'], df['stoch_d'] = calculate_stochastic(df)
            
            # ATR
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_ratio'] = df['atr'] / df['close']
            
            # Williams %R
            def williams_r(df, period=14):
                high_max = df['high'].rolling(window=period).max()
                low_min = df['low'].rolling(window=period).min()
                return -100 * ((high_max - df['close']) / (high_max - low_min))
                
            df['williams_r'] = williams_r(df)
            
            # CCI
            def calculate_cci(df, period=14):
                tp = (df['high'] + df['low'] + df['close']) / 3
                ma = tp.rolling(period).mean()
                md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
                return (tp - ma) / (0.015 * md)
                
            df['cci'] = calculate_cci(df)
            
            # Momentum
            for period in [1, 5, 10, 20]:
                df[f'momentum_{period}'] = df['close'].pct_change(period)
                df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
                
            # Volatility
            for period in [10, 20, 50]:
                df[f'volatility_{period}'] = df['close'].rolling(period).std()
                df[f'vol_ratio_{period}'] = df[f'volatility_{period}'] / df['close']
                
            # Volume analysis (if available)
            if 'tick_volume' in df.columns:
                df['volume_sma'] = df['tick_volume'].rolling(20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
                
            # Support/Resistance
            df['support'] = df['low'].rolling(20).min()
            df['resistance'] = df['high'].rolling(20).max()
            df['support_dist'] = (df['close'] - df['support']) / df['close']
            df['resistance_dist'] = (df['resistance'] - df['close']) / df['close']
            
            # Price patterns
            df['doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1).astype(int)
            df['hammer'] = ((df['close'] > df['open']) & 
                           ((df['high'] - df['close']) < (df['close'] - df['open']) * 0.3) &
                           ((df['open'] - df['low']) > (df['close'] - df['open']) * 2)).astype(int)
            
            # Gaps
            df['gap_up'] = (df['open'] > df['close'].shift()).astype(int)
            df['gap_down'] = (df['open'] < df['close'].shift()).astype(int)
            
            # Time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            
            logger.info(f"Created {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Feature creation error: {e}")
            return df
            
    def create_targets(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Tạo target labels"""
        try:
            # Định nghĩa horizons theo timeframe
            horizon_map = {
                'M1': [5, 15, 30],
                'M5': [3, 6, 12], 
                'M15': [2, 4, 8],
                'M30': [2, 4, 8],
                'H1': [2, 4, 12],
                'H4': [2, 6, 12],
                'D1': [2, 5, 10]
            }
            
            horizons = horizon_map.get(timeframe, [2, 4, 8])
            
            for horizon in horizons:
                # Future returns
                future_return = df['close'].shift(-horizon) / df['close'] - 1
                df[f'return_{horizon}'] = future_return
                
                # Classification labels - adjusted thresholds for XAU
                threshold = 0.002  # 0.2% threshold for gold
                df[f'direction_{horizon}'] = np.where(
                    future_return > threshold, 1,
                    np.where(future_return < -threshold, -1, 0)
                )
                
                # Strong signals
                strong_threshold = threshold * 2
                df[f'strong_signal_{horizon}'] = np.where(
                    abs(future_return) > strong_threshold,
                    np.sign(future_return), 0
                )
                
            return df
            
        except Exception as e:
            logger.error(f"Target creation error: {e}")
            return df
            
    def prepare_training_data(self, timeframe: str) -> Dict:
        """Chuẩn bị dữ liệu training"""
        try:
            logger.info(f"Preparing data for {timeframe}")
            
            # Thu thập dữ liệu
            df = self.collect_data(timeframe, 10000)
            if df.empty:
                return {}
                
            # Tạo features
            df = self.create_features(df)
            df = self.create_targets(df, timeframe)
            
            # Loại bỏ NaN
            df.dropna(inplace=True)
            
            if len(df) < 1000:
                logger.warning(f"Insufficient data: {len(df)}")
                return {}
                
            # Chọn features
            target_cols = [col for col in df.columns if 
                          any(x in col for x in ['direction_', 'return_', 'signal_'])]
            feature_cols = [col for col in df.columns if col not in 
                          ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'] + target_cols]
            
            X = df[feature_cols].values
            
            # Chuẩn hóa
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[timeframe] = scaler
            
            # Tạo training data
            training_data = {
                'X': X_scaled,
                'feature_names': feature_cols,
                'timestamps': df.index.values
            }
            
            # Thêm targets
            for horizon in [2, 4, 8]:
                if f'direction_{horizon}' in df.columns:
                    training_data[f'y_direction_{horizon}'] = df[f'direction_{horizon}'].values
                if f'return_{horizon}' in df.columns:
                    training_data[f'y_return_{horizon}'] = df[f'return_{horizon}'].values
                    
            self.training_data[timeframe] = training_data
            
            # Lưu data
            with open(f"training/xauusdc/data/{timeframe}_data.pkl", 'wb') as f:
                pickle.dump(training_data, f)
                
            logger.info(f"Data prepared: {X_scaled.shape}")
            return training_data
            
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return {}
            
    def create_model(self, input_dim: int, task: str = 'classification') -> tf.keras.Model:
        """Tạo neural network model"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Dense(
                    3 if task == 'classification' else 1,
                    activation='softmax' if task == 'classification' else 'tanh'
                )
            ])
            
            if task == 'classification':
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(0.001),
                    loss='mse',
                    metrics=['mae']
                )
                
            return model
            
        except Exception as e:
            logger.error(f"Model creation error: {e}")
            return None
            
    def train_models(self, timeframe: str) -> Dict:
        """Training models cho timeframe"""
        try:
            logger.info(f"Training models for {timeframe}")
            
            data = self.training_data.get(timeframe)
            if not data:
                return {}
                
            X = data['X']
            results = {}
            
            for horizon in [2, 4, 8]:
                if f'y_direction_{horizon}' not in data:
                    continue
                    
                logger.info(f"Training {timeframe} - horizon {horizon}")
                
                # Classification model
                y = data[f'y_direction_{horizon}'] + 1  # Convert to 0,1,2
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Create model
                model = self.create_model(X.shape[1], 'classification')
                if model is None:
                    continue
                    
                # Callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
                    tf.keras.callbacks.ModelCheckpoint(
                        f"training/xauusdc/models/{timeframe}_dir_{horizon}.h5",
                        save_best_only=True
                    )
                ]
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate
                train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
                test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
                
                # Save model
                self.models[f"{timeframe}_dir_{horizon}"] = model
                
                results[f"direction_{horizon}"] = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'samples': len(X_train)
                }
                
                logger.info(f"  Direction {horizon}: Train={train_acc:.3f}, Test={test_acc:.3f}")
                
            return results
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {}
            
    def run_training(self) -> Dict:
        """Chạy training cho tất cả timeframes"""
        try:
            print("🚀 XAU/USDc Multi-Timeframe Training System")
            print("=" * 60)
            
            if not self.connect_mt5():
                return {}
                
            all_results = {}
            
            for tf in self.timeframes.keys():
                print(f"\n📊 Processing {tf}...")
                
                try:
                    # Prepare data
                    data = self.prepare_training_data(tf)
                    if not data:
                        print(f"  ⚠️ Skipping {tf} - no data")
                        continue
                        
                    # Train models
                    results = self.train_models(tf)
                    
                    all_results[tf] = {
                        'results': results,
                        'samples': len(data['X']),
                        'features': len(data['feature_names'])
                    }
                    
                    print(f"  ✅ {tf} completed - {len(data['X'])} samples")
                    
                except Exception as e:
                    print(f"  ❌ {tf} failed: {e}")
                    continue
                    
            # Save results
            with open("training/xauusdc/results/training_results.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
                
            self.generate_report(all_results)
            return all_results
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {}
        finally:
            mt5.shutdown()
            
    def generate_report(self, results: Dict):
        """Tạo báo cáo training"""
        try:
            print("\n" + "="*60)
            print("📈 TRAINING RESULTS SUMMARY")
            print("="*60)
            
            total_models = 0
            avg_accuracy = []
            
            for tf, tf_data in results.items():
                tf_results = tf_data.get('results', {})
                samples = tf_data.get('samples', 0)
                features = tf_data.get('features', 0)
                
                print(f"\n{tf}:")
                print(f"  Data: {samples:,} samples, {features} features")
                
                tf_acc = []
                for key, metrics in tf_results.items():
                    if 'direction_' in key:
                        test_acc = metrics.get('test_acc', 0)
                        print(f"  {key}: {test_acc:.1%}")
                        tf_acc.append(test_acc)
                        total_models += 1
                        
                if tf_acc:
                    tf_avg = np.mean(tf_acc)
                    avg_accuracy.extend(tf_acc)
                    print(f"  Average: {tf_avg:.1%}")
                    
            if avg_accuracy:
                overall_avg = np.mean(avg_accuracy)
                print(f"\n🎯 OVERALL RESULTS:")
                print(f"  Total models: {total_models}")
                print(f"  Average accuracy: {overall_avg:.1%}")
                print(f"  Symbol: {self.symbol}")
                
            print("="*60)
            print("✅ Training completed successfully!")
            print("📁 Results saved to: training/xauusdc/")
            
        except Exception as e:
            logger.error(f"Report error: {e}")


def main():
    """Main function"""
    
    system = XAUUSDcTrainingSystem()
    results = system.run_training()
    
    if results:
        print(f"\n🎉 Training successful!")
        print(f"📊 Timeframes trained: {len(results)}")
    else:
        print(f"\n❌ Training failed!")
        
    return results

if __name__ == "__main__":
    main() 