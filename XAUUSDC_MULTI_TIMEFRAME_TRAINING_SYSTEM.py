#!/usr/bin/env python3
"""
XAU/USDc Multi-Timeframe Training System
Ultimate XAU Super System V4.0

Training cho cặp XAU/USDc trên tất cả timeframes:
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
import talib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XAUUSDcMultiTimeframeTrainingSystem:
    """Hệ thống training XAU/USDc đa khung thời gian"""
    
    def __init__(self):
        self.symbol = "XAUUSDc"  # Gold vs US Dollar cent
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
        
        # Tạo thư mục lưu trữ
        os.makedirs("training/xauusdc", exist_ok=True)
        os.makedirs("training/xauusdc/models", exist_ok=True)
        os.makedirs("training/xauusdc/data", exist_ok=True)
        os.makedirs("training/xauusdc/results", exist_ok=True)
        
    def connect_mt5(self) -> bool:
        """Kết nối MT5"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
                
            # Kiểm tra symbol
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return False
                
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select {self.symbol}")
                    return False
                    
            logger.info(f"MT5 connected successfully. Symbol {self.symbol} ready.")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
            
    def collect_historical_data(self, timeframe_key: str, bars: int = 10000) -> pd.DataFrame:
        """Thu thập dữ liệu lịch sử từ MT5"""
        try:
            timeframe = self.timeframes[timeframe_key]
            
            # Lấy dữ liệu từ MT5
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No data received for {timeframe_key}")
                return pd.DataFrame()
                
            # Chuyển đổi sang DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            logger.info(f"Collected {len(df)} bars for {timeframe_key}")
            return df
            
        except Exception as e:
            logger.error(f"Data collection error for {timeframe_key}: {e}")
            return pd.DataFrame()
            
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo các chỉ báo kỹ thuật"""
        try:
            if len(df) < 100:
                logger.warning("Insufficient data for technical indicators")
                return df
                
            # Price action features
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            df['oc_ratio'] = (df['close'] - df['open']) / df['open']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
            df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
            
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = talib.SMA(df['close'].values, timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(df['close'].values, timeperiod=period)
                
            # Price relative to MAs
            for period in [20, 50, 100]:
                df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
                df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
                
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # RSI
            for period in [14, 21, 30]:
                df[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
                
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(
                df['high'].values, df['low'].values, df['close'].values,
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            
            # Williams %R
            df['williams_r'] = talib.WILLR(
                df['high'].values, df['low'].values, df['close'].values, timeperiod=14
            )
            
            # ADX
            df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # ATR
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['atr_ratio'] = df['atr'] / df['close']
            
            # CCI
            df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # Volume indicators (if available)
            if 'tick_volume' in df.columns:
                df['volume_sma'] = talib.SMA(df['tick_volume'].values, timeperiod=20)
                df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
                
            # Price patterns
            df['doji'] = talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            df['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            df['engulfing'] = talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            
            # Momentum features
            for period in [1, 5, 10, 20]:
                df[f'momentum_{period}'] = df['close'].pct_change(period)
                df[f'price_change_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
                
            # Volatility features
            for period in [5, 10, 20]:
                df[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close']
                
            # Support/Resistance levels
            df['support'] = df['low'].rolling(20).min()
            df['resistance'] = df['high'].rolling(20).max()
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
            
            # Fibonacci levels
            df['fib_high'] = df['high'].rolling(100).max()
            df['fib_low'] = df['low'].rolling(100).min()
            df['fib_range'] = df['fib_high'] - df['fib_low']
            df['fib_23.6'] = df['fib_low'] + 0.236 * df['fib_range']
            df['fib_38.2'] = df['fib_low'] + 0.382 * df['fib_range']
            df['fib_50.0'] = df['fib_low'] + 0.500 * df['fib_range']
            df['fib_61.8'] = df['fib_low'] + 0.618 * df['fib_range']
            
            logger.info(f"Created {len(df.columns)} technical features")
            return df
            
        except Exception as e:
            logger.error(f"Technical features creation error: {e}")
            return df
            
    def create_multi_timeframe_features(self, timeframe_key: str) -> pd.DataFrame:
        """Tạo features từ nhiều timeframe"""
        try:
            # Thu thập dữ liệu timeframe chính
            main_df = self.collect_historical_data(timeframe_key, 10000)
            if main_df.empty:
                return pd.DataFrame()
                
            # Tạo technical features cho timeframe chính
            main_df = self.create_technical_features(main_df)
            
            # Thu thập dữ liệu từ timeframes khác để tạo context
            timeframe_order = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
            current_index = timeframe_order.index(timeframe_key)
            
            # Thêm features từ timeframe cao hơn (trend context)
            for i in range(current_index + 1, len(timeframe_order)):
                higher_tf = timeframe_order[i]
                higher_df = self.collect_historical_data(higher_tf, 2000)
                
                if not higher_df.empty:
                    # Resample dữ liệu timeframe cao hơn xuống timeframe hiện tại
                    higher_features = self.extract_higher_timeframe_features(higher_df, higher_tf)
                    main_df = self.merge_timeframe_features(main_df, higher_features, higher_tf)
                    
            # Thêm features từ timeframe thấp hơn (micro structure)
            for i in range(current_index - 1, -1, -1):
                lower_tf = timeframe_order[i]
                if lower_tf in ['M1', 'M5'] and timeframe_key in ['H4', 'D1']:
                    # Chỉ lấy features từ timeframe thấp cho những timeframe cao
                    continue
                    
                lower_df = self.collect_historical_data(lower_tf, 5000)
                if not lower_df.empty:
                    lower_features = self.extract_lower_timeframe_features(lower_df, lower_tf)
                    main_df = self.merge_timeframe_features(main_df, lower_features, lower_tf)
                    
            return main_df
            
        except Exception as e:
            logger.error(f"Multi-timeframe features creation error: {e}")
            return pd.DataFrame()
            
    def extract_higher_timeframe_features(self, df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
        """Trích xuất features từ timeframe cao hơn"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Trend direction
            features[f'{tf_name}_trend_sma'] = np.where(df['close'] > df['sma_20'], 1, -1)
            features[f'{tf_name}_trend_ema'] = np.where(df['close'] > df['ema_20'], 1, -1)
            
            # Momentum
            features[f'{tf_name}_rsi'] = df['rsi_14']
            features[f'{tf_name}_macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
            
            # Volatility
            features[f'{tf_name}_atr_ratio'] = df['atr_ratio']
            features[f'{tf_name}_bb_position'] = df['bb_position']
            
            # Support/Resistance
            features[f'{tf_name}_support_distance'] = df['support_distance']
            features[f'{tf_name}_resistance_distance'] = df['resistance_distance']
            
            return features
            
        except Exception as e:
            logger.error(f"Higher timeframe features extraction error: {e}")
            return pd.DataFrame()
            
    def extract_lower_timeframe_features(self, df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
        """Trích xuất features từ timeframe thấp hơn"""
        try:
            # Aggregate lower timeframe data
            agg_data = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'tick_volume': 'sum'
            }).dropna()
            
            features = pd.DataFrame(index=agg_data.index)
            
            # Micro-structure features
            features[f'{tf_name}_volatility'] = agg_data['close'].rolling(10).std()
            features[f'{tf_name}_volume_avg'] = agg_data['tick_volume'].rolling(10).mean()
            features[f'{tf_name}_price_velocity'] = agg_data['close'].diff()
            
            return features
            
        except Exception as e:
            logger.error(f"Lower timeframe features extraction error: {e}")
            return pd.DataFrame()
            
    def merge_timeframe_features(self, main_df: pd.DataFrame, 
                                other_features: pd.DataFrame, tf_name: str) -> pd.DataFrame:
        """Gộp features từ các timeframe khác"""
        try:
            # Forward fill để match với timeframe chính
            other_features_resampled = other_features.resample(main_df.index.freq or '1min').ffill()
            
            # Merge với main dataframe
            merged_df = main_df.join(other_features_resampled, how='left')
            merged_df.fillna(method='ffill', inplace=True)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Timeframe merge error: {e}")
            return main_df
            
    def create_target_labels(self, df: pd.DataFrame, timeframe_key: str) -> pd.DataFrame:
        """Tạo target labels cho training"""
        try:
            # Định nghĩa horizons khác nhau tùy theo timeframe
            horizon_map = {
                'M1': [5, 15, 30],      # 5, 15, 30 phút
                'M5': [3, 6, 12],       # 15, 30, 60 phút  
                'M15': [2, 4, 8],       # 30, 60, 120 phút
                'M30': [2, 4, 8],       # 1, 2, 4 giờ
                'H1': [2, 4, 12],       # 2, 4, 12 giờ
                'H4': [2, 6, 12],       # 8, 24, 48 giờ
                'D1': [2, 5, 10]        # 2, 5, 10 ngày
            }
            
            horizons = horizon_map.get(timeframe_key, [2, 4, 8])
            
            for horizon in horizons:
                # Price direction
                future_return = df['close'].shift(-horizon) / df['close'] - 1
                
                # Classification labels
                df[f'direction_{horizon}'] = np.where(future_return > 0.001, 1,  # Buy
                                                    np.where(future_return < -0.001, -1, 0))  # Sell, Hold
                
                # Regression targets
                df[f'return_{horizon}'] = future_return
                
                # Volatility-adjusted returns
                volatility = df['atr_ratio'].rolling(20).mean()
                df[f'vol_adj_return_{horizon}'] = future_return / (volatility + 1e-8)
                
                # Risk-adjusted targets
                df[f'sharpe_target_{horizon}'] = df[f'return_{horizon}'] / (df[f'return_{horizon}'].rolling(20).std() + 1e-8)
                
            # Strong signal labels (cho high-confidence predictions)
            for horizon in horizons:
                strong_threshold = df[f'return_{horizon}'].std() * 1.5
                df[f'strong_signal_{horizon}'] = np.where(
                    abs(df[f'return_{horizon}']) > strong_threshold,
                    np.sign(df[f'return_{horizon}']), 0
                )
                
            logger.info(f"Created target labels for {timeframe_key}")
            return df
            
        except Exception as e:
            logger.error(f"Target labels creation error: {e}")
            return df
            
    def prepare_training_data(self, timeframe_key: str) -> Dict:
        """Chuẩn bị dữ liệu training"""
        try:
            logger.info(f"Preparing training data for {timeframe_key}")
            
            # Thu thập và xử lý dữ liệu
            df = self.create_multi_timeframe_features(timeframe_key)
            if df.empty:
                return {}
                
            # Tạo target labels
            df = self.create_target_labels(df, timeframe_key)
            
            # Loại bỏ NaN values
            df.dropna(inplace=True)
            
            if len(df) < 1000:
                logger.warning(f"Insufficient data for {timeframe_key}: {len(df)} samples")
                return {}
                
            # Chọn features (loại bỏ OHLC gốc và targets)
            target_cols = [col for col in df.columns if 'direction_' in col or 'return_' in col or 'signal_' in col]
            feature_cols = [col for col in df.columns if col not in 
                          ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'] + target_cols]
            
            X = df[feature_cols].values
            
            # Chuẩn hóa features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Lưu scaler
            self.scalers[timeframe_key] = scaler
            
            # Tạo training data cho các tasks khác nhau
            training_data = {
                'X': X_scaled,
                'feature_names': feature_cols,
                'timestamps': df.index.values
            }
            
            # Direction prediction (classification)
            for horizon in [2, 4, 8]:  # Horizons chung
                if f'direction_{horizon}' in df.columns:
                    y_direction = df[f'direction_{horizon}'].values
                    training_data[f'y_direction_{horizon}'] = y_direction
                    
                if f'return_{horizon}' in df.columns:
                    y_return = df[f'return_{horizon}'].values
                    training_data[f'y_return_{horizon}'] = y_return
                    
                if f'strong_signal_{horizon}' in df.columns:
                    y_strong = df[f'strong_signal_{horizon}'].values
                    training_data[f'y_strong_{horizon}'] = y_strong
            
            # Lưu dữ liệu
            self.training_data[timeframe_key] = training_data
            
            # Lưu ra file
            with open(f"training/xauusdc/data/{timeframe_key}_training_data.pkl", 'wb') as f:
                pickle.dump(training_data, f)
                
            logger.info(f"Training data prepared for {timeframe_key}: {X_scaled.shape}")
            return training_data
            
        except Exception as e:
            logger.error(f"Training data preparation error for {timeframe_key}: {e}")
            return {}
            
    def create_neural_network(self, input_shape: int, task_type: str = 'classification') -> tf.keras.Model:
        """Tạo neural network cho training"""
        try:
            model = tf.keras.Sequential([
                # Input layer
                tf.keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                # Hidden layers
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                # Output layer
                tf.keras.layers.Dense(
                    3 if task_type == 'classification' else 1,
                    activation='softmax' if task_type == 'classification' else 'tanh'
                )
            ])
            
            # Compile model
            if task_type == 'classification':
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae']
                )
                
            return model
            
        except Exception as e:
            logger.error(f"Neural network creation error: {e}")
            return None
            
    def train_timeframe_models(self, timeframe_key: str) -> Dict:
        """Training models cho một timeframe"""
        try:
            logger.info(f"Training models for {timeframe_key}")
            
            training_data = self.training_data.get(timeframe_key)
            if not training_data:
                logger.error(f"No training data for {timeframe_key}")
                return {}
                
            X = training_data['X']
            results = {}
            
            # Train models cho các horizons khác nhau
            for horizon in [2, 4, 8]:
                if f'y_direction_{horizon}' not in training_data:
                    continue
                    
                logger.info(f"Training {timeframe_key} models for horizon {horizon}")
                
                # Direction prediction model
                y_direction = training_data[f'y_direction_{horizon}']
                
                # Convert labels (-1, 0, 1) to (0, 1, 2)
                y_direction_encoded = y_direction + 1
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_direction_encoded, test_size=0.2, random_state=42, stratify=y_direction_encoded
                )
                
                # Create and train classification model
                clf_model = self.create_neural_network(X.shape[1], 'classification')
                if clf_model is None:
                    continue
                    
                # Training callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
                    tf.keras.callbacks.ModelCheckpoint(
                        f"training/xauusdc/models/{timeframe_key}_direction_{horizon}.h5",
                        save_best_only=True
                    )
                ]
                
                # Train classification model
                history = clf_model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=64,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate classification model
                train_acc = clf_model.evaluate(X_train, y_train, verbose=0)[1]
                test_acc = clf_model.evaluate(X_test, y_test, verbose=0)[1]
                
                # Predictions for analysis
                y_pred = clf_model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)
                
                # Lưu model
                model_key = f"{timeframe_key}_direction_{horizon}"
                self.models[model_key] = clf_model
                
                results[f"direction_{horizon}"] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'model_path': f"training/xauusdc/models/{timeframe_key}_direction_{horizon}.h5"
                }
                
                # Train regression model nếu có data
                if f'y_return_{horizon}' in training_data:
                    y_return = training_data[f'y_return_{horizon}']
                    
                    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                        X, y_return, test_size=0.2, random_state=42
                    )
                    
                    reg_model = self.create_neural_network(X.shape[1], 'regression')
                    if reg_model is not None:
                        reg_callbacks = [
                            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                            tf.keras.callbacks.ModelCheckpoint(
                                f"training/xauusdc/models/{timeframe_key}_return_{horizon}.h5",
                                save_best_only=True
                            )
                        ]
                        
                        reg_model.fit(
                            X_train_reg, y_train_reg,
                            validation_data=(X_test_reg, y_test_reg),
                            epochs=100,
                            batch_size=64,
                            callbacks=reg_callbacks,
                            verbose=0
                        )
                        
                        train_mse = reg_model.evaluate(X_train_reg, y_train_reg, verbose=0)[0]
                        test_mse = reg_model.evaluate(X_test_reg, y_test_reg, verbose=0)[0]
                        
                        model_key_reg = f"{timeframe_key}_return_{horizon}"
                        self.models[model_key_reg] = reg_model
                        
                        results[f"return_{horizon}"] = {
                            'train_mse': train_mse,
                            'test_mse': test_mse,
                            'model_path': f"training/xauusdc/models/{timeframe_key}_return_{horizon}.h5"
                        }
                        
            logger.info(f"Training completed for {timeframe_key}")
            return results
            
        except Exception as e:
            logger.error(f"Training error for {timeframe_key}: {e}")
            return {}
            
    def evaluate_model_performance(self, timeframe_key: str) -> Dict:
        """Đánh giá hiệu suất model"""
        try:
            training_data = self.training_data.get(timeframe_key)
            if not training_data:
                return {}
                
            X = training_data['X']
            performance = {}
            
            for horizon in [2, 4, 8]:
                # Evaluate direction model
                model_key = f"{timeframe_key}_direction_{horizon}"
                if model_key in self.models:
                    model = self.models[model_key]
                    y_true = training_data.get(f'y_direction_{horizon}')
                    
                    if y_true is not None:
                        y_true_encoded = y_true + 1
                        y_pred = model.predict(X)
                        y_pred_classes = np.argmax(y_pred, axis=1)
                        
                        # Accuracy by class
                        from sklearn.metrics import classification_report, accuracy_score
                        accuracy = accuracy_score(y_true_encoded, y_pred_classes)
                        
                        # Trading-specific metrics
                        buy_signals = (y_pred_classes == 2)  # Class 2 = Buy (original 1)
                        sell_signals = (y_pred_classes == 0)  # Class 0 = Sell (original -1)
                        
                        buy_accuracy = accuracy_score(
                            y_true_encoded[buy_signals], 
                            y_pred_classes[buy_signals]
                        ) if np.sum(buy_signals) > 0 else 0
                        
                        sell_accuracy = accuracy_score(
                            y_true_encoded[sell_signals], 
                            y_pred_classes[sell_signals]
                        ) if np.sum(sell_signals) > 0 else 0
                        
                        performance[f"direction_{horizon}"] = {
                            'overall_accuracy': accuracy,
                            'buy_accuracy': buy_accuracy,
                            'sell_accuracy': sell_accuracy,
                            'buy_signals_count': np.sum(buy_signals),
                            'sell_signals_count': np.sum(sell_signals)
                        }
                        
                # Evaluate return model
                return_model_key = f"{timeframe_key}_return_{horizon}"
                if return_model_key in self.models:
                    return_model = self.models[return_model_key]
                    y_true_return = training_data.get(f'y_return_{horizon}')
                    
                    if y_true_return is not None:
                        y_pred_return = return_model.predict(X).flatten()
                        
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        
                        mse = mean_squared_error(y_true_return, y_pred_return)
                        mae = mean_absolute_error(y_true_return, y_pred_return)
                        r2 = r2_score(y_true_return, y_pred_return)
                        
                        # Direction accuracy of return predictions
                        direction_acc = np.mean(np.sign(y_true_return) == np.sign(y_pred_return))
                        
                        performance[f"return_{horizon}"] = {
                            'mse': mse,
                            'mae': mae,
                            'r2_score': r2,
                            'direction_accuracy': direction_acc
                        }
                        
            return performance
            
        except Exception as e:
            logger.error(f"Performance evaluation error for {timeframe_key}: {e}")
            return {}
            
    def run_comprehensive_training(self) -> Dict:
        """Chạy training toàn diện cho tất cả timeframes"""
        try:
            logger.info("Starting comprehensive XAU/USDc training")
            
            # Kết nối MT5
            if not self.connect_mt5():
                logger.error("Failed to connect to MT5")
                return {}
                
            all_results = {}
            
            # Training cho từng timeframe
            for timeframe_key in self.timeframes.keys():
                logger.info(f"Processing timeframe: {timeframe_key}")
                
                try:
                    # Chuẩn bị dữ liệu
                    training_data = self.prepare_training_data(timeframe_key)
                    if not training_data:
                        logger.warning(f"Skipping {timeframe_key} due to insufficient data")
                        continue
                        
                    # Training models
                    training_results = self.train_timeframe_models(timeframe_key)
                    
                    # Đánh giá performance
                    performance = self.evaluate_model_performance(timeframe_key)
                    
                    # Lưu performance metrics
                    self.performance_metrics[timeframe_key] = performance
                    
                    all_results[timeframe_key] = {
                        'training_results': training_results,
                        'performance_metrics': performance,
                        'data_samples': len(training_data['X']),
                        'feature_count': len(training_data['feature_names'])
                    }
                    
                    logger.info(f"Completed training for {timeframe_key}")
                    
                except Exception as e:
                    logger.error(f"Error processing {timeframe_key}: {e}")
                    continue
                    
            # Lưu kết quả tổng hợp
            with open("training/xauusdc/results/comprehensive_training_results.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
                
            # Tạo báo cáo tổng kết
            self.generate_training_report(all_results)
            
            logger.info("Comprehensive training completed")
            return all_results
            
        except Exception as e:
            logger.error(f"Comprehensive training error: {e}")
            return {}
        finally:
            mt5.shutdown()
            
    def generate_training_report(self, results: Dict):
        """Tạo báo cáo training"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'timeframes_trained': list(results.keys()),
                'summary': {}
            }
            
            total_models = 0
            best_performances = {}
            
            for tf, tf_results in results.items():
                performance = tf_results.get('performance_metrics', {})
                
                # Tìm best accuracy cho mỗi timeframe
                best_acc = 0
                for horizon in [2, 4, 8]:
                    direction_key = f"direction_{horizon}"
                    if direction_key in performance:
                        acc = performance[direction_key].get('overall_accuracy', 0)
                        if acc > best_acc:
                            best_acc = acc
                            
                best_performances[tf] = best_acc
                
                # Đếm models
                training_results = tf_results.get('training_results', {})
                total_models += len(training_results)
                
                report['summary'][tf] = {
                    'best_accuracy': best_acc,
                    'models_trained': len(training_results),
                    'data_samples': tf_results.get('data_samples', 0),
                    'features': tf_results.get('feature_count', 0)
                }
                
            # Overall statistics
            report['overall'] = {
                'total_models_trained': total_models,
                'timeframes_completed': len(results),
                'average_best_accuracy': np.mean(list(best_performances.values())) if best_performances else 0,
                'best_timeframe': max(best_performances.items(), key=lambda x: x[1])[0] if best_performances else None
            }
            
            # Lưu báo cáo
            with open("training/xauusdc/results/training_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            # In báo cáo
            print("\n" + "="*80)
            print("XAU/USDc MULTI-TIMEFRAME TRAINING REPORT")
            print("="*80)
            print(f"Training completed at: {report['timestamp']}")
            print(f"Symbol: {self.symbol}")
            print(f"Total models trained: {report['overall']['total_models_trained']}")
            print(f"Timeframes completed: {report['overall']['timeframes_completed']}/7")
            print(f"Average best accuracy: {report['overall']['average_best_accuracy']:.1%}")
            if report['overall']['best_timeframe']:
                print(f"Best performing timeframe: {report['overall']['best_timeframe']}")
                
            print(f"\nTimeframe Details:")
            for tf, summary in report['summary'].items():
                print(f"  {tf}:")
                print(f"    Best accuracy: {summary['best_accuracy']:.1%}")
                print(f"    Models trained: {summary['models_trained']}")
                print(f"    Data samples: {summary['data_samples']:,}")
                print(f"    Features: {summary['features']}")
                
            print("="*80)
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")


def main():
    """Main execution function"""
    
    # Tạo training system
    training_system = XAUUSDcMultiTimeframeTrainingSystem()
    
    print("🚀 Starting XAU/USDc Multi-Timeframe Training System")
    print("=" * 80)
    print("Timeframes: M1, M5, M15, M30, H1, H4, D1")
    print("Symbol: XAU/USDc (Gold/US Dollar cent)")
    print("Data source: MetaTrader 5")
    print("="*80)
    
    # Chạy training toàn diện
    results = training_system.run_comprehensive_training()
    
    if results:
        print("\n✅ Training completed successfully!")
        print(f"📊 Results saved to: training/xauusdc/results/")
        print(f"🤖 Models saved to: training/xauusdc/models/")
        print(f"💾 Data saved to: training/xauusdc/data/")
    else:
        print("\n❌ Training failed!")
        
    return results

if __name__ == "__main__":
    main() 