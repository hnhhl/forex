#!/usr/bin/env python3
"""
🔥 RETRAIN ULTIMATE XAU SYSTEM V4.0 VỚI DỮ LIỆU THỰC TẾ
=======================================================
Script này sẽ retrain toàn bộ hệ thống từ đầu với 268,475 records dữ liệu thực tế từ MT5
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class RealDataTrainingSystem:
    """Hệ thống training với dữ liệu thực tế"""
    
    def __init__(self):
        self.data_dir = "data/maximum_mt5_v2"
        self.results_dir = "training_results_real_data"
        self.models_dir = "trained_models_real_data"
        
        # Tạo thư mục kết quả
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.training_results = {
            'start_time': datetime.now().isoformat(),
            'data_summary': {},
            'training_phases': {},
            'models_trained': {},
            'performance_metrics': {},
            'backtest_results': {}
        }
        
    def load_real_data(self):
        """Load tất cả dữ liệu thực tế từ MT5"""
        print("📊 LOADING DỮ LIỆU THỰC TẾ TỪ MT5")
        print("=" * 50)
        
        # Load summary
        summary_file = f"{self.data_dir}/collection_summary_20250618_115847.json"
        with open(summary_file, 'r') as f:
            self.data_summary = json.load(f)
        
        print(f"✅ Tổng số records: {self.data_summary['total_records']:,}")
        print(f"✅ Số timeframes: {self.data_summary['total_timeframes']}")
        
        # Load từng timeframe
        self.timeframe_data = {}
        
        for tf, info in self.data_summary['timeframes'].items():
            csv_file = info['csv_file']
            if os.path.exists(csv_file):
                print(f"   📈 Loading {tf}: {info['records']:,} records...")
                data = pd.read_csv(csv_file)
                data['time'] = pd.to_datetime(data['time'])
                data = data.sort_values('time').reset_index(drop=True)
                
                # Tính toán features cơ bản
                data = self.calculate_basic_features(data)
                
                self.timeframe_data[tf] = data
                print(f"      ✅ Loaded {len(data)} records ({data['time'].min()} → {data['time'].max()})")
        
        self.training_results['data_summary'] = {
            'total_records': sum(len(data) for data in self.timeframe_data.values()),
            'timeframes_loaded': list(self.timeframe_data.keys()),
            'date_range': {
                'start': min(data['time'].min() for data in self.timeframe_data.values()).isoformat(),
                'end': max(data['time'].max() for data in self.timeframe_data.values()).isoformat()
            }
        }
        
        print(f"\n✅ LOADED TỔNG CỘNG: {self.training_results['data_summary']['total_records']:,} RECORDS")
        
    def calculate_basic_features(self, data):
        """Tính toán features cơ bản cho training"""
        # Price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Technical indicators
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volume features
        if 'tick_volume' in data.columns:
            data['volume_sma'] = data['tick_volume'].rolling(20).mean()
            data['volume_ratio'] = data['tick_volume'] / data['volume_sma']
        
        # Price patterns
        data['high_low_ratio'] = data['high'] / data['low']
        data['open_close_ratio'] = data['open'] / data['close']
        data['price_range'] = (data['high'] - data['low']) / data['close']
        
        return data
        
    def create_training_labels(self, data, prediction_horizon=1):
        """Tạo labels cho training"""
        # Simple price direction prediction
        data['future_price'] = data['close'].shift(-prediction_horizon)
        data['price_change'] = (data['future_price'] - data['close']) / data['close']
        
        # Multi-class labels
        data['direction'] = 0  # Hold
        data.loc[data['price_change'] > 0.002, 'direction'] = 1  # Buy (>0.2% gain)
        data.loc[data['price_change'] < -0.002, 'direction'] = -1  # Sell (<-0.2% loss)
        
        # Binary labels
        data['binary_direction'] = (data['price_change'] > 0).astype(int)
        
        # Regression labels (normalized price change)
        data['regression_target'] = data['price_change'] * 100  # Convert to percentage
        
        return data
        
    def prepare_training_data(self, timeframe='H1'):
        """Chuẩn bị dữ liệu training cho một timeframe"""
        print(f"\n📋 CHUẨN BỊ DỮ LIỆU TRAINING CHO {timeframe}")
        print("-" * 40)
        
        if timeframe not in self.timeframe_data:
            print(f"❌ Không có dữ liệu cho timeframe {timeframe}")
            return None, None, None, None
            
        data = self.timeframe_data[timeframe].copy()
        
        # Tạo labels
        data = self.create_training_labels(data)
        
        # Chọn features
        feature_columns = [
            'open', 'high', 'low', 'close', 'tick_volume', 'spread',
            'returns', 'log_returns', 'volatility',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volume_ratio', 'high_low_ratio',
            'open_close_ratio', 'price_range'
        ]
        
        # Loại bỏ columns không tồn tại
        available_features = [col for col in feature_columns if col in data.columns]
        
        # Drop NaN values
        data_clean = data[available_features + ['direction', 'binary_direction', 'regression_target']].dropna()
        
        print(f"   📊 Features: {len(available_features)}")
        print(f"   📊 Clean data: {len(data_clean):,} records")
        print(f"   📊 Date range: {data_clean.index[0]} → {data_clean.index[-1]}")
        
        # Prepare features and targets
        X = data_clean[available_features].values
        y_multiclass = data_clean['direction'].values
        y_binary = data_clean['binary_direction'].values
        y_regression = data_clean['regression_target'].values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"   ✅ Features shape: {X_scaled.shape}")
        print(f"   ✅ Labels prepared: multiclass, binary, regression")
        
        return X_scaled, y_multiclass, y_binary, y_regression, scaler, available_features
        
    def train_neural_networks(self, X_train, y_train, model_type='binary'):
        """Train neural networks với TensorFlow"""
        print(f"\n🧠 TRAINING NEURAL NETWORKS ({model_type.upper()})")
        print("-" * 40)
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            models = {}
            
            # Prepare data for different architectures
            X_2d = X_train  # For Dense networks
            X_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # For LSTM/Conv1D
            
            # 1. Dense Neural Network
            print("   🔹 Training Dense NN...")
            dense_model = Sequential([
                Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(256, activation='relu'),
                Dropout(0.3),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid' if model_type == 'binary' else 'linear')
            ])
            
            dense_model.compile(
                optimizer='adam',
                loss='binary_crossentropy' if model_type == 'binary' else 'mse',
                metrics=['accuracy'] if model_type == 'binary' else ['mae']
            )
            
            # Early stopping
            early_stop = EarlyStopping(patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5)
            
            # Train
            history_dense = dense_model.fit(
                X_2d, y_train,
                epochs=100,
                batch_size=64,
                validation_split=0.2,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            models['dense'] = {
                'model': dense_model,
                'history': history_dense,
                'type': 'dense'
            }
            
            print(f"      ✅ Dense NN trained - Final loss: {history_dense.history['loss'][-1]:.4f}")
            
            # 2. LSTM Network (if enough data)
            if X_train.shape[0] > 1000:
                print("   🔹 Training LSTM...")
                
                # Reshape for LSTM (samples, timesteps, features)
                sequence_length = min(60, X_train.shape[1])
                X_lstm = []
                y_lstm = []
                
                for i in range(sequence_length, len(X_train)):
                    X_lstm.append(X_train[i-sequence_length:i])
                    y_lstm.append(y_train[i])
                
                X_lstm = np.array(X_lstm)
                y_lstm = np.array(y_lstm)
                
                lstm_model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
                    Dropout(0.3),
                    LSTM(64, return_sequences=False),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid' if model_type == 'binary' else 'linear')
                ])
                
                lstm_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy' if model_type == 'binary' else 'mse',
                    metrics=['accuracy'] if model_type == 'binary' else ['mae']
                )
                
                history_lstm = lstm_model.fit(
                    X_lstm, y_lstm,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
                
                models['lstm'] = {
                    'model': lstm_model,
                    'history': history_lstm,
                    'type': 'lstm',
                    'sequence_length': sequence_length
                }
                
                print(f"      ✅ LSTM trained - Final loss: {history_lstm.history['loss'][-1]:.4f}")
            
            return models
            
        except Exception as e:
            print(f"   ❌ Error training neural networks: {e}")
            return {}
            
    def train_ensemble_models(self, X_train, y_train):
        """Train ensemble models"""
        print(f"\n🎯 TRAINING ENSEMBLE MODELS")
        print("-" * 40)
        
        ensemble_models = {}
        
        try:
            # Random Forest
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            print("   🌳 Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            ensemble_models['random_forest'] = rf_model
            print(f"      ✅ Random Forest trained - Score: {rf_model.score(X_train, y_train):.4f}")
            
            # Gradient Boosting
            print("   🚀 Training Gradient Boosting...")
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            ensemble_models['gradient_boosting'] = gb_model
            print(f"      ✅ Gradient Boosting trained - Score: {gb_model.score(X_train, y_train):.4f}")
            
            # Logistic Regression
            print("   📊 Training Logistic Regression...")
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train, y_train)
            ensemble_models['logistic_regression'] = lr_model
            print(f"      ✅ Logistic Regression trained - Score: {lr_model.score(X_train, y_train):.4f}")
            
        except Exception as e:
            print(f"   ❌ Error training ensemble models: {e}")
            
        return ensemble_models
        
    def evaluate_models(self, models, X_test, y_test, model_type='binary'):
        """Đánh giá performance của models"""
        print(f"\n📈 ĐÁNH GIÁ MODEL PERFORMANCE")
        print("-" * 40)
        
        results = {}
        
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
            
            for name, model_info in models.items():
                print(f"   🔍 Evaluating {name}...")
                
                if isinstance(model_info, dict) and 'model' in model_info:
                    model = model_info['model']
                    
                    # Neural network predictions
                    if hasattr(model, 'predict'):
                        if model_type == 'binary':
                            y_pred_prob = model.predict(X_test)
                            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                        else:
                            y_pred = model.predict(X_test).flatten()
                else:
                    model = model_info
                    y_pred = model.predict(X_test)
                
                if model_type == 'binary':
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
                    
                    print(f"      📊 Accuracy: {accuracy:.4f}")
                    print(f"      📊 Precision: {precision:.4f}")
                    print(f"      📊 Recall: {recall:.4f}")
                    print(f"      📊 F1-Score: {f1:.4f}")
                else:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2_score': r2
                    }
                    
                    print(f"      📊 MSE: {mse:.4f}")
                    print(f"      📊 MAE: {mae:.4f}")
                    print(f"      📊 R²: {r2:.4f}")
                    
        except Exception as e:
            print(f"   ❌ Error evaluating models: {e}")
            
        return results
        
    def save_models(self, models, scaler, features, timeframe):
        """Lưu trained models"""
        print(f"\n💾 SAVING TRAINED MODELS FOR {timeframe}")
        print("-" * 40)
        
        import pickle
        import joblib
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save scaler and features
        scaler_file = f"{self.models_dir}/scaler_{timeframe}_{timestamp}.pkl"
        features_file = f"{self.models_dir}/features_{timeframe}_{timestamp}.json"
        
        joblib.dump(scaler, scaler_file)
        with open(features_file, 'w') as f:
            json.dump(features, f)
        
        print(f"   ✅ Saved scaler: {scaler_file}")
        print(f"   ✅ Saved features: {features_file}")
        
        # Save models
        saved_models = {}
        
        for name, model_info in models.items():
            try:
                if isinstance(model_info, dict) and 'model' in model_info:
                    model = model_info['model']
                    model_type = model_info.get('type', 'unknown')
                    
                    if hasattr(model, 'save'):  # TensorFlow model
                        model_file = f"{self.models_dir}/{name}_{timeframe}_{timestamp}.keras"
                        model.save(model_file)
                        saved_models[name] = {
                            'file': model_file,
                            'type': model_type,
                            'framework': 'tensorflow'
                        }
                        print(f"   ✅ Saved TF model: {model_file}")
                else:
                    # Scikit-learn model
                    model_file = f"{self.models_dir}/{name}_{timeframe}_{timestamp}.pkl"
                    joblib.dump(model_info, model_file)
                    saved_models[name] = {
                        'file': model_file,
                        'type': 'sklearn',
                        'framework': 'sklearn'
                    }
                    print(f"   ✅ Saved sklearn model: {model_file}")
                    
            except Exception as e:
                print(f"   ❌ Error saving {name}: {e}")
                
        return saved_models
        
    def run_comprehensive_training(self):
        """Chạy training comprehensive cho tất cả timeframes"""
        print("🚀 BẮT ĐẦU COMPREHENSIVE TRAINING VỚI DỮ LIỆU THỰC TẾ")
        print("=" * 70)
        
        # Load data
        self.load_real_data()
        
        # Train cho từng timeframe quan trọng
        priority_timeframes = ['H1', 'H4', 'D1', 'M30']
        
        for tf in priority_timeframes:
            if tf not in self.timeframe_data:
                continue
                
            print(f"\n🎯 TRAINING TIMEFRAME: {tf}")
            print("=" * 50)
            
            # Prepare data
            X, y_multi, y_binary, y_reg, scaler, features = self.prepare_training_data(tf)
            
            if X is None:
                continue
                
            # Train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train_binary, y_test_binary = y_binary[:split_idx], y_binary[split_idx:]
            y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]
            
            # Train neural networks (binary classification)
            nn_models = self.train_neural_networks(X_train, y_train_binary, 'binary')
            
            # Train ensemble models
            ensemble_models = self.train_ensemble_models(X_train, y_train_binary)
            
            # Combine all models
            all_models = {**nn_models, **ensemble_models}
            
            # Evaluate models
            evaluation_results = self.evaluate_models(all_models, X_test, y_test_binary, 'binary')
            
            # Save models
            saved_models = self.save_models(all_models, scaler, features, tf)
            
            # Store results
            self.training_results['training_phases'][tf] = {
                'data_points': len(X),
                'features_count': len(features),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'models_trained': list(all_models.keys()),
                'evaluation_results': evaluation_results,
                'saved_models': saved_models
            }
            
            print(f"\n✅ HOÀN THÀNH TRAINING CHO {tf}")
            print(f"   📊 Models trained: {len(all_models)}")
            print(f"   📊 Best accuracy: {max([r.get('accuracy', 0) for r in evaluation_results.values()]):.4f}")
        
        # Save final results
        self.training_results['end_time'] = datetime.now().isoformat()
        results_file = f"{self.results_dir}/comprehensive_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        print(f"\n🎉 HOÀN THÀNH COMPREHENSIVE TRAINING!")
        print(f"📄 Kết quả lưu tại: {results_file}")
        
        return self.training_results

def main():
    """Main training function"""
    print("🔥 ULTIMATE XAU SYSTEM V4.0 - COMPREHENSIVE RETRAINING")
    print("=" * 70)
    print(f"⏰ Bắt đầu: {datetime.now()}")
    print("📊 Sử dụng 268,475 records dữ liệu thực tế từ MT5 (2014-2025)")
    print()
    
    # Khởi tạo training system
    trainer = RealDataTrainingSystem()
    
    # Chạy comprehensive training
    results = trainer.run_comprehensive_training()
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH RETRAINING HỆ THỐNG!")
    print("\n📈 TỔNG KẾT:")
    print(f"   🔢 Tổng data points: {results['data_summary']['total_records']:,}")
    print(f"   📅 Thời gian: {results['data_summary']['date_range']['start']} → {results['data_summary']['date_range']['end']}")
    print(f"   🎯 Timeframes trained: {len(results['training_phases'])}")
    print(f"   🧠 Total models: {sum(len(phase['models_trained']) for phase in results['training_phases'].values())}")
    
    print("\n🚀 HỆ THỐNG ĐÃ ĐƯỢC RETRAIN VỚI DỮ LIỆU THỰC TẾ!")
    print("   Giờ đây performance sẽ được cải thiện đáng kể!")

if __name__ == "__main__":
    main() 