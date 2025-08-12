#!/usr/bin/env python3
"""
Demo XAU/USDc Prediction System
Ultimate XAU Super System V4.0

Sử dụng các models đã training để dự đoán XAU/USDc real-time
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle
import json
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XAUUSDcPredictionSystem:
    """Hệ thống dự đoán XAU/USDc sử dụng trained models"""
    
    def __init__(self):
        self.symbol = "XAUUSDc"
        self.models = {}
        self.scalers = {}
        self.model_info = {
            'M15_dir_2': {'timeframe': 'M15', 'horizon': 2, 'accuracy': 0.840, 'prediction_minutes': 30},
            'M15_dir_4': {'timeframe': 'M15', 'horizon': 4, 'accuracy': 0.722, 'prediction_minutes': 60},
            'M15_dir_8': {'timeframe': 'M15', 'horizon': 8, 'accuracy': 0.621, 'prediction_minutes': 120},
            'M30_dir_2': {'timeframe': 'M30', 'horizon': 2, 'accuracy': 0.776, 'prediction_minutes': 60},
            'M30_dir_4': {'timeframe': 'M30', 'horizon': 4, 'accuracy': 0.633, 'prediction_minutes': 120},
            'M30_dir_8': {'timeframe': 'M30', 'horizon': 8, 'accuracy': 0.543, 'prediction_minutes': 240},
            'H1_dir_2': {'timeframe': 'H1', 'horizon': 2, 'accuracy': 0.671, 'prediction_minutes': 120},
            'H1_dir_4': {'timeframe': 'H1', 'horizon': 4, 'accuracy': 0.536, 'prediction_minutes': 240},
            'H4_dir_2': {'timeframe': 'H4', 'horizon': 2, 'accuracy': 0.460, 'prediction_minutes': 480},
            'D1_dir_2': {'timeframe': 'D1', 'horizon': 2, 'accuracy': 0.436, 'prediction_minutes': 2880}
        }
        
        self.timeframes = {
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
    def connect_mt5(self) -> bool:
        """Kết nối MT5"""
        try:
            if not mt5.initialize():
                return False
                
            # Kiểm tra symbol
            for symbol_test in ["XAUUSDc", "XAUUSD", "GOLD"]:
                symbol_info = mt5.symbol_info(symbol_test)
                if symbol_info is not None:
                    self.symbol = symbol_test
                    if not symbol_info.visible:
                        mt5.symbol_select(symbol_test, True)
                    logger.info(f"Connected to {self.symbol}")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
            
    def load_models(self) -> bool:
        """Load trained models và scalers"""
        try:
            models_loaded = 0
            
            for model_key, info in self.model_info.items():
                try:
                    # Load model
                    model_path = f"training/xauusdc/models/{model_key}.h5"
                    model = tf.keras.models.load_model(model_path)
                    self.models[model_key] = model
                    
                    # Load scaler
                    timeframe = info['timeframe']
                    scaler_path = f"training/xauusdc/data/{timeframe}_data.pkl"
                    with open(scaler_path, 'rb') as f:
                        data = pickle.load(f)
                        # Recreate scaler from training data
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        scaler.fit(data['X'])
                        self.scalers[timeframe] = scaler
                    
                    models_loaded += 1
                    logger.info(f"Loaded {model_key} (accuracy: {info['accuracy']:.1%})")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_key}: {e}")
                    continue
                    
            logger.info(f"Successfully loaded {models_loaded}/{len(self.model_info)} models")
            return models_loaded > 0
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return False
            
    def get_current_data(self, timeframe: str, bars: int = 200) -> pd.DataFrame:
        """Lấy dữ liệu hiện tại từ MT5"""
        try:
            tf_value = self.timeframes[timeframe]
            rates = mt5.copy_rates_from_pos(self.symbol, tf_value, 0, bars)
            
            if rates is None or len(rates) == 0:
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Data retrieval error: {e}")
            return pd.DataFrame()
            
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo features tương tự như khi training"""
        try:
            if len(df) < 100:
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
                
            # Volume analysis
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
            
            # Time features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Feature creation error: {e}")
            return df
            
    def make_prediction(self, model_key: str) -> Dict:
        """Tạo prediction cho một model"""
        try:
            if model_key not in self.models:
                return {}
                
            info = self.model_info[model_key]
            timeframe = info['timeframe']
            
            # Lấy dữ liệu hiện tại
            df = self.get_current_data(timeframe, 200)
            if df.empty:
                return {}
                
            # Tạo features
            df = self.create_features(df)
            df.dropna(inplace=True)
            
            if len(df) < 10:
                return {}
                
            # Chuẩn bị features cho prediction
            target_cols = [col for col in df.columns if 
                          any(x in col for x in ['direction_', 'return_', 'signal_'])]
            feature_cols = [col for col in df.columns if col not in 
                          ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'] + target_cols]
            
            # Lấy sample cuối cùng
            X = df[feature_cols].iloc[-1:].values
            
            # Scale features
            if timeframe in self.scalers:
                X_scaled = self.scalers[timeframe].transform(X)
            else:
                X_scaled = X
                
            # Prediction
            model = self.models[model_key]
            y_pred = model.predict(X_scaled, verbose=0)[0]
            
            # Convert prediction
            predicted_class = np.argmax(y_pred)
            confidence = np.max(y_pred)
            
            # Map class to direction
            direction_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            direction = direction_map[predicted_class]
            
            # Current price info
            current_price = df['close'].iloc[-1]
            current_time = df.index[-1]
            
            return {
                'model': model_key,
                'timeframe': timeframe,
                'horizon_periods': info['horizon'],
                'prediction_minutes': info['prediction_minutes'],
                'accuracy': info['accuracy'],
                'direction': direction,
                'confidence': confidence,
                'predicted_class': predicted_class,
                'class_probabilities': y_pred.tolist(),
                'current_price': current_price,
                'prediction_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'target_time': (current_time + pd.Timedelta(minutes=info['prediction_minutes'])).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {model_key}: {e}")
            return {}
            
    def get_ensemble_prediction(self) -> Dict:
        """Tạo ensemble prediction từ best models"""
        try:
            # Best models theo accuracy
            best_models = ['M15_dir_2', 'M30_dir_2', 'M15_dir_4', 'H1_dir_2']
            
            predictions = []
            for model_key in best_models:
                pred = self.make_prediction(model_key)
                if pred:
                    predictions.append(pred)
                    
            if not predictions:
                return {}
                
            # Weighted ensemble
            total_weight = 0
            weighted_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            for pred in predictions:
                weight = pred['accuracy'] * pred['confidence']
                weighted_scores[pred['direction']] += weight
                total_weight += weight
                
            # Normalize
            if total_weight > 0:
                for direction in weighted_scores:
                    weighted_scores[direction] /= total_weight
                    
            # Final decision
            final_direction = max(weighted_scores, key=weighted_scores.get)
            final_confidence = weighted_scores[final_direction]
            
            return {
                'ensemble_direction': final_direction,
                'ensemble_confidence': final_confidence,
                'weighted_scores': weighted_scores,
                'individual_predictions': predictions,
                'models_used': len(predictions),
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return {}
            
    def run_live_analysis(self) -> Dict:
        """Chạy phân tích live toàn diện"""
        try:
            print("🚀 XAU/USDc Live Prediction Analysis")
            print("=" * 60)
            
            if not self.connect_mt5():
                print("❌ Failed to connect to MT5")
                return {}
                
            if not self.load_models():
                print("❌ Failed to load models")
                return {}
                
            # Current market info
            current_tick = mt5.symbol_info_tick(self.symbol)
            if current_tick:
                print(f"📊 Current {self.symbol}: {current_tick.bid:.2f}")
                print(f"🕒 Server time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("\n" + "="*60)
            print("🤖 INDIVIDUAL MODEL PREDICTIONS")
            print("="*60)
            
            all_predictions = {}
            
            # Get predictions từ best models
            for model_key in sorted(self.model_info.keys(), key=lambda x: self.model_info[x]['accuracy'], reverse=True):
                pred = self.make_prediction(model_key)
                if pred:
                    all_predictions[model_key] = pred
                    
                    # Format output
                    direction_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
                    confidence_stars = '⭐' * int(pred['confidence'] * 5)
                    
                    print(f"\n{model_key} ({pred['timeframe']}):")
                    print(f"  {direction_emoji.get(pred['direction'], '⚪')} Prediction: {pred['direction']}")
                    print(f"  🎯 Confidence: {pred['confidence']:.1%} {confidence_stars}")
                    print(f"  📈 Model Accuracy: {pred['accuracy']:.1%}")
                    print(f"  ⏰ Target: {pred['prediction_minutes']} minutes ({pred['target_time']})")
                    
            # Ensemble prediction
            print(f"\n{'='*60}")
            print("🎪 ENSEMBLE PREDICTION")
            print("="*60)
            
            ensemble = self.get_ensemble_prediction()
            if ensemble:
                direction_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
                
                print(f"\n🎯 FINAL DECISION: {direction_emoji.get(ensemble['ensemble_direction'], '⚪')} {ensemble['ensemble_direction']}")
                print(f"🔥 Confidence: {ensemble['ensemble_confidence']:.1%}")
                print(f"🤖 Models used: {ensemble['models_used']}")
                
                print(f"\n📊 Weighted Scores:")
                for direction, score in ensemble['weighted_scores'].items():
                    emoji = direction_emoji.get(direction, '⚪')
                    stars = '⭐' * int(score * 5)
                    print(f"  {emoji} {direction}: {score:.1%} {stars}")
                    
            # Trading recommendations
            print(f"\n{'='*60}")
            print("💡 TRADING RECOMMENDATIONS")
            print("="*60)
            
            if ensemble and ensemble['ensemble_confidence'] > 0.6:
                direction = ensemble['ensemble_direction']
                confidence = ensemble['ensemble_confidence']
                
                if direction == 'BUY':
                    print(f"✅ STRONG BUY SIGNAL")
                    print(f"🎯 Confidence: {confidence:.1%}")
                    print(f"⏰ Best timeframes: M15 (30min), M30 (1hr)")
                    print(f"💰 Suggested position: Long XAU/USDc")
                    
                elif direction == 'SELL':
                    print(f"✅ STRONG SELL SIGNAL")
                    print(f"🎯 Confidence: {confidence:.1%}")
                    print(f"⏰ Best timeframes: M15 (30min), M30 (1hr)")
                    print(f"💰 Suggested position: Short XAU/USDc")
                    
                else:
                    print(f"⚠️ HOLD SIGNAL - Market uncertainty")
                    print(f"🎯 Confidence: {confidence:.1%}")
                    print(f"💡 Wait for clearer signals")
                    
            else:
                print(f"⚠️ LOW CONFIDENCE SIGNALS")
                print(f"💡 Wait for better market conditions")
                print(f"📊 Monitor for stronger signals")
                
            print("="*60)
            
            return {
                'individual_predictions': all_predictions,
                'ensemble_prediction': ensemble,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Live analysis error: {e}")
            return {}
        finally:
            mt5.shutdown()


def main():
    """Main demo function"""
    
    system = XAUUSDcPredictionSystem()
    results = system.run_live_analysis()
    
    # Save results
    if results:
        with open(f"live_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to live_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    return results

if __name__ == "__main__":
    main() 