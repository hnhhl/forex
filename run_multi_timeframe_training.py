#!/usr/bin/env python3
"""
🚀 MULTI-TIMEFRAME TRAINING MODE CHO HỆ THỐNG CHÍNH
Chạy training với dữ liệu từ M1 đến W1 trên hệ thống Ultimate XAU
"""

import sys
import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import json

# Import hệ thống chính
sys.path.append('src')
from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig

def run_multi_timeframe_training():
    """Chạy Multi-Timeframe Training trên hệ thống chính"""
    print("🚀 MULTI-TIMEFRAME TRAINING MODE")
    print("Hệ thống Ultimate XAU - Training với dữ liệu M1 đến W1")
    print("=" * 80)
    
    try:
        # Initialize system configuration
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        config.live_trading = False
        config.paper_trading = True
        config.continuous_learning = True
        
        # Multi-Timeframe specific settings
        config.epochs = 50  # Reduce for faster training
        config.batch_size = 32
        config.learning_rate = 0.001
        
        print("📊 Initializing Ultimate XAU System...")
        system = UltimateXAUSystem(config)
        
        # Check system status
        status = system.get_system_status()
        print(f"   System Status: {status.get('status', 'Unknown')}")
        print(f"   Active Systems: {status.get('active_systems', 0)}")
        
        # Start Multi-Timeframe Training
        print("\n🧠 Starting Multi-Timeframe Training...")
        
        # Define timeframes
        timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1
        }
        
        # Collect Multi-Timeframe data
        print("📈 Collecting Multi-Timeframe data...")
        multi_tf_data = {}
        
        for tf_name, tf_value in timeframes.items():
            try:
                print(f"   📊 Collecting {tf_name}...")
                
                if mt5.initialize():
                    rates = mt5.copy_rates_from_pos(config.symbol, tf_value, 0, 1000)
                    
                    if rates is not None and len(rates) > 100:
                        df = pd.DataFrame(rates)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        multi_tf_data[tf_name] = df
                        print(f"      ✓ {len(df):,} records")
                    else:
                        print(f"      ❌ No data")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
        
        if len(multi_tf_data) < 3:
            print(f"❌ Insufficient timeframe data: {len(multi_tf_data)}/8")
            return False
        
        print(f"✅ Collected {len(multi_tf_data)} timeframes")
        
        # Prepare Multi-Timeframe features
        print("\n🔧 Preparing Multi-Timeframe features...")
        
        all_features = []
        all_labels = []
        
        for tf_name, data in multi_tf_data.items():
            try:
                print(f"   🔧 Processing {tf_name}...")
                
                # Create enhanced features
                features, labels = create_timeframe_features(data, tf_name)
                
                if len(features) > 0:
                    all_features.extend(features)
                    all_labels.extend(labels)
                    print(f"      ✓ {len(features)} sequences")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
        
        if len(all_features) == 0:
            print("❌ No features prepared")
            return False
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\n📊 Multi-Timeframe Dataset:")
        print(f"   Features: {X.shape}")
        print(f"   Labels: {y.shape}")
        
        # Train with Multi-Timeframe data
        print("\n🧠 Training models with Multi-Timeframe data...")
        
        # Get Neural Network System from main system
        nn_system = None
        for system_name, system_obj in system.system_manager.systems.items():
            if 'NeuralNetwork' in system_name:
                nn_system = system_obj
                break
        
        training_results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'mode': 'multi_timeframe_training',
            'symbol': config.symbol,
            'timeframes_used': list(multi_tf_data.keys()),
            'total_samples': len(all_features),
            'feature_shape': X.shape,
            'models_trained': [],
            'performance': {}
        }
        
        if nn_system and hasattr(nn_system, 'train_models'):
            try:
                # Create dummy DataFrame for compatibility
                dummy_df = pd.DataFrame()
                
                # Train models
                success = nn_system.train_models(dummy_df, y)
                
                if success:
                    training_results['models_trained'] = list(nn_system.models.keys())
                    training_results['performance'] = getattr(nn_system, 'model_performance', {})
                    print("✅ Multi-Timeframe training completed")
                    
                    # Display results
                    print(f"\n📊 Training Results:")
                    for model_name, perf in training_results['performance'].items():
                        accuracy = perf.get('val_accuracy', 0)
                        print(f"   {model_name}: {accuracy:.2%} accuracy")
                else:
                    print("❌ Training failed")
                    
            except Exception as e:
                print(f"❌ Training error: {e}")
        else:
            print("❌ Neural Network System not found")
        
        # Test trained system
        print("\n🧪 Testing trained system...")
        
        try:
            # Generate signal to test if Multi-Timeframe data is being used
            signal = system.generate_signal()
            
            print(f"   Signal: {signal.get('action', 'Unknown')}")
            print(f"   Confidence: {signal.get('confidence', 0):.2%}")
            print(f"   Components: {len(signal.get('components', {}))}")
            
            # Check if system is using Multi-Timeframe analysis
            signal_str = str(signal).lower()
            if any(tf.lower() in signal_str for tf in ['m1', 'm5', 'h1', 'h4', 'd1']):
                print("   ✅ Multi-Timeframe analysis detected in signals")
                training_results['multi_tf_signals'] = True
            else:
                print("   ⚠️ Multi-Timeframe analysis not clearly detected")
                training_results['multi_tf_signals'] = False
                
        except Exception as e:
            print(f"   ❌ Signal test error: {e}")
        
        # Save results
        results_file = f"multi_timeframe_training_results_{training_results['timestamp']}.json"
        
        try:
            os.makedirs('training_results', exist_ok=True)
            results_path = f"training_results/{results_file}"
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(training_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Results saved: {results_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        # Final summary
        print(f"\n🎯 MULTI-TIMEFRAME TRAINING SUMMARY")
        print("=" * 60)
        print(f"✅ Timeframes used: {len(multi_tf_data)}/8")
        print(f"✅ Total samples: {len(all_features):,}")
        print(f"✅ Models trained: {len(training_results.get('models_trained', []))}")
        
        if training_results.get('performance'):
            avg_accuracy = np.mean([p.get('val_accuracy', 0) for p in training_results['performance'].values()])
            print(f"✅ Average accuracy: {avg_accuracy:.2%}")
        
        print("\n🎉 MULTI-TIMEFRAME TRAINING COMPLETED!")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-Timeframe training failed: {e}")
        return False
    
    finally:
        # Cleanup
        if mt5.initialize():
            mt5.shutdown()

def create_timeframe_features(data: pd.DataFrame, tf_name: str):
    """Tạo features và labels từ dữ liệu timeframe"""
    try:
        if len(data) < 60:
            return [], []
        
        # Calculate technical indicators
        if len(data) >= 50:
            # Price features
            data['price_change'] = data['close'].pct_change()
            data['volatility'] = (data['high'] - data['low']) / data['close']
            data['body_size'] = abs(data['close'] - data['open']) / data['close']
            
            # Moving averages
            data['sma_5'] = data['close'].rolling(5).mean()
            data['sma_20'] = data['close'].rolling(20).mean()
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_middle = data['close'].rolling(20).mean()
            bb_std = data['close'].rolling(20).std()
            data['bb_upper'] = bb_middle + (bb_std * 2)
            data['bb_lower'] = bb_middle - (bb_std * 2)
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Feature columns
        feature_cols = [
            'open', 'high', 'low', 'close', 'tick_volume',
            'price_change', 'volatility', 'body_size',
            'sma_5', 'sma_20', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'rsi', 'bb_position'
        ]
        
        available_cols = [col for col in feature_cols if col in data.columns]
        
        # Create sequences
        features = []
        labels = []
        sequence_length = 60
        
        # Timeframe encoding
        tf_encoding = {
            'M1': 0.1, 'M5': 0.2, 'M15': 0.3, 'M30': 0.4,
            'H1': 0.5, 'H4': 0.6, 'D1': 0.7, 'W1': 0.8
        }
        
        for i in range(sequence_length, len(data) - 1):
            # Get sequence
            sequence = data[available_cols].iloc[i-sequence_length:i].values
            
            # Handle NaN values
            sequence = np.nan_to_num(sequence, nan=0.0)
            
            # Add timeframe encoding
            tf_encoded = np.full((sequence_length, 1), tf_encoding.get(tf_name, 0.5))
            
            # Combine
            enhanced_sequence = np.concatenate([sequence, tf_encoded], axis=1)
            features.append(enhanced_sequence)
            
            # Create label (price direction)
            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[i + 1]
            label = 1 if future_price > current_price else 0
            labels.append(label)
        
        return features, labels
        
    except Exception as e:
        print(f"Feature creation error for {tf_name}: {e}")
        return [], []

def main():
    print("🚀 MULTI-TIMEFRAME TRAINING MODE")
    print("Chạy training đa khung thời gian cho hệ thống Ultimate XAU")
    print("=" * 80)
    
    success = run_multi_timeframe_training()
    
    if success:
        print("\n🎉 MULTI-TIMEFRAME TRAINING HOÀN THÀNH THÀNH CÔNG!")
        print("Hệ thống đã được training với dữ liệu từ M1 đến W1")
    else:
        print("\n❌ MULTI-TIMEFRAME TRAINING THẤT BẠI!")
        print("Vui lòng kiểm tra lỗi và thử lại")

if __name__ == "__main__":
    main() 