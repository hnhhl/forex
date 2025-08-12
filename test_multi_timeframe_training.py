#!/usr/bin/env python3
"""
TEST MULTI-TIMEFRAME TRAINING CHO HỆ THỐNG CHÍNH
Kiểm tra và training với dữ liệu từ M1 đến W1
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

def test_multi_timeframe_data_collection():
    """Test thu thập dữ liệu Multi-Timeframe"""
    print("🔍 TESTING MULTI-TIMEFRAME DATA COLLECTION")
    print("=" * 60)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return False
    
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
    
    symbol = "XAUUSDc"
    collected_data = {}
    
    print(f"📊 Collecting data for {symbol}...")
    
    for tf_name, tf_value in timeframes.items():
        try:
            # Get data for this timeframe
            rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 1000)
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                collected_data[tf_name] = df
                
                print(f"   ✓ {tf_name}: {len(df):,} records")
                print(f"     Date range: {df['time'].min()} to {df['time'].max()}")
                print(f"     Columns: {list(df.columns)}")
            else:
                print(f"   ❌ {tf_name}: No data available")
                
        except Exception as e:
            print(f"   ❌ {tf_name}: Error - {e}")
    
    mt5.shutdown()
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total timeframes collected: {len(collected_data)}/8")
    
    if len(collected_data) >= 4:
        print("   ✅ Sufficient data for Multi-Timeframe training")
        return collected_data
    else:
        print("   ❌ Insufficient data for Multi-Timeframe training")
        return None

def test_multi_timeframe_feature_engineering(data_dict):
    """Test feature engineering từ Multi-Timeframe data"""
    print("\n🔧 TESTING MULTI-TIMEFRAME FEATURE ENGINEERING")
    print("=" * 60)
    
    all_features = []
    feature_summary = {}
    
    for tf_name, data in data_dict.items():
        try:
            print(f"\n📊 Processing {tf_name} data...")
            
            if len(data) < 50:
                print(f"   ⚠️ Insufficient data ({len(data)} records)")
                continue
            
            # Calculate technical indicators
            features = {}
            
            # Basic price features
            features['price_change'] = data['close'].pct_change().iloc[-1]
            features['volatility'] = ((data['high'] - data['low']) / data['close']).iloc[-1]
            features['body_size'] = (abs(data['close'] - data['open']) / data['close']).iloc[-1]
            
            # Moving averages
            for period in [5, 10, 20]:
                if len(data) >= period:
                    sma = data['close'].rolling(period).mean()
                    features[f'sma_{period}_ratio'] = (data['close'].iloc[-1] / sma.iloc[-1] - 1)
            
            # RSI
            if len(data) >= 14:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features['rsi'] = rsi.iloc[-1] / 100.0  # Normalize
            
            # MACD
            if len(data) >= 26:
                ema12 = data['close'].ewm(span=12).mean()
                ema26 = data['close'].ewm(span=26).mean()
                macd = ema12 - ema26
                features['macd'] = macd.iloc[-1] / data['close'].iloc[-1]
            
            # Volume features
            if 'tick_volume' in data.columns and len(data) >= 20:
                vol_ma = data['tick_volume'].rolling(20).mean()
                if vol_ma.iloc[-1] != 0:
                    features['volume_ratio'] = data['tick_volume'].iloc[-1] / vol_ma.iloc[-1]
            
            # Trend strength
            if len(data) >= 20:
                trend = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
                features['trend_strength'] = trend
            
            # Support/Resistance
            if len(data) >= 50:
                recent_high = data['high'].rolling(20).max().iloc[-1]
                recent_low = data['low'].rolling(20).min().iloc[-1]
                resistance_dist = (recent_high - data['close'].iloc[-1]) / data['close'].iloc[-1]
                support_dist = (data['close'].iloc[-1] - recent_low) / data['close'].iloc[-1]
                features['resistance_distance'] = resistance_dist
                features['support_distance'] = support_dist
            
            # Clean features (remove NaN)
            clean_features = {k: v for k, v in features.items() 
                            if not (np.isnan(v) or np.isinf(v))}
            
            if clean_features:
                all_features.append(clean_features)
                feature_summary[tf_name] = {
                    'feature_count': len(clean_features),
                    'features': list(clean_features.keys())
                }
                
                print(f"   ✓ {len(clean_features)} features extracted")
                print(f"   Features: {list(clean_features.keys())[:5]}...")
            else:
                print(f"   ❌ No valid features extracted")
                
        except Exception as e:
            print(f"   ❌ Error processing {tf_name}: {e}")
    
    print(f"\n📊 FEATURE ENGINEERING SUMMARY:")
    for tf_name, summary in feature_summary.items():
        print(f"   {tf_name}: {summary['feature_count']} features")
    
    total_features = sum(len(f) for f in all_features)
    print(f"   Total features across all timeframes: {total_features}")
    
    return all_features, feature_summary

def test_main_system_integration():
    """Test tích hợp với hệ thống chính"""
    print("\n🚀 TESTING MAIN SYSTEM INTEGRATION")
    print("=" * 60)
    
    try:
        # Initialize main system
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        config.live_trading = False
        config.paper_trading = True
        
        system = UltimateXAUSystem(config)
        
        print("📊 Initializing ULTIMATE XAU SYSTEM...")
        
        # Test system status
        status = system.get_system_status()
        print(f"   System status: {status.get('status', 'Unknown')}")
        print(f"   Active systems: {status.get('active_systems', 0)}")
        print(f"   Version: {status.get('version', 'Unknown')}")
        
        # Test signal generation (which should use Multi-Timeframe if available)
        print("\n📈 Testing signal generation...")
        signal = system.generate_signal()
        
        print(f"   Signal generated: {signal.get('action', 'Unknown')}")
        print(f"   Confidence: {signal.get('confidence', 0):.2%}")
        print(f"   Components: {len(signal.get('components', {}))}")
        
        # Test if Multi-Timeframe data is being used
        if 'multi_timeframe' in str(signal).lower():
            print("   ✅ Multi-Timeframe analysis detected")
        else:
            print("   ⚠️ Multi-Timeframe analysis not clearly detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ System integration test failed: {e}")
        return False

def run_multi_timeframe_training_test():
    """Chạy test hoàn chỉnh Multi-Timeframe Training"""
    print("🚀 MULTI-TIMEFRAME TRAINING TEST FOR MAIN SYSTEM")
    print("=" * 80)
    
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'tests': {}
    }
    
    # Test 1: Data Collection
    print("\n1️⃣ STEP 1: Multi-Timeframe Data Collection")
    data_dict = test_multi_timeframe_data_collection()
    results['tests']['data_collection'] = {
        'success': data_dict is not None,
        'timeframes_collected': len(data_dict) if data_dict else 0
    }
    
    if not data_dict:
        print("❌ Cannot proceed without data")
        return results
    
    # Test 2: Feature Engineering
    print("\n2️⃣ STEP 2: Multi-Timeframe Feature Engineering")
    features, feature_summary = test_multi_timeframe_feature_engineering(data_dict)
    results['tests']['feature_engineering'] = {
        'success': len(features) > 0,
        'total_features': sum(len(f) for f in features),
        'timeframes_processed': len(feature_summary)
    }
    
    # Test 3: Main System Integration
    print("\n3️⃣ STEP 3: Main System Integration")
    integration_success = test_main_system_integration()
    results['tests']['system_integration'] = {
        'success': integration_success
    }
    
    # Final Assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 60)
    
    total_tests = len(results['tests'])
    passed_tests = sum(1 for test in results['tests'].values() if test['success'])
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        assessment = "✅ MULTI-TIMEFRAME TRAINING READY"
        recommendation = "Hệ thống chính đã sẵn sàng cho Multi-Timeframe Training"
    elif passed_tests >= 2:
        assessment = "⚠️ PARTIAL MULTI-TIMEFRAME SUPPORT"
        recommendation = "Cần khắc phục một số vấn đề để đạt hiệu quả tối ưu"
    else:
        assessment = "❌ MULTI-TIMEFRAME TRAINING NOT READY"
        recommendation = "Cần sửa chữa cơ bản trước khi tiến hành training"
    
    print(f"Assessment: {assessment}")
    print(f"Recommendation: {recommendation}")
    
    # Detailed results
    print(f"\n📊 DETAILED RESULTS:")
    for test_name, test_result in results['tests'].items():
        status = "✅" if test_result['success'] else "❌"
        print(f"   {status} {test_name}: {test_result}")
    
    # Save results
    results_file = f"real_training_results/multi_timeframe_test_{results['timestamp']}.json"
    os.makedirs('real_training_results', exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Results saved: {results_file}")
    
    return results

def main():
    print("🔍 MULTI-TIMEFRAME TRAINING CAPABILITY TEST")
    print("Kiểm tra khả năng training đa khung thời gian của hệ thống chính")
    print("=" * 80)
    
    results = run_multi_timeframe_training_test()
    
    # Summary
    if all(test['success'] for test in results['tests'].values()):
        print("\n🎉 HỆ THỐNG ĐÃ SẴN SÀNG CHO MULTI-TIMEFRAME TRAINING!")
    else:
        print("\n⚠️ HỆ THỐNG CẦN CẢI THIỆN MULTI-TIMEFRAME CAPABILITIES")

if __name__ == "__main__":
    main() 