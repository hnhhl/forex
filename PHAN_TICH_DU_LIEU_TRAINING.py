#!/usr/bin/env python3
"""
Phân tích chi tiết dữ liệu training XAU/USDc
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_training_data():
    """Phân tích chi tiết dữ liệu đã sử dụng cho training"""
    
    print("📊 PHÂN TÍCH CHI TIẾT DỮ LIỆU TRAINING XAU/USDc")
    print("=" * 70)
    
    # Connect to MT5 để lấy thông tin symbol
    if not mt5.initialize():
        print("❌ Cannot connect to MT5")
        return
        
    symbol = "XAUUSDc"
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        symbol = "XAUUSD"  # Fallback
        symbol_info = mt5.symbol_info(symbol)
        
    if symbol_info:
        print(f"🔗 Symbol analyzed: {symbol}")
        print(f"📍 Spread: {symbol_info.spread}")
        print(f"💰 Contract size: {symbol_info.trade_contract_size}")
        print(f"📏 Tick size: {symbol_info.trade_tick_size}")
        print(f"💵 Tick value: {symbol_info.trade_tick_value}")
    
    timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    
    print(f"\n🕒 Thời gian phân tích: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    total_data_points = 0
    earliest_time = None
    latest_time = None
    
    for tf_name, tf_value in timeframes.items():
        print(f"\n📈 TIMEFRAME: {tf_name}")
        print("-" * 40)
        
        try:
            # Lấy dữ liệu tương tự như khi training
            rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 10000)
            
            if rates is None or len(rates) == 0:
                print(f"  ❌ Không có dữ liệu cho {tf_name}")
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Thông tin cơ bản
            start_time = df['time'].min()
            end_time = df['time'].max()
            duration = end_time - start_time
            
            print(f"  📅 Thời gian bắt đầu: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  📅 Thời gian kết thúc: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  ⏰ Khoảng thời gian: {duration.days} ngày, {duration.seconds//3600} giờ")
            print(f"  📊 Số lượng bars: {len(df):,}")
            print(f"  💹 Giá thấp nhất: {df['low'].min():.2f}")
            print(f"  💹 Giá cao nhất: {df['high'].max():.2f}")
            print(f"  📈 Giá đầu kỳ: {df['close'].iloc[0]:.2f}")
            print(f"  📈 Giá cuối kỳ: {df['close'].iloc[-1]:.2f}")
            print(f"  📊 Biến động tổng: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
            
            # Cập nhật thông tin tổng
            total_data_points += len(df)
            if earliest_time is None or start_time < earliest_time:
                earliest_time = start_time
            if latest_time is None or end_time > latest_time:
                latest_time = end_time
                
            # Phân tích theo năm
            df['year'] = df['time'].dt.year
            yearly_counts = df['year'].value_counts().sort_index()
            print(f"  📋 Phân bố theo năm:")
            for year, count in yearly_counts.items():
                percentage = (count / len(df)) * 100
                print(f"    {year}: {count:,} bars ({percentage:.1f}%)")
                
            # Phân tích theo tháng gần đây
            recent_data = df[df['time'] >= (end_time - timedelta(days=90))]
            if len(recent_data) > 0:
                recent_volatility = recent_data['close'].pct_change().std() * np.sqrt(252) * 100
                print(f"  📊 Volatility 3 tháng gần: {recent_volatility:.2f}%/năm")
                
        except Exception as e:
            print(f"  ❌ Lỗi xử lý {tf_name}: {e}")
            continue
    
    # Tóm tắt tổng
    print(f"\n{'='*70}")
    print("📈 TỔNG KẾT DỮ LIỆU TRAINING")
    print(f"{'='*70}")
    
    if earliest_time and latest_time:
        total_duration = latest_time - earliest_time
        print(f"🕒 Khoảng thời gian tổng: {total_duration.days} ngày ({total_duration.days/365.25:.1f} năm)")
        print(f"📅 Từ: {earliest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 Đến: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Tổng data points: {total_data_points:,}")
        print(f"💾 Dung lượng ước tính: {total_data_points * 8 * 67 / (1024*1024):.1f} MB")
    
    # Đọc thông tin từ training results
    try:
        with open("training/xauusdc/results/training_results.json", 'r') as f:
            training_results = json.load(f)
            
        print(f"\n📋 CHI TIẾT TRAINING RESULTS:")
        print("-" * 40)
        
        total_samples_used = 0
        total_models = 0
        
        for tf, data in training_results.items():
            samples = data.get('samples', 0)
            features = data.get('features', 0)
            results = data.get('results', {})
            models_count = len(results)
            
            total_samples_used += samples
            total_models += models_count
            
            print(f"  {tf}:")
            print(f"    📊 Samples processed: {samples:,}")
            print(f"    🔧 Features created: {features}")
            print(f"    🤖 Models trained: {models_count}")
            
            if results:
                accuracies = []
                for model_name, metrics in results.items():
                    test_acc = metrics.get('test_acc', 0)
                    accuracies.append(test_acc)
                    print(f"    📈 {model_name}: {test_acc:.1%} accuracy")
                    
                if accuracies:
                    avg_acc = np.mean(accuracies)
                    print(f"    🎯 Average accuracy: {avg_acc:.1%}")
        
        print(f"\n🏆 THỐNG KÊ TỔNG:")
        print(f"  📊 Total samples used: {total_samples_used:,}")
        print(f"  🤖 Total models trained: {total_models}")
        print(f"  🔧 Features per sample: 67")
        
    except Exception as e:
        print(f"❌ Không thể đọc training results: {e}")
    
    # Phân tích chất lượng dữ liệu
    print(f"\n🔍 CHẤT LƯỢNG DỮ LIỆU:")
    print("-" * 40)
    
    # Kiểm tra gaps và missing data
    for tf_name, tf_value in timeframes.items():
        if tf_name in ['M1', 'M5']:  # Skip heavy timeframes
            continue
            
        try:
            rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 1000)
            if rates is None:
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Check for gaps
            if tf_name == 'M15':
                expected_interval = 15
            elif tf_name == 'M30':
                expected_interval = 30
            elif tf_name == 'H1':
                expected_interval = 60
            elif tf_name == 'H4':
                expected_interval = 240
            elif tf_name == 'D1':
                expected_interval = 1440
            else:
                continue
                
            time_diffs = df['time'].diff().dt.total_seconds() / 60  # minutes
            normal_intervals = (time_diffs == expected_interval).sum()
            total_intervals = len(time_diffs) - 1
            
            if total_intervals > 0:
                quality_pct = (normal_intervals / total_intervals) * 100
                print(f"  {tf_name}: {quality_pct:.1f}% intervals are normal")
                
                # Find gaps
                gaps = time_diffs[time_diffs > expected_interval * 2]
                if len(gaps) > 0:
                    print(f"    ⚠️ {len(gaps)} gaps detected (>2x interval)")
                    
        except Exception as e:
            continue
    
    # Market conditions coverage
    print(f"\n📊 ĐIỀU KIỆN THỊ TRƯỜNG ĐƯỢC COVERAGE:")
    print("-" * 40)
    
    try:
        # Analyze M15 data for market conditions
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 5000)
        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['returns'] = df['close'].pct_change()
            
            # Volatility periods
            df['volatility'] = df['returns'].rolling(50).std()
            high_vol_pct = (df['volatility'] > df['volatility'].quantile(0.8)).sum() / len(df) * 100
            low_vol_pct = (df['volatility'] < df['volatility'].quantile(0.2)).sum() / len(df) * 100
            
            print(f"  🔥 High volatility periods: {high_vol_pct:.1f}%")
            print(f"  😴 Low volatility periods: {low_vol_pct:.1f}%")
            
            # Trend periods
            df['sma_20'] = df['close'].rolling(20).mean()
            df['trend'] = df['close'] > df['sma_20']
            uptrend_pct = df['trend'].sum() / len(df) * 100
            
            print(f"  📈 Uptrend periods: {uptrend_pct:.1f}%")
            print(f"  📉 Downtrend periods: {100-uptrend_pct:.1f}%")
            
            # Trading sessions coverage
            df['hour'] = df['time'].dt.hour
            asian_pct = ((df['hour'] >= 0) & (df['hour'] < 8)).sum() / len(df) * 100
            london_pct = ((df['hour'] >= 8) & (df['hour'] < 16)).sum() / len(df) * 100
            ny_pct = ((df['hour'] >= 16) & (df['hour'] < 24)).sum() / len(df) * 100
            
            print(f"  🌏 Asian session: {asian_pct:.1f}%")
            print(f"  🇬🇧 London session: {london_pct:.1f}%")
            print(f"  🇺🇸 NY session: {ny_pct:.1f}%")
            
    except Exception as e:
        print(f"  ❌ Cannot analyze market conditions: {e}")
    
    mt5.shutdown()
    print(f"\n{'='*70}")
    print("✅ PHÂN TÍCH HOÀN TẤT")
    print(f"{'='*70}")

if __name__ == "__main__":
    analyze_training_data() 