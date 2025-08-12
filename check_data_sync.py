# -*- coding: utf-8 -*-
"""Check Data Synchronization in AI3.0 System"""

import sys
import os
sys.path.append('src')

from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

def check_data_sync():
    print("KIỂM TRA ĐỒNG BỘ DATA TRONG AI3.0")
    print("="*60)
    
    try:
        # 1. Initialize system
        print("1. Khởi tạo hệ thống...")
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        # 2. Kiểm tra các nguồn data
        print("\n2. Kiểm tra các nguồn data...")
        
        # 2.1 MT5 Real-time data
        print("   2.1 MT5 Real-time Data:")
        if mt5.initialize():
            rates_mt5 = mt5.copy_rates_from_pos("XAUUSDc", mt5.TIMEFRAME_M1, 0, 10)
            if rates_mt5 is not None:
                df_mt5 = pd.DataFrame(rates_mt5)
                print(f"      ✅ MT5 connected: {len(df_mt5)} records")
                print(f"      Latest price: {df_mt5['close'].iloc[-1]}")
                print(f"      Latest time: {pd.to_datetime(df_mt5['time'].iloc[-1], unit='s')}")
                print(f"      Columns: {list(df_mt5.columns)}")
            else:
                print("      ❌ MT5 no data")
            mt5.shutdown()
        else:
            print("      ❌ MT5 connection failed")
        
        # 2.2 CSV Historical data
        print("\n   2.2 CSV Historical Data:")
        csv_files = [
            "data/working_free_data/XAUUSD_M1_realistic.csv",
            "data/maximum_mt5_v2/XAUUSDc_M1_20250618_115847.csv",
            "data/maximum_mt5_v2/XAUUSDc_H1_20250618_115847.csv"
        ]
        
        csv_data = {}
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    df_csv = pd.read_csv(csv_file)
                    csv_data[csv_file] = df_csv
                    print(f"      ✅ {csv_file}")
                    print(f"         Records: {len(df_csv):,}")
                    print(f"         Columns: {list(df_csv.columns)}")
                    if 'Date' in df_csv.columns:
                        print(f"         Date range: {df_csv['Date'].iloc[0]} to {df_csv['Date'].iloc[-1]}")
                    elif 'time' in df_csv.columns:
                        print(f"         Time range: {df_csv['time'].iloc[0]} to {df_csv['time'].iloc[-1]}")
                except Exception as e:
                    print(f"      ❌ {csv_file}: Error - {str(e)[:50]}")
            else:
                print(f"      ❌ {csv_file}: File not found")
        
        # 3. Kiểm tra data được sử dụng trong system
        print("\n3. Kiểm tra data được sử dụng trong system...")
        
        # Generate signal để xem system dùng data gì
        print("   Generating signal để trace data source...")
        signal = system.generate_signal("XAUUSDc")
        print(f"   Signal: {signal.get('action')} ({signal.get('confidence', 0):.1%})")
        
        # 4. Kiểm tra neural system data
        print("\n4. Kiểm tra Neural System data...")
        neural_system = system.system_manager.systems.get('NeuralNetworkSystem')
        if neural_system:
            print("   Neural System found")
            
            # Check scalers
            print(f"   Available scalers: {list(neural_system.feature_scalers.keys())}")
            
            # Check if scaler has been fitted
            if 'fixed_5_features' in neural_system.feature_scalers:
                scaler = neural_system.feature_scalers['fixed_5_features']
                if hasattr(scaler, 'data_min_'):
                    print(f"   Scaler fitted: ✅")
                    print(f"   Feature ranges: min={scaler.data_min_}, max={scaler.data_max_}")
                else:
                    print(f"   Scaler not fitted: ❌")
            else:
                print(f"   No fixed_5_features scaler: ❌")
        
        # 5. Kiểm tra data consistency
        print("\n5. Kiểm tra tính nhất quán của data...")
        
        # So sánh giá gần nhất từ các nguồn
        if rates_mt5 is not None and csv_data:
            mt5_latest_price = df_mt5['close'].iloc[-1]
            print(f"   MT5 latest price: {mt5_latest_price}")
            
            for file_path, df_csv in csv_data.items():
                if 'Close' in df_csv.columns:
                    csv_latest_price = df_csv['Close'].iloc[-1]
                    price_diff = abs(mt5_latest_price - csv_latest_price)
                    price_diff_pct = (price_diff / mt5_latest_price) * 100
                    
                    print(f"   {os.path.basename(file_path)}: {csv_latest_price}")
                    print(f"      Difference: {price_diff:.2f} ({price_diff_pct:.2f}%)")
                    
                    if price_diff_pct > 5:
                        print(f"      ⚠️ Large price difference detected!")
                    elif price_diff_pct > 1:
                        print(f"      ⚠️ Moderate price difference")
                    else:
                        print(f"      ✅ Price sync OK")
        
        # 6. Kiểm tra feature consistency
        print("\n6. Kiểm tra tính nhất quán của features...")
        
        # Test feature preparation với data khác nhau
        test_data_sources = []
        
        # MT5 data
        if rates_mt5 is not None:
            df_test_mt5 = pd.DataFrame(rates_mt5)
            if 'volume' not in df_test_mt5.columns:
                df_test_mt5['volume'] = df_test_mt5['tick_volume']
            test_data_sources.append(("MT5", df_test_mt5))
        
        # CSV data
        for file_path, df_csv in csv_data.items():
            if len(df_csv) > 100:
                df_test_csv = df_csv.tail(100).copy()
                # Standardize column names
                column_mapping = {
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                }
                for old_col, new_col in column_mapping.items():
                    if old_col in df_test_csv.columns:
                        df_test_csv[new_col] = df_test_csv[old_col]
                
                if 'volume' not in df_test_csv.columns and 'tick_volume' in df_test_csv.columns:
                    df_test_csv['volume'] = df_test_csv['tick_volume']
                
                test_data_sources.append((os.path.basename(file_path), df_test_csv))
        
        # So sánh features từ các nguồn
        feature_stats = {}
        for source_name, df_test in test_data_sources:
            try:
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df_test.columns for col in required_cols):
                    features = df_test[required_cols].values
                    stats = {
                        'mean': np.mean(features, axis=0),
                        'std': np.std(features, axis=0),
                        'min': np.min(features, axis=0),
                        'max': np.max(features, axis=0)
                    }
                    feature_stats[source_name] = stats
                    print(f"   {source_name}: ✅ Features OK")
                    print(f"      Price range: {stats['min'][3]:.2f} - {stats['max'][2]:.2f}")
                else:
                    print(f"   {source_name}: ❌ Missing columns")
            except Exception as e:
                print(f"   {source_name}: ❌ Error - {str(e)[:50]}")
        
        # 7. Phát hiện vấn đề đồng bộ
        print("\n7. Phát hiện vấn đề đồng bộ...")
        
        sync_issues = []
        
        # Check if multiple data sources have very different price ranges
        if len(feature_stats) > 1:
            price_ranges = {}
            for source, stats in feature_stats.items():
                price_ranges[source] = (stats['min'][3], stats['max'][2])  # close prices
            
            # Compare price ranges
            all_mins = [r[0] for r in price_ranges.values()]
            all_maxs = [r[1] for r in price_ranges.values()]
            
            min_diff = max(all_mins) - min(all_mins)
            max_diff = max(all_maxs) - min(all_maxs)
            
            if min_diff > 100 or max_diff > 100:  # 100 USD difference
                sync_issues.append(f"Large price range differences: min_diff={min_diff:.2f}, max_diff={max_diff:.2f}")
        
        # Check if scaler is not fitted properly
        if neural_system and 'fixed_5_features' not in neural_system.feature_scalers:
            sync_issues.append("Neural system scaler not properly initialized")
        
        # Check if confidence is too low (indicates data issues)
        if signal.get('confidence', 0) < 0.25:
            sync_issues.append(f"Very low confidence ({signal.get('confidence', 0):.1%}) indicates data/model issues")
        
        # 8. Kết luận và khuyến nghị
        print("\n8. KẾT LUẬN VÀ KHUYẾN NGHỊ:")
        
        if sync_issues:
            print("   ❌ PHÁT HIỆN VẤN ĐỀ ĐỒNG BỘ:")
            for issue in sync_issues:
                print(f"      • {issue}")
            
            print("\n   🔧 KHUYẾN NGHỊ:")
            print("      1. Đồng bộ hóa tất cả data sources")
            print("      2. Sử dụng một nguồn data chính thống nhất")
            print("      3. Retrain neural models với data đã đồng bộ")
            print("      4. Kiểm tra và update feature scalers")
            
            return False
        else:
            print("   ✅ DATA ĐỒNG BỘ TỐT")
            print("   Hệ thống có thể hoạt động ổn định")
            return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Kiểm tra đồng bộ data AI3.0")
    print("="*60)
    
    is_synced = check_data_sync()
    
    print("\n" + "="*60)
    if is_synced:
        print("✅ DATA ĐỒNG BỘ: Hệ thống sẵn sàng")
        print("Có thể tiến hành training/optimization")
    else:
        print("❌ DATA KHÔNG ĐỒNG BỘ: Cần fix trước")
        print("Phải giải quyết vấn đề đồng bộ trước khi training")
    
    print("="*60) 