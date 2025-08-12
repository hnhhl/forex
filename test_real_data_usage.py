#!/usr/bin/env python3
"""
Test script để kiểm tra xem Ultimate XAU System có đang sử dụng dữ liệu thực tế từ MT5 hay không
"""

import sys
import os
import pandas as pd
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_real_data_availability():
    """Kiểm tra dữ liệu thực tế có sẵn không"""
    print("🔍 KIỂM TRA DỮ LIỆU THỰC TẾ")
    print("=" * 50)
    
    data_dir = "data/maximum_mt5_v2"
    
    if not os.path.exists(data_dir):
        print("❌ Thư mục dữ liệu không tồn tại!")
        return False
    
    # Kiểm tra summary file
    summary_file = f"{data_dir}/collection_summary_20250618_115847.json"
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"✅ Tổng số records: {summary['total_records']:,}")
        print(f"✅ Số timeframes: {summary['total_timeframes']}")
        print(f"✅ Symbol: {summary['symbol']}")
        print()
        
        # Kiểm tra từng timeframe
        for tf, info in summary['timeframes'].items():
            print(f"   {tf}: {info['records']:,} records ({info['start_date']} → {info['end_date']})")
    
    # Test load một file dữ liệu
    h1_file = f"{data_dir}/XAUUSDc_H1_20250618_115847.csv"
    if os.path.exists(h1_file):
        print(f"\n📊 TEST LOAD DỮ LIỆU H1:")
        data = pd.read_csv(h1_file)
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Time range: {data['time'].min()} → {data['time'].max()}")
        print(f"   Sample data:")
        print(data.head(3).to_string())
        return True
    
    return False

def test_system_data_usage():
    """Test xem hệ thống có sử dụng dữ liệu thực tế không"""
    print("\n🧪 TEST HỆ THỐNG SỬ DỤNG DỮ LIỆU")
    print("=" * 50)
    
    try:
        from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        # Tạo hệ thống với config mặc định
        config = SystemConfig()
        system = UltimateXAUSystem(config)
        
        print("✅ Hệ thống khởi tạo thành công")
        
        # Test fallback data function
        print("\n📈 TEST FALLBACK DATA FUNCTION:")
        fallback_data = system._get_fallback_data("XAUUSDc")
        
        if not fallback_data.empty:
            print(f"   Data shape: {fallback_data.shape}")
            print(f"   Columns: {list(fallback_data.columns)}")
            
            # Kiểm tra xem có phải dữ liệu thực tế không
            if 'time' in fallback_data.columns:
                time_range = pd.to_datetime(fallback_data['time'])
                print(f"   Time range: {time_range.min()} → {time_range.max()}")
                
                # Kiểm tra xem có phải dữ liệu fake không (fake data có pattern đặc biệt)
                if len(fallback_data) > 100:
                    price_data = fallback_data['close'] if 'close' in fallback_data.columns else fallback_data.get('price', [])
                    if len(price_data) > 0:
                        price_std = price_data.std()
                        price_mean = price_data.mean()
                        
                        print(f"   Price mean: ${price_mean:.2f}")
                        print(f"   Price std: ${price_std:.2f}")
                        
                        # Dữ liệu fake thường có mean ~2050 và std ~10-20
                        # Dữ liệu thực có mean ~2000-2500 và std khác
                        if 2040 <= price_mean <= 2060 and 10 <= price_std <= 25:
                            print("   ⚠️  CÓ THỂ LÀ DỮ LIỆU FAKE (mean ~2050, std ~10-20)")
                        else:
                            print("   ✅ CÓ VẺ LÀ DỮ LIỆU THỰC TẾ")
                        
                        print(f"   Sample prices: {price_data.head(5).tolist()}")
        else:
            print("   ❌ Không lấy được dữ liệu")
        
        # Test comprehensive market data
        print("\n📊 TEST COMPREHENSIVE MARKET DATA:")
        market_data = system._get_comprehensive_market_data("XAUUSDc")
        
        if not market_data.empty:
            print(f"   Data shape: {market_data.shape}")
            print(f"   Columns: {list(market_data.columns)}")
            
            if 'close' in market_data.columns:
                prices = market_data['close']
                print(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
                print(f"   Latest price: ${prices.iloc[-1]:.2f}")
        else:
            print("   ❌ Không lấy được comprehensive market data")
            
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi test hệ thống: {e}")
        return False

def test_signal_generation():
    """Test tạo signal với dữ liệu thực tế"""
    print("\n🎯 TEST SIGNAL GENERATION")
    print("=" * 50)
    
    try:
        from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        config = SystemConfig()
        system = UltimateXAUSystem(config)
        
        # Generate signal
        signal = system.generate_signal("XAUUSDc")
        
        print("Signal generated:")
        print(f"   Action: {signal.get('action', 'N/A')}")
        print(f"   Strength: {signal.get('strength', 'N/A')}")
        print(f"   Prediction: {signal.get('prediction', 'N/A')}")
        print(f"   Confidence: {signal.get('confidence', 'N/A')}")
        print(f"   Systems used: {signal.get('systems_used', 'N/A')}")
        
        # Kiểm tra xem có error về data không
        if 'error' in signal:
            print(f"   ⚠️  Error: {signal['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi test signal generation: {e}")
        return False

def main():
    """Main test function"""
    print("🔥 ULTIMATE XAU SYSTEM - KIỂM TRA SỬ DỤNG DỮ LIỆU THỰC TẾ")
    print("=" * 70)
    print(f"Thời gian test: {datetime.now()}")
    print()
    
    # Test 1: Kiểm tra dữ liệu có sẵn
    data_available = test_real_data_availability()
    
    if not data_available:
        print("\n❌ KHÔNG TÌM THẤY DỮ LIỆU THỰC TẾ!")
        print("Hãy chạy get_maximum_mt5_data_v2.py để tải dữ liệu trước")
        return
    
    # Test 2: Kiểm tra hệ thống sử dụng dữ liệu
    system_ok = test_system_data_usage()
    
    # Test 3: Test signal generation
    if system_ok:
        test_signal_generation()
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH KIỂM TRA!")
    print("\nKẾT LUẬN:")
    print("- Nếu thấy 'CÓ VẺ LÀ DỮ LIỆU THỰC TẾ' → Hệ thống đã sử dụng dữ liệu thực")
    print("- Nếu thấy 'CÓ THỂ LÀ DỮ LIỆU FAKE' → Hệ thống vẫn dùng dữ liệu giả lập")
    print("- Kiểm tra logs để xem có load được file CSV thực tế không")

if __name__ == "__main__":
    main() 