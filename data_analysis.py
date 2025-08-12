#!/usr/bin/env python3
"""
Data Analysis - Phan tich du lieu training chi tiet
"""
import pandas as pd
import os
from pathlib import Path

def analyze_training_data():
    print("="*80)
    print("PHÂN TÍCH DỮ LIỆU TRAINING CHI TIẾT")
    print("="*80)
    
    total_records = 0
    total_size_mb = 0
    
    # 1. Working Free Data
    print("1. WORKING FREE DATA:")
    working_path = "data/working_free_data"
    if os.path.exists(working_path):
        for file in sorted(os.listdir(working_path)):
            if file.endswith('.csv'):
                filepath = os.path.join(working_path, file)
                size_mb = os.path.getsize(filepath) / (1024*1024)
                
                try:
                    df = pd.read_csv(filepath)
                    records = len(df)
                    date_range = f"{df.iloc[0,0]} to {df.iloc[-1,0]}" if len(df) > 0 else "No data"
                    
                    print(f"   ✓ {file}:")
                    print(f"     - Size: {size_mb:.1f}MB")
                    print(f"     - Records: {records:,}")
                    print(f"     - Range: {date_range}")
                    
                    total_records += records
                    total_size_mb += size_mb
                    
                except Exception as e:
                    print(f"   ✗ {file}: Error reading - {e}")
    print()
    
    # 2. Maximum MT5 V2 Data
    print("2. MAXIMUM MT5 V2 DATA:")
    mt5_path = "data/maximum_mt5_v2"
    if os.path.exists(mt5_path):
        csv_files = [f for f in os.listdir(mt5_path) if f.endswith('.csv')]
        pkl_files = [f for f in os.listdir(mt5_path) if f.endswith('.pkl')]
        
        print(f"   CSV Files: {len(csv_files)}")
        print(f"   PKL Files: {len(pkl_files)}")
        
        for file in sorted(csv_files):
            filepath = os.path.join(mt5_path, file)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            
            try:
                df = pd.read_csv(filepath)
                records = len(df)
                
                print(f"   ✓ {file}: {size_mb:.1f}MB, {records:,} records")
                total_records += records
                total_size_mb += size_mb
                
            except Exception as e:
                print(f"   ✗ {file}: Error - {e}")
    print()
    
    # 3. Real Free Data
    print("3. REAL FREE DATA:")
    real_path = "data/real_free_data"
    if os.path.exists(real_path):
        csv_files = [f for f in os.listdir(real_path) if f.endswith('.csv')]
        print(f"   Files: {len(csv_files)}")
        
        for file in sorted(csv_files):
            filepath = os.path.join(real_path, file)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            
            try:
                df = pd.read_csv(filepath)
                records = len(df)
                print(f"   ✓ {file}: {size_mb:.1f}MB, {records:,} records")
                
                total_records += records
                total_size_mb += size_mb
                
            except Exception as e:
                print(f"   ✗ {file}: Error - {e}")
    print()
    
    # 4. Free Historical Data
    print("4. FREE HISTORICAL DATA:")
    hist_path = "data/free_historical_data"
    if os.path.exists(hist_path):
        csv_files = [f for f in os.listdir(hist_path) if f.endswith('.csv')]
        print(f"   Files: {len(csv_files)}")
        
        for file in sorted(csv_files):
            filepath = os.path.join(hist_path, file)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            
            try:
                df = pd.read_csv(filepath)
                records = len(df)
                print(f"   ✓ {file}: {size_mb:.1f}MB, {records:,} records")
                
                total_records += records
                total_size_mb += size_mb
                
            except Exception as e:
                print(f"   ✗ {file}: Error - {e}")
    print()
    
    # 5. Tổng kết
    print("="*80)
    print("TỔNG KẾT DỮ LIỆU:")
    print(f"   📊 Tổng records: {total_records:,}")
    print(f"   💾 Tổng dung lượng: {total_size_mb:.1f}MB")
    print(f"   📈 Trung bình: {total_records/total_size_mb:.0f} records/MB" if total_size_mb > 0 else "")
    print()
    
    # 6. Đánh giá chất lượng
    print("ĐÁNH GIÁ CHẤT LƯỢNG:")
    if total_records > 1000000:
        print("   🚀 EXCELLENT: >1M records - Đủ cho deep learning")
    elif total_records > 500000:
        print("   ✅ GOOD: >500K records - Tốt cho training")
    elif total_records > 100000:
        print("   ⚠️ MODERATE: >100K records - Cần thêm dữ liệu")
    else:
        print("   ❌ INSUFFICIENT: <100K records - Cần nhiều dữ liệu hơn")
    
    print()
    
    # 7. Khuyến nghị
    print("KHUYẾN NGHỊ:")
    if total_records < 500000:
        print("   📥 Cần download thêm dữ liệu lịch sử")
        print("   🔄 Sử dụng multiple timeframes để tăng data")
        print("   🎯 Focus vào M1 data cho maximum records")
    else:
        print("   ✅ Dữ liệu đủ cho training chất lượng cao")
        print("   🚀 Có thể bắt đầu training ngay")
    
    print("="*80)

if __name__ == "__main__":
    analyze_training_data() 