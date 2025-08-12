#!/usr/bin/env python3
"""
ULTIMATE MT5 DATA COLLECTOR V2.0
Lấy dữ liệu MT5 tối đa có thể với xử lý lỗi tốt hơn
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time

class UltimateMT5DataCollector:
    def __init__(self):
        self.symbols = ["XAUUSD", "GOLD", "GOLD.", "#GOLD"]  # Thử nhiều tên symbol
        self.active_symbol = None
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5, 
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1
        }
        self.collected_data = {}
        
    def initialize_mt5(self):
        """Khởi tạo MT5 với xử lý lỗi"""
        print("🔄 Đang khởi tạo MT5...")
        
        if not mt5.initialize():
            print(f"❌ Lỗi khởi tạo MT5: {mt5.last_error()}")
            return False
            
        # Thông tin account
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Kết nối MT5 thành công!")
            print(f"   Account: {account_info.login}")
            print(f"   Server: {account_info.server}")
            print(f"   Company: {account_info.company}")
        else:
            print("⚠️ Không lấy được thông tin account")
            
        return True
        
    def find_working_symbol(self):
        """Tìm symbol GOLD hoạt động"""
        print("🔍 Đang tìm symbol GOLD...")
        
        # Lấy tất cả symbols
        symbols = mt5.symbols_get()
        if symbols is None:
            print("❌ Không lấy được danh sách symbols")
            return None
            
        # Tìm symbol chứa GOLD hoặc XAU
        gold_symbols = []
        for symbol in symbols:
            name = symbol.name.upper()
            if any(x in name for x in ['GOLD', 'XAU', 'XAUUSD']):
                gold_symbols.append(symbol.name)
                
        print(f"📋 Tìm thấy {len(gold_symbols)} symbols GOLD: {gold_symbols[:5]}...")
        
        # Test từng symbol
        for symbol_name in gold_symbols[:10]:  # Test 10 symbols đầu
            try:
                symbol_info = mt5.symbol_info(symbol_name)
                if symbol_info is None:
                    continue
                    
                # Thử select symbol
                if not symbol_info.visible:
                    if not mt5.symbol_select(symbol_name, True):
                        continue
                        
                # Test lấy dữ liệu
                rates = mt5.copy_rates_from_pos(symbol_name, mt5.TIMEFRAME_H1, 0, 10)
                if rates is not None and len(rates) > 0:
                    print(f"✅ Symbol hoạt động: {symbol_name}")
                    print(f"   Spread: {symbol_info.spread}")
                    print(f"   Digits: {symbol_info.digits}")
                    return symbol_name
                    
            except Exception as e:
                continue
                
        print("❌ Không tìm thấy symbol GOLD hoạt động")
        return None
        
    def get_maximum_historical_data(self, symbol, timeframe_name, timeframe_code):
        """Lấy dữ liệu lịch sử tối đa"""
        print(f"\n📊 Lấy dữ liệu {timeframe_name} cho {symbol}...")
        
        all_data = []
        
        # Thử các phương pháp lấy dữ liệu
        methods = [
            # Method 1: Từ hiện tại về quá khứ
            {'name': 'copy_rates_from_pos', 'count': 100000},
            {'name': 'copy_rates_from_pos', 'count': 50000},
            {'name': 'copy_rates_from_pos', 'count': 10000},
        ]
        
        for method in methods:
            try:
                if method['name'] == 'copy_rates_from_pos':
                    rates = mt5.copy_rates_from_pos(symbol, timeframe_code, 0, method['count'])
                    
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    print(f"✅ Lấy được {len(df)} records bằng {method['name']}")
                    print(f"   Từ: {df['time'].min()}")
                    print(f"   Đến: {df['time'].max()}")
                    
                    return df
                    
            except Exception as e:
                print(f"⚠️ Lỗi {method['name']}: {e}")
                continue
                
        # Method 2: Lấy theo ngày cụ thể
        print("🔄 Thử lấy dữ liệu theo khoảng thời gian...")
        
        # Thử từ 5 năm trước
        start_date = datetime.now() - timedelta(days=5*365)
        
        try:
            rates = mt5.copy_rates_range(symbol, timeframe_code, start_date, datetime.now())
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                print(f"✅ Lấy được {len(df)} records theo range")
                print(f"   Từ: {df['time'].min()}")
                print(f"   Đến: {df['time'].max()}")
                
                return df
                
        except Exception as e:
            print(f"⚠️ Lỗi copy_rates_range: {e}")
            
        print(f"❌ Không lấy được dữ liệu {timeframe_name}")
        return None
        
    def collect_all_data(self):
        """Thu thập tất cả dữ liệu"""
        if not self.initialize_mt5():
            return False
            
        # Tìm symbol hoạt động
        self.active_symbol = self.find_working_symbol()
        if not self.active_symbol:
            return False
            
        print(f"\n🚀 BẮT ĐẦU THU THẬP DỮ LIỆU CHO {self.active_symbol}")
        print("=" * 50)
        
        results = {}
        total_records = 0
        
        for tf_name, tf_code in self.timeframes.items():
            try:
                data = self.get_maximum_historical_data(self.active_symbol, tf_name, tf_code)
                
                if data is not None and len(data) > 0:
                    self.collected_data[tf_name] = data
                    records = len(data)
                    total_records += records
                    
                    results[tf_name] = {
                        'success': True,
                        'records': records,
                        'start_date': str(data['time'].min()),
                        'end_date': str(data['time'].max()),
                        'duration_days': (data['time'].max() - data['time'].min()).days,
                        'memory_mb': round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                    }
                    
                    print(f"✅ {tf_name}: {records:,} records")
                    
                else:
                    results[tf_name] = {'success': False, 'error': 'No data'}
                    print(f"❌ {tf_name}: Không có dữ liệu")
                    
            except Exception as e:
                results[tf_name] = {'success': False, 'error': str(e)}
                print(f"❌ {tf_name}: Lỗi {e}")
                
        print(f"\n🎯 TỔNG KẾT: {total_records:,} records từ {len(self.collected_data)} timeframes")
        return results
        
    def save_collected_data(self):
        """Lưu dữ liệu đã thu thập"""
        if not self.collected_data:
            print("❌ Không có dữ liệu để lưu")
            return None
            
        # Tạo thư mục
        os.makedirs('data/maximum_mt5_v2', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            'collection_time': timestamp,
            'symbol': self.active_symbol,
            'total_timeframes': len(self.collected_data),
            'total_records': 0,
            'timeframes': {}
        }
        
        print(f"\n💾 ĐANG LƯU DỮ LIỆU...")
        
        for tf_name, data in self.collected_data.items():
            records = len(data)
            summary['total_records'] += records
            
            # Lưu pickle (nhanh)
            pkl_file = f"data/maximum_mt5_v2/{self.active_symbol}_{tf_name}_{timestamp}.pkl"
            data.to_pickle(pkl_file)
            
            # Lưu CSV (dễ đọc) - chỉ lưu sample nếu quá lớn
            csv_file = f"data/maximum_mt5_v2/{self.active_symbol}_{tf_name}_{timestamp}.csv"
            if records > 50000:
                # Lưu sample 10000 records cuối
                data.tail(10000).to_csv(csv_file, index=False)
                csv_note = f"Sample 10,000 records cuối (từ {records:,} records)"
            else:
                data.to_csv(csv_file, index=False)
                csv_note = "Full data"
                
            file_size = round(os.path.getsize(pkl_file) / 1024 / 1024, 2)
            
            summary['timeframes'][tf_name] = {
                'records': records,
                'start_date': str(data['time'].min()),
                'end_date': str(data['time'].max()),
                'duration_days': (data['time'].max() - data['time'].min()).days,
                'pkl_file': pkl_file,
                'csv_file': csv_file,
                'csv_note': csv_note,
                'file_size_mb': file_size
            }
            
            print(f"   ✅ {tf_name}: {records:,} records -> {file_size}MB")
            
        # Lưu summary
        summary_file = f"data/maximum_mt5_v2/collection_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"\n🎉 HOÀN THÀNH!")
        print(f"📊 Tổng cộng: {summary['total_records']:,} records")
        print(f"📋 Báo cáo: {summary_file}")
        
        return summary
        
    def display_summary(self):
        """Hiển thị tóm tắt dữ liệu"""
        if not self.collected_data:
            print("Chưa có dữ liệu")
            return
            
        print(f"\n📈 TỔNG QUAN DỮ LIỆU - {self.active_symbol}")
        print("=" * 60)
        
        total_records = 0
        for tf_name, data in self.collected_data.items():
            records = len(data)
            total_records += records
            days = (data['time'].max() - data['time'].min()).days
            
            print(f"{tf_name:>4}: {records:>8,} records | {days:>4} days | "
                  f"{data['time'].min().strftime('%Y-%m-%d')} -> {data['time'].max().strftime('%Y-%m-%d')}")
            
        print("-" * 60)
        print(f"TỔNG: {total_records:>8,} records từ {len(self.collected_data)} timeframes")

def main():
    print("🔥 ULTIMATE MT5 DATA COLLECTOR V2.0 🔥")
    print("Lấy dữ liệu tối đa với xử lý lỗi tốt hơn")
    print("=" * 60)
    
    collector = UltimateMT5DataCollector()
    
    try:
        # Thu thập dữ liệu
        results = collector.collect_all_data()
        
        if results and any(r.get('success', False) for r in results.values()):
            # Hiển thị tóm tắt
            collector.display_summary()
            
            # Lưu dữ liệu
            summary = collector.save_collected_data()
            
            if summary:
                print(f"\n🎯 THÀNH CÔNG!")
                print(f"Symbol: {summary['symbol']}")
                print(f"Timeframes: {summary['total_timeframes']}")
                print(f"Total Records: {summary['total_records']:,}")
                
        else:
            print("❌ Không thu thập được dữ liệu")
            
    except Exception as e:
        print(f"❌ Lỗi tổng quát: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        mt5.shutdown()
        print("\n🔚 Đã đóng kết nối MT5")

if __name__ == "__main__":
    main() 