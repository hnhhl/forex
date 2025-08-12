#!/usr/bin/env python3
"""
Script lấy dữ liệu MT5 tối đa có thể
Hỗ trợ nhiều timeframe và thời gian dài nhất
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time

class MaximumMT5DataCollector:
    def __init__(self):
        self.symbol = "XAUUSD"
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
        self.max_data = {}
        
    def initialize_mt5(self):
        """Khởi tạo kết nối MT5"""
        if not mt5.initialize():
            print(f"❌ Lỗi khởi tạo MT5: {mt5.last_error()}")
            return False
            
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Kết nối MT5 thành công - Account: {account_info.login}")
        else:
            print("⚠️ Không lấy được thông tin account")
            
        return True
        
    def get_symbol_info(self):
        """Lấy thông tin symbol"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"❌ Không tìm thấy symbol {self.symbol}")
            return False
            
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                print(f"❌ Không thể select symbol {self.symbol}")
                return False
                
        print(f"✅ Symbol {self.symbol} sẵn sàng")
        return True
        
    def find_earliest_data(self, timeframe):
        """Tìm dữ liệu sớm nhất có thể"""
        print(f"🔍 Tìm dữ liệu sớm nhất cho {timeframe}...")
        
        # Thử từ 10 năm trước
        start_years = [10, 8, 5, 3, 2, 1]
        
        for years in start_years:
            start_date = datetime.now() - timedelta(days=years*365)
            
            rates = mt5.copy_rates_from(
                self.symbol,
                self.timeframes[timeframe],
                start_date,
                100  # Chỉ lấy 100 records để test
            )
            
            if rates is not None and len(rates) > 0:
                earliest_time = pd.to_datetime(rates[0]['time'], unit='s')
                print(f"✅ {timeframe}: Dữ liệu sớm nhất từ {earliest_time}")
                return start_date
                
        # Nếu không tìm thấy, dùng 1 năm
        return datetime.now() - timedelta(days=365)
        
    def get_maximum_data_for_timeframe(self, timeframe):
        """Lấy dữ liệu tối đa cho một timeframe"""
        print(f"\n📊 Đang lấy dữ liệu tối đa cho {timeframe}...")
        
        # Tìm điểm bắt đầu sớm nhất
        start_date = self.find_earliest_data(timeframe)
        end_date = datetime.now()
        
        all_data = []
        current_date = start_date
        batch_size = 100000  # Lấy từng batch lớn
        
        while current_date < end_date:
            try:
                rates = mt5.copy_rates_from(
                    self.symbol,
                    self.timeframes[timeframe],
                    current_date,
                    batch_size
                )
                
                if rates is None or len(rates) == 0:
                    print(f"⚠️ Không có dữ liệu từ {current_date}")
                    current_date += timedelta(days=30)  # Skip 30 days
                    continue
                    
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                all_data.append(df)
                
                # Update current_date to last record time
                last_time = df['time'].iloc[-1]
                current_date = last_time + timedelta(minutes=1)
                
                print(f"📈 {timeframe}: Đã lấy {len(df)} records, tổng: {sum(len(d) for d in all_data)}")
                
                # Tránh quá tải
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Lỗi lấy dữ liệu {timeframe}: {e}")
                current_date += timedelta(days=1)
                continue
                
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['time']).sort_values('time')
            
            print(f"✅ {timeframe}: Tổng cộng {len(combined_df)} records")
            print(f"   Từ: {combined_df['time'].min()}")
            print(f"   Đến: {combined_df['time'].max()}")
            
            return combined_df
        else:
            print(f"❌ Không lấy được dữ liệu cho {timeframe}")
            return None
            
    def collect_all_maximum_data(self):
        """Lấy dữ liệu tối đa cho tất cả timeframes"""
        if not self.initialize_mt5():
            return False
            
        if not self.get_symbol_info():
            return False
            
        print(f"\n🚀 Bắt đầu lấy dữ liệu tối đa cho {self.symbol}")
        print(f"Timeframes: {list(self.timeframes.keys())}")
        
        results = {}
        
        for tf_name in self.timeframes.keys():
            try:
                data = self.get_maximum_data_for_timeframe(tf_name)
                if data is not None:
                    self.max_data[tf_name] = data
                    results[tf_name] = {
                        'records': len(data),
                        'start_date': str(data['time'].min()),
                        'end_date': str(data['time'].max()),
                        'file_size_mb': round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                    }
                else:
                    results[tf_name] = {'error': 'No data available'}
                    
            except Exception as e:
                print(f"❌ Lỗi xử lý {tf_name}: {e}")
                results[tf_name] = {'error': str(e)}
                
        return results
        
    def save_maximum_data(self):
        """Lưu dữ liệu tối đa"""
        if not self.max_data:
            print("❌ Không có dữ liệu để lưu")
            return
            
        # Tạo thư mục
        os.makedirs('data/maximum_mt5', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            'collection_time': timestamp,
            'symbol': self.symbol,
            'total_timeframes': len(self.max_data),
            'timeframes': {}
        }
        
        total_records = 0
        
        for tf_name, data in self.max_data.items():
            # Lưu từng timeframe
            filename = f"data/maximum_mt5/{self.symbol}_{tf_name}_{timestamp}.pkl"
            data.to_pickle(filename)
            
            # Lưu CSV cho dễ đọc
            csv_filename = f"data/maximum_mt5/{self.symbol}_{tf_name}_{timestamp}.csv"
            data.to_csv(csv_filename, index=False)
            
            records = len(data)
            total_records += records
            
            summary['timeframes'][tf_name] = {
                'records': records,
                'start_date': str(data['time'].min()),
                'end_date': str(data['time'].max()),
                'pkl_file': filename,
                'csv_file': csv_filename,
                'file_size_mb': round(os.path.getsize(filename) / 1024 / 1024, 2)
            }
            
            print(f"💾 Đã lưu {tf_name}: {records:,} records -> {filename}")
            
        summary['total_records'] = total_records
        
        # Lưu summary
        summary_file = f"data/maximum_mt5/collection_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f"\n🎉 HOÀN THÀNH! Tổng cộng: {total_records:,} records")
        print(f"📋 Báo cáo chi tiết: {summary_file}")
        
        return summary
        
    def get_data_statistics(self):
        """Thống kê dữ liệu đã lấy"""
        if not self.max_data:
            return "Chưa có dữ liệu"
            
        stats = {}
        total_records = 0
        
        for tf_name, data in self.max_data.items():
            records = len(data)
            total_records += records
            
            stats[tf_name] = {
                'records': f"{records:,}",
                'start': str(data['time'].min()),
                'end': str(data['time'].max()),
                'duration_days': (data['time'].max() - data['time'].min()).days,
                'memory_mb': round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            
        stats['TOTAL'] = f"{total_records:,} records"
        return stats

def main():
    print("🔥 ULTIMATE MT5 DATA COLLECTOR - MAXIMUM MODE 🔥")
    print("=" * 60)
    
    collector = MaximumMT5DataCollector()
    
    try:
        # Lấy dữ liệu tối đa
        results = collector.collect_all_maximum_data()
        
        if results:
            print("\n📊 KẾT QUẢ THU THẬP:")
            for tf, result in results.items():
                if 'error' in result:
                    print(f"❌ {tf}: {result['error']}")
                else:
                    print(f"✅ {tf}: {result['records']:,} records ({result['file_size_mb']}MB)")
                    
            # Lưu dữ liệu
            summary = collector.save_maximum_data()
            
            # Thống kê
            stats = collector.get_data_statistics()
            print(f"\n📈 THỐNG KÊ TỔNG QUAN:")
            for tf, stat in stats.items():
                if tf == 'TOTAL':
                    print(f"🎯 {tf}: {stat}")
                else:
                    print(f"   {tf}: {stat['records']} records, {stat['duration_days']} days")
                    
        else:
            print("❌ Không lấy được dữ liệu")
            
    except Exception as e:
        print(f"❌ Lỗi tổng quát: {e}")
        
    finally:
        mt5.shutdown()
        print("\n🔚 Đã ngắt kết nối MT5")

if __name__ == "__main__":
    main() 