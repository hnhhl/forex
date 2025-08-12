#!/usr/bin/env python3
"""
🆓 FREE HISTORICAL DATA DOWNLOADER
Download miễn phí từ HistData.com cho XAU/USD M1 data
"""

import requests
import pandas as pd
import zipfile
import os
from datetime import datetime, timedelta
import time
import json

class FreeHistoricalDataDownloader:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.base_url = "https://www.histdata.com/download-free-forex-historical-data"
        self.output_dir = "data/free_historical_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_histdata_xauusd(self):
        """Download XAU/USD M1 data từ HistData.com"""
        print("🆓 DOWNLOADING FREE XAU/USD DATA FROM HISTDATA.COM")
        print("=" * 60)
        
        # HistData.com có data theo tháng
        # Thử download data từ 2020-2024 (4-5 năm gần nhất)
        years = [2020, 2021, 2022, 2023, 2024]
        months = list(range(1, 13))
        
        downloaded_files = []
        total_records = 0
        
        for year in years:
            for month in months:
                if year == 2024 and month > 12:  # Không vượt quá tháng hiện tại
                    break
                    
                print(f"\n📅 Attempting to download {year}-{month:02d}...")
                
                # URL pattern cho HistData (cần research thêm)
                # Thường là: https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/XAUUSD/2024/1
                url = f"https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/{self.symbol}/{year}/{month}"
                
                try:
                    # Simulate download (cần implement thực tế)
                    print(f"   🔍 Checking: {url}")
                    time.sleep(1)  # Tránh spam server
                    
                    # Placeholder - sẽ implement thực tế
                    print(f"   ⏳ Would download from: {url}")
                    
                except Exception as e:
                    print(f"   ❌ Error downloading {year}-{month}: {e}")
                    continue
        
        return {
            'files_downloaded': len(downloaded_files),
            'total_records': total_records,
            'files': downloaded_files
        }
    
    def download_forexsb_data(self):
        """Download từ ForexSB.com - có API trực tiếp"""
        print("\n🆓 DOWNLOADING FROM FOREXSB.COM")
        print("=" * 40)
        
        # ForexSB có API trực tiếp
        base_url = "https://data.forexsb.com"
        
        # Các timeframes cần thiết
        timeframes = ['M1', 'M5', 'M15', 'M30']
        
        results = {}
        
        for tf in timeframes:
            print(f"\n📊 Downloading {tf} data...")
            
            try:
                # API call để lấy data (cần research API endpoint)
                api_url = f"{base_url}/api/data"
                
                # Simulate API call
                print(f"   🔍 API URL: {api_url}")
                print(f"   📈 Timeframe: {tf}")
                print(f"   💰 Symbol: XAU/USD")
                
                # Placeholder response
                results[tf] = {
                    'status': 'simulated',
                    'records': 'unknown',
                    'file': f"{self.output_dir}/XAUUSD_{tf}_forexsb.csv"
                }
                
                print(f"   ✅ {tf} data simulated")
                
            except Exception as e:
                print(f"   ❌ Error downloading {tf}: {e}")
                results[tf] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def create_sample_data(self):
        """Tạo sample data để test format"""
        print("\n🔧 CREATING SAMPLE DATA FOR TESTING")
        print("=" * 45)
        
        # Tạo sample M1 data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)  # 1 tuần sample
        
        dates = []
        current = start_date
        
        while current < end_date:
            # Skip weekends
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(current)
            current += timedelta(minutes=1)
        
        # Tạo fake price data
        import random
        base_price = 2000.0
        
        data = []
        for i, dt in enumerate(dates):
            # Random walk price
            if i == 0:
                open_price = base_price
            else:
                open_price = data[i-1]['Close']
            
            high = open_price + random.uniform(0, 5)
            low = open_price - random.uniform(0, 5)
            close = random.uniform(low, high)
            volume = random.randint(100, 1000)
            
            data.append({
                'Date': dt.strftime('%Y.%m.%d'),
                'Time': dt.strftime('%H:%M'),
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume
            })
        
        # Save sample data
        df = pd.DataFrame(data)
        sample_file = f"{self.output_dir}/XAUUSD_M1_sample.csv"
        df.to_csv(sample_file, index=False)
        
        print(f"   ✅ Sample data created: {len(df)} records")
        print(f"   📁 File: {sample_file}")
        print(f"   📅 Period: {start_date.date()} → {end_date.date()}")
        
        # Show sample
        print(f"\n📋 SAMPLE DATA PREVIEW:")
        print(df.head().to_string())
        
        return {
            'file': sample_file,
            'records': len(df),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
    
    def analyze_data_quality(self, csv_file):
        """Phân tích chất lượng dữ liệu"""
        if not os.path.exists(csv_file):
            return {'error': 'File not found'}
        
        print(f"\n🔍 ANALYZING DATA QUALITY: {csv_file}")
        print("=" * 50)
        
        df = pd.read_csv(csv_file)
        
        # Basic stats
        total_records = len(df)
        
        # Check for gaps
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.sort_values('DateTime')
        
        # Calculate time gaps
        time_diffs = df['DateTime'].diff()
        expected_diff = pd.Timedelta(minutes=1)  # M1 data
        
        gaps = time_diffs[time_diffs > expected_diff * 2]  # Gaps > 2 minutes
        
        # Price analysis
        price_analysis = {
            'open_range': (df['Open'].min(), df['Open'].max()),
            'high_range': (df['High'].min(), df['High'].max()),
            'low_range': (df['Low'].min(), df['Low'].max()),
            'close_range': (df['Close'].min(), df['Close'].max()),
        }
        
        # Check for anomalies
        price_changes = ((df['Close'] - df['Open']) / df['Open'] * 100).abs()
        large_moves = price_changes[price_changes > 5]  # >5% moves
        
        analysis = {
            'total_records': total_records,
            'date_range': {
                'start': df['DateTime'].min().strftime('%Y-%m-%d %H:%M'),
                'end': df['DateTime'].max().strftime('%Y-%m-%d %H:%M'),
                'duration_days': (df['DateTime'].max() - df['DateTime'].min()).days
            },
            'gaps': {
                'total_gaps': len(gaps),
                'largest_gap_minutes': time_diffs.max().total_seconds() / 60 if len(time_diffs) > 0 else 0
            },
            'price_analysis': price_analysis,
            'anomalies': {
                'large_moves_count': len(large_moves),
                'max_move_percent': price_changes.max() if len(price_changes) > 0 else 0
            }
        }
        
        # Print analysis
        print(f"   📊 Total Records: {analysis['total_records']:,}")
        print(f"   📅 Date Range: {analysis['date_range']['start']} → {analysis['date_range']['end']}")
        print(f"   ⏱️  Duration: {analysis['date_range']['duration_days']} days")
        print(f"   🕳️  Data Gaps: {analysis['gaps']['total_gaps']}")
        print(f"   📈 Price Range: ${price_analysis['low_range'][0]:.2f} - ${price_analysis['high_range'][1]:.2f}")
        print(f"   ⚠️  Large Moves: {analysis['anomalies']['large_moves_count']} (>{5}%)")
        
        return analysis

def main():
    downloader = FreeHistoricalDataDownloader()
    
    print("🚀 FREE HISTORICAL DATA DOWNLOAD TEST")
    print("=" * 50)
    
    # 1. Tạo sample data để test
    sample_result = downloader.create_sample_data()
    
    # 2. Analyze sample data quality
    if sample_result.get('file'):
        quality_analysis = downloader.analyze_data_quality(sample_result['file'])
    
    # 3. Simulate real downloads
    print(f"\n" + "="*60)
    histdata_result = downloader.download_histdata_xauusd()
    
    print(f"\n" + "="*60)
    forexsb_result = downloader.download_forexsb_data()
    
    # 4. Summary
    summary = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'sample_data': sample_result,
        'quality_analysis': quality_analysis if 'quality_analysis' in locals() else None,
        'histdata_simulation': histdata_result,
        'forexsb_simulation': forexsb_result
    }
    
    # Save summary
    summary_file = f"{downloader.output_dir}/download_test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ SUMMARY SAVED: {summary_file}")
    
    return summary

if __name__ == "__main__":
    main() 