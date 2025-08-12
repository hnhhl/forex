#!/usr/bin/env python3
"""
🆓 REAL FREE DATA DOWNLOADER
Download thực tế từ ForexSB.com cho XAU/USD
"""

import requests
import pandas as pd
import json
import os
from datetime import datetime
import time

class RealFreeDataDownloader:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.base_url = "https://data.forexsb.com"
        self.output_dir = "data/real_free_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_forexsb_xauusd(self):
        """Download XAU/USD data từ ForexSB.com"""
        print("🆓 DOWNLOADING REAL XAU/USD DATA FROM FOREXSB.COM")
        print("=" * 60)
        
        # ForexSB.com API endpoints (cần research)
        # URL thường là: https://data.forexsb.com/?symbol=XAUUSD&format=csv
        
        timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        results = {}
        
        for tf in timeframes:
            print(f"\n📊 Downloading {tf} data...")
            
            try:
                # Thử các URL patterns có thể
                possible_urls = [
                    f"{self.base_url}/?symbol={self.symbol}&timeframe={tf}&format=csv",
                    f"{self.base_url}/download/{self.symbol}_{tf}.csv",
                    f"{self.base_url}/api/data?symbol={self.symbol}&period={tf}",
                ]
                
                success = False
                for url in possible_urls:
                    print(f"   🔍 Trying: {url}")
                    
                    try:
                        response = requests.get(url, timeout=30)
                        
                        if response.status_code == 200:
                            # Check if response contains CSV data
                            content = response.text
                            
                            if 'Date' in content or 'Time' in content:
                                # Save data
                                csv_file = f"{self.output_dir}/{self.symbol}_{tf}_forexsb.csv"
                                
                                with open(csv_file, 'w') as f:
                                    f.write(content)
                                
                                # Quick analysis
                                lines = content.split('\n')
                                data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
                                
                                results[tf] = {
                                    'status': 'success',
                                    'file': csv_file,
                                    'records': len(data_lines) - 1,  # -1 for header
                                    'url': url,
                                    'size_kb': len(content) / 1024
                                }
                                
                                print(f"   ✅ Success! {len(data_lines)-1} records")
                                print(f"   📁 Saved: {csv_file}")
                                success = True
                                break
                                
                            else:
                                print(f"   ⚠️  Response doesn't look like CSV data")
                        else:
                            print(f"   ❌ HTTP {response.status_code}")
                            
                    except requests.RequestException as e:
                        print(f"   ❌ Request failed: {e}")
                        continue
                
                if not success:
                    results[tf] = {
                        'status': 'failed',
                        'error': 'No working URL found'
                    }
                    print(f"   ❌ Failed to download {tf} data")
                
                # Be nice to server
                time.sleep(2)
                
            except Exception as e:
                results[tf] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"   ❌ Error: {e}")
        
        return results
    
    def try_histdata_download(self):
        """Thử download từ HistData.com"""
        print("\n🆓 TRYING HISTDATA.COM DOWNLOAD")
        print("=" * 40)
        
        # HistData.com thường cần form submission
        # URL pattern: https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/XAUUSD/2024/1
        
        base_url = "https://www.histdata.com"
        
        # Thử download tháng gần nhất
        year = 2024
        month = 6  # June 2024
        
        url = f"{base_url}/download-free-forex-data/?/ascii/1-minute-bar-quotes/{self.symbol}/{year}/{month}"
        
        print(f"   🔍 Trying: {url}")
        
        try:
            # Headers để giả lập browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            print(f"   📊 Status: {response.status_code}")
            print(f"   📏 Content length: {len(response.content)} bytes")
            
            if response.status_code == 200:
                # Check if it's a download or webpage
                content_type = response.headers.get('content-type', '')
                
                if 'zip' in content_type or 'application' in content_type:
                    # It's a file download
                    zip_file = f"{self.output_dir}/{self.symbol}_{year}_{month:02d}_histdata.zip"
                    
                    with open(zip_file, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"   ✅ Downloaded ZIP: {zip_file}")
                    
                    # Try to extract
                    try:
                        import zipfile
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            zip_ref.extractall(self.output_dir)
                        print(f"   📂 Extracted to: {self.output_dir}")
                        
                        return {'status': 'success', 'file': zip_file}
                    except Exception as e:
                        print(f"   ⚠️  Extract failed: {e}")
                        return {'status': 'partial', 'file': zip_file}
                        
                else:
                    # It's probably a webpage - need to parse for download links
                    print(f"   ⚠️  Got webpage, not direct download")
                    print(f"   📄 Content type: {content_type}")
                    
                    # Save webpage for analysis
                    html_file = f"{self.output_dir}/histdata_response.html"
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    return {'status': 'webpage', 'file': html_file}
            else:
                return {'status': 'failed', 'code': response.status_code}
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def analyze_downloaded_data(self):
        """Phân tích dữ liệu đã download"""
        print(f"\n🔍 ANALYZING DOWNLOADED DATA")
        print("=" * 40)
        
        csv_files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
        
        analysis_results = {}
        
        for csv_file in csv_files:
            file_path = os.path.join(self.output_dir, csv_file)
            print(f"\n📊 Analyzing: {csv_file}")
            
            try:
                # Try different separators
                for sep in [',', ';', '\t']:
                    try:
                        df = pd.read_csv(file_path, sep=sep)
                        if len(df.columns) >= 5:  # OHLC + time should be at least 5 columns
                            break
                    except:
                        continue
                else:
                    print(f"   ❌ Could not parse CSV")
                    continue
                
                # Basic analysis
                total_records = len(df)
                columns = list(df.columns)
                
                # Try to identify OHLC columns
                ohlc_columns = {}
                for col in columns:
                    col_lower = col.lower()
                    if 'open' in col_lower:
                        ohlc_columns['open'] = col
                    elif 'high' in col_lower:
                        ohlc_columns['high'] = col
                    elif 'low' in col_lower:
                        ohlc_columns['low'] = col
                    elif 'close' in col_lower:
                        ohlc_columns['close'] = col
                    elif 'time' in col_lower or 'date' in col_lower:
                        ohlc_columns['time'] = col
                
                analysis = {
                    'total_records': total_records,
                    'columns': columns,
                    'ohlc_columns': ohlc_columns,
                    'sample_data': df.head(3).to_dict('records') if total_records > 0 else []
                }
                
                print(f"   📊 Records: {total_records:,}")
                print(f"   📋 Columns: {columns}")
                print(f"   🎯 OHLC mapping: {ohlc_columns}")
                
                analysis_results[csv_file] = analysis
                
            except Exception as e:
                print(f"   ❌ Analysis error: {e}")
                analysis_results[csv_file] = {'error': str(e)}
        
        return analysis_results

def main():
    downloader = RealFreeDataDownloader()
    
    print("🚀 REAL FREE DATA DOWNLOAD")
    print("=" * 40)
    
    # 1. Try ForexSB.com
    forexsb_results = downloader.download_forexsb_xauusd()
    
    # 2. Try HistData.com
    histdata_result = downloader.try_histdata_download()
    
    # 3. Analyze what we got
    analysis_results = downloader.analyze_downloaded_data()
    
    # 4. Summary
    summary = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'forexsb_results': forexsb_results,
        'histdata_result': histdata_result,
        'analysis_results': analysis_results
    }
    
    # Save summary
    summary_file = f"{downloader.output_dir}/real_download_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ SUMMARY SAVED: {summary_file}")
    
    # Print quick summary
    print(f"\n📋 QUICK SUMMARY:")
    print(f"   🎯 ForexSB attempts: {len(forexsb_results)}")
    successful_downloads = sum(1 for r in forexsb_results.values() if r.get('status') == 'success')
    print(f"   ✅ Successful downloads: {successful_downloads}")
    print(f"   📊 Files analyzed: {len(analysis_results)}")
    
    return summary

if __name__ == "__main__":
    main() 