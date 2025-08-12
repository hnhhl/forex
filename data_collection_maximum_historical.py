#!/usr/bin/env python3
"""
🔥 MAXIMUM HISTORICAL DATA COLLECTION - 11+ YEARS
Thu thập đủ 11+ năm data cho M1, M5, M15, M30
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class MaximumHistoricalCollector:
    def __init__(self):
        self.symbol = "XAUUSDc"
        self.output_dir = "data/maximum_historical_11years"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def initialize_mt5(self):
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False
        print("✅ MT5 initialized")
        return True
        
    def collect_maximum_data(self):
        """Thu thập maximum data cho M1, M5, M15, M30"""
        print("🔥 COLLECTING MAXIMUM 11+ YEARS DATA")
        print("=" * 50)
        
        # Target timeframes cần thu thập đủ data
        timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5, 
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30
        }
        
        # Start từ 2014 như H1 data
        start_date = datetime(2014, 1, 1)
        end_date = datetime.now()
        
        results = {}
        
        for tf_name, tf_value in timeframes.items():
            print(f"\n📊 Collecting {tf_name} data...")
            
            # Request maximum possible data
            max_count = 500000  # Request 500k records
            
            rates = mt5.copy_rates_range(
                self.symbol, 
                tf_value,
                start_date,
                end_date
            )
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Save data
                csv_file = f"{self.output_dir}/{self.symbol}_{tf_name}_maximum.csv"
                df.to_csv(csv_file, index=False)
                
                results[tf_name] = {
                    'records': len(df),
                    'start_date': df['time'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': df['time'].max().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_days': (df['time'].max() - df['time'].min()).days,
                    'csv_file': csv_file
                }
                
                print(f"   ✅ {len(df):,} records")
                print(f"   📅 {results[tf_name]['start_date']} → {results[tf_name]['end_date']}")
                print(f"   🎯 {results[tf_name]['duration_days']} days")
            else:
                print(f"   ❌ No data for {tf_name}")
        
        # Save summary
        summary = {
            'collection_time': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'symbol': self.symbol,
            'target_timeframes': list(timeframes.keys()),
            'results': results
        }
        
        summary_file = f"{self.output_dir}/collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n✅ Summary saved: {summary_file}")
        return results

def main():
    collector = MaximumHistoricalCollector()
    
    if collector.initialize_mt5():
        collector.collect_maximum_data()
    
    mt5.shutdown()

if __name__ == "__main__":
    main() 