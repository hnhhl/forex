#!/usr/bin/env python3
"""
📊 STEP 2: DATA ALIGNMENT SYSTEM
Hệ thống alignment data cho multi-timeframe analysis
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class MultiTimeframeDataAligner:
    """System để align data từ tất cả timeframes"""
    
    def __init__(self):
        self.symbol = "XAUUSDc"
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize():
            print("❌ Cannot connect to MT5")
            return False
        print("✅ Connected to MT5")
        return True
    
    def get_timeframe_data(self, timeframe_mt5, bars=2000):
        """Get data for specific timeframe"""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe_mt5, 0, bars)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def calculate_core_features(self, df, tf_name):
        """Calculate core features for each timeframe"""
        
        features = pd.DataFrame(index=df.index)
        
        # Price features (5)
        features[f'{tf_name}_close'] = df['close']
        features[f'{tf_name}_sma_20'] = df['close'].rolling(20).mean()
        features[f'{tf_name}_ema_12'] = df['close'].ewm(span=12).mean()
        features[f'{tf_name}_price_change'] = df['close'].pct_change()
        features[f'{tf_name}_high_low_range'] = (df['high'] - df['low']) / df['close']
        
        # Momentum features (3)
        features[f'{tf_name}_rsi'] = self.calculate_rsi(df['close'], 14)
        features[f'{tf_name}_macd'] = features[f'{tf_name}_ema_12'] - df['close'].ewm(span=26).mean()
        features[f'{tf_name}_momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # Volatility features (2)
        features[f'{tf_name}_atr'] = self.calculate_atr(df, 14)
        features[f'{tf_name}_volatility'] = df['close'].rolling(20).std() / df['close']
        
        # Fill NaN
        features = features.ffill().fillna(0)
        
        return features
    
    def calculate_rsi(self, prices, period):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, df, period):
        """Calculate Average True Range"""
        tr = np.maximum(df['high'] - df['low'], 
                       np.maximum(abs(df['high'] - df['close'].shift()), 
                                abs(df['low'] - df['close'].shift())))
        return tr.rolling(period).mean()
    
    def collect_all_timeframe_data(self):
        """Collect data from all timeframes"""
        
        print("📊 Collecting multi-timeframe data...")
        
        all_data = {}
        
        for tf_name, tf_mt5 in self.timeframes.items():
            print(f"  📈 Collecting {tf_name} data...")
            
            df = self.get_timeframe_data(tf_mt5, 2000)
            
            if df is not None and len(df) >= 100:
                # Calculate features
                features = self.calculate_core_features(df, tf_name)
                
                all_data[tf_name] = {
                    'ohlc': df[['time', 'open', 'high', 'low', 'close', 'tick_volume']],
                    'features': features,
                    'time': df['time']
                }
                
                print(f"    ✅ {tf_name}: {len(df)} bars, {features.shape[1]} features")
            else:
                print(f"    ❌ {tf_name}: Insufficient data")
        
        return all_data
    
    def align_to_common_timeline(self, all_data):
        """Align all timeframes to common timeline"""
        
        print("🔗 Aligning timeframes to common timeline...")
        
        # Find common time range
        start_times = []
        end_times = []
        
        for tf_name, data in all_data.items():
            start_times.append(data['time'].min())
            end_times.append(data['time'].max())
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        print(f"  📅 Common time range: {common_start} to {common_end}")
        
        # Use M15 as base frequency
        base_data = all_data['M15']
        base_time = base_data['time']
        base_ohlc = base_data['ohlc']
        
        # Filter to common time range
        mask = (base_time >= common_start) & (base_time <= common_end)
        base_time_filtered = base_time[mask]
        base_ohlc_filtered = base_ohlc[mask]
        
        if len(base_time_filtered) < 500:
            print(f"❌ Insufficient common data: {len(base_time_filtered)}")
            return None
        
        print(f"  📊 Base timeline (M15): {len(base_time_filtered)} samples")
        
        # Align all timeframes to base timeline
        aligned_data = {}
        
        for tf_name, data in all_data.items():
            print(f"    🔗 Aligning {tf_name}...")
            
            # Create DataFrame with time index
            tf_df = data['features'].copy()
            tf_df.index = data['time']
            
            # Resample to M15 frequency
            aligned_features = self.resample_to_base(tf_df, base_time_filtered)
            
            aligned_data[tf_name] = {
                'features': aligned_features,
                'samples': len(aligned_features)
            }
            
            print(f"      ✅ {tf_name}: {aligned_features.shape[1]} features, {len(aligned_features)} samples")
        
        return aligned_data, base_time_filtered, base_ohlc_filtered
    
    def resample_to_base(self, tf_df, base_time):
        """Resample timeframe data to base frequency"""
        
        # Create base DataFrame
        base_df = pd.DataFrame(index=base_time)
        
        # Join and forward fill
        aligned = base_df.join(tf_df, how='left').ffill().fillna(0)
        
        return aligned
    
    def create_unified_dataset(self):
        """Create unified dataset with all timeframes"""
        
        print("🔗 CREATING UNIFIED MULTI-TIMEFRAME DATASET")
        print("=" * 60)
        
        if not self.connect_mt5():
            return None
        
        try:
            # Collect data
            all_data = self.collect_all_timeframe_data()
            
            if len(all_data) < 4:
                print("❌ Need at least 4 timeframes")
                return None
            
            # Align data
            aligned_data, base_time, base_ohlc = self.align_to_common_timeline(all_data)
            
            if aligned_data is None:
                return None
            
            # Combine features from all timeframes
            print("🔄 Combining features from all timeframes...")
            
            combined_features = []
            feature_names = []
            
            for tf_name in ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']:
                if tf_name in aligned_data:
                    tf_features = aligned_data[tf_name]['features']
                    combined_features.append(tf_features.values)
                    feature_names.extend(tf_features.columns.tolist())
                    print(f"  ✅ {tf_name}: {tf_features.shape[1]} features added")
                else:
                    # Add zeros for missing timeframes
                    zero_features = np.zeros((len(base_time), 10))  # 10 features per TF
                    combined_features.append(zero_features)
                    feature_names.extend([f'{tf_name}_feature_{i}' for i in range(10)])
                    print(f"  ⚠️ {tf_name}: Missing, added zeros")
            
            # Create final dataset
            X = np.concatenate(combined_features, axis=1)
            
            # Create labels
            y = self.create_labels(base_ohlc)
            
            # Create metadata
            metadata = {
                'total_samples': len(X),
                'total_features': X.shape[1],
                'timeframes': list(self.timeframes.keys()),
                'features_per_timeframe': {tf: aligned_data[tf]['samples'] if tf in aligned_data else 0 
                                         for tf in self.timeframes.keys()},
                'time_range': {
                    'start': str(base_time.iloc[0]),
                    'end': str(base_time.iloc[-1])
                },
                'feature_names': feature_names,
                'created_at': datetime.now().isoformat()
            }
            
            print(f"\n✅ UNIFIED DATASET CREATED:")
            print(f"  📊 Samples: {len(X):,}")
            print(f"  📊 Features: {X.shape[1]}")
            print(f"  📊 Timeframes: {len(self.timeframes)}")
            print(f"  📊 Label distribution: {np.bincount(y)}")
            
            # Save dataset
            os.makedirs('training/xauusdc/unified_data', exist_ok=True)
            
            np.save('training/xauusdc/unified_data/X_unified.npy', X)
            np.save('training/xauusdc/unified_data/y_unified.npy', y)
            
            with open('training/xauusdc/unified_data/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n✅ Dataset saved:")
            print(f"  📁 X_unified.npy: {X.shape}")
            print(f"  📁 y_unified.npy: {y.shape}")
            print(f"  📁 metadata.json: {len(metadata)} fields")
            
            return X, y, metadata
            
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            return None
        
        finally:
            mt5.shutdown()
    
    def create_labels(self, ohlc_data):
        """Create trading labels"""
        
        prices = ohlc_data['close'].reset_index(drop=True)
        labels = []
        
        for i in range(len(prices) - 4):
            current = prices.iloc[i]
            future = prices.iloc[i + 4]  # 4 M15 bars = 1 hour
            
            if pd.notna(current) and pd.notna(future):
                pct_change = (future - current) / current
                
                if pct_change > 0.002:  # > 0.2% = BUY
                    labels.append(2)
                elif pct_change < -0.002:  # < -0.2% = SELL
                    labels.append(0)
                else:  # HOLD
                    labels.append(1)
            else:
                labels.append(1)
        
        return np.array(labels)

def main():
    """Main function"""
    aligner = MultiTimeframeDataAligner()
    
    print("📊 STEP 2: DATA ALIGNMENT SYSTEM")
    print("Creating unified multi-timeframe dataset")
    print("=" * 50)
    
    # Create unified dataset
    result = aligner.create_unified_dataset()
    
    if result is not None:
        X, y, metadata = result
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Unified multi-timeframe dataset created")
        print(f"✅ Ready for training unified model")
    else:
        print(f"\n❌ Failed to create dataset")

if __name__ == "__main__":
    main() 