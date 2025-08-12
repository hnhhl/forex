#!/usr/bin/env python3
"""
Debug AI2.0 Signal Generation
"""

import pandas as pd
import numpy as np

def main():
    print("🔍 DEBUGGING AI2.0 SIGNAL GENERATION")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('data/working_free_data/XAUUSD_M15_realistic.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.rename(columns={'Close': 'close'})
    
    print(f"📊 Data Info:")
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # AI2.0 parameters
    step_size = 30
    lookback = 20
    future_lookahead = 15
    
    print(f"\n🤖 AI2.0 Parameters:")
    print(f"Step size: {step_size} (every {step_size} records)")
    print(f"Lookback: {lookback} candles")
    print(f"Future lookahead: {future_lookahead} candles")
    
    total_possible_signals = (len(df) - lookback - future_lookahead) // step_size
    print(f"Expected signals: {total_possible_signals:,}")
    
    # Analyze price changes
    price_changes = df['close'].pct_change() * 100
    print(f"\n💰 Price Change Analysis:")
    print(f"Average price change: {price_changes.mean():.4f}%")
    print(f"Std price change: {price_changes.std():.4f}%")
    print(f"Changes > 0.1%: {(price_changes > 0.1).sum():,} ({(price_changes > 0.1).mean()*100:.1f}%)")
    print(f"Changes < -0.1%: {(price_changes < -0.1).sum():,} ({(price_changes < -0.1).mean()*100:.1f}%)")
    
    # Test signal generation with different parameters
    print(f"\n🧪 Testing Different Parameters:")
    
    test_params = [
        {'step': 10, 'lookback': 10, 'lookahead': 5},
        {'step': 15, 'lookback': 15, 'lookahead': 10},
        {'step': 30, 'lookback': 20, 'lookahead': 15},
        {'step': 60, 'lookback': 30, 'lookahead': 30}
    ]
    
    for params in test_params:
        step = params['step']
        lb = params['lookback']
        la = params['lookahead']
        
        signals = []
        for i in range(lb, len(df) - la, step):
            current_price = df.iloc[i]['close']
            future_price = df.iloc[min(i + la, len(df) - 1)]['close']
            price_change_pct = (future_price - current_price) / current_price * 100
            
            if price_change_pct > 0.05:  # Lower threshold
                signal = 1  # BUY
            elif price_change_pct < -0.05:
                signal = 0  # SELL
            else:
                signal = 2  # HOLD
            
            signals.append(signal)
        
        if signals:
            unique, counts = np.unique(signals, return_counts=True)
            signal_dist = dict(zip(unique, counts))
            
            print(f"  Step={step}, LB={lb}, LA={la}: {len(signals):,} signals")
            for signal, count in signal_dist.items():
                signal_name = ['SELL', 'BUY', 'HOLD'][signal]
                print(f"    {signal_name}: {count:,} ({count/len(signals)*100:.1f}%)")
    
    # Analyze best performing setup
    print(f"\n🎯 Recommended Setup:")
    print("- Reduce step_size to 15 for more signals")
    print("- Lower thresholds to 0.05% for more BUY/SELL signals")
    print("- Use shorter lookahead (5-10 candles)")
    print("- Add more sophisticated voting logic")

if __name__ == "__main__":
    main() 