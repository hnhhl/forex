# -*- coding: utf-8 -*-
"""Fix Data Sync - Option A: MT5 Real-time + Recent"""

import sys
import os
sys.path.append('src')

from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

def fix_data_sync_option_a():
    print("FIX DATA SYNC - OPTION A: MT5 Real-time + Recent")
    print("="*70)
    
    try:
        # 1. Thu thập MT5 data
        print("1. Thu thập MT5 Data...")
        
        if not mt5.initialize():
            print("   MT5 connection failed")
            return False
        
        # Real-time data
        rates_realtime = mt5.copy_rates_from_pos("XAUUSDc", mt5.TIMEFRAME_M1, 0, 50)
        # Recent data for training
        rates_recent = mt5.copy_rates_from_pos("XAUUSDc", mt5.TIMEFRAME_M1, 0, 15000)
        
        if rates_realtime is None or rates_recent is None:
            print("   Failed to get MT5 data")
            mt5.shutdown()
            return False
        
        df_realtime = pd.DataFrame(rates_realtime)
        df_recent = pd.DataFrame(rates_recent)
        
        print(f"   Real-time: {len(df_realtime)} records")
        print(f"   Recent: {len(df_recent):,} records")
        print(f"   Current price: {df_realtime['close'].iloc[-1]:.2f}")
        
        mt5.shutdown()
        
        # 2. Prepare training data
        print("\n2. Prepare Training Data...")
        
        df_train = df_recent.copy()
        if 'volume' not in df_train.columns:
            df_train['volume'] = df_train['tick_volume']
        
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        sequence_length = 60
        
        X, y = [], []
        for i in range(sequence_length, len(df_train) - 5):
            sequence = df_train[feature_columns].iloc[i-sequence_length:i].values
            X.append(sequence)
            
            current_price = df_train['close'].iloc[i]
            future_price = df_train['close'].iloc[i + 5]
            price_change = (future_price - current_price) / current_price
            
            if price_change > 0.0003:
                target = 0.8
            elif price_change < -0.0003:
                target = 0.2
            else:
                target = 0.5
            
            y.append(target)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"   Created {len(X):,} sequences")
        print(f"   BUY: {np.sum(y > 0.6):,}, SELL: {np.sum(y < 0.4):,}, HOLD: {np.sum((y >= 0.4) & (y <= 0.6)):,}")
        
        # 3. Normalize data
        print("\n3. Normalize Data...")
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape).astype(np.float32)
        
        print(f"   Price range: {scaler.data_min_[3]:.2f} - {scaler.data_max_[2]:.2f}")
        
        # 4. Initialize system
        print("\n4. Initialize AI3.0...")
        
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        neural_system = system.system_manager.systems.get('NeuralNetworkSystem')
        if not neural_system:
            return False
        
        neural_system.feature_scalers['fixed_5_features'] = scaler
        print("   Neural system updated")
        
        # 5. Split and train
        print("\n5. Train Models...")
        
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        trained_count = 0
        for model_name in ['lstm', 'cnn', 'transformer']:
            try:
                model = neural_system.models.get(model_name)
                if model is None:
                    continue
                
                print(f"   Training {model_name.upper()}...")
                
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    batch_size=64,
                    verbose=0
                )
                
                final_loss = history.history['loss'][-1]
                print(f"   {model_name}: loss={final_loss:.4f}")
                trained_count += 1
                
            except Exception as e:
                print(f"   {model_name}: Failed - {str(e)[:50]}")
        
        print(f"   Trained {trained_count}/3 models")
        
        # 6. Test system
        print("\n6. Test System...")
        
        signals = []
        for i in range(8):
            signal = system.generate_signal("XAUUSDc")
            signals.append(signal)
            action = signal.get('action', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            print(f"   Signal {i+1}: {action} ({confidence:.1%})")
        
        # 7. Results
        print("\n7. Results...")
        
        confidences = [s.get('confidence', 0) for s in signals]
        avg_confidence = np.mean(confidences)
        
        actions = [s.get('action', 'UNKNOWN') for s in signals]
        unique_actions = set(actions)
        
        print(f"   Average confidence: {avg_confidence:.1%}")
        print(f"   Signal types: {unique_actions}")
        
        # Assessment
        if avg_confidence > 0.4:
            status = "EXCELLENT"
            print("\n   EXCELLENT: High confidence achieved!")
        elif avg_confidence > 0.3:
            status = "GOOD"
            print("\n   GOOD: Significant improvement!")
        elif avg_confidence > 0.25:
            status = "IMPROVED"
            print("\n   IMPROVED: Some progress made!")
        else:
            status = "NEEDS_WORK"
            print("\n   NEEDS_WORK: Requires more optimization")
        
        # Save results
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'option': 'A',
            'training_records': len(df_train),
            'models_trained': trained_count,
            'average_confidence': avg_confidence,
            'signal_types': list(unique_actions),
            'status': status
        }
        
        import json
        with open('option_a_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   Results saved to: option_a_results.json")
        
        return status in ["EXCELLENT", "GOOD", "IMPROVED"]
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = fix_data_sync_option_a()
    
    print("\n" + "="*70)
    if success:
        print("OPTION A THÀNH CÔNG!")
        print("AI3.0 đã được fix với MT5 Real-time + Recent data!")
    else:
        print("OPTION A cần thêm optimization")
    print("="*70) 