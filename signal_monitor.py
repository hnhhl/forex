# -*- coding: utf-8 -*-
"""AI3.0 Signal Monitoring System"""

import sys
import os
sys.path.append('src')

import time
import json
from datetime import datetime
from collections import defaultdict

def monitor_signals():
    print("🚀 AI3.0 SIGNAL MONITORING SYSTEM")
    print("="*50)
    
    # Initialize system
    try:
        from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Monitoring variables
    signals_history = []
    signal_counts = defaultdict(int)
    confidences = []
    start_time = datetime.now()
    
    print(f"\n⏰ Starting monitoring at {start_time.strftime('%H:%M:%S')}")
    print("🔄 Generating signals every 3 seconds...")
    print("⏹️ Press Ctrl+C to stop\n")
    
    try:
        signal_num = 0
        
        while True:
            signal_num += 1
            
            # Generate signal
            signal = system.generate_signal("XAUUSDc")
            
            # Extract data
            action = signal.get('action', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            method = signal.get('ensemble_method', 'unknown')
            price = signal.get('current_price', 0)
            
            # Store data
            signal_data = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'action': action,
                'confidence': confidence,
                'price': price,
                'method': method
            }
            
            signals_history.append(signal_data)
            signal_counts[action] += 1
            confidences.append(confidence)
            
            # Print current signal
            print(f"🔴 SIGNAL #{signal_num} - {signal_data['timestamp']}")
            print(f"   🎯 Action: {action}")
            print(f"   📈 Confidence: {confidence:.1%}")
            print(f"   💰 Price: ${price:.2f}")
            print(f"   🔧 Method: {method}")
            
            # Get consensus if available
            if 'hybrid_metrics' in signal and signal['hybrid_metrics']:
                consensus = signal['hybrid_metrics'].get('hybrid_consensus', 0)
                print(f"   🤝 Consensus: {consensus:.1%}")
            
            # Show distribution every 5 signals
            if signal_num % 5 == 0:
                print(f"\n📊 SIGNAL DISTRIBUTION (Last {signal_num} signals):")
                total = len(signals_history)
                for act in ['BUY', 'SELL', 'HOLD']:
                    count = signal_counts[act]
                    pct = (count / total) * 100 if total > 0 else 0
                    print(f"   {act}: {count} ({pct:.1f}%)")
                
                # Average confidence
                avg_conf = sum(confidences) / len(confidences) if confidences else 0
                print(f"   📈 Avg Confidence: {avg_conf:.1%}")
                
                # Diversity check
                unique_actions = len(set([s['action'] for s in signals_history]))
                if unique_actions == 1:
                    print("   ⚠️ Signal Diversity: LIMITED (only 1 type)")
                elif unique_actions == 2:
                    print("   ⚡ Signal Diversity: GOOD (2 types)")
                else:
                    print("   ✅ Signal Diversity: EXCELLENT (3 types)")
                
                print("-" * 50)
            
            # Wait 3 seconds
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    
    except Exception as e:
        print(f"\n❌ Error during monitoring: {e}")
    
    finally:
        # Final summary
        end_time = datetime.now()
        runtime = end_time - start_time
        
        print("\n" + "="*50)
        print("📊 MONITORING SUMMARY")
        print("="*50)
        
        print(f"⏱️ Runtime: {str(runtime).split('.')[0]}")
        print(f"📊 Total Signals: {len(signals_history)}")
        
        if signals_history:
            print(f"\n📈 FINAL DISTRIBUTION:")
            total = len(signals_history)
            for action in ['BUY', 'SELL', 'HOLD']:
                count = signal_counts[action]
                pct = (count / total) * 100
                print(f"   {action}: {count} signals ({pct:.1f}%)")
            
            avg_confidence = sum(confidences) / len(confidences)
            print(f"\n⚡ PERFORMANCE:")
            print(f"   📈 Average Confidence: {avg_confidence:.1%}")
            
            unique_actions = len(set([s['action'] for s in signals_history]))
            print(f"   🎲 Signal Types: {unique_actions}/3")
            
            # Assessment
            if unique_actions >= 3:
                print("   ✅ Diversity: EXCELLENT")
            elif unique_actions == 2:
                print("   ⚡ Diversity: GOOD")
            else:
                print("   ⚠️ Diversity: LIMITED")
            
            if avg_confidence >= 0.4:
                print("   ✅ Confidence: HIGH")
            elif avg_confidence >= 0.25:
                print("   ⚡ Confidence: GOOD")
            else:
                print("   ⚠️ Confidence: LOW")
        
        # Save data
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signal_monitoring_{timestamp}.json"
            
            data = {
                'session_info': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'runtime_seconds': runtime.total_seconds(),
                    'total_signals': len(signals_history)
                },
                'signals': signals_history,
                'summary': {
                    'signal_counts': dict(signal_counts),
                    'average_confidence': avg_confidence if signals_history else 0,
                    'unique_signal_types': unique_actions if signals_history else 0
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"\n💾 Data saved to: {filename}")
            
        except Exception as e:
            print(f"\n❌ Failed to save data: {e}")

if __name__ == "__main__":
    monitor_signals() 