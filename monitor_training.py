#!/usr/bin/env python3
"""
🔍 TRAINING PROGRESS MONITOR
Monitor training progress and results
"""

import time
import os
import json
from datetime import datetime

def monitor_training():
    print('🔍 MONITORING TRAINING PROGRESS...')
    print('-' * 50)
    
    start_time = time.time()
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Check for result files
        result_files = []
        if os.path.exists('comprehensive_trading_results'):
            result_files = [f for f in os.listdir('comprehensive_trading_results') if f.endswith('.json')]
        
        model_files = []
        if os.path.exists('trading_models_50epochs'):
            model_files = [f for f in os.listdir('trading_models_50epochs') if f.endswith('.pkl')]
        
        print(f'⏱️  Elapsed: {elapsed:.0f}s | Results: {len(result_files)} | Models: {len(model_files)}')
        
        # Check if training completed
        if result_files and model_files:
            print('✅ TRAINING COMPLETED!')
            latest_result = max(result_files, key=lambda x: os.path.getctime(os.path.join('comprehensive_trading_results', x)))
            print(f'📊 Latest result: {latest_result}')
            
            # Show quick summary
            try:
                with open(f'comprehensive_trading_results/{latest_result}', 'r') as f:
                    data = json.load(f)
                    
                print('\n📈 QUICK SUMMARY:')
                print('-' * 30)
                
                if 'best_model' in data:
                    print(f'🏆 Best Model: {data["best_model"]}')
                
                if 'trading_performance' in data:
                    for model, perf in data['trading_performance'].items():
                        if perf.get('total_trades', 0) > 0:
                            print(f'   {model}:')
                            print(f'     💰 Trades: {perf.get("total_trades", 0)}')
                            print(f'     🎯 Win Rate: {perf.get("win_rate", 0):.1f}%')
                            print(f'     📊 Return: {perf.get("total_return", 0):.1f}%')
                            print(f'     💵 Final Balance: ${perf.get("final_balance", 0):.2f}')
                
            except Exception as e:
                print(f'⚠️ Error reading results: {e}')
            
            break
        
        # Safety timeout after 15 minutes
        if elapsed > 900:
            print('⏰ Timeout reached - training may still be running...')
            break
            
        time.sleep(15)

if __name__ == "__main__":
    monitor_training() 