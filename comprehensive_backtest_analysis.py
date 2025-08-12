#!/usr/bin/env python3
"""
COMPREHENSIVE BACKTEST ANALYSIS
Backtest toàn diện hệ thống AI3.0 với dữ liệu quá khứ thực tế
"""

import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
sys.path.append('src/core')

class ComprehensiveBacktestAnalysis:
    def __init__(self):
        self.backtest_results = {
            'start_time': datetime.now(),
            'data_analysis': {},
            'signal_analysis': {},
            'trading_performance': {},
            'risk_metrics': {},
            'detailed_trades': []
        }
        self.initial_balance = 10000.0
        self.current_balance = self.initial_balance
        self.position_size = 0.01  # 0.01 lot
        self.trades = []
        
    def load_historical_data(self):
        """Load dữ liệu lịch sử từ files có sẵn"""
        print("📊 LOADING HISTORICAL DATA FOR BACKTEST")
        print("=" * 60)
        
        try:
            # Load dữ liệu từ các timeframes khác nhau
            data_sources = {
                'H1': 'data/maximum_mt5_v2/XAUUSDc_H1_20250618_115847.csv',
                'H4': 'data/maximum_mt5_v2/XAUUSDc_H4_20250618_115847.csv',
                'D1': 'data/maximum_mt5_v2/XAUUSDc_D1_20250618_115847.csv'
            }
            
            loaded_data = {}
            
            for timeframe, file_path in data_sources.items():
                try:
                    print(f"📈 Loading {timeframe} data...")
                    df = pd.read_csv(file_path)
                    
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                        df = df.sort_values('time').reset_index(drop=True)
                        
                        # Filter dữ liệu 2024-2025 (gần đây nhất)
                        df_recent = df[df['time'] >= '2024-01-01'].copy()
                        
                        if len(df_recent) > 100:  # Đủ dữ liệu để test
                            loaded_data[timeframe] = df_recent
                            print(f"   ✅ {timeframe}: {len(df_recent)} candles from {df_recent['time'].min()} to {df_recent['time'].max()}")
                        else:
                            print(f"   ⚠️ {timeframe}: Insufficient recent data")
                    
                except Exception as e:
                    print(f"   ❌ {timeframe}: Error loading - {e}")
            
            if loaded_data:
                # Chọn timeframe có nhiều data nhất
                best_timeframe = max(loaded_data.keys(), key=lambda x: len(loaded_data[x]))
                self.historical_data = loaded_data[best_timeframe]
                
                print(f"\n🎯 Selected {best_timeframe} data for backtest:")
                print(f"   • Total candles: {len(self.historical_data):,}")
                print(f"   • Date range: {self.historical_data['time'].min()} to {self.historical_data['time'].max()}")
                print(f"   • Price range: ${self.historical_data['close'].min():.2f} - ${self.historical_data['close'].max():.2f}")
                
                self.backtest_results['data_analysis'] = {
                    'timeframe': best_timeframe,
                    'total_candles': len(self.historical_data),
                    'date_start': self.historical_data['time'].min().isoformat(),
                    'date_end': self.historical_data['time'].max().isoformat(),
                    'price_min': float(self.historical_data['close'].min()),
                    'price_max': float(self.historical_data['close'].max()),
                    'data_quality': 'GOOD'
                }
                
                return True
            else:
                print("❌ No suitable historical data found")
                return False
                
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return False
    
    def run_signal_analysis(self):
        """Chạy phân tích signals trên dữ liệu lịch sử"""
        print(f"\n🎯 RUNNING SIGNAL ANALYSIS")
        print("-" * 40)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            # Initialize system
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            # Analyze signals on historical data
            signals = []
            signal_times = []
            
            # Test trên mỗi 50 candles để tránh quá lâu
            test_indices = range(100, len(self.historical_data), 50)
            total_tests = len(test_indices)
            
            print(f"🔍 Testing {total_tests} points in historical data...")
            
            for i, idx in enumerate(test_indices):
                if i >= 20:  # Giới hạn 20 tests để tránh quá lâu
                    break
                    
                try:
                    # Lấy data window cho test
                    end_idx = min(idx + 100, len(self.historical_data))
                    data_window = self.historical_data.iloc[max(0, idx-100):end_idx].copy()
                    
                    if len(data_window) < 60:  # Cần ít nhất 60 candles
                        continue
                    
                    start_time = time.time()
                    
                    # Override system data
                    system._historical_data = data_window
                    signal = system.generate_signal()
                    
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    if signal:
                        # Thêm thông tin thời gian thực tế
                        signal['historical_time'] = data_window['time'].iloc[-1]
                        signal['historical_price'] = float(data_window['close'].iloc[-1])
                        signal['generation_time'] = generation_time
                        
                        signals.append(signal)
                        signal_times.append(generation_time)
                        
                        print(f"   Signal {i+1}/{min(20, total_tests)}: {signal['action']} | "
                              f"Price: ${signal['historical_price']:.2f} | "
                              f"Confidence: {signal['confidence']:.1%}")
                    
                except Exception as e:
                    print(f"   ❌ Signal {i+1}: Error - {e}")
                    continue
            
            if signals:
                # Phân tích signal quality
                actions = [s['action'] for s in signals]
                confidences = [s['confidence'] for s in signals]
                prices = [s['historical_price'] for s in signals]
                
                signal_distribution = {action: actions.count(action) for action in set(actions)}
                
                print(f"\n📊 SIGNAL ANALYSIS RESULTS:")
                print(f"   • Total Signals: {len(signals)}")
                print(f"   • Signal Distribution: {signal_distribution}")
                print(f"   • Average Confidence: {np.mean(confidences):.1%}")
                print(f"   • Confidence Range: {np.min(confidences):.1%} - {np.max(confidences):.1%}")
                print(f"   • Price Range: ${np.min(prices):.2f} - ${np.max(prices):.2f}")
                print(f"   • Average Generation Time: {np.mean(signal_times):.2f}s")
                
                self.signals = signals
                self.backtest_results['signal_analysis'] = {
                    'total_signals': len(signals),
                    'signal_distribution': signal_distribution,
                    'average_confidence': float(np.mean(confidences)),
                    'confidence_std': float(np.std(confidences)),
                    'average_generation_time': float(np.mean(signal_times)),
                    'price_range': [float(np.min(prices)), float(np.max(prices))]
                }
                
                return True
            else:
                print("❌ No signals generated")
                return False
                
        except Exception as e:
            print(f"❌ Signal analysis error: {e}")
            return False
    
    def simulate_trading(self):
        """Simulate trading dựa trên signals"""
        print(f"\n💰 SIMULATING TRADING PERFORMANCE")
        print("-" * 40)
        
        if not hasattr(self, 'signals') or not self.signals:
            print("❌ No signals available for trading simulation")
            return False
        
        try:
            # Reset trading variables
            self.current_balance = self.initial_balance
            self.trades = []
            current_position = None
            
            # Simulate từng signal
            for i, signal in enumerate(self.signals):
                action = signal['action']
                price = signal['historical_price']
                confidence = signal['confidence']
                timestamp = signal['historical_time']
                
                # Trading logic
                if action == 'BUY' and current_position != 'LONG':
                    # Close short position if exists
                    if current_position == 'SHORT':
                        self._close_position('SHORT', price, timestamp, 'Signal Change')
                    
                    # Open long position
                    if confidence > 0.3:  # Minimum confidence threshold
                        self._open_position('LONG', price, timestamp, confidence)
                        current_position = 'LONG'
                
                elif action == 'SELL' and current_position != 'SHORT':
                    # Close long position if exists
                    if current_position == 'LONG':
                        self._close_position('LONG', price, timestamp, 'Signal Change')
                    
                    # Open short position
                    if confidence > 0.3:  # Minimum confidence threshold
                        self._open_position('SHORT', price, timestamp, confidence)
                        current_position = 'SHORT'
                
                elif action == 'HOLD':
                    # Close any open position
                    if current_position:
                        self._close_position(current_position, price, timestamp, 'Hold Signal')
                        current_position = None
            
            # Close final position if exists
            if current_position and self.signals:
                final_price = self.signals[-1]['historical_price']
                final_time = self.signals[-1]['historical_time']
                self._close_position(current_position, final_price, final_time, 'End of Test')
            
            # Analyze trading performance
            self._analyze_trading_performance()
            
            return True
            
        except Exception as e:
            print(f"❌ Trading simulation error: {e}")
            return False
    
    def _open_position(self, direction, price, timestamp, confidence):
        """Open trading position"""
        position_value = self.current_balance * self.position_size * confidence
        
        trade = {
            'type': 'OPEN',
            'direction': direction,
            'open_price': price,
            'open_time': timestamp,
            'position_size': self.position_size,
            'confidence': confidence,
            'position_value': position_value
        }
        
        self.trades.append(trade)
        print(f"   📈 OPEN {direction} at ${price:.2f} | Confidence: {confidence:.1%}")
    
    def _close_position(self, direction, price, timestamp, reason):
        """Close trading position"""
        # Find last open position of this direction
        open_trades = [t for t in self.trades if t['type'] == 'OPEN' and t['direction'] == direction]
        
        if not open_trades:
            return
        
        last_open = open_trades[-1]
        
        # Calculate P&L
        if direction == 'LONG':
            pnl_pips = (price - last_open['open_price']) * 100  # Convert to pips
        else:  # SHORT
            pnl_pips = (last_open['open_price'] - price) * 100  # Convert to pips
        
        # Calculate dollar P&L (simplified)
        pnl_dollars = pnl_pips * last_open['position_size'] * 10  # Rough calculation
        
        # Update balance
        self.current_balance += pnl_dollars
        
        trade = {
            'type': 'CLOSE',
            'direction': direction,
            'open_price': last_open['open_price'],
            'close_price': price,
            'open_time': last_open['open_time'],
            'close_time': timestamp,
            'pnl_pips': pnl_pips,
            'pnl_dollars': pnl_dollars,
            'confidence': last_open['confidence'],
            'reason': reason
        }
        
        self.trades.append(trade)
        
        status = "✅ PROFIT" if pnl_dollars > 0 else "❌ LOSS"
        print(f"   📉 CLOSE {direction} at ${price:.2f} | P&L: {pnl_pips:.1f} pips (${pnl_dollars:.2f}) | {status}")
    
    def _analyze_trading_performance(self):
        """Phân tích hiệu quả trading"""
        print(f"\n📊 TRADING PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        # Filter close trades
        close_trades = [t for t in self.trades if t['type'] == 'CLOSE']
        
        if not close_trades:
            print("❌ No completed trades to analyze")
            return
        
        # Calculate metrics
        total_trades = len(close_trades)
        winning_trades = [t for t in close_trades if t['pnl_dollars'] > 0]
        losing_trades = [t for t in close_trades if t['pnl_dollars'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl_dollars'] for t in close_trades)
        total_pips = sum(t['pnl_pips'] for t in close_trades)
        
        avg_win = np.mean([t['pnl_dollars'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_dollars'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t['pnl_dollars'] for t in winning_trades) / 
                           sum(t['pnl_dollars'] for t in losing_trades)) if losing_trades else float('inf')
        
        # Calculate returns
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate max drawdown (simplified)
        running_balance = self.initial_balance
        max_balance = self.initial_balance
        max_drawdown = 0
        
        for trade in close_trades:
            running_balance += trade['pnl_dollars']
            max_balance = max(max_balance, running_balance)
            drawdown = (max_balance - running_balance) / max_balance * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        print(f"📈 OVERALL PERFORMANCE:")
        print(f"   • Initial Balance: ${self.initial_balance:,.2f}")
        print(f"   • Final Balance: ${self.current_balance:,.2f}")
        print(f"   • Total Return: {total_return:.2f}%")
        print(f"   • Total P&L: ${total_pnl:.2f} ({total_pips:.1f} pips)")
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   • Total Trades: {total_trades}")
        print(f"   • Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"   • Losing Trades: {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"   • Average Win: ${avg_win:.2f}")
        print(f"   • Average Loss: ${avg_loss:.2f}")
        print(f"   • Profit Factor: {profit_factor:.2f}")
        
        print(f"\n⚠️ RISK METRICS:")
        print(f"   • Max Drawdown: {max_drawdown:.2f}%")
        
        # Store results
        self.backtest_results['trading_performance'] = {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return_percent': total_return,
            'total_pnl_dollars': total_pnl,
            'total_pnl_pips': total_pips,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_percent': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_percent': max_drawdown
        }
        
        # Store detailed trades
        self.backtest_results['detailed_trades'] = [
            {
                'direction': t['direction'],
                'open_price': t['open_price'],
                'close_price': t['close_price'],
                'open_time': t['open_time'].isoformat() if hasattr(t['open_time'], 'isoformat') else str(t['open_time']),
                'close_time': t['close_time'].isoformat() if hasattr(t['close_time'], 'isoformat') else str(t['close_time']),
                'pnl_dollars': t['pnl_dollars'],
                'pnl_pips': t['pnl_pips'],
                'confidence': t['confidence'],
                'reason': t['reason']
            } for t in close_trades
        ]
    
    def analyze_system_weaknesses(self):
        """Phân tích điểm yếu của hệ thống"""
        print(f"\n🔍 ANALYZING SYSTEM WEAKNESSES")
        print("-" * 40)
        
        weaknesses = []
        
        # Check trading performance
        if 'trading_performance' in self.backtest_results:
            perf = self.backtest_results['trading_performance']
            
            if perf['total_return_percent'] < 0:
                weaknesses.append({
                    'category': 'PERFORMANCE',
                    'issue': 'Negative Returns',
                    'severity': 'HIGH',
                    'description': f"System lost {abs(perf['total_return_percent']):.2f}% over test period"
                })
            
            if perf['win_rate_percent'] < 50:
                weaknesses.append({
                    'category': 'ACCURACY',
                    'issue': 'Low Win Rate',
                    'severity': 'HIGH',
                    'description': f"Win rate only {perf['win_rate_percent']:.1f}% (below 50%)"
                })
            
            if perf['profit_factor'] < 1.0:
                weaknesses.append({
                    'category': 'PROFITABILITY',
                    'issue': 'Poor Profit Factor',
                    'severity': 'HIGH',
                    'description': f"Profit factor {perf['profit_factor']:.2f} (below 1.0)"
                })
            
            if perf['max_drawdown_percent'] > 20:
                weaknesses.append({
                    'category': 'RISK',
                    'issue': 'High Drawdown',
                    'severity': 'MEDIUM',
                    'description': f"Max drawdown {perf['max_drawdown_percent']:.2f}% (above 20%)"
                })
        
        # Check signal quality
        if 'signal_analysis' in self.backtest_results:
            sig = self.backtest_results['signal_analysis']
            
            if sig['average_confidence'] < 0.5:
                weaknesses.append({
                    'category': 'CONFIDENCE',
                    'issue': 'Low Signal Confidence',
                    'severity': 'MEDIUM',
                    'description': f"Average confidence only {sig['average_confidence']:.1%}"
                })
            
            # Check signal distribution
            if 'signal_distribution' in sig:
                dist = sig['signal_distribution']
                total_signals = sum(dist.values())
                hold_ratio = dist.get('HOLD', 0) / total_signals if total_signals > 0 else 0
                
                if hold_ratio > 0.8:
                    weaknesses.append({
                        'category': 'ACTIVITY',
                        'issue': 'Too Many HOLD Signals',
                        'severity': 'MEDIUM',
                        'description': f"{hold_ratio:.1%} of signals are HOLD (system too conservative)"
                    })
        
        # Display weaknesses
        if weaknesses:
            print("❌ IDENTIFIED WEAKNESSES:")
            for i, weakness in enumerate(weaknesses, 1):
                severity_icon = "🔴" if weakness['severity'] == 'HIGH' else "🟡"
                print(f"   {severity_icon} {i}. {weakness['category']}: {weakness['issue']}")
                print(f"      └─ {weakness['description']}")
        else:
            print("✅ No major weaknesses identified")
        
        self.backtest_results['weaknesses'] = weaknesses
        
        return weaknesses
    
    def generate_recommendations(self):
        """Tạo khuyến nghị cải thiện"""
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = []
        
        # Based on weaknesses
        if hasattr(self, 'backtest_results') and 'weaknesses' in self.backtest_results:
            for weakness in self.backtest_results['weaknesses']:
                if weakness['category'] == 'PERFORMANCE':
                    recommendations.append({
                        'priority': 'HIGH',
                        'area': 'Strategy Logic',
                        'action': 'Revise signal generation logic and entry/exit rules'
                    })
                
                elif weakness['category'] == 'ACCURACY':
                    recommendations.append({
                        'priority': 'HIGH',
                        'area': 'Model Training',
                        'action': 'Retrain AI models with more recent data and better features'
                    })
                
                elif weakness['category'] == 'RISK':
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'area': 'Risk Management',
                        'action': 'Implement stricter stop-loss and position sizing rules'
                    })
                
                elif weakness['category'] == 'CONFIDENCE':
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'area': 'Signal Quality',
                        'action': 'Improve confidence calculation and add signal filtering'
                    })
        
        # General recommendations
        recommendations.extend([
            {
                'priority': 'HIGH',
                'area': 'Data Quality',
                'action': 'Use more recent and higher quality training data'
            },
            {
                'priority': 'MEDIUM',
                'area': 'Feature Engineering',
                'action': 'Add more sophisticated technical indicators and market regime detection'
            },
            {
                'priority': 'MEDIUM',
                'area': 'Ensemble Optimization',
                'action': 'Optimize voting weights and consensus thresholds'
            },
            {
                'priority': 'LOW',
                'area': 'Performance',
                'action': 'Optimize code for faster signal generation'
            }
        ])
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            priority_icon = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🟢"
            print(f"   {priority_icon} {i}. {rec['area']}: {rec['action']}")
        
        self.backtest_results['recommendations'] = recommendations
    
    def generate_final_report(self):
        """Tạo báo cáo cuối cùng"""
        print(f"\n" + "="*80)
        print("📋 BÁO CÁO BACKTEST TOÀN DIỆN - HỆ THỐNG AI3.0")
        print("="*80)
        
        end_time = datetime.now()
        test_duration = end_time - self.backtest_results['start_time']
        
        print(f"⏰ Thời gian test: {test_duration}")
        
        # Data summary
        if 'data_analysis' in self.backtest_results:
            data = self.backtest_results['data_analysis']
            print(f"\n📊 DATA SUMMARY:")
            print(f"   • Timeframe: {data['timeframe']}")
            print(f"   • Period: {data['date_start'][:10]} to {data['date_end'][:10]}")
            print(f"   • Total Candles: {data['total_candles']:,}")
            print(f"   • Price Range: ${data['price_min']:.2f} - ${data['price_max']:.2f}")
        
        # Performance summary
        if 'trading_performance' in self.backtest_results:
            perf = self.backtest_results['trading_performance']
            print(f"\n💰 TRADING PERFORMANCE:")
            print(f"   • Total Return: {perf['total_return_percent']:.2f}%")
            print(f"   • Total Trades: {perf['total_trades']}")
            print(f"   • Win Rate: {perf['win_rate_percent']:.1f}%")
            print(f"   • Profit Factor: {perf['profit_factor']:.2f}")
            print(f"   • Max Drawdown: {perf['max_drawdown_percent']:.2f}%")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        
        if 'trading_performance' in self.backtest_results:
            perf = self.backtest_results['trading_performance']
            
            if perf['total_return_percent'] > 10 and perf['win_rate_percent'] > 60:
                print("   🎉 SYSTEM PERFORMANCE: EXCELLENT")
            elif perf['total_return_percent'] > 0 and perf['win_rate_percent'] > 50:
                print("   👍 SYSTEM PERFORMANCE: GOOD")
            elif perf['total_return_percent'] > -10:
                print("   ⚠️ SYSTEM PERFORMANCE: POOR")
            else:
                print("   ❌ SYSTEM PERFORMANCE: VERY POOR")
        
        # Save results
        self.backtest_results['end_time'] = end_time
        self.backtest_results['test_duration_seconds'] = test_duration.total_seconds()
        
        filename = f"comprehensive_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.backtest_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📁 Báo cáo chi tiết đã lưu: {filename}")
        print(f"🎉 BACKTEST HOÀN THÀNH!")

def main():
    """Chạy backtest chính"""
    backtest = ComprehensiveBacktestAnalysis()
    
    # Step 1: Load historical data
    if not backtest.load_historical_data():
        print("❌ Failed to load historical data")
        return
    
    # Step 2: Run signal analysis
    if not backtest.run_signal_analysis():
        print("❌ Failed to analyze signals")
        return
    
    # Step 3: Simulate trading
    if not backtest.simulate_trading():
        print("❌ Failed to simulate trading")
        return
    
    # Step 4: Analyze weaknesses
    backtest.analyze_system_weaknesses()
    
    # Step 5: Generate recommendations
    backtest.generate_recommendations()
    
    # Step 6: Generate final report
    backtest.generate_final_report()

if __name__ == "__main__":
    main() 