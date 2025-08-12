#!/usr/bin/env python3
"""
ADVANCED STABILITY BACKTEST
Kiểm tra tính ổn định và điểm yếu của hệ thống AI3.0 với nhiều kịch bản khác nhau
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

class AdvancedStabilityBacktest:
    def __init__(self):
        self.results = {
            'start_time': datetime.now(),
            'test_scenarios': [],
            'overall_analysis': {},
            'stability_metrics': {},
            'weakness_analysis': {}
        }
        
    def run_comprehensive_tests(self):
        """Chạy các test scenario khác nhau"""
        print("🔬 ADVANCED STABILITY BACKTEST - KIỂM TRA TÍNH ỔN ĐỊNH")
        print("=" * 70)
        
        # Test scenarios
        scenarios = [
            {
                'name': 'Trending Market (2024)',
                'data_filter': lambda df: df[df['time'] >= '2024-01-01'],
                'description': 'Test trong thị trường trending mạnh'
            },
            {
                'name': 'Volatile Period (2024 Q1)',
                'data_filter': lambda df: df[(df['time'] >= '2024-01-01') & (df['time'] < '2024-04-01')],
                'description': 'Test trong thời kỳ biến động cao'
            },
            {
                'name': 'Consolidation Period',
                'data_filter': lambda df: df[(df['time'] >= '2024-06-01') & (df['time'] < '2024-09-01')],
                'description': 'Test trong thời kỳ sideway'
            },
            {
                'name': 'Recent Data (2025)',
                'data_filter': lambda df: df[df['time'] >= '2025-01-01'],
                'description': 'Test với dữ liệu gần đây nhất'
            }
        ]
        
        for scenario in scenarios:
            print(f"\n🎯 Testing Scenario: {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            
            result = self.test_scenario(scenario)
            if result:
                self.results['test_scenarios'].append(result)
                print(f"   ✅ Completed: {result['summary']}")
            else:
                print(f"   ❌ Failed to complete scenario")
    
    def test_scenario(self, scenario):
        """Test một scenario cụ thể"""
        try:
            # Load data
            df = pd.read_csv('data/maximum_mt5_v2/XAUUSDc_H1_20250618_115847.csv')
            df['time'] = pd.to_datetime(df['time'])
            
            # Apply scenario filter
            filtered_df = scenario['data_filter'](df)
            
            if len(filtered_df) < 100:
                return None
            
            # Initialize system
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            # Test parameters
            initial_balance = 10000.0
            current_balance = initial_balance
            trades = []
            signals = []
            
            # Sample test points
            test_indices = np.linspace(100, len(filtered_df)-100, min(15, len(filtered_df)//200)).astype(int)
            
            for idx in test_indices:
                try:
                    # Get data window
                    start_idx = max(0, idx - 100)
                    end_idx = min(len(filtered_df), idx + 50)
                    data_window = filtered_df.iloc[start_idx:end_idx].copy()
                    
                    if len(data_window) < 60:
                        continue
                    
                    # Generate signal
                    system._historical_data = data_window
                    signal = system.generate_signal()
                    
                    if signal:
                        signal['test_time'] = data_window['time'].iloc[-1]
                        signal['test_price'] = float(data_window['close'].iloc[-1])
                        signals.append(signal)
                        
                except Exception as e:
                    continue
            
            # Simulate trading
            if signals:
                current_position = None
                
                for i, signal in enumerate(signals):
                    action = signal['action']
                    price = signal['test_price']
                    confidence = signal['confidence']
                    
                    # Simple trading logic
                    if action == 'BUY' and current_position != 'LONG' and confidence > 0.4:
                        if current_position == 'SHORT':
                            # Close short, calculate P&L
                            last_trade = [t for t in trades if t['type'] == 'OPEN' and t['direction'] == 'SHORT'][-1]
                            pnl = (last_trade['price'] - price) * 100 * 0.01  # Simplified
                            current_balance += pnl
                            trades.append({
                                'type': 'CLOSE',
                                'direction': 'SHORT',
                                'price': price,
                                'pnl': pnl,
                                'time': signal['test_time']
                            })
                        
                        # Open long
                        trades.append({
                            'type': 'OPEN',
                            'direction': 'LONG',
                            'price': price,
                            'confidence': confidence,
                            'time': signal['test_time']
                        })
                        current_position = 'LONG'
                    
                    elif action == 'SELL' and current_position != 'SHORT' and confidence > 0.4:
                        if current_position == 'LONG':
                            # Close long, calculate P&L
                            last_trade = [t for t in trades if t['type'] == 'OPEN' and t['direction'] == 'LONG'][-1]
                            pnl = (price - last_trade['price']) * 100 * 0.01  # Simplified
                            current_balance += pnl
                            trades.append({
                                'type': 'CLOSE',
                                'direction': 'LONG',
                                'price': price,
                                'pnl': pnl,
                                'time': signal['test_time']
                            })
                        
                        # Open short
                        trades.append({
                            'type': 'OPEN',
                            'direction': 'SHORT',
                            'price': price,
                            'confidence': confidence,
                            'time': signal['test_time']
                        })
                        current_position = 'SHORT'
                
                # Close final position
                if current_position and signals:
                    final_signal = signals[-1]
                    final_price = final_signal['test_price']
                    
                    last_trade = [t for t in trades if t['type'] == 'OPEN'][-1]
                    if current_position == 'LONG':
                        pnl = (final_price - last_trade['price']) * 100 * 0.01
                    else:
                        pnl = (last_trade['price'] - final_price) * 100 * 0.01
                    
                    current_balance += pnl
                    trades.append({
                        'type': 'CLOSE',
                        'direction': current_position,
                        'price': final_price,
                        'pnl': pnl,
                        'time': final_signal['test_time']
                    })
            
            # Calculate metrics
            close_trades = [t for t in trades if t['type'] == 'CLOSE']
            total_return = (current_balance - initial_balance) / initial_balance * 100
            
            actions = [s['action'] for s in signals]
            action_dist = {action: actions.count(action) for action in set(actions)}
            
            win_trades = [t for t in close_trades if t['pnl'] > 0]
            win_rate = len(win_trades) / len(close_trades) * 100 if close_trades else 0
            
            avg_confidence = np.mean([s['confidence'] for s in signals]) if signals else 0
            
            return {
                'scenario_name': scenario['name'],
                'data_period': f"{filtered_df['time'].min()} to {filtered_df['time'].max()}",
                'data_points': len(filtered_df),
                'signals_generated': len(signals),
                'signal_distribution': action_dist,
                'average_confidence': avg_confidence,
                'total_trades': len(close_trades),
                'win_rate': win_rate,
                'total_return': total_return,
                'final_balance': current_balance,
                'summary': f"{len(signals)} signals, {total_return:.1f}% return, {win_rate:.1f}% win rate"
            }
            
        except Exception as e:
            print(f"   ❌ Error in scenario: {e}")
            return None
    
    def analyze_stability(self):
        """Phân tích tính ổn định của hệ thống"""
        print(f"\n📊 STABILITY ANALYSIS")
        print("-" * 50)
        
        if not self.results['test_scenarios']:
            print("❌ No test scenarios completed")
            return
        
        scenarios = self.results['test_scenarios']
        
        # Analyze returns consistency
        returns = [s['total_return'] for s in scenarios]
        win_rates = [s['win_rate'] for s in scenarios]
        confidences = [s['average_confidence'] for s in scenarios]
        
        return_std = np.std(returns)
        return_mean = np.mean(returns)
        
        win_rate_std = np.std(win_rates)
        win_rate_mean = np.mean(win_rates)
        
        confidence_std = np.std(confidences)
        confidence_mean = np.mean(confidences)
        
        print(f"📈 RETURN CONSISTENCY:")
        print(f"   • Average Return: {return_mean:.2f}%")
        print(f"   • Return Std Dev: {return_std:.2f}%")
        print(f"   • Return Consistency: {'GOOD' if return_std < 10 else 'POOR'}")
        
        print(f"\n🎯 WIN RATE CONSISTENCY:")
        print(f"   • Average Win Rate: {win_rate_mean:.1f}%")
        print(f"   • Win Rate Std Dev: {win_rate_std:.1f}%")
        print(f"   • Win Rate Consistency: {'GOOD' if win_rate_std < 20 else 'POOR'}")
        
        print(f"\n🔍 CONFIDENCE CONSISTENCY:")
        print(f"   • Average Confidence: {confidence_mean:.1%}")
        print(f"   • Confidence Std Dev: {confidence_std:.1%}")
        print(f"   • Confidence Consistency: {'GOOD' if confidence_std < 0.1 else 'POOR'}")
        
        # Overall stability score
        stability_score = 0
        if return_std < 10: stability_score += 25
        if win_rate_std < 20: stability_score += 25
        if confidence_std < 0.1: stability_score += 25
        if return_mean > 0: stability_score += 25
        
        print(f"\n🏆 OVERALL STABILITY SCORE: {stability_score}/100")
        
        if stability_score >= 75:
            print("   ✅ EXCELLENT STABILITY")
        elif stability_score >= 50:
            print("   ⚠️ MODERATE STABILITY")
        else:
            print("   ❌ POOR STABILITY")
        
        self.results['stability_metrics'] = {
            'return_mean': return_mean,
            'return_std': return_std,
            'win_rate_mean': win_rate_mean,
            'win_rate_std': win_rate_std,
            'confidence_mean': confidence_mean,
            'confidence_std': confidence_std,
            'stability_score': stability_score
        }
    
    def identify_weaknesses(self):
        """Xác định điểm yếu của hệ thống"""
        print(f"\n🔍 WEAKNESS IDENTIFICATION")
        print("-" * 50)
        
        weaknesses = []
        
        if not self.results['test_scenarios']:
            return
        
        scenarios = self.results['test_scenarios']
        
        # Check for consistent poor performance
        negative_returns = [s for s in scenarios if s['total_return'] < 0]
        if len(negative_returns) > len(scenarios) / 2:
            weaknesses.append({
                'category': 'PERFORMANCE',
                'severity': 'HIGH',
                'description': f"{len(negative_returns)}/{len(scenarios)} scenarios had negative returns"
            })
        
        # Check for low win rates
        low_win_rates = [s for s in scenarios if s['win_rate'] < 50]
        if len(low_win_rates) > len(scenarios) / 2:
            weaknesses.append({
                'category': 'ACCURACY',
                'severity': 'HIGH',
                'description': f"{len(low_win_rates)}/{len(scenarios)} scenarios had win rate < 50%"
            })
        
        # Check for low confidence
        low_confidence = [s for s in scenarios if s['average_confidence'] < 0.5]
        if len(low_confidence) > len(scenarios) / 2:
            weaknesses.append({
                'category': 'CONFIDENCE',
                'severity': 'MEDIUM',
                'description': f"{len(low_confidence)}/{len(scenarios)} scenarios had low confidence"
            })
        
        # Check signal distribution issues
        for scenario in scenarios:
            dist = scenario['signal_distribution']
            total_signals = sum(dist.values())
            
            if total_signals > 0:
                hold_ratio = dist.get('HOLD', 0) / total_signals
                if hold_ratio > 0.8:
                    weaknesses.append({
                        'category': 'ACTIVITY',
                        'severity': 'MEDIUM',
                        'description': f"Scenario '{scenario['scenario_name']}' has {hold_ratio:.1%} HOLD signals (too conservative)"
                    })
                
                # Check for signal imbalance
                if len(dist) == 1 and 'HOLD' not in dist:
                    weaknesses.append({
                        'category': 'BIAS',
                        'severity': 'MEDIUM',
                        'description': f"Scenario '{scenario['scenario_name']}' shows directional bias: {dist}"
                    })
        
        # Display weaknesses
        if weaknesses:
            print("❌ IDENTIFIED WEAKNESSES:")
            for i, weakness in enumerate(weaknesses, 1):
                severity_icon = "🔴" if weakness['severity'] == 'HIGH' else "🟡"
                print(f"   {severity_icon} {i}. {weakness['category']}: {weakness['description']}")
        else:
            print("✅ No major weaknesses identified")
        
        self.results['weakness_analysis'] = weaknesses
    
    def generate_improvement_plan(self):
        """Tạo kế hoạch cải thiện"""
        print(f"\n💡 IMPROVEMENT PLAN")
        print("-" * 50)
        
        improvements = []
        
        # Based on stability metrics
        if 'stability_metrics' in self.results:
            metrics = self.results['stability_metrics']
            
            if metrics['return_std'] > 10:
                improvements.append({
                    'priority': 'HIGH',
                    'area': 'Return Consistency',
                    'action': 'Implement adaptive position sizing and risk management',
                    'target': 'Reduce return volatility to < 10%'
                })
            
            if metrics['win_rate_mean'] < 60:
                improvements.append({
                    'priority': 'HIGH',
                    'area': 'Win Rate',
                    'action': 'Improve signal quality and entry/exit timing',
                    'target': 'Achieve consistent 60%+ win rate'
                })
            
            if metrics['confidence_mean'] < 0.6:
                improvements.append({
                    'priority': 'MEDIUM',
                    'area': 'Signal Confidence',
                    'action': 'Recalibrate confidence scoring and add signal validation',
                    'target': 'Achieve average confidence > 60%'
                })
        
        # Based on weaknesses
        if 'weakness_analysis' in self.results:
            weakness_categories = set(w['category'] for w in self.results['weakness_analysis'])
            
            if 'PERFORMANCE' in weakness_categories:
                improvements.append({
                    'priority': 'CRITICAL',
                    'area': 'Core Strategy',
                    'action': 'Completely revise trading logic and signal generation',
                    'target': 'Achieve positive returns in all market conditions'
                })
            
            if 'BIAS' in weakness_categories:
                improvements.append({
                    'priority': 'HIGH',
                    'area': 'Signal Balance',
                    'action': 'Rebalance ensemble weights and add contrarian signals',
                    'target': 'Achieve balanced BUY/SELL signal distribution'
                })
        
        # General improvements
        improvements.extend([
            {
                'priority': 'HIGH',
                'area': 'Model Retraining',
                'action': 'Retrain all models with latest data and improved features',
                'target': 'Use 2023-2025 data for training'
            },
            {
                'priority': 'MEDIUM',
                'area': 'Risk Management',
                'action': 'Implement dynamic stop-loss and take-profit levels',
                'target': 'Limit max drawdown to < 15%'
            },
            {
                'priority': 'MEDIUM',
                'area': 'Market Regime Detection',
                'action': 'Add market regime classification for adaptive strategies',
                'target': 'Detect trending vs ranging markets'
            }
        ])
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        improvements.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        # Display improvements
        for i, improvement in enumerate(improvements, 1):
            priority_icon = "🚨" if improvement['priority'] == 'CRITICAL' else "🔴" if improvement['priority'] == 'HIGH' else "🟡"
            print(f"   {priority_icon} {i}. {improvement['area']}")
            print(f"      Action: {improvement['action']}")
            print(f"      Target: {improvement['target']}")
            print()
        
        self.results['improvement_plan'] = improvements
    
    def generate_final_report(self):
        """Tạo báo cáo cuối cùng"""
        print(f"\n" + "="*80)
        print("📋 BÁO CÁO STABILITY BACKTEST - HỆ THỐNG AI3.0")
        print("="*80)
        
        end_time = datetime.now()
        test_duration = end_time - self.results['start_time']
        
        print(f"⏰ Thời gian test: {test_duration}")
        print(f"🎯 Số scenarios tested: {len(self.results['test_scenarios'])}")
        
        # Scenario summary
        if self.results['test_scenarios']:
            print(f"\n📊 SCENARIO RESULTS:")
            for scenario in self.results['test_scenarios']:
                print(f"   • {scenario['scenario_name']}: {scenario['summary']}")
        
        # Stability assessment
        if 'stability_metrics' in self.results:
            score = self.results['stability_metrics']['stability_score']
            print(f"\n🏆 STABILITY SCORE: {score}/100")
            
            if score >= 75:
                print("   ✅ HỆ THỐNG ỔN ĐỊNH TỐT")
                print("   → Có thể triển khai thực tế với giám sát")
            elif score >= 50:
                print("   ⚠️ HỆ THỐNG ỔN ĐỊNH TRUNG BÌNH")
                print("   → Cần cải thiện trước khi triển khai")
            else:
                print("   ❌ HỆ THỐNG KHÔNG ỔN ĐỊNH")
                print("   → Cần thiết kế lại hoàn toàn")
        
        # Major weaknesses
        if 'weakness_analysis' in self.results:
            high_severity = [w for w in self.results['weakness_analysis'] if w['severity'] == 'HIGH']
            if high_severity:
                print(f"\n🚨 ĐIỂM YẾU NGHIÊM TRỌNG: {len(high_severity)} vấn đề")
                for weakness in high_severity[:3]:  # Top 3
                    print(f"   • {weakness['category']}: {weakness['description']}")
        
        # Save results
        self.results['end_time'] = end_time
        self.results['test_duration_seconds'] = test_duration.total_seconds()
        
        filename = f"stability_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📁 Báo cáo chi tiết đã lưu: {filename}")
        print(f"🎉 STABILITY BACKTEST HOÀN THÀNH!")

def main():
    """Chạy stability backtest chính"""
    backtest = AdvancedStabilityBacktest()
    
    # Step 1: Run comprehensive tests
    backtest.run_comprehensive_tests()
    
    # Step 2: Analyze stability
    backtest.analyze_stability()
    
    # Step 3: Identify weaknesses
    backtest.identify_weaknesses()
    
    # Step 4: Generate improvement plan
    backtest.generate_improvement_plan()
    
    # Step 5: Generate final report
    backtest.generate_final_report()

if __name__ == "__main__":
    main() 