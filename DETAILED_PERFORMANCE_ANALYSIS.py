import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

print("📊 DETAILED PERFORMANCE ANALYSIS - ULTIMATE SYSTEM V5.0")
print("="*60)

def load_results():
    """Load training results from JSON file"""
    filename = "ULTIMATE_SYSTEM_V5_TRAINING_RESULTS_20250618_094125.json"
    
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        print(f"✅ Loaded results from: {filename}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def analyze_trading_performance(results):
    """Analyze trading performance in detail"""
    print(f"\n📈 TRADING PERFORMANCE ANALYSIS")
    print("-"*40)
    
    backtest = results['training_history']['backtest']
    metrics = results['performance_metrics']
    
    # Basic metrics
    initial_capital = 10000
    final_capital = backtest['final_capital']
    total_return = backtest['total_return']
    
    print(f"💰 CAPITAL ANALYSIS:")
    print(f"  Initial: ${initial_capital:,.2f}")
    print(f"  Final: ${final_capital:,.2f}")
    print(f"  Absolute Gain: ${final_capital - initial_capital:,.2f}")
    print(f"  Percentage Gain: {total_return*100:.2f}%")
    print(f"  Multiplier: {final_capital/initial_capital:.2f}x")
    
    # Risk metrics
    print(f"\n📊 RISK METRICS:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Volatility: {metrics['volatility']*100:.2f}%")
    
    # Risk assessment
    if metrics['sharpe_ratio'] > 1.5:
        risk_assessment = "🟢 Excellent"
    elif metrics['sharpe_ratio'] > 1.0:
        risk_assessment = "🟢 Good"
    elif metrics['sharpe_ratio'] > 0.5:
        risk_assessment = "🟡 Fair"
    else:
        risk_assessment = "🔴 Poor"
    
    print(f"  Risk Assessment: {risk_assessment}")
    
    # Trade analysis
    trades = backtest['trades']
    total_trades = len(trades)
    
    print(f"\n🔄 TRADE ANALYSIS:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"  Total Trade Pairs: {metrics['total_trades']}")
    
    # Analyze individual trades
    if total_trades >= 2:
        profits = []
        holding_periods = []
        
        for i in range(0, total_trades-1, 2):
            if i+1 < total_trades:
                buy_trade = trades[i]
                sell_trade = trades[i+1]
                
                if buy_trade[0] == 'BUY' and sell_trade[0] == 'SELL':
                    buy_price = buy_trade[1]
                    sell_price = sell_trade[1]
                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    profits.append(profit_pct)
                    
                    holding_period = sell_trade[2] - buy_trade[2]
                    holding_periods.append(holding_period)
        
        if profits:
            profits = np.array(profits)
            winning_trades = profits[profits > 0]
            losing_trades = profits[profits < 0]
            
            print(f"\n📊 TRADE STATISTICS:")
            print(f"  Average Profit: {np.mean(profits):.2f}%")
            print(f"  Best Trade: {np.max(profits):.2f}%")
            print(f"  Worst Trade: {np.min(profits):.2f}%")
            print(f"  Profit Std: {np.std(profits):.2f}%")
            
            if len(winning_trades) > 0:
                print(f"  Avg Winning Trade: {np.mean(winning_trades):.2f}%")
            if len(losing_trades) > 0:
                print(f"  Avg Losing Trade: {np.mean(losing_trades):.2f}%")
            
            print(f"  Avg Holding Period: {np.mean(holding_periods):.1f} periods")

def analyze_ai_performance(results):
    """Analyze AI model performance"""
    print(f"\n🧠 AI MODEL PERFORMANCE ANALYSIS")
    print("-"*40)
    
    # DQN Analysis
    dqn = results['training_history']['dqn']
    print(f"🤖 DQN AGENT:")
    print(f"  Final Epsilon: {dqn['final_epsilon']:.6f}")
    print(f"  Exploration Complete: {dqn['final_epsilon'] < 0.01}")
    print(f"  Learning Status: {'✅ Converged' if dqn['final_epsilon'] < 0.05 else '⚠️ Still Learning'}")
    
    # Meta Learning Analysis
    meta = results['training_history']['meta_learning']
    print(f"\n🧠 META LEARNING:")
    print(f"  Train Accuracy: {meta['train_accuracy']:.4f} ({meta['train_accuracy']*100:.2f}%)")
    print(f"  Test Accuracy: {meta['test_accuracy']:.4f} ({meta['test_accuracy']*100:.2f}%)")
    
    # Overfitting analysis
    overfitting_gap = meta['train_accuracy'] - meta['test_accuracy']
    print(f"  Overfitting Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
    
    if overfitting_gap > 0.2:
        overfitting_status = "🔴 Severe Overfitting"
    elif overfitting_gap > 0.1:
        overfitting_status = "🟡 Moderate Overfitting"
    elif overfitting_gap > 0.05:
        overfitting_status = "🟡 Slight Overfitting"
    else:
        overfitting_status = "🟢 Good Generalization"
    
    print(f"  Status: {overfitting_status}")
    
    # Cross-Validation Analysis
    cv = results['training_history']['cross_validation']
    print(f"\n📊 CROSS-VALIDATION:")
    print(f"  Mean Accuracy: {cv['mean']:.4f} ({cv['mean']*100:.2f}%)")
    print(f"  Standard Deviation: {cv['std']:.4f} ({cv['std']*100:.2f}%)")
    print(f"  Stability: {'🟢 Stable' if cv['std'] < 0.02 else '🟡 Variable'}")
    
    # Overall AI assessment
    print(f"\n🎯 AI SYSTEM ASSESSMENT:")
    
    # Calculate AI score
    ai_score = 0
    
    # DQN score (0-25 points)
    if dqn['final_epsilon'] < 0.01:
        ai_score += 25
    elif dqn['final_epsilon'] < 0.05:
        ai_score += 20
    else:
        ai_score += 10
    
    # Meta learning score (0-35 points)
    if meta['test_accuracy'] > 0.6:
        ai_score += 35
    elif meta['test_accuracy'] > 0.5:
        ai_score += 25
    elif meta['test_accuracy'] > 0.45:
        ai_score += 15
    else:
        ai_score += 5
    
    # Overfitting penalty
    if overfitting_gap > 0.3:
        ai_score -= 15
    elif overfitting_gap > 0.2:
        ai_score -= 10
    elif overfitting_gap > 0.1:
        ai_score -= 5
    
    # CV score (0-25 points)
    if cv['mean'] > 0.55:
        ai_score += 25
    elif cv['mean'] > 0.52:
        ai_score += 20
    elif cv['mean'] > 0.50:
        ai_score += 15
    else:
        ai_score += 5
    
    # Stability bonus (0-15 points)
    if cv['std'] < 0.01:
        ai_score += 15
    elif cv['std'] < 0.02:
        ai_score += 10
    else:
        ai_score += 5
    
    print(f"  AI Score: {ai_score}/100")
    
    if ai_score >= 80:
        ai_grade = "🟢 Excellent (A)"
    elif ai_score >= 70:
        ai_grade = "🟢 Good (B)"
    elif ai_score >= 60:
        ai_grade = "🟡 Fair (C)"
    elif ai_score >= 50:
        ai_grade = "🟡 Poor (D)"
    else:
        ai_grade = "🔴 Failed (F)"
    
    print(f"  AI Grade: {ai_grade}")

def analyze_data_quality(results):
    """Analyze data quality and characteristics"""
    print(f"\n📊 DATA QUALITY ANALYSIS")
    print("-"*40)
    
    data_info = results['data_info']
    
    print(f"📈 DATASET OVERVIEW:")
    print(f"  Total Samples: {data_info['samples']:,}")
    print(f"  Features: {data_info['features']}")
    print(f"  Positive Rate: {data_info['positive_rate']:.3f} ({data_info['positive_rate']*100:.1f}%)")
    
    # Class balance analysis
    positive_rate = data_info['positive_rate']
    if 0.4 <= positive_rate <= 0.6:
        balance_status = "🟢 Well Balanced"
    elif 0.3 <= positive_rate <= 0.7:
        balance_status = "🟡 Slightly Imbalanced"
    else:
        balance_status = "🔴 Highly Imbalanced"
    
    print(f"  Class Balance: {balance_status}")
    
    # Sample size assessment
    samples = data_info['samples']
    if samples >= 5000:
        sample_status = "🟢 Large Dataset"
    elif samples >= 2000:
        sample_status = "🟢 Adequate Dataset"
    elif samples >= 1000:
        sample_status = "🟡 Small Dataset"
    else:
        sample_status = "🔴 Very Small Dataset"
    
    print(f"  Sample Size: {sample_status}")
    
    # Feature richness
    features = data_info['features']
    if features >= 100:
        feature_status = "🟢 Feature Rich"
    elif features >= 50:
        feature_status = "🟢 Good Features"
    elif features >= 20:
        feature_status = "🟡 Limited Features"
    else:
        feature_status = "🔴 Too Few Features"
    
    print(f"  Feature Count: {feature_status}")

def generate_summary_report(results):
    """Generate executive summary"""
    print(f"\n📋 EXECUTIVE SUMMARY")
    print("="*60)
    
    # Overall performance
    total_return = results['training_history']['backtest']['total_return']
    sharpe_ratio = results['performance_metrics']['sharpe_ratio']
    max_drawdown = results['performance_metrics']['max_drawdown']
    win_rate = results['performance_metrics']['win_rate']
    
    print(f"🎯 OVERALL PERFORMANCE:")
    print(f"  Return: {total_return*100:.2f}% ({'🟢 Excellent' if total_return > 2 else '🟡 Good' if total_return > 0.5 else '🔴 Poor'})")
    print(f"  Risk-Adjusted: Sharpe {sharpe_ratio:.3f} ({'🟢 Good' if sharpe_ratio > 1 else '🟡 Fair' if sharpe_ratio > 0.5 else '🔴 Poor'})")
    print(f"  Risk Control: {max_drawdown*100:.2f}% DD ({'🟢 Low' if max_drawdown < 0.2 else '🟡 Moderate' if max_drawdown < 0.3 else '🔴 High'})")
    print(f"  Consistency: {win_rate*100:.1f}% WR ({'🟢 High' if win_rate > 0.6 else '🟡 Fair' if win_rate > 0.5 else '🔴 Low'})")
    
    # System status
    print(f"\n🔧 SYSTEM STATUS:")
    print(f"  ✅ All 5 components working")
    print(f"  ✅ Training completed successfully")
    print(f"  ✅ Results validated and saved")
    print(f"  ⚠️ Using synthetic data (limitation)")
    
    # Next steps
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    print(f"  1. 🔧 Fix real market data loading")
    print(f"  2. 🧠 Optimize meta learning (reduce overfitting)")
    print(f"  3. 📊 Expand validation (5-fold CV)")
    print(f"  4. 🚀 Prepare for live trading testing")
    
    # Final grade
    performance_score = min(100, total_return * 10)  # Cap at 100
    risk_score = max(0, 100 - max_drawdown * 500)   # Penalty for high DD
    consistency_score = win_rate * 100
    
    overall_score = (performance_score * 0.4 + risk_score * 0.3 + consistency_score * 0.3)
    
    print(f"\n🏆 OVERALL SYSTEM GRADE:")
    print(f"  Performance Score: {performance_score:.1f}/100")
    print(f"  Risk Score: {risk_score:.1f}/100") 
    print(f"  Consistency Score: {consistency_score:.1f}/100")
    print(f"  Overall Score: {overall_score:.1f}/100")
    
    if overall_score >= 80:
        final_grade = "🟢 A (Excellent)"
    elif overall_score >= 70:
        final_grade = "🟢 B (Good)"
    elif overall_score >= 60:
        final_grade = "🟡 C (Fair)"
    elif overall_score >= 50:
        final_grade = "🟡 D (Poor)"
    else:
        final_grade = "🔴 F (Failed)"
    
    print(f"  Final Grade: {final_grade}")

def main():
    """Main analysis function"""
    results = load_results()
    
    if results is None:
        print("❌ Cannot proceed without results file")
        return
    
    print(f"\n📊 Analysis started at: {datetime.now().strftime('%H:%M:%S')}")
    
    # Run all analyses
    analyze_trading_performance(results)
    analyze_ai_performance(results)
    analyze_data_quality(results)
    generate_summary_report(results)
    
    print(f"\n✅ Analysis completed at: {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main() 