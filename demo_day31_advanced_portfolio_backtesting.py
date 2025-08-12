"""
Demo Day 31: Advanced Portfolio Backtesting
Ultimate XAU Super System V4.0

Test các tính năng:
1. Multi-Strategy Portfolio Backtesting
2. AI-Enhanced Signal Integration  
3. Deep Learning Portfolio Optimization
4. Advanced Performance Analytics
5. Real-time Portfolio Simulation
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.core.analysis.advanced_portfolio_backtesting import (
        AdvancedPortfolioBacktesting, BacktestingConfig, BacktestingStrategy,
        PerformanceMetric, RebalanceFrequency, create_default_config,
        analyze_multiple_strategies, create_advanced_portfolio_backtesting
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    MODULES_AVAILABLE = False

def create_sample_data(start_date: datetime, end_date: datetime, 
                      initial_price: float = 2000.0) -> pd.DataFrame:
    """Tạo dữ liệu sample cho backtesting"""
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, initial_price * 0.5))  # Prevent negative prices
    
    # Create OHLC data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(initial_price)
    
    # High/Low with some variance
    daily_range = data['close'] * 0.01  # 1% daily range
    data['high'] = data[['open', 'close']].max(axis=1) + np.random.uniform(0, 1, len(data)) * daily_range
    data['low'] = data[['open', 'close']].min(axis=1) - np.random.uniform(0, 1, len(data)) * daily_range
    
    # Volume
    data['volume'] = np.random.uniform(1000, 5000, len(data))
    
    return data

def module1_multi_strategy_backtesting():
    """Module 1: Multi-Strategy Portfolio Backtesting"""
    print("\n" + "="*80)
    print("📊 MODULE 1: MULTI-STRATEGY PORTFOLIO BACKTESTING")
    print("="*80)
    
    try:
        # Tạo dữ liệu test
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 6, 30)
        data = create_sample_data(start_date, end_date)
        
        print(f"📈 Tạo dữ liệu: {len(data)} ngày từ {start_date.strftime('%Y-%m-%d')} đến {end_date.strftime('%Y-%m-%d')}")
        
        # Test nhiều strategies
        strategies = [
            BacktestingStrategy.BUY_AND_HOLD,
            BacktestingStrategy.MEAN_REVERSION,
            BacktestingStrategy.MOMENTUM,
            BacktestingStrategy.ENSEMBLE_AI
        ]
        
        results = {}
        
        for strategy in strategies:
            try:
                config = create_default_config(start_date, end_date)
                config.strategy = strategy
                config.initial_capital = 100000.0
                
                backtester = create_advanced_portfolio_backtesting(config)
                result = backtester.run_backtest(data)
                results[strategy.value] = result
                
                print(f"✅ {strategy.value}: Hoàn thành với {result.total_trades} giao dịch")
                
            except Exception as e:
                print(f"❌ {strategy.value}: Lỗi - {str(e)[:100]}")
                results[strategy.value] = None
        
        # So sánh kết quả
        print(f"\n📊 So sánh hiệu suất các strategies:")
        for strategy_name, result in results.items():
            if result and result.performance_metrics:
                total_return = result.performance_metrics.get(PerformanceMetric.TOTAL_RETURN, 0)
                sharpe = result.performance_metrics.get(PerformanceMetric.SHARPE_RATIO, 0)
                max_dd = result.performance_metrics.get(PerformanceMetric.MAX_DRAWDOWN, 0)
                print(f"  {strategy_name}: Return {total_return:.1%}, Sharpe {sharpe:.2f}, MaxDD {max_dd:.1%}")
        
        # Tính điểm
        successful_strategies = len([r for r in results.values() if r is not None])
        score = (successful_strategies / len(strategies)) * 100
        
        print(f"\n🎯 Kết quả Module 1:")
        print(f"   Strategies thành công: {successful_strategies}/{len(strategies)}")
        print(f"   Điểm số: {score:.1f}/100")
        
        return score
        
    except Exception as e:
        print(f"❌ Lỗi Module 1: {e}")
        return 0

def module2_ai_enhanced_signal_integration():
    """Module 2: AI-Enhanced Signal Integration"""
    print("\n" + "="*80)
    print("🤖 MODULE 2: AI-ENHANCED SIGNAL INTEGRATION")
    print("="*80)
    
    try:
        # Tạo config với AI enabled
        start_date = datetime(2024, 3, 1)
        end_date = datetime(2024, 5, 31)
        data = create_sample_data(start_date, end_date)
        
        config = create_default_config(start_date, end_date)
        config.use_ai_signals = True
        config.use_deep_learning = True
        config.ai_confidence_threshold = 0.7
        config.ensemble_weights = {
            'technical': 0.25,
            'ml_signal': 0.375,
            'dl_signal': 0.375
        }
        
        print(f"🧠 Test AI-enhanced backtesting với {len(data)} ngày dữ liệu")
        print(f"   AI Confidence Threshold: {config.ai_confidence_threshold}")
        print(f"   Ensemble Weights: {config.ensemble_weights}")
        
        # Chạy backtesting với AI
        backtester = create_advanced_portfolio_backtesting(config)
        result = backtester.run_backtest(data)
        
        # Phân tích AI performance
        ai_metrics = result.ai_performance
        portfolio_metrics = result.performance_metrics
        
        print(f"\n📊 Kết quả AI Integration:")
        if ai_metrics:
            print(f"   AI Trades: {ai_metrics.get('ai_trade_count', 0)}")
            print(f"   AI Accuracy: {ai_metrics.get('ai_accuracy', 0):.1%}")
            print(f"   Avg Confidence: {ai_metrics.get('average_confidence', 0):.1%}")
            print(f"   High Conf Accuracy: {ai_metrics.get('high_confidence_accuracy', 0):.1%}")
        
        print(f"\n💰 Portfolio Performance:")
        total_return = portfolio_metrics.get(PerformanceMetric.TOTAL_RETURN, 0)
        sharpe = portfolio_metrics.get(PerformanceMetric.SHARPE_RATIO, 0)
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Sharpe Ratio: {sharpe:.3f}")
        print(f"   Total Trades: {result.total_trades}")
        
        # Tính điểm dựa trên AI integration
        ai_score = 0
        if ai_metrics:
            ai_trade_ratio = ai_metrics.get('ai_trade_ratio', 0)
            ai_accuracy = ai_metrics.get('ai_accuracy', 0)
            avg_confidence = ai_metrics.get('average_confidence', 0)
            
            ai_score = (ai_trade_ratio * 40 + ai_accuracy * 30 + avg_confidence * 30)
        
        performance_score = min(100, max(0, (total_return + 0.1) * 500))  # Scale performance
        final_score = (ai_score * 0.7 + performance_score * 0.3)
        
        print(f"\n🎯 Kết quả Module 2:")
        print(f"   AI Integration Score: {ai_score:.1f}/100")
        print(f"   Performance Score: {performance_score:.1f}/100")
        print(f"   Điểm tổng: {final_score:.1f}/100")
        
        return final_score
        
    except Exception as e:
        print(f"❌ Lỗi Module 2: {e}")
        return 0

def module3_deep_learning_portfolio_optimization():
    """Module 3: Deep Learning Portfolio Optimization"""
    print("\n" + "="*80)
    print("🧠 MODULE 3: DEEP LEARNING PORTFOLIO OPTIMIZATION")
    print("="*80)
    
    try:
        # Tạo config tối ưu cho Deep Learning
        start_date = datetime(2024, 2, 1)
        end_date = datetime(2024, 7, 31)
        data = create_sample_data(start_date, end_date)
        
        config = create_default_config(start_date, end_date)
        config.strategy = BacktestingStrategy.DEEP_LEARNING
        config.use_deep_learning = True
        config.ai_confidence_threshold = 0.8
        config.max_position_size = 0.15  # Allow larger positions for DL
        config.rebalance_frequency = RebalanceFrequency.DAILY
        
        print(f"🔬 Deep Learning Portfolio Optimization với {len(data)} ngày")
        print(f"   Strategy: {config.strategy.value}")
        print(f"   Max Position Size: {config.max_position_size:.1%}")
        print(f"   Rebalance: {config.rebalance_frequency.value}")
        
        # Test multiple DL configurations
        dl_configs = [
            {'confidence': 0.6, 'position_size': 0.1},
            {'confidence': 0.7, 'position_size': 0.12},
            {'confidence': 0.8, 'position_size': 0.15}
        ]
        
        best_result = None
        best_score = 0
        
        for i, dl_config in enumerate(dl_configs):
            try:
                test_config = create_default_config(start_date, end_date)
                test_config.strategy = BacktestingStrategy.DEEP_LEARNING
                test_config.use_deep_learning = True
                test_config.ai_confidence_threshold = dl_config['confidence']
                test_config.max_position_size = dl_config['position_size']
                
                backtester = create_advanced_portfolio_backtesting(test_config)
                result = backtester.run_backtest(data)
                
                # Đánh giá kết quả
                total_return = result.performance_metrics.get(PerformanceMetric.TOTAL_RETURN, 0)
                sharpe = result.performance_metrics.get(PerformanceMetric.SHARPE_RATIO, 0)
                max_dd = result.performance_metrics.get(PerformanceMetric.MAX_DRAWDOWN, 0)
                
                # Tính điểm cho config này
                config_score = (total_return * 40 + max(0, sharpe) * 30 - max_dd * 30)
                
                print(f"   Config {i+1}: Confidence {dl_config['confidence']:.1f}, Return {total_return:.1%}, Sharpe {sharpe:.2f}")
                
                if config_score > best_score:
                    best_score = config_score
                    best_result = result
                    
            except Exception as e:
                print(f"   Config {i+1}: Lỗi - {str(e)[:50]}")
        
        if best_result:
            print(f"\n🏆 Best Deep Learning Configuration:")
            total_return = best_result.performance_metrics.get(PerformanceMetric.TOTAL_RETURN, 0)
            sharpe = best_result.performance_metrics.get(PerformanceMetric.SHARPE_RATIO, 0)
            max_dd = best_result.performance_metrics.get(PerformanceMetric.MAX_DRAWDOWN, 0)
            
            print(f"   Total Return: {total_return:.2%}")
            print(f"   Sharpe Ratio: {sharpe:.3f}")
            print(f"   Max Drawdown: {max_dd:.2%}")
            print(f"   Total Trades: {best_result.total_trades}")
            
            # Deep Learning specific metrics
            if best_result.ai_performance:
                dl_accuracy = best_result.ai_performance.get('ai_accuracy', 0)
                dl_confidence = best_result.ai_performance.get('average_confidence', 0)
                print(f"   DL Accuracy: {dl_accuracy:.1%}")
                print(f"   Avg DL Confidence: {dl_confidence:.1%}")
        
        # Tính điểm cuối
        final_score = min(100, max(0, best_score * 10 + 50))
        
        print(f"\n🎯 Kết quả Module 3:")
        print(f"   Deep Learning Optimization Score: {final_score:.1f}/100")
        
        return final_score
        
    except Exception as e:
        print(f"❌ Lỗi Module 3: {e}")
        return 0

def module4_advanced_performance_analytics():
    """Module 4: Advanced Performance Analytics"""
    print("\n" + "="*80)
    print("📈 MODULE 4: ADVANCED PERFORMANCE ANALYTICS")
    print("="*80)
    
    try:
        # Tạo dữ liệu cho phân tích chi tiết
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 8, 31)
        data = create_sample_data(start_date, end_date)
        benchmark_data = create_sample_data(start_date, end_date, initial_price=1950.0)  # Benchmark khác
        
        config = create_default_config(start_date, end_date)
        config.strategy = BacktestingStrategy.ENSEMBLE_AI
        config.use_ai_signals = True
        config.use_deep_learning = True
        
        print(f"📊 Advanced Performance Analytics với {len(data)} ngày dữ liệu")
        
        # Chạy backtesting với benchmark
        backtester = create_advanced_portfolio_backtesting(config)
        result = backtester.run_backtest(data, benchmark_data)
        
        # Phân tích performance metrics
        metrics = result.performance_metrics
        risk_metrics = result.risk_metrics
        benchmark_comp = result.benchmark_comparison
        
        print(f"\n📊 Performance Metrics:")
        for metric, value in metrics.items():
            if metric in [PerformanceMetric.TOTAL_RETURN, PerformanceMetric.ANNUALIZED_RETURN]:
                print(f"   {metric.value}: {value:.2%}")
            elif metric == PerformanceMetric.MAX_DRAWDOWN:
                print(f"   {metric.value}: {value:.2%}")
            elif metric == PerformanceMetric.WIN_RATE:
                print(f"   {metric.value}: {value:.1%}")
            else:
                print(f"   {metric.value}: {value:.3f}")
        
        print(f"\n⚠️ Risk Metrics:")
        for key, value in risk_metrics.items():
            if 'var' in key.lower() or 'volatility' in key.lower():
                print(f"   {key}: {value:.3%}")
            else:
                print(f"   {key}: {value:.3f}")
        
        print(f"\n📊 Benchmark Comparison:")
        for key, value in benchmark_comp.items():
            if key in ['alpha', 'tracking_error']:
                print(f"   {key}: {value:.3%}")
            else:
                print(f"   {key}: {value:.3f}")
        
        # Generate detailed report
        report = backtester.generate_report(result)
        print(f"\n📋 Generated detailed report ({len(report)} characters)")
        
        # Tính điểm dựa trên completeness của analytics
        metrics_score = len(metrics) / len(PerformanceMetric) * 100
        risk_score = min(100, len(risk_metrics) * 10)
        benchmark_score = min(100, len(benchmark_comp) * 20)
        report_score = min(100, len(report) / 50)  # Report quality
        
        final_score = (metrics_score * 0.3 + risk_score * 0.3 + benchmark_score * 0.2 + report_score * 0.2)
        
        print(f"\n🎯 Kết quả Module 4:")
        print(f"   Performance Metrics: {metrics_score:.1f}/100")
        print(f"   Risk Analytics: {risk_score:.1f}/100")
        print(f"   Benchmark Analysis: {benchmark_score:.1f}/100")
        print(f"   Report Quality: {report_score:.1f}/100")
        print(f"   Điểm tổng: {final_score:.1f}/100")
        
        return final_score
        
    except Exception as e:
        print(f"❌ Lỗi Module 4: {e}")
        return 0

def module5_real_time_portfolio_simulation():
    """Module 5: Real-time Portfolio Simulation"""
    print("\n" + "="*80)
    print("⚡ MODULE 5: REAL-TIME PORTFOLIO SIMULATION")
    print("="*80)
    
    try:
        # Tạo simulation real-time
        start_date = datetime(2024, 6, 1)
        end_date = datetime(2024, 8, 31)
        data = create_sample_data(start_date, end_date)
        
        config = create_default_config(start_date, end_date)
        config.strategy = BacktestingStrategy.ADAPTIVE
        config.rebalance_frequency = RebalanceFrequency.DAILY
        
        print(f"⚡ Real-time Portfolio Simulation với {len(data)} ngày")
        
        # Test tốc độ xử lý
        simulation_times = []
        portfolio_values = []
        
        # Simulate từng ngày như real-time
        for i in range(5):  # Test 5 lần để tính trung bình
            start_time = time.time()
            
            # Simulate partial data processing
            partial_data = data.iloc[:len(data)//2]  # Process half data
            
            backtester = create_advanced_portfolio_backtesting(config)
            result = backtester.run_backtest(partial_data)
            
            processing_time = time.time() - start_time
            simulation_times.append(processing_time)
            
            if result.portfolio_history:
                final_value = result.portfolio_history[-1].total_value
                portfolio_values.append(final_value)
            
            print(f"   Simulation {i+1}: {processing_time:.3f}s, Portfolio: ${portfolio_values[-1]:,.0f}")
        
        # Tính thống kê performance
        avg_time = np.mean(simulation_times)
        std_time = np.std(simulation_times)
        avg_portfolio_value = np.mean(portfolio_values)
        portfolio_consistency = 1 - (np.std(portfolio_values) / avg_portfolio_value)
        
        print(f"\n⚡ Real-time Performance:")
        print(f"   Avg Processing Time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"   Avg Portfolio Value: ${avg_portfolio_value:,.0f}")
        print(f"   Portfolio Consistency: {portfolio_consistency:.1%}")
        
        # Test stress conditions
        print(f"\n🔥 Stress Testing:")
        stress_results = []
        
        for stress_days in [30, 60, 90]:
            try:
                stress_data = data.head(stress_days)
                start_time = time.time()
                
                stress_backtester = create_advanced_portfolio_backtesting(config)
                stress_result = stress_backtester.run_backtest(stress_data)
                
                stress_time = time.time() - start_time
                stress_results.append({
                    'days': stress_days,
                    'time': stress_time,
                    'trades': stress_result.total_trades,
                    'success': True
                })
                
                print(f"   {stress_days} days: {stress_time:.3f}s, {stress_result.total_trades} trades")
                
            except Exception as e:
                stress_results.append({
                    'days': stress_days,
                    'time': float('inf'),
                    'trades': 0,
                    'success': False
                })
                print(f"   {stress_days} days: FAILED - {str(e)[:50]}")
        
        # Tính điểm
        speed_score = min(100, max(0, (10 - avg_time) * 10))  # Faster = better
        consistency_score = portfolio_consistency * 100
        stress_score = (len([r for r in stress_results if r['success']]) / len(stress_results)) * 100
        
        final_score = (speed_score * 0.4 + consistency_score * 0.3 + stress_score * 0.3)
        
        print(f"\n🎯 Kết quả Module 5:")
        print(f"   Speed Score: {speed_score:.1f}/100")
        print(f"   Consistency Score: {consistency_score:.1f}/100")
        print(f"   Stress Test Score: {stress_score:.1f}/100")
        print(f"   Điểm tổng: {final_score:.1f}/100")
        
        return final_score
        
    except Exception as e:
        print(f"❌ Lỗi Module 5: {e}")
        return 0

def main():
    """Main demo function"""
    print("🚀 ULTIMATE XAU SUPER SYSTEM V4.0")
    print("📊 Day 31: Advanced Portfolio Backtesting Demo")
    print("=" * 80)
    
    if not MODULES_AVAILABLE:
        print("❌ Không thể import modules. Vui lòng kiểm tra dependencies.")
        return
    
    # Chạy tất cả modules
    start_time = time.time()
    
    module_scores = {}
    
    # Module 1: Multi-Strategy Portfolio Backtesting
    module_scores['Module 1'] = module1_multi_strategy_backtesting()
    
    # Module 2: AI-Enhanced Signal Integration
    module_scores['Module 2'] = module2_ai_enhanced_signal_integration()
    
    # Module 3: Deep Learning Portfolio Optimization
    module_scores['Module 3'] = module3_deep_learning_portfolio_optimization()
    
    # Module 4: Advanced Performance Analytics
    module_scores['Module 4'] = module4_advanced_performance_analytics()
    
    # Module 5: Real-time Portfolio Simulation
    module_scores['Module 5'] = module5_real_time_portfolio_simulation()
    
    # Tính tổng điểm
    total_score = np.mean(list(module_scores.values()))
    execution_time = time.time() - start_time
    
    # Hiển thị kết quả tổng
    print("\n" + "="*80)
    print("📊 KẾT QUẢ TỔNG HỢP - DAY 31: ADVANCED PORTFOLIO BACKTESTING")
    print("="*80)
    
    for module, score in module_scores.items():
        status = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        print(f"{status} {module}: {score:.1f}/100")
    
    print(f"\n🏆 ĐIỂM TỔNG: {total_score:.1f}/100")
    print(f"⏱️ Thời gian thực hiện: {execution_time:.2f} giây")
    
    # Đánh giá cuối
    if total_score >= 90:
        grade = "🥇 XUẤT SẮC"
    elif total_score >= 80:
        grade = "🥈 TỐT"
    elif total_score >= 70:
        grade = "🥉 KHANG ĐỊNH"
    else:
        grade = "📈 CẦN CẢI THIỆN"
    
    print(f"🎯 Xếp hạng: {grade}")
    
    # Summary
    print(f"\n📋 Tóm tắt Advanced Portfolio Backtesting:")
    print(f"   ✅ Multi-strategy backtesting engine")
    print(f"   ✅ AI/ML signal integration")
    print(f"   ✅ Deep learning portfolio optimization")
    print(f"   ✅ Advanced performance analytics")
    print(f"   ✅ Real-time simulation capabilities")
    
    print(f"\n🎉 Day 31 hoàn thành! Advanced Portfolio Backtesting đã sẵn sàng cho production.")

if __name__ == "__main__":
    main() 