"""
Demo for Master Integration System
Ultimate XAU Super System V4.0 - Unified Integration

This demo showcases the complete integration of all system components:
1. System Initialization and Configuration
2. Component Integration Testing
3. Real-time Data Processing
4. Signal Generation and Combination
5. Performance Monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import threading
import warnings
warnings.filterwarnings('ignore')

from src.core.integration.master_system import (
    MasterIntegrationSystem, SystemConfig, SystemMode, IntegrationLevel,
    MarketData, TradingSignal,
    create_development_system, create_simulation_system, create_live_trading_system
)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def generate_realistic_market_stream(duration_minutes: int = 60) -> list:
    """Generate realistic market data stream"""
    print("📊 Generating realistic market data stream...")
    
    np.random.seed(42)
    data_points = []
    
    base_price = 2000.0
    current_price = base_price
    start_time = datetime.now()
    
    for i in range(duration_minutes):
        # Generate realistic price movement
        price_change = np.random.normal(0, 0.5)  # $0.50 average movement
        current_price += price_change
        current_price = max(current_price, 1500.0)  # Floor price
        
        # Generate OHLC
        high = current_price + np.random.uniform(0, 2)
        low = current_price - np.random.uniform(0, 2)
        volume = np.random.uniform(8000, 15000)
        
        # Technical indicators (simplified)
        rsi = 50 + np.random.normal(0, 15)
        rsi = max(0, min(100, rsi))
        
        macd = np.random.normal(0, 2)
        sma_20 = current_price + np.random.normal(0, 5)
        sma_50 = current_price + np.random.normal(0, 10)
        
        data_point = MarketData(
            timestamp=start_time + timedelta(minutes=i),
            symbol="XAUUSD",
            price=current_price,
            high=high,
            low=low,
            volume=volume,
            technical_indicators={
                'rsi': rsi,
                'macd': macd,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'bb_upper': current_price + 20,
                'bb_lower': current_price - 20
            }
        )
        
        data_points.append(data_point)
    
    print(f"✅ Generated {len(data_points)} market data points")
    print(f"   Price range: ${min(d.price for d in data_points):.2f} - ${max(d.price for d in data_points):.2f}")
    
    return data_points


def demo_system_initialization():
    """Demo 1: System Initialization and Configuration"""
    print("\n" + "="*70)
    print("🔧 DEMO 1: SYSTEM INITIALIZATION AND CONFIGURATION")
    print("="*70)
    
    # Create different system configurations
    systems = {}
    
    print("\n🏗️ Creating Different System Configurations...")
    
    # Development System
    print("\n1. Development System:")
    dev_system = create_development_system()
    systems['development'] = dev_system
    
    print(f"   Mode: {dev_system.config.mode.value}")
    print(f"   Integration Level: {dev_system.config.integration_level.value}")
    print(f"   Initial Balance: ${dev_system.config.initial_balance:,.2f}")
    print(f"   Neural Ensemble: {dev_system.config.use_neural_ensemble}")
    print(f"   Reinforcement Learning: {dev_system.config.use_reinforcement_learning}")
    print(f"   Components Active: {sum(dev_system.state.components_status.values())}")
    
    # Simulation System
    print("\n2. Simulation System:")
    sim_system = create_simulation_system()
    systems['simulation'] = sim_system
    
    print(f"   Mode: {sim_system.config.mode.value}")
    print(f"   Integration Level: {sim_system.config.integration_level.value}")
    print(f"   Initial Balance: ${sim_system.config.initial_balance:,.2f}")
    print(f"   Risk Tolerance: {sim_system.config.risk_tolerance:.1%}")
    print(f"   Max Position Size: {sim_system.config.max_position_size:.1%}")
    print(f"   Components Active: {sum(sim_system.state.components_status.values())}")
    
    # Live Trading System
    print("\n3. Live Trading System:")
    live_system = create_live_trading_system()
    systems['live'] = live_system
    
    print(f"   Mode: {live_system.config.mode.value}")
    print(f"   Integration Level: {live_system.config.integration_level.value}")
    print(f"   Initial Balance: ${live_system.config.initial_balance:,.2f}")
    print(f"   Risk Tolerance: {live_system.config.risk_tolerance:.1%}")
    print(f"   RL Exploration Rate: {live_system.config.rl_exploration_rate:.1%}")
    print(f"   Update Frequency: {live_system.config.update_frequency}s")
    print(f"   Components Active: {sum(live_system.state.components_status.values())}")
    
    # Custom Configuration
    print("\n4. Custom High-Performance System:")
    custom_config = SystemConfig(
        mode=SystemMode.SIMULATION,
        integration_level=IntegrationLevel.FULL,
        initial_balance=1000000.0,  # $1M
        max_position_size=0.1,      # 10% max
        risk_tolerance=0.005,       # 0.5% daily VaR
        ensemble_confidence_threshold=0.85,  # High confidence
        rl_exploration_rate=0.02,   # Low exploration
        update_frequency=0.1,       # Very fast updates
        max_concurrent_trades=3     # Conservative
    )
    custom_system = MasterIntegrationSystem(custom_config)
    systems['custom'] = custom_system
    
    print(f"   Mode: {custom_system.config.mode.value}")
    print(f"   Balance: ${custom_system.config.initial_balance:,.2f}")
    print(f"   Max Position: {custom_system.config.max_position_size:.1%}")
    print(f"   Risk Tolerance: {custom_system.config.risk_tolerance:.1%}")
    print(f"   Ensemble Threshold: {custom_system.config.ensemble_confidence_threshold:.1%}")
    print(f"   Components Active: {sum(custom_system.state.components_status.values())}")
    
    print(f"\n✅ Created {len(systems)} different system configurations")
    return systems


def demo_component_integration(system: MasterIntegrationSystem):
    """Demo 2: Component Integration Testing"""
    print("\n" + "="*70)
    print("🔗 DEMO 2: COMPONENT INTEGRATION TESTING")
    print("="*70)
    
    print(f"\n🧪 Testing Component Integration for {system.config.mode.value} system...")
    
    # Get system status
    status = system.get_system_status()
    
    print(f"\n📊 System Status Overview:")
    print(f"   Mode: {status['mode']}")
    print(f"   Integration Level: {status['integration_level']}")
    print(f"   Components Active: {status['components_active']}/{status['total_components']}")
    
    # Portfolio Status
    portfolio = status['portfolio']
    print(f"\n💰 Portfolio Status:")
    print(f"   Total Balance: ${portfolio['total_balance']:,.2f}")
    print(f"   Available Balance: ${portfolio['available_balance']:,.2f}")
    print(f"   Total Positions: {portfolio['total_positions']}")
    print(f"   Unrealized P&L: ${portfolio['unrealized_pnl']:,.2f}")
    
    # AI Status
    ai_status = status['ai_status']
    print(f"\n🤖 AI Systems Status:")
    print(f"   Neural Ensemble Active: {ai_status['neural_ensemble_active']}")
    print(f"   RL Agent Active: {ai_status['rl_agent_active']}")
    print(f"   Last Prediction Confidence: {ai_status['last_prediction_confidence']}")
    
    # Component Details
    print(f"\n🔧 Component Details:")
    for component, active in status['components_status'].items():
        status_icon = "✅" if active else "❌"
        print(f"   {status_icon} {component}: {'Active' if active else 'Inactive'}")
    
    # Test component availability
    print(f"\n🧪 Component Availability Test:")
    
    # Check Phase 1 components
    phase1_components = ['var_calculator', 'risk_monitor', 'position_sizer', 'kelly_criterion', 'portfolio_manager']
    phase1_active = sum(1 for comp in phase1_components if status['components_status'].get(comp, False))
    print(f"   Phase 1 (Risk Management): {phase1_active}/{len(phase1_components)} components active")
    
    # Check Phase 2 components
    phase2_components = ['neural_ensemble', 'rl_agent']
    phase2_active = sum(1 for comp in phase2_components if status['components_status'].get(comp, False))
    print(f"   Phase 2 (AI Systems): {phase2_active}/{len(phase2_components)} components active")
    
    # Integration score
    total_possible = len(phase1_components) + len(phase2_components)
    total_active = phase1_active + phase2_active
    integration_score = (total_active / total_possible) * 100
    
    print(f"\n🎯 Integration Score: {integration_score:.1f}% ({total_active}/{total_possible} components)")
    
    if integration_score >= 80:
        rating = "🌟 EXCELLENT"
    elif integration_score >= 60:
        rating = "✅ GOOD"
    elif integration_score >= 40:
        rating = "⚠️ MODERATE"
    else:
        rating = "❌ POOR"
    
    print(f"   Integration Rating: {rating}")
    
    return status


def demo_real_time_processing(system: MasterIntegrationSystem, market_data: list):
    """Demo 3: Real-time Data Processing"""
    print("\n" + "="*70)
    print("⚡ DEMO 3: REAL-TIME DATA PROCESSING")
    print("="*70)
    
    print(f"\n🎮 Starting Real-time Processing Simulation...")
    print(f"   System: {system.config.mode.value}")
    print(f"   Data Points: {len(market_data)}")
    print(f"   Update Frequency: {system.config.update_frequency}s")
    
    # Start real-time processing
    system.start_real_time_processing()
    
    # Process market data in real-time
    processing_log = []
    start_time = time.time()
    
    print(f"\n📈 Processing Market Data Stream...")
    
    for i, data_point in enumerate(market_data[:30]):  # Process first 30 points for demo
        # Add market data
        system.add_market_data(data_point)
        
        # Log processing
        processing_time = time.time() - start_time
        processing_log.append({
            'step': i + 1,
            'timestamp': data_point.timestamp,
            'price': data_point.price,
            'buffer_size': len(system.market_data_buffer),
            'signals_count': len(system.signals_history),
            'processing_time': processing_time
        })
        
        # Print periodic updates
        if (i + 1) % 5 == 0:
            recent_signals = system.get_recent_signals(hours=1)
            print(f"   Step {i+1:2d}: Price=${data_point.price:7.2f}, "
                  f"Buffer={len(system.market_data_buffer):3d}, "
                  f"Signals={len(recent_signals):2d}, "
                  f"Time={processing_time:5.2f}s")
        
        # Small delay to simulate real-time
        time.sleep(0.1)
    
    # Stop real-time processing
    system.stop_real_time_processing()
    
    # Analysis
    total_time = time.time() - start_time
    avg_processing_time = total_time / len(processing_log)
    
    print(f"\n📊 Real-time Processing Results:")
    print(f"   Total Data Points Processed: {len(processing_log)}")
    print(f"   Total Processing Time: {total_time:.2f}s")
    print(f"   Average Processing Time: {avg_processing_time:.3f}s per point")
    print(f"   Processing Rate: {len(processing_log)/total_time:.1f} points/second")
    print(f"   Final Buffer Size: {len(system.market_data_buffer)}")
    print(f"   Total Signals Generated: {len(system.signals_history)}")
    
    # Performance rating
    if avg_processing_time < 0.1:
        performance = "🌟 EXCELLENT"
    elif avg_processing_time < 0.5:
        performance = "✅ GOOD"
    elif avg_processing_time < 1.0:
        performance = "⚠️ MODERATE"
    else:
        performance = "❌ SLOW"
    
    print(f"   Performance Rating: {performance}")
    
    return processing_log


def demo_signal_generation(system: MasterIntegrationSystem, market_data: list):
    """Demo 4: Signal Generation and Combination"""
    print("\n" + "="*70)
    print("📡 DEMO 4: SIGNAL GENERATION AND COMBINATION")
    print("="*70)
    
    print(f"\n🎯 Testing Signal Generation Pipeline...")
    
    # Process enough data to generate signals
    for data_point in market_data[:50]:
        system.add_market_data(data_point)
    
    # Get generated signals
    all_signals = system.signals_history
    recent_signals = system.get_recent_signals(hours=24)  # All signals for demo
    
    print(f"\n📊 Signal Generation Results:")
    print(f"   Total Signals Generated: {len(all_signals)}")
    print(f"   Recent Signals (24h): {len(recent_signals)}")
    
    if recent_signals:
        # Analyze signal types
        signal_types = {}
        signal_sources = {}
        confidence_scores = []
        
        for signal in recent_signals:
            # Count signal types
            signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1
            
            # Count signal sources
            signal_sources[signal.source] = signal_sources.get(signal.source, 0) + 1
            
            # Collect confidence scores
            confidence_scores.append(signal.confidence)
        
        print(f"\n📈 Signal Type Distribution:")
        for signal_type, count in signal_types.items():
            percentage = (count / len(recent_signals)) * 100
            print(f"   {signal_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔍 Signal Source Distribution:")
        for source, count in signal_sources.items():
            percentage = (count / len(recent_signals)) * 100
            print(f"   {source}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 Signal Quality Metrics:")
        avg_confidence = np.mean(confidence_scores)
        min_confidence = np.min(confidence_scores)
        max_confidence = np.max(confidence_scores)
        std_confidence = np.std(confidence_scores)
        
        print(f"   Average Confidence: {avg_confidence:.2f}")
        print(f"   Confidence Range: {min_confidence:.2f} - {max_confidence:.2f}")
        print(f"   Confidence Std Dev: {std_confidence:.2f}")
        
        # Quality rating
        if avg_confidence >= 0.8:
            quality = "🌟 EXCELLENT"
        elif avg_confidence >= 0.7:
            quality = "✅ GOOD"
        elif avg_confidence >= 0.6:
            quality = "⚠️ MODERATE"
        else:
            quality = "❌ POOR"
        
        print(f"   Signal Quality: {quality}")
        
        # Show recent signals
        print(f"\n📋 Recent Signals (Last 5):")
        for i, signal in enumerate(recent_signals[-5:]):
            print(f"   {i+1}. {signal.timestamp.strftime('%H:%M:%S')} | "
                  f"{signal.signal_type:4s} | "
                  f"Conf: {signal.confidence:.2f} | "
                  f"Source: {signal.source}")
    
    else:
        print("   No signals generated yet - system may need more data or time")
    
    return recent_signals


def demo_performance_monitoring(system: MasterIntegrationSystem):
    """Demo 5: Performance Monitoring"""
    print("\n" + "="*70)
    print("📈 DEMO 5: PERFORMANCE MONITORING")
    print("="*70)
    
    print(f"\n📊 System Performance Analysis...")
    
    # Get comprehensive system status
    status = system.get_system_status()
    
    # Portfolio Performance
    portfolio = status['portfolio']
    initial_balance = system.config.initial_balance
    current_balance = portfolio['total_balance']
    total_return = (current_balance - initial_balance) / initial_balance
    
    print(f"\n💰 Portfolio Performance:")
    print(f"   Initial Balance: ${initial_balance:,.2f}")
    print(f"   Current Balance: ${current_balance:,.2f}")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Available Balance: ${portfolio['available_balance']:,.2f}")
    print(f"   Active Positions: {portfolio['total_positions']}")
    print(f"   Unrealized P&L: ${portfolio['unrealized_pnl']:,.2f}")
    
    # Trading Performance
    performance = status['performance']
    print(f"\n📈 Trading Performance:")
    print(f"   Total Return: {performance['total_return']:.2%}")
    
    if performance['sharpe_ratio'] is not None:
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    else:
        print(f"   Sharpe Ratio: Not available (insufficient data)")
    
    if performance['win_rate'] is not None:
        print(f"   Win Rate: {performance['win_rate']:.1%}")
    else:
        print(f"   Win Rate: Not available (no completed trades)")
    
    # Signal Performance
    signals_info = status['signals']
    print(f"\n📡 Signal Performance:")
    print(f"   Total Signals: {signals_info['total_signals']}")
    print(f"   Recent Signals (1h): {signals_info['recent_signals']}")
    
    if signals_info['total_signals'] > 0:
        signal_rate = signals_info['recent_signals'] / 1  # per hour
        print(f"   Signal Generation Rate: {signal_rate:.1f} signals/hour")
    
    # AI Performance
    ai_status = status['ai_status']
    print(f"\n🤖 AI System Performance:")
    print(f"   Neural Ensemble: {'Active' if ai_status['neural_ensemble_active'] else 'Inactive'}")
    print(f"   RL Agent: {'Active' if ai_status['rl_agent_active'] else 'Inactive'}")
    
    if ai_status['last_prediction_confidence'] is not None:
        print(f"   Last Prediction Confidence: {ai_status['last_prediction_confidence']:.2f}")
    else:
        print(f"   Last Prediction Confidence: No recent predictions")
    
    # System Health
    print(f"\n🔧 System Health:")
    print(f"   Components Active: {status['components_active']}/{status['total_components']}")
    
    if status['total_components'] > 0:
        health_score = (status['components_active'] / status['total_components']) * 100
    else:
        health_score = 0.0
    print(f"   System Health Score: {health_score:.1f}%")
    
    if health_score >= 90:
        health_rating = "🌟 EXCELLENT"
    elif health_score >= 75:
        health_rating = "✅ GOOD"
    elif health_score >= 50:
        health_rating = "⚠️ MODERATE"
    else:
        health_rating = "❌ POOR"
    
    print(f"   Health Rating: {health_rating}")
    
    # Overall Performance Rating
    print(f"\n🎯 Overall System Rating:")
    
    # Calculate composite score
    portfolio_score = 50 if total_return >= 0 else 30
    signal_score = 30 if signals_info['total_signals'] > 0 else 10
    health_score_weighted = health_score * 0.2
    
    overall_score = portfolio_score + signal_score + health_score_weighted
    
    if overall_score >= 90:
        overall_rating = "🌟 EXCELLENT"
    elif overall_score >= 75:
        overall_rating = "✅ GOOD"
    elif overall_score >= 60:
        overall_rating = "⚠️ MODERATE"
    else:
        overall_rating = "❌ NEEDS IMPROVEMENT"
    
    print(f"   Overall Score: {overall_score:.1f}/100")
    print(f"   Overall Rating: {overall_rating}")
    
    return status


def main():
    """Main demo function"""
    print("🔧 MASTER INTEGRATION SYSTEM DEMO")
    print("Ultimate XAU Super System V4.0 - Unified Integration")
    print("=" * 80)
    
    try:
        # Generate market data
        market_data = generate_realistic_market_stream(duration_minutes=60)
        
        # Demo 1: System Initialization
        systems = demo_system_initialization()
        
        # Use simulation system for remaining demos
        main_system = systems['simulation']
        
        # Demo 2: Component Integration
        integration_status = demo_component_integration(main_system)
        
        # Demo 3: Real-time Processing
        processing_log = demo_real_time_processing(main_system, market_data)
        
        # Demo 4: Signal Generation
        signals = demo_signal_generation(main_system, market_data)
        
        # Demo 5: Performance Monitoring
        performance_status = demo_performance_monitoring(main_system)
        
        print("\n" + "="*80)
        print("🎉 MASTER INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Final Summary
        print(f"\n📋 FINAL SUMMARY:")
        print(f"   Systems Created: {len(systems)}")
        print(f"   Market Data Points: {len(market_data)}")
        print(f"   Processing Steps: {len(processing_log)}")
        print(f"   Signals Generated: {len(signals)}")
        print(f"   Components Active: {integration_status['components_active']}")
        print(f"   Integration Level: {integration_status['integration_level']}")
        
        # System Readiness Assessment
        if integration_status['total_components'] > 0:
            components_ratio = integration_status['components_active'] / integration_status['total_components']
        else:
            components_ratio = 0.0
        signals_generated = len(signals) > 0
        processing_successful = len(processing_log) > 0
        
        readiness_score = (
            (components_ratio * 40) +
            (40 if signals_generated else 0) +
            (20 if processing_successful else 0)
        )
        
        print(f"\n🎯 System Readiness Assessment:")
        print(f"   Readiness Score: {readiness_score:.1f}/100")
        
        if readiness_score >= 90:
            readiness = "🌟 FULLY READY for production"
        elif readiness_score >= 75:
            readiness = "✅ READY for advanced testing"
        elif readiness_score >= 60:
            readiness = "⚠️ PARTIALLY READY - needs optimization"
        else:
            readiness = "❌ NOT READY - requires fixes"
        
        print(f"   Status: {readiness}")
        
        print("\n✅ Master Integration System successfully demonstrated!")
        print("🚀 Ready for Phase 2 continuation with Day 17!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()