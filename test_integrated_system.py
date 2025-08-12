"""
Test AI3.0 Master System with Multi-Perspective Ensemble Integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

# Import AI3.0 Master System
from src.core.integration.master_system import (
    MasterIntegrationSystem, 
    SystemConfig, 
    MarketData, 
    SystemMode, 
    IntegrationLevel
)

def create_sample_market_data(symbol="XAUUSD", price=2000.0):
    """Create sample market data for testing"""
    return MarketData(
        timestamp=datetime.now(),
        symbol=symbol,
        price=price,
        high=price * 1.005,
        low=price * 0.995,
        volume=1000.0,
        technical_indicators={
            'rsi': 55.0,
            'macd': 0.5,
            'bb_upper': price * 1.02,
            'bb_lower': price * 0.98,
            'sma_20': price * 0.999,
            'ema_20': price * 1.001,
            'volatility': 0.6,
            'volume_ratio': 1.2
        }
    )

def test_system_initialization():
    """Test system initialization with Multi-Perspective Ensemble"""
    print("🔧 TESTING SYSTEM INITIALIZATION")
    print("=" * 50)
    
    # Create configuration with Multi-Perspective enabled
    config = SystemConfig(
        mode=SystemMode.SIMULATION,
        integration_level=IntegrationLevel.FULL,
        use_multi_perspective=True,
        enable_democratic_voting=True,
        multi_perspective_consensus_threshold=0.6
    )
    
    # Initialize system
    system = MasterIntegrationSystem(config)
    
    # Check system status
    status = system.get_system_status()
    
    print(f"✅ System Mode: {status['mode']}")
    print(f"✅ Integration Level: {status['integration_level']}")
    print(f"✅ Multi-Perspective Enabled: {config.use_multi_perspective}")
    print(f"✅ Democratic Voting: {config.enable_democratic_voting}")
    print(f"✅ Components Active: {status['components_active']}/{status['total_components']}")
    
    # Print component status
    print("\n📊 COMPONENT STATUS:")
    for component, active in status['components_status'].items():
        status_icon = "✅" if active else "❌"
        print(f"{status_icon} {component}: {'ACTIVE' if active else 'INACTIVE'}")
    
    return system

def test_multi_perspective_signal_generation(system):
    """Test Multi-Perspective signal generation"""
    print("\n🎯 TESTING MULTI-PERSPECTIVE SIGNAL GENERATION")
    print("=" * 50)
    
    # Create sample market data
    market_data = create_sample_market_data()
    
    print(f"📈 Market Data: {market_data.symbol} @ ${market_data.price}")
    print(f"📊 RSI: {market_data.technical_indicators['rsi']}")
    print(f"📊 MACD: {market_data.technical_indicators['macd']}")
    print(f"📊 Volatility: {market_data.technical_indicators['volatility']}")
    
    # Process market data through system
    system.add_market_data(market_data)
    
    # Get recent signals
    recent_signals = system.get_recent_signals(hours=1)
    
    print(f"\n🔥 SIGNALS GENERATED: {len(recent_signals)}")
    
    for i, signal in enumerate(recent_signals, 1):
        print(f"\n📡 SIGNAL {i}:")
        print(f"   Source: {signal.source}")
        print(f"   Type: {signal.signal_type}")
        print(f"   Confidence: {signal.confidence:.3f}")
        print(f"   Timestamp: {signal.timestamp}")
        
        if signal.metadata:
            print(f"   Metadata: {signal.metadata}")
    
    return recent_signals

def test_system_performance():
    """Test system performance with multiple data points"""
    print("\n⚡ TESTING SYSTEM PERFORMANCE")
    print("=" * 50)
    
    # Initialize system
    config = SystemConfig(
        mode=SystemMode.SIMULATION,
        use_multi_perspective=True,
        enable_democratic_voting=True
    )
    system = MasterIntegrationSystem(config)
    
    # Generate multiple market data points
    base_price = 2000.0
    num_tests = 10
    
    start_time = time.time()
    
    for i in range(num_tests):
        # Simulate price movement
        price_change = np.random.normal(0, 0.01)
        current_price = base_price * (1 + price_change)
        
        # Create market data
        market_data = create_sample_market_data(price=current_price)
        market_data.technical_indicators['rsi'] = 30 + (i * 5)  # Varying RSI
        
        # Process data
        system.add_market_data(market_data)
        
        print(f"📊 Test {i+1}/{num_tests}: Price=${current_price:.2f}, RSI={market_data.technical_indicators['rsi']:.1f}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Get all signals
    all_signals = system.get_recent_signals(hours=1)
    
    print(f"\n⚡ PERFORMANCE RESULTS:")
    print(f"   Processing Time: {processing_time:.3f} seconds")
    print(f"   Average per Signal: {processing_time/num_tests:.3f} seconds")
    print(f"   Total Signals: {len(all_signals)}")
    
    # Analyze signal distribution
    signal_types = {}
    sources = {}
    
    for signal in all_signals:
        signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1
        sources[signal.source] = sources.get(signal.source, 0) + 1
    
    print(f"\n📊 SIGNAL DISTRIBUTION:")
    for signal_type, count in signal_types.items():
        print(f"   {signal_type}: {count}")
    
    print(f"\n🔧 SOURCE DISTRIBUTION:")
    for source, count in sources.items():
        print(f"   {source}: {count}")
    
    return all_signals

def test_democratic_voting_integration():
    """Test democratic voting integration specifically"""
    print("\n🗳️ TESTING DEMOCRATIC VOTING INTEGRATION")
    print("=" * 50)
    
    # Create system with high Multi-Perspective focus
    config = SystemConfig(
        mode=SystemMode.SIMULATION,
        use_multi_perspective=True,
        enable_democratic_voting=True,
        multi_perspective_consensus_threshold=0.5,  # Lower threshold for testing
        use_neural_ensemble=False,  # Disable others to focus on MP
        use_reinforcement_learning=False
    )
    
    system = MasterIntegrationSystem(config)
    
    # Test with different market conditions
    test_conditions = [
        {"price": 2000.0, "rsi": 20, "macd": -1.0, "volatility": 0.8, "condition": "Oversold + High Volatility"},
        {"price": 2010.0, "rsi": 80, "macd": 1.0, "volatility": 0.3, "condition": "Overbought + Low Volatility"},
        {"price": 2005.0, "rsi": 50, "macd": 0.0, "volatility": 0.5, "condition": "Neutral Market"},
    ]
    
    for i, condition in enumerate(test_conditions, 1):
        print(f"\n🧪 TEST CONDITION {i}: {condition['condition']}")
        
        # Create market data
        market_data = create_sample_market_data(price=condition['price'])
        market_data.technical_indicators.update({
            'rsi': condition['rsi'],
            'macd': condition['macd'],
            'volatility': condition['volatility']
        })
        
        # Process data
        system.add_market_data(market_data)
        
        # Get signals
        signals = system.get_recent_signals(hours=1)
        
        # Find Multi-Perspective signals
        mp_signals = [s for s in signals if 'multi_perspective' in s.source]
        
        print(f"   📊 Market: RSI={condition['rsi']}, MACD={condition['macd']}, Vol={condition['volatility']}")
        print(f"   🔥 MP Signals: {len(mp_signals)}")
        
        for signal in mp_signals:
            print(f"   📡 {signal.signal_type} (confidence: {signal.confidence:.3f})")
            if signal.metadata and 'vote_distribution' in signal.metadata:
                print(f"   🗳️ Votes: {signal.metadata['vote_distribution']}")

def main():
    """Main test function"""
    print("🚀 AI3.0 MASTER SYSTEM WITH MULTI-PERSPECTIVE ENSEMBLE")
    print("=" * 60)
    print("Testing complete integration and functionality...")
    print()
    
    try:
        # Test system initialization
        system = test_system_initialization()
        
        # Test signal generation
        print("\n🎯 TESTING SIGNAL GENERATION")
        print("=" * 50)
        
        market_data = create_sample_market_data()
        print(f"📈 Processing: {market_data.symbol} @ ${market_data.price}")
        
        system.add_market_data(market_data)
        
        signals = system.get_recent_signals(hours=1)
        print(f"🔥 Signals Generated: {len(signals)}")
        
        for signal in signals:
            print(f"📡 {signal.source}: {signal.signal_type} (conf: {signal.confidence:.3f})")
        
        print("\n🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 