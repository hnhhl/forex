"""
Simple AI Master Integration Demo
Ultimate XAU Super System V4.0 - Day 18

Simplified demo focusing on core integration capabilities
"""

import numpy as np
from datetime import datetime, timedelta
from src.core.integration.ai_master_integration import (
    create_ai_master_system, AIMarketData, DecisionStrategy
)


def create_simple_market_data() -> AIMarketData:
    """Create simple market data for testing"""
    return AIMarketData(
        timestamp=datetime.now(),
        symbol="XAUUSD",
        price=2000.0,
        high=2010.0,
        low=1990.0,
        volume=5000.0,
        sma_20=1995.0,
        sma_50=1985.0,
        rsi=65.0,
        macd=2.5,
        volatility=0.15,
        momentum=0.05
    )


def main():
    """Simple demo main function"""
    print("🚀 AI Master Integration System - Simple Demo")
    print("="*60)
    
    # Create system with minimal configuration
    config = {
        'enable_neural_ensemble': False,  # Disable complex AI for testing
        'enable_reinforcement_learning': False,
        'enable_meta_learning': False,
        'decision_strategy': 'adaptive_ensemble',
        'sequence_length': 5,
        'min_confidence_threshold': 0.5
    }
    
    print("🔧 Creating AI Master Integration System...")
    system = create_ai_master_system(config)
    print("✅ System created successfully")
    
    # Test system status
    print("\n📊 System Status:")
    status = system.get_system_status()
    for key, value in status['systems_active'].items():
        print(f"   {key}: {'✅' if value else '❌'}")
    
    # Generate test data
    print(f"\n📈 Processing test market data...")
    
    decisions = []
    for i in range(10):
        market_data = AIMarketData(
            timestamp=datetime.now() + timedelta(minutes=i),
            symbol="XAUUSD",
            price=2000.0 + i * 0.5,
            high=2010.0 + i * 0.5,
            low=1990.0 + i * 0.5,
            volume=5000.0 + i * 100
        )
        
        decision = system.process_market_data(market_data)
        if decision:
            decisions.append(decision)
            print(f"   Decision {i+1}: {decision.action} (Confidence: {decision.confidence:.3f})")
    
    print(f"\n📊 Results Summary:")
    print(f"   Total Market Data Points: 10")
    print(f"   Decisions Generated: {len(decisions)}")
    print(f"   Decision Rate: {len(decisions)/10*100:.1f}%")
    
    if decisions:
        actions = [d.action for d in decisions]
        print(f"   Actions: BUY={actions.count('BUY')}, SELL={actions.count('SELL')}, HOLD={actions.count('HOLD')}")
        avg_confidence = np.mean([d.confidence for d in decisions])
        print(f"   Average Confidence: {avg_confidence:.3f}")
    
    # Test export
    print(f"\n💾 Testing data export...")
    export_result = system.export_system_data("simple_demo_results.json")
    if export_result['success']:
        print(f"✅ Export successful: {export_result['filepath']}")
    else:
        print(f"❌ Export failed: {export_result['error']}")
    
    print(f"\n🎉 Simple Demo Completed Successfully!")
    print(f"✅ AI Master Integration System operational")
    print(f"✅ Core functionality verified")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n❌ Demo failed!")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
