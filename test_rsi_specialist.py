"""
Test RSI Specialist
================================================================================
Test script để kiểm tra RSI Specialist đầu tiên hoạt động
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample XAUUSD data for testing"""
    
    # Generate realistic XAU price data
    base_price = 2000.0
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='H')
    
    # Generate price movements with some trend and volatility
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0, 0.002, periods)  # 0.2% hourly volatility
    
    # Add some trend and mean reversion
    trend = np.sin(np.arange(periods) * 0.1) * 0.001
    returns += trend
    
    # Generate OHLC data
    prices = base_price * np.cumprod(1 + returns)
    
    # Create OHLC with realistic spreads
    spread = 0.5  # $0.50 spread for XAU
    
    opens = prices
    closes = prices * (1 + np.random.normal(0, 0.0005, periods))
    highs = np.maximum(opens, closes) + np.random.exponential(spread/2, periods)
    lows = np.minimum(opens, closes) - np.random.exponential(spread/2, periods)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.randint(100, 1000, periods)
    })
    
    df.set_index('timestamp', inplace=True)
    
    return df

def test_rsi_specialist():
    """Test RSI Specialist functionality"""
    
    print("🧪 TESTING RSI SPECIALIST")
    print("=" * 60)
    
    try:
        # Import RSI Specialist
        from core.specialists.rsi_specialist import RSISpecialist, create_rsi_specialist
        
        print("✅ Successfully imported RSI Specialist")
        
        # Create test data
        print("\n📊 Creating sample XAU data...")
        data = create_sample_data(100)
        current_price = data['close'].iloc[-1]
        
        print(f"   Data points: {len(data)}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Current price: ${current_price:.2f}")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Test RSI Specialist creation
        print("\n🔧 Creating RSI Specialist...")
        rsi_specialist = create_rsi_specialist(period=14, oversold=30, overbought=70)
        
        print(f"   Name: {rsi_specialist.name}")
        print(f"   Category: {rsi_specialist.category}")
        print(f"   Description: {rsi_specialist.description}")
        print(f"   Enabled: {rsi_specialist.enabled}")
        print(f"   Weight: {rsi_specialist.weight}")
        
        # Test RSI calculation
        print("\n📈 Testing RSI calculation...")
        rsi_series = rsi_specialist.calculate_rsi(data['close'])
        current_rsi = rsi_series.iloc[-1]
        
        print(f"   Current RSI: {current_rsi:.2f}")
        print(f"   RSI range: {rsi_series.min():.2f} - {rsi_series.max():.2f}")
        print(f"   RSI mean: {rsi_series.mean():.2f}")
        
        # Test vote generation
        print("\n🗳️ Testing vote generation...")
        vote = rsi_specialist.analyze(data, current_price)
        
        print(f"   Specialist: {vote.specialist_name}")
        print(f"   Vote: {vote.vote}")
        print(f"   Confidence: {vote.confidence:.3f}")
        print(f"   Reasoning: {vote.reasoning}")
        print(f"   Timestamp: {vote.timestamp}")
        
        # Test with different RSI conditions
        print("\n🔄 Testing different RSI conditions...")
        
        # Test oversold condition (RSI < 30)
        oversold_data = data.copy()
        # Simulate price drop to create oversold condition
        oversold_data['close'].iloc[-10:] *= 0.95  # 5% drop
        
        oversold_vote = rsi_specialist.analyze(oversold_data, oversold_data['close'].iloc[-1])
        print(f"   Oversold test - Vote: {oversold_vote.vote}, Confidence: {oversold_vote.confidence:.3f}")
        
        # Test overbought condition (RSI > 70)
        overbought_data = data.copy()
        # Simulate price rise to create overbought condition
        overbought_data['close'].iloc[-10:] *= 1.05  # 5% rise
        
        overbought_vote = rsi_specialist.analyze(overbought_data, overbought_data['close'].iloc[-1])
        print(f"   Overbought test - Vote: {overbought_vote.vote}, Confidence: {overbought_vote.confidence:.3f}")
        
        # Test performance tracking
        print("\n📊 Testing performance tracking...")
        performance = rsi_specialist.get_performance_summary()
        
        print(f"   Total votes: {performance['total_votes']}")
        print(f"   Recent accuracy: {performance['recent_accuracy']:.1%}")
        print(f"   Overall accuracy: {performance['overall_accuracy']:.1%}")
        
        # Test vote history
        print(f"\n📝 Vote history: {len(rsi_specialist.vote_history)} votes")
        if rsi_specialist.vote_history:
            latest_vote = rsi_specialist.vote_history[-1]
            print(f"   Latest vote: {latest_vote['vote']} (confidence: {latest_vote['confidence']:.3f})")
            print(f"   RSI value: {latest_vote['rsi_value']:.2f}")
        
        print("\n✅ RSI SPECIALIST TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the specialist files are created correctly")
        return False
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run RSI Specialist tests"""
    
    print("🚀 MULTI-PERSPECTIVE ENSEMBLE - RSI SPECIALIST TEST")
    print("=" * 80)
    
    success = test_rsi_specialist()
    
    if success:
        print("\n🎉 MILESTONE 1 ACHIEVED: First Specialist Working!")
        print("✅ RSI Specialist is operational and ready for integration")
        
        # Update progress tracking
        print("\n📋 PROGRESS UPDATE:")
        print("□ RSI_Specialist.py created & tested ✅")
        print("□ MACD_Specialist.py created & tested")
        print("□ Fibonacci_Specialist.py created & tested")
        print("...")
        print(f"Phase 1 Progress: 1/18 specialists completed (5.6%)")
        
    else:
        print("\n❌ TEST FAILED - Need to fix issues before proceeding")
    
    return success

if __name__ == "__main__":
    main() 