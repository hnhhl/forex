"""
Test Two Specialists - RSI & MACD
================================================================================
Test script để kiểm tra 2 specialists đầu tiên hoạt động
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
    
    base_price = 2000.0
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='h')
    
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, periods)
    trend = np.sin(np.arange(periods) * 0.1) * 0.001
    returns += trend
    
    prices = base_price * np.cumprod(1 + returns)
    spread = 0.5
    
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

def test_two_specialists():
    """Test RSI and MACD Specialists"""
    
    print("🧪 TESTING TWO SPECIALISTS - RSI & MACD")
    print("=" * 80)
    
    try:
        # Import specialists
        from core.specialists.rsi_specialist import RSISpecialist, create_rsi_specialist
        from core.specialists.macd_specialist import MACDSpecialist, create_macd_specialist
        
        print("✅ Successfully imported both specialists")
        
        # Create test data
        print("\n📊 Creating sample XAU data...")
        data = create_sample_data(100)
        current_price = data['close'].iloc[-1]
        
        print(f"   Data points: {len(data)}")
        print(f"   Current price: ${current_price:.2f}")
        
        # Create specialists
        print("\n🔧 Creating specialists...")
        rsi_specialist = create_rsi_specialist(period=14, oversold=30, overbought=70)
        macd_specialist = create_macd_specialist(fast=12, slow=26, signal=9)
        
        print(f"   RSI Specialist: {rsi_specialist.name} ({rsi_specialist.category})")
        print(f"   MACD Specialist: {macd_specialist.name} ({macd_specialist.category})")
        
        # Test both specialists
        print("\n🗳️ Testing vote generation...")
        
        # RSI Analysis
        rsi_vote = rsi_specialist.analyze(data, current_price)
        print(f"\n📈 RSI Analysis:")
        print(f"   Vote: {rsi_vote.vote}")
        print(f"   Confidence: {rsi_vote.confidence:.3f}")
        print(f"   Reasoning: {rsi_vote.reasoning}")
        
        # MACD Analysis
        macd_vote = macd_specialist.analyze(data, current_price)
        print(f"\n📉 MACD Analysis:")
        print(f"   Vote: {macd_vote.vote}")
        print(f"   Confidence: {macd_vote.confidence:.3f}")
        print(f"   Reasoning: {macd_vote.reasoning}")
        
        # Compare votes
        print(f"\n🤝 Vote Comparison:")
        print(f"   RSI Vote: {rsi_vote.vote} (confidence: {rsi_vote.confidence:.3f})")
        print(f"   MACD Vote: {macd_vote.vote} (confidence: {macd_vote.confidence:.3f})")
        
        if rsi_vote.vote == macd_vote.vote:
            print(f"   ✅ CONSENSUS: Both specialists agree on {rsi_vote.vote}")
            consensus_confidence = (rsi_vote.confidence + macd_vote.confidence) / 2
            print(f"   Combined confidence: {consensus_confidence:.3f}")
        else:
            print(f"   ⚠️ DISAGREEMENT: RSI says {rsi_vote.vote}, MACD says {macd_vote.vote}")
            print(f"   Need more specialists for consensus")
        
        # Test with different market conditions
        print(f"\n🔄 Testing different market conditions...")
        
        # Trending up market
        trending_data = data.copy()
        trending_data.loc[:, 'close'] = trending_data['close'] * np.linspace(1.0, 1.05, len(trending_data))
        
        rsi_trending = rsi_specialist.analyze(trending_data, trending_data['close'].iloc[-1])
        macd_trending = macd_specialist.analyze(trending_data, trending_data['close'].iloc[-1])
        
        print(f"   Trending Market:")
        print(f"     RSI: {rsi_trending.vote} ({rsi_trending.confidence:.3f})")
        print(f"     MACD: {macd_trending.vote} ({macd_trending.confidence:.3f})")
        
        # Volatile market
        volatile_data = data.copy()
        volatile_data.loc[:, 'close'] *= (1 + np.random.normal(0, 0.01, len(volatile_data)))
        
        rsi_volatile = rsi_specialist.analyze(volatile_data, volatile_data['close'].iloc[-1])
        macd_volatile = macd_specialist.analyze(volatile_data, volatile_data['close'].iloc[-1])
        
        print(f"   Volatile Market:")
        print(f"     RSI: {rsi_volatile.vote} ({rsi_volatile.confidence:.3f})")
        print(f"     MACD: {macd_volatile.vote} ({macd_volatile.confidence:.3f})")
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        
        rsi_performance = rsi_specialist.get_performance_summary()
        macd_performance = macd_specialist.get_performance_summary()
        
        print(f"   RSI Specialist:")
        print(f"     Total votes: {rsi_performance['total_votes']}")
        print(f"     Recent accuracy: {rsi_performance['recent_accuracy']:.1%}")
        print(f"     Weight: {rsi_performance['weight']}")
        
        print(f"   MACD Specialist:")
        print(f"     Total votes: {macd_performance['total_votes']}")
        print(f"     Recent accuracy: {macd_performance['recent_accuracy']:.1%}")
        print(f"     Weight: {macd_performance['weight']}")
        
        print("\n✅ TWO SPECIALISTS TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run two specialists test"""
    
    print("🚀 MULTI-PERSPECTIVE ENSEMBLE - TWO SPECIALISTS TEST")
    print("=" * 80)
    
    success = test_two_specialists()
    
    if success:
        print("\n🎉 MILESTONE 2 PROGRESS: 2/5 Specialists Working!")
        print("✅ RSI & MACD Specialists are operational")
        
        # Update progress tracking
        print("\n📋 PROGRESS UPDATE:")
        print("☑ RSI_Specialist.py created & tested ✅")
        print("☑ MACD_Specialist.py created & tested ✅")
        print("□ Fibonacci_Specialist.py created & tested")
        print("□ News_Sentiment_Specialist.py structure ready")
        print("□ Social_Media_Specialist.py structure ready")
        print("...")
        print(f"Phase 1 Progress: 2/18 specialists completed (11.1%)")
        print(f"Week 1 Progress: 2/8 tasks completed (25.0%)")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Create Fibonacci_Specialist.py")
        print("2. Create basic sentiment specialists structure")
        print("3. Create pattern specialists")
        print("4. Build democratic voting engine")
        
    else:
        print("\n❌ TEST FAILED - Need to fix issues before proceeding")
    
    return success

if __name__ == "__main__":
    main() 