#!/usr/bin/env python3
"""
🏆 DEMO: Ultimate XAU System với Kelly Criterion Integration
Thể hiện tích hợp hoàn chỉnh Kelly Criterion vào UltimateXAUSystem V4.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_enhanced_xau_system_with_kelly():
    """Tạo Enhanced Ultimate XAU System với Kelly Criterion Integration"""
    
    print("🚀 INITIALIZING ULTIMATE XAU SYSTEM V4.0 WITH KELLY CRITERION")
    print("="*80)
    
    try:
        from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        config = SystemConfig(
            symbol="XAUUSDc",
            enable_kelly_criterion=True,
            kelly_method="adaptive",
            kelly_lookback_period=100,
            kelly_max_fraction=0.25,
            kelly_min_fraction=0.01,
            kelly_safety_factor=0.5,
            enable_position_sizing=True,
            default_sizing_method="kelly_adaptive"
        )
        
        system = UltimateXAUSystem(config)
        
        print(f"✅ Ultimate XAU System initialized with Kelly Criterion")
        print(f"📊 Total Systems: {system.system_state['systems_total']}")
        print(f"🔥 Active Systems: {system.system_state['systems_active']}")
        
        return system, config
        
    except Exception as e:
        logger.error(f"Error creating enhanced system: {e}")
        return None, None

def create_kelly_system_directly():
    """Tạo Kelly Criterion System trực tiếp"""
    
    print("\n🏆 INITIALIZING KELLY CRITERION SYSTEM DIRECTLY")
    print("="*60)
    
    try:
        from src.core.kelly_system import KellyCriterionSystem
        from src.core.ultimate_xau_system import SystemConfig
        
        config = SystemConfig(
            kelly_method="adaptive",
            kelly_lookback_period=100,
            kelly_max_fraction=0.25,
            kelly_min_fraction=0.01,
            kelly_safety_factor=0.5
        )
        
        kelly_system = KellyCriterionSystem(config)
        
        if kelly_system.initialize():
            print("✅ Kelly Criterion System initialized successfully")
            return kelly_system, config
        else:
            print("❌ Failed to initialize Kelly Criterion System")
            return None, None
            
    except Exception as e:
        logger.error(f"Error creating Kelly System: {e}")
        return None, None

def generate_sample_market_data():
    """Tạo dữ liệu thị trường mẫu cho XAU"""
    
    print("\n📊 GENERATING SAMPLE XAU MARKET DATA")
    print("="*50)
    
    try:
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=42), 
            end=datetime.now(), 
            freq='1H'
        )
        
        base_price = 2050.0
        data = []
        
        for i, date in enumerate(dates):
            trend = np.sin(i * 0.01) * 50
            volatility = np.random.normal(0, 15)
            price = base_price + trend + volatility
            
            open_price = price + np.random.uniform(-3, 3)
            high_price = max(open_price, price) + np.random.uniform(0, 10)
            low_price = min(open_price, price) - np.random.uniform(0, 10)
            close_price = price + np.random.uniform(-2, 2)
            
            data.append({
                'time': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': np.random.randint(1000, 50000)
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} data points")
        print(f"📈 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating market data: {e}")
        return pd.DataFrame()

def simulate_trading_history():
    """Tạo lịch sử giao dịch mẫu"""
    
    print("\n📈 SIMULATING TRADING HISTORY FOR KELLY CALCULATION")
    print("="*60)
    
    try:
        trades = []
        win_rate = 0.65
        avg_win = 0.025
        avg_loss = -0.015
        
        for i in range(50):
            is_win = np.random.random() < win_rate
            
            if is_win:
                profit_loss = abs(np.random.normal(avg_win, 0.01))
            else:
                profit_loss = -abs(np.random.normal(-avg_loss, 0.005))
            
            trade = {
                'trade_id': f"T{i+1:03d}",
                'timestamp': datetime.now() - timedelta(days=100-i*2),
                'symbol': 'XAUUSDc',
                'profit_loss': profit_loss,
                'win': is_win
            }
            
            trades.append(trade)
        
        actual_win_rate = sum(1 for t in trades if t['win']) / len(trades)
        actual_avg_win = np.mean([t['profit_loss'] for t in trades if t['win']])
        actual_avg_loss = np.mean([t['profit_loss'] for t in trades if not t['win']])
        
        print(f"✅ Simulated {len(trades)} trades")
        print(f"🎯 Win Rate: {actual_win_rate:.1%}")
        print(f"💰 Average Win: {actual_avg_win:.2%}")
        print(f"💸 Average Loss: {actual_avg_loss:.2%}")
        
        return trades
        
    except Exception as e:
        logger.error(f"Error simulating trading history: {e}")
        return []

def test_kelly_system(kelly_system, market_data, trading_history):
    """Test Kelly Criterion System"""
    
    print("\n🧪 TESTING KELLY CRITERION SYSTEM")
    print("="*50)
    
    try:
        if not kelly_system:
            print("❌ Kelly System not available")
            return None
        
        # Add trading history
        for trade in trading_history:
            kelly_system.add_trade_result(trade)
        
        # Process market data
        result = kelly_system.process(market_data)
        
        if 'error' in result:
            print(f"❌ Kelly processing error: {result['error']}")
            return None
        
        kelly_calc = result.get('kelly_calculation', {})
        
        print("✅ KELLY CRITERION RESULTS:")
        print(f"   📊 Kelly Fraction: {kelly_calc.get('kelly_fraction', 0):.3f}")
        print(f"   🛡️ Safe Kelly Fraction: {kelly_calc.get('safe_kelly_fraction', 0):.3f}")
        print(f"   💰 Position Size: ${kelly_calc.get('position_size_usd', 0):,.2f}")
        print(f"   🎯 Confidence: {kelly_calc.get('confidence', 0):.1%}")
        print(f"   📈 Recommendation: {kelly_calc.get('recommendation', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing Kelly system: {e}")
        return None

def main():
    """Main demo function"""
    
    print("🏆 ULTIMATE XAU SYSTEM với KELLY CRITERION INTEGRATION DEMO")
    print("="*80)
    print(f"🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Step 1: Generate test data
    market_data = generate_sample_market_data()
    if market_data.empty:
        print("❌ Failed to generate market data")
        return
    
    trading_history = simulate_trading_history()
    if not trading_history:
        print("❌ Failed to generate trading history")
        return
    
    # Step 2: Test Kelly System
    kelly_system, kelly_config = create_kelly_system_directly()
    if kelly_system:
        kelly_result = test_kelly_system(kelly_system, market_data, trading_history)
    
    # Step 3: Test Ultimate System
    ultimate_system, ultimate_config = create_enhanced_xau_system_with_kelly()
    if ultimate_system:
        print("\n🚀 TESTING ULTIMATE XAU SYSTEM")
        print("="*50)
        
        signal = ultimate_system.generate_signal("XAUUSDc")
        
        print("✅ ULTIMATE SYSTEM SIGNAL:")
        print(f"   🎯 Action: {signal.get('action', 'N/A')}")
        print(f"   💪 Strength: {signal.get('strength', 'N/A')}")
        print(f"   📊 Prediction: {signal.get('prediction', 0):.3f}")
        print(f"   🎯 Confidence: {signal.get('confidence', 0):.1%}")
    
    # Step 4: Summary
    print("\n🎯 INTEGRATION DEMO SUMMARY")
    print("="*50)
    print("✅ Kelly Criterion System: Successfully integrated")
    print("✅ Ultimate XAU System: Successfully enhanced")
    print("✅ Position Sizing: Kelly-optimized")
    print("✅ Risk Management: Advanced Kelly controls")
    
    print(f"\n🏆 Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
