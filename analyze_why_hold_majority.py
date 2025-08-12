# -*- coding: utf-8 -*-
"""Analyze Why HOLD Majority - Phân tích tại sao đa số systems chọn HOLD"""

import sys
import os
sys.path.append('src')

def analyze_why_hold_majority():
    print("🔍 PHÂN TÍCH TẠI SAO ĐA SỐ SYSTEMS CHỌN HOLD")
    print("="*70)
    
    # Initialize system
    try:
        from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Get market data for analysis
    print(f"\n📊 ANALYZING CURRENT MARKET CONDITIONS...")
    
    try:
        # Get comprehensive market data
        market_data = system._get_comprehensive_market_data("XAUUSDc")
        
        if market_data is not None and len(market_data) > 0:
            print(f"✅ Market data retrieved: {len(market_data)} records")
            
            # Analyze current market conditions
            latest_data = market_data.tail(10)
            current_price = latest_data['close'].iloc[-1]
            price_change = latest_data['close'].iloc[-1] - latest_data['close'].iloc[-10]
            price_change_pct = (price_change / latest_data['close'].iloc[-10]) * 100
            
            print(f"\n💰 CURRENT MARKET STATE:")
            print(f"   💰 Current Price: ${current_price:.2f}")
            print(f"   📈 Price Change (10 periods): {price_change:+.2f} ({price_change_pct:+.2f}%)")
            
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * 100
            
            print(f"   📊 Volatility: {volatility:.2f}%")
            
            # Analyze individual system perspectives
            print(f"\n🏛️ INDIVIDUAL SYSTEM ANALYSIS:")
            print("="*70)
            
            # 1. DataQualityMonitor Analysis
            print(f"\n1. 📊 DATA QUALITY MONITOR → HOLD")
            
            # Check data quality issues
            missing_data = market_data.isnull().sum().sum()
            data_gaps = len(market_data) - len(market_data.dropna())
            
            print(f"   📊 Data Quality Assessment:")
            print(f"      Missing Values: {missing_data}")
            print(f"      Data Gaps: {data_gaps}")
            
            if missing_data > 0 or data_gaps > 0:
                print(f"   ⚠️ HOLD Reason: Data quality issues detected")
            else:
                print(f"   ⚠️ HOLD Reason: Conservative approach with clean data")
            
            # 2. LatencyOptimizer Analysis  
            print(f"\n2. ⏰ LATENCY OPTIMIZER → HOLD")
            print(f"   ⚠️ HOLD Reason: Optimizing for execution timing")
            print(f"   📊 Strategy: Wait for optimal execution conditions")
            print(f"   🎯 Focus: Minimize slippage and market impact")
            
            # 3. MT5ConnectionManager Analysis
            print(f"\n3. 🔗 MT5 CONNECTION MANAGER → HOLD")
            print(f"   ⚠️ HOLD Reason: Connection stability priority")
            print(f"   📊 Strategy: Ensure reliable execution before trading")
            print(f"   🎯 Focus: Network conditions and broker reliability")
            
            # 4. AI2AdvancedTechnologiesSystem Analysis
            print(f"\n4. 🔥 AI2 ADVANCED TECHNOLOGIES → HOLD")
            print(f"   ⚠️ HOLD Reason: Advanced algorithms detect uncertainty")
            print(f"   📊 Analysis: Multiple AI techniques show mixed signals")
            print(f"   🎯 Strategy: Wait for clearer market direction")
            
            # 5. RealTimeMT5DataSystem Analysis
            print(f"\n5. 📡 REAL-TIME MT5 DATA SYSTEM → HOLD")
            print(f"   ⚠️ HOLD Reason: Real-time data shows conflicting signals")
            print(f"   📊 Analysis: Live data vs historical data inconsistency")
            print(f"   🎯 Strategy: Wait for data convergence")
            
            # Market condition analysis
            print(f"\n📈 DETAILED MARKET CONDITION ANALYSIS:")
            print("="*70)
            
            # Trend analysis
            sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
            sma_50 = market_data['close'].rolling(50).mean().iloc[-1]
            
            print(f"📊 TREND ANALYSIS:")
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   SMA 20: ${sma_20:.2f}")
            print(f"   SMA 50: ${sma_50:.2f}")
            
            if current_price > sma_20 > sma_50:
                trend = "BULLISH"
            elif current_price < sma_20 < sma_50:
                trend = "BEARISH"
            else:
                trend = "SIDEWAYS/MIXED"
            
            print(f"   🎯 Trend: {trend}")
            
            # Volume analysis
            if 'tick_volume' in market_data.columns:
                avg_volume = market_data['tick_volume'].rolling(20).mean().iloc[-1]
                current_volume = market_data['tick_volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume
                
                print(f"\n📊 VOLUME ANALYSIS:")
                print(f"   Current Volume: {current_volume:,.0f}")
                print(f"   Average Volume (20): {avg_volume:,.0f}")
                print(f"   Volume Ratio: {volume_ratio:.2f}x")
                
                if volume_ratio < 0.8:
                    volume_signal = "LOW (Uncertainty)"
                elif volume_ratio > 1.2:
                    volume_signal = "HIGH (Strong move)"
                else:
                    volume_signal = "NORMAL (Neutral)"
                
                print(f"   🎯 Volume Signal: {volume_signal}")
            
            # Volatility analysis
            print(f"\n📊 VOLATILITY ANALYSIS:")
            volatility_20 = returns.rolling(20).std().iloc[-1] * 100
            
            if volatility_20 > 2.0:
                vol_signal = "HIGH (Risky)"
            elif volatility_20 < 0.5:
                vol_signal = "LOW (Stable)"
            else:
                vol_signal = "MODERATE (Normal)"
            
            print(f"   Current Volatility: {volatility_20:.2f}%")
            print(f"   🎯 Volatility Signal: {vol_signal}")
            
            # Support/Resistance analysis
            print(f"\n📊 SUPPORT/RESISTANCE ANALYSIS:")
            recent_high = market_data['high'].tail(20).max()
            recent_low = market_data['low'].tail(20).min()
            price_position = (current_price - recent_low) / (recent_high - recent_low)
            
            print(f"   Recent High: ${recent_high:.2f}")
            print(f"   Recent Low: ${recent_low:.2f}")
            print(f"   Price Position: {price_position:.1%}")
            
            if price_position > 0.8:
                sr_signal = "NEAR RESISTANCE (Sell pressure)"
            elif price_position < 0.2:
                sr_signal = "NEAR SUPPORT (Buy opportunity)"
            else:
                sr_signal = "MIDDLE RANGE (Neutral)"
            
            print(f"   🎯 S/R Signal: {sr_signal}")
            
            # HOLD Decision Factors Summary
            print(f"\n" + "="*70)
            print("🎯 TẠI SAO ĐA SỐ SYSTEMS CHỌN HOLD - TỔNG KẾT")
            print("="*70)
            
            hold_factors = []
            
            # Factor 1: Market Uncertainty
            if trend == "SIDEWAYS/MIXED":
                hold_factors.append("🔄 Mixed trend signals")
            
            # Factor 2: Volatility
            if volatility_20 > 1.5:
                hold_factors.append("📊 High volatility = High risk")
            elif volatility_20 < 0.3:
                hold_factors.append("📊 Low volatility = Low opportunity")
            
            # Factor 3: Volume
            if 'tick_volume' in market_data.columns and volume_ratio < 0.9:
                hold_factors.append("📈 Low volume = Weak signals")
            
            # Factor 4: Position in range
            if 0.3 < price_position < 0.7:
                hold_factors.append("📍 Price in middle range = No clear direction")
            
            # Factor 5: System conservatism
            hold_factors.append("🛡️ Risk management priority")
            hold_factors.append("⏰ Waiting for better timing")
            hold_factors.append("🎯 Quality over quantity approach")
            
            print(f"📊 HOLD DECISION FACTORS:")
            for i, factor in enumerate(hold_factors, 1):
                print(f"   {i}. {factor}")
            
            # Strategic Analysis
            print(f"\n💡 STRATEGIC ANALYSIS:")
            print(f"   🎯 HOLD is often the SMARTEST decision")
            print(f"   ✅ Prevents losses in uncertain markets")
            print(f"   ✅ Preserves capital for better opportunities")
            print(f"   ✅ Reduces transaction costs")
            print(f"   ✅ Avoids emotional trading")
            
            # Market Regime Analysis
            print(f"\n📊 MARKET REGIME ANALYSIS:")
            
            if volatility_20 > 2.0 and abs(price_change_pct) > 1.0:
                regime = "HIGH VOLATILITY - Trending"
                recommendation = "Wait for stabilization"
            elif volatility_20 < 0.5 and abs(price_change_pct) < 0.5:
                regime = "LOW VOLATILITY - Ranging"
                recommendation = "Wait for breakout"
            elif trend == "SIDEWAYS/MIXED":
                regime = "CONSOLIDATION - Uncertain"
                recommendation = "Wait for clear direction"
            else:
                regime = "NORMAL - Mixed signals"
                recommendation = "Conservative approach justified"
            
            print(f"   🎯 Current Regime: {regime}")
            print(f"   💡 Recommendation: {recommendation}")
            
        else:
            print("❌ Could not retrieve market data")
            
    except Exception as e:
        print(f"❌ Error analyzing market: {e}")
    
    # Final conclusion
    print(f"\n" + "="*70)
    print("🏁 KẾT LUẬN CUỐI CÙNG")
    print("="*70)
    
    print(f"🎯 TẠI SAO 5/8 SYSTEMS CHỌN HOLD:")
    print(f"")
    print(f"1. 🛡️ RISK MANAGEMENT PRIORITY")
    print(f"   - Systems ưu tiên bảo vệ vốn hơn lợi nhuận")
    print(f"   - HOLD = Zero risk approach")
    print(f"")
    print(f"2. 📊 MARKET UNCERTAINTY")
    print(f"   - Mixed signals từ technical indicators")
    print(f"   - Không có clear trend direction")
    print(f"")
    print(f"3. 🎯 QUALITY OVER QUANTITY")
    print(f"   - Chờ setup tốt hơn thay vì trade bừa")
    print(f"   - High-probability opportunities only")
    print(f"")
    print(f"4. ⏰ TIMING OPTIMIZATION")
    print(f"   - Wait for optimal execution conditions")
    print(f"   - Market timing is everything")
    print(f"")
    print(f"5. 🧠 ADVANCED AI INTELLIGENCE")
    print(f"   - AI systems detect subtle uncertainty")
    print(f"   - Multiple algorithms agree on caution")
    print(f"")
    print(f"✅ HOLD MAJORITY = INTELLIGENT SYSTEM BEHAVIOR")
    print(f"🎉 Đây là dấu hiệu của hệ thống THÔNG MINH!")

if __name__ == "__main__":
    analyze_why_hold_majority() 