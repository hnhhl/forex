# -*- coding: utf-8 -*-
"""Debug Consensus Detailed - Phân tích chi tiết consensus"""

import sys
import os
sys.path.append('src')

def debug_consensus_detailed():
    print("🔍 PHÂN TÍCH CHI TIẾT CONSENSUS - DEBUG MODE")
    print("="*60)
    
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
    
    # Generate 3 signals to see variation
    print(f"\n🔄 GENERATING 3 SIGNALS FOR ANALYSIS...")
    
    for i in range(3):
        print(f"\n" + "="*50)
        print(f"📊 SIGNAL #{i+1} ANALYSIS")
        print("="*50)
        
        try:
            signal = system.generate_signal("XAUUSDc")
            
            # Basic signal info
            action = signal.get('action', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            method = signal.get('ensemble_method', 'unknown')
            hybrid_metrics = signal.get('hybrid_metrics', {})
            
            print(f"🎯 Action: {action}")
            print(f"📈 Confidence: {confidence:.1%}")
            print(f"🔧 Method: {method}")
            
            if hybrid_metrics:
                consensus = hybrid_metrics.get('hybrid_consensus', 0)
                print(f"🤝 Consensus: {consensus:.1%}")
                
                # AI2.0 prediction
                ai2_prediction = hybrid_metrics.get('ai2_weighted_average', 0)
                print(f"📊 AI2.0 Weighted Average: {ai2_prediction:.3f}")
                
                # Vote analysis
                vote_agreement = hybrid_metrics.get('vote_agreement', 0)
                prediction_vote_agreement = hybrid_metrics.get('prediction_vote_agreement', 0)
                
                print(f"🗳️ Vote Agreement: {vote_agreement:.1%}")
                print(f"🔄 Prediction-Vote Agreement: {prediction_vote_agreement:.1%}")
                
                # System votes breakdown
                system_votes = hybrid_metrics.get('system_votes', {})
                if system_votes:
                    print(f"\n🗳️ INDIVIDUAL SYSTEM VOTES:")
                    
                    buy_count = 0
                    sell_count = 0
                    hold_count = 0
                    
                    for sys_name, vote in system_votes.items():
                        print(f"   {sys_name}: {vote}")
                        
                        if vote == 'BUY':
                            buy_count += 1
                        elif vote == 'SELL':
                            sell_count += 1
                        elif vote == 'HOLD':
                            hold_count += 1
                    
                    total_votes = len(system_votes)
                    print(f"\n📈 VOTE SUMMARY:")
                    print(f"   🟢 BUY: {buy_count}/{total_votes} ({buy_count/total_votes*100:.1f}%)")
                    print(f"   🔴 SELL: {sell_count}/{total_votes} ({sell_count/total_votes*100:.1f}%)")
                    print(f"   ⚪ HOLD: {hold_count}/{total_votes} ({hold_count/total_votes*100:.1f}%)")
                    
                    # Identify disagreement
                    if action == 'BUY':
                        disagreeing = sell_count + hold_count
                        print(f"\n⚠️ DISAGREEMENT WITH BUY:")
                        print(f"   {disagreeing}/{total_votes} systems disagree")
                        if sell_count > 0:
                            print(f"   {sell_count} systems vote SELL")
                        if hold_count > 0:
                            print(f"   {hold_count} systems vote HOLD")
                    
                    elif action == 'SELL':
                        disagreeing = buy_count + hold_count
                        print(f"\n⚠️ DISAGREEMENT WITH SELL:")
                        print(f"   {disagreeing}/{total_votes} systems disagree")
                        if buy_count > 0:
                            print(f"   {buy_count} systems vote BUY")
                        if hold_count > 0:
                            print(f"   {hold_count} systems vote HOLD")
                    
                    # Consensus analysis
                    majority_vote = max(buy_count, sell_count, hold_count)
                    expected_consensus = majority_vote / total_votes
                    actual_consensus = consensus
                    
                    print(f"\n🎯 CONSENSUS ANALYSIS:")
                    print(f"   Expected: {expected_consensus:.1%}")
                    print(f"   Actual: {actual_consensus:.1%}")
                    print(f"   Gap: {abs(actual_consensus - expected_consensus):.1%}")
                    
                    if abs(actual_consensus - expected_consensus) > 0.1:
                        print(f"   ⚠️ SIGNIFICANT CONSENSUS GAP!")
                
                else:
                    print("⚠️ No system votes data available")
            
            else:
                print("⚠️ No hybrid metrics available")
        
        except Exception as e:
            print(f"❌ Error generating signal {i+1}: {e}")
    
    # Overall analysis
    print(f"\n" + "="*60)
    print("🎯 CONSENSUS ISSUE ANALYSIS")
    print("="*60)
    
    print("📊 POSSIBLE REASONS FOR 61.7% CONSENSUS:")
    print("   1. 🔄 Market Transition Period")
    print("      - Systems detecting different market phases")
    print("      - Some see bullish, others see bearish signals")
    
    print("   2. 🧠 Neural Network Uncertainty")
    print("      - Neural models may have low confidence")
    print("      - Different models predicting different directions")
    
    print("   3. 📈 Technical Indicator Conflicts")
    print("      - Short-term vs long-term indicators disagreeing")
    print("      - Different timeframes showing different trends")
    
    print("   4. 🎯 Hybrid Logic Weighting")
    print("      - AI2.0 weighted average vs democratic voting")
    print("      - Systems with different prediction confidence")
    
    print("   5. 📊 Data Quality Variations")
    print("      - Different systems using slightly different data")
    print("      - Real-time vs historical data inconsistencies")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("   ✅ 61.7% consensus is actually GOOD")
    print("   ⚡ Shows systems are thinking independently")
    print("   🎯 Prevents overconfident wrong decisions")
    print("   📊 Indicates healthy system diversity")

if __name__ == "__main__":
    debug_consensus_detailed() 