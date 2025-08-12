# -*- coding: utf-8 -*-
"""Fix Votes Display - Sửa lỗi hiển thị individual system votes"""

import sys
import os
sys.path.append('src')

def fix_votes_display():
    print("🔧 FIX VOTES DISPLAY - HIỂN THỊ ĐÚNG INDIVIDUAL SYSTEM VOTES")
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
    
    # Generate signal with correct votes extraction
    print(f"\n🔄 GENERATING SIGNAL WITH CORRECT VOTES EXTRACTION...")
    signal = system.generate_signal("XAUUSDc")
    
    # Extract basic info
    action = signal.get('action', 'UNKNOWN')
    confidence = signal.get('confidence', 0)
    method = signal.get('ensemble_method', 'unknown')
    hybrid_metrics = signal.get('hybrid_metrics', {})
    
    print(f"\n📊 SIGNAL OVERVIEW:")
    print(f"   🎯 Action: {action}")
    print(f"   📈 Confidence: {confidence:.1%}")
    print(f"   🔧 Method: {method}")
    
    if hybrid_metrics:
        consensus = hybrid_metrics.get('hybrid_consensus', 0)
        print(f"   🤝 Consensus: {consensus:.1%}")
    
    # CORRECT VOTES EXTRACTION
    voting_results = signal.get('voting_results', {})
    
    if voting_results:
        print(f"\n🗳️ CORRECT VOTING RESULTS EXTRACTION:")
        
        # Extract vote counts
        buy_votes = voting_results.get('buy_votes', 0)
        sell_votes = voting_results.get('sell_votes', 0)
        hold_votes = voting_results.get('hold_votes', 0)
        votes_list = voting_results.get('votes', [])
        
        total_votes = buy_votes + sell_votes + hold_votes
        
        print(f"   📊 VOTE COUNTS:")
        print(f"      🟢 BUY: {buy_votes}/{total_votes} ({buy_votes/total_votes*100:.1f}%)")
        print(f"      🔴 SELL: {sell_votes}/{total_votes} ({sell_votes/total_votes*100:.1f}%)")
        print(f"      ⚪ HOLD: {hold_votes}/{total_votes} ({hold_votes/total_votes*100:.1f}%)")
        
        print(f"\n🗳️ INDIVIDUAL VOTES LIST:")
        print(f"   Votes: {votes_list}")
        print(f"   Total Systems: {len(votes_list)}")
        
        # Map votes to systems (we know there are 8 systems)
        system_names = [
            'DataQualityMonitor',
            'LatencyOptimizer', 
            'MT5ConnectionManager',
            'NeuralNetworkSystem',
            'AIPhaseSystem',
            'AI2AdvancedTechnologiesSystem',
            'AdvancedAIEnsembleSystem',
            'RealTimeMT5DataSystem'
        ]
        
        print(f"\n🏛️ ESTIMATED SYSTEM VOTES MAPPING:")
        for i, vote in enumerate(votes_list):
            if i < len(system_names):
                system_name = system_names[i]
                print(f"   {system_name}: {vote}")
        
        # Disagreement analysis
        if action == 'BUY':
            disagreeing = sell_votes + hold_votes
            print(f"\n⚠️ DISAGREEMENT WITH BUY:")
            print(f"   {disagreeing}/{total_votes} systems disagree ({disagreeing/total_votes*100:.1f}%)")
            if sell_votes > 0:
                print(f"   🔴 {sell_votes} systems vote SELL")
            if hold_votes > 0:
                print(f"   ⚪ {hold_votes} systems vote HOLD")
        
        elif action == 'SELL':
            disagreeing = buy_votes + hold_votes
            print(f"\n⚠️ DISAGREEMENT WITH SELL:")
            print(f"   {disagreeing}/{total_votes} systems disagree ({disagreeing/total_votes*100:.1f}%)")
            if buy_votes > 0:
                print(f"   🟢 {buy_votes} systems vote BUY")
            if hold_votes > 0:
                print(f"   ⚪ {hold_votes} systems vote HOLD")
        
        # Consensus analysis
        majority_votes = max(buy_votes, sell_votes, hold_votes)
        expected_consensus = majority_votes / total_votes
        actual_consensus = consensus
        
        print(f"\n🎯 CONSENSUS ANALYSIS:")
        print(f"   📊 Expected Consensus: {expected_consensus:.1%}")
        print(f"   📊 Actual Consensus: {actual_consensus:.1%}")
        print(f"   📊 Consensus Gap: {abs(actual_consensus - expected_consensus):.1%}")
        
        # Detailed breakdown
        print(f"\n🔍 DETAILED BREAKDOWN:")
        print(f"   🎯 Winning Vote: {action}")
        print(f"   📊 Winning Count: {majority_votes}/{total_votes}")
        print(f"   📈 Democratic Support: {expected_consensus:.1%}")
        print(f"   🤝 Hybrid Consensus: {actual_consensus:.1%}")
        
        # Explanation of consensus calculation
        if hybrid_metrics:
            print(f"\n💡 CONSENSUS CALCULATION BREAKDOWN:")
            
            consensus_ratio = hybrid_metrics.get('consensus_ratio', 0)
            agreement = hybrid_metrics.get('agreement', 0)
            
            print(f"   📊 Consensus Ratio: {consensus_ratio:.1%}")
            print(f"   🤝 Agreement Score: {agreement:.1%}")
            print(f"   🔄 Hybrid Formula: (consensus_ratio * 0.7) + (agreement * 0.3)")
            print(f"   🎯 Result: ({consensus_ratio:.3f} * 0.7) + ({agreement:.3f} * 0.3) = {actual_consensus:.3f}")
        
    else:
        print("⚠️ No voting results data available")
    
    # Analysis of why consensus is not 100%
    print(f"\n" + "="*70)
    print("🎯 TẠI SAO CONSENSUS KHÔNG PHẢI 100%?")
    print("="*70)
    
    if voting_results:
        total_systems = len(votes_list)
        disagreeing_systems = total_systems - max(buy_votes, sell_votes, hold_votes)
        
        print(f"📊 THỐNG KÊ DISAGREEMENT:")
        print(f"   🏛️ Total Systems: {total_systems}")
        print(f"   ✅ Agreeing Systems: {max(buy_votes, sell_votes, hold_votes)}")
        print(f"   ❌ Disagreeing Systems: {disagreeing_systems}")
        print(f"   📈 Agreement Rate: {max(buy_votes, sell_votes, hold_votes)/total_systems:.1%}")
        
        print(f"\n💡 LÝ DO DISAGREEMENT:")
        print(f"   1. 🧠 Neural Networks: Có thể thấy uncertainty")
        print(f"   2. 📊 Data Quality: Systems đánh giá data khác nhau")
        print(f"   3. ⏰ Latency: Timing differences giữa systems")
        print(f"   4. 🔗 MT5 Connection: Network conditions khác nhau")
        print(f"   5. 🚀 AI Phases: Different market phase detection")
        print(f"   6. 🔥 AI2 Technologies: Advanced algorithms có view khác")
        print(f"   7. 🏆 Ensemble: Multiple models trong ensemble")
        print(f"   8. 📡 Real-time Data: Slight data variations")
        
        print(f"\n✅ KẾT LUẬN:")
        if consensus >= 0.7:
            print(f"   🎉 CONSENSUS {consensus:.1%} LÀ TỐT!")
            print(f"   ✅ Hệ thống hoạt động healthy với diversity")
            print(f"   🎯 Prevents overconfident wrong decisions")
        elif consensus >= 0.5:
            print(f"   ⚡ CONSENSUS {consensus:.1%} LÀ MODERATE")
            print(f"   📊 Cần monitor thêm nhưng vẫn acceptable")
        else:
            print(f"   ⚠️ CONSENSUS {consensus:.1%} LÀ THẤP")
            print(f"   🔧 Có thể cần điều chỉnh system parameters")

if __name__ == "__main__":
    fix_votes_display() 