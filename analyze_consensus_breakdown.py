# -*- coding: utf-8 -*-
"""Analyze Consensus Breakdown - Chi tiết phân tích đồng thuận"""

import sys
import os
sys.path.append('src')

def analyze_consensus_breakdown():
    print("🔍 PHÂN TÍCH CHI TIẾT CONSENSUS BREAKDOWN")
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
    
    print(f"\n🎯 PHÂN TÍCH 8 SYSTEMS TRONG AI3.0:")
    print("="*60)
    
    # List all systems
    systems = system.system_manager.systems
    print(f"📊 Total Systems: {len(systems)}")
    
    for name, sys_obj in systems.items():
        print(f"   - {name}: {type(sys_obj).__name__}")
    
    # Generate detailed signal with breakdown
    print(f"\n🔄 GENERATING DETAILED SIGNAL...")
    signal = system.generate_signal("XAUUSDc")
    
    # Extract detailed metrics
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
        
        # Detailed breakdown
        print(f"\n🔍 DETAILED CONSENSUS BREAKDOWN:")
        
        # AI2.0 Weighted Average
        ai2_prediction = hybrid_metrics.get('ai2_weighted_average', 0)
        print(f"   📊 AI2.0 Weighted Average: {ai2_prediction:.3f}")
        
        # Individual system votes
        system_votes = hybrid_metrics.get('system_votes', {})
        if system_votes:
            print(f"\n🗳️ INDIVIDUAL SYSTEM VOTES:")
            
            buy_votes = []
            sell_votes = []
            hold_votes = []
            
            for sys_name, vote in system_votes.items():
                if vote == 'BUY':
                    buy_votes.append(sys_name)
                elif vote == 'SELL':
                    sell_votes.append(sys_name)
                elif vote == 'HOLD':
                    hold_votes.append(sys_name)
                
                print(f"   {sys_name}: {vote}")
            
            print(f"\n📈 VOTE DISTRIBUTION:")
            print(f"   🟢 BUY votes: {len(buy_votes)} systems")
            if buy_votes:
                for sys in buy_votes:
                    print(f"      - {sys}")
            
            print(f"   🔴 SELL votes: {len(sell_votes)} systems")
            if sell_votes:
                for sys in sell_votes:
                    print(f"      - {sys}")
            
            print(f"   ⚪ HOLD votes: {len(hold_votes)} systems")
            if hold_votes:
                for sys in hold_votes:
                    print(f"      - {sys}")
        
        # Vote agreement analysis
        vote_agreement = hybrid_metrics.get('vote_agreement', 0)
        prediction_vote_agreement = hybrid_metrics.get('prediction_vote_agreement', 0)
        
        print(f"\n🎯 AGREEMENT ANALYSIS:")
        print(f"   🗳️ Vote Agreement: {vote_agreement:.1%}")
        print(f"   🔄 Prediction-Vote Agreement: {prediction_vote_agreement:.1%}")
        
        # Identify disagreement sources
        print(f"\n⚠️ DISAGREEMENT ANALYSIS:")
        
        total_systems = len(system_votes) if system_votes else 8
        sell_count = len(sell_votes) if system_votes else 0
        consensus_expected = sell_count / total_systems if total_systems > 0 else 0
        
        print(f"   📊 Expected Consensus: {consensus_expected:.1%}")
        print(f"   📊 Actual Consensus: {consensus:.1%}")
        print(f"   📊 Consensus Gap: {abs(consensus - consensus_expected):.1%}")
        
        if consensus < 0.8:
            print(f"\n🚨 CONSENSUS ISSUES IDENTIFIED:")
            
            disagreeing_systems = len(buy_votes) + len(hold_votes) if system_votes else 0
            print(f"   ⚠️ {disagreeing_systems} systems disagree with SELL")
            
            if buy_votes:
                print(f"   🟢 BUY disagreement from: {', '.join(buy_votes)}")
            if hold_votes:
                print(f"   ⚪ HOLD disagreement from: {', '.join(hold_votes)}")
    
    # Analyze system-specific predictions
    print(f"\n🔬 SYSTEM-SPECIFIC ANALYSIS:")
    print("="*60)
    
    # Try to get individual predictions if available
    try:
        # Check neural network predictions
        neural_system = systems.get('NeuralNetworkSystem')
        if neural_system and hasattr(neural_system, 'models'):
            print(f"\n🧠 NEURAL NETWORK SYSTEM:")
            print(f"   Models: {len(neural_system.models)}")
            
            # Get individual model predictions if possible
            try:
                # This might not work depending on implementation
                features = system._prepare_features("XAUUSDc")
                if features is not None:
                    print(f"   Features shape: {features.shape}")
                    
                    for model_name, model in neural_system.models.items():
                        try:
                            pred = model.predict(features, verbose=0)[0][0]
                            print(f"   {model_name}: {pred:.3f}")
                        except:
                            print(f"   {model_name}: Unable to get prediction")
            except Exception as e:
                print(f"   ⚠️ Could not analyze individual models: {e}")
        
        # Check AI Phases system
        ai_phases = systems.get('AIPhaseSystem')
        if ai_phases:
            print(f"\n🚀 AI PHASES SYSTEM:")
            print(f"   Type: {type(ai_phases).__name__}")
            print(f"   Boost: +12.0%")
        
        # Check AI2 Advanced Technologies
        ai2_advanced = systems.get('AI2AdvancedTechnologiesSystem')
        if ai2_advanced:
            print(f"\n🔥 AI2.0 ADVANCED TECHNOLOGIES:")
            print(f"   Type: {type(ai2_advanced).__name__}")
            print(f"   Boost: +15.0%")
        
        # Check other major systems
        for sys_name in ['DataQualityMonitor', 'LatencyOptimizer', 'MT5ConnectionManager', 
                        'AdvancedAIEnsembleSystem', 'RealTimeMT5DataSystem']:
            if sys_name in systems:
                sys_obj = systems[sys_name]
                print(f"\n📊 {sys_name.upper()}:")
                print(f"   Type: {type(sys_obj).__name__}")
                print(f"   Status: Active")
        
    except Exception as e:
        print(f"❌ Error in system analysis: {e}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("="*60)
    
    if consensus < 0.7:
        print("⚠️ LOW CONSENSUS RECOMMENDATIONS:")
        print("   1. Check individual system configurations")
        print("   2. Verify market data consistency across systems")
        print("   3. Consider retraining neural models")
        print("   4. Review system weights in ensemble")
    elif consensus < 0.8:
        print("⚡ MODERATE CONSENSUS RECOMMENDATIONS:")
        print("   1. Monitor system performance over time")
        print("   2. Check for conflicting market signals")
        print("   3. Consider minor parameter adjustments")
    else:
        print("✅ GOOD CONSENSUS:")
        print("   1. System is working well")
        print("   2. Continue monitoring")

if __name__ == "__main__":
    analyze_consensus_breakdown() 