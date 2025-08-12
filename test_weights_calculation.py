#!/usr/bin/env python3
"""
🎯 KIỂM TRA WEIGHTS THỰC TẾ VÀ PHÂN TÍCH HỆ THỐNG 4 CẤP
Test script để xác định chính xác cách hệ thống phân chia quyền lực
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def analyze_actual_weights():
    """Phân tích weights thực tế từ code"""
    
    print("🔍 PHÂN TÍCH WEIGHTS THỰC TẾ TRONG HỆ THỐNG")
    print("=" * 60)
    
    # Base weights từ code thực tế
    base_weights = {
        'NeuralNetworkSystem': 0.25,           # CẤP 1 - Neural Networks (25%)
        'MT5ConnectionManager': 0.20,          # CẤP 1 - Data provider (20%)
        'AdvancedAIEnsembleSystem': 0.20,      # CẤP 2 - AI Ensemble (20%)
        'DataQualityMonitor': 0.15,            # CẤP 2 - Data validation (15%)
        'AIPhaseSystem': 0.15,                 # CẤP 2 - AI Phases (+12% boost) (15%)
        'RealTimeMT5DataSystem': 0.15,         # CẤP 2 - Real-time streaming (15%)
        'AI2AdvancedTechnologiesSystem': 0.10, # CẤP 3 - Advanced AI (+15% boost) (10%)
        'LatencyOptimizer': 0.10,              # CẤP 3 - Performance optimization (10%)
        'DemocraticSpecialistsSystem': 1.0,    # CẤP 4 - Democratic layer (Full voting)
        
        # Core Trading Systems (được thêm vào)
        'PortfolioManager': 0.20,              # Portfolio management
        'OrderManager': 0.05,                  # Order execution
        'StopLossManager': 0.05,               # Stop loss management
        'PositionSizer': 0.10,                 # Position sizing
        'KellyCriterionCalculator': 0.10       # Kelly Criterion
    }
    
    print("📊 BASE WEIGHTS THEO CODE:")
    total_base = 0
    for system, weight in base_weights.items():
        print(f"  {system:<35}: {weight:>6.1%}")
        total_base += weight
    
    print(f"\n📈 TỔNG BASE WEIGHTS: {total_base:.1%}")
    print(f"⚠️  VẤN ĐỀ: Tổng weights = {total_base:.1%} (vượt quá 100%!)")
    
    return base_weights, total_base

def analyze_tier_distribution():
    """Phân tích phân chia theo 4 cấp"""
    
    print("\n🏆 PHÂN TÍCH PHÂN CHIA 4 CẤP")
    print("=" * 60)
    
    # Phân chia theo cấp (theo thiết kế ban đầu)
    tier_1_systems = {
        'NeuralNetworkSystem': 0.25,
        'MT5ConnectionManager': 0.20,
        'PortfolioManager': 0.20  # Thêm vào từ Core Trading
    }
    
    tier_2_systems = {
        'AdvancedAIEnsembleSystem': 0.20,
        'DataQualityMonitor': 0.15,
        'AIPhaseSystem': 0.15,  # +12% boost riêng biệt
        'RealTimeMT5DataSystem': 0.15
    }
    
    tier_3_systems = {
        'AI2AdvancedTechnologiesSystem': 0.10,  # +15% boost riêng biệt
        'LatencyOptimizer': 0.10
    }
    
    tier_4_systems = {
        'DemocraticSpecialistsSystem': 1.0  # 18 specialists với equal voting
    }
    
    # Core Trading Systems (cần phân loại lại)
    core_trading = {
        'OrderManager': 0.05,
        'StopLossManager': 0.05, 
        'PositionSizer': 0.10,
        'KellyCriterionCalculator': 0.10
    }
    
    print("🥇 CẤP 1 - HỆ THỐNG CHÍNH:")
    tier_1_total = sum(tier_1_systems.values())
    for system, weight in tier_1_systems.items():
        print(f"  {system:<35}: {weight:>6.1%}")
    print(f"  TỔNG CẤP 1: {tier_1_total:.1%}")
    
    print("\n🥈 CẤP 2 - HỆ THỐNG HỖ TRỢ:")
    tier_2_total = sum(tier_2_systems.values())
    for system, weight in tier_2_systems.items():
        boost = ""
        if system == 'AIPhaseSystem':
            boost = " (+12% boost)"
        print(f"  {system:<35}: {weight:>6.1%}{boost}")
    print(f"  TỔNG CẤP 2: {tier_2_total:.1%}")
    
    print("\n🥉 CẤP 3 - HỆ THỐNG PHỤ:")
    tier_3_total = sum(tier_3_systems.values())
    for system, weight in tier_3_systems.items():
        boost = ""
        if system == 'AI2AdvancedTechnologiesSystem':
            boost = " (+15% boost)"
        print(f"  {system:<35}: {weight:>6.1%}{boost}")
    print(f"  TỔNG CẤP 3: {tier_3_total:.1%}")
    
    print("\n🗳️ CẤP 4 - DEMOCRATIC LAYER:")
    tier_4_total = sum(tier_4_systems.values())
    for system, weight in tier_4_systems.items():
        print(f"  {system:<35}: {weight:>6.1%} (18 specialists)")
    print(f"  TỔNG CẤP 4: {tier_4_total:.1%}")
    
    print("\n⚙️ CORE TRADING SYSTEMS (Cần phân loại lại):")
    core_total = sum(core_trading.values())
    for system, weight in core_trading.items():
        print(f"  {system:<35}: {weight:>6.1%}")
    print(f"  TỔNG CORE TRADING: {core_total:.1%}")
    
    grand_total = tier_1_total + tier_2_total + tier_3_total + tier_4_total + core_total
    print(f"\n📊 TỔNG TẤT CẢ: {grand_total:.1%}")
    
    return {
        'tier_1': tier_1_total,
        'tier_2': tier_2_total, 
        'tier_3': tier_3_total,
        'tier_4': tier_4_total,
        'core_trading': core_total,
        'grand_total': grand_total
    }

def propose_optimal_weights():
    """Đề xuất phân chia weights tối ưu"""
    
    print("\n🎯 ĐỀ XUẤT HỆ THỐNG WEIGHTS TỐI ƯU")
    print("=" * 60)
    
    optimal_weights = {
        # CẤP 1 - CORE DECISION (40%)
        'NeuralNetworkSystem': 0.20,        # Primary AI engine
        'PortfolioManager': 0.15,           # Capital allocation
        'OrderManager': 0.05,               # Execution engine
        
        # CẤP 2 - AI ENHANCEMENT (35%)
        'AdvancedAIEnsembleSystem': 0.20,   # Multi-model consensus
        'AIPhaseSystem': 0.15,              # Performance boosting (+12% boost riêng)
        
        # CẤP 3 - OPTIMIZATION (15%)
        'LatencyOptimizer': 0.05,           # Speed optimization
        'AI2AdvancedTechnologiesSystem': 0.10, # Advanced AI (+15% boost riêng)
        
        # CẤP 4 - CONSENSUS (10%)
        'DemocraticSpecialistsSystem': 0.10, # Democratic validation
        
        # SUPPORT LAYER (0% voting, 100% service)
        'MT5ConnectionManager': 0.0,        # Data provider
        'DataQualityMonitor': 0.0,          # Data validator
        'RealTimeMT5DataSystem': 0.0,       # Data streamer
        'StopLossManager': 0.0,             # Risk protector
        'PositionSizer': 0.0,               # Size calculator
        'KellyCriterionCalculator': 0.0     # Optimization calculator
    }
    
    print("🎯 PHÂN CHIA TỐI ƯU (100% TOTAL):")
    
    # Group by tiers
    tier_1_optimal = ['NeuralNetworkSystem', 'PortfolioManager', 'OrderManager']
    tier_2_optimal = ['AdvancedAIEnsembleSystem', 'AIPhaseSystem']
    tier_3_optimal = ['LatencyOptimizer', 'AI2AdvancedTechnologiesSystem']
    tier_4_optimal = ['DemocraticSpecialistsSystem']
    support_optimal = ['MT5ConnectionManager', 'DataQualityMonitor', 'RealTimeMT5DataSystem', 
                      'StopLossManager', 'PositionSizer', 'KellyCriterionCalculator']
    
    print("\n🥇 CẤP 1 - CORE DECISION (40%):")
    tier_1_total = 0
    for system in tier_1_optimal:
        weight = optimal_weights[system]
        tier_1_total += weight
        print(f"  {system:<35}: {weight:>6.1%}")
    print(f"  TỔNG CẤP 1: {tier_1_total:.1%}")
    
    print("\n🥈 CẤP 2 - AI ENHANCEMENT (35%):")
    tier_2_total = 0
    for system in tier_2_optimal:
        weight = optimal_weights[system]
        tier_2_total += weight
        boost = " (+12% boost riêng)" if system == 'AIPhaseSystem' else ""
        print(f"  {system:<35}: {weight:>6.1%}{boost}")
    print(f"  TỔNG CẤP 2: {tier_2_total:.1%}")
    
    print("\n🥉 CẤP 3 - OPTIMIZATION (15%):")
    tier_3_total = 0
    for system in tier_3_optimal:
        weight = optimal_weights[system]
        tier_3_total += weight
        boost = " (+15% boost riêng)" if system == 'AI2AdvancedTechnologiesSystem' else ""
        print(f"  {system:<35}: {weight:>6.1%}{boost}")
    print(f"  TỔNG CẤP 3: {tier_3_total:.1%}")
    
    print("\n🗳️ CẤP 4 - CONSENSUS (10%):")
    tier_4_total = 0
    for system in tier_4_optimal:
        weight = optimal_weights[system]
        tier_4_total += weight
        print(f"  {system:<35}: {weight:>6.1%}")
    print(f"  TỔNG CẤP 4: {tier_4_total:.1%}")
    
    print("\n📊 SUPPORT LAYER (0% voting, 100% service):")
    for system in support_optimal:
        weight = optimal_weights[system]
        print(f"  {system:<35}: {weight:>6.1%} (Support only)")
    
    optimal_total = tier_1_total + tier_2_total + tier_3_total + tier_4_total
    print(f"\n✅ TỔNG VOTING POWER: {optimal_total:.1%}")
    
    return optimal_weights

def calculate_boost_effects():
    """Tính toán hiệu ứng của boost mechanisms"""
    
    print("\n🚀 PHÂN TÍCH BOOST MECHANISMS")
    print("=" * 60)
    
    base_prediction = 0.65  # Ví dụ prediction
    
    print(f"📊 BASE PREDICTION: {base_prediction:.1%}")
    
    # AI Phases boost (+12%)
    ai_phases_boost = 0.12
    prediction_with_ai_phases = base_prediction * (1 + ai_phases_boost)
    print(f"🔄 AI Phases Boost: +{ai_phases_boost:.1%}")
    print(f"   Prediction sau AI Phases: {prediction_with_ai_phases:.1%}")
    
    # AI2 boost (+15%)
    ai2_boost = 0.15
    prediction_with_ai2 = prediction_with_ai_phases * (1 + ai2_boost)
    print(f"🤖 AI2 Advanced Boost: +{ai2_boost:.1%}")
    print(f"   Prediction sau AI2: {prediction_with_ai2:.1%}")
    
    # Combined boost effect
    total_boost = (1 + ai_phases_boost) * (1 + ai2_boost) - 1
    final_prediction = base_prediction * (1 + total_boost)
    
    print(f"\n📈 TỔNG HỢP:")
    print(f"   Combined Boost Effect: +{total_boost:.1%}")
    print(f"   Final Prediction: {final_prediction:.1%}")
    print(f"   Improvement: {(final_prediction - base_prediction)/base_prediction:.1%}")
    
    return {
        'base_prediction': base_prediction,
        'ai_phases_boost': ai_phases_boost,
        'ai2_boost': ai2_boost,
        'total_boost': total_boost,
        'final_prediction': final_prediction
    }

def analyze_democratic_layer():
    """Phân tích Democratic Layer chi tiết"""
    
    print("\n🗳️ PHÂN TÍCH DEMOCRATIC LAYER CHI TIẾT")
    print("=" * 60)
    
    # 18 specialists theo categories
    specialist_categories = {
        'Technical Analysis': [
            'RSI_Specialist', 'MACD_Specialist', 'Bollinger_Specialist'
        ],
        'Sentiment Analysis': [
            'News_Sentiment_Specialist', 'Social_Media_Specialist', 'Market_Fear_Greed_Specialist'
        ],
        'Pattern Recognition': [
            'Chart_Pattern_Specialist', 'Candlestick_Specialist', 'Support_Resistance_Specialist'
        ],
        'Risk Management': [
            'Volatility_Specialist', 'Correlation_Specialist', 'Drawdown_Specialist'
        ],
        'Momentum Analysis': [
            'Trend_Following_Specialist', 'Mean_Reversion_Specialist', 'Breakout_Specialist'
        ],
        'Volatility Analysis': [
            'VIX_Specialist', 'ATR_Specialist', 'GARCH_Specialist'
        ]
    }
    
    total_specialists = sum(len(specialists) for specialists in specialist_categories.values())
    vote_per_specialist = 1.0 / total_specialists  # Equal voting
    
    print(f"📊 DEMOCRATIC STRUCTURE:")
    print(f"   Total Specialists: {total_specialists}")
    print(f"   Vote per Specialist: {vote_per_specialist:.3%}")
    print(f"   Categories: {len(specialist_categories)}")
    
    print(f"\n🏷️ SPECIALIST CATEGORIES:")
    for category, specialists in specialist_categories.items():
        category_vote = len(specialists) * vote_per_specialist
        print(f"   {category:<20}: {len(specialists)} specialists ({category_vote:.1%} total vote)")
        for specialist in specialists:
            print(f"     - {specialist}")
    
    # Voting scenarios
    print(f"\n📊 VOTING SCENARIOS:")
    
    # Scenario 1: Strong consensus (16/18 agree)
    strong_consensus = 16/18
    print(f"   Strong Consensus (16/18): {strong_consensus:.1%} agreement")
    
    # Scenario 2: Weak consensus (10/18 agree)
    weak_consensus = 10/18
    print(f"   Weak Consensus (10/18): {weak_consensus:.1%} agreement")
    
    # Scenario 3: No consensus (9/18 agree)
    no_consensus = 9/18
    print(f"   No Consensus (9/18): {no_consensus:.1%} agreement")
    
    print(f"\n⚖️ CONSENSUS THRESHOLDS:")
    print(f"   Recommended threshold: 67% (12/18 specialists)")
    print(f"   Strong signal threshold: 78% (14/18 specialists)")
    print(f"   Emergency override: 89% (16/18 specialists)")
    
    return {
        'total_specialists': total_specialists,
        'vote_per_specialist': vote_per_specialist,
        'categories': specialist_categories,
        'strong_consensus': strong_consensus,
        'weak_consensus': weak_consensus,
        'no_consensus': no_consensus
    }

def generate_implementation_code():
    """Tạo code implementation cho optimal weights"""
    
    print("\n💻 IMPLEMENTATION CODE")
    print("=" * 60)
    
    code = '''
def get_optimal_system_weights():
    """Optimal weights distribution for AI3.0 system"""
    return {
        # CẤP 1 - CORE DECISION (40%)
        'NeuralNetworkSystem': 0.20,
        'PortfolioManager': 0.15,
        'OrderManager': 0.05,
        
        # CẤP 2 - AI ENHANCEMENT (35%)
        'AdvancedAIEnsembleSystem': 0.20,
        'AIPhaseSystem': 0.15,  # +12% boost separate
        
        # CẤP 3 - OPTIMIZATION (15%)
        'LatencyOptimizer': 0.05,
        'AI2AdvancedTechnologiesSystem': 0.10,  # +15% boost separate
        
        # CẤP 4 - CONSENSUS (10%)
        'DemocraticSpecialistsSystem': 0.10,
        
        # SUPPORT LAYER (0% voting)
        'MT5ConnectionManager': 0.0,
        'DataQualityMonitor': 0.0,
        'RealTimeMT5DataSystem': 0.0,
        'StopLossManager': 0.0,
        'PositionSizer': 0.0,
        'KellyCriterionCalculator': 0.0
    }

def calculate_final_prediction_with_boosts(base_prediction, ai_phases_active=True, ai2_active=True):
    """Calculate final prediction with boost mechanisms"""
    prediction = base_prediction
    
    # Apply AI Phases boost (+12%)
    if ai_phases_active:
        prediction *= 1.12
        
    # Apply AI2 Advanced boost (+15%)
    if ai2_active:
        prediction *= 1.15
        
    # Ensure prediction stays within bounds
    return min(1.0, max(0.0, prediction))

def democratic_consensus_with_threshold(specialist_votes, threshold=0.67):
    """Democratic consensus with configurable threshold"""
    if len(specialist_votes) == 0:
        return 0.5  # Neutral
        
    # Count votes
    buy_votes = sum(1 for vote in specialist_votes if vote > 0.5)
    total_votes = len(specialist_votes)
    
    # Calculate consensus strength
    consensus_strength = buy_votes / total_votes
    
    # Apply threshold
    if consensus_strength >= threshold:
        return consensus_strength
    elif (1 - consensus_strength) >= threshold:
        return 1 - consensus_strength
    else:
        return 0.5  # No consensus, return neutral
    '''
    
    print(code)
    
    return code

def main():
    """Main analysis function"""
    
    print("🎯 PHÂN TÍCH TOÀN DIỆN HỆ THỐNG 4 CẤP QUYẾT ĐỊNH AI3.0")
    print("=" * 80)
    print(f"⏰ Thời gian phân tích: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Phân tích weights thực tế
    base_weights, total_base = analyze_actual_weights()
    
    # 2. Phân tích phân chia 4 cấp
    tier_analysis = analyze_tier_distribution()
    
    # 3. Đề xuất weights tối ưu
    optimal_weights = propose_optimal_weights()
    
    # 4. Phân tích boost effects
    boost_analysis = calculate_boost_effects()
    
    # 5. Phân tích Democratic Layer
    democratic_analysis = analyze_democratic_layer()
    
    # 6. Tạo implementation code
    implementation_code = generate_implementation_code()
    
    # 7. Kết luận và khuyến nghị
    print("\n🎯 KẾT LUẬN VÀ KHUYẾN NGHỊ")
    print("=" * 60)
    
    print("❌ VẤN ĐỀ HIỆN TẠI:")
    print(f"   - Tổng weights: {total_base:.1%} (vượt quá 100%)")
    print(f"   - Democratic layer quá mạnh: 100% voting power")
    print(f"   - Data systems có quyền vote trading decisions")
    print(f"   - Thiếu cân bằng giữa prediction và execution")
    
    print("\n✅ GIẢI PHÁP ĐỀ XUẤT:")
    print("   - Rebalance weights: 40-35-15-10 distribution")
    print("   - Tách voting systems và support systems")
    print("   - Democratic layer chỉ 10% để validation")
    print("   - Boost mechanisms tính riêng biệt")
    
    print("\n🚀 HÀNH ĐỘNG CẦN THỰC HIỆN:")
    print("   1. Update _get_system_weight() method")
    print("   2. Implement optimal weights distribution")
    print("   3. Separate boost calculations")
    print("   4. Add democratic consensus threshold")
    print("   5. Move data systems to support layer")
    
    # Save results
    results = {
        'analysis_time': datetime.now().isoformat(),
        'current_weights': base_weights,
        'current_total': total_base,
        'tier_analysis': tier_analysis,
        'optimal_weights': optimal_weights,
        'boost_analysis': boost_analysis,
        'democratic_analysis': democratic_analysis,
        'implementation_code': implementation_code
    }
    
    with open('weights_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Kết quả đã được lưu vào: weights_analysis_results.json")
    
    return results

if __name__ == "__main__":
    results = main() 