#!/usr/bin/env python3
"""
PHÂN TÍCH VAI TRÒ VÀ MỨC ĐỘ QUYẾT ĐỊNH CỦA TỪNG THÀNH PHẦN
Trong quá trình tạo signal của hệ thống AI3.0
"""

import sys
sys.path.append('src/core')

def analyze_system_roles():
    """Phân tích vai trò của từng hệ thống trong signal generation"""
    print("🎯 PHÂN TÍCH VAI TRÒ VÀ MỨC ĐỘ QUYẾT ĐỊNH")
    print("=" * 60)
    
    try:
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        # Initialize system
        config = SystemConfig()
        system = UltimateXAUSystem(config)
        
        print("\n📊 1. TRỌNG SỐ CƠ BẢN CỦA CÁC HỆ THỐNG")
        print("-" * 40)
        
        # Analyze base weights
        base_weights = {
            'NeuralNetworkSystem': 0.25,        # 25% - VAI TRÒ CHÍNH
            'MT5ConnectionManager': 0.20,       # 20% - VAI TRÒ QUAN TRỌNG
            'AdvancedAIEnsembleSystem': 0.20,   # 20% - VAI TRÒ QUAN TRỌNG
            'DataQualityMonitor': 0.15,         # 15% - VAI TRÒ HỖ TRỢ
            'AIPhaseSystem': 0.15,              # 15% - VAI TRÒ HỖ TRỢ
            'RealTimeMT5DataSystem': 0.15,      # 15% - VAI TRÒ HỖ TRỢ
            'AI2AdvancedTechnologiesSystem': 0.10,  # 10% - VAI TRÒ PHỤ
            'LatencyOptimizer': 0.10,           # 10% - VAI TRÒ PHỤ
        }
        
        # Sort by weight
        sorted_systems = sorted(base_weights.items(), key=lambda x: x[1], reverse=True)
        
        for system_name, weight in sorted_systems:
            role = get_system_role(weight)
            print(f"   {role} {system_name}: {weight:.1%}")
        
        print("\n🎭 2. HYBRID ENSEMBLE DECISION PROCESS")
        print("-" * 40)
        
        print("   Step 1: AI2.0 Weighted Average (70% influence)")
        print("   • Mỗi system đóng góp theo trọng số")
        print("   • NeuralNetworkSystem có ảnh hưởng lớn nhất")
        print("   • Weighted prediction = Σ(prediction × weight)")
        
        print("\n   Step 2: AI3.0 Democratic Consensus (30% influence)")
        print("   • Mỗi system vote BUY/SELL/HOLD")
        print("   • Đếm votes và tính consensus ratio")
        print("   • Democratic decision = majority vote")
        
        print("\n   Step 3: Hybrid Consensus Calculation")
        print("   • Hybrid = (consensus_ratio × 0.7) + (agreement × 0.3)")
        print("   • Final confidence = base_confidence × hybrid_consensus")
        
        print("\n🏆 3. CÁC HỆ THỐNG VAI TRÒ CHÍNH")
        print("-" * 40)
        
        main_systems = [
            ('NeuralNetworkSystem', 25, 'Quyết định chính - AI prediction'),
            ('MT5ConnectionManager', 20, 'Dữ liệu thị trường real-time'),
            ('AdvancedAIEnsembleSystem', 20, 'Ensemble AI models'),
        ]
        
        for name, weight, description in main_systems:
            print(f"   🎯 {name}")
            print(f"      • Trọng số: {weight}%")
            print(f"      • Vai trò: {description}")
            print(f"      • Mức độ ảnh hưởng: CHÍNH")
            print()
        
        print("🔧 4. CÁC HỆ THỐNG VAI TRÒ HỖ TRỢ")
        print("-" * 40)
        
        support_systems = [
            ('DataQualityMonitor', 15, 'Kiểm tra chất lượng dữ liệu'),
            ('AIPhaseSystem', 15, '6 Phases AI enhancement (+12% boost)'),
            ('RealTimeMT5DataSystem', 15, 'Streaming data từ MT5'),
        ]
        
        for name, weight, description in support_systems:
            print(f"   🛠️ {name}")
            print(f"      • Trọng số: {weight}%")
            print(f"      • Vai trò: {description}")
            print(f"      • Mức độ ảnh hưởng: HỖ TRỢ")
            print()
        
        print("⚙️ 5. CÁC HỆ THỐNG VAI TRÒ PHỤ")
        print("-" * 40)
        
        minor_systems = [
            ('AI2AdvancedTechnologiesSystem', 10, 'Advanced AI techniques (+15% boost)'),
            ('LatencyOptimizer', 10, 'Tối ưu hiệu suất hệ thống'),
        ]
        
        for name, weight, description in minor_systems:
            print(f"   ⚡ {name}")
            print(f"      • Trọng số: {weight}%")
            print(f"      • Vai trò: {description}")
            print(f"      • Mức độ ảnh hưởng: PHỤ")
            print()
        
        print("🎲 6. DYNAMIC WEIGHT ADJUSTMENT")
        print("-" * 40)
        
        print("   Performance Multiplier Formula:")
        print("   • accuracy_multiplier = 0.5 + avg_accuracy")
        print("   • confidence_multiplier = 1.0 + (total_votes / 1000)")
        print("   • final_weight = base_weight × accuracy_multiplier × confidence_multiplier")
        print()
        print("   Ví dụ:")
        print("   • NeuralNetwork với 80% accuracy: 0.25 × (0.5 + 0.8) = 0.325 (30% tăng)")
        print("   • DataQuality với 60% accuracy: 0.15 × (0.5 + 0.6) = 0.165 (10% tăng)")
        
        print("\n🎯 7. QUYẾT ĐỊNH CUỐI CÙNG")
        print("-" * 40)
        
        print("   Decision Logic Thresholds:")
        print("   • STRONG BUY: signal_strength > 0.2 AND hybrid_consensus >= 0.5")
        print("   • MODERATE BUY: signal_strength > 0.04 AND hybrid_consensus >= 0.5")
        print("   • STRONG SELL: signal_strength < -0.2 AND hybrid_consensus >= 0.5")
        print("   • MODERATE SELL: signal_strength < -0.04 AND hybrid_consensus >= 0.5")
        print("   • HOLD: Tất cả trường hợp khác")
        
        print("\n📈 8. MỨC ĐỘ ẢNH HƯỞNG THỰC TẾ")
        print("-" * 40)
        
        # Generate a test signal to show real influence
        signal = system.generate_signal()
        
        if 'systems_used' in signal:
            print(f"   Số systems tham gia: {signal['systems_used']}")
            print(f"   Action quyết định: {signal['action']}")
            print(f"   Confidence: {signal['confidence']:.1%}")
            
            if 'voting_results' in signal:
                votes = signal['voting_results']
                total = votes['buy_votes'] + votes['sell_votes'] + votes['hold_votes']
                print(f"   Vote distribution:")
                print(f"   • BUY: {votes['buy_votes']}/{total} ({votes['buy_votes']/total:.1%})")
                print(f"   • SELL: {votes['sell_votes']}/{total} ({votes['sell_votes']/total:.1%})")
                print(f"   • HOLD: {votes['hold_votes']}/{total} ({votes['hold_votes']/total:.1%})")
            
            if 'hybrid_metrics' in signal:
                metrics = signal['hybrid_metrics']
                print(f"   Hybrid consensus: {metrics['hybrid_consensus']:.1%}")
                print(f"   Signal strength: {metrics['signal_strength']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_system_role(weight):
    """Xác định vai trò dựa trên trọng số"""
    if weight >= 0.20:
        return "🎯 VAI TRÒ CHÍNH   "
    elif weight >= 0.15:
        return "🛠️ VAI TRÒ HỖ TRỢ "
    else:
        return "⚡ VAI TRÒ PHỤ    "

def analyze_18_specialists_roles():
    """Phân tích vai trò của 18 specialists trong democratic voting"""
    print("\n" + "="*60)
    print("🗳️ PHÂN TÍCH VAI TRÒ 18 SPECIALISTS")
    print("="*60)
    
    specialists_by_category = {
        'Technical (3 specialists)': {
            'weight': '16.7%',
            'specialists': ['RSI_Specialist', 'MACD_Specialist', 'Fibonacci_Specialist'],
            'role': 'Phân tích kỹ thuật cơ bản',
            'influence': 'TRUNG BÌNH'
        },
        'Sentiment (3 specialists)': {
            'weight': '16.7%', 
            'specialists': ['News_Sentiment', 'Social_Media', 'Fear_Greed'],
            'role': 'Phân tích tâm lý thị trường',
            'influence': 'THẤP (cần data thực)'
        },
        'Pattern (3 specialists)': {
            'weight': '16.7%',
            'specialists': ['Chart_Pattern', 'Candlestick', 'Wave_Analysis'],
            'role': 'Nhận dạng mô hình giá',
            'influence': 'CAO'
        },
        'Risk (3 specialists)': {
            'weight': '16.7%',
            'specialists': ['VaR_Risk', 'Drawdown', 'Position_Size'],
            'role': 'Quản lý rủi ro',
            'influence': 'TRUNG BÌNH'
        },
        'Momentum (3 specialists)': {
            'weight': '16.7%',
            'specialists': ['Trend', 'Mean_Reversion', 'Breakout'],
            'role': 'Phân tích động lượng',
            'influence': 'RẤT CAO (75% accuracy)'
        },
        'Volatility (3 specialists)': {
            'weight': '16.7%',
            'specialists': ['ATR', 'Bollinger', 'Volatility_Clustering'],
            'role': 'Phân tích biến động',
            'influence': 'TRUNG BÌNH'
        }
    }
    
    print("📊 Democratic Voting Process:")
    print("   • Mỗi specialist vote: BUY/SELL/HOLD")
    print("   • Consensus threshold: 12/18 specialists (67%)")
    print("   • Category weighting dựa trên market conditions")
    print("   • Final decision: Majority vote với confidence weighting")
    
    print("\n🏆 Ranking theo Performance (từ test scenarios):")
    performance_ranking = [
        ('Fibonacci_Specialist', 87.5, 'Technical'),
        ('Mean_Reversion_Specialist', 87.5, 'Momentum'),
        ('Candlestick_Specialist', 75.0, 'Pattern'),
        ('Trend_Specialist', 75.0, 'Momentum'),
        ('MACD_Specialist', 62.5, 'Technical'),
    ]
    
    for i, (name, accuracy, category) in enumerate(performance_ranking, 1):
        print(f"   {i}. {name}: {accuracy}% accuracy ({category})")
    
    print(f"\n📈 Category Performance:")
    for category, info in specialists_by_category.items():
        print(f"   {category}: {info['influence']} influence")
        print(f"      • Role: {info['role']}")
        print(f"      • Weight: {info['weight']} per category")
    
    print("\n🎯 Consensus Analysis từ Real Data:")
    print("   • High consensus (>70%): 100% accuracy")
    print("   • Medium consensus (55-70%): 75% accuracy")  
    print("   • Low consensus (<55%): 50% accuracy")
    print("   • Transaction rate: 12.4% (conservative approach)")

if __name__ == "__main__":
    success = analyze_system_roles()
    if success:
        analyze_18_specialists_roles()
        print("\n🎉 PHÂN TÍCH HOÀN TẤT!")
        print("\n📋 TÓM TẮT VAI TRÒ:")
        print("   🎯 CHÍNH: NeuralNetwork (25%), MT5Connection (20%), AIEnsemble (20%)")
        print("   🛠️ HỖ TRỢ: DataQuality (15%), AIPhases (15%), RealTimeMT5 (15%)")
        print("   ⚡ PHỤ: AI2Advanced (10%), LatencyOptimizer (10%)")
        print("   🗳️ DEMOCRATIC: 18 specialists với equal voting rights")
    else:
        print("\n❌ PHÂN TÍCH THẤT BẠI!") 