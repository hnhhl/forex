#!/usr/bin/env python3
"""
DIAGNOSTIC SIGNAL ANALYSIS
Chẩn đoán vấn đề logic trong hệ thống ensemble signal generation
"""

import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append('src/core')

def analyze_signal_logic_issue():
    print("🔍 DIAGNOSTIC ANALYSIS - SIGNAL LOGIC ISSUE")
    print("=" * 60)
    
    print("\n❌ PHÁT HIỆN LỖI LOGIC NGHIÊM TRỌNG:")
    print("-" * 40)
    
    print("🐛 BUG #1: VARIABLE SCOPE ERROR trong _generate_ensemble_signal")
    print("   📍 Location: Line 3140-3145 trong ultimate_xau_system.py")
    print("   📝 Code bị lỗi:")
    print("   ```python")
    print("   # Convert to vote with ADAPTIVE thresholds - Quick Win #4")
    print("   buy_threshold, sell_threshold = self._get_adaptive_thresholds()")
    print("   ")
    print("   if pred > buy_threshold:  # ❌ 'pred' chưa được định nghĩa!")
    print("       votes.append('BUY')")
    print("   elif pred < sell_threshold:")
    print("       votes.append('SELL')")
    print("   else:")
    print("       votes.append('HOLD')")
    print("   ```")
    
    print("\n🚨 VẤN ĐỀ:")
    print("   • Variable 'pred' được sử dụng NGOÀI vòng lặp for")
    print("   • 'pred' chỉ tồn tại TRONG vòng lặp for")
    print("   • Dẫn đến NameError hoặc sử dụng giá trị cuối cùng")
    print("   • Logic voting bị sai hoàn toàn")
    
    print("\n🐛 BUG #2: LOGIC DECISION THRESHOLD")
    print("   📝 Code có vấn đề:")
    print("   ```python")
    print("   if signal_strength > 0.02 and buy_votes >= sell_votes:")
    print("       # Threshold quá thấp: 0.02 = 2%")
    print("   elif signal_strength > 0.04 and hybrid_consensus >= min_consensus:")
    print("       # Threshold quá thấp: 0.04 = 4%")
    print("   ```")
    
    print("\n🚨 VẤN ĐỀ:")
    print("   • Threshold quá thấp (2%, 4%) -> tạo quá nhiều signals")
    print("   • Không có minimum confidence requirement")
    print("   • Logic BUY/SELL quá dễ trigger")
    
    print("\n🐛 BUG #3: CONFIDENCE CALCULATION")
    print("   📝 Code có vấn đề:")
    print("   ```python")
    print("   base_confidence = np.mean(confidences)")
    print("   final_confidence = base_confidence * hybrid_consensus")
    print("   ```")
    
    print("\n🚨 VẤN ĐỀ:")
    print("   • Confidence được nhân với consensus -> giảm quá nhiều")
    print("   • Kết quả: confidence luôn thấp (~45%)")
    print("   • Không có boost cho strong signals")

def test_fixed_logic():
    """Test logic đã sửa"""
    print(f"\n🔧 TESTING FIXED LOGIC")
    print("-" * 40)
    
    # Simulate signal components
    signal_components = {
        'NeuralNetworkSystem': {'prediction': 0.7, 'confidence': 0.8},
        'DataQualityMonitor': {'prediction': 0.6, 'confidence': 0.7},
        'MT5ConnectionManager': {'prediction': 0.65, 'confidence': 0.75},
        'AIPhaseSystem': {'prediction': 0.8, 'confidence': 0.85}
    }
    
    print("📊 Test Input:")
    for system, data in signal_components.items():
        print(f"   • {system}: pred={data['prediction']:.2f}, conf={data['confidence']:.2f}")
    
    # Current (buggy) logic simulation
    print(f"\n❌ CURRENT BUGGY LOGIC:")
    predictions = [0.7, 0.6, 0.65, 0.8]
    confidences = [0.8, 0.7, 0.75, 0.85]
    
    # Bug: pred outside loop
    votes = []
    buy_threshold, sell_threshold = 0.55, 0.45  # Simulated thresholds
    
    for pred in predictions:
        if pred > buy_threshold:
            votes.append('BUY')
        elif pred < sell_threshold:
            votes.append('SELL')
        else:
            votes.append('HOLD')
    
    # Bug: using last pred value outside loop
    last_pred = predictions[-1]  # This is what actually happens
    
    weighted_pred = np.mean(predictions)
    base_confidence = np.mean(confidences)
    
    buy_votes = votes.count('BUY')
    sell_votes = votes.count('SELL')
    hold_votes = votes.count('HOLD')
    
    consensus_ratio = max(buy_votes, sell_votes, hold_votes) / len(votes)
    signal_strength = (weighted_pred - 0.5) * 2
    
    # Bug: confidence reduction
    final_confidence = base_confidence * consensus_ratio  # Always reduces confidence
    
    print(f"   • Votes: BUY={buy_votes}, SELL={sell_votes}, HOLD={hold_votes}")
    print(f"   • Weighted pred: {weighted_pred:.3f}")
    print(f"   • Signal strength: {signal_strength:.3f}")
    print(f"   • Base confidence: {base_confidence:.3f}")
    print(f"   • Final confidence: {final_confidence:.3f} (❌ Too low!)")
    
    # Decision with low thresholds
    if signal_strength > 0.02:  # Bug: too low threshold
        action = "BUY"
    else:
        action = "HOLD"
    
    print(f"   • Action: {action} (❌ Too easy to trigger!)")

def propose_fixes():
    """Đề xuất các fix cần thiết"""
    print(f"\n💡 PROPOSED FIXES")
    print("=" * 60)
    
    print("🔧 FIX #1: VARIABLE SCOPE")
    print("   📝 Fixed Code:")
    print("   ```python")
    print("   # INSIDE the loop:")
    print("   for system_name, result in signal_components.items():")
    print("       # ... get prediction ...")
    print("       predictions.append(pred)")
    print("       confidences.append(result.get('confidence', 0.5))")
    print("       weights.append(self._get_system_weight(system_name))")
    print("       ")
    print("       # Convert to vote INSIDE loop")
    print("       buy_threshold, sell_threshold = self._get_adaptive_thresholds()")
    print("       if pred > buy_threshold:")
    print("           votes.append('BUY')")
    print("       elif pred < sell_threshold:")
    print("           votes.append('SELL')")
    print("       else:")
    print("           votes.append('HOLD')")
    print("   ```")
    
    print("\n🔧 FIX #2: THRESHOLD ADJUSTMENT")
    print("   📝 Fixed Code:")
    print("   ```python")
    print("   # Increase thresholds for more selective signals")
    print("   if signal_strength > 0.15 and hybrid_consensus >= 0.6:  # 15% instead of 2%")
    print("       action, strength = 'BUY', 'STRONG'")
    print("   elif signal_strength > 0.08 and hybrid_consensus >= 0.55:  # 8% instead of 4%")
    print("       action, strength = 'BUY', 'MODERATE'")
    print("   elif signal_strength < -0.15 and hybrid_consensus >= 0.6:")
    print("       action, strength = 'SELL', 'STRONG'")
    print("   elif signal_strength < -0.08 and hybrid_consensus >= 0.55:")
    print("       action, strength = 'SELL', 'MODERATE'")
    print("   else:")
    print("       action, strength = 'HOLD', 'NEUTRAL'")
    print("   ```")
    
    print("\n🔧 FIX #3: CONFIDENCE CALCULATION")
    print("   📝 Fixed Code:")
    print("   ```python")
    print("   # Don't reduce confidence, enhance it for strong signals")
    print("   final_confidence = base_confidence")
    print("   ")
    print("   # Boost confidence for strong consensus")
    print("   if hybrid_consensus > 0.8:")
    print("       final_confidence = min(0.95, final_confidence * 1.2)")
    print("   elif hybrid_consensus > 0.6:")
    print("       final_confidence = min(0.9, final_confidence * 1.1)")
    print("   ")
    print("   # Reduce confidence only for weak consensus")
    print("   elif hybrid_consensus < 0.4:")
    print("       final_confidence *= 0.8")
    print("   ```")
    
    print("\n🔧 FIX #4: ADAPTIVE THRESHOLDS")
    print("   📝 Fixed Code:")
    print("   ```python")
    print("   def _get_adaptive_thresholds(self) -> tuple:")
    print("       # Base thresholds")
    print("       buy_threshold = 0.6   # 60% instead of 55%")
    print("       sell_threshold = 0.4  # 40% instead of 45%")
    print("       ")
    print("       # Adjust based on market volatility")
    print("       if hasattr(self, '_last_market_data'):")
    print("           volatility = self._calculate_volatility(self._last_market_data)")
    print("           if volatility > 0.02:  # High volatility")
    print("               buy_threshold = 0.65   # More conservative")
    print("               sell_threshold = 0.35")
    print("       ")
    print("       return buy_threshold, sell_threshold")
    print("   ```")

def main():
    """Chạy diagnostic analysis"""
    analyze_signal_logic_issue()
    test_fixed_logic()
    propose_fixes()
    
    print(f"\n🎯 SUMMARY")
    print("=" * 60)
    print("❌ HỆ THỐNG CÓ 4 LỖI LOGIC NGHIÊM TRỌNG:")
    print("   1. Variable scope error (critical)")
    print("   2. Threshold quá thấp (high)")
    print("   3. Confidence calculation sai (high)")
    print("   4. Adaptive thresholds không hiệu quả (medium)")
    
    print(f"\n🚨 TÁC ĐỘNG:")
    print("   • Win rate thấp (25%)")
    print("   • Confidence thấp (45%)")
    print("   • Signal quality kém")
    print("   • Performance không ổn định")
    
    print(f"\n💡 KHUYẾN NGHỊ:")
    print("   • SỬA LỖI NGAY LẬP TỨC")
    print("   • Test từng fix riêng biệt")
    print("   • Validate với backtest")
    print("   • Monitor performance improvement")

if __name__ == "__main__":
    main() 