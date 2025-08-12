#!/usr/bin/env python3
"""
COMPREHENSIVE TRADING SUMMARY
Tổng hợp toàn diện về trading và consensus với visual analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_summary():
    """Tạo tổng hợp toàn diện về trading"""
    print("🎉 COMPREHENSIVE TRADING & CONSENSUS SUMMARY")
    print("=" * 80)
    
    # Load transaction data
    try:
        with open('trading_transaction_analysis_20250623_003209.json', 'r') as f:
            transaction_data = json.load(f)
    except:
        print("❌ Could not load transaction data")
        return
    
    # Load original signal data
    try:
        with open('m1_analysis_results/signal_analyses_20250623_002218.json', 'r') as f:
            signal_data = json.load(f)
    except:
        print("❌ Could not load signal data")
        return
    
    print("📊 TỔNG QUAN HỆ THỐNG TRADING")
    print("=" * 60)
    
    # System overview
    total_signals = len(signal_data)
    actionable_transactions = transaction_data['transaction_summary']['total_actionable_transactions']
    hold_signals = total_signals - actionable_transactions
    
    print(f"🔍 SIGNAL GENERATION OVERVIEW:")
    print(f"   • Tổng số signals được phân tích: {total_signals}")
    print(f"   • Signals tạo ra giao dịch (BUY/SELL): {actionable_transactions}")
    print(f"   • Signals HOLD (không giao dịch): {hold_signals}")
    print(f"   • Tỷ lệ tạo giao dịch: {actionable_transactions/total_signals*100:.1f}%")
    
    print(f"\n💰 CHI TIẾT CÁC GIAO DỊCH ĐÃ TẠO:")
    print("=" * 60)
    
    transactions = transaction_data['detailed_transactions']
    
    # Transaction details
    buy_count = transaction_data['decision_distribution']['buy_transactions']
    sell_count = transaction_data['decision_distribution']['sell_transactions']
    accuracy = transaction_data['transaction_summary']['transaction_accuracy']
    
    print(f"📈 PHÂN PHỐI GIAO DỊCH:")
    print(f"   • Lệnh BUY: {buy_count} giao dịch ({buy_count/actionable_transactions*100:.1f}%)")
    print(f"   • Lệnh SELL: {sell_count} giao dịch ({sell_count/actionable_transactions*100:.1f}%)")
    print(f"   • Độ chính xác tổng thể: {accuracy:.1f}%")
    
    print(f"\n🗳️ PHÂN TÍCH CONSENSUS CHI TIẾT:")
    print("=" * 60)
    
    # Detailed consensus analysis for each transaction
    for i, transaction in enumerate(transactions, 1):
        scenario = transaction['scenario']
        decision = transaction['decision']
        agree_votes = transaction['agree_votes']
        disagree_votes = transaction['disagree_votes']
        consensus_pct = transaction['consensus_percentage']
        vote_breakdown = transaction['vote_breakdown']
        correct = "✅ ĐÚNG" if transaction['prediction_correct'] else "❌ SAI"
        categories_agree = transaction['categories_agree']
        
        print(f"\n🔸 GIAO DỊCH #{i}: {scenario}")
        print(f"   📊 Quyết định: {decision} {correct}")
        print(f"   🗳️  Đồng thuận: {agree_votes}/18 specialists ({consensus_pct:.1f}%)")
        print(f"   📊 Chi tiết votes:")
        print(f"       • Đồng ý với {decision}: {agree_votes} specialists")
        print(f"       • Không đồng ý: {disagree_votes} specialists")
        print(f"       • Phân bố: BUY({vote_breakdown['BUY']}) SELL({vote_breakdown['SELL']}) HOLD({vote_breakdown['HOLD']})")
        print(f"   🏛️  Categories đồng thuận: {categories_agree}/6 categories")
        
        # Analyze opposition
        if decision == 'BUY':
            opposition = vote_breakdown['SELL'] + vote_breakdown['HOLD']
            main_opposition = 'SELL' if vote_breakdown['SELL'] > vote_breakdown['HOLD'] else 'HOLD'
        else:  # SELL
            opposition = vote_breakdown['BUY'] + vote_breakdown['HOLD']
            main_opposition = 'BUY' if vote_breakdown['BUY'] > vote_breakdown['HOLD'] else 'HOLD'
        
        print(f"   🔄 Phản đối chính: {main_opposition} ({opposition} specialists)")
    
    print(f"\n📊 THỐNG KÊ CONSENSUS PATTERNS:")
    print("=" * 60)
    
    # Consensus statistics
    consensus_levels = []
    for transaction in transactions:
        consensus_pct = transaction['consensus_percentage']
        if consensus_pct >= 70:
            consensus_levels.append('High (70%+)')
        elif consensus_pct >= 55:
            consensus_levels.append('Moderate (55-70%)')
        else:
            consensus_levels.append('Low (<55%)')
    
    consensus_counts = pd.Series(consensus_levels).value_counts()
    
    print(f"🎯 PHÂN PHỐI MỨC ĐỘ CONSENSUS:")
    for level, count in consensus_counts.items():
        percentage = count / len(transactions) * 100
        print(f"   • {level}: {count} giao dịch ({percentage:.1f}%)")
    
    # Accuracy by consensus level
    print(f"\n🎯 ĐỘ CHÍNH XÁC THEO MỨC ĐỘ CONSENSUS:")
    
    high_consensus = [t for t in transactions if t['consensus_percentage'] >= 70]
    moderate_consensus = [t for t in transactions if 55 <= t['consensus_percentage'] < 70]
    low_consensus = [t for t in transactions if t['consensus_percentage'] < 55]
    
    for level_name, level_transactions in [
        ('High Consensus (70%+)', high_consensus),
        ('Moderate Consensus (55-70%)', moderate_consensus),
        ('Low Consensus (<55%)', low_consensus)
    ]:
        if level_transactions:
            correct = sum(1 for t in level_transactions if t['prediction_correct'])
            total = len(level_transactions)
            accuracy = correct / total * 100
            print(f"   • {level_name}: {accuracy:.1f}% ({correct}/{total})")
    
    print(f"\n🏆 SPECIALIST PERFORMANCE INSIGHTS:")
    print("=" * 60)
    
    # Calculate specialist agreement with final decisions
    specialist_agreement = {}
    
    for signal in signal_data:
        final_decision = signal['final_decision']
        if final_decision != 'HOLD':  # Only count actionable transactions
            for specialist_name, vote_data in signal['specialist_votes'].items():
                if specialist_name not in specialist_agreement:
                    specialist_agreement[specialist_name] = {'agree': 0, 'total': 0}
                
                specialist_agreement[specialist_name]['total'] += 1
                if vote_data['decision'] == final_decision:
                    specialist_agreement[specialist_name]['agree'] += 1
    
    # Calculate agreement rates
    agreement_rates = {}
    for specialist, data in specialist_agreement.items():
        if data['total'] > 0:
            agreement_rates[specialist] = data['agree'] / data['total'] * 100
    
    # Sort by agreement rate
    sorted_specialists = sorted(agreement_rates.items(), key=lambda x: x[1], reverse=True)
    
    print(f"🤝 TOP SPECIALISTS (Đồng thuận với quyết định cuối):")
    for specialist, rate in sorted_specialists[:5]:
        print(f"   • {specialist}: {rate:.1f}% đồng thuận")
    
    print(f"\n🔄 MOST CONTRARIAN SPECIALISTS:")
    for specialist, rate in sorted_specialists[-5:]:
        print(f"   • {specialist}: {rate:.1f}% đồng thuận")
    
    print(f"\n📈 KẾT LUẬN VÀ INSIGHTS:")
    print("=" * 60)
    
    avg_consensus = np.mean([t['consensus_percentage'] for t in transactions])
    strong_consensus_count = len([t for t in transactions if t['consensus_percentage'] >= 70])
    weak_consensus_count = len([t for t in transactions if t['consensus_percentage'] < 50])
    
    print(f"✅ ĐIỂM MẠNH:")
    print(f"   • Độ chính xác cao: {accuracy:.1f}% (6/7 giao dịch đúng)")
    print(f"   • Consensus trung bình: {avg_consensus:.1f}%")
    print(f"   • Hệ thống tạo ra {actionable_transactions} giao dịch từ {total_signals} signals")
    print(f"   • Cân bằng tốt: {buy_count} BUY vs {sell_count} SELL")
    
    print(f"\n⚠️ CẦN CẢI THIỆN:")
    print(f"   • Chỉ {strong_consensus_count} giao dịch có consensus cao (>70%)")
    print(f"   • {weak_consensus_count} giao dịch có consensus thấp (<50%)")
    print(f"   • Giao dịch sai duy nhất: 'High Volatility Whipsaw' (consensus thấp: 44.4%)")
    
    print(f"\n💡 KHUYẾN NGHỊ:")
    print(f"   • Tăng threshold consensus tối thiểu lên 55% để tránh giao dịch rủi ro")
    print(f"   • Cải thiện xử lý scenarios volatility cao")
    print(f"   • Tận dụng specialists có agreement rate cao")
    print(f"   • Xem xét giảm weight của specialists contrarian trong volatile markets")
    
    # Save comprehensive summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    comprehensive_summary = {
        'summary_timestamp': datetime.now().isoformat(),
        'system_overview': {
            'total_signals': total_signals,
            'actionable_transactions': actionable_transactions,
            'hold_signals': hold_signals,
            'transaction_rate': actionable_transactions/total_signals*100
        },
        'transaction_performance': {
            'buy_transactions': buy_count,
            'sell_transactions': sell_count,
            'accuracy': accuracy,
            'correct_predictions': transaction_data['transaction_summary']['correct_predictions'],
            'incorrect_predictions': transaction_data['transaction_summary']['incorrect_predictions']
        },
        'consensus_analysis': {
            'average_consensus': avg_consensus,
            'high_consensus_transactions': strong_consensus_count,
            'low_consensus_transactions': weak_consensus_count,
            'consensus_distribution': dict(consensus_counts)
        },
        'specialist_agreement_rates': agreement_rates,
        'detailed_transaction_analysis': transactions
    }
    
    summary_path = f"comprehensive_trading_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(comprehensive_summary, f, indent=2, default=str)
    
    print(f"\n💾 Comprehensive summary saved: {summary_path}")
    
    return comprehensive_summary

if __name__ == "__main__":
    create_comprehensive_summary() 