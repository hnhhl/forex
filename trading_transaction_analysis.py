#!/usr/bin/env python3
"""
TRADING TRANSACTION ANALYSIS
Phân tích chi tiết các giao dịch được tạo ra và mức độ đồng thuận trong từng giao dịch
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_signal_analyses():
    """Load kết quả phân tích signal"""
    print("📊 LOADING SIGNAL ANALYSES...")
    print("=" * 60)
    
    try:
        with open('m1_analysis_results/signal_analyses_20250623_002218.json', 'r') as f:
            signal_analyses = json.load(f)
        
        print(f"✅ Loaded {len(signal_analyses)} signal analyses")
        return signal_analyses
        
    except Exception as e:
        print(f"❌ Error loading signal analyses: {e}")
        return None

def analyze_trading_transactions(signal_analyses):
    """Phân tích chi tiết các giao dịch trading"""
    print("\n💰 ANALYZING TRADING TRANSACTIONS...")
    print("=" * 60)
    
    transactions = []
    
    for i, analysis in enumerate(signal_analyses, 1):
        scenario = analysis['scenario']
        final_decision = analysis['final_decision']
        
        # Chỉ tạo transaction nếu không phải HOLD
        if final_decision != 'HOLD':
            
            # Phân tích votes chi tiết
            specialist_votes = analysis['specialist_votes']
            vote_counts = analysis['vote_distribution']
            
            # Tính toán consensus details
            total_specialists = 18
            agree_votes = vote_counts[final_decision]
            disagree_votes = total_specialists - agree_votes
            
            # Phân tích theo từng vote
            buy_votes = vote_counts.get('BUY', 0)
            sell_votes = vote_counts.get('SELL', 0)
            hold_votes = vote_counts.get('HOLD', 0)
            
            # Tính consensus strength
            consensus_strength = agree_votes / total_specialists
            
            # Phân tích confidence của specialists đồng thuận vs phản đối
            agree_confidences = []
            disagree_confidences = []
            
            for specialist_name, vote_data in specialist_votes.items():
                if vote_data['decision'] == final_decision:
                    agree_confidences.append(vote_data['confidence'])
                else:
                    disagree_confidences.append(vote_data['confidence'])
            
            # Phân tích theo category
            category_consensus = analysis['category_consensus']
            categories_agree = 0
            categories_disagree = 0
            
            for category, consensus in category_consensus.items():
                if consensus['majority_decision'] == final_decision:
                    categories_agree += 1
                else:
                    categories_disagree += 1
            
            # Tạo transaction record
            transaction = {
                'transaction_id': i,
                'scenario_name': scenario['name'],
                'scenario_type': scenario['type'],
                'signal_decision': final_decision,
                'actual_direction': scenario['actual_direction'],
                'prediction_correct': analysis['prediction_correct'],
                'consensus_details': {
                    'total_specialists': total_specialists,
                    'agree_votes': agree_votes,
                    'disagree_votes': disagree_votes,
                    'consensus_percentage': consensus_strength * 100,
                    'vote_breakdown': {
                        'BUY': buy_votes,
                        'SELL': sell_votes,
                        'HOLD': hold_votes
                    }
                },
                'confidence_analysis': {
                    'agree_avg_confidence': np.mean(agree_confidences) if agree_confidences else 0,
                    'disagree_avg_confidence': np.mean(disagree_confidences) if disagree_confidences else 0,
                    'confidence_difference': np.mean(agree_confidences) - np.mean(disagree_confidences) if agree_confidences and disagree_confidences else 0,
                    'total_specialists_high_confidence': len([c for c in agree_confidences + disagree_confidences if c > 0.7])
                },
                'category_consensus': {
                    'categories_agree': categories_agree,
                    'categories_disagree': categories_disagree,
                    'category_consensus_percentage': categories_agree / 6 * 100,
                    'category_details': category_consensus
                },
                'weighted_confidence': analysis['weighted_confidence'],
                'signal_quality_score': calculate_signal_quality(analysis)
            }
            
            transactions.append(transaction)
    
    print(f"✅ Analyzed {len(transactions)} trading transactions")
    print(f"📊 Total signals generated: {len(signal_analyses)}")
    print(f"📊 Actionable transactions (BUY/SELL): {len(transactions)}")
    print(f"📊 HOLD signals: {len(signal_analyses) - len(transactions)}")
    
    return transactions

def calculate_signal_quality(analysis):
    """Tính toán signal quality score"""
    consensus_strength = analysis['consensus_strength']
    weighted_confidence = analysis['weighted_confidence']
    
    # Count high confidence votes
    high_confidence_votes = 0
    for vote_data in analysis['specialist_votes'].values():
        if vote_data['confidence'] > 0.7:
            high_confidence_votes += 1
    
    # Count category agreements
    category_agreements = 0
    final_decision = analysis['final_decision']
    for category_consensus in analysis['category_consensus'].values():
        if category_consensus['majority_decision'] == final_decision:
            category_agreements += 1
    
    # Calculate quality score
    quality_score = (
        consensus_strength * 0.4 +
        weighted_confidence * 0.3 +
        (category_agreements / 6) * 0.2 +
        (high_confidence_votes / 18) * 0.1
    )
    
    return quality_score

def analyze_consensus_patterns(transactions):
    """Phân tích patterns consensus"""
    print("\n🤝 ANALYZING CONSENSUS PATTERNS...")
    print("=" * 60)
    
    # Consensus distribution
    consensus_ranges = {
        'Very High (80-100%)': 0,
        'High (60-80%)': 0,
        'Moderate (50-60%)': 0,
        'Low (40-50%)': 0,
        'Very Low (<40%)': 0
    }
    
    for transaction in transactions:
        consensus_pct = transaction['consensus_details']['consensus_percentage']
        
        if consensus_pct >= 80:
            consensus_ranges['Very High (80-100%)'] += 1
        elif consensus_pct >= 60:
            consensus_ranges['High (60-80%)'] += 1
        elif consensus_pct >= 50:
            consensus_ranges['Moderate (50-60%)'] += 1
        elif consensus_pct >= 40:
            consensus_ranges['Low (40-50%)'] += 1
        else:
            consensus_ranges['Very Low (<40%)'] += 1
    
    print("📊 CONSENSUS DISTRIBUTION:")
    for range_name, count in consensus_ranges.items():
        percentage = count / len(transactions) * 100 if transactions else 0
        print(f"   • {range_name}: {count} transactions ({percentage:.1f}%)")
    
    # Accuracy by consensus level
    print(f"\n🎯 ACCURACY BY CONSENSUS LEVEL:")
    
    for range_name in consensus_ranges.keys():
        range_transactions = []
        
        for transaction in transactions:
            consensus_pct = transaction['consensus_details']['consensus_percentage']
            
            if range_name == 'Very High (80-100%)' and consensus_pct >= 80:
                range_transactions.append(transaction)
            elif range_name == 'High (60-80%)' and 60 <= consensus_pct < 80:
                range_transactions.append(transaction)
            elif range_name == 'Moderate (50-60%)' and 50 <= consensus_pct < 60:
                range_transactions.append(transaction)
            elif range_name == 'Low (40-50%)' and 40 <= consensus_pct < 50:
                range_transactions.append(transaction)
            elif range_name == 'Very Low (<40%)' and consensus_pct < 40:
                range_transactions.append(transaction)
        
        if range_transactions:
            correct_predictions = sum(1 for t in range_transactions if t['prediction_correct'])
            accuracy = correct_predictions / len(range_transactions) * 100
            print(f"   • {range_name}: {accuracy:.1f}% accuracy ({correct_predictions}/{len(range_transactions)})")

def analyze_specialist_agreement_details(transactions):
    """Phân tích chi tiết sự đồng thuận của specialists"""
    print("\n👥 ANALYZING SPECIALIST AGREEMENT DETAILS...")
    print("=" * 60)
    
    agreement_stats = {
        'unanimous_decisions': 0,
        'strong_majority': 0,
        'simple_majority': 0,
        'plurality': 0
    }
    
    detailed_breakdowns = []
    
    for transaction in transactions:
        consensus_details = transaction['consensus_details']
        agree_votes = consensus_details['agree_votes']
        total_votes = consensus_details['total_specialists']
        
        # Classify agreement level
        if agree_votes == total_votes:
            agreement_level = 'unanimous'
            agreement_stats['unanimous_decisions'] += 1
        elif agree_votes >= total_votes * 0.75:  # 75%+
            agreement_level = 'strong_majority'
            agreement_stats['strong_majority'] += 1
        elif agree_votes >= total_votes * 0.5:   # 50%+
            agreement_level = 'simple_majority'
            agreement_stats['simple_majority'] += 1
        else:
            agreement_level = 'plurality'
            agreement_stats['plurality'] += 1
        
        # Detailed breakdown
        breakdown = {
            'transaction_id': transaction['transaction_id'],
            'scenario': transaction['scenario_name'],
            'decision': transaction['signal_decision'],
            'agreement_level': agreement_level,
            'agree_votes': agree_votes,
            'disagree_votes': consensus_details['disagree_votes'],
            'consensus_percentage': consensus_details['consensus_percentage'],
            'vote_breakdown': consensus_details['vote_breakdown'],
            'prediction_correct': transaction['prediction_correct'],
            'categories_agree': transaction['category_consensus']['categories_agree'],
            'categories_disagree': transaction['category_consensus']['categories_disagree']
        }
        
        detailed_breakdowns.append(breakdown)
    
    print("📊 AGREEMENT LEVEL DISTRIBUTION:")
    total_transactions = len(transactions)
    for level, count in agreement_stats.items():
        percentage = count / total_transactions * 100 if total_transactions > 0 else 0
        print(f"   • {level.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    return detailed_breakdowns

def create_transaction_summary_report(transactions, detailed_breakdowns):
    """Tạo báo cáo tóm tắt giao dịch"""
    print("\n📋 CREATING TRANSACTION SUMMARY REPORT...")
    print("=" * 60)
    
    # Overall statistics
    total_transactions = len(transactions)
    correct_predictions = sum(1 for t in transactions if t['prediction_correct'])
    accuracy = correct_predictions / total_transactions * 100 if total_transactions > 0 else 0
    
    # Consensus statistics
    avg_consensus = np.mean([t['consensus_details']['consensus_percentage'] for t in transactions])
    avg_confidence = np.mean([t['weighted_confidence'] for t in transactions])
    avg_quality_score = np.mean([t['signal_quality_score'] for t in transactions])
    
    # Vote distribution statistics
    total_buy_transactions = len([t for t in transactions if t['signal_decision'] == 'BUY'])
    total_sell_transactions = len([t for t in transactions if t['signal_decision'] == 'SELL'])
    
    # Category consensus statistics
    avg_categories_agree = np.mean([t['category_consensus']['categories_agree'] for t in transactions])
    
    report = {
        'report_timestamp': datetime.now().isoformat(),
        'transaction_summary': {
            'total_signals_analyzed': len(transactions) + sum(1 for analysis in [None] if analysis), # Placeholder for total signals
            'total_actionable_transactions': total_transactions,
            'hold_signals': 0,  # Will be calculated from original data
            'transaction_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': total_transactions - correct_predictions
        },
        'decision_distribution': {
            'buy_transactions': total_buy_transactions,
            'sell_transactions': total_sell_transactions,
            'buy_percentage': total_buy_transactions / total_transactions * 100 if total_transactions > 0 else 0,
            'sell_percentage': total_sell_transactions / total_transactions * 100 if total_transactions > 0 else 0
        },
        'consensus_statistics': {
            'average_consensus_percentage': avg_consensus,
            'average_weighted_confidence': avg_confidence,
            'average_quality_score': avg_quality_score,
            'average_categories_agreeing': avg_categories_agree
        },
        'detailed_transactions': detailed_breakdowns
    }
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"trading_transaction_analysis_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Transaction report saved: {report_path}")
    
    return report

def print_detailed_transaction_analysis(transactions, detailed_breakdowns, report):
    """In phân tích chi tiết giao dịch"""
    print(f"\n🎉 DETAILED TRADING TRANSACTION ANALYSIS")
    print("=" * 80)
    
    summary = report['transaction_summary']
    decisions = report['decision_distribution']
    consensus = report['consensus_statistics']
    
    print(f"📊 TRANSACTION OVERVIEW:")
    print(f"   • Total actionable transactions: {summary['total_actionable_transactions']}")
    print(f"   • Transaction accuracy: {summary['transaction_accuracy']:.1f}%")
    print(f"   • Correct predictions: {summary['correct_predictions']}")
    print(f"   • Incorrect predictions: {summary['incorrect_predictions']}")
    
    print(f"\n💰 DECISION DISTRIBUTION:")
    print(f"   • BUY transactions: {decisions['buy_transactions']} ({decisions['buy_percentage']:.1f}%)")
    print(f"   • SELL transactions: {decisions['sell_transactions']} ({decisions['sell_percentage']:.1f}%)")
    
    print(f"\n🤝 CONSENSUS STATISTICS:")
    print(f"   • Average consensus: {consensus['average_consensus_percentage']:.1f}%")
    print(f"   • Average confidence: {consensus['average_weighted_confidence']:.1f}%")
    print(f"   • Average quality score: {consensus['average_quality_score']:.3f}")
    print(f"   • Average categories agreeing: {consensus['average_categories_agreeing']:.1f}/6")
    
    print(f"\n📋 DETAILED TRANSACTION BREAKDOWN:")
    print("=" * 80)
    
    for breakdown in detailed_breakdowns:
        transaction_id = breakdown['transaction_id']
        scenario = breakdown['scenario']
        decision = breakdown['decision']
        agree_votes = breakdown['agree_votes']
        disagree_votes = breakdown['disagree_votes']
        consensus_pct = breakdown['consensus_percentage']
        vote_breakdown = breakdown['vote_breakdown']
        correct = "✅" if breakdown['prediction_correct'] else "❌"
        categories_agree = breakdown['categories_agree']
        
        print(f"\n🔸 Transaction #{transaction_id}: {scenario}")
        print(f"   📊 Decision: {decision} {correct}")
        print(f"   🗳️  Consensus: {agree_votes}/18 specialists agree ({consensus_pct:.1f}%)")
        print(f"   📊 Vote breakdown: BUY({vote_breakdown['BUY']}) SELL({vote_breakdown['SELL']}) HOLD({vote_breakdown['HOLD']})")
        print(f"   🏛️  Categories: {categories_agree}/6 categories agree")
        print(f"   🎯 Agreement level: {breakdown['agreement_level'].replace('_', ' ').title()}")

def run_transaction_analysis():
    """Chạy phân tích giao dịch toàn diện"""
    print("🚀 TRADING TRANSACTION ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # 1. Load signal analyses
    signal_analyses = load_signal_analyses()
    if not signal_analyses:
        return
    
    # 2. Analyze trading transactions
    transactions = analyze_trading_transactions(signal_analyses)
    
    # 3. Analyze consensus patterns
    analyze_consensus_patterns(transactions)
    
    # 4. Analyze specialist agreement details
    detailed_breakdowns = analyze_specialist_agreement_details(transactions)
    
    # 5. Create summary report
    report = create_transaction_summary_report(transactions, detailed_breakdowns)
    
    # 6. Print detailed analysis
    print_detailed_transaction_analysis(transactions, detailed_breakdowns, report)
    
    print(f"\nEnd time: {datetime.now()}")
    print("=" * 80)
    
    return report

if __name__ == "__main__":
    run_transaction_analysis() 