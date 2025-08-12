#!/usr/bin/env python3
"""
DETAILED SIGNAL TRACKING ANALYSIS
Phân tích chi tiết tracking signal với thống kê từng specialist và category
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_analysis_results():
    """Load kết quả phân tích từ files JSON"""
    print("📊 LOADING ANALYSIS RESULTS...")
    print("=" * 60)
    
    try:
        # Load specialist performance
        with open('m1_analysis_results/specialist_performance_20250623_002218.json', 'r') as f:
            specialist_performance = json.load(f)
        
        # Load signal analyses
        with open('m1_analysis_results/signal_analyses_20250623_002218.json', 'r') as f:
            signal_analyses = json.load(f)
        
        # Load summary
        with open('m1_analysis_results/summary_20250623_002218.json', 'r') as f:
            summary = json.load(f)
        
        print(f"✅ Loaded specialist performance: {len(specialist_performance)} specialists")
        print(f"✅ Loaded signal analyses: {len(signal_analyses)} scenarios")
        print(f"✅ Loaded summary data")
        
        return specialist_performance, signal_analyses, summary
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None, None, None

def analyze_specialist_voting_patterns(specialist_performance, signal_analyses):
    """Phân tích patterns voting của specialists"""
    print("\n🔍 ANALYZING VOTING PATTERNS...")
    print("=" * 60)
    
    # Create voting matrix
    specialists = list(specialist_performance.keys())
    scenarios = [analysis['scenario']['name'] for analysis in signal_analyses]
    
    voting_matrix = []
    confidence_matrix = []
    
    for analysis in signal_analyses:
        scenario_votes = []
        scenario_confidences = []
        
        for specialist in specialists:
            vote_data = analysis['specialist_votes'][specialist]
            
            # Convert vote to numeric
            vote_numeric = {'BUY': 1, 'HOLD': 0, 'SELL': -1}[vote_data['decision']]
            scenario_votes.append(vote_numeric)
            scenario_confidences.append(vote_data['confidence'])
        
        voting_matrix.append(scenario_votes)
        confidence_matrix.append(scenario_confidences)
    
    # Convert to DataFrames
    voting_df = pd.DataFrame(voting_matrix, columns=specialists, index=scenarios)
    confidence_df = pd.DataFrame(confidence_matrix, columns=specialists, index=scenarios)
    
    print(f"✅ Created voting matrix: {voting_df.shape}")
    
    # Analyze voting patterns
    print(f"\n📊 VOTING PATTERNS ANALYSIS:")
    
    # 1. Specialist agreement analysis
    specialist_correlations = voting_df.corr()
    
    # Find most agreeable specialists
    avg_correlations = specialist_correlations.mean().sort_values(ascending=False)
    print(f"\n🤝 MOST AGREEABLE SPECIALISTS:")
    for specialist, corr in avg_correlations.head(5).items():
        category = specialist_performance[specialist]['category']
        print(f"   • {specialist}: {corr:.3f} avg correlation ({category})")
    
    # Find most contrarian specialists
    print(f"\n🔄 MOST CONTRARIAN SPECIALISTS:")
    for specialist, corr in avg_correlations.tail(5).items():
        category = specialist_performance[specialist]['category']
        print(f"   • {specialist}: {corr:.3f} avg correlation ({category})")
    
    # 2. Category consensus analysis
    category_votes = {}
    for specialist, perf in specialist_performance.items():
        category = perf['category']
        if category not in category_votes:
            category_votes[category] = []
        
        specialist_votes = voting_df[specialist].tolist()
        category_votes[category].extend(specialist_votes)
    
    print(f"\n📊 CATEGORY VOTING TENDENCIES:")
    for category, votes in category_votes.items():
        buy_pct = (np.array(votes) == 1).mean() * 100
        sell_pct = (np.array(votes) == -1).mean() * 100
        hold_pct = (np.array(votes) == 0).mean() * 100
        
        print(f"   • {category}: BUY {buy_pct:.1f}%, SELL {sell_pct:.1f}%, HOLD {hold_pct:.1f}%")
    
    # 3. Scenario difficulty analysis
    scenario_consensus = []
    for i, scenario in enumerate(scenarios):
        votes = voting_df.iloc[i]
        consensus = max(abs(votes.sum()), len(votes) - abs(votes.sum())) / len(votes)
        scenario_consensus.append(consensus)
    
    scenario_difficulty = pd.DataFrame({
        'scenario': scenarios,
        'consensus': scenario_consensus
    }).sort_values('consensus')
    
    print(f"\n🎯 SCENARIO DIFFICULTY (Lowest consensus = Most difficult):")
    for _, row in scenario_difficulty.head(5).iterrows():
        print(f"   • {row['scenario']}: {row['consensus']:.1%} consensus")
    
    return voting_df, confidence_df, specialist_correlations

def analyze_signal_quality_metrics(signal_analyses):
    """Phân tích chất lượng signal"""
    print("\n📈 ANALYZING SIGNAL QUALITY METRICS...")
    print("=" * 60)
    
    quality_metrics = []
    
    for analysis in signal_analyses:
        scenario = analysis['scenario']
        
        # Calculate various quality metrics
        metrics = {
            'scenario': scenario['name'],
            'scenario_type': scenario['type'],
            'actual_direction': scenario['actual_direction'],
            'final_decision': analysis['final_decision'],
            'prediction_correct': analysis['prediction_correct'],
            'consensus_strength': analysis['consensus_strength'],
            'weighted_confidence': analysis['weighted_confidence'],
            'vote_distribution': analysis['vote_distribution'],
            'total_votes': sum(analysis['vote_distribution'].values()),
            'category_agreements': 0,
            'high_confidence_votes': 0,
            'unanimous_categories': 0
        }
        
        # Count category agreements
        category_consensus = analysis['category_consensus']
        final_decision = analysis['final_decision']
        
        for category, consensus in category_consensus.items():
            if consensus['majority_decision'] == final_decision:
                metrics['category_agreements'] += 1
            
            if consensus['consensus_strength'] == 1.0:
                metrics['unanimous_categories'] += 1
        
        # Count high confidence votes
        for specialist_vote in analysis['specialist_votes'].values():
            if specialist_vote['confidence'] > 0.7:
                metrics['high_confidence_votes'] += 1
        
        # Calculate quality score
        quality_score = (
            metrics['consensus_strength'] * 0.3 +
            metrics['weighted_confidence'] * 0.3 +
            (metrics['category_agreements'] / 6) * 0.2 +
            (metrics['high_confidence_votes'] / 18) * 0.2
        )
        metrics['quality_score'] = quality_score
        
        quality_metrics.append(metrics)
    
    quality_df = pd.DataFrame(quality_metrics)
    
    print(f"✅ Analyzed {len(quality_metrics)} signals")
    
    # Quality analysis
    print(f"\n📊 SIGNAL QUALITY ANALYSIS:")
    print(f"   • Average consensus strength: {quality_df['consensus_strength'].mean():.1%}")
    print(f"   • Average weighted confidence: {quality_df['weighted_confidence'].mean():.1%}")
    print(f"   • Average quality score: {quality_df['quality_score'].mean():.3f}")
    print(f"   • Signals with >70% consensus: {(quality_df['consensus_strength'] > 0.7).sum()}/{len(quality_df)}")
    print(f"   • Signals with >5 high confidence votes: {(quality_df['high_confidence_votes'] > 5).sum()}/{len(quality_df)}")
    
    # Best and worst quality signals
    print(f"\n🏆 HIGHEST QUALITY SIGNALS:")
    best_signals = quality_df.nlargest(3, 'quality_score')
    for _, signal in best_signals.iterrows():
        print(f"   • {signal['scenario']}: {signal['quality_score']:.3f} "
              f"({signal['consensus_strength']:.1%} consensus, "
              f"{'✅' if signal['prediction_correct'] else '❌'})")
    
    print(f"\n⚠️ LOWEST QUALITY SIGNALS:")
    worst_signals = quality_df.nsmallest(3, 'quality_score')
    for _, signal in worst_signals.iterrows():
        print(f"   • {signal['scenario']}: {signal['quality_score']:.3f} "
              f"({signal['consensus_strength']:.1%} consensus, "
              f"{'✅' if signal['prediction_correct'] else '❌'})")
    
    return quality_df

def analyze_specialist_accuracy_vs_confidence(specialist_performance):
    """Phân tích mối quan hệ accuracy vs confidence"""
    print("\n🎯 ANALYZING ACCURACY VS CONFIDENCE...")
    print("=" * 60)
    
    accuracy_confidence_data = []
    
    for name, perf in specialist_performance.items():
        accuracy_confidence_data.append({
            'specialist': name,
            'category': perf['category'],
            'accuracy_rate': perf.get('accuracy_rate', 0),
            'avg_confidence': perf.get('avg_confidence', 0),
            'total_votes': perf['total_votes'],
            'bias': perf['bias']
        })
    
    ac_df = pd.DataFrame(accuracy_confidence_data)
    
    # Calculate correlation
    correlation = ac_df['accuracy_rate'].corr(ac_df['avg_confidence'])
    print(f"📊 Accuracy vs Confidence correlation: {correlation:.3f}")
    
    # Find over/under confident specialists
    ac_df['confidence_accuracy_diff'] = ac_df['avg_confidence'] - ac_df['accuracy_rate']
    
    print(f"\n😤 OVERCONFIDENT SPECIALISTS (High confidence, Low accuracy):")
    overconfident = ac_df.nlargest(5, 'confidence_accuracy_diff')
    for _, spec in overconfident.iterrows():
        print(f"   • {spec['specialist']}: {spec['avg_confidence']:.3f} confidence, "
              f"{spec['accuracy_rate']:.3f} accuracy ({spec['category']})")
    
    print(f"\n😔 UNDERCONFIDENT SPECIALISTS (Low confidence, High accuracy):")
    underconfident = ac_df.nsmallest(5, 'confidence_accuracy_diff')
    for _, spec in underconfident.iterrows():
        print(f"   • {spec['specialist']}: {spec['avg_confidence']:.3f} confidence, "
              f"{spec['accuracy_rate']:.3f} accuracy ({spec['category']})")
    
    # Category analysis
    print(f"\n📊 CATEGORY ACCURACY VS CONFIDENCE:")
    category_stats = ac_df.groupby('category').agg({
        'accuracy_rate': 'mean',
        'avg_confidence': 'mean',
        'total_votes': 'sum'
    }).round(3)
    
    for category, stats in category_stats.iterrows():
        print(f"   • {category}: {stats['accuracy_rate']:.3f} accuracy, "
              f"{stats['avg_confidence']:.3f} confidence")
    
    return ac_df

def analyze_market_scenario_responses(signal_analyses):
    """Phân tích phản ứng với các scenarios thị trường"""
    print("\n📊 ANALYZING MARKET SCENARIO RESPONSES...")
    print("=" * 60)
    
    scenario_responses = {}
    
    for analysis in signal_analyses:
        scenario_type = analysis['scenario']['type']
        
        if scenario_type not in scenario_responses:
            scenario_responses[scenario_type] = {
                'total_signals': 0,
                'correct_predictions': 0,
                'avg_consensus': [],
                'avg_confidence': [],
                'vote_distributions': []
            }
        
        resp = scenario_responses[scenario_type]
        resp['total_signals'] += 1
        
        if analysis['prediction_correct']:
            resp['correct_predictions'] += 1
        
        resp['avg_consensus'].append(analysis['consensus_strength'])
        resp['avg_confidence'].append(analysis['weighted_confidence'])
        resp['vote_distributions'].append(analysis['vote_distribution'])
    
    print(f"📊 SCENARIO TYPE PERFORMANCE:")
    for scenario_type, resp in scenario_responses.items():
        accuracy = resp['correct_predictions'] / resp['total_signals'] if resp['total_signals'] > 0 else 0
        avg_consensus = np.mean(resp['avg_consensus'])
        avg_confidence = np.mean(resp['avg_confidence'])
        
        print(f"   • {scenario_type.upper()}: {accuracy:.1%} accuracy, "
              f"{avg_consensus:.1%} consensus, {avg_confidence:.1%} confidence "
              f"({resp['total_signals']} signals)")
    
    return scenario_responses

def create_detailed_report(specialist_performance, signal_analyses, quality_df, ac_df):
    """Tạo báo cáo chi tiết"""
    print("\n📋 CREATING DETAILED REPORT...")
    print("=" * 60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report = {
        'report_timestamp': datetime.now().isoformat(),
        'executive_summary': {
            'total_specialists': len(specialist_performance),
            'total_scenarios': len(signal_analyses),
            'overall_signal_accuracy': quality_df['prediction_correct'].mean(),
            'avg_consensus_strength': quality_df['consensus_strength'].mean(),
            'avg_quality_score': quality_df['quality_score'].mean()
        },
        'specialist_rankings': {
            'by_accuracy': ac_df.nlargest(10, 'accuracy_rate')[['specialist', 'category', 'accuracy_rate', 'avg_confidence']].to_dict('records'),
            'by_confidence': ac_df.nlargest(10, 'avg_confidence')[['specialist', 'category', 'accuracy_rate', 'avg_confidence']].to_dict('records'),
            'most_active': ac_df.nlargest(10, 'total_votes')[['specialist', 'category', 'total_votes', 'accuracy_rate']].to_dict('records')
        },
        'signal_quality_analysis': {
            'high_quality_signals': quality_df[quality_df['quality_score'] > 0.6].to_dict('records'),
            'low_quality_signals': quality_df[quality_df['quality_score'] < 0.4].to_dict('records'),
            'quality_distribution': {
                'excellent': (quality_df['quality_score'] > 0.7).sum(),
                'good': ((quality_df['quality_score'] > 0.5) & (quality_df['quality_score'] <= 0.7)).sum(),
                'fair': ((quality_df['quality_score'] > 0.3) & (quality_df['quality_score'] <= 0.5)).sum(),
                'poor': (quality_df['quality_score'] <= 0.3).sum()
            }
        },
        'category_performance': {},
        'improvement_recommendations': []
    }
    
    # Category performance
    for category in ac_df['category'].unique():
        category_data = ac_df[ac_df['category'] == category]
        report['category_performance'][category] = {
            'specialist_count': len(category_data),
            'avg_accuracy': category_data['accuracy_rate'].mean(),
            'avg_confidence': category_data['avg_confidence'].mean(),
            'best_specialist': category_data.loc[category_data['accuracy_rate'].idxmax(), 'specialist'],
            'worst_specialist': category_data.loc[category_data['accuracy_rate'].idxmin(), 'specialist']
        }
    
    # Improvement recommendations
    # 1. Overconfident specialists
    overconfident = ac_df[ac_df['confidence_accuracy_diff'] > 0.2]
    if len(overconfident) > 0:
        report['improvement_recommendations'].append({
            'type': 'confidence_calibration',
            'description': 'Reduce confidence for overconfident specialists',
            'affected_specialists': overconfident['specialist'].tolist(),
            'priority': 'high'
        })
    
    # 2. Low performing categories
    category_performance = ac_df.groupby('category')['accuracy_rate'].mean()
    low_performing = category_performance[category_performance < 0.5]
    if len(low_performing) > 0:
        report['improvement_recommendations'].append({
            'type': 'category_improvement',
            'description': 'Improve algorithms for underperforming categories',
            'affected_categories': low_performing.index.tolist(),
            'priority': 'medium'
        })
    
    # 3. Low consensus signals
    low_consensus = quality_df[quality_df['consensus_strength'] < 0.4]
    if len(low_consensus) > 0:
        report['improvement_recommendations'].append({
            'type': 'consensus_improvement',
            'description': 'Review scenarios with low consensus',
            'affected_scenarios': low_consensus['scenario'].tolist(),
            'priority': 'low'
        })
    
    # Save report
    report_path = f"detailed_signal_tracking_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Detailed report saved: {report_path}")
    
    return report

def print_comprehensive_analysis(report):
    """In phân tích toàn diện"""
    print(f"\n🎉 COMPREHENSIVE SIGNAL TRACKING ANALYSIS")
    print("=" * 80)
    
    exec_summary = report['executive_summary']
    print(f"📊 EXECUTIVE SUMMARY:")
    print(f"   • Total specialists: {exec_summary['total_specialists']}")
    print(f"   • Total scenarios: {exec_summary['total_scenarios']}")
    print(f"   • Overall signal accuracy: {exec_summary['overall_signal_accuracy']:.1%}")
    print(f"   • Average consensus strength: {exec_summary['avg_consensus_strength']:.1%}")
    print(f"   • Average quality score: {exec_summary['avg_quality_score']:.3f}")
    
    print(f"\n🏆 TOP PERFORMING SPECIALISTS (By Accuracy):")
    for i, spec in enumerate(report['specialist_rankings']['by_accuracy'][:5], 1):
        print(f"   {i}. {spec['specialist']}: {spec['accuracy_rate']:.1%} accuracy ({spec['category']})")
    
    print(f"\n📊 CATEGORY PERFORMANCE:")
    for category, perf in report['category_performance'].items():
        print(f"   • {category}: {perf['avg_accuracy']:.1%} accuracy, {perf['specialist_count']} specialists")
        print(f"     Best: {perf['best_specialist']}, Worst: {perf['worst_specialist']}")
    
    print(f"\n📈 SIGNAL QUALITY DISTRIBUTION:")
    quality_dist = report['signal_quality_analysis']['quality_distribution']
    total_signals = sum(quality_dist.values())
    for quality, count in quality_dist.items():
        pct = count / total_signals * 100 if total_signals > 0 else 0
        print(f"   • {quality.title()}: {count} signals ({pct:.1f}%)")
    
    print(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
    for i, rec in enumerate(report['improvement_recommendations'], 1):
        print(f"   {i}. {rec['description']} (Priority: {rec['priority']})")
        if 'affected_specialists' in rec:
            print(f"      Affected: {len(rec['affected_specialists'])} specialists")
        elif 'affected_categories' in rec:
            print(f"      Affected: {', '.join(rec['affected_categories'])}")

def run_detailed_analysis():
    """Chạy phân tích chi tiết toàn diện"""
    print("🚀 DETAILED SIGNAL TRACKING ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # 1. Load results
    specialist_performance, signal_analyses, summary = load_analysis_results()
    if not all([specialist_performance, signal_analyses, summary]):
        print("❌ Failed to load analysis results")
        return
    
    # 2. Analyze voting patterns
    voting_df, confidence_df, correlations = analyze_specialist_voting_patterns(
        specialist_performance, signal_analyses
    )
    
    # 3. Analyze signal quality
    quality_df = analyze_signal_quality_metrics(signal_analyses)
    
    # 4. Analyze accuracy vs confidence
    ac_df = analyze_specialist_accuracy_vs_confidence(specialist_performance)
    
    # 5. Analyze market scenario responses
    scenario_responses = analyze_market_scenario_responses(signal_analyses)
    
    # 6. Create detailed report
    report = create_detailed_report(specialist_performance, signal_analyses, quality_df, ac_df)
    
    # 7. Print comprehensive analysis
    print_comprehensive_analysis(report)
    
    print(f"\nEnd time: {datetime.now()}")
    print("=" * 80)
    
    return report

if __name__ == "__main__":
    run_detailed_analysis() 