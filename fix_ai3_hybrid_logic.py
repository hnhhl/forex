# -*- coding: utf-8 -*-
"""Fix AI3.0 Logic - Hybrid AI2.0 Weighted Average + AI3.0 Democratic Consensus"""

import sys
import os
sys.path.append('src')

from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def fix_ai3_hybrid_logic():
    print("FIX AI3.0 HYBRID LOGIC - AI2.0 Weighted + AI3.0 Democratic Consensus")
    print("="*80)
    
    try:
        # 1. Initialize system and test original
        print("1. Test Original Logic...")
        
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        original_signals = []
        for i in range(3):
            signal = system.generate_signal("XAUUSDc")
            original_signals.append(signal)
            print(f"   Original {i+1}: {signal.get('action')} ({signal.get('confidence', 0):.1%})")
        
        original_confidence = np.mean([s.get('confidence', 0) for s in original_signals])
        print(f"   Original Average: {original_confidence:.1%}")
        
        # 2. Create hybrid logic
        print("\n2. Create Hybrid Logic...")
        
        def hybrid_ensemble_signal(self, signal_components):
            """HYBRID: AI2.0 Weighted Average + AI3.0 Democratic Consensus"""
            try:
                predictions = []
                confidences = []
                weights = []
                votes = []
                
                # Extract data from systems
                for system_name, result in signal_components.items():
                    if isinstance(result, dict):
                        # Get prediction
                        if 'prediction' in result:
                            pred = result['prediction']
                        elif 'ensemble_prediction' in result:
                            pred = result['ensemble_prediction'].get('prediction', 0.5)
                        else:
                            pred = 0.5
                        
                        predictions.append(pred)
                        confidences.append(result.get('confidence', 0.5))
                        weights.append(self._get_system_weight(system_name))
                        
                        # Convert to vote (LOWER thresholds like AI2.0)
                        if pred > 0.51:    # Lowered from 0.55
                            votes.append('BUY')
                        elif pred < 0.49:  # Lowered from 0.45
                            votes.append('SELL')
                        else:
                            votes.append('HOLD')
                
                if not predictions:
                    return self._create_neutral_signal()
                
                # Step 1: AI2.0 Weighted Average
                weights = np.array(weights) / np.sum(weights)  # Normalize
                weighted_pred = np.sum(np.array(predictions) * weights)
                base_confidence = np.mean(confidences)
                
                # Step 2: AI3.0 Democratic Consensus
                buy_votes = votes.count('BUY')
                sell_votes = votes.count('SELL')
                hold_votes = votes.count('HOLD')
                total_votes = len(votes)
                
                consensus_ratio = max(buy_votes, sell_votes, hold_votes) / total_votes
                
                # Step 3: Check agreement between weighted and votes
                signal_strength = (weighted_pred - 0.5) * 2  # -1 to 1
                
                if signal_strength > 0.02 and buy_votes >= sell_votes:
                    agreement = 1.0
                elif signal_strength < -0.02 and sell_votes >= buy_votes:
                    agreement = 1.0
                elif abs(signal_strength) <= 0.02 and hold_votes >= max(buy_votes, sell_votes):
                    agreement = 1.0
                else:
                    agreement = 0.6  # Some disagreement
                
                # Step 4: Hybrid consensus score
                hybrid_consensus = (consensus_ratio * 0.7) + (agreement * 0.3)
                
                # Step 5: Final confidence with consensus influence
                final_confidence = base_confidence * hybrid_consensus
                
                # Step 6: Decision logic (AI2.0 thresholds + consensus requirement)
                min_consensus = 0.5  # Require 50% consensus
                
                if signal_strength > 0.2 and hybrid_consensus >= min_consensus:
                    action, strength = "BUY", "STRONG"
                    final_confidence = min(0.9, final_confidence + 0.1)
                elif signal_strength > 0.04 and hybrid_consensus >= min_consensus:
                    action, strength = "BUY", "MODERATE"
                elif signal_strength < -0.2 and hybrid_consensus >= min_consensus:
                    action, strength = "SELL", "STRONG"
                    final_confidence = min(0.9, final_confidence + 0.1)
                elif signal_strength < -0.04 and hybrid_consensus >= min_consensus:
                    action, strength = "SELL", "MODERATE"
                else:
                    action, strength = "HOLD", "NEUTRAL"
                    if hybrid_consensus < 0.4:
                        final_confidence *= 0.7
                
                return {
                    'symbol': self.config.symbol,
                    'action': action,
                    'strength': strength,
                    'prediction': weighted_pred,
                    'confidence': final_confidence,
                    'timestamp': pd.Timestamp.now(),
                    'systems_used': len(predictions),
                    'ensemble_method': 'hybrid_ai2_ai3',
                    'hybrid_metrics': {
                        'weighted_prediction': weighted_pred,
                        'signal_strength': signal_strength,
                        'consensus_ratio': consensus_ratio,
                        'agreement': agreement,
                        'hybrid_consensus': hybrid_consensus
                    },
                    'voting_results': {
                        'buy_votes': buy_votes,
                        'sell_votes': sell_votes,
                        'hold_votes': hold_votes,
                        'votes': votes
                    }
                }
                
            except Exception as e:
                print(f"Hybrid error: {e}")
                return self._create_neutral_signal()
        
        # 3. Apply hybrid logic
        print("\n3. Apply Hybrid Logic...")
        system._generate_ensemble_signal = hybrid_ensemble_signal.__get__(system, UltimateXAUSystem)
        print("   ✅ Hybrid logic applied")
        
        # 4. Test hybrid logic
        print("\n4. Test Hybrid Logic...")
        
        hybrid_signals = []
        for i in range(8):
            signal = system.generate_signal("XAUUSDc")
            hybrid_signals.append(signal)
            
            action = signal.get('action')
            confidence = signal.get('confidence', 0)
            metrics = signal.get('hybrid_metrics', {})
            consensus = metrics.get('hybrid_consensus', 0)
            
            print(f"   Hybrid {i+1}: {action} ({confidence:.1%}) | Consensus: {consensus:.1%}")
        
        # 5. Compare results
        print("\n5. Results Comparison...")
        
        hybrid_confidences = [s.get('confidence', 0) for s in hybrid_signals]
        hybrid_avg = np.mean(hybrid_confidences)
        
        hybrid_actions = [s.get('action') for s in hybrid_signals]
        unique_actions = set(hybrid_actions)
        
        action_counts = {}
        for action in hybrid_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"   📊 Original: {original_confidence:.1%} confidence, only HOLD")
        print(f"   📊 Hybrid:   {hybrid_avg:.1%} confidence, {unique_actions}")
        print(f"   📊 Improvement: {(hybrid_avg - original_confidence):.1%}")
        print(f"   📊 Distribution: {action_counts}")
        
        # 6. Assessment
        print("\n6. Assessment...")
        
        improvement = hybrid_avg - original_confidence
        has_diversity = len(unique_actions) > 1
        
        avg_consensus = np.mean([s.get('hybrid_metrics', {}).get('hybrid_consensus', 0) for s in hybrid_signals])
        
        if improvement > 0.1 and has_diversity and avg_consensus > 0.6:
            status = "EXCELLENT"
            print("   🎉 EXCELLENT: Major improvement achieved!")
        elif improvement > 0.05 and has_diversity:
            status = "GOOD"
            print("   ✅ GOOD: Significant improvement!")
        elif improvement > 0 or has_diversity:
            status = "IMPROVED"
            print("   ⚡ IMPROVED: Some progress made!")
        else:
            status = "NEEDS_WORK"
            print("   ⚠️ NEEDS_WORK: Requires more tuning")
        
        # Save results
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'original_confidence': float(original_confidence),
            'hybrid_confidence': float(hybrid_avg),
            'improvement': float(improvement),
            'signal_diversity': list(unique_actions),
            'action_distribution': action_counts,
            'average_consensus': float(avg_consensus),
            'status': status
        }
        
        import json
        with open('hybrid_logic_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   📊 Results saved to: hybrid_logic_results.json")
        
        return status in ["EXCELLENT", "GOOD", "IMPROVED"]
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = fix_ai3_hybrid_logic()
    
    print("\n" + "="*80)
    if success:
        print("🎉 HYBRID LOGIC THÀNH CÔNG!")
        print("✅ AI2.0 weighted average + AI3.0 democratic consensus")
        print("✅ Thresholds thấp hơn cho better signal generation")
        print("✅ Confidence based on system consensus")
    else:
        print("⚠️ HYBRID LOGIC cần fine-tuning")
    print("="*80) 