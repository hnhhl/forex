#!/usr/bin/env python3
"""
DEEP SYSTEM ANALYSIS
Phân tích chi tiết toàn bộ hệ thống để tìm điểm mất cân bằng dẫn đến lỗi
"""

import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
sys.path.append('src/core')

class DeepSystemAnalysis:
    def __init__(self):
        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'data_analysis': {},
            'model_analysis': {},
            'logic_analysis': {},
            'flow_analysis': {},
            'balance_issues': []
        }
    
    def analyze_complete_system(self):
        """Phân tích toàn bộ hệ thống một cách chi tiết"""
        print("🔬 DEEP SYSTEM ANALYSIS - COMPREHENSIVE INVESTIGATION")
        print("=" * 80)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            # Initialize system
            config = SystemConfig()
            config.symbol = "XAUUSDc"
            system = UltimateXAUSystem(config)
            
            print("✅ System initialized for deep analysis")
            
            # Step 1: Analyze Data Flow
            print("\n📊 STEP 1: DATA FLOW ANALYSIS")
            print("-" * 50)
            self.analyze_data_flow(system)
            
            # Step 2: Analyze Individual Components
            print("\n🧩 STEP 2: COMPONENT ANALYSIS")
            print("-" * 50)
            self.analyze_individual_components(system)
            
            # Step 3: Analyze Signal Generation Logic
            print("\n🎯 STEP 3: SIGNAL GENERATION LOGIC")
            print("-" * 50)
            self.analyze_signal_generation_logic(system)
            
            # Step 4: Analyze Ensemble Logic
            print("\n🔄 STEP 4: ENSEMBLE LOGIC ANALYSIS")
            print("-" * 50)
            self.analyze_ensemble_logic(system)
            
            # Step 5: Analyze Balance Issues
            print("\n⚖️ STEP 5: BALANCE ISSUES ANALYSIS")
            print("-" * 50)
            self.analyze_balance_issues(system)
            
            # Step 6: Generate Detailed Report
            print("\n📋 STEP 6: DETAILED REPORT")
            print("-" * 50)
            self.generate_detailed_report()
            
        except Exception as e:
            print(f"❌ Deep analysis failed: {e}")
            traceback.print_exc()
    
    def analyze_data_flow(self, system):
        """Phân tích luồng dữ liệu"""
        try:
            # Get raw data
            data = system._get_comprehensive_market_data("XAUUSDc")
            
            print(f"📈 Raw Data Analysis:")
            print(f"   • Total records: {len(data)}")
            print(f"   • Columns: {list(data.columns)}")
            print(f"   • Date range: {data['time'].min()} to {data['time'].max()}")
            
            # Check data quality
            print(f"\n🔍 Data Quality Check:")
            for col in ['open', 'high', 'low', 'close', 'tick_volume']:
                if col in data.columns:
                    null_count = data[col].isnull().sum()
                    zero_count = (data[col] == 0).sum()
                    print(f"   • {col}: {null_count} nulls, {zero_count} zeros")
            
            # Check price consistency
            price_issues = 0
            if 'high' in data.columns and 'low' in data.columns:
                invalid_hl = (data['high'] < data['low']).sum()
                price_issues += invalid_hl
                print(f"   • Invalid high/low: {invalid_hl} records")
            
            if 'open' in data.columns and 'close' in data.columns:
                if 'high' in data.columns and 'low' in data.columns:
                    invalid_ohlc = ((data['open'] > data['high']) | 
                                   (data['open'] < data['low']) |
                                   (data['close'] > data['high']) | 
                                   (data['close'] < data['low'])).sum()
                    price_issues += invalid_ohlc
                    print(f"   • Invalid OHLC: {invalid_ohlc} records")
            
            self.analysis_results['data_analysis'] = {
                'total_records': len(data),
                'columns': list(data.columns),
                'price_issues': price_issues,
                'data_quality_score': max(0, 100 - price_issues)
            }
            
            # Sample recent data for detailed analysis
            recent_data = data.tail(5)
            print(f"\n📋 Recent Data Sample:")
            for idx, row in recent_data.iterrows():
                print(f"   {row['time']}: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f}")
            
        except Exception as e:
            print(f"❌ Data flow analysis failed: {e}")
            self.analysis_results['data_analysis']['error'] = str(e)
    
    def analyze_individual_components(self, system):
        """Phân tích từng component riêng lẻ"""
        try:
            # Get market data for testing
            data = system._get_comprehensive_market_data("XAUUSDc")
            if data.empty:
                print("❌ No data available for component analysis")
                return
            
            # Test each system component
            components_results = {}
            
            print("🧪 Testing Individual Components:")
            
            for system_name, subsystem in system.system_manager.systems.items():
                try:
                    print(f"\n   Testing {system_name}...")
                    
                    # Process data through component
                    result = subsystem.process(data)
                    
                    if isinstance(result, dict):
                        prediction = result.get('prediction', 'N/A')
                        confidence = result.get('confidence', 'N/A')
                        
                        print(f"      ✅ {system_name}: prediction={prediction}, confidence={confidence}")
                        
                        components_results[system_name] = {
                            'status': 'success',
                            'prediction': prediction,
                            'confidence': confidence,
                            'result_type': type(result).__name__
                        }
                    else:
                        print(f"      ⚠️ {system_name}: Unexpected result type: {type(result)}")
                        components_results[system_name] = {
                            'status': 'warning',
                            'result_type': type(result).__name__,
                            'result': str(result)[:100]
                        }
                        
                except Exception as e:
                    print(f"      ❌ {system_name}: Error - {e}")
                    components_results[system_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            self.analysis_results['model_analysis'] = components_results
            
            # Analyze component predictions
            predictions = []
            confidences = []
            
            for comp_name, comp_result in components_results.items():
                if comp_result['status'] == 'success':
                    try:
                        pred = float(comp_result['prediction'])
                        conf = float(comp_result['confidence'])
                        predictions.append(pred)
                        confidences.append(conf)
                    except:
                        pass
            
            if predictions:
                print(f"\n📊 Component Predictions Summary:")
                print(f"   • Predictions range: {min(predictions):.3f} to {max(predictions):.3f}")
                print(f"   • Average prediction: {np.mean(predictions):.3f}")
                print(f"   • Prediction std: {np.std(predictions):.3f}")
                print(f"   • Confidences range: {min(confidences):.3f} to {max(confidences):.3f}")
                print(f"   • Average confidence: {np.mean(confidences):.3f}")
                
                # Check for bias
                if np.mean(predictions) > 0.6:
                    print("   ⚠️ BIAS DETECTED: Predictions skewed towards BUY")
                    self.analysis_results['balance_issues'].append("Prediction bias towards BUY")
                elif np.mean(predictions) < 0.4:
                    print("   ⚠️ BIAS DETECTED: Predictions skewed towards SELL")
                    self.analysis_results['balance_issues'].append("Prediction bias towards SELL")
                else:
                    print("   ✅ Predictions appear balanced")
                
                # Check confidence levels
                if np.mean(confidences) < 0.5:
                    print("   ⚠️ LOW CONFIDENCE: Average confidence too low")
                    self.analysis_results['balance_issues'].append("Low average confidence")
                
        except Exception as e:
            print(f"❌ Component analysis failed: {e}")
            self.analysis_results['model_analysis']['error'] = str(e)
    
    def analyze_signal_generation_logic(self, system):
        """Phân tích logic tạo signal chi tiết"""
        try:
            print("🎯 Signal Generation Deep Dive:")
            
            # Generate multiple signals and analyze
            signals = []
            for i in range(10):
                signal = system.generate_signal("XAUUSDc")
                signals.append(signal)
                time.sleep(0.1)  # Small delay
            
            # Analyze signal patterns
            actions = [s.get('action', 'UNKNOWN') for s in signals]
            confidences = [s.get('confidence', 0) for s in signals]
            strengths = [s.get('strength', 'UNKNOWN') for s in signals]
            
            print(f"\n📈 Signal Pattern Analysis (10 signals):")
            print(f"   • Actions: {dict(pd.Series(actions).value_counts())}")
            print(f"   • Strengths: {dict(pd.Series(strengths).value_counts())}")
            print(f"   • Confidence range: {min(confidences):.3f} to {max(confidences):.3f}")
            print(f"   • Average confidence: {np.mean(confidences):.3f}")
            
            # Deep dive into signal components
            if signals:
                sample_signal = signals[0]
                print(f"\n🔍 Sample Signal Deep Analysis:")
                print(f"   • Action: {sample_signal.get('action')}")
                print(f"   • Strength: {sample_signal.get('strength')}")
                print(f"   • Confidence: {sample_signal.get('confidence'):.3f}")
                print(f"   • Prediction: {sample_signal.get('prediction'):.3f}")
                print(f"   • Systems used: {sample_signal.get('systems_used')}")
                
                # Analyze hybrid metrics
                hybrid_metrics = sample_signal.get('hybrid_metrics', {})
                if hybrid_metrics:
                    print(f"\n⚖️ Hybrid Metrics Analysis:")
                    for key, value in hybrid_metrics.items():
                        print(f"   • {key}: {value}")
                    
                    # Check for logic issues
                    signal_strength = hybrid_metrics.get('signal_strength', 0)
                    consensus = hybrid_metrics.get('hybrid_consensus', 0)
                    
                    if abs(signal_strength) > 10:
                        print(f"   ⚠️ EXTREME SIGNAL STRENGTH: {signal_strength}")
                        self.analysis_results['balance_issues'].append(f"Extreme signal strength: {signal_strength}")
                    
                    if consensus > 0.9 and sample_signal.get('confidence', 0) < 0.6:
                        print(f"   ⚠️ LOGIC INCONSISTENCY: High consensus ({consensus:.2f}) but low confidence ({sample_signal.get('confidence', 0):.2f})")
                        self.analysis_results['balance_issues'].append("High consensus but low confidence")
                
                # Analyze voting results
                voting_results = sample_signal.get('voting_results', {})
                if voting_results:
                    print(f"\n🗳️ Voting Results Analysis:")
                    buy_votes = voting_results.get('buy_votes', 0)
                    sell_votes = voting_results.get('sell_votes', 0)
                    hold_votes = voting_results.get('hold_votes', 0)
                    
                    print(f"   • BUY votes: {buy_votes}")
                    print(f"   • SELL votes: {sell_votes}")
                    print(f"   • HOLD votes: {hold_votes}")
                    
                    total_votes = buy_votes + sell_votes + hold_votes
                    if total_votes > 0:
                        print(f"   • Vote distribution: BUY {buy_votes/total_votes:.1%}, SELL {sell_votes/total_votes:.1%}, HOLD {hold_votes/total_votes:.1%}")
                        
                        # Check for voting bias
                        if sell_votes > buy_votes * 2 and sell_votes > hold_votes * 2:
                            print("   ⚠️ VOTING BIAS: Strong SELL bias detected")
                            self.analysis_results['balance_issues'].append("Strong SELL voting bias")
                        elif buy_votes > sell_votes * 2 and buy_votes > hold_votes * 2:
                            print("   ⚠️ VOTING BIAS: Strong BUY bias detected")
                            self.analysis_results['balance_issues'].append("Strong BUY voting bias")
            
            self.analysis_results['logic_analysis'] = {
                'signal_consistency': len(set(actions)) == 1,  # All same action
                'confidence_consistency': max(confidences) - min(confidences) < 0.1,
                'average_confidence': np.mean(confidences),
                'dominant_action': max(set(actions), key=actions.count) if actions else None
            }
            
        except Exception as e:
            print(f"❌ Signal generation analysis failed: {e}")
            self.analysis_results['logic_analysis']['error'] = str(e)
    
    def analyze_ensemble_logic(self, system):
        """Phân tích logic ensemble chi tiết"""
        try:
            print("🔄 Ensemble Logic Deep Analysis:")
            
            # Get market data
            data = system._get_comprehensive_market_data("XAUUSDc")
            if data.empty:
                print("❌ No data for ensemble analysis")
                return
            
            # Process all systems to get components
            signal_components = system._process_all_systems(data)
            
            print(f"\n📊 Signal Components Analysis:")
            print(f"   • Total components: {len(signal_components)}")
            
            # Analyze each component
            predictions = []
            confidences = []
            weights = []
            
            for comp_name, comp_result in signal_components.items():
                if isinstance(comp_result, dict):
                    pred = comp_result.get('prediction')
                    conf = comp_result.get('confidence')
                    
                    if pred is not None and conf is not None:
                        try:
                            pred_val = float(pred)
                            conf_val = float(conf)
                            weight = system._get_system_weight(comp_name)
                            
                            predictions.append(pred_val)
                            confidences.append(conf_val)
                            weights.append(weight)
                            
                            print(f"   • {comp_name}: pred={pred_val:.3f}, conf={conf_val:.3f}, weight={weight:.3f}")
                            
                        except Exception as e:
                            print(f"   • {comp_name}: Error parsing values - {e}")
                    else:
                        print(f"   • {comp_name}: Missing prediction or confidence")
                else:
                    print(f"   • {comp_name}: Invalid result type: {type(comp_result)}")
            
            # Analyze ensemble calculations
            if predictions and weights:
                print(f"\n🧮 Ensemble Calculations:")
                
                # Weighted prediction
                weighted_pred = np.average(predictions, weights=weights)
                print(f"   • Weighted prediction: {weighted_pred:.3f}")
                
                # Signal strength calculation
                signal_strength = (weighted_pred - 0.5) * 2
                print(f"   • Signal strength: {signal_strength:.3f}")
                
                # Check for calculation issues
                if abs(signal_strength) > 1:
                    print(f"   ⚠️ CALCULATION ERROR: Signal strength out of range [-1, 1]: {signal_strength}")
                    self.analysis_results['balance_issues'].append(f"Signal strength out of range: {signal_strength}")
                
                # Weight analysis
                total_weight = sum(weights)
                print(f"   • Total weights: {total_weight:.3f}")
                
                if abs(total_weight - 1.0) > 0.1:
                    print(f"   ⚠️ WEIGHT IMBALANCE: Total weights != 1.0: {total_weight}")
                    self.analysis_results['balance_issues'].append(f"Weight imbalance: {total_weight}")
                
                # Prediction distribution
                print(f"\n📈 Prediction Distribution:")
                buy_predictions = sum(1 for p in predictions if p > 0.6)
                sell_predictions = sum(1 for p in predictions if p < 0.4)
                neutral_predictions = len(predictions) - buy_predictions - sell_predictions
                
                print(f"   • BUY predictions (>0.6): {buy_predictions}")
                print(f"   • SELL predictions (<0.4): {sell_predictions}")
                print(f"   • NEUTRAL predictions: {neutral_predictions}")
                
                # Check for prediction bias
                if sell_predictions > buy_predictions * 2:
                    print("   ⚠️ PREDICTION BIAS: Strong SELL bias in components")
                    self.analysis_results['balance_issues'].append("Component prediction SELL bias")
                elif buy_predictions > sell_predictions * 2:
                    print("   ⚠️ PREDICTION BIAS: Strong BUY bias in components")
                    self.analysis_results['balance_issues'].append("Component prediction BUY bias")
            
            # Test ensemble signal generation directly
            print(f"\n🎯 Direct Ensemble Test:")
            ensemble_signal = system._generate_ensemble_signal(signal_components)
            
            print(f"   • Final action: {ensemble_signal.get('action')}")
            print(f"   • Final confidence: {ensemble_signal.get('confidence'):.3f}")
            print(f"   • Final prediction: {ensemble_signal.get('prediction'):.3f}")
            
            # Check thresholds
            hybrid_metrics = ensemble_signal.get('hybrid_metrics', {})
            if hybrid_metrics:
                signal_strength = hybrid_metrics.get('signal_strength', 0)
                consensus = hybrid_metrics.get('hybrid_consensus', 0)
                
                print(f"   • Signal strength: {signal_strength:.3f}")
                print(f"   • Hybrid consensus: {consensus:.3f}")
                
                # Check threshold logic
                if abs(signal_strength) > 0.15 and consensus >= 0.6:
                    print("   • Should trigger STRONG signal")
                elif abs(signal_strength) > 0.08 and consensus >= 0.55:
                    print("   • Should trigger MODERATE signal")
                else:
                    print("   • Should trigger HOLD signal")
                
                actual_action = ensemble_signal.get('action')
                if actual_action == 'HOLD' and (abs(signal_strength) > 0.15 and consensus >= 0.6):
                    print("   ⚠️ THRESHOLD LOGIC ERROR: Should be STRONG but got HOLD")
                    self.analysis_results['balance_issues'].append("Threshold logic error")
            
            self.analysis_results['flow_analysis'] = {
                'components_count': len(signal_components),
                'valid_predictions': len(predictions),
                'weighted_prediction': weighted_pred if 'weighted_pred' in locals() else None,
                'signal_strength': signal_strength if 'signal_strength' in locals() else None,
                'final_action': ensemble_signal.get('action'),
                'final_confidence': ensemble_signal.get('confidence')
            }
            
        except Exception as e:
            print(f"❌ Ensemble logic analysis failed: {e}")
            self.analysis_results['flow_analysis']['error'] = str(e)
    
    def analyze_balance_issues(self, system):
        """Phân tích các vấn đề mất cân bằng"""
        try:
            print("⚖️ Balance Issues Deep Analysis:")
            
            # Check for systematic bias over multiple runs
            print(f"\n🔄 Systematic Bias Check (50 signals):")
            
            actions_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            confidences = []
            signal_strengths = []
            
            for i in range(50):
                signal = system.generate_signal("XAUUSDc")
                
                action = signal.get('action', 'UNKNOWN')
                if action in actions_count:
                    actions_count[action] += 1
                
                confidences.append(signal.get('confidence', 0))
                
                hybrid_metrics = signal.get('hybrid_metrics', {})
                if hybrid_metrics:
                    signal_strengths.append(hybrid_metrics.get('signal_strength', 0))
                
                if i % 10 == 9:
                    print(f"   Processed {i+1}/50 signals...")
            
            # Analyze systematic bias
            total_signals = sum(actions_count.values())
            if total_signals > 0:
                print(f"\n📊 Systematic Bias Analysis:")
                for action, count in actions_count.items():
                    percentage = count / total_signals * 100
                    print(f"   • {action}: {count} ({percentage:.1f}%)")
                    
                    if percentage > 80:
                        print(f"   ⚠️ CRITICAL BIAS: {action} dominates with {percentage:.1f}%")
                        self.analysis_results['balance_issues'].append(f"Critical {action} bias: {percentage:.1f}%")
                    elif percentage > 60:
                        print(f"   ⚠️ MODERATE BIAS: {action} bias detected: {percentage:.1f}%")
                        self.analysis_results['balance_issues'].append(f"Moderate {action} bias: {percentage:.1f}%")
            
            # Analyze confidence distribution
            if confidences:
                print(f"\n📈 Confidence Distribution Analysis:")
                avg_conf = np.mean(confidences)
                std_conf = np.std(confidences)
                min_conf = np.min(confidences)
                max_conf = np.max(confidences)
                
                print(f"   • Average: {avg_conf:.3f}")
                print(f"   • Std Dev: {std_conf:.3f}")
                print(f"   • Range: {min_conf:.3f} to {max_conf:.3f}")
                
                # Check for confidence issues
                if std_conf < 0.01:
                    print(f"   ⚠️ CONFIDENCE STUCK: Very low variance ({std_conf:.4f})")
                    self.analysis_results['balance_issues'].append(f"Confidence stuck at ~{avg_conf:.3f}")
                
                if avg_conf < 0.5:
                    print(f"   ⚠️ LOW CONFIDENCE: Average below 50%")
                    self.analysis_results['balance_issues'].append(f"Low average confidence: {avg_conf:.3f}")
            
            # Analyze signal strength distribution
            if signal_strengths:
                print(f"\n💪 Signal Strength Distribution:")
                avg_strength = np.mean(signal_strengths)
                std_strength = np.std(signal_strengths)
                min_strength = np.min(signal_strengths)
                max_strength = np.max(signal_strengths)
                
                print(f"   • Average: {avg_strength:.3f}")
                print(f"   • Std Dev: {std_strength:.3f}")
                print(f"   • Range: {min_strength:.3f} to {max_strength:.3f}")
                
                # Check for strength issues
                if std_strength < 0.01:
                    print(f"   ⚠️ STRENGTH STUCK: Very low variance ({std_strength:.4f})")
                    self.analysis_results['balance_issues'].append(f"Signal strength stuck at ~{avg_strength:.3f}")
                
                if abs(avg_strength) > 10:
                    print(f"   ⚠️ EXTREME STRENGTH: Average too high ({avg_strength:.3f})")
                    self.analysis_results['balance_issues'].append(f"Extreme signal strength: {avg_strength:.3f}")
            
        except Exception as e:
            print(f"❌ Balance analysis failed: {e}")
            self.analysis_results['balance_issues'].append(f"Balance analysis error: {str(e)}")
    
    def generate_detailed_report(self):
        """Tạo báo cáo chi tiết"""
        print("📋 COMPREHENSIVE SYSTEM DIAGNOSIS REPORT")
        print("=" * 80)
        
        # Summary of issues found
        total_issues = len(self.analysis_results['balance_issues'])
        print(f"\n🚨 TOTAL ISSUES FOUND: {total_issues}")
        
        if total_issues == 0:
            print("✅ No major issues detected")
        else:
            print("❌ Critical issues detected:")
            for i, issue in enumerate(self.analysis_results['balance_issues'], 1):
                print(f"   {i}. {issue}")
        
        # Data quality assessment
        data_analysis = self.analysis_results.get('data_analysis', {})
        if 'data_quality_score' in data_analysis:
            score = data_analysis['data_quality_score']
            print(f"\n📊 Data Quality Score: {score}/100")
            if score < 80:
                print("   ⚠️ Data quality issues may affect system performance")
        
        # Model analysis summary
        model_analysis = self.analysis_results.get('model_analysis', {})
        successful_components = sum(1 for comp in model_analysis.values() 
                                  if isinstance(comp, dict) and comp.get('status') == 'success')
        total_components = len(model_analysis)
        
        if total_components > 0:
            success_rate = successful_components / total_components * 100
            print(f"\n🧩 Component Success Rate: {successful_components}/{total_components} ({success_rate:.1f}%)")
            
            if success_rate < 80:
                print("   ⚠️ Multiple component failures detected")
        
        # Logic analysis summary
        logic_analysis = self.analysis_results.get('logic_analysis', {})
        if 'average_confidence' in logic_analysis:
            avg_conf = logic_analysis['average_confidence']
            print(f"\n🎯 Average Signal Confidence: {avg_conf:.1%}")
            
            if avg_conf < 0.6:
                print("   ⚠️ Low confidence may prevent trading")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if total_issues == 0:
            print("   ✅ System appears to be functioning correctly")
        else:
            print("   🔧 CRITICAL FIXES NEEDED:")
            
            # Specific recommendations based on issues found
            issues_text = ' '.join(self.analysis_results['balance_issues'])
            
            if 'bias' in issues_text.lower():
                print("   1. Fix prediction/voting bias - check model training data")
            
            if 'confidence stuck' in issues_text.lower():
                print("   2. Fix confidence calculation - likely formula error")
            
            if 'signal strength stuck' in issues_text.lower():
                print("   3. Fix signal strength calculation - check ensemble logic")
            
            if 'threshold logic error' in issues_text.lower():
                print("   4. Fix threshold logic - review decision criteria")
            
            if 'weight imbalance' in issues_text.lower():
                print("   5. Fix weight normalization - ensure weights sum to 1.0")
        
        # Save detailed report
        filename = f"deep_system_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed analysis saved to: {filename}")
        
        return self.analysis_results

def main():
    """Main analysis function"""
    print("🔬 DEEP SYSTEM ANALYSIS - FINDING ROOT CAUSES")
    print("=" * 80)
    print(f"🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    analyzer = DeepSystemAnalysis()
    results = analyzer.analyze_complete_system()
    
    print(f"\n🏁 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    main() 