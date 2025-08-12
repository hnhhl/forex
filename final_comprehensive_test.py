#!/usr/bin/env python3
"""
🎯 FINAL COMPREHENSIVE TEST - Test cuối cùng toàn diện
Đánh giá chính xác hệ thống sau tất cả các sửa chữa
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append('src')

def comprehensive_final_test():
    """Test toàn diện cuối cùng"""
    print("🎯 FINAL COMPREHENSIVE SYSTEM TEST")
    print("=" * 50)
    print("🔍 Objective: Đánh giá chính xác hệ thống sau sửa chữa")
    print()
    
    test_results = {
        "test_timestamp": datetime.now().isoformat(),
        "system_status": {},
        "signal_generation": {},
        "performance_metrics": {},
        "error_analysis": {},
        "final_assessment": {}
    }
    
    try:
        # 1. System Initialization Test
        print("🚀 SYSTEM INITIALIZATION TEST")
        print("-" * 35)
        
        from core.ultimate_xau_system import UltimateXAUSystem
        
        start_time = time.time()
        system = UltimateXAUSystem()
        init_time = time.time() - start_time
        
        test_results["system_status"] = {
            "initialization_success": True,
            "initialization_time": init_time,
            "components_initialized": True
        }
        
        print(f"✅ System initialized in {init_time:.2f}s")
        
        # 2. Signal Generation Test
        print("\n📡 SIGNAL GENERATION TEST")
        print("-" * 30)
        
        # Create comprehensive test data
        test_data = pd.DataFrame({
            'open': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0, 2005.0, 2006.0, 2007.0, 2008.0, 2009.0],
            'high': [2005.0, 2006.0, 2007.0, 2008.0, 2009.0, 2010.0, 2011.0, 2012.0, 2013.0, 2014.0],
            'low': [1995.0, 1996.0, 1997.0, 1998.0, 1999.0, 2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
            'close': [2003.0, 2004.0, 2005.0, 2006.0, 2007.0, 2008.0, 2009.0, 2010.0, 2011.0, 2012.0],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Test multiple signals
        signals = []
        processing_times = []
        errors = []
        
        for i in range(20):  # Test 20 signals
            try:
                signal_start = time.time()
                signal = system.generate_signal(test_data)
                signal_time = time.time() - signal_start
                
                processing_times.append(signal_time)
                
                # Extract signal information
                signal_info = {
                    "test_id": i + 1,
                    "action": signal.get('action', 'UNKNOWN'),
                    "strength": signal.get('strength', 'UNKNOWN'),
                    "prediction": signal.get('prediction', 0),
                    "confidence": signal.get('confidence', 0),
                    "processing_time": signal_time,
                    "has_error": 'error' in signal,
                    "error_message": signal.get('error', None),
                    "systems_used": signal.get('systems_used', 0)
                }
                
                signals.append(signal_info)
                
                if signal_info["has_error"]:
                    errors.append(signal_info["error_message"])
                
            except Exception as e:
                errors.append(str(e))
                signals.append({
                    "test_id": i + 1,
                    "action": "ERROR",
                    "error": str(e),
                    "processing_time": 0
                })
        
        # Analyze signal results
        successful_signals = [s for s in signals if s.get("action") != "ERROR"]
        actions = [s["action"] for s in successful_signals]
        confidences = [s["confidence"] for s in successful_signals if s["confidence"] > 0]
        
        signal_distribution = {
            'BUY': actions.count('BUY'),
            'SELL': actions.count('SELL'),
            'HOLD': actions.count('HOLD'),
            'ERROR': len(signals) - len(successful_signals)
        }
        
        test_results["signal_generation"] = {
            "total_tests": len(signals),
            "successful_signals": len(successful_signals),
            "success_rate": len(successful_signals) / len(signals) * 100,
            "signal_distribution": signal_distribution,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "avg_processing_time": np.mean(processing_times) if processing_times else 0,
            "max_processing_time": max(processing_times) if processing_times else 0,
            "error_count": len(errors),
            "unique_errors": len(set(errors))
        }
        
        print(f"✅ Signal Generation Results:")
        print(f"   Success Rate: {test_results['signal_generation']['success_rate']:.1f}%")
        print(f"   Signal Distribution: {signal_distribution}")
        print(f"   Avg Confidence: {test_results['signal_generation']['avg_confidence']:.2f}%")
        print(f"   Avg Processing Time: {test_results['signal_generation']['avg_processing_time']:.3f}s")
        print(f"   Error Count: {test_results['signal_generation']['error_count']}")
        
        # 3. Performance Metrics
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 25)
        
        test_results["performance_metrics"] = {
            "initialization_time": init_time,
            "avg_signal_time": test_results["signal_generation"]["avg_processing_time"],
            "max_signal_time": test_results["signal_generation"]["max_processing_time"],
            "throughput_signals_per_second": 1 / test_results["signal_generation"]["avg_processing_time"] if test_results["signal_generation"]["avg_processing_time"] > 0 else 0
        }
        
        print(f"   Initialization Time: {init_time:.2f}s")
        print(f"   Avg Signal Time: {test_results['performance_metrics']['avg_signal_time']:.3f}s")
        print(f"   Max Signal Time: {test_results['performance_metrics']['max_signal_time']:.3f}s")
        print(f"   Throughput: {test_results['performance_metrics']['throughput_signals_per_second']:.1f} signals/sec")
        
        # 4. Error Analysis
        print("\n🔍 ERROR ANALYSIS")
        print("-" * 20)
        
        if errors:
            error_frequency = {}
            for error in errors:
                if error in error_frequency:
                    error_frequency[error] += 1
                else:
                    error_frequency[error] = 1
            
            test_results["error_analysis"] = {
                "total_errors": len(errors),
                "unique_errors": len(error_frequency),
                "error_frequency": error_frequency,
                "most_common_error": max(error_frequency.items(), key=lambda x: x[1]) if error_frequency else None
            }
            
            print(f"   Total Errors: {len(errors)}")
            print(f"   Unique Errors: {len(error_frequency)}")
            if test_results["error_analysis"]["most_common_error"]:
                most_common = test_results["error_analysis"]["most_common_error"]
                print(f"   Most Common: {most_common[0][:50]}... ({most_common[1]} times)")
        else:
            test_results["error_analysis"] = {
                "total_errors": 0,
                "unique_errors": 0,
                "error_frequency": {},
                "most_common_error": None
            }
            print("   🎉 No errors detected!")
        
        # 5. Final Assessment
        print("\n🎯 FINAL ASSESSMENT")
        print("-" * 25)
        
        # Calculate comprehensive score
        scores = {
            "initialization": 25 if test_results["system_status"]["initialization_success"] else 0,
            "signal_success": min(test_results["signal_generation"]["success_rate"] / 100 * 25, 25),
            "performance": 25 if test_results["performance_metrics"]["avg_signal_time"] < 0.1 else 20 if test_results["performance_metrics"]["avg_signal_time"] < 0.5 else 10,
            "reliability": 25 if test_results["error_analysis"]["total_errors"] == 0 else max(25 - test_results["error_analysis"]["total_errors"] * 2, 0)
        }
        
        # Bonus for confidence
        if test_results["signal_generation"]["avg_confidence"] > 30:
            scores["confidence_bonus"] = 10
        elif test_results["signal_generation"]["avg_confidence"] > 10:
            scores["confidence_bonus"] = 5
        else:
            scores["confidence_bonus"] = 0
        
        total_score = sum(scores.values())
        
        # Determine status
        if total_score >= 90:
            status = "EXCELLENT"
            status_emoji = "🟢"
        elif total_score >= 75:
            status = "GOOD"
            status_emoji = "🟡"
        elif total_score >= 60:
            status = "FAIR"
            status_emoji = "🟠"
        else:
            status = "POOR"
            status_emoji = "🔴"
        
        test_results["final_assessment"] = {
            "total_score": total_score,
            "status": status,
            "scores_breakdown": scores,
            "recommendation": f"{status} - System performance is {status.lower()}"
        }
        
        print(f"   {status_emoji} Overall Score: {total_score:.1f}/100 ({status})")
        print(f"   📊 Score Breakdown:")
        for component, score in scores.items():
            print(f"      {component.title()}: {score:.1f}")
        
        # 6. Detailed Signal Analysis
        if successful_signals:
            print(f"\n📈 DETAILED SIGNAL ANALYSIS")
            print("-" * 30)
            
            # Analyze signal patterns
            strengths = [s.get("strength", "UNKNOWN") for s in successful_signals]
            strength_dist = {
                'STRONG': strengths.count('STRONG'),
                'MODERATE': strengths.count('MODERATE'),
                'WEAK': strengths.count('WEAK'),
                'NEUTRAL': strengths.count('NEUTRAL')
            }
            
            print(f"   Signal Strength Distribution: {strength_dist}")
            
            # Confidence analysis
            if confidences:
                print(f"   Confidence Statistics:")
                print(f"      Average: {np.mean(confidences):.2f}%")
                print(f"      Median: {np.median(confidences):.2f}%")
                print(f"      Min: {min(confidences):.2f}%")
                print(f"      Max: {max(confidences):.2f}%")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        test_results["final_assessment"] = {
            "total_score": 0,
            "status": "FAILED",
            "error": str(e)
        }
    
    # Save results
    with open("final_comprehensive_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test results saved: final_comprehensive_test_results.json")
    
    return test_results

def main():
    """Main test function"""
    print("🎯 FINAL COMPREHENSIVE SYSTEM TEST")
    print("=" * 50)
    print("🔍 Testing system after all repairs...")
    print()
    
    results = comprehensive_final_test()
    
    print("\n" + "=" * 50)
    print("🏁 FINAL COMPREHENSIVE TEST COMPLETE")
    
    if results.get("final_assessment"):
        score = results["final_assessment"]["total_score"]
        status = results["final_assessment"]["status"]
        print(f"📊 Final Score: {score:.1f}/100 ({status})")
        
        if score >= 90:
            print("🎉 EXCELLENT! System is ready for production!")
        elif score >= 75:
            print("✅ GOOD! System is functional with minor issues!")
        elif score >= 60:
            print("⚠️ FAIR! System needs some improvements!")
        else:
            print("❌ POOR! System needs significant work!")
    
    return results

if __name__ == "__main__":
    main() 