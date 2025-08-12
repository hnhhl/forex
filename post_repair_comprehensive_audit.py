#!/usr/bin/env python3
"""
🔍 POST-REPAIR COMPREHENSIVE AUDIT - Đánh giá kết quả sửa chữa
Kiểm tra toàn diện hệ thống sau khi sửa chữa triệt để
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

def comprehensive_post_repair_audit():
    """Audit toàn diện sau sửa chữa"""
    print("🔍 POST-REPAIR COMPREHENSIVE SYSTEM AUDIT")
    print("=" * 60)
    print("🎯 Objective: Đánh giá kết quả sửa chữa triệt để")
    print()
    
    audit_results = {
        "audit_timestamp": datetime.now().isoformat(),
        "audit_type": "Post-Repair Comprehensive",
        "system_status": {},
        "component_status": {},
        "signal_quality": {},
        "performance_metrics": {},
        "improvements": {},
        "remaining_issues": [],
        "overall_assessment": {}
    }
    
    try:
        # Import và khởi tạo hệ thống
        print("🚀 Initializing system...")
        from core.ultimate_xau_system import UltimateXAUSystem
        
        start_time = time.time()
        system = UltimateXAUSystem()
        init_time = time.time() - start_time
        
        print(f"✅ System initialized in {init_time:.2f}s")
        
        # 1. System Status Assessment
        print("\n📊 SYSTEM STATUS ASSESSMENT")
        print("-" * 40)
        
        audit_results["system_status"] = {
            "initialization_time": init_time,
            "initialization_success": True,
            "total_systems": len(system.systems),
            "active_systems": len([s for s in system.systems.values() if hasattr(s, 'active') and s.active]),
            "system_version": getattr(system, 'version', 'Unknown')
        }
        
        print(f"   Total Systems: {audit_results['system_status']['total_systems']}")
        print(f"   Active Systems: {audit_results['system_status']['active_systems']}")
        
        # 2. Component Status Check
        print("\n🔧 COMPONENT STATUS CHECK")
        print("-" * 40)
        
        component_status = {}
        critical_components = [
            'data_quality_monitor',
            'latency_optimizer',
            'mt5_connection_manager',
            'neural_network_system',
            'ai_phase_system',
            'ai2_advanced_technologies',
            'advanced_ai_ensemble',
            'realtime_mt5_data'
        ]
        
        for component in critical_components:
            if hasattr(system, component):
                comp_obj = getattr(system, component)
                status = "ACTIVE" if hasattr(comp_obj, 'active') and comp_obj.active else "INACTIVE"
                component_status[component] = {
                    "exists": True,
                    "status": status,
                    "has_connection_state": hasattr(comp_obj, 'connection_state') if component == 'mt5_connection_manager' else None
                }
                print(f"   ✅ {component}: {status}")
            else:
                component_status[component] = {
                    "exists": False,
                    "status": "MISSING"
                }
                print(f"   ❌ {component}: MISSING")
        
        audit_results["component_status"] = component_status
        
        # 3. Signal Generation Test
        print("\n📡 SIGNAL GENERATION TEST")
        print("-" * 40)
        
        # Tạo test data
        test_data = pd.DataFrame({
            'open': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
            'high': [2005.0, 2006.0, 2007.0, 2008.0, 2009.0],
            'low': [1995.0, 1996.0, 1997.0, 1998.0, 1999.0],
            'close': [2003.0, 2004.0, 2005.0, 2006.0, 2007.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        signal_results = []
        signal_errors = []
        
        # Test multiple signals
        for i in range(5):
            try:
                signal_start = time.time()
                signal = system.generate_signal(test_data)
                signal_time = time.time() - signal_start
                
                signal_results.append({
                    "test_id": i + 1,
                    "signal": signal.get('signal', 'N/A'),
                    "confidence": signal.get('confidence', 0),
                    "processing_time": signal_time,
                    "success": True
                })
                
                print(f"   Test {i+1}: {signal.get('signal', 'N/A')} (Confidence: {signal.get('confidence', 0):.2f}%, Time: {signal_time:.3f}s)")
                
            except Exception as e:
                signal_errors.append(str(e))
                signal_results.append({
                    "test_id": i + 1,
                    "error": str(e),
                    "success": False
                })
                print(f"   Test {i+1}: ERROR - {str(e)[:50]}...")
        
        # Signal Quality Analysis
        successful_signals = [s for s in signal_results if s.get('success', False)]
        if successful_signals:
            signals = [s['signal'] for s in successful_signals]
            confidences = [s['confidence'] for s in successful_signals]
            times = [s['processing_time'] for s in successful_signals]
            
            signal_distribution = {
                'BUY': signals.count('BUY'),
                'SELL': signals.count('SELL'),
                'HOLD': signals.count('HOLD')
            }
            
            audit_results["signal_quality"] = {
                "success_rate": len(successful_signals) / len(signal_results) * 100,
                "signal_distribution": signal_distribution,
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "avg_processing_time": np.mean(times) if times else 0,
                "max_processing_time": max(times) if times else 0,
                "errors": signal_errors
            }
            
            print(f"\n   📊 Signal Quality Summary:")
            print(f"      Success Rate: {audit_results['signal_quality']['success_rate']:.1f}%")
            print(f"      Signal Distribution: {signal_distribution}")
            print(f"      Avg Confidence: {audit_results['signal_quality']['avg_confidence']:.2f}%")
            print(f"      Avg Processing Time: {audit_results['signal_quality']['avg_processing_time']:.3f}s")
        
        # 4. Performance Metrics
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 40)
        
        audit_results["performance_metrics"] = {
            "initialization_time": init_time,
            "avg_signal_time": audit_results["signal_quality"]["avg_processing_time"] if "signal_quality" in audit_results else 0,
            "memory_usage": "Not measured",
            "cpu_usage": "Not measured"
        }
        
        print(f"   Initialization Time: {init_time:.2f}s")
        print(f"   Avg Signal Time: {audit_results['performance_metrics']['avg_signal_time']:.3f}s")
        
        # 5. Improvements Assessment
        print("\n📈 IMPROVEMENTS ASSESSMENT")
        print("-" * 40)
        
        improvements = []
        
        # Check MT5ConnectionManager fix
        if 'mt5_connection_manager' in component_status and component_status['mt5_connection_manager']['has_connection_state']:
            improvements.append("✅ MT5ConnectionManager connection_state fixed")
        
        # Check signal generation success
        if audit_results["signal_quality"]["success_rate"] > 0:
            improvements.append("✅ Signal generation working")
        
        # Check system initialization
        if audit_results["system_status"]["initialization_success"]:
            improvements.append("✅ System initialization successful")
        
        # Check component activation
        active_count = audit_results["system_status"]["active_systems"]
        total_count = audit_results["system_status"]["total_systems"]
        if active_count >= 7:  # At least 7/8 active
            improvements.append(f"✅ Component activation improved ({active_count}/{total_count})")
        
        audit_results["improvements"] = improvements
        
        for improvement in improvements:
            print(f"   {improvement}")
        
        # 6. Remaining Issues
        print("\n⚠️ REMAINING ISSUES")
        print("-" * 40)
        
        remaining_issues = []
        
        # Check for signal errors
        if signal_errors:
            remaining_issues.append(f"Signal generation errors: {len(signal_errors)} occurrences")
        
        # Check confidence levels
        if "signal_quality" in audit_results and audit_results["signal_quality"]["avg_confidence"] < 30:
            remaining_issues.append(f"Low confidence levels: {audit_results['signal_quality']['avg_confidence']:.2f}%")
        
        # Check signal bias
        if "signal_quality" in audit_results:
            dist = audit_results["signal_quality"]["signal_distribution"]
            total_signals = sum(dist.values())
            if total_signals > 0:
                buy_ratio = dist.get('BUY', 0) / total_signals
                if buy_ratio > 0.8:  # More than 80% BUY
                    remaining_issues.append(f"Signal bias towards BUY: {buy_ratio*100:.1f}%")
        
        # Check processing time
        if audit_results["performance_metrics"]["avg_signal_time"] > 0.3:  # More than 300ms
            remaining_issues.append(f"Slow processing time: {audit_results['performance_metrics']['avg_signal_time']*1000:.0f}ms")
        
        audit_results["remaining_issues"] = remaining_issues
        
        if remaining_issues:
            for issue in remaining_issues:
                print(f"   ⚠️ {issue}")
        else:
            print("   🎉 No major issues detected!")
        
        # 7. Overall Assessment
        print("\n🎯 OVERALL ASSESSMENT")
        print("-" * 40)
        
        # Calculate overall score
        score_components = {
            "initialization": 20 if audit_results["system_status"]["initialization_success"] else 0,
            "component_activation": (active_count / total_count) * 20 if total_count > 0 else 0,
            "signal_generation": (audit_results["signal_quality"]["success_rate"] / 100) * 20 if "signal_quality" in audit_results else 0,
            "signal_quality": min(audit_results["signal_quality"]["avg_confidence"] / 30 * 20, 20) if "signal_quality" in audit_results else 0,
            "performance": 20 if audit_results["performance_metrics"]["avg_signal_time"] < 0.2 else 10
        }
        
        overall_score = sum(score_components.values())
        
        # Determine status
        if overall_score >= 80:
            status = "EXCELLENT"
            status_emoji = "🟢"
        elif overall_score >= 65:
            status = "GOOD"
            status_emoji = "🟡"
        elif overall_score >= 50:
            status = "FAIR"
            status_emoji = "🟠"
        else:
            status = "POOR"
            status_emoji = "🔴"
        
        audit_results["overall_assessment"] = {
            "overall_score": overall_score,
            "status": status,
            "score_components": score_components,
            "improvements_count": len(improvements),
            "remaining_issues_count": len(remaining_issues)
        }
        
        print(f"   {status_emoji} Overall Score: {overall_score:.1f}/100 ({status})")
        print(f"   📈 Improvements: {len(improvements)}")
        print(f"   ⚠️ Remaining Issues: {len(remaining_issues)}")
        
        # Score breakdown
        print(f"\n   📊 Score Breakdown:")
        for component, score in score_components.items():
            print(f"      {component.title()}: {score:.1f}/20")
        
    except Exception as e:
        print(f"❌ Audit failed: {e}")
        audit_results["error"] = str(e)
        audit_results["overall_assessment"] = {
            "overall_score": 0,
            "status": "FAILED",
            "error": str(e)
        }
    
    # Save audit results
    with open("post_repair_audit_results.json", 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Audit results saved: post_repair_audit_results.json")
    
    return audit_results

def compare_with_previous_audit():
    """So sánh với audit trước đó"""
    print("\n🔄 COMPARING WITH PREVIOUS AUDIT")
    print("-" * 40)
    
    try:
        # Load previous audit
        if os.path.exists("comprehensive_system_audit_results.json"):
            with open("comprehensive_system_audit_results.json", 'r', encoding='utf-8') as f:
                previous = json.load(f)
            
            # Load current audit
            if os.path.exists("post_repair_audit_results.json"):
                with open("post_repair_audit_results.json", 'r', encoding='utf-8') as f:
                    current = json.load(f)
                
                # Compare scores
                prev_score = previous.get("overall_assessment", {}).get("overall_score", 0)
                curr_score = current.get("overall_assessment", {}).get("overall_score", 0)
                
                improvement = curr_score - prev_score
                
                print(f"   Previous Score: {prev_score:.1f}/100")
                print(f"   Current Score: {curr_score:.1f}/100")
                print(f"   Improvement: {improvement:+.1f} points")
                
                if improvement > 0:
                    print("   📈 System performance IMPROVED!")
                elif improvement == 0:
                    print("   ➡️ System performance UNCHANGED")
                else:
                    print("   📉 System performance DECLINED")
                
                return improvement
        else:
            print("   ⚠️ No previous audit found for comparison")
            return None
            
    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")
        return None

def main():
    """Main audit function"""
    audit_results = comprehensive_post_repair_audit()
    improvement = compare_with_previous_audit()
    
    print("\n" + "=" * 60)
    print("🎯 POST-REPAIR AUDIT COMPLETE")
    
    if audit_results.get("overall_assessment"):
        score = audit_results["overall_assessment"]["overall_score"]
        status = audit_results["overall_assessment"]["status"]
        print(f"📊 Final Score: {score:.1f}/100 ({status})")
        
        if improvement is not None:
            print(f"📈 Improvement: {improvement:+.1f} points")
    
    return audit_results

if __name__ == "__main__":
    main() 