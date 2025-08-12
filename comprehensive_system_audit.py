#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE SYSTEM AUDIT - Rà soát toàn diện hệ thống
Kiểm tra tính đồng nhất, logic consistency, và các vấn đề tiềm ẩn
"""

import sys
import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import ast

sys.path.append('src')

class SystemAuditor:
    """Class thực hiện audit toàn diện hệ thống"""
    
    def __init__(self):
        self.system_file = "src/core/ultimate_xau_system.py"
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
    def audit_code_consistency(self):
        """Kiểm tra tính nhất quán trong code"""
        print("🔍 AUDITING CODE CONSISTENCY")
        print("=" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. Kiểm tra method naming consistency
        method_patterns = re.findall(r'def (\w+)\(', content)
        
        # Check naming conventions
        inconsistent_naming = []
        for method in method_patterns:
            if not method.startswith('_') and method not in ['__init__', '__str__', '__repr__']:
                if not method.islower() or '_' not in method:
                    inconsistent_naming.append(method)
        
        if inconsistent_naming:
            self.warnings.append(f"Inconsistent method naming: {inconsistent_naming[:5]}")
        
        # 2. Kiểm tra confidence calculation patterns
        confidence_patterns = re.findall(r'confidence\s*=\s*([^;\n]+)', content)
        unique_patterns = set(confidence_patterns)
        
        print(f"📊 Found {len(confidence_patterns)} confidence assignments")
        print(f"📊 Unique patterns: {len(unique_patterns)}")
        
        # Check for hardcoded values
        hardcoded_confidence = []
        for pattern in unique_patterns:
            if re.search(r'\b(0\.0|0|25\.0|50\.0)\b', pattern):
                hardcoded_confidence.append(pattern)
        
        if hardcoded_confidence:
            self.warnings.append(f"Hardcoded confidence values found: {len(hardcoded_confidence)} patterns")
            print(f"   ⚠️ Hardcoded patterns: {hardcoded_confidence[:3]}")
        
        # 3. Kiểm tra prediction calculation patterns
        prediction_patterns = re.findall(r'prediction\s*=\s*([^;\n]+)', content)
        unique_pred_patterns = set(prediction_patterns)
        
        print(f"📊 Found {len(prediction_patterns)} prediction assignments")
        print(f"📊 Unique prediction patterns: {len(unique_pred_patterns)}")
        
        # 4. Kiểm tra error handling consistency
        try_blocks = len(re.findall(r'try:', content))
        except_blocks = len(re.findall(r'except', content))
        
        if try_blocks != except_blocks:
            self.issues.append(f"Mismatched try/except blocks: {try_blocks} try vs {except_blocks} except")
        
        print(f"📊 Error handling: {try_blocks} try blocks, {except_blocks} except blocks")
        
        return {
            'method_count': len(method_patterns),
            'confidence_patterns': len(unique_patterns),
            'prediction_patterns': len(unique_pred_patterns),
            'error_handling_balance': try_blocks == except_blocks
        }
    
    def audit_parameter_consistency(self):
        """Kiểm tra tính nhất quán của parameters"""
        print(f"\n🔧 AUDITING PARAMETER CONSISTENCY")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. Kiểm tra threshold values
        threshold_patterns = {
            'buy_threshold': re.findall(r'buy_threshold\s*=\s*([0-9.]+)', content),
            'sell_threshold': re.findall(r'sell_threshold\s*=\s*([0-9.]+)', content),
            'confidence_threshold': re.findall(r'confidence[_\w]*threshold\s*=\s*([0-9.]+)', content),
            'min_consensus': re.findall(r'min_consensus\s*=\s*([0-9.]+)', content)
        }
        
        print("📊 Threshold values found:")
        for name, values in threshold_patterns.items():
            unique_values = set(values)
            print(f"   {name}: {unique_values}")
            if len(unique_values) > 1:
                self.warnings.append(f"Inconsistent {name} values: {unique_values}")
        
        # 2. Kiểm tra default values
        default_patterns = {
            'default_confidence': re.findall(r'(?:confidence\s*=\s*|return\s+)([0-9.]+)(?:\s*#.*default)', content, re.IGNORECASE),
            'default_prediction': re.findall(r'(?:prediction\s*=\s*|return\s+)([0-9.]+)(?:\s*#.*default)', content, re.IGNORECASE),
            'fallback_values': re.findall(r'\.get\([\'"][\w_]+[\'"],\s*([0-9.]+)\)', content)
        }
        
        print(f"\n📊 Default values:")
        for name, values in default_patterns.items():
            unique_values = set(values)
            print(f"   {name}: {unique_values}")
        
        # 3. Kiểm tra range validation
        range_checks = {
            'confidence_min': re.findall(r'max\([^,]+,\s*([0-9.]+)\)', content),
            'confidence_max': re.findall(r'min\([^,]+,\s*([0-9.]+)\)', content),
            'prediction_bounds': re.findall(r'max\(0\.0,\s*min\(1\.0', content)
        }
        
        print(f"\n📊 Range validations:")
        for name, values in range_checks.items():
            if name != 'prediction_bounds':
                unique_values = set(values)
                print(f"   {name}: {unique_values}")
            else:
                print(f"   {name}: {len(values)} occurrences")
        
        return threshold_patterns
    
    def audit_method_signatures(self):
        """Kiểm tra consistency của method signatures"""
        print(f"\n🔍 AUDITING METHOD SIGNATURES")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract all method definitions
        method_pattern = r'def (\w+)\((.*?)\):'
        methods = re.findall(method_pattern, content, re.DOTALL)
        
        # Analyze common patterns
        data_methods = []
        confidence_methods = []
        prediction_methods = []
        
        for method_name, params in methods:
            if 'data' in params.lower():
                data_methods.append((method_name, params.replace('\n', ' ').strip()))
            if 'confidence' in method_name.lower():
                confidence_methods.append((method_name, params.replace('\n', ' ').strip()))
            if 'prediction' in method_name.lower() or 'predict' in method_name.lower():
                prediction_methods.append((method_name, params.replace('\n', ' ').strip()))
        
        print(f"📊 Methods accepting data: {len(data_methods)}")
        print(f"📊 Confidence-related methods: {len(confidence_methods)}")
        print(f"📊 Prediction-related methods: {len(prediction_methods)}")
        
        # Check for missing type hints
        methods_without_types = []
        for method_name, params in methods:
            if '->' not in params and method_name != '__init__':
                methods_without_types.append(method_name)
        
        if methods_without_types:
            self.warnings.append(f"Methods without return type hints: {len(methods_without_types)}")
            print(f"   ⚠️ Missing return types: {methods_without_types[:5]}")
        
        return {
            'total_methods': len(methods),
            'data_methods': len(data_methods),
            'confidence_methods': len(confidence_methods),
            'prediction_methods': len(prediction_methods)
        }
    
    def audit_data_flow_consistency(self):
        """Kiểm tra tính nhất quán của data flow"""
        print(f"\n🌊 AUDITING DATA FLOW CONSISTENCY")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. Kiểm tra DataFrame operations
        dataframe_ops = {
            'empty_checks': re.findall(r'\.empty\b', content),
            'shape_checks': re.findall(r'\.shape\b', content),
            'column_access': re.findall(r'\[[\'"][a-zA-Z_]+[\'\"]\]', content),
            'iloc_access': re.findall(r'\.iloc\[', content),
            'loc_access': re.findall(r'\.loc\[', content)
        }
        
        print("📊 DataFrame operations:")
        for op_type, occurrences in dataframe_ops.items():
            print(f"   {op_type}: {len(occurrences)}")
        
        # 2. Kiểm tra data validation patterns
        validation_patterns = {
            'none_checks': len(re.findall(r'if\s+\w+\s+is\s+None', content)),
            'empty_checks': len(re.findall(r'if\s+.*\.empty', content)),
            'length_checks': len(re.findall(r'if\s+len\(', content)),
            'type_checks': len(re.findall(r'isinstance\(', content))
        }
        
        print(f"\n📊 Data validation patterns:")
        for pattern, count in validation_patterns.items():
            print(f"   {pattern}: {count}")
        
        # 3. Kiểm tra error handling trong data processing
        data_error_handling = {
            'try_in_process': len(re.findall(r'def\s+process.*?try:', content, re.DOTALL)),
            'except_dataframe': len(re.findall(r'except.*DataFrame', content)),
            'except_generic': len(re.findall(r'except\s+Exception', content))
        }
        
        print(f"\n📊 Data error handling:")
        for pattern, count in data_error_handling.items():
            print(f"   {pattern}: {count}")
        
        return dataframe_ops
    
    def audit_system_integration(self):
        """Kiểm tra tính nhất quán của system integration"""
        print(f"\n🔗 AUDITING SYSTEM INTEGRATION")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. Tìm tất cả system classes
        system_classes = re.findall(r'class (\w+)(?:\(BaseSystem\))?:', content)
        print(f"📊 Found {len(system_classes)} system classes")
        
        # 2. Kiểm tra required methods trong BaseSystem
        required_methods = ['initialize', 'process', 'cleanup']
        system_method_compliance = {}
        
        for system_class in system_classes:
            if 'System' in system_class:
                methods_found = []
                for method in required_methods:
                    pattern = f'class {system_class}.*?def {method}'
                    if re.search(pattern, content, re.DOTALL):
                        methods_found.append(method)
                
                system_method_compliance[system_class] = methods_found
                if len(methods_found) != len(required_methods):
                    missing = set(required_methods) - set(methods_found)
                    self.warnings.append(f"{system_class} missing methods: {missing}")
        
        print(f"📊 System method compliance:")
        for system, methods in system_method_compliance.items():
            compliance = len(methods) / len(required_methods) * 100
            print(f"   {system}: {compliance:.1f}% ({len(methods)}/{len(required_methods)})")
        
        # 3. Kiểm tra return format consistency
        return_patterns = re.findall(r'return\s+\{[^}]*\}', content)
        print(f"📊 Found {len(return_patterns)} dictionary returns")
        
        # Check for consistent keys
        common_keys = ['prediction', 'confidence', 'timestamp', 'error']
        key_usage = {}
        for key in common_keys:
            key_usage[key] = len(re.findall(f'[\'\"]{key}[\'\"]\s*:', content))
        
        print(f"📊 Common return keys usage:")
        for key, count in key_usage.items():
            print(f"   {key}: {count} occurrences")
        
        return system_method_compliance
    
    def test_system_functionality(self):
        """Test thực tế functionality của hệ thống"""
        print(f"\n🧪 TESTING SYSTEM FUNCTIONALITY")
        print("-" * 35)
        
        try:
            from core.ultimate_xau_system import UltimateXAUSystem
            
            # Test 1: System initialization
            print("🔄 Testing system initialization...")
            system = UltimateXAUSystem()
            print("   ✅ System initialization successful")
            
            # Test 2: Method availability
            required_methods = ['_validate_confidence', '_safe_dataframe_check', 'generate_signal']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(system, method):
                    missing_methods.append(method)
            
            if missing_methods:
                self.issues.append(f"Missing critical methods: {missing_methods}")
                print(f"   ❌ Missing methods: {missing_methods}")
            else:
                print("   ✅ All critical methods present")
            
            # Test 3: Confidence validation
            print("🔄 Testing confidence validation...")
            test_values = [0, 0.0, None, 50, 150, -10, "75"]
            confidence_results = {}
            
            for value in test_values:
                try:
                    result = system._validate_confidence(value)
                    confidence_results[str(value)] = result
                    if result <= 0 or result > 100:
                        self.issues.append(f"Invalid confidence result: {value} -> {result}")
                except Exception as e:
                    self.issues.append(f"Confidence validation error for {value}: {e}")
            
            print(f"   ✅ Confidence validation results: {confidence_results}")
            
            # Test 4: Signal generation
            print("🔄 Testing signal generation...")
            try:
                signal = system.generate_signal()
                
                if isinstance(signal, dict):
                    required_keys = ['action', 'confidence', 'prediction']
                    missing_keys = [key for key in required_keys if key not in signal]
                    
                    if missing_keys:
                        self.issues.append(f"Signal missing keys: {missing_keys}")
                    
                    confidence = signal.get('confidence', 0)
                    if confidence <= 0:
                        self.issues.append(f"Signal confidence still zero: {confidence}")
                    else:
                        print(f"   ✅ Signal confidence: {confidence}")
                    
                    print(f"   ✅ Signal generated: {signal.get('action', 'UNKNOWN')}")
                else:
                    self.issues.append("Signal generation returned non-dict")
                    
            except Exception as e:
                self.issues.append(f"Signal generation failed: {e}")
                print(f"   ❌ Signal generation error: {e}")
            
            return True
            
        except Exception as e:
            self.issues.append(f"System import/initialization failed: {e}")
            print(f"   ❌ System test failed: {e}")
            return False
    
    def generate_recommendations(self):
        """Tạo recommendations dựa trên audit results"""
        print(f"\n💡 GENERATING RECOMMENDATIONS")
        print("-" * 35)
        
        # Dựa trên issues và warnings tìm thấy
        if any('confidence' in issue.lower() for issue in self.issues):
            self.recommendations.append("Implement comprehensive confidence validation across all methods")
        
        if any('missing' in issue.lower() for issue in self.issues):
            self.recommendations.append("Add missing methods to ensure system completeness")
        
        if any('inconsistent' in warning.lower() for warning in self.warnings):
            self.recommendations.append("Standardize naming conventions and parameter patterns")
        
        if any('hardcoded' in warning.lower() for warning in self.warnings):
            self.recommendations.append("Replace hardcoded values with configurable parameters")
        
        # General recommendations
        self.recommendations.extend([
            "Implement automated testing for all critical methods",
            "Add comprehensive logging for debugging and monitoring",
            "Create validation scripts for regular system health checks",
            "Document all configuration parameters and their valid ranges",
            "Implement performance monitoring for all system components"
        ])
        
        print("📋 Recommendations:")
        for i, rec in enumerate(self.recommendations, 1):
            print(f"   {i}. {rec}")
    
    def generate_audit_report(self):
        """Tạo báo cáo audit chi tiết"""
        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_issues': len(self.issues),
                'total_warnings': len(self.warnings),
                'total_recommendations': len(self.recommendations)
            },
            'issues': self.issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'status': 'CRITICAL' if len(self.issues) > 5 else 'WARNING' if len(self.warnings) > 3 else 'GOOD'
        }
        
        # Save report
        report_file = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report, report_file
    
    def run_comprehensive_audit(self):
        """Chạy audit toàn diện"""
        print("🔍 COMPREHENSIVE SYSTEM AUDIT")
        print("=" * 50)
        print("🎯 Objective: Kiểm tra tính đồng nhất và consistency của hệ thống")
        print()
        
        # Run all audit components
        code_consistency = self.audit_code_consistency()
        param_consistency = self.audit_parameter_consistency()
        method_signatures = self.audit_method_signatures()
        data_flow = self.audit_data_flow_consistency()
        integration = self.audit_system_integration()
        functionality_test = self.test_system_functionality()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Generate final report
        report, report_file = self.generate_audit_report()
        
        # Print summary
        print(f"\n📋 AUDIT SUMMARY")
        print("=" * 20)
        print(f"📊 Total Issues: {len(self.issues)}")
        print(f"⚠️ Total Warnings: {len(self.warnings)}")
        print(f"💡 Total Recommendations: {len(self.recommendations)}")
        print(f"📁 Report saved: {report_file}")
        print(f"🎯 Overall Status: {report['status']}")
        
        if self.issues:
            print(f"\n🚨 CRITICAL ISSUES:")
            for i, issue in enumerate(self.issues[:5], 1):
                print(f"   {i}. {issue}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS:")
            for i, warning in enumerate(self.warnings[:5], 1):
                print(f"   {i}. {warning}")
        
        return report

def main():
    """Main function"""
    auditor = SystemAuditor()
    report = auditor.run_comprehensive_audit()
    
    print(f"\n✅ COMPREHENSIVE AUDIT COMPLETED!")
    print(f"📊 Status: {report['status']}")
    
    if report['status'] == 'GOOD':
        print("🎉 System is in good condition!")
    elif report['status'] == 'WARNING':
        print("⚠️ System has some issues that should be addressed")
    else:
        print("🚨 System has critical issues requiring immediate attention")
    
    return report

if __name__ == "__main__":
    main() 