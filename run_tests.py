#!/usr/bin/env python3
"""
Test Runner Script
Ultimate XAU Super System V4.0
"""

import sys
import subprocess
import os
from datetime import datetime

def run_tests():
    """Run all test suites with reporting"""
    
    print("🧪 ULTIMATE XAU SUPER SYSTEM V4.0 - TEST EXECUTION")
    print("=" * 60)
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    test_suites = [
        ("Unit Tests", "tests/unit", "unit"),
        ("Integration Tests", "tests/integration", "integration"), 
        ("Performance Tests", "tests/performance", "performance"),
        ("Load Tests", "tests/load", "load"),
        ("E2E Tests", "tests/e2e", "e2e")
    ]
    
    results = {}
    
    for suite_name, test_path, marker in test_suites:
        print(f"
🔬 Running {suite_name}...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            f"-m", marker,
            "--verbose",
            f"--html=reports/{marker}_report.html",
            f"--junitxml=reports/{marker}_junit.xml"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ {suite_name} PASSED")
                results[suite_name] = "PASSED"
            else:
                print(f"   ❌ {suite_name} FAILED")
                results[suite_name] = "FAILED"
                print(f"   Error: {result.stderr}")
                
        except Exception as e:
            print(f"   ⚠️ {suite_name} ERROR: {e}")
            results[suite_name] = "ERROR"
    
    # Generate summary report
    generate_summary_report(results)
    
    return results

def generate_summary_report(results):
    """Generate test execution summary report"""
    
    report_content = f"""
# Test Execution Summary Report
Ultimate XAU Super System V4.0

**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Results Summary

| Test Suite | Status |
|------------|--------|
"""
    
    for suite, status in results.items():
        status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
        report_content += f"| {suite} | {status_icon} {status} |\n"
    
    passed_count = sum(1 for status in results.values() if status == "PASSED")
    total_count = len(results)
    success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
    
    report_content += f"""
## Summary Statistics

- **Total Test Suites:** {total_count}
- **Passed:** {passed_count}
- **Failed:** {total_count - passed_count}
- **Success Rate:** {success_rate:.1f}%

## Recommendations

"""
    
    if success_rate == 100:
        report_content += "🎉 All tests passed! System is ready for production."
    elif success_rate >= 80:
        report_content += "⚠️ Most tests passed. Review failed tests before production."
    else:
        report_content += "❌ Multiple test failures detected. System needs attention before production."
    
    with open("reports/test_summary.md", "w") as f:
        f.write(report_content)
    
    print(f"
📊 Test Summary Report generated: reports/test_summary.md")
    print(f"🎯 Overall Success Rate: {success_rate:.1f}%")

if __name__ == "__main__":
    run_tests()
