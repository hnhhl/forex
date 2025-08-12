#!/usr/bin/env python3
"""
🔧 FIX DOUBLE COLON FINAL - Fix cụ thể vấn đề double colon
Giải quyết triệt để vấn đề double colon (::) trong code
"""

import sys
import os
import re
from datetime import datetime

sys.path.append('src')

class DoubleColonFixer:
    """Class fix double colon issues"""
    
    def __init__(self):
        self.system_file = "src/core/ultimate_xau_system.py"
        self.fixes_applied = []
        
    def create_colon_backup(self):
        """Tạo backup cho colon fix"""
        backup_file = f"{self.system_file}.colon_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Colon backup: {backup_file}")
        return backup_file
    
    def fix_double_colons(self):
        """Fix tất cả double colons"""
        print("🔧 FIXING DOUBLE COLONS")
        print("-" * 25)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count double colons
        double_colon_count = content.count('::')
        print(f"🔍 Found {double_colon_count} double colons")
        
        if double_colon_count == 0:
            print("✅ No double colons found")
            return True
        
        # Fix double colons - replace :: with :
        fixed_content = content.replace('::', ':')
        
        # Additional patterns to fix
        patterns_to_fix = [
            (r':\s*:', ':'),  # : followed by space and :
            (r'pass:', 'pass'),  # pass followed by colon
            (r'return\s+[^:]+:', lambda m: m.group(0)[:-1]),  # return statement with colon
        ]
        
        for pattern, replacement in patterns_to_fix:
            if callable(replacement):
                fixed_content = re.sub(pattern, replacement, fixed_content)
            else:
                fixed_content = re.sub(pattern, replacement, fixed_content)
        
        # Write fixed content
        with open(self.system_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        self.fixes_applied.append(f"Fixed {double_colon_count} double colons")
        print(f"✅ Fixed {double_colon_count} double colons")
        return True
    
    def fix_method_definition_colons(self):
        """Fix method definition colons specifically"""
        print(f"\n🔧 FIXING METHOD DEFINITION COLONS")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_lines = []
        fixes_count = 0
        
        for i, line in enumerate(lines):
            # Check for method definitions with issues
            if re.match(r'^\s*def\s+\w+.*', line):
                # Pattern: def method(): pass:
                if re.match(r'^\s*def\s+\w+.*:\s*pass:\s*$', line):
                    fixed_line = re.sub(r':\s*pass:\s*$', ': pass', line)
                    fixed_lines.append(fixed_line)
                    fixes_count += 1
                    print(f"🔧 Fixed method definition at line {i+1}")
                
                # Pattern: def method(): return something:
                elif re.match(r'^\s*def\s+\w+.*:\s*return.*:\s*$', line):
                    fixed_line = re.sub(r':\s*$', '', line)
                    fixed_lines.append(fixed_line)
                    fixes_count += 1
                    print(f"🔧 Fixed method return at line {i+1}")
                
                # Pattern: def method()::
                elif line.rstrip().endswith('::'):
                    fixed_line = line.rstrip()[:-1]  # Remove one colon
                    fixed_lines.append(fixed_line)
                    fixes_count += 1
                    print(f"🔧 Fixed double colon at line {i+1}")
                
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        if fixes_count > 0:
            new_content = '\n'.join(fixed_lines)
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.fixes_applied.append(f"Fixed {fixes_count} method definition colons")
            print(f"✅ Fixed {fixes_count} method definition colons")
            return True
        else:
            print("✅ No method definition colon issues found")
            return False
    
    def fix_class_definition_colons(self):
        """Fix class definition colons"""
        print(f"\n🔧 FIXING CLASS DEFINITION COLONS")
        print("-" * 30)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_lines = []
        fixes_count = 0
        
        for i, line in enumerate(lines):
            # Check for class definitions with double colons
            if re.match(r'^\s*class\s+\w+.*', line):
                if line.rstrip().endswith('::'):
                    fixed_line = line.rstrip()[:-1]  # Remove one colon
                    fixed_lines.append(fixed_line)
                    fixes_count += 1
                    print(f"🔧 Fixed class double colon at line {i+1}")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        if fixes_count > 0:
            new_content = '\n'.join(fixed_lines)
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.fixes_applied.append(f"Fixed {fixes_count} class definition colons")
            print(f"✅ Fixed {fixes_count} class definition colons")
            return True
        else:
            print("✅ No class definition colon issues found")
            return False
    
    def fix_pass_statement_colons(self):
        """Fix pass statement colons"""
        print(f"\n🔧 FIXING PASS STATEMENT COLONS")
        print("-" * 30)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix pass: patterns
        pass_patterns = [
            (r'\bpass:\s*$', 'pass'),  # pass: at end of line
            (r'\bpass:\s*\n', 'pass\n'),  # pass: followed by newline
        ]
        
        fixes_count = 0
        for pattern, replacement in pass_patterns:
            matches = len(re.findall(pattern, content, re.MULTILINE))
            if matches > 0:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                fixes_count += matches
                print(f"🔧 Fixed {matches} pass statement colons")
        
        if fixes_count > 0:
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.fixes_applied.append(f"Fixed {fixes_count} pass statement colons")
            print(f"✅ Fixed {fixes_count} pass statement colons")
            return True
        else:
            print("✅ No pass statement colon issues found")
            return False
    
    def fix_return_statement_colons(self):
        """Fix return statement colons"""
        print(f"\n🔧 FIXING RETURN STATEMENT COLONS")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix return: patterns
        return_patterns = [
            (r'\breturn\s+([^:\n]+):\s*$', r'return \1'),  # return value: at end of line
            (r'\breturn\s+([^:\n]+):\s*\n', r'return \1\n'),  # return value: followed by newline
        ]
        
        fixes_count = 0
        for pattern, replacement in return_patterns:
            matches = len(re.findall(pattern, content, re.MULTILINE))
            if matches > 0:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                fixes_count += matches
                print(f"🔧 Fixed {matches} return statement colons")
        
        if fixes_count > 0:
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.fixes_applied.append(f"Fixed {fixes_count} return statement colons")
            print(f"✅ Fixed {fixes_count} return statement colons")
            return True
        else:
            print("✅ No return statement colon issues found")
            return False
    
    def comprehensive_colon_check(self):
        """Comprehensive colon check"""
        print(f"\n🔍 COMPREHENSIVE COLON CHECK")
        print("-" * 30)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for remaining issues
        issues = []
        
        # Check for double colons
        if '::' in content:
            double_colon_lines = []
            for i, line in enumerate(content.split('\n')):
                if '::' in line:
                    double_colon_lines.append(i + 1)
            issues.append(f"Double colons found on lines: {double_colon_lines}")
        
        # Check for pass:
        if 'pass:' in content:
            pass_colon_lines = []
            for i, line in enumerate(content.split('\n')):
                if 'pass:' in line:
                    pass_colon_lines.append(i + 1)
            issues.append(f"'pass:' found on lines: {pass_colon_lines}")
        
        if issues:
            print("⚠️ Remaining colon issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ No colon issues found")
            return True
    
    def final_syntax_test(self):
        """Final syntax test"""
        print(f"\n🧪 FINAL SYNTAX TEST")
        print("-" * 20)
        
        try:
            import ast
            with open(self.system_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            print("✅ Syntax validation: PASSED")
            
            # Test import
            from core.ultimate_xau_system import UltimateXAUSystem
            print("✅ Import test: PASSED")
            
            # Test initialization
            system = UltimateXAUSystem()
            print("✅ Initialization test: PASSED")
            
            # Test signal generation
            signal = system.generate_signal()
            if isinstance(signal, dict) and 'action' in signal:
                print(f"✅ Signal test: PASSED - {signal.get('action')} ({signal.get('confidence')}%)")
                return True
            else:
                print("⚠️ Signal test: PARTIAL")
                return False
                
        except Exception as e:
            print(f"❌ Syntax test failed: {e}")
            return False
    
    def run_double_colon_fix(self):
        """Chạy fix double colon"""
        print("🔧 DOUBLE COLON FIX")
        print("=" * 30)
        print("🎯 Objective: Fix triệt để double colon issues")
        print()
        
        # Create backup
        self.create_colon_backup()
        
        # Run fixes
        steps = [
            ("Fix Double Colons", self.fix_double_colons),
            ("Fix Method Colons", self.fix_method_definition_colons),
            ("Fix Class Colons", self.fix_class_definition_colons),
            ("Fix Pass Colons", self.fix_pass_statement_colons),
            ("Fix Return Colons", self.fix_return_statement_colons),
            ("Colon Check", self.comprehensive_colon_check),
            ("Final Test", self.final_syntax_test)
        ]
        
        results = {}
        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            try:
                result = step_func()
                results[step_name] = "✅ SUCCESS" if result else "⚠️ PARTIAL"
            except Exception as e:
                results[step_name] = f"❌ ERROR: {e}"
                print(f"❌ Error: {e}")
        
        # Summary
        print(f"\n📋 DOUBLE COLON FIX SUMMARY")
        print("=" * 35)
        print(f"🔧 Fixes Applied: {len(self.fixes_applied)}")
        
        if self.fixes_applied:
            print(f"\n✅ FIXES APPLIED:")
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"   {i}. {fix}")
        
        print(f"\n🎯 STEP RESULTS:")
        for step, result in results.items():
            print(f"   {result} {step}")
        
        # Calculate success
        success_count = sum(1 for result in results.values() if "✅" in result)
        total_count = len(results)
        success_rate = success_count / total_count * 100
        
        print(f"\n📊 Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        if success_rate >= 95:
            final_status = "🌟 HOÀN TOÀN THÀNH CÔNG"
            print("🎉 DOUBLE COLON ISSUES ĐÃ ĐƯỢC FIX TRIỆT ĐỂ!")
        elif success_rate >= 85:
            final_status = "✅ THÀNH CÔNG TỐT"
        elif success_rate >= 70:
            final_status = "⚠️ THÀNH CÔNG PARTIAL"
        else:
            final_status = "❌ CẦN THÊM CÔNG VIỆC"
        
        print(f"🎯 Final Status: {final_status}")
        
        return {
            'fixes_applied': self.fixes_applied,
            'step_results': results,
            'success_rate': success_rate,
            'final_status': final_status
        }

def main():
    """Main function"""
    fixer = DoubleColonFixer()
    result = fixer.run_double_colon_fix()
    
    print(f"\n🎯 DOUBLE COLON FIX COMPLETED!")
    print(f"📊 Success Rate: {result['success_rate']:.1f}%")
    print(f"🎯 Status: {result['final_status']}")
    
    return result

if __name__ == "__main__":
    main() 