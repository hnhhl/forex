#!/usr/bin/env python3
"""
🎯 FINAL SYNTAX COMPLETE FIX - Fix hoàn toàn syntax error cuối cùng
Giải quyết triệt để class definition và indentation issues
"""

import sys
import os
import re
import ast
from datetime import datetime

sys.path.append('src')

class FinalSyntaxFixer:
    """Class fix hoàn toàn syntax cuối cùng"""
    
    def __init__(self):
        self.system_file = "src/core/ultimate_xau_system.py"
        self.fixes_applied = []
        
    def create_final_backup(self):
        """Tạo backup cuối cùng"""
        backup_file = f"{self.system_file}.final_syntax_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Final backup: {backup_file}")
        return backup_file
    
    def fix_class_definition_issues(self):
        """Fix class definition issues"""
        print("🔧 FIXING CLASS DEFINITION ISSUES")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_lines = []
        fixes_count = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for class definition
            if re.match(r'^\s*class\s+\w+.*:?\s*$', line):
                fixed_lines.append(line)
                
                # Ensure class definition ends with colon
                if not line.rstrip().endswith(':'):
                    fixed_lines[-1] = line.rstrip() + ':'
                    fixes_count += 1
                    print(f"🔧 Added colon to class definition: {line.strip()}")
                
                # Check next line for proper indentation
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    
                    # If next line is empty, add a pass statement
                    if not next_line.strip():
                        # Look for the next non-empty line
                        j = i + 1
                        while j < len(lines) and not lines[j].strip():
                            fixed_lines.append(lines[j])
                            j += 1
                        
                        # If no proper content follows, add pass
                        if j >= len(lines) or not lines[j].startswith('    '):
                            fixed_lines.append('    pass')
                            fixes_count += 1
                            print(f"🔧 Added 'pass' to empty class")
                        
                        i = j - 1
                    
                    # If next line exists but not properly indented
                    elif next_line.strip() and not next_line.startswith('    '):
                        # Check if it's another class/function definition
                        if not re.match(r'^\s*(class|def|import|from)\s+', next_line):
                            # Add pass to complete the class
                            fixed_lines.append('    pass')
                            fixes_count += 1
                            print(f"🔧 Added 'pass' to complete class definition")
            else:
                fixed_lines.append(line)
            
            i += 1
        
        if fixes_count > 0:
            new_content = '\n'.join(fixed_lines)
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.fixes_applied.append(f"Fixed {fixes_count} class definition issues")
            print(f"✅ Fixed {fixes_count} class definition issues")
            return True
        else:
            print("✅ No class definition issues found")
            return False
    
    def fix_multiple_assignment_line_560(self):
        """Fix multiple assignment issue ở line 560"""
        print(f"\n🔧 FIXING MULTIPLE ASSIGNMENT LINE 560")
        print("-" * 40)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the specific multiple assignment pattern
        pattern = r'SYSTEM_NAME = "ULTIMATE_XAU_SUPER_SYSTEM": DEFAULT_SYMBOL = "XAUUSDc": DEFAULT_TIMEFRAME = mt5\.TIMEFRAME_M1'
        
        if re.search(pattern, content):
            replacement = '''SYSTEM_NAME = "ULTIMATE_XAU_SUPER_SYSTEM"
DEFAULT_SYMBOL = "XAUUSDc"
DEFAULT_TIMEFRAME = mt5.TIMEFRAME_M1'''
            
            content = re.sub(pattern, replacement, content)
            
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.fixes_applied.append("Fixed multiple assignment in line 560")
            print("✅ Fixed multiple assignment in line 560")
            return True
        else:
            print("✅ No multiple assignment issue found")
            return False
    
    def fix_empty_class_bodies(self):
        """Fix empty class bodies"""
        print(f"\n🔧 FIXING EMPTY CLASS BODIES")
        print("-" * 30)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_lines = []
        fixes_count = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            fixed_lines.append(line)
            
            # Check for class definition
            if re.match(r'^\s*class\s+\w+.*:\s*$', line):
                class_indent = len(line) - len(line.lstrip())
                
                # Look ahead to see if class has content
                j = i + 1
                has_content = False
                
                # Skip empty lines
                while j < len(lines) and not lines[j].strip():
                    j += 1
                
                # Check if next content is properly indented for this class
                if j < len(lines):
                    next_line = lines[j]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    # If next line is not indented more than class, class is empty
                    if next_indent <= class_indent:
                        has_content = False
                    else:
                        has_content = True
                
                # If class is empty, add pass
                if not has_content:
                    # Add empty line and pass
                    if i + 1 < len(lines) and lines[i + 1].strip():
                        fixed_lines.append('')
                    fixed_lines.append(' ' * (class_indent + 4) + 'pass')
                    fixes_count += 1
                    print(f"🔧 Added 'pass' to empty class at line {i+1}")
            
            i += 1
        
        if fixes_count > 0:
            new_content = '\n'.join(fixed_lines)
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.fixes_applied.append(f"Fixed {fixes_count} empty class bodies")
            print(f"✅ Fixed {fixes_count} empty class bodies")
            return True
        else:
            print("✅ No empty class bodies found")
            return False
    
    def fix_malformed_method_definitions(self):
        """Fix malformed method definitions"""
        print(f"\n🔧 FIXING MALFORMED METHOD DEFINITIONS")
        print("-" * 40)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_lines = []
        fixes_count = 0
        
        for i, line in enumerate(lines):
            # Check for malformed def statements
            if re.match(r'^\s*def\s+\w+.*[^:]\s*$', line.rstrip()):
                # Add missing colon
                fixed_line = line.rstrip() + ':'
                fixed_lines.append(fixed_line)
                fixes_count += 1
                print(f"🔧 Added colon to method definition at line {i+1}")
            else:
                fixed_lines.append(line)
        
        if fixes_count > 0:
            new_content = '\n'.join(fixed_lines)
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.fixes_applied.append(f"Fixed {fixes_count} malformed method definitions")
            print(f"✅ Fixed {fixes_count} malformed method definitions")
            return True
        else:
            print("✅ No malformed method definitions found")
            return False
    
    def comprehensive_syntax_check(self):
        """Comprehensive syntax check"""
        print(f"\n🔍 COMPREHENSIVE SYNTAX CHECK")
        print("-" * 30)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            ast.parse(content)
            print("✅ Syntax validation: PASSED")
            return True
        except SyntaxError as e:
            print(f"❌ Syntax Error: Line {e.lineno} - {e.msg}")
            print(f"📝 Error text: {e.text}")
            
            # Show context
            lines = content.split('\n')
            if e.lineno <= len(lines):
                start = max(0, e.lineno - 3)
                end = min(len(lines), e.lineno + 3)
                
                print(f"\n📄 Context around line {e.lineno}:")
                for i in range(start, end):
                    line_num = i + 1
                    marker = ">>> " if line_num == e.lineno else "    "
                    print(f"{marker}{line_num:4d}: {lines[i]}")
            
            return False
        except Exception as e:
            print(f"❌ Parse Error: {e}")
            return False
    
    def final_system_validation(self):
        """Final system validation"""
        print(f"\n🧪 FINAL SYSTEM VALIDATION")
        print("-" * 25)
        
        try:
            # Test import
            from core.ultimate_xau_system import UltimateXAUSystem
            print("✅ Import: SUCCESS")
            
            # Test initialization
            system = UltimateXAUSystem()
            print("✅ Initialization: SUCCESS")
            
            # Test signal generation
            signal = system.generate_signal()
            if isinstance(signal, dict) and 'action' in signal:
                print(f"✅ Signal Generation: SUCCESS - {signal.get('action')} ({signal.get('confidence')}%)")
                return True
            else:
                print("⚠️ Signal Generation: PARTIAL")
                return False
                
        except Exception as e:
            print(f"❌ System validation failed: {e}")
            return False
    
    def run_final_complete_fix(self):
        """Chạy fix hoàn toàn cuối cùng"""
        print("🎯 FINAL SYNTAX COMPLETE FIX")
        print("=" * 40)
        print("🎯 Objective: Fix hoàn toàn tất cả syntax errors")
        print()
        
        # Create final backup
        self.create_final_backup()
        
        # Run all fixes
        steps = [
            ("Fix Multiple Assignment", self.fix_multiple_assignment_line_560),
            ("Fix Class Definitions", self.fix_class_definition_issues),
            ("Fix Empty Class Bodies", self.fix_empty_class_bodies),
            ("Fix Method Definitions", self.fix_malformed_method_definitions),
            ("Syntax Check", self.comprehensive_syntax_check),
            ("System Validation", self.final_system_validation)
        ]
        
        results = {}
        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            try:
                result = step_func()
                results[step_name] = "✅ SUCCESS" if result else "⚠️ PARTIAL"
            except Exception as e:
                results[step_name] = f"❌ ERROR: {e}"
                print(f"❌ Error in {step_name}: {e}")
        
        # Summary
        print(f"\n📋 FINAL FIX SUMMARY")
        print("=" * 25)
        print(f"🔧 Fixes Applied: {len(self.fixes_applied)}")
        
        if self.fixes_applied:
            print(f"\n✅ FIXES APPLIED:")
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"   {i}. {fix}")
        
        print(f"\n🎯 STEP RESULTS:")
        for step, result in results.items():
            print(f"   {result} {step}")
        
        # Calculate success rate
        success_count = sum(1 for result in results.values() if "✅" in result)
        total_count = len(results)
        success_rate = success_count / total_count * 100
        
        print(f"\n📊 Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        if success_rate >= 95:
            final_status = "🌟 HOÀN TOÀN THÀNH CÔNG"
            print("🎉 HỆ THỐNG ĐÃ ĐƯỢC FIX TRIỆT ĐỂ!")
        elif success_rate >= 85:
            final_status = "✅ THÀNH CÔNG TỐT"
            print("✅ Hệ thống đã được fix tốt!")
        elif success_rate >= 70:
            final_status = "⚠️ THÀNH CÔNG PARTIAL"
            print("⚠️ Hệ thống được fix partial, cần check thêm")
        else:
            final_status = "❌ CẦN THÊM CÔNG VIỆC"
            print("❌ Cần thêm công việc để hoàn thiện")
        
        print(f"🎯 Final Status: {final_status}")
        
        return {
            'fixes_applied': self.fixes_applied,
            'step_results': results,
            'success_rate': success_rate,
            'final_status': final_status
        }

def main():
    """Main function"""
    fixer = FinalSyntaxFixer()
    result = fixer.run_final_complete_fix()
    
    print(f"\n🎯 FINAL SYNTAX FIX COMPLETED!")
    print(f"📊 Success Rate: {result['success_rate']:.1f}%")
    print(f"🎯 Status: {result['final_status']}")
    
    return result

if __name__ == "__main__":
    main() 