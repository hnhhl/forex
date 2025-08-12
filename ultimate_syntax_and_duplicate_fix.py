#!/usr/bin/env python3
"""
🔧 ULTIMATE SYNTAX AND DUPLICATE FIX - Fix triệt để syntax error và duplicate methods
Giải quyết hoàn toàn các vấn đề syntax và duplicate để hệ thống hoạt động 100%
"""

import sys
import os
import re
import ast
from datetime import datetime
from typing import Dict, List, Set

sys.path.append('src')

class UltimateSyntaxFixer:
    """Class fix triệt để syntax và duplicate issues"""
    
    def __init__(self):
        self.system_file = "src/core/ultimate_xau_system.py"
        self.fixes_applied = []
        
    def create_emergency_backup(self):
        """Tạo backup khẩn cấp"""
        backup_file = f"{self.system_file}.emergency_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Emergency backup: {backup_file}")
        return backup_file
    
    def identify_syntax_error_location(self):
        """Xác định chính xác vị trí syntax error"""
        print("🔍 IDENTIFYING SYNTAX ERROR LOCATION")
        print("-" * 40)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        try:
            ast.parse(content)
            print("✅ No syntax errors found")
            return None
        except SyntaxError as e:
            print(f"❌ Syntax Error at line {e.lineno}: {e.msg}")
            print(f"📝 Error text: {e.text}")
            
            # Show context around error
            start_line = max(0, e.lineno - 5)
            end_line = min(len(lines), e.lineno + 5)
            
            print(f"\n📄 Context around line {e.lineno}:")
            for i in range(start_line, end_line):
                line_num = i + 1
                marker = ">>> " if line_num == e.lineno else "    "
                print(f"{marker}{line_num:4d}: {lines[i]}")
            
            return {
                'line_number': e.lineno,
                'error_message': e.msg,
                'error_text': e.text,
                'context_lines': lines[start_line:end_line]
            }
    
    def fix_syntax_error_line_560(self):
        """Fix cụ thể syntax error ở line 560"""
        print(f"\n🔧 FIXING SYNTAX ERROR AT LINE 560")
        print("-" * 35)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Check line 560 and surrounding lines
        if len(lines) > 560:
            problem_line = lines[559]  # 0-indexed
            print(f"Problem line 560: {problem_line}")
            
            # Common syntax fixes
            fixed_line = problem_line
            
            # Fix multiple assignment issues
            if '=' in problem_line and '==' not in problem_line:
                # Check for problematic patterns
                if re.search(r'(\w+)\s*=\s*(\w+)\s*=', problem_line):
                    print("🔧 Fixing multiple assignment")
                    # Split multiple assignments
                    parts = problem_line.split('=')
                    if len(parts) > 2:
                        # Take the last assignment
                        var_name = parts[0].strip()
                        value = '='.join(parts[1:]).strip()
                        fixed_line = f"        {var_name} = {value}"
            
            # Fix missing colons
            if re.search(r'(if|elif|else|try|except|finally|for|while|def|class)\s+.*[^:]$', problem_line.strip()):
                if not problem_line.strip().endswith(':'):
                    print("🔧 Adding missing colon")
                    fixed_line = problem_line.rstrip() + ':'
            
            # Fix indentation
            if problem_line.strip() and not problem_line.startswith(' ') and not problem_line.startswith('#'):
                if 'class ' not in problem_line and 'def ' not in problem_line and 'import ' not in problem_line:
                    print("🔧 Fixing indentation")
                    fixed_line = '    ' + problem_line.lstrip()
            
            # Apply fix
            if fixed_line != problem_line:
                lines[559] = fixed_line
                content = '\n'.join(lines)
                
                with open(self.system_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixes_applied.append(f"Fixed syntax error at line 560: '{problem_line}' → '{fixed_line}'")
                print(f"✅ Fixed line 560: {fixed_line}")
                return True
            else:
                print("⚠️ Could not auto-fix line 560")
                return False
        else:
            print("⚠️ File too short to have line 560")
            return False
    
    def remove_duplicate_methods(self):
        """Loại bỏ duplicate methods"""
        print(f"\n🔧 REMOVING DUPLICATE METHODS")
        print("-" * 30)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all method definitions with their positions
        method_pattern = r'^\s*def\s+(\w+)\s*\('
        methods_found = {}
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            match = re.match(method_pattern, line)
            if match:
                method_name = match.group(1)
                if method_name in methods_found:
                    methods_found[method_name].append(i)
                else:
                    methods_found[method_name] = [i]
        
        # Find duplicates
        duplicate_methods = {name: positions for name, positions in methods_found.items() if len(positions) > 1}
        
        if not duplicate_methods:
            print("✅ No duplicate methods found")
            return True
        
        print(f"🔍 Found {len(duplicate_methods)} duplicate methods:")
        for method, positions in duplicate_methods.items():
            print(f"   📝 {method}: lines {[p+1 for p in positions]}")
        
        # Remove duplicates (keep the first occurrence)
        lines_to_remove = set()
        
        for method_name, positions in duplicate_methods.items():
            # Keep first occurrence, remove others
            for pos in positions[1:]:
                # Find the end of the method
                method_start = pos
                method_end = self._find_method_end(lines, method_start)
                
                # Mark lines for removal
                for line_idx in range(method_start, method_end + 1):
                    lines_to_remove.add(line_idx)
                
                print(f"🗑️ Removing duplicate {method_name} at lines {method_start+1}-{method_end+1}")
        
        # Remove marked lines
        if lines_to_remove:
            new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
            new_content = '\n'.join(new_lines)
            
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            removed_count = len(lines_to_remove)
            self.fixes_applied.append(f"Removed {removed_count} duplicate method lines")
            print(f"✅ Removed {removed_count} duplicate lines")
            return True
        
        return False
    
    def _find_method_end(self, lines: List[str], start_idx: int) -> int:
        """Tìm dòng kết thúc của method"""
        method_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            
            # Empty lines or comments continue the method
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            # If we find a line with same or less indentation, method ends
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= method_indent and line.strip():
                return i - 1
        
        # If we reach end of file, method ends there
        return len(lines) - 1
    
    def fix_indentation_issues(self):
        """Fix indentation issues"""
        print(f"\n🔧 FIXING INDENTATION ISSUES")
        print("-" * 30)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        fixed_lines = []
        indentation_fixes = 0
        
        for i, line in enumerate(lines):
            if not line.strip():
                # Keep empty lines as is
                fixed_lines.append(line)
                continue
            
            if line.strip().startswith('#'):
                # Keep comments as is
                fixed_lines.append(line)
                continue
            
            # Check indentation
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            # Fix indentation to be multiple of 4
            if current_indent > 0 and current_indent % 4 != 0:
                # Round to nearest multiple of 4
                correct_indent = ((current_indent + 2) // 4) * 4
                fixed_line = ' ' * correct_indent + stripped
                fixed_lines.append(fixed_line)
                indentation_fixes += 1
            else:
                fixed_lines.append(line)
        
        if indentation_fixes > 0:
            new_content = '\n'.join(fixed_lines)
            with open(self.system_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.fixes_applied.append(f"Fixed indentation for {indentation_fixes} lines")
            print(f"✅ Fixed indentation for {indentation_fixes} lines")
            return True
        else:
            print("✅ No indentation issues found")
            return False
    
    def comprehensive_syntax_validation(self):
        """Validation syntax toàn diện"""
        print(f"\n🔍 COMPREHENSIVE SYNTAX VALIDATION")
        print("-" * 40)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        validation_results = {
            'syntax_valid': False,
            'issues_found': [],
            'warnings': []
        }
        
        try:
            # Parse the entire file
            ast.parse(content)
            validation_results['syntax_valid'] = True
            print("✅ Syntax validation: PASSED")
            
        except SyntaxError as e:
            validation_results['syntax_valid'] = False
            validation_results['issues_found'].append({
                'type': 'SyntaxError',
                'line': e.lineno,
                'message': e.msg,
                'text': e.text
            })
            print(f"❌ Syntax Error: Line {e.lineno} - {e.msg}")
            
        except Exception as e:
            validation_results['syntax_valid'] = False
            validation_results['issues_found'].append({
                'type': 'ParseError',
                'message': str(e)
            })
            print(f"❌ Parse Error: {e}")
        
        # Additional checks
        lines = content.split('\n')
        
        # Check for common issues
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Check for tabs mixed with spaces
            if '\t' in line and ' ' in line:
                validation_results['warnings'].append(f"Line {line_num}: Mixed tabs and spaces")
            
            # Check for trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                validation_results['warnings'].append(f"Line {line_num}: Trailing whitespace")
            
            # Check for very long lines
            if len(line) > 120:
                validation_results['warnings'].append(f"Line {line_num}: Very long line ({len(line)} chars)")
        
        print(f"📊 Validation Results:")
        print(f"   ✅ Syntax Valid: {validation_results['syntax_valid']}")
        print(f"   ⚠️ Issues Found: {len(validation_results['issues_found'])}")
        print(f"   💡 Warnings: {len(validation_results['warnings'])}")
        
        return validation_results
    
    def final_system_test_after_fixes(self):
        """Test hệ thống sau khi fix"""
        print(f"\n🧪 FINAL SYSTEM TEST AFTER FIXES")
        print("-" * 35)
        
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
            else:
                print("⚠️ Signal Generation: PARTIAL")
            
            # Test key methods
            key_methods = ['start_trading', 'stop_trading', 'emergency_stop', 'trading_loop']
            for method in key_methods:
                if hasattr(system, method):
                    print(f"✅ {method}: Available")
                else:
                    print(f"❌ {method}: Missing")
            
            return True
            
        except Exception as e:
            print(f"❌ System test failed: {e}")
            return False
    
    def run_ultimate_fix(self):
        """Chạy fix triệt để"""
        print("🔧 ULTIMATE SYNTAX AND DUPLICATE FIX")
        print("=" * 50)
        print("🎯 Objective: Fix triệt để syntax error và duplicate methods")
        print()
        
        # Create emergency backup
        self.create_emergency_backup()
        
        # Run fixes in sequence
        steps = [
            ("Identify Syntax Error", self.identify_syntax_error_location),
            ("Fix Line 560 Syntax", self.fix_syntax_error_line_560),
            ("Remove Duplicates", self.remove_duplicate_methods),
            ("Fix Indentation", self.fix_indentation_issues),
            ("Validate Syntax", self.comprehensive_syntax_validation),
            ("Final Test", self.final_system_test_after_fixes)
        ]
        
        results = {}
        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            try:
                result = step_func()
                if step_name == "Validate Syntax":
                    results[step_name] = "✅ PASSED" if result['syntax_valid'] else "❌ FAILED"
                else:
                    results[step_name] = "✅ SUCCESS" if result else "⚠️ PARTIAL"
            except Exception as e:
                results[step_name] = f"❌ ERROR: {e}"
                print(f"❌ Error in {step_name}: {e}")
        
        # Summary
        print(f"\n📋 ULTIMATE FIX SUMMARY")
        print("=" * 30)
        print(f"🔧 Fixes Applied: {len(self.fixes_applied)}")
        
        if self.fixes_applied:
            print(f"\n✅ FIXES APPLIED:")
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"   {i}. {fix}")
        
        print(f"\n🎯 STEP RESULTS:")
        for step, result in results.items():
            print(f"   {result} {step}")
        
        # Final status
        success_count = sum(1 for result in results.values() if "✅" in result)
        total_count = len(results)
        success_rate = success_count / total_count * 100
        
        print(f"\n📊 Overall Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            final_status = "🌟 HOÀN TOÀN THÀNH CÔNG"
        elif success_rate >= 80:
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
    fixer = UltimateSyntaxFixer()
    result = fixer.run_ultimate_fix()
    
    print(f"\n🎉 ULTIMATE FIX COMPLETED!")
    print(f"📊 Success Rate: {result['success_rate']:.1f}%")
    print(f"🎯 Status: {result['final_status']}")
    
    return result

if __name__ == "__main__":
    main() 