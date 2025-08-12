#!/usr/bin/env python3
"""
🔧 FIX SYNTAX ERROR - Sửa lỗi syntax error
Sửa lỗi indentation error on line 1495
"""

import sys
import os
import re
from datetime import datetime

sys.path.append('src')

def fix_syntax_error():
    """Sửa lỗi syntax error"""
    print("🔧 FIXING SYNTAX ERROR")
    print("=" * 25)
    
    system_file = "src/core/ultimate_xau_system.py"
    
    # Read file
    with open(system_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📄 File has {len(lines)} lines")
    
    # Check around line 1495-1496
    error_area = lines[1490:1500]
    print(f"📍 Lines around 1495:")
    for i, line in enumerate(error_area, 1491):
        print(f"   {i}: {repr(line)}")
    
    # Look for class definition without proper body
    fixes_applied = []
    
    for i, line in enumerate(lines):
        # Check for class definition followed by improper indentation
        if line.strip().startswith('class ') and line.strip().endswith(':'):
            # Check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # If next line is not properly indented or is empty without proper class body
                if next_line.strip() == '' or (not next_line.startswith('    ') and next_line.strip() != ''):
                    # Add proper class body
                    lines[i + 1] = '    """Auto-generated class body"""\n    pass\n\n'
                    fixes_applied.append(f"Fixed class definition at line {i + 1}")
                    print(f"✅ Fixed class definition at line {i + 1}")
    
    # Look for other common syntax issues
    for i, line in enumerate(lines):
        # Fix missing colons in if/for/while statements
        if re.match(r'^\s*(if|for|while|def|class)\s+.*[^:]$', line.strip()):
            if not line.strip().endswith(':'):
                lines[i] = line.rstrip() + ':\n'
                fixes_applied.append(f"Added missing colon at line {i + 1}")
                print(f"✅ Added missing colon at line {i + 1}")
    
    # Check for indentation issues around the error line
    if len(lines) > 1495:
        line_1495 = lines[1494]  # 0-indexed
        line_1496 = lines[1495] if len(lines) > 1495 else ""
        
        print(f"📍 Line 1495: {repr(line_1495)}")
        print(f"📍 Line 1496: {repr(line_1496)}")
        
        # If line 1495 is a class definition and line 1496 is not indented
        if line_1495.strip().endswith(':') and line_1496.strip() and not line_1496.startswith('    '):
            # Fix indentation
            lines[1495] = '    ' + line_1496.lstrip()
            fixes_applied.append("Fixed indentation at line 1496")
            print("✅ Fixed indentation at line 1496")
    
    # Write back the fixed file
    if fixes_applied:
        with open(system_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ Applied {len(fixes_applied)} syntax fixes:")
        for fix in fixes_applied:
            print(f"   - {fix}")
    else:
        print("ℹ️ No syntax fixes needed")
    
    return len(fixes_applied)

def test_syntax():
    """Test syntax bằng cách compile"""
    print(f"\n🧪 TESTING SYNTAX")
    print("-" * 20)
    
    try:
        import ast
        with open("src/core/ultimate_xau_system.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the file
        ast.parse(content)
        print("✅ Syntax is valid!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error still exists:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def main():
    """Main function"""
    print("🔧 SYNTAX ERROR FIX")
    print("=" * 30)
    
    fixes = fix_syntax_error()
    syntax_ok = test_syntax()
    
    print(f"\n📋 SUMMARY")
    print("-" * 15)
    print(f"🔧 Fixes applied: {fixes}")
    print(f"🧪 Syntax valid: {'✅ YES' if syntax_ok else '❌ NO'}")
    
    if syntax_ok:
        print("🎉 Syntax error fixed successfully!")
        
        # Test import
        try:
            print("\n🔄 Testing import...")
            from core.ultimate_xau_system import UltimateXAUSystem
            print("✅ Import successful!")
        except Exception as e:
            print(f"⚠️ Import warning: {e}")
    else:
        print("⚠️ Syntax error still exists. Manual inspection needed.")
    
    return syntax_ok

if __name__ == "__main__":
    main() 