#!/usr/bin/env python3
"""
🔧 FIX DOUBLE COLON ERROR - Sửa lỗi double colon
Sửa lỗi invalid syntax: pass:
"""

import re

def fix_double_colon():
    """Sửa lỗi double colon"""
    print("🔧 FIXING DOUBLE COLON ERROR")
    print("=" * 30)
    
    system_file = "src/core/ultimate_xau_system.py"
    
    with open(system_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixes = []
    
    # Fix common double colon issues
    patterns_to_fix = [
        (r'pass:', 'pass'),
        (r'break:', 'break'),
        (r'continue:', 'continue'),
        (r'return:', 'return'),
        (r'(\w+):\s*:', r'\1:'),  # Remove double colons
    ]
    
    for pattern, replacement in patterns_to_fix:
        matches = len(re.findall(pattern, content))
        if matches > 0:
            content = re.sub(pattern, replacement, content)
            fixes.append(f"Fixed {matches} instances of '{pattern}' → '{replacement}'")
    
    # Save fixed content
    with open(system_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if fixes:
        print("✅ Applied fixes:")
        for fix in fixes:
            print(f"   - {fix}")
    else:
        print("ℹ️ No double colon fixes needed")
    
    return len(fixes)

def test_syntax_again():
    """Test syntax lại"""
    print(f"\n🧪 TESTING SYNTAX AGAIN")
    print("-" * 25)
    
    try:
        import ast
        with open("src/core/ultimate_xau_system.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        print("✅ Syntax is now valid!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}:")
        print(f"   Text: {e.text}")
        print(f"   Error: {e.msg}")
        
        # Show context around error
        lines = content.split('\n')
        start = max(0, e.lineno - 3)
        end = min(len(lines), e.lineno + 2)
        
        print(f"   Context:")
        for i in range(start, end):
            marker = ">>> " if i == e.lineno - 1 else "    "
            print(f"   {marker}{i+1}: {lines[i]}")
        
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def main():
    """Main function"""
    
    print("🔧 DOUBLE COLON ERROR FIX")
    print("=" * 35)
    
    fixes = fix_double_colon()
    syntax_ok = test_syntax_again()
    
    print(f"\n📋 SUMMARY")
    print("-" * 15)
    print(f"🔧 Fixes applied: {fixes}")
    print(f"🧪 Syntax valid: {'✅ YES' if syntax_ok else '❌ NO'}")
    
    if syntax_ok:
        print("🎉 All syntax errors fixed!")
        
        # Test import
        try:
            print("\n🔄 Testing system import...")
            from core.ultimate_xau_system import UltimateXAUSystem
            print("✅ System import successful!")
            
            # Quick functionality test
            system = UltimateXAUSystem()
            print("✅ System initialization successful!")
            
        except Exception as e:
            print(f"⚠️ System test warning: {e}")
    
    return syntax_ok

if __name__ == "__main__":
    main() 