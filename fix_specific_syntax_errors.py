#!/usr/bin/env python3
"""
🔧 FIX SPECIFIC SYNTAX ERRORS - Sửa các lỗi syntax cụ thể
"""

import re

def fix_specific_errors():
    """Sửa các lỗi syntax cụ thể"""
    print("🔧 FIXING SPECIFIC SYNTAX ERRORS")
    print("=" * 35)
    
    system_file = "src/core/ultimate_xau_system.py"
    
    with open(system_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixes = []
    
    # 1. Fix lambda syntax errors
    lambda_fixes = [
        (r'lambda \[([^\]]+)\]', r'lambda: [\1]'),  # lambda [list] -> lambda: [list]
        (r'lambda \[([^\]]+)\]', r'lambda: [\1]'),  # lambda ["list"] -> lambda: ["list"]
    ]
    
    for pattern, replacement in lambda_fixes:
        matches = len(re.findall(pattern, content))
        if matches > 0:
            content = re.sub(pattern, replacement, content)
            fixes.append(f"Fixed {matches} lambda syntax errors")
    
    # 2. Fix method definition with parameters
    method_fixes = [
        (r'def __init__\(self, input_size = (\d+), ([^)]+)\)', r'def __init__(self, input_size=\1, \2)'),
        (r'def place_order\([^)]*\):\s*([^:]+) = ([^:]+):', r'def place_order(self, symbol: str, order_type: int, volume: float, price: float = 0.0, sl: float = 0.0, tp: float = 0.0) -> Dict:\n        """\1 = \2"""'),
    ]
    
    for pattern, replacement in method_fixes:
        matches = len(re.findall(pattern, content))
        if matches > 0:
            content = re.sub(pattern, replacement, content)
            fixes.append(f"Fixed {matches} method definition errors")
    
    # 3. Fix specific line 629 - lambda syntax
    if 'lambda [512, 256, 128, 64]' in content:
        content = content.replace('lambda [512, 256, 128, 64]', 'lambda: [512, 256, 128, 64]')
        fixes.append("Fixed line 629 lambda syntax")
    
    # 4. Fix other lambda list patterns
    other_lambda_patterns = [
        ('lambda [5, 10, 20, 50, 100, 200]', 'lambda: [5, 10, 20, 50, 100, 200]'),
        ('lambda [\'M1\', \'M5\', \'M15\', \'M30\', \'H1\', \'H4\', \'D1\', \'W1\']', 'lambda: [\'M1\', \'M5\', \'M15\', \'M30\', \'H1\', \'H4\', \'D1\', \'W1\']'),
        ('lambda ["email", "telegram", "discord"]', 'lambda: ["email", "telegram", "discord"]'),
    ]
    
    for old_pattern, new_pattern in other_lambda_patterns:
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            fixes.append(f"Fixed lambda pattern: {old_pattern}")
    
    # 5. Fix method definition issues
    # Fix place_order method specifically
    place_order_pattern = r'def place_order\(self, symbol: str, order_type: int, volume: float,:\s*price: float = 0\.0, sl float = 0\.0, tp float = 0\.0\) -> Dict'
    if re.search(place_order_pattern, content):
        content = re.sub(place_order_pattern, 'def place_order(self, symbol: str, order_type: int, volume: float, price: float = 0.0, sl: float = 0.0, tp: float = 0.0) -> Dict', content)
        fixes.append("Fixed place_order method definition")
    
    # 6. Fix __init__ method definitions with syntax errors
    init_pattern = r'def __init__\(self, input_size = (\d+), ([^)]+)\)([^:]*)$'
    matches = re.findall(init_pattern, content, re.MULTILINE)
    if matches:
        for match in matches:
            old_def = f'def __init__(self, input_size = {match[0]}, {match[1]}){match[2]}'
            new_def = f'def __init__(self, input_size={match[0]}, {match[1]}):'
            if old_def in content:
                content = content.replace(old_def, new_def)
                fixes.append(f"Fixed __init__ method definition")
    
    # Save fixed content
    with open(system_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if fixes:
        print("✅ Applied specific fixes:")
        for fix in fixes:
            print(f"   - {fix}")
    else:
        print("ℹ️ No specific fixes needed")
    
    return len(fixes)

def validate_syntax():
    """Validate syntax"""
    print(f"\n🧪 VALIDATING SYNTAX")
    print("-" * 25)
    
    try:
        import ast
        with open("src/core/ultimate_xau_system.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        print("✅ Syntax is valid!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}:")
        print(f"   Text: {e.text}")
        print(f"   Error: {e.msg}")
        
        # Show more context
        lines = content.split('\n')
        start = max(0, e.lineno - 5)
        end = min(len(lines), e.lineno + 3)
        
        print(f"   Context (lines {start+1}-{end}):")
        for i in range(start, end):
            marker = ">>> " if i == e.lineno - 1 else "    "
            print(f"   {marker}{i+1}: {lines[i]}")
        
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def main():
    """Main function"""
    print("🔧 SPECIFIC SYNTAX ERROR FIX")
    print("=" * 35)
    
    fixes = fix_specific_errors()
    syntax_ok = validate_syntax()
    
    print(f"\n📋 SUMMARY")
    print("-" * 15)
    print(f"🔧 Fixes applied: {fixes}")
    print(f"🧪 Syntax valid: {'✅ YES' if syntax_ok else '❌ NO'}")
    
    if syntax_ok:
        print("🎉 All syntax errors fixed!")
        
        # Test system
        try:
            print("\n🔄 Testing system...")
            from core.ultimate_xau_system import UltimateXAUSystem
            print("✅ System import successful!")
            
            system = UltimateXAUSystem()
            print("✅ System initialization successful!")
            
        except Exception as e:
            print(f"⚠️ System test warning: {e}")
    
    return syntax_ok

if __name__ == "__main__":
    main() 