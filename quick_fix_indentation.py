#!/usr/bin/env python3
"""
🔧 QUICK FIX INDENTATION ERROR
Sửa nhanh lỗi indentation ở line 3401 trong ultimate_xau_system.py
"""

import shutil
from datetime import datetime

def fix_indentation_error():
    """Sửa lỗi indentation ở line 3401"""
    print("🔧 FIXING INDENTATION ERROR")
    print("=" * 30)
    
    system_file = "src/core/ultimate_xau_system.py"
    backup_file = f"ultimate_xau_system_backup_indent_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    # Create backup
    shutil.copy2(system_file, backup_file)
    print(f"📦 Backup created: {backup_file}")
    
    with open(system_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix the problematic area around line 3401
    fixed_lines = []
    in_problematic_area = False
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Detect problematic area
        if line_num >= 3395 and line_num <= 3410:
            in_problematic_area = True
            
            # Fix specific issues
            if line_num == 3401 and line.strip() == "if not predictions:":
                # Add proper indentation and code block
                fixed_lines.append("            if not predictions:\n")
                fixed_lines.append("                return self._create_neutral_signal()\n")
                fixed_lines.append("            \n")
                continue
            elif line_num == 3402 and "# Calculate confidence" in line:
                # Skip this line as it's duplicated
                continue
            elif line_num in [3403, 3404, 3405, 3406, 3407, 3408] and ("if prediction" in line or "base_confidence" in line or "strength_multiplier" in line or "confidence =" in line or "else:" in line):
                # Skip these lines as they're duplicated
                continue
            elif line_num == 3409 and line.strip() == "":
                # Skip empty line
                continue
            elif line_num == 3410 and "return self._create_neutral_signal()" in line:
                # Skip this line as it's already added
                continue
        
        # Add the line if not skipped
        fixed_lines.append(line)
    
    # Write fixed content
    with open(system_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed indentation error")
    print("✅ Removed duplicate code blocks")
    
    return True

def test_syntax():
    """Test syntax after fix"""
    print("\n🧪 TESTING SYNTAX AFTER FIX")
    print("-" * 30)
    
    try:
        import sys
        sys.path.append('src')
        from core.ultimate_xau_system import UltimateXAUSystem
        
        print("✅ No syntax errors - import successful")
        
        # Test initialization
        system = UltimateXAUSystem()
        print("✅ System initialization successful")
        
        # Test confidence validation
        if hasattr(system, '_validate_confidence'):
            test_confidence = system._validate_confidence(50.0)
            print(f"✅ Confidence validation works: 50.0 -> {test_confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Syntax error still exists: {e}")
        return False

def main():
    """Main function"""
    print("🔧 QUICK INDENTATION FIX")
    print("=" * 25)
    
    # Fix indentation
    fix_success = fix_indentation_error()
    
    # Test syntax
    if fix_success:
        test_success = test_syntax()
        
        if test_success:
            print(f"\n🎉 INDENTATION FIX SUCCESSFUL!")
            print(f"   • Syntax errors resolved")
            print(f"   • System can be imported")
            print(f"   • Ready for confidence testing")
        else:
            print(f"\n⚠️ Additional syntax issues may remain")
    
    return fix_success

if __name__ == "__main__":
    main() 