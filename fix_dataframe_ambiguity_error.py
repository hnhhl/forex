#!/usr/bin/env python3
"""
🔧 FIX DATAFRAME AMBIGUITY ERROR - Sửa lỗi cuối cùng
Giải quyết "The truth value of a DataFrame is ambiguous" error
"""

import sys
import os
import re
import shutil
from datetime import datetime

def fix_dataframe_ambiguity():
    """Sửa lỗi DataFrame ambiguity trong signal generation"""
    print("🔧 FIXING DATAFRAME AMBIGUITY ERROR")
    print("=" * 45)
    print("🎯 Target: Fix final remaining issue for EXCELLENT score")
    print()
    
    system_file = "src/core/ultimate_xau_system.py"
    backup_file = f"ultimate_xau_system_backup_dataframe_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    # Create backup
    print("📦 Creating backup...")
    shutil.copy2(system_file, backup_file)
    print(f"✅ Backup created: {backup_file}")
    
    # Read current file
    with open(system_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixes_applied = []
    
    # Fix 1: DataFrame boolean operations
    print("\n🔧 Fix 1: DataFrame boolean operations")
    
    # Find and fix DataFrame conditions that cause ambiguity
    patterns_to_fix = [
        # Pattern 1: if dataframe_var:
        (r'if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'if not \1.empty:'),
        
        # Pattern 2: if not dataframe_var:
        (r'if\s+not\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'if \1.empty:'),
        
        # Pattern 3: dataframe and condition
        (r'([a-zA-Z_][a-zA-Z0-9_]*)\s+and\s+', r'\1.any() and '),
        
        # Pattern 4: dataframe or condition  
        (r'([a-zA-Z_][a-zA-Z0-9_]*)\s+or\s+', r'\1.any() or '),
    ]
    
    # Apply specific fixes for known problematic patterns
    original_content = content
    
    # Fix specific DataFrame conditions in signal generation
    if 'if data:' in content:
        content = content.replace('if data:', 'if not data.empty:')
        fixes_applied.append("Fixed 'if data:' condition")
    
    if 'if not data:' in content:
        content = content.replace('if not data:', 'if data.empty:')
        fixes_applied.append("Fixed 'if not data:' condition")
    
    if 'if features:' in content:
        content = content.replace('if features:', 'if not features.empty:')
        fixes_applied.append("Fixed 'if features:' condition")
    
    if 'if not features:' in content:
        content = content.replace('if not features:', 'if features.empty:')
        fixes_applied.append("Fixed 'if not features:' condition")
    
    # Fix DataFrame conditions in signal validation
    if 'if signal_data:' in content:
        content = content.replace('if signal_data:', 'if not signal_data.empty:')
        fixes_applied.append("Fixed 'if signal_data:' condition")
    
    # Fix 2: DataFrame comparison operations
    print("\n🔧 Fix 2: DataFrame comparison operations")
    
    # Fix DataFrame comparisons that might cause ambiguity
    comparison_fixes = [
        ('data > 0', 'data.any() > 0'),
        ('data < 0', 'data.any() < 0'),
        ('features > threshold', '(features > threshold).any()'),
        ('features < threshold', '(features < threshold).any()'),
    ]
    
    for old_pattern, new_pattern in comparison_fixes:
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            fixes_applied.append(f"Fixed DataFrame comparison: {old_pattern}")
    
    # Fix 3: DataFrame logical operations
    print("\n🔧 Fix 3: DataFrame logical operations")
    
    # Find and fix logical operations with DataFrames
    logical_fixes = [
        # Fix DataFrame in boolean context
        (r'if\s+(\w+_df)\s*:', r'if not \1.empty:'),
        (r'if\s+not\s+(\w+_df)\s*:', r'if \1.empty:'),
        
        # Fix DataFrame in and/or operations
        (r'(\w+_df)\s+and\s+', r'\1.any() and '),
        (r'(\w+_df)\s+or\s+', r'\1.any() or '),
    ]
    
    for pattern, replacement in logical_fixes:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            fixes_applied.append(f"Fixed logical operation with DataFrame: {pattern}")
    
    # Fix 4: Specific signal generation logic
    print("\n🔧 Fix 4: Signal generation logic")
    
    # Find generate_signal method and fix DataFrame issues
    signal_method_pattern = r'def generate_signal\(self, data\):(.*?)(?=def|\Z)'
    signal_match = re.search(signal_method_pattern, content, re.DOTALL)
    
    if signal_match:
        method_content = signal_match.group(1)
        original_method = method_content
        
        # Fix common DataFrame ambiguity patterns in signal generation
        method_fixes = [
            # Ensure data validation uses proper DataFrame methods
            ('if data:', 'if not data.empty:'),
            ('if not data:', 'if data.empty:'),
            ('len(data) == 0', 'data.empty'),
            ('len(data) > 0', 'not data.empty'),
            
            # Fix threshold comparisons
            ('signal > buy_threshold', '(signal > buy_threshold).any() if hasattr(signal, "any") else signal > buy_threshold'),
            ('signal < sell_threshold', '(signal < sell_threshold).any() if hasattr(signal, "any") else signal < sell_threshold'),
            
            # Fix DataFrame boolean operations
            ('and data', 'and not data.empty'),
            ('or data', 'or not data.empty'),
        ]
        
        for old_pattern, new_pattern in method_fixes:
            if old_pattern in method_content:
                method_content = method_content.replace(old_pattern, new_pattern)
                fixes_applied.append(f"Fixed in generate_signal: {old_pattern}")
        
        # Replace the method in content
        if method_content != original_method:
            content = content.replace(original_method, method_content)
    
    # Fix 5: Add DataFrame validation helper
    print("\n🔧 Fix 5: Add DataFrame validation helper")
    
    # Add helper method for safe DataFrame operations
    helper_method = '''
    def _safe_dataframe_check(self, df, check_type='not_empty'):
        """
        Safe DataFrame checking to avoid ambiguity errors
        """
        if df is None:
            return False
        
        if not hasattr(df, 'empty'):
            return bool(df)  # Not a DataFrame
        
        if check_type == 'not_empty':
            return not df.empty
        elif check_type == 'empty':
            return df.empty
        elif check_type == 'any':
            return df.any().any() if hasattr(df, 'any') else bool(df)
        else:
            return not df.empty
    '''
    
    # Insert helper method before generate_signal
    if 'def generate_signal(self, data):' in content and '_safe_dataframe_check' not in content:
        content = content.replace(
            'def generate_signal(self, data):',
            helper_method + '\n    def generate_signal(self, data):'
        )
        fixes_applied.append("Added _safe_dataframe_check helper method")
    
    # Fix 6: Update signal generation to use helper
    print("\n🔧 Fix 6: Update signal generation to use helper")
    
    # Replace problematic DataFrame checks with helper calls
    helper_replacements = [
        ('if not data.empty:', 'if self._safe_dataframe_check(data, "not_empty"):'),
        ('if data.empty:', 'if self._safe_dataframe_check(data, "empty"):'),
        ('if not features.empty:', 'if self._safe_dataframe_check(features, "not_empty"):'),
        ('if features.empty:', 'if self._safe_dataframe_check(features, "empty"):'),
    ]
    
    for old_pattern, new_pattern in helper_replacements:
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            fixes_applied.append(f"Updated to use helper: {old_pattern}")
    
    # Save the fixed file
    with open(system_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ DATAFRAME AMBIGUITY FIXES COMPLETED")
    print(f"📝 Total fixes applied: {len(fixes_applied)}")
    
    for i, fix in enumerate(fixes_applied, 1):
        print(f"   {i}. {fix}")
    
    print(f"\n💾 Backup saved: {backup_file}")
    print(f"🎯 Ready for validation!")
    
    return len(fixes_applied)

def main():
    """Main function"""
    print("🚀 CONTINUING SYSTEM REPAIR - FINAL PHASE")
    print("=" * 50)
    print("🎯 Objective: Fix DataFrame ambiguity for EXCELLENT score")
    print()
    
    fixes_count = fix_dataframe_ambiguity()
    
    print(f"\n🎉 FINAL REPAIR PHASE COMPLETED!")
    print(f"📊 Fixes applied: {fixes_count}")
    print("🔍 Ready for final validation!")
    
    return fixes_count

if __name__ == "__main__":
    main() 