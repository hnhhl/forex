#!/usr/bin/env python3
"""
🔍 DIAGNOSE DATAFRAME ERROR - Tìm chính xác vị trí lỗi
Chẩn đoán và sửa DataFrame ambiguity error một cách chính xác
"""

import sys
import os
import traceback
import pandas as pd
from datetime import datetime

sys.path.append('src')

def diagnose_dataframe_error():
    """Chẩn đoán chính xác lỗi DataFrame ambiguity"""
    print("🔍 DIAGNOSING DATAFRAME AMBIGUITY ERROR")
    print("=" * 50)
    print("🎯 Objective: Find exact location of DataFrame error")
    print()
    
    try:
        # Import system
        from core.ultimate_xau_system import UltimateXAUSystem
        
        print("✅ System import successful")
        
        # Initialize system
        system = UltimateXAUSystem()
        print("✅ System initialization successful")
        
        # Create test data
        test_data = pd.DataFrame({
            'open': [2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
            'high': [2005.0, 2006.0, 2007.0, 2008.0, 2009.0],
            'low': [1995.0, 1996.0, 1997.0, 1998.0, 1999.0],
            'close': [2003.0, 2004.0, 2005.0, 2006.0, 2007.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        print("✅ Test data created")
        print(f"   Data shape: {test_data.shape}")
        print(f"   Data columns: {list(test_data.columns)}")
        
        # Try to generate signal with detailed error tracking
        print("\n🧪 Testing signal generation with error tracking...")
        
        try:
            signal = system.generate_signal(test_data)
            print("✅ Signal generation successful!")
            print(f"   Signal: {signal}")
            
        except Exception as e:
            print(f"❌ Signal generation failed: {e}")
            print("\n📍 DETAILED ERROR TRACEBACK:")
            traceback.print_exc()
            
            # Try to identify the specific line causing the issue
            tb = traceback.extract_tb(e.__traceback__)
            for frame in tb:
                if 'ultimate_xau_system.py' in frame.filename:
                    print(f"\n🎯 Error in file: {frame.filename}")
                    print(f"   Line {frame.lineno}: {frame.line}")
                    print(f"   Function: {frame.name}")
            
            return False
            
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def find_dataframe_conditions():
    """Tìm tất cả DataFrame conditions trong code"""
    print("\n🔍 FINDING DATAFRAME CONDITIONS IN CODE")
    print("-" * 45)
    
    system_file = "src/core/ultimate_xau_system.py"
    
    with open(system_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    problematic_patterns = [
        'if data:',
        'if not data:',
        'if features:',
        'if not features:',
        'if signal_data:',
        'if not signal_data:',
        'data and ',
        'data or ',
        'features and ',
        'features or ',
    ]
    
    found_issues = []
    
    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        for pattern in problematic_patterns:
            if pattern in line_stripped and not line_stripped.startswith('#'):
                found_issues.append({
                    'line_number': i,
                    'line_content': line_stripped,
                    'pattern': pattern
                })
    
    if found_issues:
        print(f"🎯 Found {len(found_issues)} potential DataFrame ambiguity issues:")
        for issue in found_issues:
            print(f"   Line {issue['line_number']}: {issue['line_content']}")
            print(f"      Pattern: {issue['pattern']}")
    else:
        print("✅ No obvious DataFrame condition issues found")
    
    return found_issues

def create_targeted_fix():
    """Tạo fix có mục tiêu dựa trên chẩn đoán"""
    print("\n🔧 CREATING TARGETED FIX")
    print("-" * 30)
    
    # Find the issues first
    issues = find_dataframe_conditions()
    
    if not issues:
        print("⚠️ No specific issues found to fix")
        return False
    
    system_file = "src/core/ultimate_xau_system.py"
    backup_file = f"ultimate_xau_system_backup_targeted_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    # Create backup
    import shutil
    shutil.copy2(system_file, backup_file)
    print(f"📦 Backup created: {backup_file}")
    
    # Read file
    with open(system_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply targeted fixes
    fixes_applied = []
    
    # Specific fixes based on common DataFrame ambiguity patterns
    targeted_fixes = [
        # Most common patterns that cause ambiguity
        ('if data:', 'if not data.empty:'),
        ('if not data:', 'if data.empty:'),
        ('if features:', 'if not features.empty:'),
        ('if not features:', 'if features.empty:'),
        ('if signal_data:', 'if not signal_data.empty:'),
        ('if not signal_data:', 'if signal_data.empty:'),
        
        # Boolean operations with DataFrames
        ('data and ', 'not data.empty and '),
        ('data or ', 'not data.empty or '),
        ('features and ', 'not features.empty and '),
        ('features or ', 'not features.empty or '),
        
        # Common DataFrame variables
        ('if df:', 'if not df.empty:'),
        ('if not df:', 'if df.empty:'),
        ('if market_data:', 'if not market_data.empty:'),
        ('if not market_data:', 'if market_data.empty:'),
        
        # Specific to signal generation
        ('if len(data) > 0:', 'if not data.empty:'),
        ('if len(data) == 0:', 'if data.empty:'),
        ('len(data) > 0', 'not data.empty'),
        ('len(data) == 0', 'data.empty'),
    ]
    
    for old_pattern, new_pattern in targeted_fixes:
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            fixes_applied.append(f"Fixed: {old_pattern} → {new_pattern}")
    
    # Special fix for pandas DataFrame boolean evaluation
    # Add a helper function if not exists
    if '_is_dataframe_valid' not in content:
        helper_function = '''
    def _is_dataframe_valid(self, df):
        """Check if DataFrame is valid and not empty"""
        if df is None:
            return False
        if hasattr(df, 'empty'):
            return not df.empty
        return bool(df)
    
    def _is_dataframe_empty(self, df):
        """Check if DataFrame is empty"""
        if df is None:
            return True
        if hasattr(df, 'empty'):
            return df.empty
        return not bool(df)
'''
        
        # Insert before the first method
        first_method = content.find('    def ')
        if first_method != -1:
            content = content[:first_method] + helper_function + content[first_method:]
            fixes_applied.append("Added DataFrame validation helper functions")
    
    # Save fixed file
    with open(system_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Applied {len(fixes_applied)} targeted fixes:")
    for fix in fixes_applied:
        print(f"   • {fix}")
    
    return len(fixes_applied) > 0

def main():
    """Main diagnostic function"""
    print("🔍 DATAFRAME ERROR DIAGNOSTIC & FIX")
    print("=" * 45)
    
    # Step 1: Diagnose the error
    print("Step 1: Diagnosing current error...")
    success = diagnose_dataframe_error()
    
    if success:
        print("✅ No DataFrame errors detected!")
        return True
    
    # Step 2: Find problematic conditions
    print("\nStep 2: Finding problematic DataFrame conditions...")
    issues = find_dataframe_conditions()
    
    # Step 3: Create targeted fix
    print("\nStep 3: Creating targeted fix...")
    fix_applied = create_targeted_fix()
    
    if fix_applied:
        print("\n✅ TARGETED FIX APPLIED!")
        print("🔍 Ready for validation...")
        
        # Step 4: Test the fix
        print("\nStep 4: Testing the fix...")
        success_after_fix = diagnose_dataframe_error()
        
        if success_after_fix:
            print("🎉 DATAFRAME ERROR FIXED SUCCESSFULLY!")
            return True
        else:
            print("⚠️ Error still persists, may need manual intervention")
            return False
    else:
        print("⚠️ No fixes could be applied automatically")
        return False

if __name__ == "__main__":
    main() 