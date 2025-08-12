#!/usr/bin/env python3
"""
🧹 SYSTEM CLEANUP - AI3.0 TRADING SYSTEM
Cleanup disk space và optimize storage
"""

import os
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

def analyze_disk_usage():
    """Phân tích disk usage chi tiết"""
    print("📊 ANALYZING DISK USAGE...")
    print("=" * 50)
    
    analysis = {}
    
    # Analyze major directories
    major_dirs = [
        'trained_models',
        'trained_models_optimized', 
        'trained_models_real_data',
        'data',
        'learning_results',
        'training_results',
        'backtest_results',
        'logs'
    ]
    
    total_size = 0
    
    for dir_name in major_dirs:
        if os.path.exists(dir_name):
            dir_size = get_directory_size(dir_name)
            file_count = count_files(dir_name)
            
            analysis[dir_name] = {
                'size_mb': round(dir_size / (1024*1024), 2),
                'file_count': file_count,
                'exists': True
            }
            
            total_size += dir_size
            
            print(f"📁 {dir_name}: {analysis[dir_name]['size_mb']} MB ({file_count} files)")
        else:
            analysis[dir_name] = {'exists': False}
            print(f"❌ {dir_name}: Not found")
    
    analysis['total_size_mb'] = round(total_size / (1024*1024), 2)
    analysis['total_size_gb'] = round(total_size / (1024*1024*1024), 2)
    
    print(f"\n📊 TOTAL SIZE: {analysis['total_size_gb']} GB")
    
    return analysis

def get_directory_size(directory):
    """Get total size of directory"""
    total_size = 0
    
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"   ⚠️  Error calculating size for {directory}: {e}")
    
    return total_size

def count_files(directory):
    """Count files in directory"""
    count = 0
    
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            count += len(filenames)
    except Exception as e:
        print(f"   ⚠️  Error counting files in {directory}: {e}")
    
    return count

def cleanup_duplicate_models():
    """Cleanup duplicate và old models"""
    print("\n🗑️  CLEANING UP DUPLICATE MODELS...")
    print("=" * 50)
    
    cleanup_stats = {
        'files_removed': 0,
        'space_freed_mb': 0,
        'directories_cleaned': []
    }
    
    # Cleanup trained_models (keep only latest)
    models_dir = 'trained_models'
    if os.path.exists(models_dir):
        print(f"📁 Cleaning {models_dir}...")
        
        # Group files by type
        files_by_type = {}
        
        for filename in os.listdir(models_dir):
            filepath = os.path.join(models_dir, filename)
            if os.path.isfile(filepath):
                # Extract base type (before timestamp)
                base_type = filename.split('_20')[0] if '_20' in filename else filename
                
                if base_type not in files_by_type:
                    files_by_type[base_type] = []
                
                files_by_type[base_type].append({
                    'filename': filename,
                    'filepath': filepath,
                    'modified': os.path.getmtime(filepath),
                    'size': os.path.getsize(filepath)
                })
        
        # Keep only latest file for each type
        for base_type, files in files_by_type.items():
            if len(files) > 1:
                # Sort by modification time (newest first)
                files.sort(key=lambda x: x['modified'], reverse=True)
                
                # Keep first (newest), remove others
                for file_info in files[1:]:
                    try:
                        os.remove(file_info['filepath'])
                        cleanup_stats['files_removed'] += 1
                        cleanup_stats['space_freed_mb'] += file_info['size'] / (1024*1024)
                        print(f"   🗑️  Removed: {file_info['filename']}")
                    except Exception as e:
                        print(f"   ❌ Error removing {file_info['filename']}: {e}")
        
        cleanup_stats['directories_cleaned'].append(models_dir)
    
    return cleanup_stats

def cleanup_old_results():
    """Cleanup old result files"""
    print("\n🗑️  CLEANING UP OLD RESULTS...")
    print("=" * 50)
    
    cleanup_stats = {
        'files_removed': 0,
        'space_freed_mb': 0,
        'directories_cleaned': []
    }
    
    # Directories with timestamped results
    result_dirs = [
        'learning_results',
        'training_results', 
        'backtest_results',
        'analysis_reports',
        'comprehensive_trading_results',
        'detailed_trading_results'
    ]
    
    cutoff_date = datetime.now() - timedelta(days=7)  # Keep last 7 days
    
    for dir_name in result_dirs:
        if os.path.exists(dir_name):
            print(f"📁 Cleaning {dir_name} (keeping last 7 days)...")
            
            files_removed = 0
            space_freed = 0
            
            for filename in os.listdir(dir_name):
                filepath = os.path.join(dir_name, filename)
                
                if os.path.isfile(filepath):
                    file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_modified < cutoff_date:
                        try:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            
                            files_removed += 1
                            space_freed += file_size / (1024*1024)
                            
                            print(f"   🗑️  Removed old: {filename}")
                        except Exception as e:
                            print(f"   ❌ Error removing {filename}: {e}")
            
            cleanup_stats['files_removed'] += files_removed
            cleanup_stats['space_freed_mb'] += space_freed
            
            if files_removed > 0:
                cleanup_stats['directories_cleaned'].append(dir_name)
                print(f"   ✅ Cleaned {files_removed} old files ({space_freed:.2f} MB)")
    
    return cleanup_stats

def cleanup_logs():
    """Cleanup old log files"""
    print("\n🗑️  CLEANING UP LOG FILES...")
    print("=" * 50)
    
    cleanup_stats = {
        'files_removed': 0,
        'space_freed_mb': 0
    }
    
    logs_dir = 'logs'
    if os.path.exists(logs_dir):
        cutoff_date = datetime.now() - timedelta(days=3)  # Keep last 3 days
        
        for filename in os.listdir(logs_dir):
            filepath = os.path.join(logs_dir, filename)
            
            if os.path.isfile(filepath) and filename.endswith('.log'):
                file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_modified < cutoff_date:
                    try:
                        file_size = os.path.getsize(filepath)
                        os.remove(filepath)
                        
                        cleanup_stats['files_removed'] += 1
                        cleanup_stats['space_freed_mb'] += file_size / (1024*1024)
                        
                        print(f"   🗑️  Removed old log: {filename}")
                    except Exception as e:
                        print(f"   ❌ Error removing {filename}: {e}")
        
        if cleanup_stats['files_removed'] > 0:
            print(f"   ✅ Cleaned {cleanup_stats['files_removed']} log files")
    
    return cleanup_stats

def cleanup_empty_directories():
    """Remove empty directories"""
    print("\n🗑️  REMOVING EMPTY DIRECTORIES...")
    print("=" * 50)
    
    removed_dirs = []
    
    # Check common directories
    check_dirs = [
        'trained_models_smart',
        'continuous_models',
        'continuous_results',
        'performance_profiles'
    ]
    
    for dir_name in check_dirs:
        if os.path.exists(dir_name):
            try:
                if not os.listdir(dir_name):  # Empty directory
                    os.rmdir(dir_name)
                    removed_dirs.append(dir_name)
                    print(f"   🗑️  Removed empty directory: {dir_name}")
            except Exception as e:
                print(f"   ❌ Error removing {dir_name}: {e}")
    
    return removed_dirs

def create_backup_important_files():
    """Backup important files before cleanup"""
    print("\n💾 CREATING BACKUP OF IMPORTANT FILES...")
    print("=" * 50)
    
    backup_dir = f"system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Important files to backup
    important_files = [
        'src/core/ultimate_xau_system.py',
        'src/core/gpu_neural_system.py',
        'trained_models/ai_coordination_config.json'
    ]
    
    backed_up = []
    
    for file_path in important_files:
        if os.path.exists(file_path):
            try:
                backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                shutil.copy2(file_path, backup_path)
                backed_up.append(file_path)
                print(f"   💾 Backed up: {file_path}")
            except Exception as e:
                print(f"   ❌ Backup failed for {file_path}: {e}")
    
    return backup_dir, backed_up

def main():
    """Main cleanup execution"""
    print("🧹 AI3.0 SYSTEM CLEANUP")
    print("=" * 70)
    print(f"🕒 Start Time: {datetime.now()}")
    
    # Analyze current usage
    initial_analysis = analyze_disk_usage()
    
    # Create backup
    backup_dir, backed_up = create_backup_important_files()
    
    # Perform cleanup
    total_cleanup = {
        'files_removed': 0,
        'space_freed_mb': 0.0,
        'directories_cleaned': [],
        'empty_dirs_removed': []
    }
    
    # Cleanup duplicate models
    model_cleanup = cleanup_duplicate_models()
    total_cleanup['files_removed'] += model_cleanup['files_removed']
    total_cleanup['space_freed_mb'] += model_cleanup['space_freed_mb']
    total_cleanup['directories_cleaned'].extend(model_cleanup['directories_cleaned'])
    
    # Cleanup old results
    results_cleanup = cleanup_old_results()
    total_cleanup['files_removed'] += results_cleanup['files_removed']
    total_cleanup['space_freed_mb'] += results_cleanup['space_freed_mb']
    total_cleanup['directories_cleaned'].extend(results_cleanup['directories_cleaned'])
    
    # Cleanup logs
    logs_cleanup = cleanup_logs()
    total_cleanup['files_removed'] += logs_cleanup['files_removed']
    total_cleanup['space_freed_mb'] += logs_cleanup['space_freed_mb']
    
    # Remove empty directories
    empty_dirs = cleanup_empty_directories()
    total_cleanup['empty_dirs_removed'] = empty_dirs
    
    # Final analysis
    print(f"\n📊 FINAL ANALYSIS...")
    final_analysis = analyze_disk_usage()
    
    # Save cleanup report
    cleanup_report = {
        'timestamp': datetime.now().isoformat(),
        'initial_analysis': initial_analysis,
        'final_analysis': final_analysis,
        'cleanup_summary': total_cleanup,
        'backup_created': backup_dir,
        'files_backed_up': backed_up,
        'space_freed_gb': round(total_cleanup['space_freed_mb'] / 1024, 2)
    }
    
    report_file = f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(cleanup_report, f, indent=2)
    
    # Summary
    print(f"\n🎯 CLEANUP SUMMARY:")
    print(f"   Files Removed: {total_cleanup['files_removed']}")
    print(f"   Space Freed: {cleanup_report['space_freed_gb']} GB")
    print(f"   Directories Cleaned: {len(set(total_cleanup['directories_cleaned']))}")
    print(f"   Empty Dirs Removed: {len(empty_dirs)}")
    print(f"   Backup Created: {backup_dir}")
    print(f"   Report Saved: {report_file}")
    
    print(f"\n✅ CLEANUP COMPLETED!")
    print(f"🕒 End Time: {datetime.now()}")

if __name__ == "__main__":
    main() 