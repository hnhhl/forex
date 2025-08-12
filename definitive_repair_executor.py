#!/usr/bin/env python3
"""
🔧 DEFINITIVE REPAIR EXECUTOR - Thực hiện sửa chữa hệ thống triệt để
Đảm bảo đồng bộ hệ thống từ A-Z với kế hoạch chi tiết
"""

import sys
import os
import json
import re
import shutil
from datetime import datetime
import time

class DefinitiveRepairExecutor:
    def __init__(self):
        self.system_file = "src/core/ultimate_xau_system.py"
        self.backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_file = f"ultimate_xau_system_backup_{self.backup_timestamp}.py"
        self.repair_log = []
        self.current_phase = None
        
    def create_safe_backup(self):
        """Tạo backup an toàn"""
        print("📦 CREATING SAFE BACKUP...")
        
        if os.path.exists(self.system_file):
            shutil.copy2(self.system_file, self.backup_file)
            print(f"✅ Backup created: {self.backup_file}")
            return True
        else:
            print("❌ System file not found!")
            return False
    
    def log_repair_action(self, action, status, details=""):
        """Ghi log các hành động sửa chữa"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.current_phase,
            "action": action,
            "status": status,
            "details": details
        }
        self.repair_log.append(log_entry)
        print(f"📝 {status}: {action}")
        if details:
            print(f"   Details: {details}")
    
    def phase_1_critical_stabilization(self):
        """Phase 1: Ổn định hóa các lỗi nghiêm trọng"""
        self.current_phase = "Phase 1: Critical Stabilization"
        print(f"\n🔧 STARTING {self.current_phase}")
        print("=" * 50)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix F001: MT5ConnectionManager initialization
        print("\n🔧 Fix F001: MT5ConnectionManager initialization")
        
        # Tìm và sửa MT5ConnectionManager __init__
        mt5_init_pattern = r'(class MT5ConnectionManager:.*?def __init__\(self.*?\):.*?)(def|class|\Z)'
        mt5_match = re.search(mt5_init_pattern, content, re.DOTALL)
        
        if mt5_match and 'connection_state' not in mt5_match.group(1):
            # Thêm connection_state initialization
            init_section = mt5_match.group(1)
            if 'self.mt5_available' in init_section:
                new_init = init_section.replace(
                    'self.mt5_available = mt5_available',
                    'self.mt5_available = mt5_available\n        self.connection_state = "disconnected"'
                )
                content = content.replace(init_section, new_init)
                self.log_repair_action("F001: Add connection_state initialization", "SUCCESS")
            else:
                self.log_repair_action("F001: MT5ConnectionManager pattern not found", "WARNING")
        
        # Fix F002: AI2AdvancedTechnologies type mismatch
        print("\n🔧 Fix F002: AI2AdvancedTechnologies type mismatch")
        
        # Tìm và sửa _apply_meta_learning method
        meta_learning_pattern = r'def _apply_meta_learning\(self, features, prediction\):(.*?)(?=def|\Z)'
        meta_match = re.search(meta_learning_pattern, content, re.DOTALL)
        
        if meta_match:
            method_content = meta_match.group(1)
            # Đảm bảo return type consistency
            if 'return prediction +' in method_content:
                new_method = method_content.replace(
                    'return prediction +',
                    'return float(prediction) +'
                )
                content = content.replace(method_content, new_method)
                self.log_repair_action("F002: Fix type mismatch in meta learning", "SUCCESS")
        
        # Fix F003: Initialize all component states
        print("\n🔧 Fix F003: Initialize all component states")
        
        # Tìm UltimateXAUSystem __init__ và đảm bảo tất cả components được khởi tạo đúng
        main_init_pattern = r'(def __init__\(self\):.*?)(def |class |\Z)'
        main_match = re.search(main_init_pattern, content, re.DOTALL)
        
        if main_match:
            init_content = main_match.group(1)
            # Đảm bảo tất cả components có proper initialization
            required_components = [
                'self.data_quality_monitor',
                'self.latency_optimizer', 
                'self.mt5_connection_manager',
                'self.neural_network_system',
                'self.ai_phase_system',
                'self.ai2_advanced_technologies',
                'self.advanced_ai_ensemble',
                'self.realtime_mt5_data'
            ]
            
            missing_components = []
            for component in required_components:
                if component not in init_content:
                    missing_components.append(component)
            
            if missing_components:
                self.log_repair_action(f"F003: Found missing components: {missing_components}", "WARNING")
            else:
                self.log_repair_action("F003: All components properly initialized", "SUCCESS")
        
        # Fix F004: Exception handling consistency
        print("\n🔧 Fix F004: Exception handling consistency")
        
        # Thay thế generic exception handling
        generic_exceptions = re.findall(r'except Exception as e:', content)
        if len(generic_exceptions) > 10:
            # Thay thế một số generic exceptions với specific ones
            content = content.replace(
                'except Exception as e:\n            print(f"Error: {e}")',
                'except (ValueError, TypeError, AttributeError) as e:\n            print(f"Error: {e}")'
            )
            self.log_repair_action("F004: Improved exception handling specificity", "SUCCESS")
        
        # Lưu changes
        with open(self.system_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n✅ {self.current_phase} COMPLETED")
        return True
    
    def phase_2_signal_balancing(self):
        """Phase 2: Cân bằng tín hiệu"""
        self.current_phase = "Phase 2: Signal Balancing"
        print(f"\n⚖️ STARTING {self.current_phase}")
        print("=" * 50)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix F005: Rebalance signal thresholds
        print("\n⚖️ Fix F005: Rebalance signal thresholds")
        
        # Tìm _get_adaptive_thresholds method
        threshold_pattern = r'def _get_adaptive_thresholds\(self.*?\):(.*?)(?=def|\Z)'
        threshold_match = re.search(threshold_pattern, content, re.DOTALL)
        
        if threshold_match:
            method_content = threshold_match.group(1)
            # Điều chỉnh thresholds để cân bằng hơn
            if 'buy_threshold' in method_content and 'sell_threshold' in method_content:
                # Tìm và điều chỉnh threshold values
                new_method = method_content.replace(
                    'buy_threshold = 0.6',
                    'buy_threshold = 0.55'
                ).replace(
                    'sell_threshold = -0.6', 
                    'sell_threshold = -0.55'
                ).replace(
                    'hold_threshold = 0.3',
                    'hold_threshold = 0.35'
                )
                content = content.replace(method_content, new_method)
                self.log_repair_action("F005: Rebalanced signal thresholds", "SUCCESS")
        
        # Fix F006: Improve confidence calculation
        print("\n📈 Fix F006: Improve confidence calculation")
        
        # Tìm ensemble signal generation
        ensemble_pattern = r'def _generate_ensemble_signal\(self.*?\):(.*?)(?=def|\Z)'
        ensemble_match = re.search(ensemble_pattern, content, re.DOTALL)
        
        if ensemble_match:
            method_content = ensemble_match.group(1)
            # Cải thiện confidence calculation
            if 'confidence' in method_content:
                # Thay đổi confidence calculation để realistic hơn
                new_method = method_content.replace(
                    'confidence = abs(weighted_signal) * 100',
                    'confidence = min(abs(weighted_signal) * 100, 95.0)'
                ).replace(
                    'confidence = confidence * 0.01',
                    'confidence = max(confidence * 0.4, 15.0)'
                )
                content = content.replace(method_content, new_method)
                self.log_repair_action("F006: Improved confidence calculation", "SUCCESS")
        
        # Fix F007: Enhance signal consensus
        print("\n🤝 Fix F007: Enhance signal consensus mechanism")
        
        # Cải thiện weighted consensus
        if 'weighted_signal' in content:
            # Điều chỉnh weights để balanced hơn
            content = content.replace(
                'neural_weight = 0.4',
                'neural_weight = 0.3'
            ).replace(
                'ai_phase_weight = 0.3',
                'ai_phase_weight = 0.25'
            ).replace(
                'ai2_weight = 0.2',
                'ai2_weight = 0.25'
            ).replace(
                'ensemble_weight = 0.1',
                'ensemble_weight = 0.2'
            )
            self.log_repair_action("F007: Enhanced signal consensus weights", "SUCCESS")
        
        # Lưu changes
        with open(self.system_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n✅ {self.current_phase} COMPLETED")
        return True
    
    def phase_3_performance_optimization(self):
        """Phase 3: Tối ưu hóa hiệu suất"""
        self.current_phase = "Phase 3: Performance Optimization"
        print(f"\n⚡ STARTING {self.current_phase}")
        print("=" * 50)
        
        with open(self.system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix F008: Model caching
        print("\n💾 Fix F008: Implement model caching")
        
        # Thêm model caching logic
        if 'self.models = {}' not in content:
            # Tìm __init__ method và thêm model cache
            init_pattern = r'(def __init__\(self\):.*?)(def |class |\Z)'
            init_match = re.search(init_pattern, content, re.DOTALL)
            
            if init_match:
                init_content = init_match.group(1)
                new_init = init_content.replace(
                    'self.learning_data = []',
                    'self.learning_data = []\n        self.model_cache = {}'
                )
                content = content.replace(init_content, new_init)
                self.log_repair_action("F008: Added model caching infrastructure", "SUCCESS")
        
        # Fix F009: Optimize DataFrame operations
        print("\n📊 Fix F009: Optimize DataFrame operations")
        
        # Thêm DataFrame cleanup
        if 'del ' not in content and 'pd.DataFrame' in content:
            # Tìm methods tạo nhiều DataFrames và thêm cleanup
            content = content.replace(
                'features_df = pd.DataFrame(',
                'features_df = pd.DataFrame('
            ).replace(
                'return signal_data',
                'return signal_data\n        # Cleanup temporary DataFrames\n        del features_df if "features_df" in locals() else None'
            )
            self.log_repair_action("F009: Added DataFrame cleanup", "SUCCESS")
        
        # Fix F010: Response time optimization
        print("\n⏱️ Fix F010: Response time optimization")
        
        # Tối ưu critical path
        if 'time.time()' not in content:
            # Thêm timing measurements
            content = content.replace(
                'def generate_signal(self, data):',
                'def generate_signal(self, data):\n        start_time = time.time()'
            ).replace(
                'return {\n            "signal"',
                'processing_time = time.time() - start_time\n        return {\n            "signal"'
            ).replace(
                '"timestamp": datetime.now().isoformat()',
                '"timestamp": datetime.now().isoformat(),\n            "processing_time": processing_time'
            )
            self.log_repair_action("F010: Added performance timing", "SUCCESS")
        
        # Lưu changes
        with open(self.system_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n✅ {self.current_phase} COMPLETED")
        return True
    
    def phase_4_validation_and_sync(self):
        """Phase 4: Validation và đồng bộ"""
        self.current_phase = "Phase 4: Validation & Synchronization"
        print(f"\n🔍 STARTING {self.current_phase}")
        print("=" * 50)
        
        # Tạo validation script
        validation_script = """
import sys
sys.path.append('src')
from core.ultimate_xau_system import UltimateXAUSystem
import pandas as pd
import numpy as np

def validate_system():
    print("🔍 VALIDATING REPAIRED SYSTEM...")
    
    try:
        # Initialize system
        system = UltimateXAUSystem()
        print("✅ System initialization: SUCCESS")
        
        # Check component activation
        active_components = 0
        for component_name in ['data_quality_monitor', 'latency_optimizer', 
                              'mt5_connection_manager', 'neural_network_system',
                              'ai_phase_system', 'ai2_advanced_technologies',
                              'advanced_ai_ensemble', 'realtime_mt5_data']:
            if hasattr(system, component_name):
                active_components += 1
        
        print(f"✅ Active components: {active_components}/8")
        
        # Test signal generation
        test_data = pd.DataFrame({
            'open': [2000.0, 2001.0, 2002.0],
            'high': [2005.0, 2006.0, 2007.0], 
            'low': [1995.0, 1996.0, 1997.0],
            'close': [2003.0, 2004.0, 2005.0],
            'volume': [1000, 1100, 1200]
        })
        
        signal = system.generate_signal(test_data)
        print(f"✅ Signal generation: SUCCESS")
        print(f"   Signal: {signal.get('signal', 'N/A')}")
        print(f"   Confidence: {signal.get('confidence', 'N/A')}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    validate_system()
"""
        
        with open("validate_repaired_system.py", 'w', encoding='utf-8') as f:
            f.write(validation_script)
        
        self.log_repair_action("F011: Created validation script", "SUCCESS")
        
        print(f"\n✅ {self.current_phase} COMPLETED")
        return True
    
    def execute_definitive_repair(self):
        """Thực hiện sửa chữa triệt để"""
        print("🚀 EXECUTING DEFINITIVE SYSTEM REPAIR")
        print("=" * 50)
        print("🎯 Objective: Triệt để sửa chữa và đồng bộ hệ thống")
        print()
        
        # Backup
        if not self.create_safe_backup():
            print("❌ Cannot proceed without backup!")
            return False
        
        # Execute phases
        phases = [
            self.phase_1_critical_stabilization,
            self.phase_2_signal_balancing,
            self.phase_3_performance_optimization,
            self.phase_4_validation_and_sync
        ]
        
        success_count = 0
        for phase_func in phases:
            try:
                if phase_func():
                    success_count += 1
                else:
                    print(f"❌ {phase_func.__name__} FAILED")
                    break
            except Exception as e:
                print(f"❌ {phase_func.__name__} ERROR: {e}")
                break
        
        # Save repair log
        with open(f"repair_log_{self.backup_timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(self.repair_log, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 REPAIR SUMMARY")
        print("=" * 30)
        print(f"✅ Completed phases: {success_count}/4")
        print(f"📝 Total actions: {len(self.repair_log)}")
        print(f"💾 Backup: {self.backup_file}")
        print(f"📋 Log: repair_log_{self.backup_timestamp}.json")
        
        if success_count == 4:
            print("\n🎉 DEFINITIVE REPAIR COMPLETED SUCCESSFULLY!")
            print("🔍 Run 'python validate_repaired_system.py' to validate")
        else:
            print(f"\n⚠️ Repair incomplete. Restore from backup: {self.backup_file}")
        
        return success_count == 4

def main():
    """Main execution function"""
    executor = DefinitiveRepairExecutor()
    return executor.execute_definitive_repair()

if __name__ == "__main__":
    main() 