#!/usr/bin/env python3
"""
🔄 UPDATE ULTIMATE XAU SYSTEM V4.0 VỚI MODELS MỚI
================================================
Script này sẽ update hệ thống chính để sử dụng models mới được train với dữ liệu thực tế
"""

import sys
import os
import json
import glob
from datetime import datetime
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class SystemUpdater:
    """Update hệ thống với models mới"""
    
    def __init__(self):
        self.models_dir = "trained_models_real_data"
        self.results_dir = "training_results_real_data"
        self.system_models_dir = "trained_models"
        self.backup_dir = "trained_models_backup"
        
        # Tạo backup directory
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.system_models_dir, exist_ok=True)
        
    def backup_old_models(self):
        """Backup models cũ"""
        print("💾 BACKUP MODELS CŨ")
        print("=" * 30)
        
        if os.path.exists(self.system_models_dir):
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.backup_dir}/backup_{backup_timestamp}"
            
            if os.listdir(self.system_models_dir):
                shutil.copytree(self.system_models_dir, backup_path)
                print(f"✅ Backed up old models to: {backup_path}")
            else:
                print("ℹ️  No old models to backup")
        
    def find_latest_training_results(self):
        """Tìm kết quả training mới nhất"""
        print("🔍 TÌM KẾT QUẢ TRAINING MỚI NHẤT")
        print("=" * 40)
        
        if not os.path.exists(self.results_dir):
            print("❌ Không tìm thấy thư mục kết quả training")
            return None
            
        # Tìm file results mới nhất
        result_files = glob.glob(f"{self.results_dir}/comprehensive_training_results_*.json")
        
        if not result_files:
            print("❌ Không tìm thấy file kết quả training")
            return None
            
        latest_file = max(result_files, key=os.path.getctime)
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        print(f"✅ Tìm thấy kết quả training: {latest_file}")
        print(f"   📅 Training time: {results.get('start_time', 'N/A')}")
        print(f"   🎯 Timeframes: {list(results.get('training_phases', {}).keys())}")
        
        return results
        
    def copy_new_models(self, training_results):
        """Copy models mới vào system directory"""
        print("\n📦 COPY MODELS MỚI VÀO HỆ THỐNG")
        print("=" * 40)
        
        copied_models = {}
        
        for timeframe, phase_results in training_results.get('training_phases', {}).items():
            print(f"\n   📈 Processing {timeframe} models...")
            
            saved_models = phase_results.get('saved_models', {})
            
            for model_name, model_info in saved_models.items():
                source_file = model_info.get('file', '')
                
                if os.path.exists(source_file):
                    # Tạo tên file mới trong system directory
                    filename = os.path.basename(source_file)
                    dest_file = os.path.join(self.system_models_dir, filename)
                    
                    # Copy file
                    shutil.copy2(source_file, dest_file)
                    
                    copied_models[f"{timeframe}_{model_name}"] = {
                        'source': source_file,
                        'destination': dest_file,
                        'type': model_info.get('type', 'unknown'),
                        'framework': model_info.get('framework', 'unknown')
                    }
                    
                    print(f"      ✅ Copied {model_name}: {filename}")
                else:
                    print(f"      ❌ Model file not found: {source_file}")
        
        print(f"\n✅ Đã copy {len(copied_models)} models vào hệ thống")
        return copied_models
        
    def create_model_config(self, training_results, copied_models):
        """Tạo config file cho models mới"""
        print("\n⚙️  TẠO CONFIG CHO MODELS MỚI")
        print("=" * 40)
        
        config = {
            'created_at': datetime.now().isoformat(),
            'training_summary': {
                'total_data_points': training_results.get('data_summary', {}).get('total_records', 0),
                'date_range': training_results.get('data_summary', {}).get('date_range', {}),
                'timeframes_trained': list(training_results.get('training_phases', {}).keys())
            },
            'models': {},
            'performance_summary': {}
        }
        
        # Thông tin từng model
        for model_key, model_info in copied_models.items():
            timeframe, model_name = model_key.split('_', 1)
            
            if timeframe not in config['models']:
                config['models'][timeframe] = {}
            
            config['models'][timeframe][model_name] = {
                'file': model_info['destination'],
                'type': model_info['type'],
                'framework': model_info['framework'],
                'trained_with_real_data': True
            }
        
        # Performance summary
        for timeframe, phase_results in training_results.get('training_phases', {}).items():
            evaluation_results = phase_results.get('evaluation_results', {})
            
            if evaluation_results:
                best_accuracy = max([r.get('accuracy', 0) for r in evaluation_results.values()])
                best_model = max(evaluation_results.items(), key=lambda x: x[1].get('accuracy', 0))[0]
                
                config['performance_summary'][timeframe] = {
                    'best_model': best_model,
                    'best_accuracy': best_accuracy,
                    'models_count': len(evaluation_results),
                    'data_points': phase_results.get('data_points', 0)
                }
        
        # Lưu config
        config_file = os.path.join(self.system_models_dir, 'real_data_models_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Đã tạo config: {config_file}")
        return config
        
    def update_system_code(self):
        """Update code hệ thống để sử dụng models mới"""
        print("\n🔧 UPDATE SYSTEM CODE")
        print("=" * 30)
        
        # Tạo script helper để load models mới
        helper_code = '''"""
Helper functions để load models được train với dữ liệu thực tế
"""

import os
import json
import joblib
import tensorflow as tf
from typing import Dict, Any, Optional

class RealDataModelLoader:
    """Load models được train với dữ liệu thực tế"""
    
    def __init__(self, models_dir="trained_models"):
        self.models_dir = models_dir
        self.config_file = os.path.join(models_dir, "real_data_models_config.json")
        self.config = self.load_config()
        self.loaded_models = {}
        
    def load_config(self) -> Dict:
        """Load model configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_available_timeframes(self) -> list:
        """Get list of available timeframes"""
        return list(self.config.get('models', {}).keys())
    
    def get_best_model_for_timeframe(self, timeframe: str) -> Optional[str]:
        """Get best performing model for a timeframe"""
        perf_summary = self.config.get('performance_summary', {})
        if timeframe in perf_summary:
            return perf_summary[timeframe].get('best_model')
        return None
    
    def load_model(self, timeframe: str, model_name: str) -> Optional[Any]:
        """Load a specific model"""
        model_key = f"{timeframe}_{model_name}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        models_config = self.config.get('models', {})
        if timeframe not in models_config or model_name not in models_config[timeframe]:
            return None
        
        model_info = models_config[timeframe][model_name]
        model_file = model_info['file']
        framework = model_info['framework']
        
        if not os.path.exists(model_file):
            return None
        
        try:
            if framework == 'tensorflow':
                model = tf.keras.models.load_model(model_file)
            elif framework == 'sklearn':
                model = joblib.load(model_file)
            else:
                return None
            
            self.loaded_models[model_key] = model
            return model
            
        except Exception as e:
            print(f"Error loading model {model_key}: {e}")
            return None
    
    def load_scaler_and_features(self, timeframe: str) -> tuple:
        """Load scaler and features for a timeframe"""
        # Tìm scaler và features files
        import glob
        
        scaler_pattern = f"{self.models_dir}/scaler_{timeframe}_*.pkl"
        features_pattern = f"{self.models_dir}/features_{timeframe}_*.json"
        
        scaler_files = glob.glob(scaler_pattern)
        features_files = glob.glob(features_pattern)
        
        scaler, features = None, None
        
        if scaler_files:
            latest_scaler = max(scaler_files, key=os.path.getctime)
            scaler = joblib.load(latest_scaler)
        
        if features_files:
            latest_features = max(features_files, key=os.path.getctime)
            with open(latest_features, 'r') as f:
                features = json.load(f)
        
        return scaler, features
    
    def get_ensemble_prediction(self, timeframe: str, X: Any) -> Dict:
        """Get ensemble prediction from all models for a timeframe"""
        models_config = self.config.get('models', {}).get(timeframe, {})
        
        predictions = []
        confidences = []
        model_names = []
        
        for model_name in models_config.keys():
            model = self.load_model(timeframe, model_name)
            if model is not None:
                try:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)
                        pred = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]
                        conf = max(pred_proba[0])
                    elif hasattr(model, 'predict'):
                        pred = model.predict(X)
                        if len(pred.shape) > 1:
                            pred = pred.flatten()
                        conf = 0.7  # Default confidence
                    else:
                        continue
                    
                    predictions.append(float(pred[0]) if hasattr(pred, '__len__') else float(pred))
                    confidences.append(float(conf))
                    model_names.append(model_name)
                    
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
                    continue
        
        if not predictions:
            return {'prediction': 0.5, 'confidence': 0.0, 'models_used': 0}
        
        # Ensemble prediction (weighted average)
        ensemble_pred = sum(predictions) / len(predictions)
        ensemble_conf = sum(confidences) / len(confidences)
        
        return {
            'prediction': ensemble_pred,
            'confidence': ensemble_conf,
            'models_used': len(predictions),
            'individual_predictions': dict(zip(model_names, predictions)),
            'trained_with_real_data': True
        }

# Global instance
real_data_model_loader = RealDataModelLoader()
'''
        
        helper_file = os.path.join('src', 'core', 'real_data_model_loader.py')
        os.makedirs(os.path.dirname(helper_file), exist_ok=True)
        
        with open(helper_file, 'w') as f:
            f.write(helper_code)
        
        print(f"✅ Đã tạo helper: {helper_file}")
        
        # Tạo script test models mới
        test_script = '''#!/usr/bin/env python3
"""
Test script cho models mới được train với dữ liệu thực tế
"""

import sys
import os
sys.path.append('src')

from core.real_data_model_loader import real_data_model_loader
import pandas as pd
import numpy as np

def test_new_models():
    """Test models mới"""
    print("🧪 TEST MODELS MỚI VỚI DỮ LIỆU THỰC TẾ")
    print("=" * 50)
    
    # Kiểm tra config
    config = real_data_model_loader.config
    print(f"✅ Config loaded: {len(config)} sections")
    
    if 'training_summary' in config:
        summary = config['training_summary']
        print(f"📊 Training data: {summary.get('total_data_points', 0):,} records")
        print(f"📅 Date range: {summary.get('date_range', {})}")
    
    # Test từng timeframe
    timeframes = real_data_model_loader.get_available_timeframes()
    print(f"\\n🎯 Available timeframes: {timeframes}")
    
    for tf in timeframes:
        print(f"\\n--- Testing {tf} ---")
        
        # Load scaler và features
        scaler, features = real_data_model_loader.load_scaler_and_features(tf)
        
        if scaler is not None and features is not None:
            print(f"   ✅ Scaler loaded: {type(scaler).__name__}")
            print(f"   ✅ Features loaded: {len(features)} features")
            
            # Tạo sample data để test
            sample_data = np.random.randn(1, len(features))
            sample_data_scaled = scaler.transform(sample_data)
            
            # Test ensemble prediction
            result = real_data_model_loader.get_ensemble_prediction(tf, sample_data_scaled)
            
            print(f"   🎯 Ensemble prediction: {result['prediction']:.4f}")
            print(f"   🎯 Confidence: {result['confidence']:.4f}")
            print(f"   🎯 Models used: {result['models_used']}")
            print(f"   🎯 Real data trained: {result['trained_with_real_data']}")
        else:
            print(f"   ❌ Could not load scaler/features for {tf}")

if __name__ == "__main__":
    test_new_models()
'''
        
        test_file = "test_new_models.py"
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        print(f"✅ Đã tạo test script: {test_file}")
        
    def run_update(self):
        """Chạy toàn bộ quá trình update"""
        print("🔄 BẮT ĐẦU UPDATE HỆ THỐNG VỚI MODELS MỚI")
        print("=" * 60)
        
        # Backup old models
        self.backup_old_models()
        
        # Find latest training results
        training_results = self.find_latest_training_results()
        if training_results is None:
            print("❌ Không thể tìm thấy kết quả training. Hãy chạy training trước!")
            return False
        
        # Copy new models
        copied_models = self.copy_new_models(training_results)
        if not copied_models:
            print("❌ Không có models nào được copy!")
            return False
        
        # Create model config
        config = self.create_model_config(training_results, copied_models)
        
        # Update system code
        self.update_system_code()
        
        print("\n🎉 HOÀN THÀNH UPDATE HỆ THỐNG!")
        print("=" * 40)
        print(f"📦 Models copied: {len(copied_models)}")
        print(f"🎯 Timeframes: {list(config['performance_summary'].keys())}")
        print("✅ Hệ thống đã sẵn sàng sử dụng models mới!")
        
        # Show performance summary
        print("\\n📈 PERFORMANCE SUMMARY:")
        for tf, perf in config['performance_summary'].items():
            print(f"   {tf}: {perf['best_accuracy']:.4f} accuracy ({perf['best_model']})")
        
        return True

def main():
    """Main update function"""
    print("🔄 ULTIMATE XAU SYSTEM V4.0 - MODEL UPDATE")
    print("=" * 50)
    print(f"⏰ Thời gian: {datetime.now()}")
    print()
    
    updater = SystemUpdater()
    success = updater.run_update()
    
    if success:
        print("\\n🚀 HỆ THỐNG ĐÃ ĐƯỢC UPDATE!")
        print("   Bạn có thể chạy 'python test_new_models.py' để test")
    else:
        print("\\n❌ UPDATE THẤT BẠI!")

if __name__ == "__main__":
    main() 