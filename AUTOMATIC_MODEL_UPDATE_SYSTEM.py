#!/usr/bin/env python3
"""
🔄 AUTOMATIC MODEL UPDATE SYSTEM
======================================================================
🎯 Tự động cập nhật models mới vào hệ thống production
📊 Model versioning và performance monitoring
🚀 Seamless deployment với rollback capability
"""

import os
import json
import pickle
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import glob

class ModelVersionManager:
    """Quản lý versions của models"""
    
    def __init__(self):
        self.models_dir = "trained_models"
        self.new_models_dir = "optimized_models_100epochs"
        self.backup_dir = "model_backups"
        self.config_file = "model_deployment_config.json"
        
        # Create directories
        for dir_path in [self.models_dir, self.backup_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_current_models_info(self) -> Dict:
        """Lấy thông tin models hiện tại"""
        print("📊 ANALYZING CURRENT MODELS")
        print("-" * 50)
        
        current_models = {}
        
        # Scan trained_models directory
        if os.path.exists(self.models_dir):
            model_files = glob.glob(f"{self.models_dir}/*.pkl") + glob.glob(f"{self.models_dir}/*.keras")
            
            for model_file in model_files:
                file_name = os.path.basename(model_file)
                file_size = os.path.getsize(model_file)
                modified_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                
                current_models[file_name] = {
                    'path': model_file,
                    'size': file_size,
                    'modified': modified_time,
                    'type': 'keras' if file_name.endswith('.keras') else 'pickle'
                }
        
        print(f"✅ Found {len(current_models)} current models")
        return current_models
    
    def get_new_models_info(self) -> Dict:
        """Lấy thông tin models mới (100 epochs)"""
        print("📊 ANALYZING NEW MODELS (100 EPOCHS)")
        print("-" * 50)
        
        new_models = {}
        
        if os.path.exists(self.new_models_dir):
            model_files = glob.glob(f"{self.new_models_dir}/*.pkl")
            
            for model_file in model_files:
                file_name = os.path.basename(model_file)
                file_size = os.path.getsize(model_file)
                modified_time = datetime.fromtimestamp(os.path.getmtime(model_file))
                
                new_models[file_name] = {
                    'path': model_file,
                    'size': file_size,
                    'modified': modified_time,
                    'type': 'pickle',
                    'training_type': '100_epochs_intensive',
                    'performance': self.get_model_performance(file_name)
                }
        
        print(f"✅ Found {len(new_models)} new models")
        return new_models
    
    def get_model_performance(self, model_name: str) -> Dict:
        """Lấy performance metrics của model"""
        # Load performance từ training report
        report_file = "training_reports/comprehensive_training_report_20250619_131327.json"
        
        if os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            model_performance = report.get('model_performance', {})
            
            if 'enhanced_random_forest' in model_name:
                return model_performance.get('enhanced_random_forest', {})
            elif 'neural_network' in model_name:
                return model_performance.get('neural_network_100epochs', {})
            else:
                return {}
        
        return {}
    
    def create_deployment_plan(self) -> Dict:
        """Tạo kế hoạch deployment"""
        print("📋 CREATING DEPLOYMENT PLAN")
        print("-" * 50)
        
        current_models = self.get_current_models_info()
        new_models = self.get_new_models_info()
        
        deployment_plan = {
            'timestamp': datetime.now().isoformat(),
            'deployment_type': 'model_update_100epochs',
            'current_models_count': len(current_models),
            'new_models_count': len(new_models),
            'actions': [],
            'backup_required': True,
            'rollback_plan': True
        }
        
        # Plan actions for each new model
        for model_name, model_info in new_models.items():
            performance = model_info['performance']
            
            action = {
                'model_name': model_name,
                'action_type': 'deploy',
                'source_path': model_info['path'],
                'target_path': f"{self.models_dir}/{model_name}",
                'performance_metrics': performance,
                'deployment_priority': self.get_deployment_priority(performance)
            }
            
            deployment_plan['actions'].append(action)
        
        # Sort by priority
        deployment_plan['actions'].sort(key=lambda x: x['deployment_priority'], reverse=True)
        
        print(f"📊 Deployment plan created:")
        print(f"   🔄 Actions planned: {len(deployment_plan['actions'])}")
        print(f"   🎯 High priority: {sum(1 for a in deployment_plan['actions'] if a['deployment_priority'] >= 0.8)}")
        print(f"   📊 Medium priority: {sum(1 for a in deployment_plan['actions'] if 0.6 <= a['deployment_priority'] < 0.8)}")
        
        return deployment_plan
    
    def get_deployment_priority(self, performance: Dict) -> float:
        """Tính deployment priority dựa trên performance"""
        if not performance:
            return 0.5
        
        test_accuracy = performance.get('test_accuracy', 0.5)
        improvement = performance.get('vs_previous', {}).get('improvement', 0)
        
        # Priority = accuracy + improvement_bonus
        priority = test_accuracy
        if improvement > 0:
            priority += min(improvement * 2, 0.2)  # Bonus for improvement
        
        return min(priority, 1.0)
    
    def backup_current_models(self) -> bool:
        """Backup models hiện tại"""
        print("💾 BACKING UP CURRENT MODELS")
        print("-" * 50)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_subdir = f"{self.backup_dir}/backup_{timestamp}"
        os.makedirs(backup_subdir, exist_ok=True)
        
        try:
            current_models = self.get_current_models_info()
            backed_up = 0
            
            for model_name, model_info in current_models.items():
                source_path = model_info['path']
                backup_path = f"{backup_subdir}/{model_name}"
                
                shutil.copy2(source_path, backup_path)
                backed_up += 1
                print(f"   ✅ Backed up: {model_name}")
            
            # Save backup metadata
            backup_metadata = {
                'timestamp': timestamp,
                'models_backed_up': backed_up,
                'backup_directory': backup_subdir,
                'original_models': current_models
            }
            
            metadata_file = f"{backup_subdir}/backup_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(backup_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Backup completed: {backed_up} models backed up to {backup_subdir}")
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def deploy_new_models(self, deployment_plan: Dict) -> Dict:
        """Deploy models mới"""
        print("🚀 DEPLOYING NEW MODELS")
        print("-" * 50)
        
        deployment_results = {
            'timestamp': datetime.now().isoformat(),
            'total_actions': len(deployment_plan['actions']),
            'successful_deployments': 0,
            'failed_deployments': 0,
            'deployed_models': [],
            'failed_models': []
        }
        
        for action in deployment_plan['actions']:
            try:
                model_name = action['model_name']
                source_path = action['source_path']
                target_path = action['target_path']
                
                # Copy model to production directory
                shutil.copy2(source_path, target_path)
                
                # Verify deployment
                if os.path.exists(target_path):
                    deployment_results['successful_deployments'] += 1
                    deployment_results['deployed_models'].append({
                        'model_name': model_name,
                        'target_path': target_path,
                        'performance': action['performance_metrics'],
                        'priority': action['deployment_priority']
                    })
                    print(f"   ✅ Deployed: {model_name} (priority: {action['deployment_priority']:.2f})")
                else:
                    raise Exception("File not found after copy")
                    
            except Exception as e:
                deployment_results['failed_deployments'] += 1
                deployment_results['failed_models'].append({
                    'model_name': action['model_name'],
                    'error': str(e)
                })
                print(f"   ❌ Failed: {action['model_name']} - {e}")
        
        print(f"\n📊 Deployment Results:")
        print(f"   ✅ Successful: {deployment_results['successful_deployments']}")
        print(f"   ❌ Failed: {deployment_results['failed_deployments']}")
        
        return deployment_results
    
    def update_system_configuration(self, deployment_results: Dict):
        """Cập nhật system configuration"""
        print("⚙️ UPDATING SYSTEM CONFIGURATION")
        print("-" * 50)
        
        # Create/update model configuration
        model_config = {
            'last_update': datetime.now().isoformat(),
            'deployment_type': '100_epochs_intensive',
            'active_models': {},
            'performance_metrics': {},
            'deployment_history': []
        }
        
        # Add deployed models to config
        for deployed_model in deployment_results['deployed_models']:
            model_name = deployed_model['model_name']
            
            # Extract model type and create config entry
            if 'enhanced_random_forest' in model_name:
                model_config['active_models']['enhanced_random_forest'] = {
                    'path': deployed_model['target_path'],
                    'type': 'sklearn_random_forest',
                    'performance': deployed_model['performance'],
                    'priority': deployed_model['priority'],
                    'deployment_date': datetime.now().isoformat()
                }
            elif 'neural_network' in model_name:
                model_config['active_models']['neural_network_100epochs'] = {
                    'path': deployed_model['target_path'],
                    'type': 'sklearn_mlp',
                    'performance': deployed_model['performance'],
                    'priority': deployed_model['priority'],
                    'deployment_date': datetime.now().isoformat()
                }
        
        # Save configuration
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configuration updated: {self.config_file}")
        return model_config

class AutomaticModelUpdater:
    """Hệ thống tự động cập nhật models"""
    
    def __init__(self):
        self.version_manager = ModelVersionManager()
        self.update_log = []
    
    def check_for_new_models(self) -> bool:
        """Kiểm tra có models mới không"""
        print("🔍 CHECKING FOR NEW MODELS")
        print("-" * 50)
        
        new_models = self.version_manager.get_new_models_info()
        
        if not new_models:
            print("ℹ️ No new models found")
            return False
        
        # Check if models are better than current
        has_improvements = False
        for model_name, model_info in new_models.items():
            performance = model_info.get('performance', {})
            improvement = performance.get('vs_previous', {}).get('improvement', 0)
            
            if improvement > 0:
                has_improvements = True
                print(f"   ✅ {model_name}: +{improvement:.3f} improvement")
        
        if has_improvements:
            print("🚀 New improved models found!")
            return True
        else:
            print("ℹ️ New models found but no improvements")
            return False
    
    def perform_automatic_update(self) -> Dict:
        """Thực hiện automatic update"""
        print("🔄 PERFORMING AUTOMATIC MODEL UPDATE")
        print("=" * 70)
        
        update_results = {
            'timestamp': datetime.now().isoformat(),
            'update_type': 'automatic_100_epochs',
            'success': False,
            'steps_completed': [],
            'error': None
        }
        
        try:
            # Step 1: Check for new models
            if not self.check_for_new_models():
                update_results['error'] = "No improved models found"
                return update_results
            
            update_results['steps_completed'].append('new_models_detected')
            
            # Step 2: Create deployment plan
            deployment_plan = self.version_manager.create_deployment_plan()
            update_results['steps_completed'].append('deployment_plan_created')
            
            # Step 3: Backup current models
            if self.version_manager.backup_current_models():
                update_results['steps_completed'].append('current_models_backed_up')
            else:
                raise Exception("Failed to backup current models")
            
            # Step 4: Deploy new models
            deployment_results = self.version_manager.deploy_new_models(deployment_plan)
            
            if deployment_results['successful_deployments'] > 0:
                update_results['steps_completed'].append('new_models_deployed')
                update_results['deployment_results'] = deployment_results
            else:
                raise Exception("No models were successfully deployed")
            
            # Step 5: Update system configuration
            model_config = self.version_manager.update_system_configuration(deployment_results)
            update_results['steps_completed'].append('system_configuration_updated')
            update_results['model_config'] = model_config
            
            # Step 6: Verify deployment
            if self.verify_deployment():
                update_results['steps_completed'].append('deployment_verified')
                update_results['success'] = True
            else:
                raise Exception("Deployment verification failed")
            
            print(f"\n🎉 AUTOMATIC UPDATE COMPLETED SUCCESSFULLY!")
            print(f"📊 Models deployed: {deployment_results['successful_deployments']}")
            print(f"✅ All steps completed: {len(update_results['steps_completed'])}")
            
        except Exception as e:
            update_results['error'] = str(e)
            print(f"\n❌ AUTOMATIC UPDATE FAILED: {e}")
        
        # Log update attempt
        self.update_log.append(update_results)
        
        return update_results
    
    def verify_deployment(self) -> bool:
        """Verify deployment thành công"""
        print("🔍 VERIFYING DEPLOYMENT")
        print("-" * 50)
        
        try:
            # Check if config file exists
            if not os.path.exists(self.version_manager.config_file):
                print("❌ Configuration file not found")
                return False
            
            # Load and verify config
            with open(self.version_manager.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            active_models = config.get('active_models', {})
            
            if not active_models:
                print("❌ No active models in configuration")
                return False
            
            # Verify each model file exists
            verified_models = 0
            for model_name, model_info in active_models.items():
                model_path = model_info['path']
                if os.path.exists(model_path):
                    verified_models += 1
                    print(f"   ✅ {model_name}: {model_path}")
                else:
                    print(f"   ❌ {model_name}: File not found - {model_path}")
            
            if verified_models == len(active_models):
                print(f"✅ Deployment verified: {verified_models} models active")
                return True
            else:
                print(f"❌ Verification failed: {verified_models}/{len(active_models)} models found")
                return False
                
        except Exception as e:
            print(f"❌ Verification error: {e}")
            return False
    
    def create_model_loader_integration(self):
        """Tạo integration code cho hệ thống chính"""
        print("🔧 CREATING SYSTEM INTEGRATION")
        print("-" * 50)
        
        integration_code = '''"""
AUTOMATIC MODEL LOADER INTEGRATION
Generated by Automatic Model Update System
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class ProductionModelLoader:
    """Load models được deploy tự động"""
    
    def __init__(self, config_file="model_deployment_config.json"):
        self.config_file = config_file
        self.loaded_models = {}
        self.model_config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load model deployment configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def get_active_models(self) -> Dict:
        """Get list of active models"""
        return self.model_config.get('active_models', {})
    
    def load_best_model(self) -> Optional[Any]:
        """Load model với performance tốt nhất"""
        active_models = self.get_active_models()
        
        if not active_models:
            return None
        
        # Find model with highest priority
        best_model_name = None
        best_priority = 0
        
        for model_name, model_info in active_models.items():
            priority = model_info.get('priority', 0)
            if priority > best_priority:
                best_priority = priority
                best_model_name = model_name
        
        if best_model_name:
            return self.load_model(best_model_name)
        
        return None
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load specific model"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        active_models = self.get_active_models()
        
        if model_name not in active_models:
            return None
        
        model_info = active_models[model_name]
        model_path = model_info['path']
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.loaded_models[model_name] = model
            logging.info(f"Loaded production model: {model_name}")
            
            return model
            
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            return None
    
    def get_model_performance(self, model_name: str) -> Dict:
        """Get performance metrics của model"""
        active_models = self.get_active_models()
        
        if model_name in active_models:
            return active_models[model_name].get('performance', {})
        
        return {}
    
    def predict_with_best_model(self, X) -> Dict:
        """Predict sử dụng model tốt nhất"""
        model = self.load_best_model()
        
        if model is None:
            return {'prediction': 0.5, 'confidence': 0.0, 'model_used': None}
        
        try:
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(X)
                prediction = prediction_proba[:, 1] if prediction_proba.shape[1] > 1 else prediction_proba[:, 0]
                confidence = max(prediction_proba[0])
            else:
                prediction = model.predict(X)
                confidence = 0.7  # Default confidence
            
            return {
                'prediction': float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction),
                'confidence': float(confidence),
                'model_used': self.get_best_model_name(),
                'deployment_date': self.model_config.get('last_update'),
                'model_type': '100_epochs_intensive'
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return {'prediction': 0.5, 'confidence': 0.0, 'model_used': None, 'error': str(e)}
    
    def get_best_model_name(self) -> Optional[str]:
        """Get tên của model tốt nhất"""
        active_models = self.get_active_models()
        
        if not active_models:
            return None
        
        best_model_name = None
        best_priority = 0
        
        for model_name, model_info in active_models.items():
            priority = model_info.get('priority', 0)
            if priority > best_priority:
                best_priority = priority
                best_model_name = model_name
        
        return best_model_name

# Global instance for easy import
production_model_loader = ProductionModelLoader()
'''
        
        # Save integration code
        integration_file = "production_model_loader.py"
        with open(integration_file, 'w', encoding='utf-8') as f:
            f.write(integration_code)
        
        print(f"✅ Integration code created: {integration_file}")
        
        # Create usage example
        example_code = '''# USAGE EXAMPLE - How to use in your trading system

from production_model_loader import production_model_loader

# Load best model and make prediction
def get_trading_signal(features):
    """Get trading signal using best available model"""
    
    # Reshape features for prediction
    X = features.reshape(1, -1)
    
    # Get prediction from best model
    result = production_model_loader.predict_with_best_model(X)
    
    # Convert to trading signal
    prediction = result['prediction']
    confidence = result['confidence']
    model_used = result['model_used']
    
    # Decision logic
    if prediction > 0.6 and confidence > 0.7:
        return {
            'signal': 'BUY',
            'confidence': confidence,
            'model': model_used,
            'prediction_value': prediction
        }
    elif prediction < 0.4 and confidence > 0.7:
        return {
            'signal': 'SELL', 
            'confidence': confidence,
            'model': model_used,
            'prediction_value': prediction
        }
    else:
        return {
            'signal': 'HOLD',
            'confidence': confidence,
            'model': model_used,
            'prediction_value': prediction
        }

# Example usage
# signal = get_trading_signal(your_features)
# print(f"Signal: {signal['signal']}, Confidence: {signal['confidence']:.2f}")
'''
        
        example_file = "model_integration_example.py"
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(example_code)
        
        print(f"✅ Usage example created: {example_file}")
        
        return integration_file, example_file

def main():
    """Main function để chạy automatic update"""
    print("🚀 AUTOMATIC MODEL UPDATE SYSTEM")
    print("=" * 70)
    
    # Initialize updater
    updater = AutomaticModelUpdater()
    
    # Perform automatic update
    update_results = updater.perform_automatic_update()
    
    if update_results['success']:
        # Create integration code
        integration_file, example_file = updater.create_model_loader_integration()
        
        print(f"\n🎉 SYSTEM UPDATE COMPLETED!")
        print(f"✅ Models deployed and integrated")
        print(f"📄 Integration file: {integration_file}")
        print(f"📄 Example usage: {example_file}")
        print(f"⚙️ Configuration: model_deployment_config.json")
        
    else:
        print(f"\n❌ SYSTEM UPDATE FAILED!")
        print(f"Error: {update_results.get('error', 'Unknown error')}")
    
    return update_results

if __name__ == "__main__":
    main() 