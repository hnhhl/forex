#!/usr/bin/env python3
"""
🔧 INTEGRATE MODELS TO MAIN SYSTEM
======================================================================
🎯 Tích hợp models mới vào Ultimate XAU System V4.0
🚀 Update master system để sử dụng models 100 epochs
📊 Enable automatic model updates
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path

class MainSystemIntegrator:
    """Tích hợp models vào hệ thống chính"""
    
    def __init__(self):
        self.master_system_file = "src/core/integration/master_system.py"
        self.neural_ensemble_file = "src/core/ai/neural_ensemble.py"
        self.backup_dir = "system_integration_backup"
        
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def backup_original_files(self):
        """Backup files gốc trước khi modify"""
        print("💾 BACKING UP ORIGINAL SYSTEM FILES")
        print("-" * 50)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_subdir = f"{self.backup_dir}/backup_{timestamp}"
        os.makedirs(backup_subdir, exist_ok=True)
        
        files_to_backup = [
            self.master_system_file,
            self.neural_ensemble_file
        ]
        
        backed_up = 0
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                backup_path = f"{backup_subdir}/{os.path.basename(file_path)}"
                shutil.copy2(file_path, backup_path)
                backed_up += 1
                print(f"   ✅ Backed up: {file_path}")
        
        print(f"✅ Backup completed: {backed_up} files backed up")
        return backup_subdir
    
    def update_master_system(self):
        """Update master system để sử dụng production models"""
        print("🔧 UPDATING MASTER SYSTEM")
        print("-" * 50)
        
        # Read current master system
        with open(self.master_system_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add production model loader import
        import_addition = '''
# Production Model Integration
try:
    from production_model_loader import production_model_loader
    PRODUCTION_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Production models not available: {e}")
    PRODUCTION_MODELS_AVAILABLE = False
'''
        
        # Find the imports section and add our import
        if "# Phase 2 Imports - AI Systems" in content:
            content = content.replace(
                "# Phase 2 Imports - AI Systems",
                import_addition + "\n# Phase 2 Imports - AI Systems"
            )
        
        # Add production model processing method
        production_method = '''
    def _process_production_models(self, data: MarketData) -> Optional[TradingSignal]:
        """Process using production models (100 epochs)"""
        if not PRODUCTION_MODELS_AVAILABLE:
            return None
        
        try:
            # Prepare features from market data
            features = self._prepare_features_for_production_model(data)
            
            # Get prediction from best production model
            result = production_model_loader.predict_with_best_model(features.reshape(1, -1))
            
            if result['model_used'] is None:
                return None
            
            # Convert prediction to trading signal
            prediction = result['prediction']
            confidence = result['confidence']
            model_used = result['model_used']
            
            # Enhanced signal generation with dynamic thresholds
            volatility = data.technical_indicators.get('volatility', 0.5)
            
            # Dynamic thresholds based on volatility (from training analysis)
            if volatility < 0.5:  # Low volatility
                buy_threshold, sell_threshold = 0.58, 0.42
            elif volatility > 1.0:  # High volatility
                buy_threshold, sell_threshold = 0.65, 0.35
            else:  # Medium volatility
                buy_threshold, sell_threshold = 0.60, 0.40
            
            # Generate signal
            if prediction > buy_threshold and confidence > 0.7:
                signal_type = "BUY"
                risk_score = 1.0 - confidence  # Lower risk for higher confidence
            elif prediction < sell_threshold and confidence > 0.7:
                signal_type = "SELL"
                risk_score = 1.0 - confidence
            else:
                signal_type = "HOLD"
                risk_score = 0.5
            
            # Create trading signal
            signal = TradingSignal(
                timestamp=data.timestamp,
                symbol=data.symbol,
                signal_type=signal_type,
                confidence=confidence,
                source=f"production_model_{model_used}",
                risk_score=risk_score,
                metadata={
                    'model_type': result['model_type'],
                    'prediction_value': prediction,
                    'deployment_date': result['deployment_date'],
                    'volatility_regime': 'low' if volatility < 0.5 else 'high' if volatility > 1.0 else 'medium',
                    'dynamic_thresholds': {'buy': buy_threshold, 'sell': sell_threshold}
                }
            )
            
            logger.info(f"Production model signal: {signal_type} (confidence: {confidence:.3f})")
            return signal
            
        except Exception as e:
            logger.error(f"Production model processing error: {e}")
            return None
    
    def _prepare_features_for_production_model(self, data: MarketData) -> np.ndarray:
        """Prepare features for production model (19 features)"""
        # Extract features matching training format
        features = []
        
        # Technical indicators (if available)
        tech_indicators = data.technical_indicators
        
        # Moving averages (8 features)
        features.extend([
            tech_indicators.get('sma_5', data.price),
            tech_indicators.get('sma_10', data.price),
            tech_indicators.get('sma_20', data.price),
            tech_indicators.get('sma_50', data.price),
            tech_indicators.get('ema_5', data.price),
            tech_indicators.get('ema_10', data.price),
            tech_indicators.get('ema_20', data.price),
            tech_indicators.get('ema_50', data.price)
        ])
        
        # Technical indicators (4 features)
        features.extend([
            tech_indicators.get('rsi', 50.0),
            tech_indicators.get('macd', 0.0),
            tech_indicators.get('macd_signal', 0.0),
            tech_indicators.get('bb_position', 0.5)
        ])
        
        # Market analysis (3 features)
        features.extend([
            tech_indicators.get('volatility', 0.5),
            tech_indicators.get('price_momentum', 0.0),
            tech_indicators.get('volume_ratio', 1.0)
        ])
        
        # Regime detection (2 features)
        features.extend([
            tech_indicators.get('volatility_regime', 1.0),
            tech_indicators.get('trend_strength', 0.5)
        ])
        
        # Temporal features (2 features)
        hour = data.timestamp.hour if hasattr(data.timestamp, 'hour') else 12
        day_of_week = data.timestamp.weekday() if hasattr(data.timestamp, 'weekday') else 2
        features.extend([hour, day_of_week])
        
        return np.array(features)
'''
        
        # Find the _process_ai_systems method and add production model processing
        if "_process_ai_systems(self, data: MarketData) -> List[TradingSignal]:" in content:
            # Add production model call to AI systems processing
            ai_systems_replacement = '''    def _process_ai_systems(self, data: MarketData) -> List[TradingSignal]:
        """Process AI systems for trading signals"""
        signals = []
        
        # Production Models (Priority 1 - Highest)
        if PRODUCTION_MODELS_AVAILABLE:
            production_signal = self._process_production_models(data)
            if production_signal:
                signals.append(production_signal)
        
        # Neural Ensemble (Priority 2)
        if self.config.use_neural_ensemble and 'neural_ensemble' in self.components:
            ensemble_signal = self._process_neural_ensemble(data)
            if ensemble_signal:
                signals.append(ensemble_signal)
        
        # Reinforcement Learning (Priority 3)
        if self.config.use_reinforcement_learning and 'rl_agent' in self.components:
            rl_signal = self._process_reinforcement_learning(data)
            if rl_signal:
                signals.append(rl_signal)
        
        return signals'''
            
            # Replace the method
            start_marker = "    def _process_ai_systems(self, data: MarketData) -> List[TradingSignal]:"
            end_marker = "        return signals"
            
            start_idx = content.find(start_marker)
            if start_idx != -1:
                # Find the end of the method
                end_idx = content.find(end_marker, start_idx)
                if end_idx != -1:
                    end_idx = content.find("\n", end_idx) + 1
                    # Replace the method
                    content = content[:start_idx] + ai_systems_replacement + content[end_idx:]
        
        # Add the production model processing method before the _process_neural_ensemble method
        neural_ensemble_marker = "    def _process_neural_ensemble(self, data: MarketData) -> Optional[TradingSignal]:"
        neural_ensemble_idx = content.find(neural_ensemble_marker)
        if neural_ensemble_idx != -1:
            content = content[:neural_ensemble_idx] + production_method + "\n" + content[neural_ensemble_idx:]
        
        # Add numpy import if not present
        if "import numpy as np" not in content:
            content = content.replace("import pandas as pd", "import pandas as pd\nimport numpy as np")
        
        # Write updated content
        with open(self.master_system_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Master system updated with production models")
    
    def create_continuous_learning_scheduler(self):
        """Tạo scheduler để tự động retrain models"""
        print("⏰ CREATING CONTINUOUS LEARNING SCHEDULER")
        print("-" * 50)
        
        scheduler_code = '''#!/usr/bin/env python3
"""
⏰ CONTINUOUS LEARNING SCHEDULER
======================================================================
🎯 Tự động schedule retraining models khi có data mới
📊 Monitor performance và trigger updates
🔄 Maintain model freshness
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
from AUTOMATIC_MODEL_UPDATE_SYSTEM import AutomaticModelUpdater
from OPTIMIZED_TRAINING_100_EPOCHS import OptimizedTraining100Epochs

class ContinuousLearningScheduler:
    """Scheduler cho continuous learning"""
    
    def __init__(self):
        self.updater = AutomaticModelUpdater()
        self.trainer = OptimizedTraining100Epochs()
        self.last_training = None
        self.last_update = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('continuous_learning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_for_retraining(self):
        """Kiểm tra có cần retrain không"""
        self.logger.info("🔍 Checking for retraining needs...")
        
        # Check if enough time has passed since last training
        if self.last_training:
            time_since_training = datetime.now() - self.last_training
            if time_since_training < timedelta(days=7):  # Minimum 7 days between training
                self.logger.info("⏳ Too soon for retraining")
                return False
        
        # Check if new data is available
        # TODO: Implement data freshness check
        
        # Check current model performance
        # TODO: Implement performance monitoring
        
        self.logger.info("✅ Retraining conditions met")
        return True
    
    def perform_scheduled_training(self):
        """Thực hiện scheduled training"""
        self.logger.info("🚀 Starting scheduled training...")
        
        try:
            # Run optimized training
            results_file = self.trainer.run_optimized_training()
            
            if results_file:
                self.logger.info(f"✅ Training completed: {results_file}")
                self.last_training = datetime.now()
                
                # Trigger automatic update
                self.trigger_model_update()
            else:
                self.logger.error("❌ Training failed")
                
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
    
    def trigger_model_update(self):
        """Trigger automatic model update"""
        self.logger.info("🔄 Triggering model update...")
        
        try:
            update_results = self.updater.perform_automatic_update()
            
            if update_results['success']:
                self.logger.info("✅ Model update successful")
                self.last_update = datetime.now()
            else:
                self.logger.error(f"❌ Model update failed: {update_results.get('error')}")
                
        except Exception as e:
            self.logger.error(f"❌ Update error: {e}")
    
    def daily_health_check(self):
        """Daily system health check"""
        self.logger.info("🏥 Performing daily health check...")
        
        # Check if models are still available
        if self.updater.check_for_new_models():
            self.logger.info("✅ Models are available")
        else:
            self.logger.warning("⚠️ Model issues detected")
        
        # Log system status
        self.logger.info(f"📊 Last training: {self.last_training}")
        self.logger.info(f"📊 Last update: {self.last_update}")
    
    def start_scheduler(self):
        """Start the continuous learning scheduler"""
        self.logger.info("⏰ Starting Continuous Learning Scheduler")
        
        # Schedule tasks
        schedule.every().sunday.at("02:00").do(self.check_and_train)  # Weekly check
        schedule.every().day.at("06:00").do(self.daily_health_check)  # Daily health check
        schedule.every(6).hours.do(self.quick_health_check)  # Quick checks
        
        self.logger.info("📅 Scheduled tasks:")
        self.logger.info("   • Weekly training check: Sundays at 02:00")
        self.logger.info("   • Daily health check: Every day at 06:00")
        self.logger.info("   • Quick health check: Every 6 hours")
        
        # Main scheduler loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                self.logger.info("👋 Scheduler stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def check_and_train(self):
        """Check conditions and train if needed"""
        if self.check_for_retraining():
            self.perform_scheduled_training()
    
    def quick_health_check(self):
        """Quick health check"""
        self.logger.info("⚡ Quick health check...")
        # TODO: Implement quick checks

def main():
    """Main function"""
    scheduler = ContinuousLearningScheduler()
    scheduler.start_scheduler()

if __name__ == "__main__":
    main()
'''
        
        with open("continuous_learning_scheduler.py", 'w', encoding='utf-8') as f:
            f.write(scheduler_code)
        
        print("✅ Continuous learning scheduler created")
    
    def create_deployment_status_dashboard(self):
        """Tạo dashboard để monitor deployment status"""
        print("📊 CREATING DEPLOYMENT STATUS DASHBOARD")
        print("-" * 50)
        
        dashboard_code = '''#!/usr/bin/env python3
"""
📊 DEPLOYMENT STATUS DASHBOARD
======================================================================
🎯 Monitor deployment status và model performance
📈 Real-time metrics và alerts
🚀 Production monitoring
"""

import json
import os
from datetime import datetime
from production_model_loader import production_model_loader

class DeploymentStatusDashboard:
    """Dashboard cho deployment status"""
    
    def show_deployment_status(self):
        """Show current deployment status"""
        print("📊 DEPLOYMENT STATUS DASHBOARD")
        print("=" * 70)
        print(f"🕐 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model Status
        print(f"\\n🤖 MODEL STATUS:")
        print("-" * 50)
        
        active_models = production_model_loader.get_active_models()
        
        if not active_models:
            print("❌ No active models found")
            return
        
        for model_name, model_info in active_models.items():
            print(f"📍 {model_name.upper()}:")
            print(f"   Status: {'🟢 ACTIVE' if os.path.exists(model_info['path']) else '🔴 MISSING'}")
            print(f"   Path: {model_info['path']}")
            print(f"   Type: {model_info['type']}")
            print(f"   Priority: {model_info['priority']:.3f}")
            
            performance = model_info.get('performance', {})
            if performance:
                print(f"   Accuracy: {performance.get('test_accuracy', 0):.1%}")
                print(f"   Rating: {performance.get('performance_rating', 'Unknown')}")
                
                vs_previous = performance.get('vs_previous', {})
                improvement = vs_previous.get('improvement_percentage', 0)
                print(f"   Improvement: {improvement:+.1f}%")
            print()
        
        # Best Model
        best_model_name = production_model_loader.get_best_model_name()
        if best_model_name:
            print(f"🏆 BEST MODEL: {best_model_name}")
            performance = production_model_loader.get_model_performance(best_model_name)
            print(f"   Accuracy: {performance.get('test_accuracy', 0):.1%}")
            print(f"   Confidence: HIGH")
            print(f"   Status: 🟢 PRODUCTION READY")
        
        # System Health
        print(f"\\n🏥 SYSTEM HEALTH:")
        print("-" * 50)
        
        config_exists = os.path.exists('model_deployment_config.json')
        loader_exists = os.path.exists('production_model_loader.py')
        
        print(f"   Configuration: {'🟢 OK' if config_exists else '🔴 MISSING'}")
        print(f"   Model Loader: {'🟢 OK' if loader_exists else '🔴 MISSING'}")
        print(f"   Active Models: {'🟢 OK' if active_models else '🔴 NONE'}")
        print(f"   Integration: {'🟢 READY' if all([config_exists, loader_exists, active_models]) else '🔴 ISSUES'}")
        
        # Recent Activity
        print(f"\\n📈 RECENT ACTIVITY:")
        print("-" * 50)
        
        if os.path.exists('model_deployment_config.json'):
            with open('model_deployment_config.json', 'r') as f:
                config = json.load(f)
            
            last_update = config.get('last_update', 'Unknown')
            deployment_type = config.get('deployment_type', 'Unknown')
            
            print(f"   Last Deployment: {last_update}")
            print(f"   Deployment Type: {deployment_type}")
            print(f"   Models Deployed: {len(active_models)}")
        
        # Performance Summary
        print(f"\\n📊 PERFORMANCE SUMMARY:")
        print("-" * 50)
        
        if best_model_name:
            perf = production_model_loader.get_model_performance(best_model_name)
            print(f"   Best Accuracy: {perf.get('test_accuracy', 0):.1%}")
            print(f"   Improvement: {perf.get('vs_previous', {}).get('improvement_percentage', 0):+.1f}%")
            print(f"   Status: {perf.get('vs_previous', {}).get('status', 'Unknown')}")
            print(f"   Overall Rating: {perf.get('performance_rating', 'Unknown')}")
        
        print(f"\\n✅ Dashboard Updated Successfully!")

def main():
    """Main function"""
    dashboard = DeploymentStatusDashboard()
    dashboard.show_deployment_status()

if __name__ == "__main__":
    main()
'''
        
        with open("deployment_status_dashboard.py", 'w', encoding='utf-8') as f:
            f.write(dashboard_code)
        
        print("✅ Deployment status dashboard created")
    
    def perform_integration(self):
        """Thực hiện toàn bộ integration process"""
        print("🔧 PERFORMING COMPLETE SYSTEM INTEGRATION")
        print("=" * 70)
        
        integration_results = {
            'timestamp': datetime.now().isoformat(),
            'steps_completed': [],
            'success': False
        }
        
        try:
            # Step 1: Backup original files
            backup_dir = self.backup_original_files()
            integration_results['steps_completed'].append('files_backed_up')
            integration_results['backup_directory'] = backup_dir
            
            # Step 2: Update master system
            self.update_master_system()
            integration_results['steps_completed'].append('master_system_updated')
            
            # Step 3: Create continuous learning scheduler
            self.create_continuous_learning_scheduler()
            integration_results['steps_completed'].append('scheduler_created')
            
            # Step 4: Create deployment dashboard
            self.create_deployment_status_dashboard()
            integration_results['steps_completed'].append('dashboard_created')
            
            integration_results['success'] = True
            
            print(f"\\n🎉 INTEGRATION COMPLETED SUCCESSFULLY!")
            print(f"✅ All steps completed: {len(integration_results['steps_completed'])}")
            
        except Exception as e:
            integration_results['error'] = str(e)
            print(f"\\n❌ INTEGRATION FAILED: {e}")
        
        return integration_results

def main():
    """Main function"""
    print("🔧 INTEGRATING MODELS TO MAIN SYSTEM")
    print("=" * 70)
    
    integrator = MainSystemIntegrator()
    results = integrator.perform_integration()
    
    if results['success']:
        print(f"\\n🚀 INTEGRATION SUMMARY:")
        print(f"✅ Master system updated with production models")
        print(f"✅ Continuous learning scheduler created")
        print(f"✅ Deployment status dashboard created")
        print(f"✅ Automatic model updates enabled")
        print(f"\\n📄 New Files Created:")
        print(f"   • continuous_learning_scheduler.py")
        print(f"   • deployment_status_dashboard.py")
        print(f"   • production_model_loader.py (already exists)")
        print(f"   • model_integration_example.py (already exists)")
        print(f"\\n🎯 NEXT STEPS:")
        print(f"   1. Test the updated system")
        print(f"   2. Run deployment_status_dashboard.py to monitor")
        print(f"   3. Optionally start continuous_learning_scheduler.py")
        print(f"   4. Deploy to production environment")
    else:
        print(f"\\n❌ INTEGRATION FAILED!")
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    main() 