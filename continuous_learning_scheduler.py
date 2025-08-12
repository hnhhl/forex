#!/usr/bin/env python3
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
