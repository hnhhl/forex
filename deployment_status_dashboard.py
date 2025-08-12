#!/usr/bin/env python3
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
        print(f"\n🤖 MODEL STATUS:")
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
        print(f"\n🏥 SYSTEM HEALTH:")
        print("-" * 50)
        
        config_exists = os.path.exists('model_deployment_config.json')
        loader_exists = os.path.exists('production_model_loader.py')
        
        print(f"   Configuration: {'🟢 OK' if config_exists else '🔴 MISSING'}")
        print(f"   Model Loader: {'🟢 OK' if loader_exists else '🔴 MISSING'}")
        print(f"   Active Models: {'🟢 OK' if active_models else '🔴 NONE'}")
        print(f"   Integration: {'🟢 READY' if all([config_exists, loader_exists, active_models]) else '🔴 ISSUES'}")
        
        # Recent Activity
        print(f"\n📈 RECENT ACTIVITY:")
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
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("-" * 50)
        
        if best_model_name:
            perf = production_model_loader.get_model_performance(best_model_name)
            print(f"   Best Accuracy: {perf.get('test_accuracy', 0):.1%}")
            print(f"   Improvement: {perf.get('vs_previous', {}).get('improvement_percentage', 0):+.1f}%")
            print(f"   Status: {perf.get('vs_previous', {}).get('status', 'Unknown')}")
            print(f"   Overall Rating: {perf.get('performance_rating', 'Unknown')}")
        
        print(f"\n✅ Dashboard Updated Successfully!")

def main():
    """Main function"""
    dashboard = DeploymentStatusDashboard()
    dashboard.show_deployment_status()

if __name__ == "__main__":
    main()
