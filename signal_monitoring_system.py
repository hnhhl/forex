# -*- coding: utf-8 -*-
"""AI3.0 Signal Monitoring System - Real-time Signal Tracking"""

import sys
import os
sys.path.append('src')

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import threading
import warnings
warnings.filterwarnings('ignore')

class SignalMonitor:
    def __init__(self):
        self.signals_history = []
        self.signal_counts = defaultdict(int)
        self.confidence_history = deque(maxlen=100)
        self.consensus_history = deque(maxlen=100)
        self.performance_metrics = {}
        self.running = False
        self.start_time = None
        
    def initialize_system(self):
        """Khởi tạo hệ thống AI3.0"""
        try:
            from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            config = SystemConfig()
            config.symbol = "XAUUSDc"
            self.system = UltimateXAUSystem(config)
            
            print("✅ AI3.0 System initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            return False
    
    def generate_signal_sample(self):
        """Tạo một signal sample và thu thập metrics"""
        try:
            signal = self.system.generate_signal("XAUUSDc")
            
            signal_data = {
                'timestamp': datetime.now(),
                'action': signal.get('action', 'UNKNOWN'),
                'confidence': signal.get('confidence', 0),
                'method': signal.get('ensemble_method', 'unknown'),
                'hybrid_metrics': signal.get('hybrid_metrics', {}),
                'price': signal.get('current_price', 0)
            }
            
            # Lưu vào history
            self.signals_history.append(signal_data)
            self.signal_counts[signal_data['action']] += 1
            self.confidence_history.append(signal_data['confidence'])
            
            # Lưu consensus nếu có
            if 'hybrid_metrics' in signal and signal['hybrid_metrics']:
                consensus = signal['hybrid_metrics'].get('hybrid_consensus', 0)
                self.consensus_history.append(consensus)
            
            return signal_data
            
        except Exception as e:
            print(f"❌ Error generating signal: {e}")
            return None
    
    def calculate_diversity_metrics(self):
        """Tính toán metrics về signal diversity"""
        if not self.signals_history:
            return {}
        
        recent_signals = self.signals_history[-50:]  # 50 signals gần nhất
        actions = [s['action'] for s in recent_signals]
        
        # Diversity metrics
        unique_actions = set(actions)
        diversity_ratio = len(unique_actions) / len(set(['BUY', 'SELL', 'HOLD']))
        
        # Distribution
        action_distribution = {}
        for action in ['BUY', 'SELL', 'HOLD']:
            count = actions.count(action)
            action_distribution[action] = {
                'count': count,
                'percentage': (count / len(actions)) * 100 if actions else 0
            }
        
        # Confidence stats
        confidences = [s['confidence'] for s in recent_signals]
        
        return {
            'total_signals': len(self.signals_history),
            'recent_signals': len(recent_signals),
            'unique_actions': list(unique_actions),
            'diversity_ratio': diversity_ratio,
            'action_distribution': action_distribution,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0,
            'avg_consensus': np.mean(list(self.consensus_history)) if self.consensus_history else 0
        }
    
    def print_real_time_status(self, signal_data, metrics):
        """In trạng thái real-time"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Header
        print(f"\n🔴 LIVE SIGNAL MONITORING - {current_time}")
        print("=" * 60)
        
        # Current Signal
        print(f"📊 CURRENT SIGNAL:")
        print(f"   🎯 Action: {signal_data['action']}")
        print(f"   📈 Confidence: {signal_data['confidence']:.1%}")
        print(f"   💰 Price: ${signal_data['price']:.2f}")
        print(f"   🔧 Method: {signal_data['method']}")
        
        if signal_data['hybrid_metrics']:
            consensus = signal_data['hybrid_metrics'].get('hybrid_consensus', 0)
            print(f"   🤝 Consensus: {consensus:.1%}")
        
        # Diversity Analysis
        print(f"\n📈 SIGNAL DIVERSITY ANALYSIS:")
        dist = metrics['action_distribution']
        print(f"   📊 BUY: {dist['BUY']['count']} ({dist['BUY']['percentage']:.1f}%)")
        print(f"   📊 SELL: {dist['SELL']['count']} ({dist['SELL']['percentage']:.1f}%)")
        print(f"   📊 HOLD: {dist['HOLD']['count']} ({dist['HOLD']['percentage']:.1f}%)")
        print(f"   🎲 Diversity: {metrics['diversity_ratio']:.1%}")
        
        # Performance Stats
        print(f"\n⚡ PERFORMANCE STATS:")
        print(f"   📊 Total Signals: {metrics['total_signals']}")
        print(f"   📈 Avg Confidence: {metrics['avg_confidence']:.1%}")
        print(f"   🤝 Avg Consensus: {metrics['avg_consensus']:.1%}")
        print(f"   ⏱️ Runtime: {self.get_runtime()}")
        
        # Signal Quality Assessment
        self.assess_signal_quality(metrics)
    
    def assess_signal_quality(self, metrics):
        """Đánh giá chất lượng signal"""
        print(f"\n🎯 SIGNAL QUALITY ASSESSMENT:")
        
        # Diversity Assessment
        diversity = metrics['diversity_ratio']
        if diversity >= 0.67:  # All 3 types
            print("   ✅ Signal Diversity: EXCELLENT")
        elif diversity >= 0.33:  # 2 types
            print("   ⚡ Signal Diversity: GOOD")
        else:  # 1 type
            print("   ⚠️ Signal Diversity: LIMITED")
        
        # Confidence Assessment
        avg_conf = metrics['avg_confidence']
        if avg_conf >= 0.4:
            print("   ✅ Confidence Level: HIGH")
        elif avg_conf >= 0.25:
            print("   ⚡ Confidence Level: GOOD")
        else:
            print("   ⚠️ Confidence Level: LOW")
        
        # Consensus Assessment
        avg_consensus = metrics['avg_consensus']
        if avg_consensus >= 0.7:
            print("   ✅ System Consensus: HIGH")
        elif avg_consensus >= 0.5:
            print("   ⚡ System Consensus: GOOD")
        else:
            print("   ⚠️ System Consensus: LOW")
        
        # Overall Assessment
        quality_score = (diversity + avg_conf + avg_consensus) / 3
        if quality_score >= 0.6:
            print("   🎉 Overall Quality: EXCELLENT")
        elif quality_score >= 0.4:
            print("   ⚡ Overall Quality: GOOD")
        else:
            print("   ⚠️ Overall Quality: NEEDS IMPROVEMENT")
    
    def get_runtime(self):
        """Lấy thời gian chạy"""
        if self.start_time:
            runtime = datetime.now() - self.start_time
            return str(runtime).split('.')[0]
        return "Unknown"
    
    def save_monitoring_data(self):
        """Lưu dữ liệu monitoring"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signal_monitoring_{timestamp}.json"
            
            data = {
                'monitoring_session': {
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': datetime.now().isoformat(),
                    'total_signals': len(self.signals_history),
                    'runtime': self.get_runtime()
                },
                'signals_history': [
                    {
                        'timestamp': s['timestamp'].isoformat(),
                        'action': s['action'],
                        'confidence': s['confidence'],
                        'price': s['price'],
                        'method': s['method']
                    } for s in self.signals_history
                ],
                'signal_counts': dict(self.signal_counts),
                'final_metrics': self.calculate_diversity_metrics()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Monitoring data saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Failed to save data: {e}")
            return None
    
    def start_monitoring(self, duration_minutes=10, interval_seconds=5):
        """Bắt đầu monitoring"""
        print("🚀 STARTING AI3.0 SIGNAL MONITORING SYSTEM")
        print("=" * 60)
        
        if not self.initialize_system():
            return
        
        self.running = True
        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(minutes=duration_minutes)
        
        print(f"⏰ Monitoring Duration: {duration_minutes} minutes")
        print(f"🔄 Check Interval: {interval_seconds} seconds")
        print(f"🎯 End Time: {end_time.strftime('%H:%M:%S')}")
        print(f"⏹️ Press Ctrl+C to stop early")
        
        signal_count = 0
        
        try:
            while self.running and datetime.now() < end_time:
                # Generate signal
                signal_data = self.generate_signal_sample()
                
                if signal_data:
                    signal_count += 1
                    
                    # Calculate metrics
                    metrics = self.calculate_diversity_metrics()
                    
                    # Print status
                    self.print_real_time_status(signal_data, metrics)
                    
                    # Special alerts
                    self.check_alerts(signal_data, metrics)
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
        
        finally:
            self.running = False
            self.print_final_summary()
            self.save_monitoring_data()
    
    def check_alerts(self, signal_data, metrics):
        """Kiểm tra và đưa ra alerts"""
        alerts = []
        
        # Low confidence alert
        if signal_data['confidence'] < 0.2:
            alerts.append(f"🚨 LOW CONFIDENCE: {signal_data['confidence']:.1%}")
        
        # High confidence alert
        if signal_data['confidence'] > 0.5:
            alerts.append(f"🔥 HIGH CONFIDENCE: {signal_data['confidence']:.1%}")
        
        # Diversity improvement alert
        if metrics['diversity_ratio'] > 0.33 and len(self.signals_history) > 10:
            if len(set([s['action'] for s in self.signals_history[-10:]])) > 1:
                alerts.append("🎉 SIGNAL DIVERSITY IMPROVED!")
        
        # Consensus alerts
        if signal_data['hybrid_metrics']:
            consensus = signal_data['hybrid_metrics'].get('hybrid_consensus', 0)
            if consensus > 0.8:
                alerts.append(f"🤝 HIGH CONSENSUS: {consensus:.1%}")
            elif consensus < 0.3:
                alerts.append(f"⚠️ LOW CONSENSUS: {consensus:.1%}")
        
        if alerts:
            print(f"\n🚨 ALERTS:")
            for alert in alerts:
                print(f"   {alert}")
    
    def print_final_summary(self):
        """In tóm tắt cuối cùng"""
        print("\n" + "=" * 60)
        print("📊 FINAL MONITORING SUMMARY")
        print("=" * 60)
        
        metrics = self.calculate_diversity_metrics()
        
        print(f"⏱️ Total Runtime: {self.get_runtime()}")
        print(f"📊 Total Signals Generated: {metrics['total_signals']}")
        
        print(f"\n📈 SIGNAL DISTRIBUTION:")
        dist = metrics['action_distribution']
        for action in ['BUY', 'SELL', 'HOLD']:
            count = dist[action]['count']
            pct = dist[action]['percentage']
            print(f"   {action}: {count} signals ({pct:.1f}%)")
        
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"   📈 Average Confidence: {metrics['avg_confidence']:.1%}")
        print(f"   🤝 Average Consensus: {metrics['avg_consensus']:.1%}")
        print(f"   🎲 Signal Diversity: {metrics['diversity_ratio']:.1%}")
        
        # Final assessment
        self.assess_signal_quality(metrics)

def main():
    print("🎯 AI3.0 SIGNAL MONITORING SYSTEM")
    print("=" * 60)
    
    monitor = SignalMonitor()
    
    # Configuration
    print("⚙️ MONITORING CONFIGURATION:")
    print("   Duration: 10 minutes (default)")
    print("   Interval: 5 seconds between signals")
    print("   Symbol: XAUUSDc")
    print("   Method: hybrid_ai2_ai3_consensus")
    
    # Start monitoring
    try:
        monitor.start_monitoring(duration_minutes=10, interval_seconds=5)
    except Exception as e:
        print(f"❌ Critical error: {e}")

if __name__ == "__main__":
    main() 