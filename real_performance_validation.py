#!/usr/bin/env python3
"""
📊 REAL PERFORMANCE VALIDATION - Thu thập số liệu thực tế
Đo đạc và chứng minh hiệu suất thực tế của hệ thống
"""

import sys
import os
import time
import json
import psutil
import traceback
from datetime import datetime
from typing import Dict, List, Any

sys.path.append('src')

class RealPerformanceValidator:
    """Class đo đạc hiệu suất thực tế"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.start_time = time.time()
        
    def measure_system_startup_time(self):
        """Đo thời gian khởi động hệ thống thực tế"""
        print("📊 MEASURING SYSTEM STARTUP TIME")
        print("-" * 35)
        
        startup_times = []
        errors = []
        
        for i in range(10):  # Test 10 lần để có số liệu chính xác
            try:
                start = time.time()
                
                # Import và khởi tạo hệ thống
                from core.ultimate_xau_system import UltimateXAUSystem
                system = UltimateXAUSystem()
                
                end = time.time()
                startup_time = (end - start) * 1000  # Convert to milliseconds
                startup_times.append(startup_time)
                
                print(f"   Test {i+1}: {startup_time:.2f}ms")
                
                # Clear module cache for next test
                if 'core.ultimate_xau_system' in sys.modules:
                    del sys.modules['core.ultimate_xau_system']
                
            except Exception as e:
                errors.append(f"Test {i+1}: {str(e)}")
                print(f"   Test {i+1}: ERROR - {e}")
        
        if startup_times:
            avg_time = sum(startup_times) / len(startup_times)
            min_time = min(startup_times)
            max_time = max(startup_times)
            
            self.results['startup_performance'] = {
                'tests_run': len(startup_times),
                'average_time_ms': round(avg_time, 2),
                'min_time_ms': round(min_time, 2),
                'max_time_ms': round(max_time, 2),
                'success_rate': len(startup_times) / 10 * 100,
                'errors': errors
            }
            
            print(f"\n📊 Startup Performance Results:")
            print(f"   ⚡ Average: {avg_time:.2f}ms")
            print(f"   🚀 Fastest: {min_time:.2f}ms")
            print(f"   🐌 Slowest: {max_time:.2f}ms")
            print(f"   ✅ Success Rate: {len(startup_times)}/10 ({len(startup_times)/10*100}%)")
            
            return True
        else:
            print("❌ All startup tests failed")
            return False
    
    def measure_signal_generation_performance(self):
        """Đo hiệu suất tạo signal thực tế"""
        print(f"\n📊 MEASURING SIGNAL GENERATION PERFORMANCE")
        print("-" * 45)
        
        try:
            from core.ultimate_xau_system import UltimateXAUSystem
            system = UltimateXAUSystem()
            
            signal_times = []
            signal_results = []
            errors = []
            
            # Test 100 signals để có số liệu đáng tin cậy
            for i in range(100):
                try:
                    start = time.time()
                    signal = system.generate_signal()
                    end = time.time()
                    
                    signal_time = (end - start) * 1000  # ms
                    signal_times.append(signal_time)
                    signal_results.append(signal)
                    
                    if (i + 1) % 20 == 0:
                        print(f"   Completed {i+1}/100 signals...")
                        
                except Exception as e:
                    errors.append(f"Signal {i+1}: {str(e)}")
            
            if signal_times:
                avg_time = sum(signal_times) / len(signal_times)
                min_time = min(signal_times)
                max_time = max(signal_times)
                
                # Analyze signal quality
                actions = [s.get('action') for s in signal_results if isinstance(s, dict)]
                confidences = [s.get('confidence', 0) for s in signal_results if isinstance(s, dict)]
                
                action_distribution = {}
                for action in actions:
                    action_distribution[action] = action_distribution.get(action, 0) + 1
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                self.results['signal_generation'] = {
                    'total_signals': len(signal_times),
                    'average_time_ms': round(avg_time, 3),
                    'min_time_ms': round(min_time, 3),
                    'max_time_ms': round(max_time, 3),
                    'signals_per_second': round(1000 / avg_time, 2),
                    'success_rate': len(signal_times) / 100 * 100,
                    'average_confidence': round(avg_confidence, 2),
                    'action_distribution': action_distribution,
                    'errors': len(errors)
                }
                
                print(f"\n📊 Signal Generation Results:")
                print(f"   ⚡ Average: {avg_time:.3f}ms")
                print(f"   🚀 Fastest: {min_time:.3f}ms")
                print(f"   🐌 Slowest: {max_time:.3f}ms")
                print(f"   📈 Throughput: {1000/avg_time:.2f} signals/second")
                print(f"   🎯 Avg Confidence: {avg_confidence:.2f}%")
                print(f"   ✅ Success Rate: {len(signal_times)}/100")
                print(f"   📊 Actions: {action_distribution}")
                
                return True
            else:
                print("❌ All signal generation tests failed")
                return False
                
        except Exception as e:
            print(f"❌ Signal generation test failed: {e}")
            self.errors.append(f"Signal generation: {e}")
            return False
    
    def measure_memory_usage(self):
        """Đo sử dụng memory thực tế"""
        print(f"\n📊 MEASURING MEMORY USAGE")
        print("-" * 25)
        
        try:
            process = psutil.Process()
            
            # Memory before import
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Import system
            from core.ultimate_xau_system import UltimateXAUSystem
            memory_after_import = process.memory_info().rss / 1024 / 1024  # MB
            
            # Initialize system
            system = UltimateXAUSystem()
            memory_after_init = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate 50 signals
            for i in range(50):
                signal = system.generate_signal()
            memory_after_signals = process.memory_info().rss / 1024 / 1024  # MB
            
            self.results['memory_usage'] = {
                'before_import_mb': round(memory_before, 2),
                'after_import_mb': round(memory_after_import, 2),
                'after_initialization_mb': round(memory_after_init, 2),
                'after_50_signals_mb': round(memory_after_signals, 2),
                'import_overhead_mb': round(memory_after_import - memory_before, 2),
                'init_overhead_mb': round(memory_after_init - memory_after_import, 2),
                'signal_overhead_mb': round(memory_after_signals - memory_after_init, 2),
                'total_overhead_mb': round(memory_after_signals - memory_before, 2)
            }
            
            print(f"📊 Memory Usage Results:")
            print(f"   💾 Before Import: {memory_before:.2f} MB")
            print(f"   💾 After Import: {memory_after_import:.2f} MB (+{memory_after_import-memory_before:.2f} MB)")
            print(f"   💾 After Init: {memory_after_init:.2f} MB (+{memory_after_init-memory_after_import:.2f} MB)")
            print(f"   💾 After 50 Signals: {memory_after_signals:.2f} MB (+{memory_after_signals-memory_after_init:.2f} MB)")
            print(f"   📊 Total Overhead: {memory_after_signals-memory_before:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Memory measurement failed: {e}")
            self.errors.append(f"Memory measurement: {e}")
            return False
    
    def measure_cpu_usage(self):
        """Đo CPU usage thực tế"""
        print(f"\n📊 MEASURING CPU USAGE")
        print("-" * 20)
        
        try:
            from core.ultimate_xau_system import UltimateXAUSystem
            system = UltimateXAUSystem()
            
            # Measure CPU during signal generation
            cpu_measurements = []
            
            for i in range(20):  # 20 measurements
                cpu_before = psutil.cpu_percent(interval=0.1)
                
                # Generate signal (CPU intensive task)
                start = time.time()
                signal = system.generate_signal()
                end = time.time()
                
                cpu_after = psutil.cpu_percent(interval=0.1)
                cpu_measurements.append({
                    'cpu_before': cpu_before,
                    'cpu_after': cpu_after,
                    'signal_time': (end - start) * 1000
                })
                
                if (i + 1) % 5 == 0:
                    print(f"   CPU measurement {i+1}/20...")
            
            avg_cpu_before = sum(m['cpu_before'] for m in cpu_measurements) / len(cpu_measurements)
            avg_cpu_after = sum(m['cpu_after'] for m in cpu_measurements) / len(cpu_measurements)
            avg_signal_time = sum(m['signal_time'] for m in cpu_measurements) / len(cpu_measurements)
            
            self.results['cpu_usage'] = {
                'measurements': len(cpu_measurements),
                'avg_cpu_before_percent': round(avg_cpu_before, 2),
                'avg_cpu_after_percent': round(avg_cpu_after, 2),
                'cpu_impact_percent': round(avg_cpu_after - avg_cpu_before, 2),
                'avg_signal_time_ms': round(avg_signal_time, 3)
            }
            
            print(f"📊 CPU Usage Results:")
            print(f"   🖥️ Avg CPU Before: {avg_cpu_before:.2f}%")
            print(f"   🖥️ Avg CPU After: {avg_cpu_after:.2f}%")
            print(f"   📊 CPU Impact: {avg_cpu_after - avg_cpu_before:.2f}%")
            print(f"   ⚡ Avg Signal Time: {avg_signal_time:.3f}ms")
            
            return True
            
        except Exception as e:
            print(f"❌ CPU measurement failed: {e}")
            self.errors.append(f"CPU measurement: {e}")
            return False
    
    def stress_test_system(self):
        """Stress test hệ thống với load cao"""
        print(f"\n📊 STRESS TESTING SYSTEM")
        print("-" * 25)
        
        try:
            from core.ultimate_xau_system import UltimateXAUSystem
            system = UltimateXAUSystem()
            
            stress_results = {
                'signals_generated': 0,
                'errors': 0,
                'start_time': time.time(),
                'signal_times': [],
                'error_messages': []
            }
            
            print("🔥 Running stress test (1000 signals)...")
            
            # Generate 1000 signals rapidly
            for i in range(1000):
                try:
                    start = time.time()
                    signal = system.generate_signal()
                    end = time.time()
                    
                    stress_results['signals_generated'] += 1
                    stress_results['signal_times'].append((end - start) * 1000)
                    
                    if (i + 1) % 200 == 0:
                        print(f"   Generated {i+1}/1000 signals...")
                        
                except Exception as e:
                    stress_results['errors'] += 1
                    stress_results['error_messages'].append(str(e))
            
            stress_results['end_time'] = time.time()
            stress_results['total_duration'] = stress_results['end_time'] - stress_results['start_time']
            
            if stress_results['signal_times']:
                avg_time = sum(stress_results['signal_times']) / len(stress_results['signal_times'])
                throughput = stress_results['signals_generated'] / stress_results['total_duration']
                
                self.results['stress_test'] = {
                    'signals_generated': stress_results['signals_generated'],
                    'errors': stress_results['errors'],
                    'success_rate': stress_results['signals_generated'] / 1000 * 100,
                    'total_duration_seconds': round(stress_results['total_duration'], 2),
                    'average_signal_time_ms': round(avg_time, 3),
                    'throughput_signals_per_second': round(throughput, 2),
                    'error_rate': stress_results['errors'] / 1000 * 100
                }
                
                print(f"📊 Stress Test Results:")
                print(f"   ✅ Signals Generated: {stress_results['signals_generated']}/1000")
                print(f"   ❌ Errors: {stress_results['errors']}")
                print(f"   📈 Success Rate: {stress_results['signals_generated']/1000*100:.1f}%")
                print(f"   ⏱️ Total Duration: {stress_results['total_duration']:.2f}s")
                print(f"   ⚡ Throughput: {throughput:.2f} signals/second")
                print(f"   🎯 Avg Signal Time: {avg_time:.3f}ms")
                
                return True
            else:
                print("❌ Stress test failed - no signals generated")
                return False
                
        except Exception as e:
            print(f"❌ Stress test failed: {e}")
            self.errors.append(f"Stress test: {e}")
            return False
    
    def measure_reliability_over_time(self):
        """Đo độ tin cậy theo thời gian"""
        print(f"\n📊 MEASURING RELIABILITY OVER TIME")
        print("-" * 35)
        
        try:
            from core.ultimate_xau_system import UltimateXAUSystem
            system = UltimateXAUSystem()
            
            reliability_data = []
            start_time = time.time()
            
            print("🔄 Running 5-minute reliability test...")
            
            while time.time() - start_time < 300:  # 5 minutes
                try:
                    signal_start = time.time()
                    signal = system.generate_signal()
                    signal_end = time.time()
                    
                    reliability_data.append({
                        'timestamp': signal_end,
                        'signal_time_ms': (signal_end - signal_start) * 1000,
                        'success': True,
                        'confidence': signal.get('confidence', 0) if isinstance(signal, dict) else 0
                    })
                    
                except Exception as e:
                    reliability_data.append({
                        'timestamp': time.time(),
                        'success': False,
                        'error': str(e)
                    })
                
                time.sleep(1)  # 1 signal per second
            
            # Analyze reliability data
            total_attempts = len(reliability_data)
            successful = sum(1 for d in reliability_data if d.get('success', False))
            failed = total_attempts - successful
            
            if successful > 0:
                avg_signal_time = sum(d.get('signal_time_ms', 0) for d in reliability_data if d.get('success', False)) / successful
                avg_confidence = sum(d.get('confidence', 0) for d in reliability_data if d.get('success', False)) / successful
                
                self.results['reliability_test'] = {
                    'duration_seconds': 300,
                    'total_attempts': total_attempts,
                    'successful_signals': successful,
                    'failed_signals': failed,
                    'success_rate': round(successful / total_attempts * 100, 2),
                    'avg_signal_time_ms': round(avg_signal_time, 3),
                    'avg_confidence': round(avg_confidence, 2),
                    'signals_per_minute': round(successful / 5, 2)
                }
                
                print(f"📊 Reliability Test Results:")
                print(f"   ⏱️ Duration: 5 minutes")
                print(f"   ✅ Successful: {successful}/{total_attempts}")
                print(f"   📈 Success Rate: {successful/total_attempts*100:.2f}%")
                print(f"   ⚡ Avg Signal Time: {avg_signal_time:.3f}ms")
                print(f"   🎯 Avg Confidence: {avg_confidence:.2f}%")
                print(f"   📊 Signals/Minute: {successful/5:.2f}")
                
                return True
            else:
                print("❌ Reliability test failed - no successful signals")
                return False
                
        except Exception as e:
            print(f"❌ Reliability test failed: {e}")
            self.errors.append(f"Reliability test: {e}")
            return False
    
    def generate_performance_report(self):
        """Tạo báo cáo hiệu suất chi tiết"""
        print(f"\n📋 GENERATING PERFORMANCE REPORT")
        print("-" * 35)
        
        total_time = time.time() - self.start_time
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_duration_seconds': round(total_time, 2),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / 1024**3, 2),
                'python_version': sys.version
            },
            'test_results': self.results,
            'errors': self.errors,
            'overall_assessment': self._calculate_overall_assessment()
        }
        
        # Save report
        report_file = f"real_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Performance report saved: {report_file}")
        
        # Display summary
        self._display_performance_summary(report)
        
        return report
    
    def _calculate_overall_assessment(self):
        """Tính toán đánh giá tổng thể"""
        scores = []
        
        # Startup performance score
        if 'startup_performance' in self.results:
            startup = self.results['startup_performance']
            if startup['success_rate'] >= 90 and startup['average_time_ms'] < 1000:
                scores.append(100)
            elif startup['success_rate'] >= 80:
                scores.append(80)
            else:
                scores.append(60)
        
        # Signal generation score
        if 'signal_generation' in self.results:
            signal = self.results['signal_generation']
            if signal['success_rate'] >= 95 and signal['average_time_ms'] < 10:
                scores.append(100)
            elif signal['success_rate'] >= 90:
                scores.append(85)
            else:
                scores.append(70)
        
        # Memory usage score (lower is better)
        if 'memory_usage' in self.results:
            memory = self.results['memory_usage']
            if memory['total_overhead_mb'] < 50:
                scores.append(100)
            elif memory['total_overhead_mb'] < 100:
                scores.append(80)
            else:
                scores.append(60)
        
        # Stress test score
        if 'stress_test' in self.results:
            stress = self.results['stress_test']
            if stress['success_rate'] >= 95 and stress['error_rate'] < 5:
                scores.append(100)
            elif stress['success_rate'] >= 90:
                scores.append(85)
            else:
                scores.append(70)
        
        # Reliability score
        if 'reliability_test' in self.results:
            reliability = self.results['reliability_test']
            if reliability['success_rate'] >= 95:
                scores.append(100)
            elif reliability['success_rate'] >= 90:
                scores.append(85)
            else:
                scores.append(70)
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        if overall_score >= 95:
            assessment = "EXCELLENT"
        elif overall_score >= 85:
            assessment = "GOOD"
        elif overall_score >= 70:
            assessment = "FAIR"
        else:
            assessment = "POOR"
        
        return {
            'overall_score': round(overall_score, 1),
            'assessment': assessment,
            'individual_scores': scores
        }
    
    def _display_performance_summary(self, report):
        """Hiển thị tóm tắt hiệu suất"""
        print(f"\n📊 REAL PERFORMANCE SUMMARY")
        print("=" * 40)
        
        assessment = report['overall_assessment']
        print(f"🎯 Overall Score: {assessment['overall_score']}/100")
        print(f"📊 Assessment: {assessment['assessment']}")
        print(f"⏱️ Test Duration: {report['test_duration_seconds']}s")
        
        if 'startup_performance' in self.results:
            startup = self.results['startup_performance']
            print(f"\n🚀 STARTUP PERFORMANCE:")
            print(f"   ⚡ Average: {startup['average_time_ms']}ms")
            print(f"   ✅ Success Rate: {startup['success_rate']}%")
        
        if 'signal_generation' in self.results:
            signal = self.results['signal_generation']
            print(f"\n📊 SIGNAL GENERATION:")
            print(f"   ⚡ Average: {signal['average_time_ms']}ms")
            print(f"   📈 Throughput: {signal['signals_per_second']} signals/sec")
            print(f"   ✅ Success Rate: {signal['success_rate']}%")
            print(f"   🎯 Avg Confidence: {signal['average_confidence']}%")
        
        if 'memory_usage' in self.results:
            memory = self.results['memory_usage']
            print(f"\n💾 MEMORY USAGE:")
            print(f"   📊 Total Overhead: {memory['total_overhead_mb']} MB")
            print(f"   🔧 Init Overhead: {memory['init_overhead_mb']} MB")
        
        if 'stress_test' in self.results:
            stress = self.results['stress_test']
            print(f"\n🔥 STRESS TEST:")
            print(f"   ✅ Success Rate: {stress['success_rate']}%")
            print(f"   📈 Throughput: {stress['throughput_signals_per_second']} signals/sec")
            print(f"   ❌ Error Rate: {stress['error_rate']}%")
        
        if 'reliability_test' in self.results:
            reliability = self.results['reliability_test']
            print(f"\n🔄 RELIABILITY TEST:")
            print(f"   ✅ Success Rate: {reliability['success_rate']}%")
            print(f"   📊 Signals/Minute: {reliability['signals_per_minute']}")
        
        if self.errors:
            print(f"\n⚠️ ERRORS ENCOUNTERED: {len(self.errors)}")
            for error in self.errors[:3]:  # Show first 3 errors
                print(f"   - {error}")
        
        print("=" * 40)
    
    def run_comprehensive_performance_test(self):
        """Chạy test hiệu suất toàn diện"""
        print("📊 REAL PERFORMANCE VALIDATION")
        print("=" * 50)
        print("🎯 Objective: Thu thập số liệu thực tế về hiệu suất hệ thống")
        print()
        
        tests = [
            ("System Startup Time", self.measure_system_startup_time),
            ("Signal Generation Performance", self.measure_signal_generation_performance),
            ("Memory Usage", self.measure_memory_usage),
            ("CPU Usage", self.measure_cpu_usage),
            ("Stress Test", self.stress_test_system),
            ("Reliability Over Time", self.measure_reliability_over_time)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n🔄 Running: {test_name}")
            try:
                result = test_func()
                results[test_name] = "✅ SUCCESS" if result else "⚠️ PARTIAL"
            except Exception as e:
                results[test_name] = f"❌ ERROR: {e}"
                self.errors.append(f"{test_name}: {e}")
                print(f"❌ {test_name} failed: {e}")
        
        # Generate final report
        report = self.generate_performance_report()
        
        print(f"\n🎯 TEST RESULTS:")
        for test, result in results.items():
            print(f"   {result} {test}")
        
        return report

def main():
    """Main function"""
    validator = RealPerformanceValidator()
    report = validator.run_comprehensive_performance_test()
    
    print(f"\n🎉 REAL PERFORMANCE VALIDATION COMPLETED!")
    print(f"📊 Overall Score: {report['overall_assessment']['overall_score']}/100")
    print(f"🎯 Assessment: {report['overall_assessment']['assessment']}")
    
    return report

if __name__ == "__main__":
    main() 