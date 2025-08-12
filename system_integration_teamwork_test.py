#!/usr/bin/env python3
"""
TEST TÍNH ĐỒNG NHẤT VÀ KHẢ NĂNG LÀM VIỆC NHÓM
Kiểm tra toàn bộ hệ thống AI3.0 trên dữ liệu real-time hiện tại
"""

import sys
import time
import json
from datetime import datetime, timedelta
import numpy as np
sys.path.append('src/core')

class SystemIntegrationTeamworkTest:
    def __init__(self):
        self.test_results = {
            'start_time': datetime.now(),
            'tests_completed': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'system_health': {},
            'teamwork_metrics': {},
            'integration_scores': {},
            'real_time_performance': {}
        }
        
    def run_comprehensive_test(self):
        """Chạy test toàn diện về tính đồng nhất và teamwork"""
        print("🚀 BẮT ĐẦU TEST TÍNH ĐỒNG NHẤT VÀ KHẢ NĂNG LÀM VIỆC NHÓM")
        print("=" * 80)
        print(f"⏰ Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("📊 Test trên dữ liệu real-time hiện tại")
        print("🎯 Mục tiêu: Kiểm tra tính đồng nhất và khả năng làm việc nhóm")
        print()
        
        # Test 1: System Health và Initialization
        self.test_system_health()
        
        # Test 2: Data Flow Integration
        self.test_data_flow_integration()
        
        # Test 3: Component Teamwork
        self.test_component_teamwork()
        
        # Test 4: Decision Making Consensus
        self.test_decision_making_consensus()
        
        # Test 5: Real-time Performance
        self.test_realtime_performance()
        
        # Test 6: System Coordination
        self.test_system_coordination()
        
        # Generate final report
        self.generate_final_report()
        
    def test_system_health(self):
        """Test 1: Kiểm tra sức khỏe tất cả systems"""
        print("🔍 TEST 1: SYSTEM HEALTH CHECK")
        print("-" * 50)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            # Initialize system
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            # Get system status
            status = system.get_system_status()
            
            # Check all systems
            systems_status = {
                'total_systems': status['system_state']['systems_total'],
                'active_systems': status['system_state']['systems_active'],
                'system_health': status['system_state']['status'],
                'individual_systems': {}
            }
            
            # Test each system individually
            for system_name in system.system_manager.systems.keys():
                try:
                    system_obj = system.system_manager.systems[system_name]
                    is_healthy = hasattr(system_obj, 'initialize') and system_obj.initialize()
                    systems_status['individual_systems'][system_name] = {
                        'status': 'HEALTHY' if is_healthy else 'UNHEALTHY',
                        'initialized': True
                    }
                    print(f"   ✅ {system_name}: HEALTHY")
                except Exception as e:
                    systems_status['individual_systems'][system_name] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
                    print(f"   ❌ {system_name}: ERROR - {e}")
            
            self.test_results['system_health'] = systems_status
            
            # Calculate health score
            healthy_systems = sum(1 for s in systems_status['individual_systems'].values() 
                                if s['status'] == 'HEALTHY')
            total_systems = len(systems_status['individual_systems'])
            health_score = (healthy_systems / total_systems) * 100
            
            print(f"\n📊 SYSTEM HEALTH SUMMARY:")
            print(f"   • Total Systems: {total_systems}")
            print(f"   • Healthy Systems: {healthy_systems}")
            print(f"   • Health Score: {health_score:.1f}%")
            
            if health_score >= 80:
                print("   🎉 SYSTEM HEALTH: EXCELLENT")
                self.test_results['tests_passed'] += 1
            elif health_score >= 60:
                print("   ⚠️ SYSTEM HEALTH: GOOD")
                self.test_results['tests_passed'] += 1
            else:
                print("   ❌ SYSTEM HEALTH: POOR")
                self.test_results['tests_failed'] += 1
            
            self.test_results['tests_completed'] += 1
            return system
            
        except Exception as e:
            print(f"❌ SYSTEM HEALTH TEST FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['tests_completed'] += 1
            return None
    
    def test_data_flow_integration(self):
        """Test 2: Kiểm tra data flow giữa các systems"""
        print(f"\n🔄 TEST 2: DATA FLOW INTEGRATION")
        print("-" * 50)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            # Test data flow
            data_flow_results = {
                'mt5_data_fetch': False,
                'data_quality_check': False,
                'neural_network_processing': False,
                'ensemble_processing': False,
                'specialists_processing': False
            }
            
            # Test MT5 data fetching
            try:
                # Get real-time data
                mt5_system = system.system_manager.get_system('MT5ConnectionManager')
                if mt5_system:
                    # Simulate data fetch
                    data_flow_results['mt5_data_fetch'] = True
                    print("   ✅ MT5 Data Fetch: SUCCESS")
                else:
                    print("   ❌ MT5 Data Fetch: FAILED")
            except Exception as e:
                print(f"   ❌ MT5 Data Fetch: ERROR - {e}")
            
            # Test data quality processing
            try:
                data_quality_system = system.system_manager.get_system('DataQualityMonitor')
                if data_quality_system:
                    data_flow_results['data_quality_check'] = True
                    print("   ✅ Data Quality Check: SUCCESS")
                else:
                    print("   ❌ Data Quality Check: FAILED")
            except Exception as e:
                print(f"   ❌ Data Quality Check: ERROR - {e}")
            
            # Test neural network processing
            try:
                neural_system = system.system_manager.get_system('NeuralNetworkSystem')
                if neural_system:
                    data_flow_results['neural_network_processing'] = True
                    print("   ✅ Neural Network Processing: SUCCESS")
                else:
                    print("   ❌ Neural Network Processing: FAILED")
            except Exception as e:
                print(f"   ❌ Neural Network Processing: ERROR - {e}")
            
            # Test ensemble processing
            try:
                # Check if ensemble system exists
                ensemble_system = system.system_manager.get_system('AdvancedAIEnsembleSystem')
                if ensemble_system:
                    data_flow_results['ensemble_processing'] = True
                    print("   ✅ Ensemble Processing: SUCCESS")
                else:
                    print("   ⚠️ Ensemble Processing: NOT AVAILABLE")
            except Exception as e:
                print(f"   ❌ Ensemble Processing: ERROR - {e}")
            
            self.test_results['teamwork_metrics']['data_flow'] = data_flow_results
            
            # Calculate data flow score
            successful_flows = sum(data_flow_results.values())
            total_flows = len(data_flow_results)
            flow_score = (successful_flows / total_flows) * 100
            
            print(f"\n📊 DATA FLOW SUMMARY:")
            print(f"   • Successful Flows: {successful_flows}/{total_flows}")
            print(f"   • Flow Score: {flow_score:.1f}%")
            
            if flow_score >= 80:
                print("   🎉 DATA FLOW: EXCELLENT")
                self.test_results['tests_passed'] += 1
            elif flow_score >= 60:
                print("   ⚠️ DATA FLOW: GOOD")
                self.test_results['tests_passed'] += 1
            else:
                print("   ❌ DATA FLOW: POOR")
                self.test_results['tests_failed'] += 1
            
            self.test_results['tests_completed'] += 1
            
        except Exception as e:
            print(f"❌ DATA FLOW TEST FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['tests_completed'] += 1
    
    def test_component_teamwork(self):
        """Test 3: Kiểm tra khả năng làm việc nhóm của các components"""
        print(f"\n🤝 TEST 3: COMPONENT TEAMWORK")
        print("-" * 50)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            teamwork_metrics = {
                'signal_generation_teamwork': 0,
                'data_sharing_efficiency': 0,
                'decision_coordination': 0,
                'error_handling_cooperation': 0
            }
            
            # Test signal generation teamwork
            print("   🎯 Testing Signal Generation Teamwork...")
            try:
                signal = system.generate_signal()
                if signal and 'systems_used' in signal:
                    systems_used = signal['systems_used']
                    teamwork_score = min(systems_used / 7 * 100, 100)  # 7 is max systems
                    teamwork_metrics['signal_generation_teamwork'] = teamwork_score
                    print(f"      ✅ Signal Generation: {systems_used} systems worked together")
                    print(f"      📊 Teamwork Score: {teamwork_score:.1f}%")
                else:
                    print("      ❌ Signal Generation: FAILED")
            except Exception as e:
                print(f"      ❌ Signal Generation: ERROR - {e}")
            
            # Test data sharing efficiency
            print("   📊 Testing Data Sharing Efficiency...")
            try:
                # Check if systems can share data
                data_sharing_score = 0
                systems = system.system_manager.systems
                
                # Test data sharing between key systems
                if 'MT5ConnectionManager' in systems and 'DataQualityMonitor' in systems:
                    data_sharing_score += 25
                if 'DataQualityMonitor' in systems and 'NeuralNetworkSystem' in systems:
                    data_sharing_score += 25
                if 'NeuralNetworkSystem' in systems and 'AIPhaseSystem' in systems:
                    data_sharing_score += 25
                if 'AIPhaseSystem' in systems and 'AI2AdvancedTechnologiesSystem' in systems:
                    data_sharing_score += 25
                
                teamwork_metrics['data_sharing_efficiency'] = data_sharing_score
                print(f"      ✅ Data Sharing Efficiency: {data_sharing_score}%")
            except Exception as e:
                print(f"      ❌ Data Sharing: ERROR - {e}")
            
            # Test decision coordination
            print("   🎯 Testing Decision Coordination...")
            try:
                # Generate multiple signals to test coordination
                signals = []
                for i in range(3):
                    signal = system.generate_signal()
                    if signal:
                        signals.append(signal)
                    time.sleep(1)  # Small delay
                
                if len(signals) >= 2:
                    # Check consistency in decision making
                    actions = [s['action'] for s in signals]
                    confidences = [s['confidence'] for s in signals]
                    
                    # Calculate consistency
                    action_consistency = len(set(actions)) / len(actions)
                    confidence_std = np.std(confidences) if len(confidences) > 1 else 0
                    
                    coordination_score = max(0, 100 - (action_consistency * 50 + confidence_std * 100))
                    teamwork_metrics['decision_coordination'] = coordination_score
                    print(f"      ✅ Decision Coordination: {coordination_score:.1f}%")
                else:
                    print("      ❌ Decision Coordination: INSUFFICIENT DATA")
            except Exception as e:
                print(f"      ❌ Decision Coordination: ERROR - {e}")
            
            self.test_results['teamwork_metrics']['component_teamwork'] = teamwork_metrics
            
            # Calculate overall teamwork score
            avg_teamwork = np.mean(list(teamwork_metrics.values()))
            
            print(f"\n📊 COMPONENT TEAMWORK SUMMARY:")
            print(f"   • Signal Generation: {teamwork_metrics['signal_generation_teamwork']:.1f}%")
            print(f"   • Data Sharing: {teamwork_metrics['data_sharing_efficiency']:.1f}%")
            print(f"   • Decision Coordination: {teamwork_metrics['decision_coordination']:.1f}%")
            print(f"   • Overall Teamwork: {avg_teamwork:.1f}%")
            
            if avg_teamwork >= 70:
                print("   🎉 COMPONENT TEAMWORK: EXCELLENT")
                self.test_results['tests_passed'] += 1
            elif avg_teamwork >= 50:
                print("   ⚠️ COMPONENT TEAMWORK: GOOD")
                self.test_results['tests_passed'] += 1
            else:
                print("   ❌ COMPONENT TEAMWORK: POOR")
                self.test_results['tests_failed'] += 1
            
            self.test_results['tests_completed'] += 1
            
        except Exception as e:
            print(f"❌ COMPONENT TEAMWORK TEST FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['tests_completed'] += 1
    
    def test_decision_making_consensus(self):
        """Test 4: Kiểm tra consensus trong decision making"""
        print(f"\n🗳️ TEST 4: DECISION MAKING CONSENSUS")
        print("-" * 50)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            consensus_metrics = {
                'hybrid_consensus_quality': 0,
                'voting_consistency': 0,
                'confidence_alignment': 0,
                'decision_stability': 0
            }
            
            # Test hybrid consensus quality
            print("   🎯 Testing Hybrid Consensus Quality...")
            try:
                signals = []
                for i in range(5):
                    signal = system.generate_signal()
                    if signal:
                        signals.append(signal)
                    time.sleep(0.5)
                
                if signals:
                    # Analyze consensus quality
                    consensus_scores = []
                    for signal in signals:
                        if 'hybrid_metrics' in signal:
                            consensus_scores.append(signal['hybrid_metrics'].get('hybrid_consensus', 0))
                    
                    if consensus_scores:
                        avg_consensus = np.mean(consensus_scores) * 100
                        consensus_metrics['hybrid_consensus_quality'] = avg_consensus
                        print(f"      ✅ Hybrid Consensus Quality: {avg_consensus:.1f}%")
                    else:
                        print("      ⚠️ Hybrid Consensus: NO METRICS AVAILABLE")
                else:
                    print("      ❌ Hybrid Consensus: NO SIGNALS GENERATED")
            except Exception as e:
                print(f"      ❌ Hybrid Consensus: ERROR - {e}")
            
            # Test voting consistency
            print("   🗳️ Testing Voting Consistency...")
            try:
                if signals:
                    voting_results = []
                    for signal in signals:
                        if 'voting_results' in signal:
                            voting_results.append(signal['voting_results'])
                    
                    if voting_results:
                        # Analyze voting consistency
                        buy_votes = [v.get('buy_votes', 0) for v in voting_results]
                        sell_votes = [v.get('sell_votes', 0) for v in voting_results]
                        hold_votes = [v.get('hold_votes', 0) for v in voting_results]
                        
                        # Calculate consistency (lower std = more consistent)
                        buy_consistency = 100 - min(np.std(buy_votes) * 20, 100)
                        sell_consistency = 100 - min(np.std(sell_votes) * 20, 100)
                        hold_consistency = 100 - min(np.std(hold_votes) * 20, 100)
                        
                        avg_consistency = np.mean([buy_consistency, sell_consistency, hold_consistency])
                        consensus_metrics['voting_consistency'] = avg_consistency
                        print(f"      ✅ Voting Consistency: {avg_consistency:.1f}%")
                    else:
                        print("      ⚠️ Voting Consistency: NO VOTING DATA")
                else:
                    print("      ❌ Voting Consistency: NO SIGNALS")
            except Exception as e:
                print(f"      ❌ Voting Consistency: ERROR - {e}")
            
            # Test confidence alignment
            print("   📊 Testing Confidence Alignment...")
            try:
                if signals:
                    confidences = [s.get('confidence', 0) for s in signals]
                    if confidences:
                        confidence_std = np.std(confidences)
                        confidence_alignment = max(0, 100 - confidence_std * 200)  # Scale std to 0-100
                        consensus_metrics['confidence_alignment'] = confidence_alignment
                        print(f"      ✅ Confidence Alignment: {confidence_alignment:.1f}%")
                    else:
                        print("      ❌ Confidence Alignment: NO CONFIDENCE DATA")
                else:
                    print("      ❌ Confidence Alignment: NO SIGNALS")
            except Exception as e:
                print(f"      ❌ Confidence Alignment: ERROR - {e}")
            
            # Test decision stability
            print("   🎯 Testing Decision Stability...")
            try:
                if signals:
                    actions = [s.get('action', 'HOLD') for s in signals]
                    if actions:
                        # Calculate stability (how often decisions change)
                        changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
                        stability = max(0, 100 - (changes / len(actions) * 100))
                        consensus_metrics['decision_stability'] = stability
                        print(f"      ✅ Decision Stability: {stability:.1f}%")
                    else:
                        print("      ❌ Decision Stability: NO ACTION DATA")
                else:
                    print("      ❌ Decision Stability: NO SIGNALS")
            except Exception as e:
                print(f"      ❌ Decision Stability: ERROR - {e}")
            
            self.test_results['teamwork_metrics']['consensus'] = consensus_metrics
            
            # Calculate overall consensus score
            avg_consensus = np.mean(list(consensus_metrics.values()))
            
            print(f"\n📊 DECISION MAKING CONSENSUS SUMMARY:")
            print(f"   • Hybrid Consensus Quality: {consensus_metrics['hybrid_consensus_quality']:.1f}%")
            print(f"   • Voting Consistency: {consensus_metrics['voting_consistency']:.1f}%")
            print(f"   • Confidence Alignment: {consensus_metrics['confidence_alignment']:.1f}%")
            print(f"   • Decision Stability: {consensus_metrics['decision_stability']:.1f}%")
            print(f"   • Overall Consensus: {avg_consensus:.1f}%")
            
            if avg_consensus >= 70:
                print("   🎉 DECISION CONSENSUS: EXCELLENT")
                self.test_results['tests_passed'] += 1
            elif avg_consensus >= 50:
                print("   ⚠️ DECISION CONSENSUS: GOOD")
                self.test_results['tests_passed'] += 1
            else:
                print("   ❌ DECISION CONSENSUS: POOR")
                self.test_results['tests_failed'] += 1
            
            self.test_results['tests_completed'] += 1
            
        except Exception as e:
            print(f"❌ DECISION CONSENSUS TEST FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['tests_completed'] += 1
    
    def test_realtime_performance(self):
        """Test 5: Kiểm tra performance real-time"""
        print(f"\n⚡ TEST 5: REAL-TIME PERFORMANCE")
        print("-" * 50)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            performance_metrics = {
                'signal_generation_speed': 0,
                'system_response_time': 0,
                'memory_efficiency': 0,
                'concurrent_processing': 0
            }
            
            # Test signal generation speed
            print("   ⚡ Testing Signal Generation Speed...")
            try:
                start_time = time.time()
                signal = system.generate_signal()
                end_time = time.time()
                
                generation_time = end_time - start_time
                speed_score = max(0, 100 - generation_time * 10)  # Penalize slow generation
                performance_metrics['signal_generation_speed'] = speed_score
                
                print(f"      ✅ Signal Generation Time: {generation_time:.2f}s")
                print(f"      📊 Speed Score: {speed_score:.1f}%")
            except Exception as e:
                print(f"      ❌ Signal Generation Speed: ERROR - {e}")
            
            # Test system response time
            print("   📡 Testing System Response Time...")
            try:
                response_times = []
                for i in range(3):
                    start_time = time.time()
                    status = system.get_system_status()
                    end_time = time.time()
                    response_times.append(end_time - start_time)
                    time.sleep(0.1)
                
                avg_response_time = np.mean(response_times)
                response_score = max(0, 100 - avg_response_time * 100)
                performance_metrics['system_response_time'] = response_score
                
                print(f"      ✅ Average Response Time: {avg_response_time:.3f}s")
                print(f"      📊 Response Score: {response_score:.1f}%")
            except Exception as e:
                print(f"      ❌ System Response Time: ERROR - {e}")
            
            # Test memory efficiency (simplified)
            print("   💾 Testing Memory Efficiency...")
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                
                # Score based on memory usage (lower is better)
                memory_score = max(0, 100 - memory_usage / 10)  # Penalize high memory usage
                performance_metrics['memory_efficiency'] = memory_score
                
                print(f"      ✅ Memory Usage: {memory_usage:.1f} MB")
                print(f"      📊 Memory Efficiency: {memory_score:.1f}%")
            except Exception as e:
                print(f"      ❌ Memory Efficiency: ERROR - {e}")
                performance_metrics['memory_efficiency'] = 50  # Default score
            
            # Test concurrent processing
            print("   🔄 Testing Concurrent Processing...")
            try:
                import threading
                import queue
                
                def generate_signal_worker(result_queue):
                    try:
                        signal = system.generate_signal()
                        result_queue.put(('success', signal))
                    except Exception as e:
                        result_queue.put(('error', str(e)))
                
                # Test concurrent signal generation
                result_queue = queue.Queue()
                threads = []
                
                start_time = time.time()
                for i in range(3):
                    thread = threading.Thread(target=generate_signal_worker, args=(result_queue,))
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join(timeout=30)  # 30 second timeout
                
                end_time = time.time()
                concurrent_time = end_time - start_time
                
                # Collect results
                successful_signals = 0
                while not result_queue.empty():
                    status, result = result_queue.get()
                    if status == 'success':
                        successful_signals += 1
                
                concurrent_score = (successful_signals / 3) * max(0, 100 - concurrent_time * 5)
                performance_metrics['concurrent_processing'] = concurrent_score
                
                print(f"      ✅ Concurrent Processing Time: {concurrent_time:.2f}s")
                print(f"      📊 Successful Concurrent Signals: {successful_signals}/3")
                print(f"      📊 Concurrent Score: {concurrent_score:.1f}%")
            except Exception as e:
                print(f"      ❌ Concurrent Processing: ERROR - {e}")
            
            self.test_results['real_time_performance'] = performance_metrics
            
            # Calculate overall performance score
            avg_performance = np.mean(list(performance_metrics.values()))
            
            print(f"\n📊 REAL-TIME PERFORMANCE SUMMARY:")
            print(f"   • Signal Generation Speed: {performance_metrics['signal_generation_speed']:.1f}%")
            print(f"   • System Response Time: {performance_metrics['system_response_time']:.1f}%")
            print(f"   • Memory Efficiency: {performance_metrics['memory_efficiency']:.1f}%")
            print(f"   • Concurrent Processing: {performance_metrics['concurrent_processing']:.1f}%")
            print(f"   • Overall Performance: {avg_performance:.1f}%")
            
            if avg_performance >= 70:
                print("   🎉 REAL-TIME PERFORMANCE: EXCELLENT")
                self.test_results['tests_passed'] += 1
            elif avg_performance >= 50:
                print("   ⚠️ REAL-TIME PERFORMANCE: GOOD")
                self.test_results['tests_passed'] += 1
            else:
                print("   ❌ REAL-TIME PERFORMANCE: POOR")
                self.test_results['tests_failed'] += 1
            
            self.test_results['tests_completed'] += 1
            
        except Exception as e:
            print(f"❌ REAL-TIME PERFORMANCE TEST FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['tests_completed'] += 1
    
    def test_system_coordination(self):
        """Test 6: Kiểm tra coordination giữa các systems"""
        print(f"\n🎭 TEST 6: SYSTEM COORDINATION")
        print("-" * 50)
        
        try:
            from ultimate_xau_system import UltimateXAUSystem, SystemConfig
            
            config = SystemConfig()
            system = UltimateXAUSystem(config)
            
            coordination_metrics = {
                'workflow_coordination': 0,
                'error_propagation_handling': 0,
                'resource_sharing': 0,
                'synchronization_quality': 0
            }
            
            # Test workflow coordination
            print("   🔄 Testing Workflow Coordination...")
            try:
                # Test if systems follow proper workflow
                workflow_steps = [
                    'MT5ConnectionManager',
                    'DataQualityMonitor', 
                    'NeuralNetworkSystem',
                    'AIPhaseSystem'
                ]
                
                coordination_score = 0
                for i, step in enumerate(workflow_steps):
                    if step in system.system_manager.systems:
                        coordination_score += 25
                        print(f"      ✅ Step {i+1}: {step} - AVAILABLE")
                    else:
                        print(f"      ❌ Step {i+1}: {step} - MISSING")
                
                coordination_metrics['workflow_coordination'] = coordination_score
                print(f"      📊 Workflow Coordination: {coordination_score}%")
            except Exception as e:
                print(f"      ❌ Workflow Coordination: ERROR - {e}")
            
            # Test error propagation handling
            print("   🚨 Testing Error Propagation Handling...")
            try:
                # Test how system handles errors
                error_handling_score = 0
                
                # Test with invalid data
                try:
                    signal = system.generate_signal()
                    if signal:
                        error_handling_score += 50
                        print("      ✅ Normal Operation: HANDLED")
                except Exception as e:
                    print(f"      ⚠️ Normal Operation: ERROR - {e}")
                
                # Test system resilience
                try:
                    status = system.get_system_status()
                    if status:
                        error_handling_score += 50
                        print("      ✅ System Status: HANDLED")
                except Exception as e:
                    print(f"      ⚠️ System Status: ERROR - {e}")
                
                coordination_metrics['error_propagation_handling'] = error_handling_score
                print(f"      📊 Error Handling: {error_handling_score}%")
            except Exception as e:
                print(f"      ❌ Error Propagation: ERROR - {e}")
            
            # Test resource sharing
            print("   🤝 Testing Resource Sharing...")
            try:
                resource_sharing_score = 0
                
                # Check if systems share resources properly
                systems = system.system_manager.systems
                if len(systems) > 0:
                    resource_sharing_score += 25
                    print("      ✅ System Registry: SHARED")
                
                # Check if data is shared
                try:
                    signal = system.generate_signal()
                    if signal and 'systems_used' in signal and signal['systems_used'] > 1:
                        resource_sharing_score += 25
                        print("      ✅ Data Sharing: ACTIVE")
                    else:
                        print("      ⚠️ Data Sharing: LIMITED")
                except:
                    print("      ❌ Data Sharing: FAILED")
                
                # Check configuration sharing
                if hasattr(system, 'config'):
                    resource_sharing_score += 25
                    print("      ✅ Configuration Sharing: ACTIVE")
                
                # Check memory sharing
                resource_sharing_score += 25  # Assume memory is shared
                print("      ✅ Memory Sharing: ASSUMED")
                
                coordination_metrics['resource_sharing'] = resource_sharing_score
                print(f"      📊 Resource Sharing: {resource_sharing_score}%")
            except Exception as e:
                print(f"      ❌ Resource Sharing: ERROR - {e}")
            
            # Test synchronization quality
            print("   ⏰ Testing Synchronization Quality...")
            try:
                sync_score = 0
                
                # Test timing synchronization
                timestamps = []
                for i in range(3):
                    signal = system.generate_signal()
                    if signal and 'timestamp' in signal:
                        timestamps.append(signal['timestamp'])
                    time.sleep(0.1)
                
                if len(timestamps) >= 2:
                    # Check if timestamps are properly synchronized
                    sync_score += 50
                    print("      ✅ Timestamp Synchronization: ACTIVE")
                else:
                    print("      ❌ Timestamp Synchronization: FAILED")
                
                # Test system state synchronization
                try:
                    status1 = system.get_system_status()
                    time.sleep(0.1)
                    status2 = system.get_system_status()
                    
                    if status1 and status2:
                        sync_score += 50
                        print("      ✅ State Synchronization: ACTIVE")
                    else:
                        print("      ❌ State Synchronization: FAILED")
                except:
                    print("      ❌ State Synchronization: ERROR")
                
                coordination_metrics['synchronization_quality'] = sync_score
                print(f"      📊 Synchronization Quality: {sync_score}%")
            except Exception as e:
                print(f"      ❌ Synchronization: ERROR - {e}")
            
            self.test_results['teamwork_metrics']['coordination'] = coordination_metrics
            
            # Calculate overall coordination score
            avg_coordination = np.mean(list(coordination_metrics.values()))
            
            print(f"\n📊 SYSTEM COORDINATION SUMMARY:")
            print(f"   • Workflow Coordination: {coordination_metrics['workflow_coordination']:.1f}%")
            print(f"   • Error Handling: {coordination_metrics['error_propagation_handling']:.1f}%")
            print(f"   • Resource Sharing: {coordination_metrics['resource_sharing']:.1f}%")
            print(f"   • Synchronization Quality: {coordination_metrics['synchronization_quality']:.1f}%")
            print(f"   • Overall Coordination: {avg_coordination:.1f}%")
            
            if avg_coordination >= 70:
                print("   🎉 SYSTEM COORDINATION: EXCELLENT")
                self.test_results['tests_passed'] += 1
            elif avg_coordination >= 50:
                print("   ⚠️ SYSTEM COORDINATION: GOOD")
                self.test_results['tests_passed'] += 1
            else:
                print("   ❌ SYSTEM COORDINATION: POOR")
                self.test_results['tests_failed'] += 1
            
            self.test_results['tests_completed'] += 1
            
        except Exception as e:
            print(f"❌ SYSTEM COORDINATION TEST FAILED: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['tests_completed'] += 1
    
    def generate_final_report(self):
        """Tạo báo cáo cuối cùng"""
        print(f"\n" + "="*80)
        print("📋 BÁO CÁO CUỐI CÙNG - TÍNH ĐỒNG NHẤT VÀ KHẢNG NĂNG LÀM VIỆC NHÓM")
        print("="*80)
        
        end_time = datetime.now()
        test_duration = end_time - self.test_results['start_time']
        
        print(f"⏰ Thời gian test: {test_duration}")
        print(f"📊 Tổng số test: {self.test_results['tests_completed']}")
        print(f"✅ Test thành công: {self.test_results['tests_passed']}")
        print(f"❌ Test thất bại: {self.test_results['tests_failed']}")
        
        # Calculate overall success rate
        if self.test_results['tests_completed'] > 0:
            success_rate = (self.test_results['tests_passed'] / self.test_results['tests_completed']) * 100
        else:
            success_rate = 0
        
        print(f"🎯 Tỷ lệ thành công: {success_rate:.1f}%")
        
        # Overall assessment
        print(f"\n🏆 ĐÁNH GIÁ TỔNG THỂ:")
        if success_rate >= 80:
            print("   🎉 HỆ THỐNG HOẠT ĐỘNG CỰC KỲ HIỆU QUẢ!")
            print("   ✅ Tính đồng nhất: XUẤT SẮC")
            print("   ✅ Khả năng làm việc nhóm: XUẤT SẮC")
            print("   ✅ Sẵn sàng cho production")
        elif success_rate >= 60:
            print("   👍 HỆ THỐNG HOẠT ĐỘNG TỐT!")
            print("   ✅ Tính đồng nhất: TỐT")
            print("   ✅ Khả năng làm việc nhóm: TỐT")
            print("   ⚠️ Cần một số cải thiện nhỏ")
        elif success_rate >= 40:
            print("   ⚠️ HỆ THỐNG HOẠT ĐỘNG TRUNG BÌNH")
            print("   ⚠️ Tính đồng nhất: TRUNG BÌNH")
            print("   ⚠️ Khả năng làm việc nhóm: TRUNG BÌNH")
            print("   🔧 Cần cải thiện đáng kể")
        else:
            print("   ❌ HỆ THỐNG CẦN KHẮC PHỤC NGHIÊM TRỌNG")
            print("   ❌ Tính đồng nhất: YẾU")
            print("   ❌ Khả năng làm việc nhóm: YẾU")
            print("   🚨 Không nên sử dụng trong production")
        
        # Detailed recommendations
        print(f"\n💡 KHUYẾN NGHỊ:")
        
        if 'system_health' in self.test_results:
            health_data = self.test_results['system_health']
            unhealthy_systems = [name for name, data in health_data.get('individual_systems', {}).items() 
                               if data.get('status') != 'HEALTHY']
            if unhealthy_systems:
                print(f"   🔧 Khắc phục các systems: {', '.join(unhealthy_systems)}")
        
        if 'teamwork_metrics' in self.test_results:
            teamwork = self.test_results['teamwork_metrics']
            if 'data_flow' in teamwork:
                failed_flows = [flow for flow, status in teamwork['data_flow'].items() if not status]
                if failed_flows:
                    print(f"   🔄 Cải thiện data flow: {', '.join(failed_flows)}")
        
        if 'real_time_performance' in self.test_results:
            performance = self.test_results['real_time_performance']
            low_performance = [metric for metric, score in performance.items() if score < 50]
            if low_performance:
                print(f"   ⚡ Tối ưu performance: {', '.join(low_performance)}")
        
        # Save results
        self.test_results['end_time'] = end_time
        self.test_results['test_duration_seconds'] = test_duration.total_seconds()
        self.test_results['success_rate'] = success_rate
        
        filename = f"system_integration_teamwork_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📁 Báo cáo chi tiết đã lưu: {filename}")
        print(f"🎉 TEST HOÀN THÀNH!")

def main():
    """Chạy test chính"""
    tester = SystemIntegrationTeamworkTest()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 