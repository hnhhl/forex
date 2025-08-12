"""
Test Suite for Day 19 Performance Optimization
Ultimate XAU Super System V4.0

Comprehensive testing of performance optimization components:
- Performance Optimizer System
- AI Performance Integration
- Response time optimization
- Memory optimization
- System health monitoring
"""

import pytest
import time
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add source path
sys.path.append('src')

# Import test targets
try:
    from src.core.optimization.performance_optimizer import (
        PerformanceOptimizer, OptimizationConfig, OptimizationLevel,
        PerformanceMetrics, MemoryOptimizer, CPUOptimizer,
        ResponseTimeOptimizer, benchmark_function,
        create_performance_optimizer
    )
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False

try:
    from src.core.optimization.ai_performance_integrator import (
        AIPerformanceIntegrator, AIPerformanceConfig, AISystemOptimizer,
        AISystemPerformance, create_ai_performance_integrator
    )
    AI_PERFORMANCE_AVAILABLE = True
except ImportError:
    AI_PERFORMANCE_AVAILABLE = False


@pytest.mark.skipif(not PERFORMANCE_OPTIMIZER_AVAILABLE, reason="Performance Optimizer not available")
class TestPerformanceOptimizer:
    """Test Performance Optimizer System"""
    
    def setup_method(self):
        """Setup test method"""
        self.config = OptimizationConfig(
            level=OptimizationLevel.ADVANCED,
            target_response_time_ms=25.0,
            enable_parallel_processing=True
        )
        self.optimizer = PerformanceOptimizer(self.config)
    
    def teardown_method(self):
        """Cleanup after test"""
        if hasattr(self, 'optimizer'):
            self.optimizer.cleanup()
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        assert self.optimizer is not None
        assert self.optimizer.config.level == OptimizationLevel.ADVANCED
        assert self.optimizer.config.target_response_time_ms == 25.0
        assert hasattr(self.optimizer, 'performance_optimizer')
        assert hasattr(self.optimizer, 'memory_optimizer')
        assert hasattr(self.optimizer, 'cpu_optimizer')
        assert hasattr(self.optimizer, 'response_time_optimizer')
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        # Start monitoring
        self.optimizer.start_monitoring()
        assert self.optimizer.monitoring_active is True
        
        # Wait briefly for monitoring to collect data
        time.sleep(1)
        
        # Stop monitoring
        self.optimizer.stop_monitoring()
        assert self.optimizer.monitoring_active is False
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        metrics = self.optimizer.collect_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.timestamp is not None
        assert metrics.cpu_usage_percent >= 0
        assert metrics.memory_usage_mb >= 0
        assert metrics.system_health_score >= 0
        assert metrics.system_health_score <= 100
    
    def test_optimization_report(self):
        """Test optimization report generation"""
        report = self.optimizer.get_optimization_report()
        
        assert 'timestamp' in report
        assert 'optimization_level' in report
        assert 'current_metrics' in report
        assert 'targets' in report
        assert 'target_achievement' in report
        assert 'cache_performance' in report
        
        # Check current metrics structure
        current = report['current_metrics']
        assert 'response_time_ms' in current
        assert 'cpu_usage_percent' in current
        assert 'memory_usage_mb' in current
        assert 'system_health_score' in current
    
    def test_function_optimization(self):
        """Test function optimization decorator"""
        @self.optimizer.optimize_function
        def test_function():
            time.sleep(0.01)
            return "test_result"
        
        result = test_function()
        assert result == "test_result"
    
    def test_benchmark_function(self):
        """Test function benchmarking"""
        def simple_function():
            return sum(range(100))
        
        benchmark = benchmark_function(simple_function, iterations=10)
        
        assert 'iterations' in benchmark
        assert 'avg_ms' in benchmark
        assert 'min_ms' in benchmark
        assert 'max_ms' in benchmark
        assert 'ops_per_second' in benchmark
        assert benchmark['iterations'] == 10
        assert benchmark['avg_ms'] > 0


@pytest.mark.skipif(not PERFORMANCE_OPTIMIZER_AVAILABLE, reason="Performance Optimizer not available")
class TestMemoryOptimizer:
    """Test Memory Optimizer"""
    
    def setup_method(self):
        """Setup test method"""
        self.config = OptimizationConfig()
        self.memory_optimizer = MemoryOptimizer(self.config)
    
    def test_memory_optimization(self):
        """Test memory optimization functionality"""
        result = self.memory_optimizer.optimize_memory_usage()
        
        assert 'optimizations_applied' in result
        assert 'memory_usage_mb' in result
        assert 'cache_hit_rate' in result
        assert isinstance(result['optimizations_applied'], list)
        assert result['memory_usage_mb'] >= 0
    
    def test_caching_functionality(self):
        """Test result caching"""
        cache_key = "test_key"
        test_data = {"test": "data"}
        
        # Test cache miss
        cached_result = self.memory_optimizer.get_cached_result(cache_key)
        assert cached_result is None
        
        # Cache data
        self.memory_optimizer.cache_result(cache_key, test_data)
        
        # Test cache hit
        cached_result = self.memory_optimizer.get_cached_result(cache_key)
        assert cached_result == test_data
    
    def test_cache_expiration(self):
        """Test cache TTL functionality"""
        # Set short TTL for testing
        self.memory_optimizer.config.cache_ttl_seconds = 0.1
        
        cache_key = "test_expiry"
        test_data = {"expire": "me"}
        
        # Cache data
        self.memory_optimizer.cache_result(cache_key, test_data)
        
        # Should be available immediately
        cached_result = self.memory_optimizer.get_cached_result(cache_key)
        assert cached_result == test_data
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        cached_result = self.memory_optimizer.get_cached_result(cache_key)
        assert cached_result is None


@pytest.mark.skipif(not PERFORMANCE_OPTIMIZER_AVAILABLE, reason="Performance Optimizer not available")
class TestCPUOptimizer:
    """Test CPU Optimizer"""
    
    def setup_method(self):
        """Setup test method"""
        self.config = OptimizationConfig()
        self.cpu_optimizer = CPUOptimizer(self.config)
    
    def teardown_method(self):
        """Cleanup after test"""
        if hasattr(self, 'cpu_optimizer'):
            self.cpu_optimizer.cleanup()
    
    def test_cpu_intensive_optimization(self):
        """Test CPU intensive task optimization"""
        def cpu_task():
            return sum(range(1000))
        
        result = self.cpu_optimizer.optimize_cpu_intensive_task(cpu_task)
        assert result == sum(range(1000))
    
    def test_io_task_optimization(self):
        """Test I/O task optimization"""
        def io_task():
            time.sleep(0.001)
            return "io_completed"
        
        result = self.cpu_optimizer.optimize_io_task(io_task)
        assert result == "io_completed"
    
    def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring"""
        cpu_usage = self.cpu_optimizer.get_cpu_usage()
        assert isinstance(cpu_usage, float)
        assert cpu_usage >= 0
        assert cpu_usage <= 100


@pytest.mark.skipif(not PERFORMANCE_OPTIMIZER_AVAILABLE, reason="Performance Optimizer not available")
class TestResponseTimeOptimizer:
    """Test Response Time Optimizer"""
    
    def setup_method(self):
        """Setup test method"""
        self.config = OptimizationConfig()
        self.response_optimizer = ResponseTimeOptimizer(self.config)
    
    def test_function_call_optimization(self):
        """Test optimized function call"""
        def test_func():
            time.sleep(0.01)
            return "optimized"
        
        result, response_time = self.response_optimizer.optimize_function_call(
            test_func, cache_key="test_cache"
        )
        
        assert result == "optimized"
        assert response_time > 0
        assert response_time < 1.0  # Should be less than 1 second
    
    def test_response_time_stats(self):
        """Test response time statistics"""
        # Generate some response times
        for i in range(10):
            def dummy_func():
                return i
            
            self.response_optimizer.optimize_function_call(dummy_func)
        
        stats = self.response_optimizer.get_response_time_stats()
        
        assert 'avg_ms' in stats
        assert 'min_ms' in stats
        assert 'max_ms' in stats
        assert 'p95_ms' in stats
        assert 'p99_ms' in stats
        assert 'count' in stats
        assert stats['count'] == 10


@pytest.mark.skipif(not AI_PERFORMANCE_AVAILABLE, reason="AI Performance Integrator not available")
class TestAIPerformanceIntegrator:
    """Test AI Performance Integrator"""
    
    def setup_method(self):
        """Setup test method"""
        self.config = AIPerformanceConfig(
            ai_response_time_target_ms=20.0,
            parallel_ai_processing=True
        )
        self.integrator = AIPerformanceIntegrator(self.config)
    
    def teardown_method(self):
        """Cleanup after test"""
        if hasattr(self, 'integrator'):
            self.integrator.cleanup()
    
    def test_integrator_initialization(self):
        """Test integrator initialization"""
        assert self.integrator is not None
        assert self.integrator.config.ai_response_time_target_ms == 20.0
        assert hasattr(self.integrator, 'ai_optimizers')
        assert 'neural_ensemble' in self.integrator.ai_optimizers
        assert 'reinforcement_learning' in self.integrator.ai_optimizers
        assert 'meta_learning' in self.integrator.ai_optimizers
    
    def test_performance_monitoring(self):
        """Test AI performance monitoring"""
        # Start monitoring
        self.integrator.start_performance_monitoring()
        assert self.integrator.monitoring_active is True
        
        # Wait briefly
        time.sleep(1)
        
        # Stop monitoring
        self.integrator.stop_performance_monitoring()
        assert self.integrator.monitoring_active is False
    
    def test_ai_system_optimizer(self):
        """Test individual AI system optimizer"""
        optimizer = self.integrator.ai_optimizers['neural_ensemble']
        
        def mock_prediction():
            time.sleep(0.01)
            return {'prediction': 0.75, 'confidence': 0.9}
        
        result, response_time = optimizer.optimize_prediction_call(mock_prediction)
        
        assert result['prediction'] == 0.75
        assert result['confidence'] == 0.9
        assert response_time > 0
    
    def test_performance_metrics(self):
        """Test AI system performance metrics"""
        optimizer = self.integrator.ai_optimizers['neural_ensemble']
        
        # Generate some predictions
        def mock_prediction():
            return {'result': np.random.random()}
        
        for i in range(5):
            optimizer.optimize_prediction_call(mock_prediction)
        
        metrics = optimizer.get_performance_metrics()
        
        assert isinstance(metrics, AISystemPerformance)
        assert metrics.system_name == 'neural_ensemble'
        assert metrics.prediction_count >= 5
        assert metrics.health_score >= 0
        assert metrics.health_score <= 100
    
    def test_comprehensive_performance_report(self):
        """Test comprehensive performance report"""
        report = self.integrator.get_comprehensive_performance_report()
        
        assert 'timestamp' in report
        assert 'base_performance' in report
        assert 'ai_systems_performance' in report
        assert 'overall_ai_performance' in report
        assert 'configuration' in report
        
        # Check AI systems performance
        ai_systems = report['ai_systems_performance']
        assert 'neural_ensemble' in ai_systems
        assert 'reinforcement_learning' in ai_systems
        assert 'meta_learning' in ai_systems
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing optimization"""
        # Create mock market data
        class MockMarketData:
            def __init__(self, price):
                self.price = price
                self.timestamp = datetime.now()
        
        # Create batch
        batch = [MockMarketData(2000 + i) for i in range(3)]
        
        # Mock the AI master system integration
        with patch.object(self.integrator, 'ai_master_system') as mock_ai_system:
            mock_ai_system.process_market_data.return_value = Mock()
            
            # Mock the optimize_market_data_processing method
            async def mock_process(data):
                return Mock(), {'total_processing_time_ms': 25.0}
            
            with patch.object(self.integrator, 'optimize_market_data_processing', side_effect=mock_process):
                results = await self.integrator.optimize_batch_processing(batch)
                
                assert len(results) == 3
                for result in results:
                    assert result is not None


@pytest.mark.skipif(not AI_PERFORMANCE_AVAILABLE, reason="AI Performance Integrator not available")
class TestAISystemOptimizer:
    """Test AI System Optimizer"""
    
    def setup_method(self):
        """Setup test method"""
        self.config = AIPerformanceConfig()
        self.optimizer = AISystemOptimizer('test_system', self.config)
    
    def test_optimizer_initialization(self):
        """Test AI system optimizer initialization"""
        assert self.optimizer.system_name == 'test_system'
        assert self.optimizer.config == self.config
        assert hasattr(self.optimizer, 'performance_history')
        assert hasattr(self.optimizer, 'optimization_cache')
    
    def test_prediction_optimization(self):
        """Test prediction call optimization"""
        def mock_prediction():
            time.sleep(0.005)
            return {'prediction': 0.8}
        
        result, response_time = self.optimizer.optimize_prediction_call(mock_prediction)
        
        assert result['prediction'] == 0.8
        assert response_time > 0
        assert len(self.optimizer.performance_history) == 1
    
    def test_caching_functionality(self):
        """Test prediction caching"""
        def mock_prediction(*args, **kwargs):
            return {'prediction': np.random.random()}
        
        # First call should execute function
        result1, time1 = self.optimizer.optimize_prediction_call(
            mock_prediction, "arg1", kwarg1="value1"
        )
        
        # Second call with same args should use cache
        result2, time2 = self.optimizer.optimize_prediction_call(
            mock_prediction, "arg1", kwarg1="value1"
        )
        
        # Results should be same (cached)
        assert result1 == result2
        # Second call should be faster (cached)
        assert time2 < time1
    
    def test_performance_metrics_generation(self):
        """Test performance metrics generation"""
        # Generate some performance history
        def mock_prediction():
            return {'prediction': 0.5}
        
        for i in range(10):
            self.optimizer.optimize_prediction_call(mock_prediction)
        
        metrics = self.optimizer.get_performance_metrics()
        
        assert isinstance(metrics, AISystemPerformance)
        assert metrics.system_name == 'test_system'
        assert metrics.prediction_count == 10
        assert metrics.successful_predictions == 10
        assert metrics.failed_predictions == 0


class TestFactoryFunctions:
    """Test factory functions"""
    
    @pytest.mark.skipif(not PERFORMANCE_OPTIMIZER_AVAILABLE, reason="Performance Optimizer not available")
    def test_create_performance_optimizer(self):
        """Test performance optimizer factory"""
        optimizer = create_performance_optimizer(
            level="advanced",
            target_response_time_ms=30.0
        )
        
        assert optimizer is not None
        assert optimizer.config.level == OptimizationLevel.ADVANCED
        assert optimizer.config.target_response_time_ms == 30.0
        
        optimizer.cleanup()
    
    @pytest.mark.skipif(not AI_PERFORMANCE_AVAILABLE, reason="AI Performance Integrator not available")
    def test_create_ai_performance_integrator(self):
        """Test AI performance integrator factory"""
        integrator = create_ai_performance_integrator(
            ai_response_time_target_ms=15.0,
            enable_parallel_processing=True
        )
        
        assert integrator is not None
        assert integrator.config.ai_response_time_target_ms == 15.0
        assert integrator.config.parallel_ai_processing is True
        
        integrator.cleanup()


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.mark.skipif(not (PERFORMANCE_OPTIMIZER_AVAILABLE and AI_PERFORMANCE_AVAILABLE), 
                       reason="Both optimizers required")
    def test_full_integration(self):
        """Test full performance optimization integration"""
        # Create both optimizers
        perf_optimizer = create_performance_optimizer()
        ai_integrator = create_ai_performance_integrator()
        
        # Start monitoring
        perf_optimizer.start_monitoring()
        ai_integrator.start_performance_monitoring()
        
        # Simulate some work
        time.sleep(1)
        
        # Get reports
        perf_report = perf_optimizer.get_optimization_report()
        ai_report = ai_integrator.get_comprehensive_performance_report()
        
        assert perf_report is not None
        assert ai_report is not None
        
        # Cleanup
        perf_optimizer.cleanup()
        ai_integrator.cleanup()
    
    def test_performance_boost_calculation(self):
        """Test performance boost calculation"""
        # Simulate baseline metrics
        baseline = {
            'response_time_ms': 50.0,
            'cpu_usage_percent': 80.0,
            'memory_usage_mb': 600.0
        }
        
        # Simulate optimized metrics
        optimized = {
            'response_time_ms': 25.0,
            'cpu_usage_percent': 45.0,
            'memory_usage_mb': 300.0
        }
        
        # Calculate improvements
        response_improvement = ((baseline['response_time_ms'] - optimized['response_time_ms']) / 
                              baseline['response_time_ms']) * 100
        cpu_improvement = ((baseline['cpu_usage_percent'] - optimized['cpu_usage_percent']) / 
                          baseline['cpu_usage_percent']) * 100
        memory_improvement = ((baseline['memory_usage_mb'] - optimized['memory_usage_mb']) / 
                             baseline['memory_usage_mb']) * 100
        
        assert response_improvement == 50.0  # 50% improvement
        assert cpu_improvement == 43.75  # 43.75% improvement
        assert memory_improvement == 50.0  # 50% improvement


@pytest.mark.skipif(not PERFORMANCE_OPTIMIZER_AVAILABLE, reason="Performance Optimizer not available")
class TestPerformanceOptimizationEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_optimization_level(self):
        """Test handling of invalid optimization level"""
        with pytest.raises(ValueError):
            OptimizationLevel("invalid_level")
    
    def test_optimizer_with_disabled_features(self):
        """Test optimizer with disabled features"""
        config = OptimizationConfig(
            enable_parallel_processing=False,
            enable_memory_optimization=False,
            enable_cpu_optimization=False,
            enable_result_caching=False
        )
        
        optimizer = PerformanceOptimizer(config)
        
        # Should still work with features disabled
        report = optimizer.get_optimization_report()
        assert report is not None
        
        optimizer.cleanup()
    
    def test_metrics_collection_failure_handling(self):
        """Test handling of metrics collection failures"""
        optimizer = PerformanceOptimizer()
        
        # This should not raise an exception even if some metrics fail
        metrics = optimizer.collect_metrics()
        assert metrics is not None
        
        optimizer.cleanup()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=5"
    ])