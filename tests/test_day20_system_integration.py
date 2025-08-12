"""
Test Suite for Day 20 System Integration Testing
Ultimate XAU Super System V4.0

Comprehensive tests for:
- System Integration Tester
- Component integration testing
- Performance testing
- Production readiness assessment
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
import sys
import os

# Add source path
sys.path.append('src')

# Import system integration components
try:
    from src.core.testing.system_integration_tester import (
        SystemIntegrationTester, IntegrationTestConfig, TestResult, 
        IntegrationTestReport, ComponentTester, PerformanceTester,
        create_integration_tester
    )
    INTEGRATION_TESTER_AVAILABLE = True
except ImportError:
    INTEGRATION_TESTER_AVAILABLE = False
    pytest.skip("System Integration Tester not available", allow_module_level=True)


class TestIntegrationTestConfig:
    """Test integration test configuration"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = IntegrationTestConfig()
        
        assert config.enable_performance_tests is True
        assert config.enable_load_tests is True
        assert config.enable_stress_tests is True
        assert config.enable_reliability_tests is True
        assert config.max_response_time_ms == 100.0
        assert config.min_throughput_ops_sec == 50.0
        assert config.max_memory_usage_mb == 1024.0
        assert config.min_accuracy_score == 0.85
        assert config.concurrent_users == 10
        assert config.test_duration_seconds == 60
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = IntegrationTestConfig(
            enable_performance_tests=False,
            max_response_time_ms=50.0,
            concurrent_users=5
        )
        
        assert config.enable_performance_tests is False
        assert config.max_response_time_ms == 50.0
        assert config.concurrent_users == 5


class TestTestResult:
    """Test TestResult data structure"""
    
    def test_test_result_creation(self):
        """Test test result creation"""
        result = TestResult(
            test_name="Test Component",
            test_type="COMPONENT",
            status="PASS",
            duration_seconds=1.5,
            details={'test': 'value'}
        )
        
        assert result.test_name == "Test Component"
        assert result.test_type == "COMPONENT"
        assert result.status == "PASS"
        assert result.duration_seconds == 1.5
        assert result.details == {'test': 'value'}
        assert result.error_message is None
        assert isinstance(result.timestamp, datetime)


class TestComponentTester:
    """Test component testing functionality"""
    
    def test_component_tester_initialization(self):
        """Test component tester initialization"""
        config = IntegrationTestConfig()
        tester = ComponentTester(config)
        
        assert tester.config == config
        assert tester.test_results == []
    
    @pytest.mark.asyncio
    async def test_trading_systems_test(self):
        """Test trading systems integration test"""
        config = IntegrationTestConfig()
        tester = ComponentTester(config)
        
        result = await tester.test_trading_systems()
        
        assert isinstance(result, TestResult)
        assert result.test_name == "Trading Systems Integration"
        assert result.test_type == "COMPONENT"
        assert result.status in ["PASS", "FAIL"]
        assert result.duration_seconds > 0
        assert isinstance(result.details, dict)
    
    @pytest.mark.asyncio
    async def test_risk_management_test(self):
        """Test risk management integration test"""
        config = IntegrationTestConfig()
        tester = ComponentTester(config)
        
        result = await tester.test_risk_management()
        
        assert isinstance(result, TestResult)
        assert result.test_name == "Risk Management Integration"
        assert result.test_type == "COMPONENT"
        assert result.status in ["PASS", "FAIL"]
        assert result.duration_seconds > 0
        assert isinstance(result.details, dict)
    
    @pytest.mark.asyncio
    async def test_ai_systems_test(self):
        """Test AI systems integration test"""
        config = IntegrationTestConfig()
        tester = ComponentTester(config)
        
        result = await tester.test_ai_systems()
        
        assert isinstance(result, TestResult)
        assert result.test_name == "AI Systems Integration"
        assert result.test_type == "COMPONENT"
        assert result.status in ["PASS", "FAIL"]
        assert result.duration_seconds > 0
        assert isinstance(result.details, dict)
        
        # Check AI systems status tracking
        if result.status == "PASS":
            assert 'working_systems' in result.details
            assert 'total_systems' in result.details
            assert 'integration_rate' in result.details
    
    @pytest.mark.asyncio
    async def test_performance_optimization_test(self):
        """Test performance optimization integration test"""
        config = IntegrationTestConfig()
        tester = ComponentTester(config)
        
        result = await tester.test_performance_optimization()
        
        assert isinstance(result, TestResult)
        assert result.test_name == "Performance Optimization Integration"
        assert result.test_type == "COMPONENT"
        assert result.status in ["PASS", "FAIL"]
        assert result.duration_seconds > 0
        assert isinstance(result.details, dict)


class TestPerformanceTester:
    """Test performance testing functionality"""
    
    def test_performance_tester_initialization(self):
        """Test performance tester initialization"""
        config = IntegrationTestConfig()
        tester = PerformanceTester(config)
        
        assert tester.config == config
        assert isinstance(tester.performance_data, dict)
        assert 'response_times' in tester.performance_data
        assert 'throughput' in tester.performance_data
    
    @pytest.mark.asyncio
    async def test_simulate_user_session(self):
        """Test user session simulation"""
        config = IntegrationTestConfig(concurrent_users=1)
        tester = PerformanceTester(config)
        
        session_result = await tester._simulate_user_session()
        
        assert isinstance(session_result, dict)
        assert 'operations' in session_result
        assert 'avg_response_time' in session_result
        assert 'session_duration' in session_result
        assert session_result['operations'] > 0
        assert session_result['avg_response_time'] > 0
    
    @pytest.mark.asyncio
    async def test_load_test(self):
        """Test load testing"""
        config = IntegrationTestConfig(concurrent_users=2, test_duration_seconds=5)
        tester = PerformanceTester(config)
        
        result = await tester.run_load_test()
        
        assert isinstance(result, TestResult)
        assert result.test_name == "Load Test"
        assert result.test_type == "PERFORMANCE"
        assert result.status in ["PASS", "FAIL"]
        assert result.duration_seconds > 0
        
        # Check load test details
        assert 'concurrent_users' in result.details
        assert 'successful_sessions' in result.details
        assert 'failed_sessions' in result.details
        assert 'throughput_ops_sec' in result.details
        assert result.details['concurrent_users'] == 2
    
    @pytest.mark.asyncio
    async def test_stress_test(self):
        """Test stress testing"""
        config = IntegrationTestConfig(
            concurrent_users=2, 
            max_load_multiplier=2.0,
            stress_duration_seconds=5
        )
        tester = PerformanceTester(config)
        
        result = await tester.run_stress_test()
        
        assert isinstance(result, TestResult)
        assert result.test_name == "Stress Test"
        assert result.test_type == "PERFORMANCE"
        assert result.status in ["PASS", "FAIL"]
        assert result.duration_seconds > 0
        
        # Check stress test details
        assert 'max_load_tested' in result.details
        assert 'breaking_point' in result.details
        assert 'system_resilience' in result.details
    
    @pytest.mark.asyncio
    async def test_simulate_stress_operation(self):
        """Test stress operation simulation"""
        config = IntegrationTestConfig()
        tester = PerformanceTester(config)
        
        # Run multiple operations to test both success and failure paths
        results = []
        for _ in range(10):
            result = await tester._simulate_stress_operation()
            results.append(result)
        
        # Should have mix of True/False results due to random failures
        assert len(results) == 10
        assert all(isinstance(r, bool) for r in results)


class TestSystemIntegrationTester:
    """Test main system integration tester"""
    
    def test_integration_tester_initialization(self):
        """Test system integration tester initialization"""
        config = IntegrationTestConfig()
        tester = SystemIntegrationTester(config)
        
        assert tester.config == config
        assert isinstance(tester.component_tester, ComponentTester)
        assert isinstance(tester.performance_tester, PerformanceTester)
        assert tester.test_results == []
        assert tester.start_time is None
        assert tester.end_time is None
    
    def test_integration_tester_default_config(self):
        """Test system integration tester with default config"""
        tester = SystemIntegrationTester()
        
        assert isinstance(tester.config, IntegrationTestConfig)
        assert tester.config.enable_performance_tests is True
    
    @pytest.mark.asyncio
    async def test_component_tests_execution(self):
        """Test component tests execution"""
        config = IntegrationTestConfig()
        tester = SystemIntegrationTester(config)
        
        await tester._run_component_tests()
        
        # Should have 4 component tests
        component_tests = [r for r in tester.test_results if r.test_type == "COMPONENT"]
        assert len(component_tests) == 4
        
        # Check test names
        test_names = [r.test_name for r in component_tests]
        expected_names = [
            "Trading Systems Integration",
            "Risk Management Integration", 
            "AI Systems Integration",
            "Performance Optimization Integration"
        ]
        for name in expected_names:
            assert name in test_names
    
    @pytest.mark.asyncio
    async def test_performance_tests_execution(self):
        """Test performance tests execution"""
        config = IntegrationTestConfig(
            concurrent_users=2,
            test_duration_seconds=5,
            stress_duration_seconds=5
        )
        tester = SystemIntegrationTester(config)
        
        await tester._run_performance_tests()
        
        # Should have performance tests
        performance_tests = [r for r in tester.test_results if r.test_type == "PERFORMANCE"]
        assert len(performance_tests) >= 1  # At least load test
        
        # Check for load test
        load_tests = [r for r in performance_tests if "Load Test" in r.test_name]
        assert len(load_tests) >= 1
    
    @pytest.mark.asyncio
    async def test_reliability_tests_execution(self):
        """Test reliability tests execution"""
        config = IntegrationTestConfig()
        tester = SystemIntegrationTester(config)
        
        await tester._run_reliability_tests()
        
        # Should have reliability tests
        reliability_tests = [r for r in tester.test_results if r.test_type == "RELIABILITY"]
        assert len(reliability_tests) >= 1
        
        # Check for system recovery test
        recovery_tests = [r for r in reliability_tests if "System Recovery" in r.test_name]
        assert len(recovery_tests) >= 1
    
    def test_system_health_score_calculation(self):
        """Test system health score calculation"""
        tester = SystemIntegrationTester()
        
        # Mock some component test results
        tester.test_results = [
            TestResult("Test 1", "COMPONENT", "PASS", 1.0, {}),
            TestResult("Test 2", "COMPONENT", "PASS", 1.0, {}),
            TestResult("Test 3", "COMPONENT", "FAIL", 1.0, {}),
            TestResult("Test 4", "OTHER", "PASS", 1.0, {})
        ]
        
        score = tester._calculate_system_health_score()
        
        # 2 out of 3 component tests passed = 66.67%
        assert abs(score - 66.67) < 0.1
    
    def test_reliability_score_calculation(self):
        """Test reliability score calculation"""
        tester = SystemIntegrationTester()
        
        # Mock some reliability test results
        tester.test_results = [
            TestResult("Test 1", "RELIABILITY", "PASS", 1.0, {}),
            TestResult("Test 2", "RELIABILITY", "FAIL", 1.0, {}),
            TestResult("Test 3", "COMPONENT", "PASS", 1.0, {})
        ]
        
        score = tester._calculate_reliability_score()
        
        # 1 out of 2 reliability tests passed = 50%
        assert score == 50.0
    
    def test_performance_score_calculation(self):
        """Test performance score calculation"""
        tester = SystemIntegrationTester()
        
        # Mock some performance test results
        tester.test_results = [
            TestResult("Test 1", "PERFORMANCE", "PASS", 1.0, {}),
            TestResult("Test 2", "PERFORMANCE", "PASS", 1.0, {}),
            TestResult("Test 3", "COMPONENT", "PASS", 1.0, {})
        ]
        
        score = tester._calculate_performance_score()
        
        # 2 out of 2 performance tests passed = 100%
        assert score == 100.0
    
    def test_issues_identification(self):
        """Test issues identification"""
        tester = SystemIntegrationTester()
        
        # Mock test results with failures
        tester.test_results = [
            TestResult("Test 1", "COMPONENT", "PASS", 1.0, {}),
            TestResult("Test 2", "COMPONENT", "FAIL", 1.0, {}, "Connection failed"),
            TestResult("Test 3", "PERFORMANCE", "FAIL", 1.0, {}, "Timeout")
        ]
        
        issues = tester._identify_issues()
        
        assert len(issues) == 2
        assert "Test 2: Connection failed" in issues
        assert "Test 3: Timeout" in issues
    
    def test_recommendations_generation(self):
        """Test recommendations generation"""
        tester = SystemIntegrationTester()
        
        # Mock test results with low success rate
        tester.test_results = [
            TestResult("Test 1", "COMPONENT", "PASS", 1.0, {}),
            TestResult("Test 2", "COMPONENT", "FAIL", 1.0, {}),
            TestResult("Test 3", "COMPONENT", "FAIL", 1.0, {})
        ]
        
        recommendations = tester._generate_recommendations()
        
        assert len(recommendations) > 0
        # Should recommend addressing failing tests
        failing_test_rec = any("failing tests" in rec.lower() for rec in recommendations)
        assert failing_test_rec
    
    def test_integration_report_generation(self):
        """Test integration report generation"""
        tester = SystemIntegrationTester()
        tester.start_time = datetime.now()
        tester.end_time = datetime.now()
        
        # Mock test results
        tester.test_results = [
            TestResult("Test 1", "COMPONENT", "PASS", 1.0, {'avg_response_time_ms': 50}),
            TestResult("Test 2", "PERFORMANCE", "PASS", 1.0, {'avg_response_time_ms': 75})
        ]
        
        report = tester._generate_integration_report("test_session")
        
        assert isinstance(report, IntegrationTestReport)
        assert report.test_session_id == "test_session"
        assert report.total_tests == 2
        assert report.passed_tests == 2
        assert report.failed_tests == 0
        assert report.success_rate == 1.0
        assert report.avg_response_time_ms == 62.5  # Average of 50 and 75
        assert report.system_health_score > 0
        assert report.reliability_score > 0
        assert report.performance_score > 0


class TestFactoryFunction:
    """Test factory function"""
    
    def test_create_integration_tester_default(self):
        """Test factory function with default settings"""
        tester = create_integration_tester()
        
        assert isinstance(tester, SystemIntegrationTester)
        assert tester.config.enable_performance_tests is True
        assert tester.config.enable_load_tests is True
        assert tester.config.concurrent_users == 10
    
    def test_create_integration_tester_limited(self):
        """Test factory function with limited tests"""
        tester = create_integration_tester(enable_all_tests=False)
        
        assert isinstance(tester, SystemIntegrationTester)
        assert tester.config.enable_performance_tests is False
        assert tester.config.enable_load_tests is False
        assert tester.config.concurrent_users == 5


class TestIntegrationTestExecution:
    """Test full integration test execution"""
    
    @pytest.mark.asyncio
    async def test_minimal_integration_test(self):
        """Test minimal integration test execution"""
        config = IntegrationTestConfig(
            enable_performance_tests=False,
            enable_load_tests=False, 
            enable_stress_tests=False,
            concurrent_users=1
        )
        tester = SystemIntegrationTester(config)
        
        # Run a minimal integration test
        report = await tester.run_full_integration_test()
        
        assert isinstance(report, IntegrationTestReport)
        assert report.total_tests >= 4  # At least component tests
        assert report.success_rate >= 0
        assert report.integration_score >= 0
        assert isinstance(report.test_results, list)
        assert len(report.test_results) >= 4
    
    @pytest.mark.asyncio
    async def test_full_integration_test_short(self):
        """Test full integration test with short durations"""
        config = IntegrationTestConfig(
            concurrent_users=2,
            test_duration_seconds=3,
            stress_duration_seconds=3
        )
        tester = SystemIntegrationTester(config)
        
        # Run full integration test with short timeouts
        report = await tester.run_full_integration_test()
        
        assert isinstance(report, IntegrationTestReport)
        assert report.total_tests >= 6  # Component + performance + reliability
        assert report.success_rate >= 0
        assert report.integration_score >= 0
        
        # Check that we have different test types
        test_types = set(r.test_type for r in report.test_results)
        assert "COMPONENT" in test_types
        assert "PERFORMANCE" in test_types or "RELIABILITY" in test_types


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])