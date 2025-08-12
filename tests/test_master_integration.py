"""
Test Suite for Master Integration System
Tests the unified integration of all system components
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import threading
import time

from src.core.integration.master_system import (
    MasterIntegrationSystem, SystemConfig, SystemMode, IntegrationLevel,
    MarketData, TradingSignal, SystemState,
    create_development_system, create_simulation_system, create_live_trading_system
)


class TestMasterIntegration:
    """Test Master Integration System"""
    
    @pytest.fixture
    def basic_config(self):
        """Create basic system configuration"""
        return SystemConfig(
            mode=SystemMode.TESTING,
            integration_level=IntegrationLevel.BASIC,
            initial_balance=100000.0,
            use_neural_ensemble=False,
            use_reinforcement_learning=False,
            enable_logging=False
        )
    
    @pytest.fixture
    def full_config(self):
        """Create full system configuration"""
        return SystemConfig(
            mode=SystemMode.SIMULATION,
            integration_level=IntegrationLevel.FULL,
            initial_balance=250000.0,
            use_neural_ensemble=True,
            use_reinforcement_learning=True,
            enable_logging=True
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        return MarketData(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            price=2000.0,
            high=2010.0,
            low=1990.0,
            volume=10000.0,
            technical_indicators={
                'rsi': 55.0,
                'macd': 2.5,
                'sma_20': 1995.0,
                'sma_50': 1985.0
            }
        )
    
    def test_system_config_creation(self):
        """Test SystemConfig creation"""
        config = SystemConfig()
        
        assert config.mode == SystemMode.SIMULATION
        assert config.integration_level == IntegrationLevel.FULL
        assert config.initial_balance == 100000.0
        assert config.use_neural_ensemble == True
        assert config.use_reinforcement_learning == True
    
    def test_market_data_creation(self, sample_market_data):
        """Test MarketData creation and conversion"""
        data = sample_market_data
        
        assert data.symbol == "XAUUSD"
        assert data.price == 2000.0
        assert data.high == 2010.0
        assert data.low == 1990.0
        assert data.volume == 10000.0
        
        # Test conversion to dict
        data_dict = data.to_dict()
        assert 'timestamp' in data_dict
        assert 'symbol' in data_dict
        assert 'close' in data_dict
        assert data_dict['close'] == 2000.0
        assert 'rsi' in data_dict
        assert data_dict['rsi'] == 55.0
    
    def test_trading_signal_creation(self):
        """Test TradingSignal creation"""
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            signal_type="BUY",
            confidence=0.8,
            source="test",
            target_price=2010.0,
            position_size=0.1,
            expected_return=0.005
        )
        
        assert signal.symbol == "XAUUSD"
        assert signal.signal_type == "BUY"
        assert signal.confidence == 0.8
        assert signal.source == "test"
        assert signal.target_price == 2010.0
        assert signal.position_size == 0.1
        assert signal.expected_return == 0.005
    
    def test_system_state_creation(self):
        """Test SystemState creation"""
        state = SystemState(
            timestamp=datetime.now(),
            mode=SystemMode.TESTING,
            total_balance=100000.0,
            available_balance=95000.0,
            total_positions=2,
            unrealized_pnl=500.0
        )
        
        assert state.mode == SystemMode.TESTING
        assert state.total_balance == 100000.0
        assert state.available_balance == 95000.0
        assert state.total_positions == 2
        assert state.unrealized_pnl == 500.0
    
    def test_master_system_initialization_basic(self, basic_config):
        """Test basic system initialization"""
        system = MasterIntegrationSystem(basic_config)
        
        assert system.config == basic_config
        assert system.state.mode == SystemMode.TESTING
        assert system.state.total_balance == 100000.0
        assert system.state.available_balance == 100000.0
        assert system.state.total_positions == 0
        assert isinstance(system.components, dict)
        assert isinstance(system.signals_history, list)
        assert isinstance(system.market_data_buffer, list)
    
    def test_master_system_initialization_full(self, full_config):
        """Test full system initialization"""
        system = MasterIntegrationSystem(full_config)
        
        assert system.config == full_config
        assert system.state.mode == SystemMode.SIMULATION
        assert system.state.total_balance == 250000.0
        
        # Check component status update
        assert isinstance(system.state.components_status, dict)
        assert isinstance(system.state.neural_ensemble_active, bool)
        assert isinstance(system.state.rl_agent_active, bool)
    
    def test_component_status_update(self, basic_config):
        """Test component status tracking"""
        system = MasterIntegrationSystem(basic_config)
        
        # Initially should have some components
        status = system.state.components_status
        assert isinstance(status, dict)
        
        # Check that status reflects actual components
        for name, active in status.items():
            assert isinstance(active, bool)
            if active:
                assert name in system.components
    
    def test_add_market_data(self, basic_config, sample_market_data):
        """Test adding market data"""
        system = MasterIntegrationSystem(basic_config)
        
        initial_buffer_size = len(system.market_data_buffer)
        system.add_market_data(sample_market_data)
        
        assert len(system.market_data_buffer) == initial_buffer_size + 1
        assert system.market_data_buffer[-1] == sample_market_data
    
    def test_market_data_buffer_limit(self, basic_config):
        """Test market data buffer size limit"""
        system = MasterIntegrationSystem(basic_config)
        
        # Add more than 1000 data points
        for i in range(1050):
            data = MarketData(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                price=2000.0 + i,
                high=2010.0 + i,
                low=1990.0 + i,
                volume=10000.0
            )
            system.add_market_data(data)
        
        # Should be limited to 1000
        assert len(system.market_data_buffer) == 1000
        
        # Should contain the most recent data
        assert system.market_data_buffer[-1].price == 2000.0 + 1049
    
    def test_signal_combination_empty(self, basic_config, sample_market_data):
        """Test signal combination with no signals"""
        system = MasterIntegrationSystem(basic_config)
        
        combined = system._combine_signals([], sample_market_data)
        assert combined is None
    
    def test_signal_combination_single(self, basic_config, sample_market_data):
        """Test signal combination with single signal"""
        system = MasterIntegrationSystem(basic_config)
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            signal_type="BUY",
            confidence=0.8,
            source="test",
            position_size=0.1
        )
        
        combined = system._combine_signals([signal], sample_market_data)
        
        assert combined is not None
        assert combined.symbol == "XAUUSD"
        assert combined.source == "integrated_system"
        assert combined.confidence == 0.8
    
    def test_signal_combination_multiple(self, basic_config, sample_market_data):
        """Test signal combination with multiple signals"""
        system = MasterIntegrationSystem(basic_config)
        
        signals = [
            TradingSignal(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                signal_type="BUY",
                confidence=0.8,
                source="neural_ensemble",
                position_size=0.1
            ),
            TradingSignal(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                signal_type="BUY",
                confidence=0.7,
                source="reinforcement_learning",
                position_size=0.15
            ),
            TradingSignal(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                signal_type="HOLD",
                confidence=0.6,
                source="risk_management",
                position_size=0.05
            )
        ]
        
        combined = system._combine_signals(signals, sample_market_data)
        
        assert combined is not None
        assert combined.symbol == "XAUUSD"
        assert combined.source == "integrated_system"
        assert 0.6 <= combined.confidence <= 0.8
        assert 'signal_scores' in combined.metadata
        assert 'component_signals' in combined.metadata
        assert combined.metadata['component_signals'] == 3
    
    def test_signal_execution(self, basic_config):
        """Test signal execution"""
        system = MasterIntegrationSystem(basic_config)
        
        initial_balance = system.state.available_balance
        initial_positions = system.state.total_positions
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            signal_type="BUY",
            confidence=0.8,
            source="test",
            position_size=0.1
        )
        
        system._execute_signal(signal)
        
        # Check that state was updated
        assert system.state.last_update is not None
        assert system.state.last_prediction_confidence == 0.8
    
    def test_system_metrics_update(self, basic_config):
        """Test system metrics calculation"""
        system = MasterIntegrationSystem(basic_config)
        
        # Add some signals with returns
        for i in range(10):
            signal = TradingSignal(
                timestamp=datetime.now() - timedelta(minutes=i),
                symbol="XAUUSD",
                signal_type="BUY",
                confidence=0.8,
                source="test",
                expected_return=0.01 if i % 2 == 0 else -0.005
            )
            system.signals_history.append(signal)
        
        system._update_system_metrics()
        
        assert system.state.total_return is not None
        assert system.state.sharpe_ratio is not None
        assert system.state.win_rate is not None
        assert 0 <= system.state.win_rate <= 1
    
    def test_get_system_status(self, basic_config):
        """Test system status retrieval"""
        system = MasterIntegrationSystem(basic_config)
        
        status = system.get_system_status()
        
        assert 'timestamp' in status
        assert 'mode' in status
        assert 'integration_level' in status
        assert 'components_active' in status
        assert 'total_components' in status
        assert 'portfolio' in status
        assert 'performance' in status
        assert 'ai_status' in status
        assert 'signals' in status
        assert 'components_status' in status
        
        # Check portfolio section
        portfolio = status['portfolio']
        assert 'total_balance' in portfolio
        assert 'available_balance' in portfolio
        assert 'total_positions' in portfolio
        assert 'unrealized_pnl' in portfolio
        
        # Check performance section
        performance = status['performance']
        assert 'total_return' in performance
        assert 'sharpe_ratio' in performance
        assert 'win_rate' in performance
        
        # Check AI status
        ai_status = status['ai_status']
        assert 'neural_ensemble_active' in ai_status
        assert 'rl_agent_active' in ai_status
        assert 'last_prediction_confidence' in ai_status
    
    def test_get_recent_signals(self, basic_config):
        """Test recent signals retrieval"""
        system = MasterIntegrationSystem(basic_config)
        
        # Add signals at different times
        now = datetime.now()
        signals = [
            TradingSignal(
                timestamp=now - timedelta(minutes=30),
                symbol="XAUUSD",
                signal_type="BUY",
                confidence=0.8,
                source="test"
            ),
            TradingSignal(
                timestamp=now - timedelta(hours=2),
                symbol="XAUUSD",
                signal_type="SELL",
                confidence=0.7,
                source="test"
            ),
            TradingSignal(
                timestamp=now - timedelta(minutes=10),
                symbol="XAUUSD",
                signal_type="HOLD",
                confidence=0.6,
                source="test"
            )
        ]
        
        system.signals_history.extend(signals)
        
        # Get recent signals (last hour)
        recent = system.get_recent_signals(hours=1)
        
        assert len(recent) == 2  # Only signals within last hour
        assert all(s.timestamp > now - timedelta(hours=1) for s in recent)
    
    def test_system_reset(self, basic_config, sample_market_data):
        """Test system reset functionality"""
        system = MasterIntegrationSystem(basic_config)
        
        # Modify system state
        system.add_market_data(sample_market_data)
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            signal_type="BUY",
            confidence=0.8,
            source="test"
        )
        system.signals_history.append(signal)
        system.state.total_balance = 95000.0
        
        # Reset system
        system.reset_system()
        
        # Check that system is back to initial state
        assert system.state.total_balance == basic_config.initial_balance
        assert system.state.available_balance == basic_config.initial_balance
        assert system.state.total_positions == 0
        assert len(system.signals_history) == 0
        assert len(system.market_data_buffer) == 0
    
    def test_real_time_processing_start_stop(self, basic_config):
        """Test real-time processing start and stop"""
        system = MasterIntegrationSystem(basic_config)
        
        assert not system._running
        
        # Start processing
        system.start_real_time_processing()
        assert system._running
        assert system._update_thread is not None
        
        # Give it a moment to start
        time.sleep(0.1)
        
        # Stop processing
        system.stop_real_time_processing()
        assert not system._running
    
    def test_factory_functions(self):
        """Test factory functions for different system types"""
        # Development system
        dev_system = create_development_system()
        assert dev_system.config.mode == SystemMode.DEVELOPMENT
        assert dev_system.config.integration_level == IntegrationLevel.FULL
        assert dev_system.config.initial_balance == 100000.0
        
        # Simulation system
        sim_system = create_simulation_system()
        assert sim_system.config.mode == SystemMode.SIMULATION
        assert sim_system.config.integration_level == IntegrationLevel.FULL
        assert sim_system.config.initial_balance == 250000.0
        
        # Live trading system
        live_system = create_live_trading_system()
        assert live_system.config.mode == SystemMode.LIVE_TRADING
        assert live_system.config.integration_level == IntegrationLevel.FULL
        assert live_system.config.initial_balance == 500000.0
        assert live_system.config.rl_exploration_rate == 0.05  # Lower for live
    
    def test_integration_levels(self):
        """Test different integration levels"""
        # Basic integration
        basic_config = SystemConfig(integration_level=IntegrationLevel.BASIC)
        basic_system = MasterIntegrationSystem(basic_config)
        
        # Moderate integration
        moderate_config = SystemConfig(integration_level=IntegrationLevel.MODERATE)
        moderate_system = MasterIntegrationSystem(moderate_config)
        
        # Advanced integration
        advanced_config = SystemConfig(integration_level=IntegrationLevel.ADVANCED)
        advanced_system = MasterIntegrationSystem(advanced_config)
        
        # Full integration
        full_config = SystemConfig(integration_level=IntegrationLevel.FULL)
        full_system = MasterIntegrationSystem(full_config)
        
        # Each should have different component configurations
        assert isinstance(basic_system.components, dict)
        assert isinstance(moderate_system.components, dict)
        assert isinstance(advanced_system.components, dict)
        assert isinstance(full_system.components, dict)
    
    def test_error_handling_invalid_signal(self, basic_config):
        """Test error handling with invalid signals"""
        system = MasterIntegrationSystem(basic_config)
        
        # Test with None signal
        system._execute_signal(None)  # Should not crash
        
        # Test with invalid signal type
        invalid_signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="INVALID",
            signal_type="INVALID_TYPE",
            confidence=1.5,  # Invalid confidence > 1
            source="test"
        )
        
        system._execute_signal(invalid_signal)  # Should handle gracefully
    
    def test_concurrent_access(self, basic_config, sample_market_data):
        """Test thread safety with concurrent access"""
        system = MasterIntegrationSystem(basic_config)
        
        def add_data_worker():
            for i in range(10):
                data = MarketData(
                    timestamp=datetime.now(),
                    symbol="XAUUSD",
                    price=2000.0 + i,
                    high=2010.0 + i,
                    low=1990.0 + i,
                    volume=10000.0
                )
                system.add_market_data(data)
                time.sleep(0.001)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_data_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have 30 data points (3 threads × 10 each)
        assert len(system.market_data_buffer) == 30
    
    def test_system_mode_enum(self):
        """Test SystemMode enum values"""
        assert SystemMode.DEVELOPMENT.value == "development"
        assert SystemMode.TESTING.value == "testing"
        assert SystemMode.SIMULATION.value == "simulation"
        assert SystemMode.LIVE_TRADING.value == "live_trading"
    
    def test_integration_level_enum(self):
        """Test IntegrationLevel enum values"""
        assert IntegrationLevel.BASIC.value == "basic"
        assert IntegrationLevel.MODERATE.value == "moderate"
        assert IntegrationLevel.ADVANCED.value == "advanced"
        assert IntegrationLevel.FULL.value == "full"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])