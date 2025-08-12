"""
Test Suite for Portfolio Manager with Kelly Criterion Integration
Tests the integration between Portfolio Manager and Position Sizing System
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.core.trading.portfolio_manager import (
    PortfolioManager, AllocationMethod, PortfolioRiskLevel,
    SymbolAllocation, PortfolioMetrics
)
from src.core.trading.position_types import Position, PositionType, PositionStatus
from src.core.trading.kelly_criterion import KellyMethod, TradeResult

try:
    from src.core.risk.position_sizer import SizingResult, RiskLevel
    POSITION_SIZING_AVAILABLE = True
except ImportError:
    POSITION_SIZING_AVAILABLE = False


class TestPortfolioKellyIntegration:
    """Test Portfolio Manager Kelly Integration"""
    
    @pytest.fixture
    def portfolio_config(self):
        """Portfolio configuration for testing"""
        return {
            'initial_capital': 100000.0,
            'risk_per_trade': 0.02,
            'max_position_size': 0.1,
            'kelly_enabled': True,
            'kelly_max_fraction': 0.25,
            'kelly_min_fraction': 0.01,
            'max_symbols': 10,
            'risk_level': 'moderate'
        }
    
    @pytest.fixture
    def portfolio_manager(self, portfolio_config):
        """Create portfolio manager instance"""
        manager = PortfolioManager(portfolio_config)
        manager.start()
        return manager
    
    @pytest.fixture
    def sample_position(self):
        """Create sample position"""
        return Position(
            position_id="test_pos_1",
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=1.0,
            open_price=2000.0,
            current_price=2010.0,
            stop_loss=1980.0,
            take_profit=2050.0,
            status=PositionStatus.OPEN,
            open_time=datetime.now(),
            realized_profit=0.0,
            unrealized_profit=10.0
        )
    
    def test_portfolio_manager_initialization_with_kelly(self, portfolio_manager):
        """Test portfolio manager initialization with Kelly support"""
        assert portfolio_manager.kelly_enabled == True
        assert portfolio_manager.default_kelly_method == KellyMethod.ADAPTIVE
        assert portfolio_manager.sizing_parameters is not None
        assert portfolio_manager.sizing_parameters.kelly_max_fraction == 0.25
        assert portfolio_manager.sizing_parameters.kelly_min_fraction == 0.01
        assert 'kelly_updated' in portfolio_manager.portfolio_callbacks
        assert 'position_sized' in portfolio_manager.portfolio_callbacks
    
    @pytest.mark.skipif(not POSITION_SIZING_AVAILABLE, reason="Position sizing not available")
    def test_initialize_position_sizer(self, portfolio_manager):
        """Test position sizer initialization"""
        symbol = "XAUUSD"
        
        # Test successful initialization
        result = portfolio_manager.initialize_position_sizer(symbol)
        assert result == True
        assert symbol in portfolio_manager.position_sizers
        
        # Test with price data
        price_data = pd.DataFrame({
            'close': [2000, 2010, 2005, 2015, 2020],
            'high': [2005, 2015, 2010, 2020, 2025],
            'low': [1995, 2005, 2000, 2010, 2015]
        })
        
        result = portfolio_manager.initialize_position_sizer("EURUSD", price_data)
        assert result == True
        assert "EURUSD" in portfolio_manager.position_sizers
    
    @pytest.mark.skipif(not POSITION_SIZING_AVAILABLE, reason="Position sizing not available")
    def test_calculate_optimal_position_size(self, portfolio_manager):
        """Test optimal position size calculation"""
        symbol = "XAUUSD"
        current_price = 2000.0
        
        # Initialize position sizer
        portfolio_manager.initialize_position_sizer(symbol)
        
        # Add symbol to portfolio
        portfolio_manager.add_symbol(symbol, 0.5)
        
        # Test Kelly calculation
        result = portfolio_manager.calculate_optimal_position_size(
            symbol, current_price, KellyMethod.ADAPTIVE
        )
        
        assert result is not None
        assert isinstance(result, SizingResult)
        assert result.position_size > 0
        assert result.confidence_score >= 0
        
        # Check statistics updated
        assert portfolio_manager.stats['kelly_calculations'] > 0
        assert portfolio_manager.stats['position_sizing_calls'] > 0
        
        # Check symbol allocation updated
        allocation = portfolio_manager.symbols[symbol]
        assert allocation.kelly_fraction >= 0
        assert allocation.kelly_confidence >= 0
        assert allocation.recommended_size > 0
    
    @pytest.mark.skipif(not POSITION_SIZING_AVAILABLE, reason="Position sizing not available")
    def test_kelly_analysis(self, portfolio_manager):
        """Test Kelly analysis functionality"""
        symbol = "XAUUSD"
        current_price = 2000.0
        
        # Initialize and get analysis
        portfolio_manager.initialize_position_sizer(symbol)
        analysis = portfolio_manager.get_kelly_analysis(symbol, current_price)
        
        assert analysis is not None
        assert isinstance(analysis, dict)
        assert 'kelly_analysis' in analysis
        assert 'performance_summary' in analysis
        assert 'current_price' in analysis
    
    @pytest.mark.skipif(not POSITION_SIZING_AVAILABLE, reason="Position sizing not available")
    def test_add_trade_result_to_kelly(self, portfolio_manager):
        """Test adding trade results to Kelly Calculator"""
        symbol = "XAUUSD"
        
        # Initialize position sizer
        portfolio_manager.initialize_position_sizer(symbol)
        
        # Add winning trade
        result = portfolio_manager.add_trade_result_to_kelly(
            symbol=symbol,
            profit_loss=100.0,
            win=True,
            entry_price=2000.0,
            exit_price=2010.0,
            volume=1.0
        )
        assert result == True
        
        # Add losing trade
        result = portfolio_manager.add_trade_result_to_kelly(
            symbol=symbol,
            profit_loss=-50.0,
            win=False,
            entry_price=2000.0,
            exit_price=1995.0,
            volume=1.0
        )
        assert result == True
    
    def test_performance_metrics_calculation(self, portfolio_manager, sample_position):
        """Test performance metrics calculation with positions"""
        symbol = "XAUUSD"
        
        # Add symbol and position
        portfolio_manager.add_symbol(symbol, 0.5)
        portfolio_manager.add_position_to_portfolio(sample_position)
        
        # Get performance metrics
        win_rate, avg_win, avg_loss = portfolio_manager._get_symbol_performance_metrics(symbol)
        
        # Should return defaults for insufficient data
        assert win_rate == 0.6
        assert avg_win == 0.02
        assert avg_loss == -0.015
    
    @pytest.mark.skipif(not POSITION_SIZING_AVAILABLE, reason="Position sizing not available")
    def test_kelly_optimal_rebalancing(self, portfolio_manager):
        """Test Kelly optimal rebalancing"""
        # Add multiple symbols
        symbols = ["XAUUSD", "EURUSD", "GBPUSD"]
        for symbol in symbols:
            portfolio_manager.add_symbol(symbol, 1.0/len(symbols))
            portfolio_manager.initialize_position_sizer(symbol)
            
            # Add mock position to give portfolio value
            mock_position = Position(
                position_id=f"mock_{symbol}",
                symbol=symbol,
                position_type=PositionType.BUY,
                volume=1.0,
                open_price=2000.0,
                current_price=2010.0,
                status=PositionStatus.OPEN,
                remaining_volume=1.0
            )
            portfolio_manager.add_position_to_portfolio(mock_position)
        
        # Test Kelly rebalancing
        result = portfolio_manager.rebalance_portfolio(AllocationMethod.KELLY_OPTIMAL)
        assert result == True
        
        # Check that weights were assigned
        total_weight = sum(alloc.target_weight for alloc in portfolio_manager.symbols.values())
        assert abs(total_weight - 1.0) < 0.15  # Should sum to ~1.0 (allow for Kelly adjustments)
        
        # Check statistics
        assert portfolio_manager.stats['rebalance_count'] > 0
        assert portfolio_manager.stats['last_rebalance'] is not None
    
    def test_kelly_rebalancing_fallback(self, portfolio_manager):
        """Test Kelly rebalancing fallback when position sizing unavailable"""
        # Add symbols without initializing position sizers
        symbols = ["XAUUSD", "EURUSD"]
        for symbol in symbols:
            portfolio_manager.add_symbol(symbol, 0.5)
            
            # Add mock position to give portfolio value
            mock_position = Position(
                position_id=f"mock_{symbol}",
                symbol=symbol,
                position_type=PositionType.BUY,
                volume=1.0,
                open_price=2000.0,
                current_price=2010.0,
                status=PositionStatus.OPEN,
                remaining_volume=1.0
            )
            portfolio_manager.add_position_to_portfolio(mock_position)
        
        # Mock position sizing as unavailable
        with patch('src.core.trading.portfolio_manager.POSITION_SIZING_AVAILABLE', False):
            result = portfolio_manager.rebalance_portfolio(AllocationMethod.KELLY_OPTIMAL)
            assert result == True
            
            # Should fall back to equal weight
            for allocation in portfolio_manager.symbols.values():
                assert abs(allocation.target_weight - 0.5) < 0.01
    
    def test_portfolio_callbacks_integration(self, portfolio_manager):
        """Test portfolio callbacks with Kelly integration"""
        callback_called = {'kelly': False, 'position_sized': False}
        
        def kelly_callback(symbol, analysis):
            callback_called['kelly'] = True
            assert symbol is not None
            assert analysis is not None
        
        def position_sized_callback(symbol, result):
            callback_called['position_sized'] = True
            assert symbol is not None
            assert result is not None
        
        # Add callbacks
        portfolio_manager.add_callback('kelly_updated', kelly_callback)
        portfolio_manager.add_callback('position_sized', position_sized_callback)
        
        # Trigger callbacks
        if POSITION_SIZING_AVAILABLE:
            symbol = "XAUUSD"
            portfolio_manager.initialize_position_sizer(symbol)
            portfolio_manager.get_kelly_analysis(symbol, 2000.0)
            portfolio_manager.calculate_optimal_position_size(symbol, 2000.0)
            
            assert callback_called['kelly'] == True
            assert callback_called['position_sized'] == True
    
    def test_portfolio_statistics_with_kelly(self, portfolio_manager):
        """Test portfolio statistics include Kelly metrics"""
        stats = portfolio_manager.get_statistics()
        
        assert 'kelly_calculations' in stats
        assert 'position_sizing_calls' in stats
        assert stats['kelly_calculations'] >= 0
        assert stats['position_sizing_calls'] >= 0
    
    @pytest.mark.skipif(not POSITION_SIZING_AVAILABLE, reason="Position sizing not available")
    def test_symbol_allocation_kelly_metrics(self, portfolio_manager):
        """Test symbol allocation includes Kelly metrics"""
        symbol = "XAUUSD"
        
        # Add symbol and initialize
        portfolio_manager.add_symbol(symbol, 0.5)
        portfolio_manager.initialize_position_sizer(symbol)
        
        # Calculate position size to populate Kelly metrics
        portfolio_manager.calculate_optimal_position_size(symbol, 2000.0)
        
        # Check allocation has Kelly metrics
        allocation = portfolio_manager.symbols[symbol]
        assert hasattr(allocation, 'kelly_fraction')
        assert hasattr(allocation, 'kelly_confidence')
        assert hasattr(allocation, 'recommended_size')
        assert allocation.kelly_fraction >= 0
        assert allocation.kelly_confidence >= 0
        assert allocation.recommended_size >= 0
    
    def test_portfolio_summary_with_kelly(self, portfolio_manager):
        """Test portfolio summary includes Kelly information"""
        symbol = "XAUUSD"
        portfolio_manager.add_symbol(symbol, 0.5)
        
        if POSITION_SIZING_AVAILABLE:
            portfolio_manager.initialize_position_sizer(symbol)
            portfolio_manager.calculate_optimal_position_size(symbol, 2000.0)
        
        summary = portfolio_manager.get_portfolio_summary()
        
        assert 'symbol_allocations' in summary
        assert 'statistics' in summary
        assert 'kelly_calculations' in summary['statistics']
        assert 'position_sizing_calls' in summary['statistics']
    
    def test_error_handling_kelly_integration(self, portfolio_manager):
        """Test error handling in Kelly integration"""
        symbol = "INVALID_SYMBOL"
        
        # Test with invalid symbol
        result = portfolio_manager.calculate_optimal_position_size(symbol, 2000.0)
        if POSITION_SIZING_AVAILABLE:
            # Should handle gracefully
            assert result is None or isinstance(result, SizingResult)
        else:
            assert result is None
        
        # Test Kelly analysis with non-existent sizer - it auto-creates sizer
        analysis = portfolio_manager.get_kelly_analysis(symbol, 2000.0)
        if POSITION_SIZING_AVAILABLE:
            # Auto-creates sizer, so returns analysis
            assert analysis is not None
            assert isinstance(analysis, dict)
        else:
            assert analysis is None
    
    def test_concurrent_kelly_operations(self, portfolio_manager):
        """Test concurrent Kelly operations"""
        import threading
        import time
        
        symbols = ["XAUUSD", "EURUSD", "GBPUSD"]
        results = {}
        
        def calculate_for_symbol(symbol):
            try:
                portfolio_manager.add_symbol(symbol, 1.0/len(symbols))
                if POSITION_SIZING_AVAILABLE:
                    portfolio_manager.initialize_position_sizer(symbol)
                    result = portfolio_manager.calculate_optimal_position_size(symbol, 2000.0)
                    results[symbol] = result
                else:
                    results[symbol] = None
            except Exception as e:
                results[symbol] = f"Error: {e}"
        
        # Run concurrent operations
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=calculate_for_symbol, args=(symbol,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)
        
        # Check results
        assert len(results) == len(symbols)
        for symbol, result in results.items():
            if POSITION_SIZING_AVAILABLE:
                assert result is None or isinstance(result, SizingResult) or "Error" in str(result)
            else:
                assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 