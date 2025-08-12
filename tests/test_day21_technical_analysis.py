"""
Test Suite for Day 21 Technical Analysis Foundation
Ultimate XAU Super System V4.0

Comprehensive testing for technical analysis capabilities:
- Technical indicator calculations
- Pattern recognition engine
- Signal generation and validation
- Multi-timeframe analysis
- Performance validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add source path
sys.path.append('src')

# Import technical analysis components
try:
    from src.core.analysis.technical_analysis import (
        create_technical_analyzer, TechnicalAnalyzer, IndicatorConfig,
        TechnicalSignal, PatternResult, TechnicalIndicators, PatternRecognition
    )
    TECHNICAL_ANALYZER_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYZER_AVAILABLE = False
    pytest.skip("Technical Analysis not available", allow_module_level=True)


class TestIndicatorConfig:
    """Test IndicatorConfig class"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = IndicatorConfig()
        
        assert isinstance(config.sma_periods, list)
        assert isinstance(config.ema_periods, list)
        assert isinstance(config.rsi_period, int)
        assert isinstance(config.bb_period, int)
        assert isinstance(config.bb_std, float)
        
        # Check default values
        assert config.rsi_period == 14
        assert config.bb_period == 20
        assert config.bb_std == 2.0
    
    def test_custom_config_creation(self):
        """Test custom configuration creation"""
        config = IndicatorConfig(
            sma_periods=[10, 20, 50],
            rsi_period=21,
            bb_std=2.5
        )
        
        assert config.sma_periods == [10, 20, 50]
        assert config.rsi_period == 21
        assert config.bb_std == 2.5


class TestTechnicalSignal:
    """Test TechnicalSignal class"""
    
    def test_signal_creation(self):
        """Test signal creation"""
        signal = TechnicalSignal(
            indicator_name="RSI",
            signal_type="BUY",
            strength=0.8,
            value=30.5,
            timestamp=datetime.now(),
            timeframe="1h",
            description="RSI oversold"
        )
        
        assert signal.indicator_name == "RSI"
        assert signal.signal_type == "BUY"
        assert signal.strength == 0.8
        assert signal.value == 30.5
        assert isinstance(signal.timestamp, datetime)
        assert signal.timeframe == "1h"
        assert signal.description == "RSI oversold"
        assert signal.confidence == 0.5  # default value


class TestPatternResult:
    """Test PatternResult class"""
    
    def test_pattern_creation(self):
        """Test pattern result creation"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=4)
        
        pattern = PatternResult(
            pattern_name="Double Top",
            pattern_type="BEARISH",
            confidence=0.85,
            start_time=start_time,
            end_time=end_time,
            target_price=2000.0,
            stop_loss=2050.0
        )
        
        assert pattern.pattern_name == "Double Top"
        assert pattern.pattern_type == "BEARISH"
        assert pattern.confidence == 0.85
        assert pattern.start_time == start_time
        assert pattern.end_time == end_time
        assert pattern.target_price == 2000.0
        assert pattern.stop_loss == 2050.0


class TestTechnicalIndicators:
    """Test TechnicalIndicators class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        
        # Generate realistic price data
        base_price = 2000.0
        returns = np.random.normal(0, 0.01, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = {
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def indicators(self):
        """Create indicators instance"""
        return TechnicalIndicators()
    
    def test_simple_moving_average(self, indicators, sample_data):
        """Test SMA calculation"""
        sma = indicators.simple_moving_average(sample_data['close'], 10)
        
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_data)
        assert pd.isna(sma.iloc[0:9]).all()  # First 9 values should be NaN
        assert not pd.isna(sma.iloc[9])  # 10th value should not be NaN
        
        # Test calculation correctness
        expected_sma_10 = sample_data['close'].iloc[0:10].mean()
        assert abs(sma.iloc[9] - expected_sma_10) < 0.01
    
    def test_exponential_moving_average(self, indicators, sample_data):
        """Test EMA calculation"""
        ema = indicators.exponential_moving_average(sample_data['close'], 12)
        
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_data)
        assert not pd.isna(ema.iloc[-1])  # Last value should not be NaN
    
    def test_relative_strength_index(self, indicators, sample_data):
        """Test RSI calculation"""
        rsi = indicators.relative_strength_index(sample_data['close'], 14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_stochastic_oscillator(self, indicators, sample_data):
        """Test Stochastic Oscillator calculation"""
        k_percent, d_percent = indicators.stochastic_oscillator(
            sample_data['high'], sample_data['low'], sample_data['close'], 14, 3
        )
        
        assert isinstance(k_percent, pd.Series)
        assert isinstance(d_percent, pd.Series)
        assert len(k_percent) == len(sample_data)
        assert len(d_percent) == len(sample_data)
        
        # Stochastic should be between 0 and 100
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()
    
    def test_macd(self, indicators, sample_data):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = indicators.macd(sample_data['close'])
        
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd_line) == len(sample_data)
        assert len(signal_line) == len(sample_data)
        assert len(histogram) == len(sample_data)
        
        # Histogram should equal MACD - Signal
        valid_indices = histogram.dropna().index
        for idx in valid_indices[-10:]:  # Check last 10 valid values
            expected_histogram = macd_line.loc[idx] - signal_line.loc[idx]
            assert abs(histogram.loc[idx] - expected_histogram) < 0.001
    
    def test_bollinger_bands(self, indicators, sample_data):
        """Test Bollinger Bands calculation"""
        upper_band, middle_band, lower_band = indicators.bollinger_bands(sample_data['close'])
        
        assert isinstance(upper_band, pd.Series)
        assert isinstance(middle_band, pd.Series)
        assert isinstance(lower_band, pd.Series)
        assert len(upper_band) == len(sample_data)
        assert len(middle_band) == len(sample_data)
        assert len(lower_band) == len(sample_data)
        
        # Upper band should be above middle, middle above lower
        valid_indices = upper_band.dropna().index
        for idx in valid_indices:
            assert upper_band.loc[idx] > middle_band.loc[idx]
            assert middle_band.loc[idx] > lower_band.loc[idx]
    
    def test_average_true_range(self, indicators, sample_data):
        """Test ATR calculation"""
        atr = indicators.average_true_range(
            sample_data['high'], sample_data['low'], sample_data['close']
        )
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_data)
        
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
    
    def test_vwap(self, indicators, sample_data):
        """Test VWAP calculation"""
        vwap = indicators.volume_weighted_average_price(
            sample_data['high'], sample_data['low'], 
            sample_data['close'], sample_data['volume']
        )
        
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
        assert not pd.isna(vwap.iloc[-1])  # Last value should not be NaN
    
    def test_commodity_channel_index(self, indicators, sample_data):
        """Test CCI calculation"""
        cci = indicators.commodity_channel_index(
            sample_data['high'], sample_data['low'], sample_data['close']
        )
        
        assert isinstance(cci, pd.Series)
        assert len(cci) == len(sample_data)


class TestPatternRecognition:
    """Test PatternRecognition class"""
    
    @pytest.fixture
    def sample_data_with_patterns(self):
        """Create sample data with embedded patterns"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='H')
        
        # Create data with double top pattern
        base_price = 2000.0
        prices = []
        
        for i in range(200):
            if i < 50:
                # Rising trend
                price = base_price + (i * 2)
            elif i < 70:
                # First peak
                price = base_price + 100 + np.sin((i-50) * 0.3) * 10
            elif i < 120:
                # Valley
                price = base_price + 80 + np.sin((i-70) * 0.2) * 15
            elif i < 140:
                # Second peak (similar to first)
                price = base_price + 95 + np.sin((i-120) * 0.3) * 10
            else:
                # Decline
                price = base_price + 90 - ((i-140) * 1.5)
            
            prices.append(max(price, base_price * 0.8))  # Prevent extreme drops
        
        data = {
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        }
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def pattern_recognition(self):
        """Create pattern recognition instance"""
        return PatternRecognition()
    
    def test_detect_double_top(self, pattern_recognition, sample_data_with_patterns):
        """Test double top pattern detection"""
        patterns = pattern_recognition.detect_double_top(sample_data_with_patterns)
        
        assert isinstance(patterns, list)
        # Should detect at least some patterns in constructed data
        for pattern in patterns:
            assert isinstance(pattern, PatternResult)
            assert pattern.pattern_name == "Double Top"
            assert pattern.pattern_type == "BEARISH"
            assert 0 <= pattern.confidence <= 1
    
    def test_detect_head_and_shoulders(self, pattern_recognition, sample_data_with_patterns):
        """Test head and shoulders pattern detection"""
        patterns = pattern_recognition.detect_head_and_shoulders(sample_data_with_patterns)
        
        assert isinstance(patterns, list)
        for pattern in patterns:
            assert isinstance(pattern, PatternResult)
            assert pattern.pattern_name == "Head and Shoulders"
            assert pattern.pattern_type == "BEARISH"
            assert 0 <= pattern.confidence <= 1
    
    def test_detect_support_resistance(self, pattern_recognition, sample_data_with_patterns):
        """Test support and resistance level detection"""
        levels = pattern_recognition.detect_support_resistance(sample_data_with_patterns)
        
        assert isinstance(levels, dict)
        assert 'support' in levels
        assert 'resistance' in levels
        assert isinstance(levels['support'], list)
        assert isinstance(levels['resistance'], list)
        
        # Support levels should be lower than resistance levels
        if levels['support'] and levels['resistance']:
            max_support = max(levels['support'])
            min_resistance = min(levels['resistance'])
            assert max_support <= min_resistance


class TestTechnicalAnalyzer:
    """Test TechnicalAnalyzer class"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create comprehensive sample market data"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        
        base_price = 2000.0
        returns = np.random.normal(0, 0.015, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = {
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def technical_analyzer(self):
        """Create technical analyzer instance"""
        config = IndicatorConfig(
            sma_periods=[10, 20, 50],
            ema_periods=[12, 26],
            rsi_period=14
        )
        return TechnicalAnalyzer(config)
    
    def test_analyzer_initialization(self, technical_analyzer):
        """Test analyzer initialization"""
        assert isinstance(technical_analyzer, TechnicalAnalyzer)
        assert isinstance(technical_analyzer.config, IndicatorConfig)
        assert isinstance(technical_analyzer.indicators, TechnicalIndicators)
        assert isinstance(technical_analyzer.pattern_recognition, PatternRecognition)
    
    def test_analyze_market_data_structure(self, technical_analyzer, sample_market_data):
        """Test market data analysis structure"""
        results = technical_analyzer.analyze_market_data(sample_market_data, "1h")
        
        # Check main structure
        assert isinstance(results, dict)
        required_keys = ['timeframe', 'timestamp', 'indicators', 'signals', 
                        'patterns', 'support_resistance', 'trend_analysis', 'summary']
        for key in required_keys:
            assert key in results
        
        # Check data types
        assert isinstance(results['timeframe'], str)
        assert isinstance(results['timestamp'], datetime)
        assert isinstance(results['indicators'], dict)
        assert isinstance(results['signals'], list)
        assert isinstance(results['patterns'], list)
        assert isinstance(results['support_resistance'], dict)
        assert isinstance(results['trend_analysis'], dict)
        assert isinstance(results['summary'], dict)
    
    def test_analyze_market_data_indicators(self, technical_analyzer, sample_market_data):
        """Test indicator calculations in market analysis"""
        results = technical_analyzer.analyze_market_data(sample_market_data, "1h")
        indicators = results['indicators']
        
        # Check SMA indicators
        assert 'sma' in indicators
        for period in technical_analyzer.config.sma_periods:
            assert f'sma_{period}' in indicators['sma']
            assert isinstance(indicators['sma'][f'sma_{period}'], pd.Series)
        
        # Check EMA indicators
        assert 'ema' in indicators
        for period in technical_analyzer.config.ema_periods:
            assert f'ema_{period}' in indicators['ema']
            assert isinstance(indicators['ema'][f'ema_{period}'], pd.Series)
        
        # Check other indicators
        assert 'rsi' in indicators
        assert 'stochastic' in indicators
        assert 'macd' in indicators
        assert 'bollinger_bands' in indicators
        assert 'atr' in indicators
        assert 'vwap' in indicators
        assert 'cci' in indicators
    
    def test_analyze_market_data_signals(self, technical_analyzer, sample_market_data):
        """Test signal generation in market analysis"""
        results = technical_analyzer.analyze_market_data(sample_market_data, "1h")
        signals = results['signals']
        
        # Check signal structure
        for signal in signals:
            assert hasattr(signal, 'indicator_name')
            assert hasattr(signal, 'signal_type')
            assert hasattr(signal, 'strength')
            assert hasattr(signal, 'value')
            assert hasattr(signal, 'timestamp')
            assert hasattr(signal, 'timeframe')
            assert hasattr(signal, 'description')
            assert hasattr(signal, 'confidence')
            
            # Check signal values
            assert signal.signal_type in ['BUY', 'SELL']
            assert 0 <= signal.strength <= 1
            assert 0 <= signal.confidence <= 1
            assert isinstance(signal.timestamp, datetime)
    
    def test_analyze_market_data_trend_analysis(self, technical_analyzer, sample_market_data):
        """Test trend analysis in market analysis"""
        results = technical_analyzer.analyze_market_data(sample_market_data, "1h")
        trend_analysis = results['trend_analysis']
        
        # Check trend analysis structure
        required_keys = ['short_term', 'medium_term', 'long_term', 'strength', 'description']
        for key in required_keys:
            assert key in trend_analysis
        
        # Check trend values
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            assert trend_analysis[timeframe] in ['BULLISH', 'BEARISH', 'NEUTRAL']
        
        assert 0 <= trend_analysis['strength'] <= 1
        assert isinstance(trend_analysis['description'], str)
    
    def test_analyze_market_data_summary(self, technical_analyzer, sample_market_data):
        """Test summary generation in market analysis"""
        results = technical_analyzer.analyze_market_data(sample_market_data, "1h")
        summary = results['summary']
        
        # Check summary structure
        required_keys = ['total_signals', 'buy_signals', 'sell_signals', 'patterns_detected',
                        'trend_direction', 'market_sentiment', 'confidence_score', 'recommendation']
        for key in required_keys:
            assert key in summary
        
        # Check summary values
        assert isinstance(summary['total_signals'], int)
        assert isinstance(summary['buy_signals'], int)
        assert isinstance(summary['sell_signals'], int)
        assert isinstance(summary['patterns_detected'], int)
        assert summary['market_sentiment'] in ['BULLISH', 'BEARISH', 'NEUTRAL']
        assert summary['recommendation'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= summary['confidence_score'] <= 1
    
    def test_analyze_insufficient_data(self, technical_analyzer):
        """Test analysis with insufficient data"""
        # Create data with only 10 periods (insufficient for most indicators)
        small_data = pd.DataFrame({
            'open': [2000, 2001, 2002, 2003, 2004],
            'high': [2005, 2006, 2007, 2008, 2009],
            'low': [1995, 1996, 1997, 1998, 1999],
            'close': [2002, 2003, 2004, 2005, 2006],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            technical_analyzer.analyze_market_data(small_data)
    
    def test_analyze_invalid_columns(self, technical_analyzer):
        """Test analysis with invalid columns"""
        invalid_data = pd.DataFrame({
            'price': [2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2010] * 10,
            'vol': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900] * 10
        })
        
        with pytest.raises(ValueError, match="Data must contain columns"):
            technical_analyzer.analyze_market_data(invalid_data)


class TestFactoryFunction:
    """Test factory function"""
    
    def test_create_technical_analyzer_default(self):
        """Test creating analyzer with default config"""
        analyzer = create_technical_analyzer()
        
        assert isinstance(analyzer, TechnicalAnalyzer)
        assert isinstance(analyzer.config, IndicatorConfig)
    
    def test_create_technical_analyzer_custom_config(self):
        """Test creating analyzer with custom config"""
        custom_config = {
            'sma_periods': [5, 15, 30],
            'rsi_period': 21,
            'bb_std': 2.5
        }
        
        analyzer = create_technical_analyzer(custom_config)
        
        assert isinstance(analyzer, TechnicalAnalyzer)
        assert analyzer.config.sma_periods == [5, 15, 30]
        assert analyzer.config.rsi_period == 21
        assert analyzer.config.bb_std == 2.5


class TestPerformanceMetrics:
    """Test performance and quality metrics"""
    
    @pytest.fixture
    def large_market_data(self):
        """Create large dataset for performance testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
        
        base_price = 2000.0
        returns = np.random.normal(0, 0.01, 1000)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = {
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_analysis_performance(self, large_market_data):
        """Test analysis performance with large dataset"""
        analyzer = create_technical_analyzer()
        
        import time
        start_time = time.time()
        results = analyzer.analyze_market_data(large_market_data, "1h")
        end_time = time.time()
        
        # Analysis should complete within reasonable time
        analysis_time = end_time - start_time
        assert analysis_time < 10  # Should complete within 10 seconds
        
        # Should produce comprehensive results
        assert len(results['indicators']) > 5
        assert 'rsi' in results['indicators']
        assert 'sma' in results['indicators']
        assert 'macd' in results['indicators']
    
    def test_signal_quality_metrics(self, large_market_data):
        """Test signal quality metrics"""
        analyzer = create_technical_analyzer()
        results = analyzer.analyze_market_data(large_market_data, "1h")
        
        signals = results['signals']
        if signals:
            # Check signal quality
            strengths = [s.strength for s in signals]
            confidences = [s.confidence for s in signals]
            
            # All strengths should be valid
            assert all(0 <= strength <= 1 for strength in strengths)
            assert all(0 <= confidence <= 1 for confidence in confidences)
            
            # Should have reasonable average quality
            avg_strength = np.mean(strengths)
            avg_confidence = np.mean(confidences)
            
            assert avg_strength > 0.1  # Should have some signal strength
            assert avg_confidence > 0.1  # Should have some confidence


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])