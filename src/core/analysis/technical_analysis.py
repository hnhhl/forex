"""
Technical Analysis Foundation
Ultimate XAU Super System V4.0 - Day 21 Implementation

Advanced technical analysis system:
- Comprehensive indicator library
- Pattern recognition engine
- Multi-timeframe analysis
- Signal generation and validation
- Real-time chart analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    
    # Moving averages
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50, 100])
    
    # Oscillators
    rsi_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR
    atr_period: int = 14
    
    # Volume indicators
    volume_sma_period: int = 20


@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    
    indicator_name: str
    signal_type: str  # BUY, SELL, NEUTRAL
    strength: float   # 0.0 to 1.0
    value: float
    timestamp: datetime
    timeframe: str
    description: str
    confidence: float = 0.5


@dataclass
class PatternResult:
    """Pattern recognition result"""
    
    pattern_name: str
    pattern_type: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float
    start_time: datetime
    end_time: datetime
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None


class TechnicalIndicators:
    """Comprehensive technical indicators library"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def simple_moving_average(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    def exponential_moving_average(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    def relative_strength_index(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                             k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = self.exponential_moving_average(data, fast)
        ema_slow = self.exponential_moving_average(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.simple_moving_average(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def average_true_range(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def volume_weighted_average_price(self, high: pd.Series, low: pd.Series, 
                                     close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate VWAP"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    def commodity_channel_index(self, high: pd.Series, low: pd.Series, 
                               close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate CCI"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci


class PatternRecognition:
    """Advanced pattern recognition engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_double_top(self, data: pd.DataFrame, lookback: int = 50) -> List[PatternResult]:
        """Detect Double Top pattern"""
        patterns = []
        
        if len(data) < lookback * 2:
            return patterns
        
        highs = data['high'].rolling(window=5).max()
        
        for i in range(lookback, len(data) - lookback):
            # Find first peak
            if highs.iloc[i] == data['high'].iloc[i-2:i+3].max():
                first_peak = i
                
                # Look for second peak
                for j in range(i + 20, min(i + lookback, len(data))):
                    if highs.iloc[j] == data['high'].iloc[j-2:j+3].max():
                        # Check if peaks are similar height
                        if abs(data['high'].iloc[i] - data['high'].iloc[j]) / data['high'].iloc[i] < 0.02:
                            pattern = PatternResult(
                                pattern_name="Double Top",
                                pattern_type="BEARISH",
                                confidence=0.7,
                                start_time=data.index[i],
                                end_time=data.index[j],
                                target_price=data['low'].iloc[i:j].min() * 0.98,
                                stop_loss=data['high'].iloc[j] * 1.02
                            )
                            patterns.append(pattern)
                            break
        
        return patterns
    
    def detect_head_and_shoulders(self, data: pd.DataFrame, lookback: int = 50) -> List[PatternResult]:
        """Detect Head and Shoulders pattern"""
        patterns = []
        
        if len(data) < lookback:
            return patterns
        
        highs = data['high'].rolling(window=3).max()
        
        for i in range(20, len(data) - 20):
            # Look for three peaks
            peaks = []
            for j in range(i-15, i+16, 5):
                if j >= 0 and j < len(data) and highs.iloc[j] == data['high'].iloc[j-2:j+3].max():
                    peaks.append((j, data['high'].iloc[j]))
            
            if len(peaks) >= 3:
                # Sort by height
                peaks.sort(key=lambda x: x[1], reverse=True)
                
                # Check H&S pattern
                head = peaks[0]
                left_shoulder = None
                right_shoulder = None
                
                for peak in peaks[1:]:
                    if peak[0] < head[0] and left_shoulder is None:
                        left_shoulder = peak
                    elif peak[0] > head[0] and right_shoulder is None:
                        right_shoulder = peak
                
                if left_shoulder and right_shoulder:
                    # Validate pattern
                    if (abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05 and
                        head[1] > left_shoulder[1] * 1.05):
                        
                        pattern = PatternResult(
                            pattern_name="Head and Shoulders",
                            pattern_type="BEARISH",
                            confidence=0.8,
                            start_time=data.index[left_shoulder[0]],
                            end_time=data.index[right_shoulder[0]],
                            target_price=min(left_shoulder[1], right_shoulder[1]) * 0.95,
                            stop_loss=head[1] * 1.02
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def detect_support_resistance(self, data: pd.DataFrame, min_touches: int = 3) -> Dict[str, List[float]]:
        """Detect support and resistance levels"""
        
        highs = data['high'].values
        lows = data['low'].values
        
        # Find potential resistance levels
        resistance_levels = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
        
        # Find potential support levels
        support_levels = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(lows[i])
        
        # Group similar levels
        def group_levels(levels, threshold=0.005):
            if not levels:
                return []
            
            levels.sort()
            grouped = []
            current_group = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_group[-1]) / current_group[-1] < threshold:
                    current_group.append(level)
                else:
                    if len(current_group) >= min_touches:
                        grouped.append(np.mean(current_group))
                    current_group = [level]
            
            if len(current_group) >= min_touches:
                grouped.append(np.mean(current_group))
            
            return grouped
        
        return {
            'resistance': group_levels(resistance_levels),
            'support': group_levels(support_levels)
        }


class TechnicalAnalyzer:
    """Main technical analysis engine"""
    
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
        self.indicators = TechnicalIndicators()
        self.pattern_recognition = PatternRecognition()
        self.logger = logging.getLogger(__name__)
        
        logger.info("Technical Analyzer initialized")
    
    def analyze_market_data(self, data: pd.DataFrame, timeframe: str = "1h") -> Dict[str, Any]:
        """Comprehensive market analysis"""
        
        if data.empty or len(data) < 50:
            raise ValueError("Insufficient data for analysis")
        
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        results = {
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'indicators': {},
            'signals': [],
            'patterns': [],
            'support_resistance': {},
            'trend_analysis': {},
            'summary': {}
        }
        
        try:
            # Calculate indicators
            results['indicators'] = self._calculate_all_indicators(data)
            
            # Generate signals
            results['signals'] = self._generate_signals(data, results['indicators'], timeframe)
            
            # Pattern recognition
            results['patterns'] = self._detect_patterns(data)
            
            # Support/Resistance
            results['support_resistance'] = self.pattern_recognition.detect_support_resistance(data)
            
            # Trend analysis
            results['trend_analysis'] = self._analyze_trend(data, results['indicators'])
            
            # Generate summary
            results['summary'] = self._generate_analysis_summary(results)
            
            logger.info(f"Technical analysis completed for {timeframe}")
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            raise
        
        return results
    
    def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        
        indicators = {}
        
        try:
            # Moving Averages
            indicators['sma'] = {}
            for period in self.config.sma_periods:
                indicators['sma'][f'sma_{period}'] = self.indicators.simple_moving_average(data['close'], period)
            
            indicators['ema'] = {}
            for period in self.config.ema_periods:
                indicators['ema'][f'ema_{period}'] = self.indicators.exponential_moving_average(data['close'], period)
            
            # Oscillators
            indicators['rsi'] = self.indicators.relative_strength_index(data['close'], self.config.rsi_period)
            
            stoch_k, stoch_d = self.indicators.stochastic_oscillator(
                data['high'], data['low'], data['close'],
                self.config.stoch_k_period, self.config.stoch_d_period
            )
            indicators['stochastic'] = {'%K': stoch_k, '%D': stoch_d}
            
            macd_line, signal_line, histogram = self.indicators.macd(
                data['close'], self.config.macd_fast, 
                self.config.macd_slow, self.config.macd_signal
            )
            indicators['macd'] = {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(
                data['close'], self.config.bb_period, self.config.bb_std
            )
            indicators['bollinger_bands'] = {
                'upper': bb_upper,
                'middle': bb_middle,
                'lower': bb_lower
            }
            
            # ATR
            indicators['atr'] = self.indicators.average_true_range(
                data['high'], data['low'], data['close'], self.config.atr_period
            )
            
            # VWAP
            indicators['vwap'] = self.indicators.volume_weighted_average_price(
                data['high'], data['low'], data['close'], data['volume']
            )
            
            # CCI
            indicators['cci'] = self.indicators.commodity_channel_index(
                data['high'], data['low'], data['close']
            )
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
        
        return indicators
    
    def _generate_signals(self, data: pd.DataFrame, indicators: Dict, timeframe: str) -> List[TechnicalSignal]:
        """Generate trading signals from indicators"""
        
        signals = []
        current_time = data.index[-1] if hasattr(data.index[-1], 'to_pydatetime') else datetime.now()
        
        try:
            # RSI signals
            if 'rsi' in indicators:
                rsi_current = indicators['rsi'].iloc[-1]
                if rsi_current < 30:
                    signals.append(TechnicalSignal(
                        indicator_name="RSI",
                        signal_type="BUY",
                        strength=min((30 - rsi_current) / 30, 1.0),
                        value=rsi_current,
                        timestamp=current_time,
                        timeframe=timeframe,
                        description=f"RSI oversold at {rsi_current:.2f}",
                        confidence=0.7
                    ))
                elif rsi_current > 70:
                    signals.append(TechnicalSignal(
                        indicator_name="RSI",
                        signal_type="SELL",
                        strength=min((rsi_current - 70) / 30, 1.0),
                        value=rsi_current,
                        timestamp=current_time,
                        timeframe=timeframe,
                        description=f"RSI overbought at {rsi_current:.2f}",
                        confidence=0.7
                    ))
            
            # MACD signals
            if 'macd' in indicators:
                macd_current = indicators['macd']['macd'].iloc[-1]
                signal_current = indicators['macd']['signal'].iloc[-1]
                macd_prev = indicators['macd']['macd'].iloc[-2]
                signal_prev = indicators['macd']['signal'].iloc[-2]
                
                # MACD crossover
                if macd_prev <= signal_prev and macd_current > signal_current:
                    signals.append(TechnicalSignal(
                        indicator_name="MACD",
                        signal_type="BUY",
                        strength=0.8,
                        value=macd_current - signal_current,
                        timestamp=current_time,
                        timeframe=timeframe,
                        description="MACD bullish crossover",
                        confidence=0.75
                    ))
                elif macd_prev >= signal_prev and macd_current < signal_current:
                    signals.append(TechnicalSignal(
                        indicator_name="MACD",
                        signal_type="SELL",
                        strength=0.8,
                        value=signal_current - macd_current,
                        timestamp=current_time,
                        timeframe=timeframe,
                        description="MACD bearish crossover",
                        confidence=0.75
                    ))
            
            # Bollinger Bands signals
            if 'bollinger_bands' in indicators:
                price_current = data['close'].iloc[-1]
                bb_upper = indicators['bollinger_bands']['upper'].iloc[-1]
                bb_lower = indicators['bollinger_bands']['lower'].iloc[-1]
                bb_middle = indicators['bollinger_bands']['middle'].iloc[-1]
                
                if price_current <= bb_lower:
                    signals.append(TechnicalSignal(
                        indicator_name="Bollinger_Bands",
                        signal_type="BUY",
                        strength=min((bb_lower - price_current) / bb_lower, 1.0),
                        value=price_current,
                        timestamp=current_time,
                        timeframe=timeframe,
                        description="Price at lower Bollinger Band",
                        confidence=0.6
                    ))
                elif price_current >= bb_upper:
                    signals.append(TechnicalSignal(
                        indicator_name="Bollinger_Bands",
                        signal_type="SELL",
                        strength=min((price_current - bb_upper) / bb_upper, 1.0),
                        value=price_current,
                        timestamp=current_time,
                        timeframe=timeframe,
                        description="Price at upper Bollinger Band",
                        confidence=0.6
                    ))
            
            # Moving Average signals
            if 'sma' in indicators and 'sma_20' in indicators['sma'] and 'sma_50' in indicators['sma']:
                sma20 = indicators['sma']['sma_20'].iloc[-1]
                sma50 = indicators['sma']['sma_50'].iloc[-1]
                sma20_prev = indicators['sma']['sma_20'].iloc[-2]
                sma50_prev = indicators['sma']['sma_50'].iloc[-2]
                
                # Golden Cross
                if sma20_prev <= sma50_prev and sma20 > sma50:
                    signals.append(TechnicalSignal(
                        indicator_name="Moving_Average",
                        signal_type="BUY",
                        strength=0.9,
                        value=sma20 - sma50,
                        timestamp=current_time,
                        timeframe=timeframe,
                        description="Golden Cross (SMA20 > SMA50)",
                        confidence=0.8
                    ))
                # Death Cross
                elif sma20_prev >= sma50_prev and sma20 < sma50:
                    signals.append(TechnicalSignal(
                        indicator_name="Moving_Average",
                        signal_type="SELL",
                        strength=0.9,
                        value=sma50 - sma20,
                        timestamp=current_time,
                        timeframe=timeframe,
                        description="Death Cross (SMA20 < SMA50)",
                        confidence=0.8
                    ))
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def _detect_patterns(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect chart patterns"""
        
        patterns = []
        
        try:
            # Double Top
            double_tops = self.pattern_recognition.detect_double_top(data)
            patterns.extend(double_tops)
            
            # Head and Shoulders
            head_shoulders = self.pattern_recognition.detect_head_and_shoulders(data)
            patterns.extend(head_shoulders)
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Analyze market trend"""
        
        trend_analysis = {
            'short_term': 'NEUTRAL',
            'medium_term': 'NEUTRAL',
            'long_term': 'NEUTRAL',
            'strength': 0.5,
            'description': ''
        }
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Short-term trend (SMA 20)
            if 'sma' in indicators and 'sma_20' in indicators['sma']:
                sma20 = indicators['sma']['sma_20'].iloc[-1]
                if current_price > sma20 * 1.01:
                    trend_analysis['short_term'] = 'BULLISH'
                elif current_price < sma20 * 0.99:
                    trend_analysis['short_term'] = 'BEARISH'
            
            # Medium-term trend (SMA 50)
            if 'sma' in indicators and 'sma_50' in indicators['sma']:
                sma50 = indicators['sma']['sma_50'].iloc[-1]
                if current_price > sma50 * 1.02:
                    trend_analysis['medium_term'] = 'BULLISH'
                elif current_price < sma50 * 0.98:
                    trend_analysis['medium_term'] = 'BEARISH'
            
            # Long-term trend (SMA 200)
            if 'sma' in indicators and 'sma_200' in indicators['sma']:
                sma200 = indicators['sma']['sma_200'].iloc[-1]
                if current_price > sma200 * 1.05:
                    trend_analysis['long_term'] = 'BULLISH'
                elif current_price < sma200 * 0.95:
                    trend_analysis['long_term'] = 'BEARISH'
            
            # Calculate overall trend strength
            bullish_count = sum(1 for trend in [trend_analysis['short_term'], 
                                               trend_analysis['medium_term'], 
                                               trend_analysis['long_term']] if trend == 'BULLISH')
            bearish_count = sum(1 for trend in [trend_analysis['short_term'], 
                                               trend_analysis['medium_term'], 
                                               trend_analysis['long_term']] if trend == 'BEARISH')
            
            if bullish_count > bearish_count:
                trend_analysis['strength'] = 0.5 + (bullish_count / 6)
                trend_analysis['description'] = f"Bullish trend ({bullish_count}/3 timeframes)"
            elif bearish_count > bullish_count:
                trend_analysis['strength'] = 0.5 - (bearish_count / 6)
                trend_analysis['description'] = f"Bearish trend ({bearish_count}/3 timeframes)"
            else:
                trend_analysis['description'] = "Mixed/Neutral trend"
        
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
        
        return trend_analysis
    
    def _generate_analysis_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate analysis summary"""
        
        summary = {
            'total_signals': len(results['signals']),
            'buy_signals': len([s for s in results['signals'] if s.signal_type == 'BUY']),
            'sell_signals': len([s for s in results['signals'] if s.signal_type == 'SELL']),
            'patterns_detected': len(results['patterns']),
            'trend_direction': results['trend_analysis'].get('description', 'Unknown'),
            'market_sentiment': 'NEUTRAL',
            'confidence_score': 0.5,
            'recommendation': 'HOLD'
        }
        
        try:
            # Calculate market sentiment
            buy_strength = sum(s.strength for s in results['signals'] if s.signal_type == 'BUY')
            sell_strength = sum(s.strength for s in results['signals'] if s.signal_type == 'SELL')
            
            if buy_strength > sell_strength * 1.2:
                summary['market_sentiment'] = 'BULLISH'
                summary['recommendation'] = 'BUY'
            elif sell_strength > buy_strength * 1.2:
                summary['market_sentiment'] = 'BEARISH'
                summary['recommendation'] = 'SELL'
            
            # Calculate confidence score
            if results['signals']:
                avg_confidence = np.mean([s.confidence for s in results['signals']])
                summary['confidence_score'] = avg_confidence
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        
        return summary


def create_technical_analyzer(custom_config: Dict = None) -> TechnicalAnalyzer:
    """Factory function to create technical analyzer"""
    
    config = IndicatorConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return TechnicalAnalyzer(config)


if __name__ == "__main__":
    # Example usage
    analyzer = create_technical_analyzer()
    
    # Generate sample data for testing
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    np.random.seed(42)
    
    # Simulate XAUUSD price data
    base_price = 2000
    returns = np.random.normal(0, 0.01, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    sample_data = pd.DataFrame({
        'open': [p * 0.999 for p in prices],
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Perform analysis
    results = analyzer.analyze_market_data(sample_data, '1h')
    
    print("Technical Analysis Results:")
    print(f"Total signals: {results['summary']['total_signals']}")
    print(f"Market sentiment: {results['summary']['market_sentiment']}")
    print(f"Recommendation: {results['summary']['recommendation']}")
    print(f"Confidence: {results['summary']['confidence_score']:.2f}") 