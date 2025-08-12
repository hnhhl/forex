"""
Custom Technical Indicators System
Ultimate XAU Super System V4.0 - Day 23 Implementation

Advanced custom indicator framework:
- User-defined indicator creation
- High-performance calculation engine
- Multi-timeframe analysis
- Advanced visualization integration
- Real-time indicator streaming
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import inspect
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for custom technical indicators"""
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel_processing: bool = True
    
    # Calculation settings
    precision: int = 6
    fill_method: str = 'ffill'  # forward fill, backward fill, interpolate
    min_periods: int = 1
    
    # Visualization settings
    enable_plotting: bool = True
    plot_style: str = 'line'  # line, bar, area, scatter
    color_scheme: str = 'default'
    
    # Real-time settings
    enable_streaming: bool = True
    update_frequency: int = 1  # seconds
    buffer_size: int = 100


@dataclass
class IndicatorResult:
    """Result container for indicator calculations"""
    
    name: str
    values: pd.Series
    parameters: Dict[str, Any]
    calculation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Signal information
    signals: Optional[pd.Series] = None
    levels: Optional[Dict[str, float]] = None
    
    # Performance metrics
    accuracy: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


class BaseIndicator(ABC):
    """Base class for all custom technical indicators"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.result_cache = {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate indicator values"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        required_columns = ['open', 'high', 'low', 'close']
        return all(col in data.columns for col in required_columns)
    
    def generate_signals(self, values: pd.Series) -> pd.Series:
        """Generate trading signals from indicator values"""
        signals = pd.Series(index=values.index, dtype=float)
        signals.fillna(0, inplace=True)
        return signals


class MovingAverageCustom(BaseIndicator):
    """Custom Moving Average with multiple calculation methods"""
    
    def __init__(self, period: int = 20, method: str = 'sma', **kwargs):
        super().__init__(f"MA_{method.upper()}_{period}", {'period': period, 'method': method, **kwargs})
        self.period = period
        self.method = method
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate custom moving average"""
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        start_time = datetime.now()
        
        try:
            prices = data['close']
            
            if self.method == 'sma':
                values = prices.rolling(window=self.period).mean()
            elif self.method == 'ema':
                values = prices.ewm(span=self.period).mean()
            elif self.method == 'wma':
                weights = np.arange(1, self.period + 1)
                values = prices.rolling(window=self.period).apply(
                    lambda x: np.dot(x, weights) / weights.sum(), raw=True
                )
            elif self.method == 'hull':
                # Hull Moving Average
                half_period = int(self.period / 2)
                sqrt_period = int(np.sqrt(self.period))
                wma1 = prices.rolling(window=half_period).apply(
                    lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True
                )
                wma2 = prices.rolling(window=self.period).apply(
                    lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True
                )
                hull_data = 2 * wma1 - wma2
                values = hull_data.rolling(window=sqrt_period).apply(
                    lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True
                )
            elif self.method == 'adaptive':
                # Adaptive Moving Average using volatility
                volatility = prices.rolling(window=self.period).std()
                alpha = 2 / (self.period + 1)
                adaptive_alpha = alpha * (volatility / volatility.rolling(window=self.period).mean())
                values = prices.ewm(alpha=adaptive_alpha).mean()
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Generate signals
            signals = self.generate_signals(values)
            if len(values) > 1:
                signals[prices > values] = 1  # Buy signal
                signals[prices < values] = -1  # Sell signal
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            return IndicatorResult(
                name=self.name,
                values=values,
                parameters=self.parameters,
                calculation_time=calculation_time,
                signals=signals,
                metadata={
                    'method': self.method,
                    'period': self.period,
                    'data_points': len(data)
                }
            )
        
        except Exception as e:
            logger.error(f"Error calculating {self.name}: {e}")
            raise


class RSICustom(BaseIndicator):
    """Custom RSI with enhanced features"""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30, **kwargs):
        super().__init__(f"RSI_{period}", {'period': period, 'overbought': overbought, 'oversold': oversold, **kwargs})
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate enhanced RSI"""
        start_time = datetime.now()
        
        try:
            prices = data['close']
            delta = prices.diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.period).mean()
            avg_loss = loss.rolling(window=self.period).mean()
            
            # Calculate RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Generate enhanced signals
            signals = pd.Series(index=rsi.index, dtype=float).fillna(0)
            
            # Overbought/Oversold signals
            signals[rsi > self.overbought] = -1  # Sell signal
            signals[rsi < self.oversold] = 1     # Buy signal
            
            # Divergence detection (simplified)
            price_peaks = prices.rolling(window=5).max() == prices
            rsi_peaks = rsi.rolling(window=5).max() == rsi
            
            # Bullish divergence: price makes lower lows, RSI makes higher lows
            for i in range(10, len(prices)):
                if (prices.iloc[i] < prices.iloc[i-5] and 
                    rsi.iloc[i] > rsi.iloc[i-5] and 
                    rsi.iloc[i] < self.oversold):
                    signals.iloc[i] = 2  # Strong buy signal
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            return IndicatorResult(
                name=self.name,
                values=rsi,
                parameters=self.parameters,
                calculation_time=calculation_time,
                signals=signals,
                levels={'overbought': self.overbought, 'oversold': self.oversold},
                metadata={
                    'period': self.period,
                    'avg_rsi': rsi.mean(),
                    'current_rsi': rsi.iloc[-1] if len(rsi) > 0 else None
                }
            )
        
        except Exception as e:
            logger.error(f"Error calculating {self.name}: {e}")
            raise


class MACDCustom(BaseIndicator):
    """Custom MACD with enhanced signal generation"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs):
        super().__init__(f"MACD_{fast}_{slow}_{signal}", {'fast': fast, 'slow': slow, 'signal': signal, **kwargs})
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate enhanced MACD"""
        start_time = datetime.now()
        
        try:
            prices = data['close']
            
            # Calculate MACD components
            ema_fast = prices.ewm(span=self.fast).mean()
            ema_slow = prices.ewm(span=self.slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period).mean()
            histogram = macd_line - signal_line
            
            # Combine into single result
            macd_result = pd.DataFrame({
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            })
            
            # Generate enhanced signals
            signals = pd.Series(index=macd_line.index, dtype=float).fillna(0)
            
            # Basic crossover signals
            signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1   # Buy
            signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1  # Sell
            
            # Zero line crossover
            signals[(macd_line > 0) & (macd_line.shift(1) <= 0)] = 2    # Strong buy
            signals[(macd_line < 0) & (macd_line.shift(1) >= 0)] = -2   # Strong sell
            
            # Histogram momentum
            hist_increasing = histogram > histogram.shift(1)
            hist_decreasing = histogram < histogram.shift(1)
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            return IndicatorResult(
                name=self.name,
                values=macd_line,  # Primary series
                parameters=self.parameters,
                calculation_time=calculation_time,
                signals=signals,
                metadata={
                    'macd': macd_line,
                    'signal': signal_line,
                    'histogram': histogram,
                    'fast_period': self.fast,
                    'slow_period': self.slow,
                    'signal_period': self.signal_period
                }
            )
        
        except Exception as e:
            logger.error(f"Error calculating {self.name}: {e}")
            raise


class BollingerBandsCustom(BaseIndicator):
    """Custom Bollinger Bands with adaptive features"""
    
    def __init__(self, period: int = 20, std_dev: float = 2, adaptive: bool = False, **kwargs):
        super().__init__(f"BB_{period}_{std_dev}", {'period': period, 'std_dev': std_dev, 'adaptive': adaptive, **kwargs})
        self.period = period
        self.std_dev = std_dev
        self.adaptive = adaptive
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate adaptive Bollinger Bands"""
        start_time = datetime.now()
        
        try:
            prices = data['close']
            
            # Calculate middle band (SMA)
            middle_band = prices.rolling(window=self.period).mean()
            
            # Calculate standard deviation
            if self.adaptive:
                # Adaptive standard deviation based on volatility
                volatility = prices.rolling(window=self.period).std()
                adaptive_std = self.std_dev * (volatility / volatility.rolling(window=self.period).mean())
                std_values = adaptive_std * prices.rolling(window=self.period).std()
            else:
                std_values = self.std_dev * prices.rolling(window=self.period).std()
            
            # Calculate bands
            upper_band = middle_band + std_values
            lower_band = middle_band - std_values
            
            # Calculate %B and Bandwidth
            percent_b = (prices - lower_band) / (upper_band - lower_band)
            bandwidth = (upper_band - lower_band) / middle_band
            
            # Generate signals
            signals = pd.Series(index=prices.index, dtype=float).fillna(0)
            
            # Band touch signals
            signals[prices <= lower_band] = 1   # Buy at lower band
            signals[prices >= upper_band] = -1  # Sell at upper band
            
            # Squeeze detection (low volatility)
            squeeze_threshold = bandwidth.rolling(window=self.period).quantile(0.2)
            squeeze_condition = bandwidth < squeeze_threshold
            
            # Breakout signals after squeeze
            for i in range(1, len(prices)):
                if squeeze_condition.iloc[i-1] and not squeeze_condition.iloc[i]:
                    if prices.iloc[i] > middle_band.iloc[i]:
                        signals.iloc[i] = 2   # Strong buy breakout
                    else:
                        signals.iloc[i] = -2  # Strong sell breakout
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            return IndicatorResult(
                name=self.name,
                values=middle_band,  # Primary series
                parameters=self.parameters,
                calculation_time=calculation_time,
                signals=signals,
                metadata={
                    'upper_band': upper_band,
                    'middle_band': middle_band,
                    'lower_band': lower_band,
                    'percent_b': percent_b,
                    'bandwidth': bandwidth,
                    'squeeze_condition': squeeze_condition,
                    'current_bandwidth': bandwidth.iloc[-1] if len(bandwidth) > 0 else None
                }
            )
        
        except Exception as e:
            logger.error(f"Error calculating {self.name}: {e}")
            raise


class VolumeProfileCustom(BaseIndicator):
    """Custom Volume Profile indicator"""
    
    def __init__(self, bins: int = 20, period: int = 100, **kwargs):
        super().__init__(f"VolumeProfile_{bins}_{period}", {'bins': bins, 'period': period, **kwargs})
        self.bins = bins
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate Volume Profile"""
        start_time = datetime.now()
        
        try:
            if 'volume' not in data.columns:
                raise ValueError("Volume data required for Volume Profile")
            
            prices = data['close']
            volumes = data['volume']
            highs = data['high']
            lows = data['low']
            
            # Calculate for rolling periods
            vp_results = []
            
            for i in range(self.period, len(data)):
                window_data = data.iloc[i-self.period:i]
                
                # Create price bins
                price_min = window_data['low'].min()
                price_max = window_data['high'].max()
                price_bins = np.linspace(price_min, price_max, self.bins + 1)
                
                # Calculate volume at each price level
                volume_profile = np.zeros(self.bins)
                
                for j, row in window_data.iterrows():
                    # Distribute volume across price range within the bar
                    bar_range = row['high'] - row['low']
                    if bar_range > 0:
                        for k in range(self.bins):
                            bin_low = price_bins[k]
                            bin_high = price_bins[k + 1]
                            
                            # Calculate overlap between bar range and bin range
                            overlap_low = max(bin_low, row['low'])
                            overlap_high = min(bin_high, row['high'])
                            
                            if overlap_high > overlap_low:
                                overlap_ratio = (overlap_high - overlap_low) / bar_range
                                volume_profile[k] += row['volume'] * overlap_ratio
                    else:
                        # If no range, assign all volume to the appropriate bin
                        bin_idx = np.digitize(row['close'], price_bins) - 1
                        bin_idx = np.clip(bin_idx, 0, self.bins - 1)
                        volume_profile[bin_idx] += row['volume']
                
                # Find POC (Point of Control - highest volume)
                poc_idx = np.argmax(volume_profile)
                poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
                
                # Find Value Area (70% of volume)
                total_volume = volume_profile.sum()
                sorted_indices = np.argsort(volume_profile)[::-1]
                cumulative_volume = 0
                value_area_indices = []
                
                for idx in sorted_indices:
                    cumulative_volume += volume_profile[idx]
                    value_area_indices.append(idx)
                    if cumulative_volume >= 0.7 * total_volume:
                        break
                
                va_high = price_bins[max(value_area_indices) + 1]
                va_low = price_bins[min(value_area_indices)]
                
                vp_results.append({
                    'poc': poc_price,
                    'va_high': va_high,
                    'va_low': va_low,
                    'profile': volume_profile,
                    'bins': price_bins
                })
            
            # Create result series
            poc_series = pd.Series(index=data.index, dtype=float)
            va_high_series = pd.Series(index=data.index, dtype=float)
            va_low_series = pd.Series(index=data.index, dtype=float)
            
            for i, result in enumerate(vp_results):
                idx = data.index[i + self.period]
                poc_series[idx] = result['poc']
                va_high_series[idx] = result['va_high']
                va_low_series[idx] = result['va_low']
            
            # Generate signals based on price relation to POC and Value Area
            signals = pd.Series(index=data.index, dtype=float).fillna(0)
            
            for i in range(len(poc_series)):
                if pd.notna(poc_series.iloc[i]):
                    current_price = prices.iloc[i]
                    poc = poc_series.iloc[i]
                    va_high = va_high_series.iloc[i]
                    va_low = va_low_series.iloc[i]
                    
                    # Generate signals
                    if current_price < va_low:
                        signals.iloc[i] = 1   # Buy below value area
                    elif current_price > va_high:
                        signals.iloc[i] = -1  # Sell above value area
                    elif abs(current_price - poc) < (va_high - va_low) * 0.05:
                        signals.iloc[i] = 0   # Neutral at POC
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            return IndicatorResult(
                name=self.name,
                values=poc_series,
                parameters=self.parameters,
                calculation_time=calculation_time,
                signals=signals,
                metadata={
                    'poc': poc_series,
                    'va_high': va_high_series,
                    'va_low': va_low_series,
                    'bins': self.bins,
                    'period': self.period,
                    'latest_profile': vp_results[-1] if vp_results else None
                }
            )
        
        except Exception as e:
            logger.error(f"Error calculating {self.name}: {e}")
            raise


class CustomIndicatorFactory:
    """Factory for creating and managing custom indicators"""
    
    def __init__(self):
        self.indicators = {}
        self.indicator_classes = {
            'ma': MovingAverageCustom,
            'rsi': RSICustom,
            'macd': MACDCustom,
            'bb': BollingerBandsCustom,
            'vp': VolumeProfileCustom
        }
        self.logger = logging.getLogger(__name__)
    
    def register_indicator(self, name: str, indicator_class: type):
        """Register a new indicator class"""
        if not issubclass(indicator_class, BaseIndicator):
            raise ValueError("Indicator class must inherit from BaseIndicator")
        
        self.indicator_classes[name] = indicator_class
        logger.info(f"Registered indicator: {name}")
    
    def create_indicator(self, indicator_type: str, **kwargs) -> BaseIndicator:
        """Create an indicator instance"""
        if indicator_type not in self.indicator_classes:
            raise ValueError(f"Unknown indicator type: {indicator_type}")
        
        indicator_class = self.indicator_classes[indicator_type]
        return indicator_class(**kwargs)
    
    def list_indicators(self) -> List[str]:
        """List available indicator types"""
        return list(self.indicator_classes.keys())


class MultiTimeframeAnalyzer:
    """Multi-timeframe analysis engine"""
    
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
        self.factory = CustomIndicatorFactory()
        self.timeframes = ['5T', '15T', '1H', '4H', '1D']  # 5min, 15min, 1h, 4h, 1d
        self.logger = logging.getLogger(__name__)
    
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to different timeframe"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Resample OHLCV data
        resampled = data.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def analyze_timeframes(self, data: pd.DataFrame, indicator_config: Dict[str, Any]) -> Dict[str, Dict[str, IndicatorResult]]:
        """Analyze indicator across multiple timeframes"""
        results = {}
        
        for timeframe in self.timeframes:
            try:
                # Resample data
                tf_data = self.resample_data(data, timeframe)
                
                if len(tf_data) < 20:  # Skip if insufficient data
                    continue
                
                # Calculate indicators for this timeframe
                tf_results = {}
                
                for indicator_type, params in indicator_config.items():
                    try:
                        indicator = self.factory.create_indicator(indicator_type, **params)
                        result = indicator.calculate(tf_data)
                        tf_results[indicator.name] = result
                    except Exception as e:
                        logger.warning(f"Failed to calculate {indicator_type} for {timeframe}: {e}")
                
                results[timeframe] = tf_results
                
            except Exception as e:
                logger.error(f"Error analyzing timeframe {timeframe}: {e}")
        
        return results
    
    def generate_confluence_signals(self, mtf_results: Dict[str, Dict[str, IndicatorResult]]) -> pd.Series:
        """Generate confluence signals across timeframes"""
        if not mtf_results:
            return pd.Series(dtype=float)
        
        # Get the base timeframe (highest frequency)
        base_tf = min(mtf_results.keys())
        base_results = mtf_results[base_tf]
        
        if not base_results:
            return pd.Series(dtype=float)
        
        # Use the first indicator's index as reference
        first_indicator = next(iter(base_results.values()))
        confluence_signals = pd.Series(index=first_indicator.values.index, dtype=float).fillna(0)
        
        # Count signals across timeframes and indicators
        for timestamp in confluence_signals.index:
            signal_sum = 0
            signal_count = 0
            
            for timeframe, tf_results in mtf_results.items():
                for indicator_name, result in tf_results.items():
                    if result.signals is not None and timestamp in result.signals.index:
                        signal_value = result.signals[timestamp]
                        if not pd.isna(signal_value) and signal_value != 0:
                            signal_sum += signal_value
                            signal_count += 1
            
            # Generate confluence signal
            if signal_count >= 2:  # Require at least 2 signals
                avg_signal = signal_sum / signal_count
                if abs(avg_signal) >= 0.5:  # Minimum signal strength
                    confluence_signals[timestamp] = avg_signal
        
        return confluence_signals


class CustomTechnicalIndicators:
    """Main custom technical indicators system"""
    
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
        self.factory = CustomIndicatorFactory()
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.config)
        self.indicator_cache = {}
        self.logger = logging.getLogger(__name__)
        
        logger.info("Custom Technical Indicators initialized")
    
    def calculate_indicator(self, data: pd.DataFrame, indicator_type: str, **kwargs) -> IndicatorResult:
        """Calculate a single indicator"""
        indicator = self.factory.create_indicator(indicator_type, **kwargs)
        return indicator.calculate(data)
    
    def calculate_multiple_indicators(self, data: pd.DataFrame, 
                                    indicator_configs: Dict[str, Dict[str, Any]]) -> Dict[str, IndicatorResult]:
        """Calculate multiple indicators"""
        results = {}
        
        for name, config in indicator_configs.items():
            try:
                indicator_type = config.pop('type', name)
                result = self.calculate_indicator(data, indicator_type, **config)
                results[name] = result
            except Exception as e:
                logger.error(f"Error calculating indicator {name}: {e}")
        
        return results
    
    def run_mtf_analysis(self, data: pd.DataFrame, 
                        indicator_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive multi-timeframe analysis"""
        try:
            # Multi-timeframe analysis
            mtf_results = self.mtf_analyzer.analyze_timeframes(data, indicator_configs)
            
            # Generate confluence signals
            confluence_signals = self.mtf_analyzer.generate_confluence_signals(mtf_results)
            
            # Calculate overall statistics
            total_indicators = sum(len(tf_results) for tf_results in mtf_results.values())
            successful_calcs = sum(
                sum(1 for result in tf_results.values() if result.values is not None)
                for tf_results in mtf_results.values()
            )
            
            success_rate = successful_calcs / total_indicators if total_indicators > 0 else 0
            
            return {
                'timestamp': datetime.now(),
                'mtf_results': mtf_results,
                'confluence_signals': confluence_signals,
                'timeframes_analyzed': list(mtf_results.keys()),
                'total_indicators': total_indicators,
                'successful_calculations': successful_calcs,
                'success_rate': success_rate,
                'signal_strength': confluence_signals.abs().mean() if len(confluence_signals) > 0 else 0
            }
        
        except Exception as e:
            logger.error(f"Error in MTF analysis: {e}")
            raise
    
    def create_custom_indicator(self, name: str, calculation_func: Callable, **default_params) -> type:
        """Create a custom indicator class from a function"""
        
        class DynamicIndicator(BaseIndicator):
            def __init__(self, **kwargs):
                params = {**default_params, **kwargs}
                super().__init__(name, params)
                self.calc_func = calculation_func
            
            def calculate(self, data: pd.DataFrame) -> IndicatorResult:
                start_time = datetime.now()
                
                try:
                    # Call the custom calculation function
                    values = self.calc_func(data, **self.parameters)
                    
                    if not isinstance(values, pd.Series):
                        values = pd.Series(values, index=data.index)
                    
                    # Generate basic signals
                    signals = self.generate_signals(values)
                    
                    calculation_time = (datetime.now() - start_time).total_seconds()
                    
                    return IndicatorResult(
                        name=self.name,
                        values=values,
                        parameters=self.parameters,
                        calculation_time=calculation_time,
                        signals=signals,
                        metadata={'custom_indicator': True}
                    )
                
                except Exception as e:
                    logger.error(f"Error in custom indicator {name}: {e}")
                    raise
        
        return DynamicIndicator
    
    def register_custom_indicator(self, name: str, calculation_func: Callable, **default_params):
        """Register a custom indicator"""
        indicator_class = self.create_custom_indicator(name, calculation_func, **default_params)
        self.factory.register_indicator(name.lower(), indicator_class)
        logger.info(f"Registered custom indicator: {name}")


def create_custom_technical_indicators(custom_config: Dict = None) -> CustomTechnicalIndicators:
    """Factory function to create custom technical indicators system"""
    
    config = IndicatorConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return CustomTechnicalIndicators(config)


# Example custom indicator functions
def stochastic_rsi(data: pd.DataFrame, period: int = 14, stoch_period: int = 14, **kwargs) -> pd.Series:
    """Custom Stochastic RSI indicator"""
    # Calculate RSI first
    prices = data['close']
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Apply Stochastic to RSI
    rsi_low = rsi.rolling(window=stoch_period).min()
    rsi_high = rsi.rolling(window=stoch_period).max()
    stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
    
    return stoch_rsi


def williams_r(data: pd.DataFrame, period: int = 14, **kwargs) -> pd.Series:
    """Custom Williams %R indicator"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    return williams_r


def commodity_channel_index(data: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
    """Custom Commodity Channel Index"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    tp = (high + low + close) / 3  # Typical Price
    ma_tp = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    
    cci = (tp - ma_tp) / (0.015 * md)
    
    return cci


if __name__ == "__main__":
    # Example usage
    system = create_custom_technical_indicators()
    
    # Register custom indicators
    system.register_custom_indicator('stoch_rsi', stochastic_rsi, period=14, stoch_period=14)
    system.register_custom_indicator('williams_r', williams_r, period=14)
    system.register_custom_indicator('cci', commodity_channel_index, period=20)
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='h')
    np.random.seed(42)
    
    base_price = 2000
    prices = [base_price]
    for i in range(199):
        change = np.random.normal(0.001, 0.01)
        prices.append(prices[-1] * (1 + change))
    
    sample_data = pd.DataFrame({
        'open': [p * 0.999 for p in prices],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Configure indicators
    indicator_configs = {
        'ma': {'type': 'ma', 'period': 20, 'method': 'sma'},
        'rsi': {'type': 'rsi', 'period': 14},
        'macd': {'type': 'macd', 'fast': 12, 'slow': 26, 'signal': 9},
        'bb': {'type': 'bb', 'period': 20, 'std_dev': 2},
        'stoch_rsi': {'type': 'stoch_rsi', 'period': 14, 'stoch_period': 14}
    }
    
    # Run analysis
    results = system.run_mtf_analysis(sample_data, indicator_configs)
    
    print("Custom Technical Indicators Results:")
    print(f"Timeframes analyzed: {results['timeframes_analyzed']}")
    print(f"Total indicators: {results['total_indicators']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Signal strength: {results['signal_strength']:.3f}")