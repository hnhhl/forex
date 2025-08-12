"""
Multi-Timeframe Analysis Enhancement System
Ultimate XAU Super System V4.0 - Day 24 Implementation

Advanced multi-timeframe analysis capabilities:
- Enhanced timeframe synchronization
- Advanced confluence algorithms
- Real-time streaming integration
- Performance optimization for large datasets
- Intelligent signal weighting
- Cross-timeframe correlation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TimeframeType(Enum):
    """Timeframe classification for analysis"""
    SCALPING = "scalping"      # 1m, 5m
    SHORT_TERM = "short_term"  # 15m, 30m
    MEDIUM_TERM = "medium_term" # 1h, 4h
    LONG_TERM = "long_term"    # 1d, 1w
    STRATEGIC = "strategic"    # 1M, 3M


class SignalStrength(Enum):
    """Signal strength classification"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class TimeframeConfig:
    """Configuration for multi-timeframe analysis"""
    
    # Timeframe settings
    timeframes: List[str] = field(default_factory=lambda: ['1T', '5T', '15T', '1H', '4H', '1D'])
    primary_timeframe: str = '1H'
    
    # Synchronization settings
    enable_synchronization: bool = True
    sync_tolerance: int = 2  # minutes
    interpolation_method: str = 'linear'  # linear, cubic, nearest
    
    # Confluence settings
    confluence_threshold: float = 0.6
    min_timeframes_agreement: int = 3
    weight_by_timeframe: bool = True
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    cache_size: int = 1000
    
    # Real-time settings
    enable_streaming: bool = True
    update_frequency: float = 1.0  # seconds
    buffer_size: int = 500
    
    # Advanced settings
    enable_correlation_analysis: bool = True
    correlation_window: int = 50
    enable_adaptive_weights: bool = True


@dataclass
class TimeframeResult:
    """Result container for timeframe analysis"""
    
    timeframe: str
    timeframe_type: TimeframeType
    data: pd.DataFrame
    indicators: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, pd.Series] = field(default_factory=dict)
    
    # Analysis metrics
    signal_strength: float = 0.0
    confidence: float = 0.0
    trend_direction: int = 0  # -1, 0, 1
    volatility: float = 0.0
    
    # Timing information
    calculation_time: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    # Metadata
    data_quality: float = 1.0
    completeness: float = 1.0
    reliability_score: float = 1.0


@dataclass
class ConfluenceSignal:
    """Confluence signal across timeframes"""
    
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'neutral'
    strength: SignalStrength
    confidence: float
    
    # Contributing timeframes
    contributing_timeframes: List[str] = field(default_factory=list)
    timeframe_signals: Dict[str, float] = field(default_factory=dict)
    timeframe_weights: Dict[str, float] = field(default_factory=dict)
    
    # Analysis details
    consensus_score: float = 0.0
    correlation_score: float = 0.0
    momentum_score: float = 0.0
    
    # Risk assessment
    risk_level: str = 'medium'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Metadata
    calculation_method: str = 'weighted_average'
    reliability: float = 1.0


class TimeframeSynchronizer:
    """Advanced timeframe synchronization engine"""
    
    def __init__(self, config: TimeframeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Timeframe hierarchy mapping
        self.timeframe_hierarchy = {
            '1T': (TimeframeType.SCALPING, 1),
            '5T': (TimeframeType.SCALPING, 5),
            '15T': (TimeframeType.SHORT_TERM, 15),
            '30T': (TimeframeType.SHORT_TERM, 30),
            '1H': (TimeframeType.MEDIUM_TERM, 60),
            '4H': (TimeframeType.MEDIUM_TERM, 240),
            '1D': (TimeframeType.LONG_TERM, 1440),
            '1W': (TimeframeType.LONG_TERM, 10080),
            '1M': (TimeframeType.STRATEGIC, 43200)
        }
    
    def synchronize_timeframes(self, mtf_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Synchronize data across multiple timeframes"""
        try:
            if not self.config.enable_synchronization:
                return mtf_data
            
            # Find common time range
            common_start = max(df.index.min() for df in mtf_data.values())
            common_end = min(df.index.max() for df in mtf_data.values())
            
            synchronized_data = {}
            
            for timeframe, data in mtf_data.items():
                # Filter to common time range
                filtered_data = data[common_start:common_end].copy()
                
                # Fill missing values using interpolation
                if filtered_data.isnull().any().any():
                    filtered_data = self._interpolate_missing_data(
                        filtered_data, method=self.config.interpolation_method
                    )
                
                # Align timestamps with tolerance
                if self.config.sync_tolerance > 0:
                    filtered_data = self._align_timestamps(
                        filtered_data, tolerance_minutes=self.config.sync_tolerance
                    )
                
                synchronized_data[timeframe] = filtered_data
            
            logger.info(f"Synchronized {len(synchronized_data)} timeframes")
            return synchronized_data
            
        except Exception as e:
            logger.error(f"Error synchronizing timeframes: {e}")
            return mtf_data
    
    def _interpolate_missing_data(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Interpolate missing data points"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if data[col].isnull().any():
                if method == 'linear':
                    data[col] = data[col].interpolate(method='linear')
                elif method == 'cubic':
                    data[col] = data[col].interpolate(method='cubic')
                elif method == 'nearest':
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                
                # Forward fill any remaining NaN at edges
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _align_timestamps(self, data: pd.DataFrame, tolerance_minutes: int) -> pd.DataFrame:
        """Align timestamps within tolerance"""
        if tolerance_minutes <= 0:
            return data
        
        # Round timestamps to nearest tolerance interval
        tolerance_seconds = tolerance_minutes * 60
        rounded_index = data.index.round(f'{tolerance_seconds}s')
        
        # Group by rounded timestamps and aggregate
        data_rounded = data.copy()
        data_rounded.index = rounded_index
        
        # Aggregate duplicate timestamps
        aggregated = data_rounded.groupby(level=0).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return aggregated


class AdvancedConfluenceAnalyzer:
    """Advanced confluence signal analysis engine"""
    
    def __init__(self, config: TimeframeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Timeframe weights for confluence calculation
        self.default_weights = {
            '1T': 0.05,   # Scalping
            '5T': 0.10,   # Short-term noise
            '15T': 0.15,  # Short-term trend
            '30T': 0.15,  # Medium-term
            '1H': 0.20,   # Primary timeframe
            '4H': 0.20,   # Strong medium-term
            '1D': 0.15    # Long-term trend
        }
    
    def analyze_confluence(self, mtf_results: Dict[str, TimeframeResult]) -> List[ConfluenceSignal]:
        """Analyze confluence signals across timeframes"""
        try:
            confluence_signals = []
            
            if not mtf_results:
                return confluence_signals
            
            # Get common timestamps
            common_timestamps = self._find_common_timestamps(mtf_results)
            
            for timestamp in common_timestamps:
                confluence_signal = self._calculate_confluence_at_timestamp(
                    timestamp, mtf_results
                )
                
                if confluence_signal and confluence_signal.confidence >= self.config.confluence_threshold:
                    confluence_signals.append(confluence_signal)
            
            # Sort by timestamp
            confluence_signals.sort(key=lambda x: x.timestamp)
            
            logger.info(f"Generated {len(confluence_signals)} confluence signals")
            return confluence_signals
            
        except Exception as e:
            logger.error(f"Error analyzing confluence: {e}")
            return []
    
    def _find_common_timestamps(self, mtf_results: Dict[str, TimeframeResult]) -> List[datetime]:
        """Find common timestamps across timeframes"""
        if not mtf_results:
            return []
        
        # Start with first timeframe's timestamps
        first_timeframe = next(iter(mtf_results.values()))
        common_timestamps = set(first_timeframe.data.index)
        
        # Find intersection with other timeframes
        for result in mtf_results.values():
            common_timestamps &= set(result.data.index)
        
        return sorted(list(common_timestamps))
    
    def _calculate_confluence_at_timestamp(self, timestamp: datetime, 
                                         mtf_results: Dict[str, TimeframeResult]) -> Optional[ConfluenceSignal]:
        """Calculate confluence signal at specific timestamp"""
        try:
            # Collect signals from all timeframes
            timeframe_signals = {}
            timeframe_weights = {}
            contributing_timeframes = []
            
            total_signal = 0.0
            total_weight = 0.0
            signal_count = 0
            
            for timeframe, result in mtf_results.items():
                if timestamp not in result.data.index:
                    continue
                
                # Get signal strength from timeframe
                signal_strength = self._extract_signal_strength(result, timestamp)
                
                if signal_strength != 0:  # Non-neutral signal
                    weight = self._calculate_timeframe_weight(timeframe, result)
                    
                    timeframe_signals[timeframe] = signal_strength
                    timeframe_weights[timeframe] = weight
                    contributing_timeframes.append(timeframe)
                    
                    total_signal += signal_strength * weight
                    total_weight += weight
                    signal_count += 1
            
            # Check minimum agreement requirement
            if signal_count < self.config.min_timeframes_agreement:
                return None
            
            # Calculate consensus signal
            consensus_signal = total_signal / total_weight if total_weight > 0 else 0
            
            # Calculate confidence based on agreement
            confidence = self._calculate_confidence(timeframe_signals, consensus_signal)
            
            # Determine signal type and strength
            signal_type = self._determine_signal_type(consensus_signal)
            signal_strength = self._determine_signal_strength(abs(consensus_signal))
            
            # Calculate additional scores
            correlation_score = self._calculate_correlation_score(timeframe_signals)
            momentum_score = self._calculate_momentum_score(mtf_results, timestamp)
            
            return ConfluenceSignal(
                timestamp=timestamp,
                signal_type=signal_type,
                strength=signal_strength,
                confidence=confidence,
                contributing_timeframes=contributing_timeframes,
                timeframe_signals=timeframe_signals,
                timeframe_weights=timeframe_weights,
                consensus_score=consensus_signal,
                correlation_score=correlation_score,
                momentum_score=momentum_score,
                calculation_method='weighted_average',
                reliability=min(confidence, 1.0)
            )
            
        except Exception as e:
            logger.error(f"Error calculating confluence at {timestamp}: {e}")
            return None
    
    def _extract_signal_strength(self, result: TimeframeResult, timestamp: datetime) -> float:
        """Extract signal strength from timeframe result"""
        if not result.signals:
            return 0.0
        
        # Aggregate signals from multiple indicators
        total_signal = 0.0
        signal_count = 0
        
        for indicator_name, signals in result.signals.items():
            if timestamp in signals.index:
                signal_value = signals[timestamp]
                if not pd.isna(signal_value):
                    total_signal += signal_value
                    signal_count += 1
        
        return total_signal / signal_count if signal_count > 0 else 0.0
    
    def _calculate_timeframe_weight(self, timeframe: str, result: TimeframeResult) -> float:
        """Calculate weight for timeframe based on various factors"""
        base_weight = self.default_weights.get(timeframe, 0.1)
        
        if not self.config.weight_by_timeframe:
            return base_weight
        
        # Adjust weight based on data quality and reliability
        quality_factor = result.data_quality * result.reliability_score
        
        # Adjust based on timeframe type
        timeframe_type = self._get_timeframe_type(timeframe)
        type_multiplier = {
            TimeframeType.SCALPING: 0.8,
            TimeframeType.SHORT_TERM: 1.0,
            TimeframeType.MEDIUM_TERM: 1.2,
            TimeframeType.LONG_TERM: 1.1,
            TimeframeType.STRATEGIC: 0.9
        }.get(timeframe_type, 1.0)
        
        return base_weight * quality_factor * type_multiplier
    
    def _get_timeframe_type(self, timeframe: str) -> TimeframeType:
        """Get timeframe type classification"""
        timeframe_map = {
            '1T': TimeframeType.SCALPING,
            '5T': TimeframeType.SCALPING,
            '15T': TimeframeType.SHORT_TERM,
            '30T': TimeframeType.SHORT_TERM,
            '1H': TimeframeType.MEDIUM_TERM,
            '4H': TimeframeType.MEDIUM_TERM,
            '1D': TimeframeType.LONG_TERM,
            '1W': TimeframeType.LONG_TERM,
            '1M': TimeframeType.STRATEGIC
        }
        return timeframe_map.get(timeframe, TimeframeType.MEDIUM_TERM)
    
    def _calculate_confidence(self, timeframe_signals: Dict[str, float], consensus: float) -> float:
        """Calculate confidence based on signal agreement"""
        if not timeframe_signals:
            return 0.0
        
        # Calculate agreement score
        agreements = 0
        total_comparisons = 0
        
        for signal in timeframe_signals.values():
            if (signal > 0 and consensus > 0) or (signal < 0 and consensus < 0):
                agreements += 1
            total_comparisons += 1
        
        agreement_ratio = agreements / total_comparisons if total_comparisons > 0 else 0
        
        # Factor in signal count
        signal_count_factor = min(len(timeframe_signals) / self.config.min_timeframes_agreement, 1.0)
        
        # Factor in signal strength
        strength_factor = min(abs(consensus), 1.0)
        
        return agreement_ratio * signal_count_factor * strength_factor
    
    def _determine_signal_type(self, consensus_signal: float) -> str:
        """Determine signal type from consensus"""
        if consensus_signal > 0.1:
            return 'buy'
        elif consensus_signal < -0.1:
            return 'sell'
        else:
            return 'neutral'
    
    def _determine_signal_strength(self, abs_consensus: float) -> SignalStrength:
        """Determine signal strength from absolute consensus"""
        if abs_consensus >= 0.8:
            return SignalStrength.VERY_STRONG
        elif abs_consensus >= 0.6:
            return SignalStrength.STRONG
        elif abs_consensus >= 0.4:
            return SignalStrength.MODERATE
        elif abs_consensus >= 0.2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _calculate_correlation_score(self, timeframe_signals: Dict[str, float]) -> float:
        """Calculate correlation score between timeframe signals"""
        if len(timeframe_signals) < 2:
            return 1.0
        
        signals = list(timeframe_signals.values())
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(signals)):
            for j in range(i+1, len(signals)):
                # Simple correlation based on sign agreement
                if (signals[i] > 0 and signals[j] > 0) or (signals[i] < 0 and signals[j] < 0):
                    correlations.append(1.0)
                elif (signals[i] > 0 and signals[j] < 0) or (signals[i] < 0 and signals[j] > 0):
                    correlations.append(-1.0)
                else:
                    correlations.append(0.0)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_momentum_score(self, mtf_results: Dict[str, TimeframeResult], 
                                timestamp: datetime) -> float:
        """Calculate momentum score across timeframes"""
        try:
            momentum_scores = []
            
            for timeframe, result in mtf_results.items():
                if timestamp not in result.data.index:
                    continue
                
                # Calculate price momentum
                data = result.data
                idx = data.index.get_loc(timestamp)
                
                if idx >= 10:  # Need enough historical data
                    current_price = data['close'].iloc[idx]
                    past_price = data['close'].iloc[idx-10]
                    
                    momentum = (current_price - past_price) / past_price
                    momentum_scores.append(momentum)
            
            return np.mean(momentum_scores) if momentum_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.0


class RealTimeStreamProcessor:
    """Real-time data streaming and processing"""
    
    def __init__(self, config: TimeframeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.data_buffer = {}
        self.subscribers = []
    
    async def start_streaming(self, data_source: Callable):
        """Start real-time data streaming"""
        self.running = True
        logger.info("Starting real-time streaming")
        
        try:
            while self.running:
                # Get new data
                new_data = await self._fetch_new_data(data_source)
                
                if new_data:
                    # Update buffer
                    self._update_buffer(new_data)
                    
                    # Notify subscribers
                    await self._notify_subscribers(new_data)
                
                # Wait for next update
                await asyncio.sleep(self.config.update_frequency)
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
        finally:
            logger.info("Real-time streaming stopped")
    
    async def _fetch_new_data(self, data_source: Callable) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch new data from source"""
        try:
            return await data_source()
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def _update_buffer(self, new_data: Dict[str, pd.DataFrame]):
        """Update data buffer with new data"""
        for timeframe, data in new_data.items():
            if timeframe not in self.data_buffer:
                self.data_buffer[timeframe] = data.copy()
            else:
                # Append new data and keep buffer size
                combined = pd.concat([self.data_buffer[timeframe], data])
                self.data_buffer[timeframe] = combined.tail(self.config.buffer_size)
    
    async def _notify_subscribers(self, new_data: Dict[str, pd.DataFrame]):
        """Notify subscribers of new data"""
        for subscriber in self.subscribers:
            try:
                await subscriber(new_data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def subscribe(self, callback: Callable):
        """Subscribe to real-time updates"""
        self.subscribers.append(callback)
    
    def stop_streaming(self):
        """Stop real-time streaming"""
        self.running = False


class PerformanceOptimizer:
    """Performance optimization for large datasets"""
    
    def __init__(self, config: TimeframeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers)
    
    async def process_parallel_timeframes(self, data: Dict[str, pd.DataFrame], 
                                        analysis_func: Callable) -> Dict[str, TimeframeResult]:
        """Process multiple timeframes in parallel"""
        if not self.config.enable_parallel_processing:
            return await self._process_sequential(data, analysis_func)
        
        try:
            loop = asyncio.get_event_loop()
            
            # Create tasks for each timeframe
            tasks = []
            for timeframe, tf_data in data.items():
                task = loop.run_in_executor(
                    self.thread_pool,
                    self._analyze_single_timeframe,
                    timeframe, tf_data, analysis_func
                )
                tasks.append((timeframe, task))
            
            # Wait for all tasks to complete
            results = {}
            for timeframe, task in tasks:
                try:
                    result = await task
                    results[timeframe] = result
                except Exception as e:
                    logger.error(f"Error processing timeframe {timeframe}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            return await self._process_sequential(data, analysis_func)
    
    def _analyze_single_timeframe(self, timeframe: str, data: pd.DataFrame, 
                                 analysis_func: Callable) -> TimeframeResult:
        """Analyze single timeframe"""
        start_time = datetime.now()
        
        try:
            # Determine timeframe type
            timeframe_type = self._get_timeframe_type(timeframe)
            
            # Check if data is empty
            if data.empty:
                return TimeframeResult(
                    timeframe=timeframe,
                    timeframe_type=timeframe_type,
                    data=data,
                    calculation_time=(datetime.now() - start_time).total_seconds(),
                    reliability_score=0.0
                )
            
            # Run analysis - handle both sync and async functions
            result = analysis_func(data)
            if asyncio.iscoroutine(result):
                # If it's a coroutine, we need to await it
                # But since we're in a sync function, we'll skip for now
                indicators, signals = {}, {}
            else:
                indicators, signals = result
            
            # Calculate metrics
            signal_strength = self._calculate_signal_strength(signals)
            confidence = self._calculate_confidence_score(signals)
            trend_direction = self._determine_trend_direction(data)
            volatility = self._calculate_volatility(data)
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            return TimeframeResult(
                timeframe=timeframe,
                timeframe_type=timeframe_type,
                data=data,
                indicators=indicators,
                signals=signals,
                signal_strength=signal_strength,
                confidence=confidence,
                trend_direction=trend_direction,
                volatility=volatility,
                calculation_time=calculation_time,
                last_update=datetime.now(),
                data_quality=self._assess_data_quality(data),
                completeness=self._calculate_completeness(data),
                reliability_score=min(confidence, 1.0)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe}: {e}")
            return TimeframeResult(
                timeframe=timeframe,
                timeframe_type=TimeframeType.MEDIUM_TERM,
                data=data,
                calculation_time=(datetime.now() - start_time).total_seconds(),
                reliability_score=0.0
            )
    
    async def _process_sequential(self, data: Dict[str, pd.DataFrame], 
                                analysis_func: Callable) -> Dict[str, TimeframeResult]:
        """Process timeframes sequentially"""
        results = {}
        
        for timeframe, tf_data in data.items():
            try:
                result = self._analyze_single_timeframe(timeframe, tf_data, analysis_func)
                results[timeframe] = result
            except Exception as e:
                logger.error(f"Error processing timeframe {timeframe}: {e}")
        
        return results
    
    def _get_timeframe_type(self, timeframe: str) -> TimeframeType:
        """Get timeframe type"""
        timeframe_map = {
            '1T': TimeframeType.SCALPING,
            '5T': TimeframeType.SCALPING,
            '15T': TimeframeType.SHORT_TERM,
            '30T': TimeframeType.SHORT_TERM,
            '1H': TimeframeType.MEDIUM_TERM,
            '4H': TimeframeType.MEDIUM_TERM,
            '1D': TimeframeType.LONG_TERM,
            '1W': TimeframeType.LONG_TERM,
            '1M': TimeframeType.STRATEGIC
        }
        return timeframe_map.get(timeframe, TimeframeType.MEDIUM_TERM)
    
    def _calculate_signal_strength(self, signals: Dict[str, pd.Series]) -> float:
        """Calculate overall signal strength"""
        if not signals:
            return 0.0
        
        total_strength = 0.0
        signal_count = 0
        
        for signal_series in signals.values():
            if len(signal_series) > 0:
                strength = signal_series.abs().mean()
                if not pd.isna(strength):
                    total_strength += strength
                    signal_count += 1
        
        return total_strength / signal_count if signal_count > 0 else 0.0
    
    def _calculate_confidence_score(self, signals: Dict[str, pd.Series]) -> float:
        """Calculate confidence score for signals"""
        if not signals:
            return 0.0
        
        # Calculate agreement between signals
        signal_values = []
        for signal_series in signals.values():
            if len(signal_series) > 0:
                latest_signal = signal_series.iloc[-1]
                if not pd.isna(latest_signal):
                    signal_values.append(latest_signal)
        
        if len(signal_values) < 2:
            return 0.5
        
        # Calculate agreement
        positive_signals = sum(1 for s in signal_values if s > 0)
        negative_signals = sum(1 for s in signal_values if s < 0)
        total_signals = len(signal_values)
        
        agreement_ratio = max(positive_signals, negative_signals) / total_signals
        return agreement_ratio
    
    def _determine_trend_direction(self, data: pd.DataFrame) -> int:
        """Determine trend direction from price data"""
        if len(data) < 20:
            return 0
        
        recent_prices = data['close'].tail(20)
        
        # Calculate simple trend
        if recent_prices.iloc[-1] > recent_prices.iloc[0]:
            return 1  # Uptrend
        elif recent_prices.iloc[-1] < recent_prices.iloc[0]:
            return -1  # Downtrend
        else:
            return 0  # Sideways
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate price volatility"""
        if len(data) < 2:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        return returns.std() if len(returns) > 0 else 0.0
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess data quality"""
        if data.empty:
            return 0.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        # Check for duplicate timestamps
        duplicate_ratio = data.index.duplicated().sum() / len(data)
        
        # Calculate quality score
        quality_score = 1.0 - (missing_ratio + duplicate_ratio)
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate data completeness"""
        if data.empty:
            return 0.0
        
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        present_columns = [col for col in expected_columns if col in data.columns]
        
        return len(present_columns) / len(expected_columns)
    
    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class MultiTimeframeAnalysisEnhancement:
    """Main multi-timeframe analysis enhancement system"""
    
    def __init__(self, config: TimeframeConfig = None):
        self.config = config or TimeframeConfig()
        
        # Initialize components
        self.synchronizer = TimeframeSynchronizer(self.config)
        self.confluence_analyzer = AdvancedConfluenceAnalyzer(self.config)
        self.stream_processor = RealTimeStreamProcessor(self.config)
        self.performance_optimizer = PerformanceOptimizer(self.config)
        
        self.logger = logging.getLogger(__name__)
        
        logger.info("Multi-Timeframe Analysis Enhancement initialized")
    
    async def analyze_multiple_timeframes(self, data: Dict[str, pd.DataFrame], 
                                        indicator_func: Callable) -> Dict[str, Any]:
        """Comprehensive multi-timeframe analysis"""
        try:
            start_time = datetime.now()
            
            # Step 1: Synchronize timeframes
            synchronized_data = self.synchronizer.synchronize_timeframes(data)
            
            # Step 2: Parallel analysis of timeframes
            mtf_results = await self.performance_optimizer.process_parallel_timeframes(
                synchronized_data, indicator_func
            )
            
            # Step 3: Confluence analysis
            confluence_signals = self.confluence_analyzer.analyze_confluence(mtf_results)
            
            # Step 4: Calculate overall metrics
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Step 5: Generate summary statistics
            summary_stats = self._generate_summary_statistics(mtf_results, confluence_signals)
            
            return {
                'timestamp': datetime.now(),
                'mtf_results': mtf_results,
                'confluence_signals': confluence_signals,
                'synchronized_data': synchronized_data,
                'summary_statistics': summary_stats,
                'analysis_time': analysis_time,
                'config': self.config,
                'performance_metrics': self._calculate_performance_metrics(mtf_results, analysis_time)
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {e}")
            raise
    
    def _generate_summary_statistics(self, mtf_results: Dict[str, TimeframeResult], 
                                   confluence_signals: List[ConfluenceSignal]) -> Dict[str, Any]:
        """Generate summary statistics"""
        try:
            # Timeframe statistics
            timeframe_stats = {}
            for timeframe, result in mtf_results.items():
                timeframe_stats[timeframe] = {
                    'signal_strength': result.signal_strength,
                    'confidence': result.confidence,
                    'trend_direction': result.trend_direction,
                    'volatility': result.volatility,
                    'data_quality': result.data_quality,
                    'calculation_time': result.calculation_time
                }
            
            # Confluence statistics
            confluence_stats = {
                'total_signals': len(confluence_signals),
                'buy_signals': len([s for s in confluence_signals if s.signal_type == 'buy']),
                'sell_signals': len([s for s in confluence_signals if s.signal_type == 'sell']),
                'neutral_signals': len([s for s in confluence_signals if s.signal_type == 'neutral']),
                'avg_confidence': np.mean([s.confidence for s in confluence_signals]) if confluence_signals else 0,
                'avg_consensus': np.mean([s.consensus_score for s in confluence_signals]) if confluence_signals else 0
            }
            
            # Signal strength distribution
            strength_distribution = {}
            for strength in SignalStrength:
                count = len([s for s in confluence_signals if s.strength == strength])
                strength_distribution[strength.name] = count
            
            return {
                'timeframe_statistics': timeframe_stats,
                'confluence_statistics': confluence_stats,
                'strength_distribution': strength_distribution,
                'total_timeframes_analyzed': len(mtf_results),
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
            return {}
    
    def _calculate_performance_metrics(self, mtf_results: Dict[str, TimeframeResult], 
                                     total_time: float) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            total_data_points = sum(len(result.data) for result in mtf_results.values())
            
            avg_calculation_time = np.mean([result.calculation_time for result in mtf_results.values()])
            avg_data_quality = np.mean([result.data_quality for result in mtf_results.values()])
            avg_reliability = np.mean([result.reliability_score for result in mtf_results.values()])
            
            throughput = total_data_points / total_time if total_time > 0 else 0
            
            return {
                'total_analysis_time': total_time,
                'avg_timeframe_calculation_time': avg_calculation_time,
                'total_data_points_processed': total_data_points,
                'throughput_points_per_second': throughput,
                'avg_data_quality': avg_data_quality,
                'avg_reliability_score': avg_reliability,
                'parallel_processing_enabled': self.config.enable_parallel_processing,
                'timeframes_processed': len(mtf_results)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def start_real_time_analysis(self, data_source: Callable, 
                                     analysis_callback: Callable):
        """Start real-time multi-timeframe analysis"""
        if not self.config.enable_streaming:
            logger.warning("Real-time streaming is disabled")
            return
        
        # Subscribe to streaming updates
        async def on_new_data(new_data: Dict[str, pd.DataFrame]):
            try:
                # Run analysis on new data
                results = await self.analyze_multiple_timeframes(new_data, analysis_callback)
                
                # Notify callback
                await analysis_callback(results)
                
            except Exception as e:
                logger.error(f"Error processing real-time data: {e}")
        
        self.stream_processor.subscribe(on_new_data)
        
        # Start streaming
        await self.stream_processor.start_streaming(data_source)
    
    def stop_real_time_analysis(self):
        """Stop real-time analysis"""
        self.stream_processor.stop_streaming()
    
    def cleanup(self):
        """Cleanup resources"""
        self.performance_optimizer.cleanup()
        logger.info("Multi-Timeframe Analysis Enhancement cleaned up")


def create_multi_timeframe_analysis_enhancement(custom_config: Dict = None) -> MultiTimeframeAnalysisEnhancement:
    """Factory function to create multi-timeframe analysis enhancement system"""
    
    config = TimeframeConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return MultiTimeframeAnalysisEnhancement(config)


# Demo analysis function for testing
def demo_indicator_analysis(data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, pd.Series]]:
    """Demo indicator analysis function"""
    indicators = {}
    signals = {}
    
    try:
        # Simple moving average
        sma_20 = data['close'].rolling(window=20).mean()
        indicators['sma_20'] = sma_20
        
        # Generate signals
        signals['sma_20'] = pd.Series(index=data.index, dtype=float)
        signals['sma_20'][data['close'] > sma_20] = 1
        signals['sma_20'][data['close'] < sma_20] = -1
        signals['sma_20'].fillna(0, inplace=True)
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi
        
        # RSI signals
        signals['rsi'] = pd.Series(index=data.index, dtype=float)
        signals['rsi'][rsi < 30] = 1   # Oversold
        signals['rsi'][rsi > 70] = -1  # Overbought
        signals['rsi'].fillna(0, inplace=True)
        
    except Exception as e:
        logger.error(f"Error in demo analysis: {e}")
    
    return indicators, signals


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create enhancement system
        system = create_multi_timeframe_analysis_enhancement({
            'timeframes': ['5T', '15T', '1H', '4H'],
            'enable_parallel_processing': True,
            'confluence_threshold': 0.6
        })
        
        # Generate sample data for multiple timeframes
        end_time = datetime.now()
        timeframes = ['5T', '15T', '1H', '4H']
        
        sample_data = {}
        for tf in timeframes:
            periods = {'5T': 200, '15T': 100, '1H': 50, '4H': 25}[tf]
            dates = pd.date_range(end_time - timedelta(hours=periods), end_time, periods=periods)
            
            np.random.seed(42)
            base_price = 2000
            prices = [base_price + i + np.random.normal(0, 10) for i in range(periods)]
            
            df = pd.DataFrame({
                'open': [p * 0.999 for p in prices],
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'close': prices,
                'volume': np.random.randint(1000, 10000, periods)
            }, index=dates)
            
            sample_data[tf] = df
        
        # Run analysis
        results = await system.analyze_multiple_timeframes(sample_data, demo_indicator_analysis)
        
        print("Multi-Timeframe Analysis Enhancement Results:")
        print(f"Timeframes analyzed: {len(results['mtf_results'])}")
        print(f"Confluence signals: {len(results['confluence_signals'])}")
        print(f"Analysis time: {results['analysis_time']:.3f} seconds")
        print(f"Throughput: {results['performance_metrics']['throughput_points_per_second']:.0f} points/sec")
        
        # Cleanup
        system.cleanup()
    
    # Run the demo
    asyncio.run(main())