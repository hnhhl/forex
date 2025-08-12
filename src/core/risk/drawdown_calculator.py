"""
Drawdown Calculator System
Advanced drawdown monitoring and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..base_system import BaseSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrawdownType(Enum):
    """Types of drawdown calculations"""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    UNDERWATER = "underwater"
    ROLLING_MAX = "rolling_max"
    PEAK_TO_TROUGH = "peak_to_trough"


class DrawdownSeverity(Enum):
    """Drawdown severity levels"""
    MINOR = "minor"          # < 5%
    MODERATE = "moderate"    # 5-10%
    SIGNIFICANT = "significant"  # 10-20%
    SEVERE = "severe"        # 20-35%
    EXTREME = "extreme"      # > 35%


@dataclass
class DrawdownPeriod:
    """Drawdown period information"""
    start_date: datetime
    end_date: Optional[datetime]
    peak_value: float
    trough_value: float
    peak_index: int
    trough_index: int
    duration_days: int
    max_drawdown: float
    recovery_days: Optional[int] = None
    is_recovered: bool = False
    severity: DrawdownSeverity = DrawdownSeverity.MINOR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'peak_value': self.peak_value,
            'trough_value': self.trough_value,
            'peak_index': self.peak_index,
            'trough_index': self.trough_index,
            'duration_days': self.duration_days,
            'max_drawdown': self.max_drawdown,
            'recovery_days': self.recovery_days,
            'is_recovered': self.is_recovered,
            'severity': self.severity.value
        }


@dataclass
class DrawdownStatistics:
    """Comprehensive drawdown statistics"""
    current_drawdown: float
    max_drawdown: float
    max_drawdown_duration: int
    average_drawdown: float
    average_duration: int
    recovery_factor: float
    pain_index: float
    ulcer_index: float
    
    # Drawdown frequency
    total_drawdown_periods: int
    drawdowns_per_year: float
    
    # Severity distribution
    minor_drawdowns: int
    moderate_drawdowns: int
    significant_drawdowns: int
    severe_drawdowns: int
    extreme_drawdowns: int
    
    # Recovery statistics
    average_recovery_time: float
    max_recovery_time: int
    recovery_rate: float
    
    # Risk metrics
    drawdown_volatility: float
    drawdown_skewness: float
    drawdown_kurtosis: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'average_drawdown': self.average_drawdown,
            'average_duration': self.average_duration,
            'recovery_factor': self.recovery_factor,
            'pain_index': self.pain_index,
            'ulcer_index': self.ulcer_index,
            'total_drawdown_periods': self.total_drawdown_periods,
            'drawdowns_per_year': self.drawdowns_per_year,
            'minor_drawdowns': self.minor_drawdowns,
            'moderate_drawdowns': self.moderate_drawdowns,
            'significant_drawdowns': self.significant_drawdowns,
            'severe_drawdowns': self.severe_drawdowns,
            'extreme_drawdowns': self.extreme_drawdowns,
            'average_recovery_time': self.average_recovery_time,
            'max_recovery_time': self.max_recovery_time,
            'recovery_rate': self.recovery_rate,
            'drawdown_volatility': self.drawdown_volatility,
            'drawdown_skewness': self.drawdown_skewness,
            'drawdown_kurtosis': self.drawdown_kurtosis
        }


class DrawdownCalculator(BaseSystem):
    """
    Advanced Drawdown Calculator
    
    Features:
    - Multiple drawdown calculation methods
    - Drawdown period identification
    - Recovery analysis
    - Risk metrics calculation
    - Real-time monitoring
    - Alert generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Drawdown Calculator"""
        super().__init__("DrawdownCalculator", config)
        
        # Configuration
        self.min_drawdown_threshold = config.get('min_drawdown_threshold', 0.01) if config else 0.01  # 1%
        self.lookback_window = config.get('lookback_window', 252) if config else 252  # 1 year
        self.rolling_window = config.get('rolling_window', 30) if config else 30  # 30 days
        
        # Data storage
        self.price_data: Optional[pd.Series] = None
        self.drawdown_series: Optional[pd.Series] = None
        self.drawdown_periods: List[DrawdownPeriod] = []
        self.underwater_curve: Optional[pd.Series] = None
        
        logger.info("DrawdownCalculator initialized")
    
    def set_data(self, price_data: Union[pd.Series, pd.DataFrame]):
        """Set price data for drawdown calculation"""
        try:
            if isinstance(price_data, pd.DataFrame):
                if len(price_data.columns) == 1:
                    self.price_data = price_data.iloc[:, 0]
                else:
                    # Use portfolio value (sum of all columns)
                    self.price_data = price_data.sum(axis=1)
            else:
                self.price_data = price_data.copy()
            
            # Ensure datetime index
            if not isinstance(self.price_data.index, pd.DatetimeIndex):
                self.price_data.index = pd.to_datetime(self.price_data.index)
            
            logger.info(f"Price data set: {len(self.price_data)} observations")
            
        except Exception as e:
            logger.error(f"Error setting price data: {e}")
            raise
    
    def calculate_drawdown(self, drawdown_type: DrawdownType = DrawdownType.RELATIVE) -> pd.Series:
        """Calculate drawdown series"""
        try:
            if self.price_data is None or self.price_data.empty:
                raise ValueError("Price data not available")
            
            prices = self.price_data.copy()
            
            if drawdown_type == DrawdownType.ABSOLUTE:
                # Absolute drawdown from initial value
                initial_value = prices.iloc[0]
                drawdown = prices - initial_value
                
            elif drawdown_type == DrawdownType.RELATIVE:
                # Relative drawdown from running maximum
                running_max = prices.expanding().max()
                drawdown = (prices - running_max) / running_max
                
            elif drawdown_type == DrawdownType.UNDERWATER:
                # Underwater curve (time below previous peak)
                running_max = prices.expanding().max()
                drawdown = (prices - running_max) / running_max
                
            elif drawdown_type == DrawdownType.ROLLING_MAX:
                # Drawdown from rolling maximum
                rolling_max = prices.rolling(window=self.rolling_window).max()
                drawdown = (prices - rolling_max) / rolling_max
                
            elif drawdown_type == DrawdownType.PEAK_TO_TROUGH:
                # Peak-to-trough drawdown
                running_max = prices.expanding().max()
                drawdown = (prices - running_max) / running_max
            
            else:
                raise ValueError(f"Unknown drawdown type: {drawdown_type}")
            
            self.drawdown_series = drawdown
            
            if drawdown_type == DrawdownType.UNDERWATER:
                self.underwater_curve = drawdown
            
            logger.info(f"Calculated {drawdown_type.value} drawdown")
            return drawdown
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            raise
    
    def identify_drawdown_periods(self, min_threshold: Optional[float] = None) -> List[DrawdownPeriod]:
        """Identify individual drawdown periods"""
        try:
            if self.price_data is None or self.drawdown_series is None:
                raise ValueError("Price data and drawdown series required")
            
            threshold = min_threshold or self.min_drawdown_threshold
            prices = self.price_data
            drawdowns = self.drawdown_series
            
            periods = []
            in_drawdown = False
            current_period = None
            
            for i, (date, dd_value) in enumerate(drawdowns.items()):
                if not in_drawdown and dd_value <= -threshold:
                    # Start of new drawdown period
                    in_drawdown = True
                    
                    # Find the peak (look back for the actual peak)
                    peak_idx = i
                    peak_value = prices.iloc[i]
                    
                    # Look back to find the true peak
                    for j in range(max(0, i-50), i):  # Look back up to 50 periods
                        if prices.iloc[j] > peak_value:
                            peak_value = prices.iloc[j]
                            peak_idx = j
                    
                    current_period = {
                        'start_date': prices.index[peak_idx],
                        'peak_value': peak_value,
                        'peak_index': peak_idx,
                        'trough_value': prices.iloc[i],
                        'trough_index': i,
                        'max_drawdown': dd_value
                    }
                
                elif in_drawdown:
                    # Update trough if deeper drawdown
                    if dd_value < current_period['max_drawdown']:
                        current_period['max_drawdown'] = dd_value
                        current_period['trough_value'] = prices.iloc[i]
                        current_period['trough_index'] = i
                    
                    # Check for recovery
                    if dd_value >= -threshold * 0.1:  # 90% recovery
                        # End of drawdown period
                        in_drawdown = False
                        
                        duration_days = (date - current_period['start_date']).days
                        
                        # Check for full recovery
                        is_recovered = prices.iloc[i] >= current_period['peak_value'] * 0.99
                        recovery_days = None
                        
                        if is_recovered:
                            recovery_days = duration_days
                        else:
                            # Look forward for recovery
                            for j in range(i+1, min(len(prices), i+252)):  # Look forward up to 1 year
                                if prices.iloc[j] >= current_period['peak_value'] * 0.99:
                                    recovery_days = (prices.index[j] - current_period['start_date']).days
                                    is_recovered = True
                                    break
                        
                        # Determine severity
                        severity = self._classify_drawdown_severity(abs(current_period['max_drawdown']))
                        
                        period = DrawdownPeriod(
                            start_date=current_period['start_date'],
                            end_date=date,
                            peak_value=current_period['peak_value'],
                            trough_value=current_period['trough_value'],
                            peak_index=current_period['peak_index'],
                            trough_index=current_period['trough_index'],
                            duration_days=duration_days,
                            max_drawdown=abs(current_period['max_drawdown']),
                            recovery_days=recovery_days,
                            is_recovered=is_recovered,
                            severity=severity
                        )
                        
                        periods.append(period)
                        current_period = None
            
            # Handle ongoing drawdown
            if in_drawdown and current_period is not None:
                duration_days = (drawdowns.index[-1] - current_period['start_date']).days
                severity = self._classify_drawdown_severity(abs(current_period['max_drawdown']))
                
                period = DrawdownPeriod(
                    start_date=current_period['start_date'],
                    end_date=None,  # Ongoing
                    peak_value=current_period['peak_value'],
                    trough_value=current_period['trough_value'],
                    peak_index=current_period['peak_index'],
                    trough_index=current_period['trough_index'],
                    duration_days=duration_days,
                    max_drawdown=abs(current_period['max_drawdown']),
                    recovery_days=None,
                    is_recovered=False,
                    severity=severity
                )
                
                periods.append(period)
            
            self.drawdown_periods = periods
            logger.info(f"Identified {len(periods)} drawdown periods")
            
            return periods
            
        except Exception as e:
            logger.error(f"Error identifying drawdown periods: {e}")
            raise
    
    def _classify_drawdown_severity(self, drawdown_magnitude: float) -> DrawdownSeverity:
        """Classify drawdown severity"""
        if drawdown_magnitude < 0.05:
            return DrawdownSeverity.MINOR
        elif drawdown_magnitude < 0.10:
            return DrawdownSeverity.MODERATE
        elif drawdown_magnitude < 0.20:
            return DrawdownSeverity.SIGNIFICANT
        elif drawdown_magnitude < 0.35:
            return DrawdownSeverity.SEVERE
        else:
            return DrawdownSeverity.EXTREME
    
    def calculate_statistics(self) -> DrawdownStatistics:
        """Calculate comprehensive drawdown statistics"""
        try:
            if self.drawdown_series is None or not self.drawdown_periods:
                raise ValueError("Drawdown data not available")
            
            drawdowns = self.drawdown_series
            periods = self.drawdown_periods
            
            # Basic statistics
            current_drawdown = abs(drawdowns.iloc[-1])
            max_drawdown = abs(drawdowns.min())
            
            # Period statistics
            if periods:
                durations = [p.duration_days for p in periods]
                magnitudes = [p.max_drawdown for p in periods]
                
                max_drawdown_duration = max(durations)
                average_drawdown = np.mean(magnitudes)
                average_duration = np.mean(durations)
                
                # Recovery statistics
                recovered_periods = [p for p in periods if p.is_recovered and p.recovery_days is not None]
                if recovered_periods:
                    recovery_times = [p.recovery_days for p in recovered_periods]
                    average_recovery_time = np.mean(recovery_times)
                    max_recovery_time = max(recovery_times)
                    recovery_rate = len(recovered_periods) / len(periods)
                else:
                    average_recovery_time = 0.0
                    max_recovery_time = 0
                    recovery_rate = 0.0
                
                # Severity distribution
                severity_counts = {
                    DrawdownSeverity.MINOR: 0,
                    DrawdownSeverity.MODERATE: 0,
                    DrawdownSeverity.SIGNIFICANT: 0,
                    DrawdownSeverity.SEVERE: 0,
                    DrawdownSeverity.EXTREME: 0
                }
                
                for period in periods:
                    severity_counts[period.severity] += 1
                
            else:
                max_drawdown_duration = 0
                average_drawdown = 0.0
                average_duration = 0.0
                average_recovery_time = 0.0
                max_recovery_time = 0
                recovery_rate = 0.0
                severity_counts = {s: 0 for s in DrawdownSeverity}
            
            # Advanced metrics
            recovery_factor = self._calculate_recovery_factor()
            pain_index = self._calculate_pain_index()
            ulcer_index = self._calculate_ulcer_index()
            
            # Frequency metrics
            total_periods = len(periods)
            years = len(drawdowns) / 252  # Assuming daily data
            drawdowns_per_year = total_periods / years if years > 0 else 0
            
            # Risk metrics
            drawdown_volatility = drawdowns.std()
            drawdown_skewness = drawdowns.skew()
            drawdown_kurtosis = drawdowns.kurtosis()
            
            statistics = DrawdownStatistics(
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                average_drawdown=average_drawdown,
                average_duration=average_duration,
                recovery_factor=recovery_factor,
                pain_index=pain_index,
                ulcer_index=ulcer_index,
                total_drawdown_periods=total_periods,
                drawdowns_per_year=drawdowns_per_year,
                minor_drawdowns=severity_counts[DrawdownSeverity.MINOR],
                moderate_drawdowns=severity_counts[DrawdownSeverity.MODERATE],
                significant_drawdowns=severity_counts[DrawdownSeverity.SIGNIFICANT],
                severe_drawdowns=severity_counts[DrawdownSeverity.SEVERE],
                extreme_drawdowns=severity_counts[DrawdownSeverity.EXTREME],
                average_recovery_time=average_recovery_time,
                max_recovery_time=max_recovery_time,
                recovery_rate=recovery_rate,
                drawdown_volatility=drawdown_volatility,
                drawdown_skewness=drawdown_skewness,
                drawdown_kurtosis=drawdown_kurtosis
            )
            
            logger.info("Calculated comprehensive drawdown statistics")
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating drawdown statistics: {e}")
            raise
    
    def _calculate_recovery_factor(self) -> float:
        """Calculate recovery factor"""
        try:
            if self.price_data is None or len(self.price_data) < 2:
                return 0.0
            
            total_return = (self.price_data.iloc[-1] / self.price_data.iloc[0]) - 1
            max_drawdown = abs(self.drawdown_series.min()) if self.drawdown_series is not None else 0.01
            
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0.0
            return recovery_factor
            
        except Exception as e:
            logger.error(f"Error calculating recovery factor: {e}")
            return 0.0
    
    def _calculate_pain_index(self) -> float:
        """Calculate pain index (average drawdown)"""
        try:
            if self.drawdown_series is None:
                return 0.0
            
            # Pain index is the average of all negative drawdowns
            negative_drawdowns = self.drawdown_series[self.drawdown_series < 0]
            pain_index = abs(negative_drawdowns.mean()) if len(negative_drawdowns) > 0 else 0.0
            
            return pain_index
            
        except Exception as e:
            logger.error(f"Error calculating pain index: {e}")
            return 0.0
    
    def _calculate_ulcer_index(self) -> float:
        """Calculate ulcer index (RMS of drawdowns)"""
        try:
            if self.drawdown_series is None:
                return 0.0
            
            # Ulcer index is the RMS of drawdowns
            squared_drawdowns = self.drawdown_series ** 2
            ulcer_index = np.sqrt(squared_drawdowns.mean())
            
            return ulcer_index
            
        except Exception as e:
            logger.error(f"Error calculating ulcer index: {e}")
            return 0.0
    
    def get_current_drawdown_info(self) -> Dict[str, Any]:
        """Get current drawdown information"""
        try:
            if self.drawdown_series is None or self.price_data is None:
                return {'message': 'No drawdown data available'}
            
            current_drawdown = abs(self.drawdown_series.iloc[-1])
            current_price = self.price_data.iloc[-1]
            
            # Find current peak
            running_max = self.price_data.expanding().max()
            current_peak = running_max.iloc[-1]
            
            # Find when peak occurred
            peak_date = None
            for i in range(len(self.price_data) - 1, -1, -1):
                if self.price_data.iloc[i] == current_peak:
                    peak_date = self.price_data.index[i]
                    break
            
            # Calculate duration
            current_date = self.price_data.index[-1]
            duration_days = (current_date - peak_date).days if peak_date else 0
            
            # Classify severity
            severity = self._classify_drawdown_severity(current_drawdown)
            
            # Check if in significant drawdown
            is_in_drawdown = current_drawdown > self.min_drawdown_threshold
            
            info = {
                'current_drawdown': current_drawdown,
                'current_price': current_price,
                'peak_price': current_peak,
                'peak_date': peak_date.isoformat() if peak_date else None,
                'duration_days': duration_days,
                'severity': severity.value,
                'is_in_drawdown': is_in_drawdown,
                'recovery_needed': (current_peak - current_price) / current_price if current_price > 0 else 0,
                'timestamp': current_date.isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting current drawdown info: {e}")
            return {'error': str(e)}
    
    def get_worst_drawdowns(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get worst drawdown periods"""
        try:
            if not self.drawdown_periods:
                return []
            
            # Sort by magnitude (worst first)
            sorted_periods = sorted(self.drawdown_periods, key=lambda x: x.max_drawdown, reverse=True)
            
            worst_periods = []
            for period in sorted_periods[:n]:
                worst_periods.append(period.to_dict())
            
            return worst_periods
            
        except Exception as e:
            logger.error(f"Error getting worst drawdowns: {e}")
            return []
    
    def calculate_rolling_drawdown(self, window: int = 30) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        try:
            if self.drawdown_series is None:
                raise ValueError("Drawdown series not available")
            
            # Calculate rolling maximum drawdown
            rolling_max_dd = self.drawdown_series.rolling(window=window).min()
            rolling_max_dd = abs(rolling_max_dd)
            
            logger.info(f"Calculated rolling {window}-period maximum drawdown")
            return rolling_max_dd
            
        except Exception as e:
            logger.error(f"Error calculating rolling drawdown: {e}")
            raise
    
    def generate_drawdown_report(self) -> Dict[str, Any]:
        """Generate comprehensive drawdown report"""
        try:
            if self.price_data is None:
                return {'error': 'No price data available'}
            
            # Calculate all metrics
            if self.drawdown_series is None:
                self.calculate_drawdown()
            
            if not self.drawdown_periods:
                self.identify_drawdown_periods()
            
            statistics = self.calculate_statistics()
            current_info = self.get_current_drawdown_info()
            worst_drawdowns = self.get_worst_drawdowns(5)
            
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'data_period': {
                    'start_date': self.price_data.index[0].isoformat(),
                    'end_date': self.price_data.index[-1].isoformat(),
                    'total_observations': len(self.price_data)
                },
                'current_drawdown': current_info,
                'statistics': statistics.to_dict(),
                'worst_drawdowns': worst_drawdowns,
                'summary': {
                    'total_drawdown_periods': len(self.drawdown_periods),
                    'average_drawdown_magnitude': statistics.average_drawdown,
                    'average_drawdown_duration': statistics.average_duration,
                    'recovery_rate': statistics.recovery_rate,
                    'pain_index': statistics.pain_index,
                    'ulcer_index': statistics.ulcer_index
                }
            }
            
            logger.info("Generated comprehensive drawdown report")
            return report
            
        except Exception as e:
            logger.error(f"Error generating drawdown report: {e}")
            return {'error': str(e)}
    
    def export_data(self, filepath: str) -> bool:
        """Export drawdown data"""
        try:
            export_data = {
                'drawdown_series': {str(k): v for k, v in self.drawdown_series.to_dict().items()} if self.drawdown_series is not None else None,
                'drawdown_periods': [period.to_dict() for period in self.drawdown_periods],
                'underwater_curve': {str(k): v for k, v in self.underwater_curve.to_dict().items()} if self.underwater_curve is not None else None,
                'configuration': {
                    'min_drawdown_threshold': self.min_drawdown_threshold,
                    'lookback_window': self.lookback_window,
                    'rolling_window': self.rolling_window
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Drawdown data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting drawdown data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get drawdown calculator statistics"""
        return {
            'price_data_available': self.price_data is not None,
            'price_data_length': len(self.price_data) if self.price_data is not None else 0,
            'drawdown_series_calculated': self.drawdown_series is not None,
            'drawdown_periods_identified': len(self.drawdown_periods),
            'underwater_curve_available': self.underwater_curve is not None,
            'min_drawdown_threshold': self.min_drawdown_threshold,
            'lookback_window': self.lookback_window,
            'rolling_window': self.rolling_window,
            'last_update': self.last_update.isoformat() if self.last_update else None
        } 