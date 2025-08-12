"""
Risk Monitor System
Real-time risk metrics calculation and monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from ..base_system import BaseSystem
from .risk_types import RiskLevel, RiskMetrics
from .var_calculator import VaRCalculator, VaRResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskMetricType(Enum):
    """Types of risk metrics"""
    VAR = "var"
    CVAR = "cvar"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    LEVERAGE = "leverage"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"


@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_type: RiskMetricType
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    portfolio_id: Optional[str] = None
    symbol: Optional[str] = None
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class RiskThreshold:
    """Risk threshold configuration"""
    metric_type: RiskMetricType
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    is_percentage: bool = False
    is_absolute: bool = True
    lookback_period: int = 1  # periods to consider
    consecutive_breaches: int = 1  # consecutive breaches to trigger


@dataclass
class RealTimeRiskMetrics:
    """Real-time risk metrics snapshot"""
    timestamp: datetime
    portfolio_value: float
    daily_pnl: float
    daily_return: float
    
    # VaR metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    drawdown_duration: int
    
    # Volatility metrics
    realized_volatility: float
    implied_volatility: Optional[float]
    volatility_percentile: float
    
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk ratios
    leverage_ratio: float
    concentration_risk: float
    correlation_risk: float
    
    # Market metrics
    market_beta: float
    tracking_error: float
    information_ratio: float
    
    # Alert status
    active_alerts: int
    critical_alerts: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'daily_return': self.daily_return,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'realized_volatility': self.realized_volatility,
            'implied_volatility': self.implied_volatility,
            'volatility_percentile': self.volatility_percentile,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'leverage_ratio': self.leverage_ratio,
            'concentration_risk': self.concentration_risk,
            'correlation_risk': self.correlation_risk,
            'market_beta': self.market_beta,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio,
            'active_alerts': self.active_alerts,
            'critical_alerts': self.critical_alerts
        }


class RiskMonitor(BaseSystem):
    """
    Real-time Risk Monitoring System
    
    Features:
    - Real-time risk metrics calculation
    - Configurable risk thresholds and alerts
    - Multi-level alert system (Info, Warning, Critical, Emergency)
    - Historical risk metrics tracking
    - Risk dashboard data generation
    - Automated risk reporting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Risk Monitor"""
        super().__init__("RiskMonitor", config)
        
        # Configuration
        self.monitoring_interval = config.get('monitoring_interval', 60) if config else 60  # seconds
        self.max_history_size = config.get('max_history_size', 10000) if config else 10000
        self.enable_real_time = config.get('enable_real_time', True) if config else True
        self.risk_free_rate = config.get('risk_free_rate', 0.02) if config else 0.02  # 2% annual
        
        # Data storage
        self.portfolio_data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.current_positions: Dict[str, float] = {}
        self.portfolio_values: deque = deque(maxlen=self.max_history_size)
        self.risk_metrics_history: deque = deque(maxlen=self.max_history_size)
        
        # Risk thresholds
        self.risk_thresholds: List[RiskThreshold] = []
        self.setup_default_thresholds()
        
        # Alerts
        self.active_alerts: List[RiskAlert] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitoring_lock = threading.RLock()
        
        # VaR Calculator integration
        self.var_calculator = VaRCalculator()
        
        logger.info(f"RiskMonitor initialized with {self.monitoring_interval}s interval")
    
    def setup_default_thresholds(self):
        """Setup default risk thresholds"""
        default_thresholds = [
            # VaR thresholds
            RiskThreshold(
                metric_type=RiskMetricType.VAR,
                metric_name="var_95",
                warning_threshold=0.02,  # 2% of portfolio
                critical_threshold=0.05,  # 5% of portfolio
                emergency_threshold=0.10,  # 10% of portfolio
                is_percentage=True
            ),
            
            # Drawdown thresholds
            RiskThreshold(
                metric_type=RiskMetricType.DRAWDOWN,
                metric_name="current_drawdown",
                warning_threshold=0.05,  # 5% drawdown
                critical_threshold=0.10,  # 10% drawdown
                emergency_threshold=0.20,  # 20% drawdown
                is_percentage=True
            ),
            
            # Volatility thresholds
            RiskThreshold(
                metric_type=RiskMetricType.VOLATILITY,
                metric_name="realized_volatility",
                warning_threshold=0.20,  # 20% annual volatility
                critical_threshold=0.35,  # 35% annual volatility
                emergency_threshold=0.50,  # 50% annual volatility
                is_percentage=True
            ),
            
            # Leverage thresholds
            RiskThreshold(
                metric_type=RiskMetricType.LEVERAGE,
                metric_name="leverage_ratio",
                warning_threshold=2.0,   # 2x leverage
                critical_threshold=5.0,  # 5x leverage
                emergency_threshold=10.0, # 10x leverage
                is_absolute=True
            ),
            
            # Concentration thresholds
            RiskThreshold(
                metric_type=RiskMetricType.CONCENTRATION,
                metric_name="concentration_risk",
                warning_threshold=0.30,  # 30% in single position
                critical_threshold=0.50,  # 50% in single position
                emergency_threshold=0.70,  # 70% in single position
                is_percentage=True
            ),
            
            # Sharpe ratio thresholds (lower is worse)
            RiskThreshold(
                metric_type=RiskMetricType.SHARPE_RATIO,
                metric_name="sharpe_ratio",
                warning_threshold=0.5,   # Below 0.5
                critical_threshold=0.0,  # Below 0.0
                emergency_threshold=-0.5, # Below -0.5
                is_absolute=True
            )
        ]
        
        self.risk_thresholds.extend(default_thresholds)
        logger.info(f"Setup {len(default_thresholds)} default risk thresholds")
    
    def set_portfolio_data(self, portfolio_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None):
        """Set portfolio and benchmark data"""
        try:
            self.portfolio_data = portfolio_data.copy()
            self.benchmark_data = benchmark_data.copy() if benchmark_data is not None else None
            
            # Initialize VaR calculator
            if len(portfolio_data.columns) > 1:
                # Multi-asset portfolio
                returns_data = portfolio_data.pct_change().dropna()
                portfolio_values = portfolio_data.sum(axis=1)
            else:
                # Single asset
                returns_data = portfolio_data.pct_change().dropna()
                portfolio_values = portfolio_data.iloc[:, 0]
            
            if not returns_data.empty:
                # Align portfolio values with returns data (same index)
                aligned_portfolio_values = portfolio_values.loc[returns_data.index]
                self.var_calculator.set_data(returns_data, aligned_portfolio_values)
            
            logger.info(f"Portfolio data set: {len(portfolio_data)} observations, {len(portfolio_data.columns)} assets")
            
        except Exception as e:
            logger.error(f"Error setting portfolio data: {e}")
            raise
    
    def update_positions(self, positions: Dict[str, float]):
        """Update current positions"""
        with self.monitoring_lock:
            self.current_positions = positions.copy()
            logger.debug(f"Updated positions: {len(positions)} instruments")
    
    def add_risk_threshold(self, threshold: RiskThreshold):
        """Add custom risk threshold"""
        self.risk_thresholds.append(threshold)
        logger.info(f"Added risk threshold for {threshold.metric_name}")
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
        logger.info("Added alert callback")
    
    def calculate_real_time_metrics(self) -> RealTimeRiskMetrics:
        """Calculate real-time risk metrics"""
        try:
            if self.portfolio_data is None or self.portfolio_data.empty:
                raise ValueError("Portfolio data not available")
            
            current_time = datetime.now()
            
            # Portfolio value and P&L
            if len(self.portfolio_data.columns) == 1:
                portfolio_value = self.portfolio_data.iloc[-1, 0]
                if len(self.portfolio_data) > 1:
                    daily_pnl = self.portfolio_data.iloc[-1, 0] - self.portfolio_data.iloc[-2, 0]
                    daily_return = daily_pnl / self.portfolio_data.iloc[-2, 0]
                else:
                    daily_pnl = 0.0
                    daily_return = 0.0
            else:
                portfolio_value = self.portfolio_data.iloc[-1].sum()
                if len(self.portfolio_data) > 1:
                    prev_value = self.portfolio_data.iloc[-2].sum()
                    daily_pnl = portfolio_value - prev_value
                    daily_return = daily_pnl / prev_value if prev_value != 0 else 0.0
                else:
                    daily_pnl = 0.0
                    daily_return = 0.0
            
            # Calculate returns for metrics
            if len(self.portfolio_data.columns) == 1:
                returns = self.portfolio_data.iloc[:, 0].pct_change().dropna()
            else:
                portfolio_values = self.portfolio_data.sum(axis=1)
                returns = portfolio_values.pct_change().dropna()
            
            # VaR metrics
            var_95 = var_99 = cvar_95 = cvar_99 = 0.0
            if len(returns) >= 30:  # Minimum data for VaR
                try:
                    var_result_95 = self.var_calculator.calculate_historical_var(0.95)
                    var_result_99 = self.var_calculator.calculate_historical_var(0.99)
                    
                    var_95 = abs(var_result_95.var_value / portfolio_value) if portfolio_value != 0 else 0
                    var_99 = abs(var_result_99.var_value / portfolio_value) if portfolio_value != 0 else 0
                    cvar_95 = abs(var_result_95.cvar_value / portfolio_value) if portfolio_value != 0 else 0
                    cvar_99 = abs(var_result_99.cvar_value / portfolio_value) if portfolio_value != 0 else 0
                except:
                    pass
            
            # Drawdown metrics
            current_drawdown, max_drawdown, drawdown_duration = self._calculate_drawdown_metrics()
            
            # Volatility metrics
            realized_volatility = self._calculate_realized_volatility(returns)
            implied_volatility = None  # Would need options data
            volatility_percentile = self._calculate_volatility_percentile(returns)
            
            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            # Risk ratios
            leverage_ratio = self._calculate_leverage_ratio()
            concentration_risk = self._calculate_concentration_risk()
            correlation_risk = self._calculate_correlation_risk()
            
            # Market metrics
            market_beta = self._calculate_market_beta(returns)
            tracking_error = self._calculate_tracking_error(returns)
            information_ratio = self._calculate_information_ratio(returns)
            
            # Alert counts
            active_alerts = len(self.active_alerts)
            critical_alerts = len([a for a in self.active_alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]])
            
            metrics = RealTimeRiskMetrics(
                timestamp=current_time,
                portfolio_value=portfolio_value,
                daily_pnl=daily_pnl,
                daily_return=daily_return,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                drawdown_duration=drawdown_duration,
                realized_volatility=realized_volatility,
                implied_volatility=implied_volatility,
                volatility_percentile=volatility_percentile,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                leverage_ratio=leverage_ratio,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                market_beta=market_beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                active_alerts=active_alerts,
                critical_alerts=critical_alerts
            )
            
            # Store metrics
            self.risk_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating real-time metrics: {e}")
            raise
    
    def _calculate_drawdown_metrics(self) -> Tuple[float, float, int]:
        """Calculate drawdown metrics"""
        try:
            if self.portfolio_data is None or len(self.portfolio_data) < 2:
                return 0.0, 0.0, 0
            
            # Get portfolio values
            if len(self.portfolio_data.columns) == 1:
                values = self.portfolio_data.iloc[:, 0]
            else:
                values = self.portfolio_data.sum(axis=1)
            
            # Calculate running maximum
            running_max = values.expanding().max()
            
            # Calculate drawdown
            drawdown = (values - running_max) / running_max
            
            current_drawdown = drawdown.iloc[-1]
            max_drawdown = drawdown.min()
            
            # Calculate drawdown duration
            drawdown_duration = 0
            for i in range(len(drawdown) - 1, -1, -1):
                if drawdown.iloc[i] < 0:
                    drawdown_duration += 1
                else:
                    break
            
            return abs(current_drawdown), abs(max_drawdown), drawdown_duration
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return 0.0, 0.0, 0
    
    def _calculate_realized_volatility(self, returns: pd.Series) -> float:
        """Calculate realized volatility (annualized)"""
        try:
            if len(returns) < 2:
                return 0.0
            
            # Use last 30 days for realized volatility
            recent_returns = returns.tail(30)
            daily_vol = recent_returns.std()
            annual_vol = daily_vol * np.sqrt(252)  # Annualize
            
            return annual_vol
            
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return 0.0
    
    def _calculate_volatility_percentile(self, returns: pd.Series) -> float:
        """Calculate volatility percentile (current vs historical)"""
        try:
            if len(returns) < 60:  # Need at least 60 days
                return 0.5
            
            # Calculate rolling 30-day volatility
            rolling_vol = returns.rolling(30).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) < 2:
                return 0.5
            
            current_vol = rolling_vol.iloc[-1]
            percentile = (rolling_vol <= current_vol).mean()
            
            return percentile
            
        except Exception as e:
            logger.error(f"Error calculating volatility percentile: {e}")
            return 0.5
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 30:
                return 0.0
            
            excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
            
            if excess_returns.std() == 0:
                return 0.0
            
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns) < 30:
                return 0.0
            
            excess_returns = returns - (self.risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            
            sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
            return sortino
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        try:
            if len(returns) < 30 or max_drawdown == 0:
                return 0.0
            
            annual_return = returns.mean() * 252
            calmar = annual_return / max_drawdown
            
            return calmar
            
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def _calculate_leverage_ratio(self) -> float:
        """Calculate leverage ratio"""
        try:
            if not self.current_positions:
                return 1.0
            
            total_exposure = sum(abs(position) for position in self.current_positions.values())
            
            if self.portfolio_data is not None and not self.portfolio_data.empty:
                if len(self.portfolio_data.columns) == 1:
                    portfolio_value = abs(self.portfolio_data.iloc[-1, 0])
                else:
                    portfolio_value = abs(self.portfolio_data.iloc[-1].sum())
                
                if portfolio_value > 0:
                    leverage = total_exposure / portfolio_value
                    return leverage
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating leverage ratio: {e}")
            return 1.0
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk (largest position as % of portfolio)"""
        try:
            if not self.current_positions:
                return 0.0
            
            position_values = list(self.current_positions.values())
            if not position_values:
                return 0.0
            
            max_position = max(abs(pos) for pos in position_values)
            total_value = sum(abs(pos) for pos in position_values)
            
            if total_value > 0:
                concentration = max_position / total_value
                return concentration
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk"""
        try:
            if self.portfolio_data is None or len(self.portfolio_data.columns) < 2:
                return 0.0
            
            returns = self.portfolio_data.pct_change().dropna()
            if len(returns) < 30:
                return 0.0
            
            corr_matrix = returns.corr()
            
            # Average correlation (excluding diagonal)
            n = len(corr_matrix)
            total_corr = 0
            count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    total_corr += abs(corr_matrix.iloc[i, j])
                    count += 1
            
            if count > 0:
                avg_correlation = total_corr / count
                return avg_correlation
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def _calculate_market_beta(self, returns: pd.Series) -> float:
        """Calculate market beta"""
        try:
            if self.benchmark_data is None or len(returns) < 30:
                return 1.0
            
            # Get benchmark returns
            if len(self.benchmark_data.columns) == 1:
                benchmark_returns = self.benchmark_data.iloc[:, 0].pct_change().dropna()
            else:
                benchmark_values = self.benchmark_data.sum(axis=1)
                benchmark_returns = benchmark_values.pct_change().dropna()
            
            # Align returns
            aligned_returns = returns.align(benchmark_returns, join='inner')
            portfolio_ret = aligned_returns[0]
            benchmark_ret = aligned_returns[1]
            
            if len(portfolio_ret) < 30:
                return 1.0
            
            # Calculate beta
            covariance = np.cov(portfolio_ret, benchmark_ret)[0, 1]
            benchmark_variance = np.var(benchmark_ret)
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                return beta
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating market beta: {e}")
            return 1.0
    
    def _calculate_tracking_error(self, returns: pd.Series) -> float:
        """Calculate tracking error"""
        try:
            if self.benchmark_data is None or len(returns) < 30:
                return 0.0
            
            # Get benchmark returns
            if len(self.benchmark_data.columns) == 1:
                benchmark_returns = self.benchmark_data.iloc[:, 0].pct_change().dropna()
            else:
                benchmark_values = self.benchmark_data.sum(axis=1)
                benchmark_returns = benchmark_values.pct_change().dropna()
            
            # Align returns
            aligned_returns = returns.align(benchmark_returns, join='inner')
            portfolio_ret = aligned_returns[0]
            benchmark_ret = aligned_returns[1]
            
            if len(portfolio_ret) < 30:
                return 0.0
            
            # Calculate tracking error
            excess_returns = portfolio_ret - benchmark_ret
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            return tracking_error
            
        except Exception as e:
            logger.error(f"Error calculating tracking error: {e}")
            return 0.0
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate information ratio"""
        try:
            if self.benchmark_data is None or len(returns) < 30:
                return 0.0
            
            # Get benchmark returns
            if len(self.benchmark_data.columns) == 1:
                benchmark_returns = self.benchmark_data.iloc[:, 0].pct_change().dropna()
            else:
                benchmark_values = self.benchmark_data.sum(axis=1)
                benchmark_returns = benchmark_values.pct_change().dropna()
            
            # Align returns
            aligned_returns = returns.align(benchmark_returns, join='inner')
            portfolio_ret = aligned_returns[0]
            benchmark_ret = aligned_returns[1]
            
            if len(portfolio_ret) < 30:
                return 0.0
            
            # Calculate information ratio
            excess_returns = portfolio_ret - benchmark_ret
            
            if excess_returns.std() == 0:
                return 0.0
            
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            return information_ratio
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0.0
    
    def check_risk_thresholds(self, metrics: RealTimeRiskMetrics):
        """Check risk thresholds and generate alerts"""
        try:
            new_alerts = []
            
            for threshold in self.risk_thresholds:
                # Get metric value
                metric_value = getattr(metrics, threshold.metric_name, None)
                if metric_value is None:
                    continue
                
                # Determine alert severity
                severity = None
                threshold_value = None
                
                if threshold.emergency_threshold is not None:
                    if (threshold.metric_name == "sharpe_ratio" and metric_value <= threshold.emergency_threshold) or \
                       (threshold.metric_name != "sharpe_ratio" and metric_value >= threshold.emergency_threshold):
                        severity = AlertSeverity.EMERGENCY
                        threshold_value = threshold.emergency_threshold
                
                if severity is None:
                    if (threshold.metric_name == "sharpe_ratio" and metric_value <= threshold.critical_threshold) or \
                       (threshold.metric_name != "sharpe_ratio" and metric_value >= threshold.critical_threshold):
                        severity = AlertSeverity.CRITICAL
                        threshold_value = threshold.critical_threshold
                
                if severity is None:
                    if (threshold.metric_name == "sharpe_ratio" and metric_value <= threshold.warning_threshold) or \
                       (threshold.metric_name != "sharpe_ratio" and metric_value >= threshold.warning_threshold):
                        severity = AlertSeverity.WARNING
                        threshold_value = threshold.warning_threshold
                
                # Create alert if threshold breached
                if severity is not None:
                    alert_id = f"{threshold.metric_name}_{severity.value}_{int(metrics.timestamp.timestamp())}"
                    
                    # Check if similar alert already exists
                    existing_alert = next(
                        (a for a in self.active_alerts 
                         if a.metric_name == threshold.metric_name and a.severity == severity),
                        None
                    )
                    
                    if existing_alert is None:
                        message = self._generate_alert_message(threshold, metric_value, threshold_value, severity)
                        
                        alert = RiskAlert(
                            alert_id=alert_id,
                            timestamp=metrics.timestamp,
                            severity=severity,
                            metric_type=threshold.metric_type,
                            metric_name=threshold.metric_name,
                            current_value=metric_value,
                            threshold_value=threshold_value,
                            message=message
                        )
                        
                        new_alerts.append(alert)
                        self.active_alerts.append(alert)
                        self.alert_history.append(alert)
                        
                        # Trigger callbacks
                        for callback in self.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                logger.error(f"Error in alert callback: {e}")
            
            # Remove resolved alerts
            self._cleanup_resolved_alerts(metrics)
            
            if new_alerts:
                logger.warning(f"Generated {len(new_alerts)} new risk alerts")
            
        except Exception as e:
            logger.error(f"Error checking risk thresholds: {e}")
    
    def _generate_alert_message(self, threshold: RiskThreshold, current_value: float, 
                              threshold_value: float, severity: AlertSeverity) -> str:
        """Generate alert message"""
        try:
            metric_display = threshold.metric_name.replace('_', ' ').title()
            
            if threshold.is_percentage:
                current_str = f"{current_value:.2%}"
                threshold_str = f"{threshold_value:.2%}"
            else:
                current_str = f"{current_value:.4f}"
                threshold_str = f"{threshold_value:.4f}"
            
            if threshold.metric_name == "sharpe_ratio":
                direction = "below"
            else:
                direction = "above"
            
            message = f"{severity.value.upper()}: {metric_display} is {current_str}, {direction} {severity.value} threshold of {threshold_str}"
            
            return message
            
        except Exception as e:
            logger.error(f"Error generating alert message: {e}")
            return f"Risk threshold breached for {threshold.metric_name}"
    
    def _cleanup_resolved_alerts(self, metrics: RealTimeRiskMetrics):
        """Remove alerts that are no longer active"""
        try:
            resolved_alerts = []
            
            for alert in self.active_alerts:
                # Get current metric value
                current_value = getattr(metrics, alert.metric_name, None)
                if current_value is None:
                    continue
                
                # Find corresponding threshold
                threshold = next(
                    (t for t in self.risk_thresholds if t.metric_name == alert.metric_name),
                    None
                )
                
                if threshold is None:
                    continue
                
                # Check if alert is resolved
                is_resolved = False
                
                if alert.severity == AlertSeverity.WARNING:
                    if threshold.metric_name == "sharpe_ratio":
                        is_resolved = current_value > threshold.warning_threshold
                    else:
                        is_resolved = current_value < threshold.warning_threshold
                elif alert.severity == AlertSeverity.CRITICAL:
                    if threshold.metric_name == "sharpe_ratio":
                        is_resolved = current_value > threshold.critical_threshold
                    else:
                        is_resolved = current_value < threshold.critical_threshold
                elif alert.severity == AlertSeverity.EMERGENCY:
                    if threshold.emergency_threshold is not None:
                        if threshold.metric_name == "sharpe_ratio":
                            is_resolved = current_value > threshold.emergency_threshold
                        else:
                            is_resolved = current_value < threshold.emergency_threshold
                
                if is_resolved:
                    resolved_alerts.append(alert)
            
            # Remove resolved alerts
            for alert in resolved_alerts:
                self.active_alerts.remove(alert)
                logger.info(f"Resolved alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up resolved alerts: {e}")
    
    def start_monitoring(self) -> bool:
        """Start real-time monitoring"""
        try:
            if self.is_monitoring:
                logger.warning("Risk monitoring already running")
                return True
            
            if not self.enable_real_time:
                logger.info("Real-time monitoring disabled")
                return False
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Risk monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting risk monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop real-time monitoring"""
        try:
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            logger.info("Risk monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping risk monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Risk monitoring loop started")
        
        while self.is_monitoring:
            try:
                with self.monitoring_lock:
                    if self.portfolio_data is not None and not self.portfolio_data.empty:
                        # Calculate metrics
                        metrics = self.calculate_real_time_metrics()
                        
                        # Check thresholds
                        self.check_risk_thresholds(metrics)
                        
                        # Update status
                        self.update_status()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
        
        logger.info("Risk monitoring loop stopped")
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get data for risk dashboard"""
        try:
            if not self.risk_metrics_history:
                return {'message': 'No risk metrics available'}
            
            latest_metrics = self.risk_metrics_history[-1]
            
            # Historical data for charts
            history_data = []
            for metrics in list(self.risk_metrics_history)[-100:]:  # Last 100 data points
                history_data.append(metrics.to_dict())
            
            # Alert summary
            alert_summary = {
                'total_active': len(self.active_alerts),
                'by_severity': {
                    'info': len([a for a in self.active_alerts if a.severity == AlertSeverity.INFO]),
                    'warning': len([a for a in self.active_alerts if a.severity == AlertSeverity.WARNING]),
                    'critical': len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]),
                    'emergency': len([a for a in self.active_alerts if a.severity == AlertSeverity.EMERGENCY])
                },
                'recent_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'timestamp': alert.timestamp.isoformat(),
                        'severity': alert.severity.value,
                        'metric_name': alert.metric_name,
                        'message': alert.message
                    }
                    for alert in list(self.alert_history)[-10:]  # Last 10 alerts
                ]
            }
            
            # Risk threshold status
            threshold_status = []
            for threshold in self.risk_thresholds:
                current_value = getattr(latest_metrics, threshold.metric_name, None)
                if current_value is not None:
                    threshold_status.append({
                        'metric_name': threshold.metric_name,
                        'current_value': current_value,
                        'warning_threshold': threshold.warning_threshold,
                        'critical_threshold': threshold.critical_threshold,
                        'emergency_threshold': threshold.emergency_threshold,
                        'is_percentage': threshold.is_percentage
                    })
            
            dashboard_data = {
                'timestamp': latest_metrics.timestamp.isoformat(),
                'current_metrics': latest_metrics.to_dict(),
                'historical_data': history_data,
                'alert_summary': alert_summary,
                'threshold_status': threshold_status,
                'monitoring_status': {
                    'is_monitoring': self.is_monitoring,
                    'monitoring_interval': self.monitoring_interval,
                    'last_update': self.last_update.isoformat() if self.last_update else None
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {'error': str(e)}
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
            
            if alert is None:
                logger.warning(f"Alert {alert_id} not found")
                return False
            
            alert.is_acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def export_risk_data(self, filepath: str) -> bool:
        """Export risk monitoring data"""
        try:
            export_data = {
                'risk_metrics_history': [metrics.to_dict() for metrics in self.risk_metrics_history],
                'alert_history': [
                    {
                        'alert_id': alert.alert_id,
                        'timestamp': alert.timestamp.isoformat(),
                        'severity': alert.severity.value,
                        'metric_type': alert.metric_type.value,
                        'metric_name': alert.metric_name,
                        'current_value': alert.current_value,
                        'threshold_value': alert.threshold_value,
                        'message': alert.message,
                        'is_acknowledged': alert.is_acknowledged,
                        'acknowledged_by': alert.acknowledged_by,
                        'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                    }
                    for alert in self.alert_history
                ],
                'risk_thresholds': [
                    {
                        'metric_type': threshold.metric_type.value,
                        'metric_name': threshold.metric_name,
                        'warning_threshold': threshold.warning_threshold,
                        'critical_threshold': threshold.critical_threshold,
                        'emergency_threshold': threshold.emergency_threshold,
                        'is_percentage': threshold.is_percentage,
                        'is_absolute': threshold.is_absolute
                    }
                    for threshold in self.risk_thresholds
                ],
                'configuration': {
                    'monitoring_interval': self.monitoring_interval,
                    'max_history_size': self.max_history_size,
                    'enable_real_time': self.enable_real_time,
                    'risk_free_rate': self.risk_free_rate
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Risk monitoring data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting risk data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get risk monitor statistics"""
        return {
            'total_metrics_calculated': len(self.risk_metrics_history),
            'active_alerts': len(self.active_alerts),
            'total_alerts_generated': len(self.alert_history),
            'risk_thresholds_configured': len(self.risk_thresholds),
            'alert_callbacks_registered': len(self.alert_callbacks),
            'is_monitoring': self.is_monitoring,
            'monitoring_interval': self.monitoring_interval,
            'max_history_size': self.max_history_size,
            'enable_real_time': self.enable_real_time,
            'last_metrics_timestamp': self.risk_metrics_history[-1].timestamp.isoformat() if self.risk_metrics_history else None,
            'portfolio_data_available': self.portfolio_data is not None,
            'benchmark_data_available': self.benchmark_data is not None,
            'current_positions_count': len(self.current_positions)
        } 