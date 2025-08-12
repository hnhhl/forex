"""
Risk Types Module
Định nghĩa các loại risk và cấu trúc dữ liệu risk management
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np


class RiskLevel(Enum):
    """Mức độ risk"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskType(Enum):
    """Loại risk"""
    MARKET_RISK = "MARKET_RISK"
    CREDIT_RISK = "CREDIT_RISK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    OPERATIONAL_RISK = "OPERATIONAL_RISK"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"
    CORRELATION_RISK = "CORRELATION_RISK"


@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    # Value at Risk
    var_1d: float = 0.0      # 1-day VaR
    var_5d: float = 0.0      # 5-day VaR
    var_10d: float = 0.0     # 10-day VaR
    
    # Conditional Value at Risk
    cvar_1d: float = 0.0     # 1-day CVaR
    cvar_5d: float = 0.0     # 5-day CVaR
    cvar_10d: float = 0.0    # 10-day CVaR
    
    # Risk ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    drawdown_duration: int = 0
    
    # Volatility metrics
    volatility_1d: float = 0.0
    volatility_annualized: float = 0.0
    
    # Beta and correlation
    beta: float = 0.0
    correlation: float = 0.0
    
    # Risk-adjusted returns
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    
    # Timestamp
    calculated_at: datetime = None
    
    def __post_init__(self):
        if self.calculated_at is None:
            self.calculated_at = datetime.now()
    
    def get_risk_level(self) -> RiskLevel:
        """Determine overall risk level"""
        # Simple risk level determination based on VaR and drawdown
        if self.var_1d > 0.05 or self.max_drawdown > 0.20:  # 5% VaR or 20% drawdown
            return RiskLevel.CRITICAL
        elif self.var_1d > 0.03 or self.max_drawdown > 0.15:  # 3% VaR or 15% drawdown
            return RiskLevel.HIGH
        elif self.var_1d > 0.02 or self.max_drawdown > 0.10:  # 2% VaR or 10% drawdown
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'var_1d': self.var_1d,
            'var_5d': self.var_5d,
            'var_10d': self.var_10d,
            'cvar_1d': self.cvar_1d,
            'cvar_5d': self.cvar_5d,
            'cvar_10d': self.cvar_10d,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'volatility_1d': self.volatility_1d,
            'volatility_annualized': self.volatility_annualized,
            'beta': self.beta,
            'correlation': self.correlation,
            'information_ratio': self.information_ratio,
            'treynor_ratio': self.treynor_ratio,
            'risk_level': self.get_risk_level().value,
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None
        }


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    # VaR limits
    max_var_1d: float = 0.02        # 2% daily VaR limit
    max_var_5d: float = 0.05        # 5% weekly VaR limit
    max_var_10d: float = 0.08       # 8% bi-weekly VaR limit
    
    # Drawdown limits
    max_drawdown: float = 0.15      # 15% maximum drawdown
    stop_loss_drawdown: float = 0.20  # 20% stop-loss drawdown
    
    # Position limits
    max_position_size: float = 0.10  # 10% max position size
    max_sector_exposure: float = 0.30  # 30% max sector exposure
    max_correlation: float = 0.70    # 70% max correlation
    
    # Leverage limits
    max_leverage: float = 3.0        # 3:1 maximum leverage
    margin_call_level: float = 0.50  # 50% margin call level
    
    # Daily limits
    max_daily_loss: float = 0.05     # 5% max daily loss
    max_daily_trades: int = 50       # 50 max daily trades
    max_daily_volume: float = 10.0   # 10 lots max daily volume
    
    # Concentration limits
    max_single_position: float = 0.05  # 5% max single position
    max_symbol_exposure: float = 0.20  # 20% max symbol exposure
    
    def validate_limits(self) -> tuple[bool, str]:
        """Validate risk limits configuration"""
        if self.max_var_1d <= 0 or self.max_var_1d > 1:
            return False, "Daily VaR limit must be between 0 and 100%"
        
        if self.max_drawdown <= 0 or self.max_drawdown > 1:
            return False, "Max drawdown must be between 0 and 100%"
        
        if self.max_leverage <= 0:
            return False, "Max leverage must be positive"
        
        if self.max_daily_trades <= 0:
            return False, "Max daily trades must be positive"
        
        return True, "Risk limits are valid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'max_var_1d': self.max_var_1d,
            'max_var_5d': self.max_var_5d,
            'max_var_10d': self.max_var_10d,
            'max_drawdown': self.max_drawdown,
            'stop_loss_drawdown': self.stop_loss_drawdown,
            'max_position_size': self.max_position_size,
            'max_sector_exposure': self.max_sector_exposure,
            'max_correlation': self.max_correlation,
            'max_leverage': self.max_leverage,
            'margin_call_level': self.margin_call_level,
            'max_daily_loss': self.max_daily_loss,
            'max_daily_trades': self.max_daily_trades,
            'max_daily_volume': self.max_daily_volume,
            'max_single_position': self.max_single_position,
            'max_symbol_exposure': self.max_symbol_exposure
        }


@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    risk_type: RiskType
    risk_level: RiskLevel
    message: str
    current_value: float
    limit_value: float
    threshold_breach: float  # How much over the limit (percentage)
    
    # Metadata
    symbol: Optional[str] = None
    position_id: Optional[str] = None
    timestamp: datetime = None
    acknowledged: bool = False
    resolved: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def acknowledge(self):
        """Acknowledge the alert"""
        self.acknowledged = True
    
    def resolve(self):
        """Mark alert as resolved"""
        self.resolved = True
        self.acknowledged = True
    
    def get_severity_score(self) -> float:
        """Get severity score (0-100)"""
        base_score = {
            RiskLevel.LOW: 25,
            RiskLevel.MEDIUM: 50,
            RiskLevel.HIGH: 75,
            RiskLevel.CRITICAL: 100
        }[self.risk_level]
        
        # Adjust based on threshold breach
        breach_multiplier = min(2.0, 1.0 + self.threshold_breach)
        return min(100, base_score * breach_multiplier)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'risk_type': self.risk_type.value,
            'risk_level': self.risk_level.value,
            'message': self.message,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'threshold_breach': self.threshold_breach,
            'symbol': self.symbol,
            'position_id': self.position_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'severity_score': self.get_severity_score()
        }


@dataclass
class PortfolioRisk:
    """Portfolio-level risk assessment"""
    # Portfolio metrics
    total_value: float = 0.0
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    
    # Risk metrics
    portfolio_var: float = 0.0
    portfolio_cvar: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_beta: float = 0.0
    
    # Concentration metrics
    concentration_ratio: float = 0.0  # HHI or similar
    largest_position_weight: float = 0.0
    top_5_concentration: float = 0.0
    
    # Correlation metrics
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    correlation_matrix: Optional[np.ndarray] = None
    
    # Sector/Asset class exposure
    sector_exposure: Dict[str, float] = None
    asset_class_exposure: Dict[str, float] = None
    
    # Leverage metrics
    leverage_ratio: float = 0.0
    margin_utilization: float = 0.0
    
    # Timestamp
    calculated_at: datetime = None
    
    def __post_init__(self):
        if self.calculated_at is None:
            self.calculated_at = datetime.now()
        if self.sector_exposure is None:
            self.sector_exposure = {}
        if self.asset_class_exposure is None:
            self.asset_class_exposure = {}
    
    def get_risk_score(self) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        # Weighted risk score based on multiple factors
        var_score = min(100, (self.portfolio_var / 0.05) * 100)  # 5% VaR = 100 score
        concentration_score = self.concentration_ratio * 100
        leverage_score = min(100, (self.leverage_ratio / 5.0) * 100)  # 5x leverage = 100 score
        correlation_score = self.max_correlation * 100
        
        # Weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # VaR, concentration, leverage, correlation
        scores = [var_score, concentration_score, leverage_score, correlation_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_value': self.total_value,
            'total_exposure': self.total_exposure,
            'net_exposure': self.net_exposure,
            'gross_exposure': self.gross_exposure,
            'portfolio_var': self.portfolio_var,
            'portfolio_cvar': self.portfolio_cvar,
            'portfolio_volatility': self.portfolio_volatility,
            'portfolio_beta': self.portfolio_beta,
            'concentration_ratio': self.concentration_ratio,
            'largest_position_weight': self.largest_position_weight,
            'top_5_concentration': self.top_5_concentration,
            'avg_correlation': self.avg_correlation,
            'max_correlation': self.max_correlation,
            'sector_exposure': self.sector_exposure,
            'asset_class_exposure': self.asset_class_exposure,
            'leverage_ratio': self.leverage_ratio,
            'margin_utilization': self.margin_utilization,
            'risk_score': self.get_risk_score(),
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None
        }


class RiskCalculator:
    """Utility class for risk calculations"""
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        returns_array = np.array(returns)
        return float(np.percentile(returns_array, (1 - confidence_level) * 100))
    
    @staticmethod
    def calculate_cvar(returns: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        returns_array = np.array(returns)
        var = RiskCalculator.calculate_var(returns, confidence_level)
        tail_returns = returns_array[returns_array <= var]
        return float(np.mean(tail_returns)) if len(tail_returns) > 0 else 0.0
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = np.mean(returns_array) - risk_free_rate
        std_dev = np.std(returns_array)
        
        return float(excess_returns / std_dev) if std_dev > 0 else 0.0
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: List[float]) -> tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if len(cumulative_returns) == 0:
            return 0.0, 0
        
        returns_array = np.array(cumulative_returns)
        running_max = np.maximum.accumulate(returns_array)
        drawdown = (returns_array - running_max) / running_max
        
        max_dd = float(np.min(drawdown))
        
        # Calculate drawdown duration
        current_duration = 0
        max_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return abs(max_dd), max_duration 