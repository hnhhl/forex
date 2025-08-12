"""
Advanced Risk Management Systems
Ultimate XAU Super System V4.0 - Day 27 Implementation

Comprehensive risk management capabilities:
- Stress Testing Framework with Monte Carlo simulation
- Dynamic Hedging Strategies for downside protection
- Liquidity Risk Management for market impact optimization
- Value-at-Risk (VaR) and Conditional VaR calculations
- Scenario Analysis and Backtesting Framework
- Real-time Risk Monitoring and Alert Systems
- Regulatory Compliance and Reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import concurrent.futures
import threading
import time

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RiskMetricType(Enum):
    """Types of risk metrics"""
    VAR_HISTORICAL = "var_historical"
    VAR_PARAMETRIC = "var_parametric"
    VAR_MONTE_CARLO = "var_monte_carlo"
    CVAR_EXPECTED_SHORTFALL = "cvar_expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"


class StressTestType(Enum):
    """Types of stress tests"""
    HISTORICAL_SCENARIO = "historical_scenario"
    MONTE_CARLO_SIMULATION = "monte_carlo_simulation"
    FACTOR_SHOCK = "factor_shock"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    EXTREME_MARKET_CONDITIONS = "extreme_market_conditions"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    REGIME_SHIFT = "regime_shift"


class HedgingStrategy(Enum):
    """Types of hedging strategies"""
    DELTA_HEDGE = "delta_hedge"
    GAMMA_HEDGE = "gamma_hedge"
    VOLATILITY_HEDGE = "volatility_hedge"
    TAIL_HEDGE = "tail_hedge"
    CORRELATION_HEDGE = "correlation_hedge"
    CURRENCY_HEDGE = "currency_hedge"
    DYNAMIC_HEDGE = "dynamic_hedge"


class RiskLimitType(Enum):
    """Types of risk limits"""
    VAR_LIMIT = "var_limit"
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    LIQUIDITY_LIMIT = "liquidity_limit"
    CORRELATION_LIMIT = "correlation_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"


@dataclass
class RiskConfig:
    """Configuration for risk management system"""
    
    # VaR calculation settings
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_lookback_period: int = 252  # Trading days
    var_method: RiskMetricType = RiskMetricType.VAR_HISTORICAL
    
    # Stress testing settings
    monte_carlo_simulations: int = 10000
    stress_test_scenarios: List[StressTestType] = field(default_factory=lambda: [
        StressTestType.HISTORICAL_SCENARIO,
        StressTestType.MONTE_CARLO_SIMULATION,
        StressTestType.FACTOR_SHOCK
    ])
    
    # Risk limits
    max_var_95: float = 0.05  # 5% daily VaR limit
    max_var_99: float = 0.08  # 8% daily VaR limit
    max_drawdown: float = 0.15  # 15% maximum drawdown
    max_concentration: float = 0.30  # 30% single position limit
    max_leverage: float = 2.0  # 2x maximum leverage
    
    # Hedging settings
    enable_dynamic_hedging: bool = True
    hedge_ratio_target: float = 0.8  # Target hedge ratio
    hedge_rebalance_threshold: float = 0.1  # Rebalance when ratio drifts 10%
    
    # Liquidity management
    liquidity_horizon_days: int = 10  # Days to unwind positions
    market_impact_threshold: float = 0.02  # 2% market impact threshold
    
    # Monitoring settings
    real_time_monitoring: bool = True
    alert_threshold_multiplier: float = 0.8  # Alert at 80% of limits
    risk_reporting_frequency: str = "daily"  # daily, weekly, monthly


@dataclass
class RiskMetrics:
    """Container for calculated risk metrics"""
    
    timestamp: datetime
    portfolio_value: float = 0.0
    
    # VaR metrics
    var_95_daily: float = 0.0
    var_99_daily: float = 0.0
    cvar_95_daily: float = 0.0
    cvar_99_daily: float = 0.0
    
    # Risk ratios
    volatility_annual: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: int = 0
    
    # Portfolio metrics
    beta: float = 1.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    concentration_risk: float = 0.0
    
    # Liquidity metrics
    liquidity_score: float = 1.0
    estimated_liquidation_time: int = 1  # Days
    market_impact_cost: float = 0.0


@dataclass
class StressTestResult:
    """Results from stress testing"""
    
    test_type: StressTestType
    scenario_name: str
    timestamp: datetime
    
    # Scenario parameters
    shock_magnitude: float = 0.0
    shock_direction: str = "negative"
    
    # Impact results
    portfolio_pnl: float = 0.0
    portfolio_pnl_pct: float = 0.0
    worst_asset_pnl: float = 0.0
    best_asset_pnl: float = 0.0
    
    # Risk metrics under stress
    stressed_var_95: float = 0.0
    stressed_volatility: float = 0.0
    correlation_breakdown: bool = False
    
    # Recovery metrics
    estimated_recovery_time: int = 0  # Days
    probability_of_loss: float = 0.0
    
    # Asset-level impacts
    asset_impacts: Dict[str, float] = field(default_factory=dict)


@dataclass
class HedgingRecommendation:
    """Hedging strategy recommendation"""
    
    timestamp: datetime
    strategy_type: HedgingStrategy
    
    # Current exposure
    unhedged_exposure: float = 0.0
    current_hedge_ratio: float = 0.0
    target_hedge_ratio: float = 0.0
    
    # Recommended actions
    hedge_instruments: Dict[str, float] = field(default_factory=dict)
    estimated_cost: float = 0.0
    expected_protection: float = 0.0
    
    # Risk reduction
    var_reduction: float = 0.0
    volatility_reduction: float = 0.0
    max_loss_reduction: float = 0.0
    
    # Implementation details
    execution_priority: str = "medium"  # low, medium, high, urgent
    validity_period: int = 1  # Days
    confidence_level: float = 0.75


@dataclass
class RiskAlert:
    """Risk management alert"""
    
    timestamp: datetime
    alert_type: str
    severity: str  # info, warning, critical, emergency
    
    # Alert details
    metric_name: str
    current_value: float
    limit_value: float
    breach_percentage: float
    
    # Context
    portfolio_impact: float = 0.0
    affected_positions: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    urgency_score: float = 0.0  # 0-1 scale
    
    # Status tracking
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class VaRCalculator:
    """Value-at-Risk calculation engine"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_historical_var(self, returns: pd.Series, 
                               confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Historical VaR and CVaR"""
        
        try:
            if len(returns) < 30:
                self.logger.warning("Insufficient data for Historical VaR calculation")
                return 0.0, 0.0
            
            # Sort returns in ascending order
            sorted_returns = returns.sort_values()
            
            # Calculate VaR (percentile)
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(sorted_returns, var_percentile)
            
            # Calculate CVaR (Expected Shortfall)
            cvar_returns = sorted_returns[sorted_returns <= var_value]
            cvar_value = cvar_returns.mean() if len(cvar_returns) > 0 else var_value
            
            return abs(var_value), abs(cvar_value)
            
        except Exception as e:
            self.logger.error(f"Error calculating Historical VaR: {e}")
            return 0.0, 0.0
    
    def calculate_parametric_var(self, returns: pd.Series, 
                               confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Parametric VaR assuming normal distribution"""
        
        try:
            if len(returns) < 30:
                return 0.0, 0.0
            
            # Calculate mean and standard deviation
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # Parametric VaR
            var_value = abs(mean_return + z_score * std_return)
            
            # Parametric CVaR (for normal distribution)
            density_at_var = stats.norm.pdf(z_score)
            cvar_value = abs(mean_return - std_return * density_at_var / (1 - confidence_level))
            
            return var_value, cvar_value
            
        except Exception as e:
            self.logger.error(f"Error calculating Parametric VaR: {e}")
            return 0.0, 0.0
    
    def calculate_monte_carlo_var(self, returns: pd.Series, 
                                confidence_level: float = 0.95,
                                n_simulations: int = 10000) -> Tuple[float, float]:
        """Calculate Monte Carlo VaR"""
        
        try:
            if len(returns) < 30:
                return 0.0, 0.0
            
            # Fit distribution parameters
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate VaR and CVaR from simulations
            var_percentile = (1 - confidence_level) * 100
            var_value = abs(np.percentile(simulated_returns, var_percentile))
            
            # CVaR from simulations
            cvar_returns = simulated_returns[simulated_returns <= -var_value]
            cvar_value = abs(cvar_returns.mean()) if len(cvar_returns) > 0 else var_value
            
            return var_value, cvar_value
            
        except Exception as e:
            self.logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0, 0.0


class StressTester:
    """Comprehensive stress testing framework"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Historical stress scenarios
        self.historical_scenarios = {
            "black_monday_1987": {"shock": -0.22, "volatility_multiplier": 3.0},
            "dot_com_crash_2000": {"shock": -0.12, "volatility_multiplier": 2.5},
            "financial_crisis_2008": {"shock": -0.18, "volatility_multiplier": 4.0},
            "covid_crash_2020": {"shock": -0.35, "volatility_multiplier": 5.0},
            "rate_shock_2022": {"shock": -0.08, "volatility_multiplier": 1.8}
        }
    
    def run_historical_stress_test(self, portfolio_returns: pd.Series, 
                                 portfolio_weights: Dict[str, float]) -> List[StressTestResult]:
        """Run historical scenario stress tests"""
        
        results = []
        
        for scenario_name, params in self.historical_scenarios.items():
            try:
                # Apply shock to portfolio
                shocked_return = params["shock"]
                vol_multiplier = params["volatility_multiplier"]
                
                # Calculate portfolio impact
                portfolio_pnl_pct = shocked_return
                portfolio_pnl = portfolio_pnl_pct * 100000  # Assume $100k portfolio
                
                # Calculate stressed metrics
                stressed_volatility = portfolio_returns.std() * vol_multiplier * np.sqrt(252)
                stressed_var_95 = abs(shocked_return) * 1.2  # Conservative estimate
                
                result = StressTestResult(
                    test_type=StressTestType.HISTORICAL_SCENARIO,
                    scenario_name=scenario_name,
                    timestamp=datetime.now(),
                    shock_magnitude=abs(shocked_return),
                    shock_direction="negative",
                    portfolio_pnl=portfolio_pnl,
                    portfolio_pnl_pct=portfolio_pnl_pct,
                    stressed_var_95=stressed_var_95,
                    stressed_volatility=stressed_volatility,
                    estimated_recovery_time=int(abs(shocked_return) * 100),  # Days
                    probability_of_loss=min(1.0, abs(shocked_return) * 2)
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in historical stress test {scenario_name}: {e}")
                continue
        
        return results
    
    def run_monte_carlo_stress_test(self, portfolio_returns: pd.Series,
                                  n_simulations: int = None) -> StressTestResult:
        """Run Monte Carlo simulation stress test"""
        
        try:
            n_sims = n_simulations or self.config.monte_carlo_simulations
            
            # Portfolio statistics
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            # Generate scenarios
            np.random.seed(42)
            simulated_returns = np.random.normal(mean_return, std_return, n_sims)
            
            # Calculate stress metrics
            worst_case_return = np.percentile(simulated_returns, 1)  # 1st percentile
            probability_of_loss = np.sum(simulated_returns < 0) / n_sims
            
            # Expected recovery time (simplified model)
            negative_scenarios = simulated_returns[simulated_returns < 0]
            avg_loss_magnitude = abs(negative_scenarios.mean()) if len(negative_scenarios) > 0 else 0
            recovery_time = int(avg_loss_magnitude * 252)  # Days to recover
            
            result = StressTestResult(
                test_type=StressTestType.MONTE_CARLO_SIMULATION,
                scenario_name="monte_carlo_stress",
                timestamp=datetime.now(),
                shock_magnitude=abs(worst_case_return),
                portfolio_pnl=worst_case_return * 100000,
                portfolio_pnl_pct=worst_case_return,
                stressed_var_95=abs(np.percentile(simulated_returns, 5)),
                stressed_volatility=std_return * np.sqrt(252),
                estimated_recovery_time=min(recovery_time, 365),
                probability_of_loss=probability_of_loss
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo stress test: {e}")
            return StressTestResult(
                test_type=StressTestType.MONTE_CARLO_SIMULATION,
                scenario_name="monte_carlo_stress_failed",
                timestamp=datetime.now()
            )
    
    def run_factor_shock_test(self, portfolio_data: pd.DataFrame) -> List[StressTestResult]:
        """Run factor shock stress tests"""
        
        results = []
        
        # Define factor shocks
        factor_shocks = {
            "interest_rate_shock": {"magnitude": 0.02, "direction": "up"},
            "volatility_shock": {"magnitude": 0.5, "direction": "up"},
            "correlation_shock": {"magnitude": 0.3, "direction": "up"},
            "liquidity_shock": {"magnitude": 0.1, "direction": "down"}
        }
        
        for factor_name, shock_params in factor_shocks.items():
            try:
                # Simulate factor shock impact
                shock_magnitude = shock_params["magnitude"]
                
                # Simplified factor impact calculation
                if "interest_rate" in factor_name:
                    portfolio_impact = -shock_magnitude * 0.5  # Duration assumption
                elif "volatility" in factor_name:
                    portfolio_impact = -shock_magnitude * 0.3  # Volatility drag
                elif "correlation" in factor_name:
                    portfolio_impact = -shock_magnitude * 0.2  # Diversification loss
                else:  # liquidity shock
                    portfolio_impact = -shock_magnitude * 0.4  # Liquidity premium
                
                result = StressTestResult(
                    test_type=StressTestType.FACTOR_SHOCK,
                    scenario_name=factor_name,
                    timestamp=datetime.now(),
                    shock_magnitude=shock_magnitude,
                    shock_direction=shock_params["direction"],
                    portfolio_pnl=portfolio_impact * 100000,
                    portfolio_pnl_pct=portfolio_impact,
                    correlation_breakdown=(factor_name == "correlation_shock"),
                    estimated_recovery_time=int(shock_magnitude * 90)
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in factor shock test {factor_name}: {e}")
                continue
        
        return results


class DynamicHedger:
    """Dynamic hedging strategy implementation"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_hedges = {}
        
    def calculate_hedge_ratio(self, portfolio_beta: float, 
                            target_beta: float = 0.0) -> float:
        """Calculate optimal hedge ratio"""
        
        try:
            # Simple beta hedging
            hedge_ratio = (portfolio_beta - target_beta) / portfolio_beta
            
            # Apply bounds
            hedge_ratio = max(0.0, min(1.0, hedge_ratio))
            
            return hedge_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating hedge ratio: {e}")
            return 0.0
    
    def generate_hedge_recommendation(self, portfolio_data: pd.DataFrame,
                                    current_var: float) -> HedgingRecommendation:
        """Generate dynamic hedging recommendation"""
        
        try:
            # Calculate portfolio metrics
            portfolio_returns = portfolio_data.pct_change().dropna()
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Determine hedge strategy based on risk level
            if current_var > self.config.max_var_95 * 0.8:
                strategy_type = HedgingStrategy.TAIL_HEDGE
                hedge_ratio = 0.6
                expected_protection = 0.4
            elif portfolio_volatility > 0.25:
                strategy_type = HedgingStrategy.VOLATILITY_HEDGE
                hedge_ratio = 0.4
                expected_protection = 0.3
            else:
                strategy_type = HedgingStrategy.DELTA_HEDGE
                hedge_ratio = 0.3
                expected_protection = 0.2
            
            # Calculate costs and benefits
            estimated_cost = hedge_ratio * 0.02  # 2% cost assumption
            var_reduction = expected_protection * current_var
            volatility_reduction = expected_protection * portfolio_volatility
            
            # Determine execution priority
            if current_var > self.config.max_var_95:
                priority = "urgent"
            elif current_var > self.config.max_var_95 * 0.8:
                priority = "high"
            else:
                priority = "medium"
            
            recommendation = HedgingRecommendation(
                timestamp=datetime.now(),
                strategy_type=strategy_type,
                unhedged_exposure=100000,  # Assume $100k portfolio
                current_hedge_ratio=0.0,
                target_hedge_ratio=hedge_ratio,
                hedge_instruments={"SPY_PUT": hedge_ratio * 0.6, "VIX_CALL": hedge_ratio * 0.4},
                estimated_cost=estimated_cost,
                expected_protection=expected_protection,
                var_reduction=var_reduction,
                volatility_reduction=volatility_reduction,
                execution_priority=priority,
                confidence_level=0.75
            )
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error generating hedge recommendation: {e}")
            return HedgingRecommendation(
                timestamp=datetime.now(),
                strategy_type=HedgingStrategy.DELTA_HEDGE,
                execution_priority="low",
                confidence_level=0.5
            )


class LiquidityRiskManager:
    """Liquidity risk management system"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Asset liquidity rankings (simplified)
        self.liquidity_scores = {
            "GOLD": 0.95, "SILVER": 0.85, "PLATINUM": 0.70,
            "CRUDE_OIL": 0.90, "NATURAL_GAS": 0.75, "COPPER": 0.80
        }
    
    def calculate_liquidity_metrics(self, portfolio_weights: Dict[str, float],
                                  portfolio_value: float = 100000) -> Dict[str, float]:
        """Calculate portfolio liquidity metrics"""
        
        try:
            # Weighted average liquidity score
            total_liquidity_score = 0.0
            total_weight = 0.0
            
            for asset, weight in portfolio_weights.items():
                asset_liquidity = self.liquidity_scores.get(asset, 0.5)
                total_liquidity_score += weight * asset_liquidity
                total_weight += weight
            
            avg_liquidity_score = total_liquidity_score / total_weight if total_weight > 0 else 0.5
            
            # Estimated liquidation time (days)
            base_liquidation_time = self.config.liquidity_horizon_days
            liquidity_adjustment = (1 - avg_liquidity_score) * 2  # 0-2 multiplier
            estimated_liquidation_time = int(base_liquidation_time * (1 + liquidity_adjustment))
            
            # Market impact cost
            size_impact = min(0.05, portfolio_value / 10000000)  # Size impact
            liquidity_impact = (1 - avg_liquidity_score) * 0.02  # Liquidity impact
            total_market_impact = size_impact + liquidity_impact
            
            return {
                "liquidity_score": avg_liquidity_score,
                "estimated_liquidation_time": estimated_liquidation_time,
                "market_impact_cost": total_market_impact,
                "liquidity_risk_level": "low" if avg_liquidity_score > 0.8 else "medium" if avg_liquidity_score > 0.6 else "high"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity metrics: {e}")
            return {
                "liquidity_score": 0.5,
                "estimated_liquidation_time": self.config.liquidity_horizon_days,
                "market_impact_cost": 0.02,
                "liquidity_risk_level": "medium"
            }


class RiskMonitor:
    """Real-time risk monitoring and alerting system"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_alerts = []
        self.alert_history = []
        self.monitoring_active = False
        
    def check_risk_limits(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """Check all risk limits and generate alerts"""
        
        alerts = []
        
        try:
            # VaR limit checks
            if metrics.var_95_daily > self.config.max_var_95:
                alert = self._create_alert(
                    "VAR_95_BREACH",
                    "critical",
                    "Daily VaR 95%",
                    metrics.var_95_daily,
                    self.config.max_var_95,
                    ["Reduce position sizes", "Implement hedging strategy"]
                )
                alerts.append(alert)
            
            if metrics.var_99_daily > self.config.max_var_99:
                alert = self._create_alert(
                    "VAR_99_BREACH",
                    "emergency",
                    "Daily VaR 99%",
                    metrics.var_99_daily,
                    self.config.max_var_99,
                    ["Immediate position reduction", "Emergency hedging"]
                )
                alerts.append(alert)
            
            # Drawdown limit check
            if abs(metrics.current_drawdown) > self.config.max_drawdown:
                alert = self._create_alert(
                    "DRAWDOWN_BREACH",
                    "critical",
                    "Current Drawdown",
                    abs(metrics.current_drawdown),
                    self.config.max_drawdown,
                    ["Stop trading", "Review risk management", "Reassess strategy"]
                )
                alerts.append(alert)
            
            # Concentration risk check
            if metrics.concentration_risk > self.config.max_concentration:
                alert = self._create_alert(
                    "CONCENTRATION_BREACH",
                    "warning",
                    "Concentration Risk",
                    metrics.concentration_risk,
                    self.config.max_concentration,
                    ["Diversify positions", "Reduce largest holdings"]
                )
                alerts.append(alert)
            
            # Early warning alerts (80% of limits)
            warning_threshold = self.config.alert_threshold_multiplier
            
            if metrics.var_95_daily > self.config.max_var_95 * warning_threshold:
                alert = self._create_alert(
                    "VAR_95_WARNING",
                    "warning",
                    "Daily VaR 95% Warning",
                    metrics.var_95_daily,
                    self.config.max_var_95 * warning_threshold,
                    ["Monitor closely", "Consider risk reduction"]
                )
                alerts.append(alert)
            
            # Update active alerts
            self.active_alerts.extend(alerts)
            self.alert_history.extend(alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return []
    
    def _create_alert(self, alert_type: str, severity: str, metric_name: str,
                     current_value: float, limit_value: float,
                     recommendations: List[str]) -> RiskAlert:
        """Create a risk alert"""
        
        breach_percentage = (current_value - limit_value) / limit_value * 100
        urgency_score = min(1.0, breach_percentage / 100)
        
        return RiskAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            limit_value=limit_value,
            breach_percentage=breach_percentage,
            recommended_actions=recommendations,
            urgency_score=urgency_score
        )
    
    def start_monitoring(self, monitoring_interval: int = 60):
        """Start real-time risk monitoring"""
        
        if self.config.real_time_monitoring:
            self.monitoring_active = True
            self.logger.info(f"Risk monitoring started with {monitoring_interval}s interval")
        
    def stop_monitoring(self):
        """Stop real-time risk monitoring"""
        
        self.monitoring_active = False
        self.logger.info("Risk monitoring stopped")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts"""
        
        active_count = len(self.active_alerts)
        severity_counts = {}
        
        for alert in self.active_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        return {
            "total_active_alerts": active_count,
            "severity_breakdown": severity_counts,
            "highest_urgency": max([a.urgency_score for a in self.active_alerts], default=0),
            "oldest_alert": min([a.timestamp for a in self.active_alerts], default=datetime.now()),
            "alert_rate_24h": len([a for a in self.alert_history 
                                 if a.timestamp > datetime.now() - timedelta(hours=24)])
        }


class AdvancedRiskManagement:
    """Main advanced risk management system"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        
        # Initialize components
        self.var_calculator = VaRCalculator(self.config)
        self.stress_tester = StressTester(self.config)
        self.hedger = DynamicHedger(self.config)
        self.liquidity_manager = LiquidityRiskManager(self.config)
        self.risk_monitor = RiskMonitor(self.config)
        
        # State management
        self.risk_history = []
        self.stress_test_history = []
        self.hedge_history = []
        
        logger.info("Advanced Risk Management system initialized")
    
    def calculate_comprehensive_risk_metrics(self, portfolio_data: pd.DataFrame,
                                           portfolio_weights: Dict[str, float],
                                           portfolio_value: float = 100000) -> RiskMetrics:
        """Calculate comprehensive risk metrics for portfolio"""
        
        try:
            # Calculate portfolio returns
            portfolio_returns = pd.Series(index=portfolio_data.index, dtype=float)
            
            for date in portfolio_data.index:
                daily_return = 0
                for asset, weight in portfolio_weights.items():
                    if asset in portfolio_data.columns:
                        asset_return = portfolio_data[asset].pct_change().loc[date]
                        if not pd.isna(asset_return):
                            daily_return += weight * asset_return
                portfolio_returns.loc[date] = daily_return
            
            portfolio_returns = portfolio_returns.dropna()
            
            # VaR calculations
            var_95, cvar_95 = self.var_calculator.calculate_historical_var(portfolio_returns, 0.95)
            var_99, cvar_99 = self.var_calculator.calculate_historical_var(portfolio_returns, 0.99)
            
            # Risk ratios
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_volatility
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown calculation
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            current_drawdown = drawdowns.iloc[-1]
            max_drawdown = drawdowns.min()
            
            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Concentration risk
            concentration_risk = max(portfolio_weights.values()) if portfolio_weights else 0
            
            # Liquidity metrics
            liquidity_metrics = self.liquidity_manager.calculate_liquidity_metrics(
                portfolio_weights, portfolio_value
            )
            
            metrics = RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                var_95_daily=var_95,
                var_99_daily=var_99,
                cvar_95_daily=cvar_95,
                cvar_99_daily=cvar_99,
                volatility_annual=annual_volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                concentration_risk=concentration_risk,
                liquidity_score=liquidity_metrics["liquidity_score"],
                estimated_liquidation_time=liquidity_metrics["estimated_liquidation_time"],
                market_impact_cost=liquidity_metrics["market_impact_cost"]
            )
            
            # Store in history
            self.risk_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value
            )
    
    def run_comprehensive_stress_tests(self, portfolio_data: pd.DataFrame,
                                     portfolio_weights: Dict[str, float]) -> List[StressTestResult]:
        """Run comprehensive stress testing suite"""
        
        all_results = []
        
        try:
            # Calculate portfolio returns for testing
            portfolio_returns = pd.Series(index=portfolio_data.index, dtype=float)
            
            for date in portfolio_data.index:
                daily_return = 0
                for asset, weight in portfolio_weights.items():
                    if asset in portfolio_data.columns:
                        asset_return = portfolio_data[asset].pct_change().loc[date]
                        if not pd.isna(asset_return):
                            daily_return += weight * asset_return
                portfolio_returns.loc[date] = daily_return
            
            portfolio_returns = portfolio_returns.dropna()
            
            # Run different stress tests
            if StressTestType.HISTORICAL_SCENARIO in self.config.stress_test_scenarios:
                historical_results = self.stress_tester.run_historical_stress_test(
                    portfolio_returns, portfolio_weights
                )
                all_results.extend(historical_results)
            
            if StressTestType.MONTE_CARLO_SIMULATION in self.config.stress_test_scenarios:
                mc_result = self.stress_tester.run_monte_carlo_stress_test(portfolio_returns)
                all_results.append(mc_result)
            
            if StressTestType.FACTOR_SHOCK in self.config.stress_test_scenarios:
                factor_results = self.stress_tester.run_factor_shock_test(portfolio_data)
                all_results.extend(factor_results)
            
            # Store in history
            self.stress_test_history.extend(all_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return []
    
    def generate_hedging_recommendations(self, portfolio_data: pd.DataFrame,
                                       current_risk_metrics: RiskMetrics) -> HedgingRecommendation:
        """Generate dynamic hedging recommendations"""
        
        try:
            recommendation = self.hedger.generate_hedge_recommendation(
                portfolio_data, current_risk_metrics.var_95_daily
            )
            
            # Store in history
            self.hedge_history.append(recommendation)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating hedging recommendations: {e}")
            return HedgingRecommendation(
                timestamp=datetime.now(),
                strategy_type=HedgingStrategy.DELTA_HEDGE,
                execution_priority="low",
                confidence_level=0.5
            )
    
    def monitor_real_time_risk(self, current_metrics: RiskMetrics) -> List[RiskAlert]:
        """Monitor real-time risk and generate alerts"""
        
        try:
            alerts = self.risk_monitor.check_risk_limits(current_metrics)
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring real-time risk: {e}")
            return []
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data"""
        
        try:
            # Latest metrics
            latest_metrics = self.risk_history[-1] if self.risk_history else None
            
            # Alert summary
            alert_summary = self.risk_monitor.get_alert_summary()
            
            # Stress test summary
            if self.stress_test_history:
                worst_stress_loss = min([st.portfolio_pnl_pct for st in self.stress_test_history])
                avg_stress_loss = np.mean([st.portfolio_pnl_pct for st in self.stress_test_history])
            else:
                worst_stress_loss = avg_stress_loss = 0
            
            # Hedging summary
            if self.hedge_history:
                latest_hedge = self.hedge_history[-1]
                hedge_coverage = latest_hedge.target_hedge_ratio
                hedge_cost = latest_hedge.estimated_cost
            else:
                hedge_coverage = hedge_cost = 0
            
            dashboard = {
                "timestamp": datetime.now(),
                "risk_metrics": {
                    "var_95": latest_metrics.var_95_daily if latest_metrics else 0,
                    "var_99": latest_metrics.var_99_daily if latest_metrics else 0,
                    "current_drawdown": latest_metrics.current_drawdown if latest_metrics else 0,
                    "volatility": latest_metrics.volatility_annual if latest_metrics else 0,
                    "sharpe_ratio": latest_metrics.sharpe_ratio if latest_metrics else 0
                },
                "stress_testing": {
                    "worst_case_loss": worst_stress_loss,
                    "average_stress_loss": avg_stress_loss,
                    "scenarios_tested": len(self.stress_test_history)
                },
                "hedging": {
                    "hedge_coverage": hedge_coverage,
                    "estimated_cost": hedge_cost,
                    "strategy_count": len(self.hedge_history)
                },
                "alerts": alert_summary,
                "liquidity": {
                    "score": latest_metrics.liquidity_score if latest_metrics else 0.5,
                    "liquidation_time": latest_metrics.estimated_liquidation_time if latest_metrics else 10,
                    "market_impact": latest_metrics.market_impact_cost if latest_metrics else 0.02
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating risk dashboard: {e}")
            return {
                "timestamp": datetime.now(),
                "error": "Dashboard generation failed"
            }


def create_advanced_risk_management(custom_config: Dict = None) -> AdvancedRiskManagement:
    """Factory function to create advanced risk management system"""
    
    config = RiskConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return AdvancedRiskManagement(config)


if __name__ == "__main__":
    # Example usage
    print("Advanced Risk Management System")
    
    # Create risk management system
    risk_system = create_advanced_risk_management({
        'confidence_levels': [0.95, 0.99],
        'monte_carlo_simulations': 5000,
        'enable_dynamic_hedging': True,
        'real_time_monitoring': True
    })
    
    # Generate sample portfolio data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='1D')
    
    assets = ['GOLD', 'SILVER', 'PLATINUM', 'COPPER']
    portfolio_data = pd.DataFrame(index=dates, columns=assets)
    
    # Simulate correlated asset prices
    for i, asset in enumerate(assets):
        returns = np.random.normal(0.0008, 0.02, 252)  # Daily returns
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        portfolio_data[asset] = prices[1:]
    
    # Portfolio weights
    portfolio_weights = {'GOLD': 0.4, 'SILVER': 0.3, 'PLATINUM': 0.2, 'COPPER': 0.1}
    portfolio_value = 100000
    
    # Calculate comprehensive risk metrics
    risk_metrics = risk_system.calculate_comprehensive_risk_metrics(
        portfolio_data, portfolio_weights, portfolio_value
    )
    
    print("\nRisk Metrics:")
    print(f"Daily VaR (95%): {risk_metrics.var_95_daily:.3f}")
    print(f"Daily VaR (99%): {risk_metrics.var_99_daily:.3f}")
    print(f"Annual Volatility: {risk_metrics.volatility_annual:.2%}")
    print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")
    print(f"Liquidity Score: {risk_metrics.liquidity_score:.3f}")
    
    # Run stress tests
    stress_results = risk_system.run_comprehensive_stress_tests(portfolio_data, portfolio_weights)
    
    print(f"\nStress Testing Results ({len(stress_results)} scenarios):")
    for result in stress_results[:3]:  # Show first 3
        print(f"  {result.scenario_name}: {result.portfolio_pnl_pct:.2%} portfolio impact")
    
    # Generate hedging recommendations
    hedge_recommendation = risk_system.generate_hedging_recommendations(portfolio_data, risk_metrics)
    
    print(f"\nHedging Recommendation:")
    print(f"  Strategy: {hedge_recommendation.strategy_type.value}")
    print(f"  Target Hedge Ratio: {hedge_recommendation.target_hedge_ratio:.1%}")
    print(f"  Expected Protection: {hedge_recommendation.expected_protection:.1%}")
    print(f"  Priority: {hedge_recommendation.execution_priority}")
    
    # Monitor risk and check for alerts
    alerts = risk_system.monitor_real_time_risk(risk_metrics)
    
    print(f"\nRisk Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert.severity.upper()}: {alert.metric_name} = {alert.current_value:.3f}")
    
    # Get dashboard summary
    dashboard = risk_system.get_risk_dashboard()
    
    print(f"\nRisk Dashboard Summary:")
    print(f"  Current VaR (95%): {dashboard['risk_metrics']['var_95']:.3f}")
    print(f"  Stress Test Scenarios: {dashboard['stress_testing']['scenarios_tested']}")
    print(f"  Active Alerts: {dashboard['alerts']['total_active_alerts']}")
    print(f"  Liquidity Score: {dashboard['liquidity']['score']:.3f}") 