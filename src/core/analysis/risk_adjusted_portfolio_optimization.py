"""
Risk-Adjusted Portfolio Optimization System
Ultimate XAU Super System V4.0 - Day 26 Implementation

Advanced portfolio optimization capabilities:
- Dynamic risk-adjusted portfolio optimization
- Multi-asset correlation analysis with regime context
- Real-time portfolio rebalancing recommendations
- Kelly Criterion integration for optimal position sizing
- Sharpe ratio maximization and drawdown minimization
- Regime-aware asset allocation strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
from scipy.optimize import minimize, differential_evolution
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    SHARPE_RATIO = "sharpe_ratio"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    KELLY_OPTIMAL = "kelly_optimal"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_DRAWDOWN = "min_drawdown"


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    DYNAMIC = "dynamic"  # Based on regime changes
    THRESHOLD = "threshold"  # Based on drift thresholds


class RiskConstraintType(Enum):
    """Types of risk constraints"""
    MAX_VOLATILITY = "max_volatility"
    MAX_DRAWDOWN = "max_drawdown"
    VAR_LIMIT = "var_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    SECTOR_LIMIT = "sector_limit"
    CORRELATION_LIMIT = "correlation_limit"


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization"""
    
    # Optimization settings
    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    lookback_period: int = 252  # Trading days for covariance estimation
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    
    # Risk constraints
    max_volatility: float = 0.20  # 20% annual volatility limit
    max_drawdown: float = 0.15    # 15% maximum drawdown
    max_concentration: float = 0.30  # 30% maximum single asset weight
    min_weight: float = 0.01      # 1% minimum weight
    max_weight: float = 0.50      # 50% maximum weight
    
    # Kelly Criterion settings
    enable_kelly_sizing: bool = True
    kelly_fraction: float = 0.25  # Kelly fraction for safety
    confidence_level: float = 0.95
    
    # Regime integration
    enable_regime_awareness: bool = True
    regime_weight_adjustment: float = 0.20  # 20% regime-based adjustment
    
    # Advanced settings
    enable_transaction_costs: bool = True
    transaction_cost_bps: float = 5.0  # 5 basis points
    enable_dynamic_rebalancing: bool = True
    drift_threshold: float = 0.05  # 5% drift threshold
    
    # Covariance estimation
    covariance_method: str = "ledoit_wolf"  # ledoit_wolf, sample, shrinkage
    alpha_decay: float = 0.94  # Exponential decay for returns
    
    # Performance attribution
    enable_attribution: bool = True
    benchmark_weights: Optional[Dict[str, float]] = None


@dataclass
class AssetMetrics:
    """Metrics for individual assets"""
    
    symbol: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    
    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Market metrics
    beta: float = 1.0
    correlation_to_market: float = 0.0
    
    # Regime-specific metrics
    regime_sensitivity: Dict[str, float] = field(default_factory=dict)
    regime_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioWeights:
    """Portfolio weight allocation"""
    
    timestamp: datetime
    weights: Dict[str, float]
    
    # Optimization results
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk metrics
    portfolio_var: float = 0.0
    portfolio_cvar: float = 0.0
    diversification_ratio: float = 0.0
    
    # Regime context
    current_regime: Optional[str] = None
    regime_confidence: float = 0.0
    
    # Rebalancing info
    turnover: float = 0.0
    transaction_costs: float = 0.0
    rebalance_trigger: str = "scheduled"


@dataclass
class PortfolioPerformance:
    """Portfolio performance metrics"""
    
    period_start: datetime
    period_end: datetime
    
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    drawdown_duration: int = 0
    recovery_time: int = 0
    
    # Risk-adjusted metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    omega_ratio: float = 0.0
    
    # Attribution metrics
    asset_contribution: Dict[str, float] = field(default_factory=dict)
    regime_performance: Dict[str, float] = field(default_factory=dict)
    
    # Transaction metrics
    total_turnover: float = 0.0
    total_transaction_costs: float = 0.0
    net_return: float = 0.0


class PortfolioOptimizer:
    """Core portfolio optimization engine"""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization state
        self.current_weights = {}
        self.covariance_matrix = None
        self.expected_returns = None
        self.asset_metrics = {}
        
        # Performance tracking
        self.performance_history = []
        self.rebalance_history = []
        
    def calculate_asset_metrics(self, price_data: pd.DataFrame) -> Dict[str, AssetMetrics]:
        """Calculate comprehensive metrics for each asset"""
        
        metrics = {}
        
        for symbol in price_data.columns:
            try:
                prices = price_data[symbol].dropna()
                returns = prices.pct_change().dropna()
                
                if len(returns) < 30:  # Minimum data requirement
                    continue
                
                # Basic metrics
                expected_return = returns.mean() * 252  # Annualized
                volatility = returns.std() * np.sqrt(252)  # Annualized
                sharpe_ratio = expected_return / volatility if volatility > 0 else 0
                
                # Risk metrics
                var_95 = np.percentile(returns, 5)
                cvar_95 = returns[returns <= var_95].mean()
                
                # Drawdown calculation
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = drawdowns.min()
                
                # Calmar ratio
                calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown != 0 else 0
                
                metrics[symbol] = AssetMetrics(
                    symbol=symbol,
                    expected_return=expected_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    max_drawdown=max_drawdown,
                    calmar_ratio=calmar_ratio
                )
                
            except Exception as e:
                self.logger.warning(f"Error calculating metrics for {symbol}: {e}")
                continue
        
        self.asset_metrics = metrics
        return metrics
    
    def estimate_covariance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix using specified method"""
        
        try:
            if self.config.covariance_method == "ledoit_wolf":
                try:
                    cov_estimator = LedoitWolf()
                    covariance_matrix = cov_estimator.fit(returns.fillna(0)).covariance_
                except:
                    # Fallback to sample covariance if LedoitWolf fails
                    covariance_matrix = returns.cov().values
            elif self.config.covariance_method == "shrinkage":
                # James-Stein shrinkage
                sample_cov = returns.cov().values
                n, p = returns.shape
                shrinkage = (p + 2) / (n + p + 2)
                target = np.trace(sample_cov) / p * np.eye(p)
                covariance_matrix = (1 - shrinkage) * sample_cov + shrinkage * target
            else:  # sample
                covariance_matrix = returns.cov().values
            
            # Annualize covariance matrix
            covariance_matrix = covariance_matrix * 252
            
            self.covariance_matrix = covariance_matrix
            return covariance_matrix
            
        except Exception as e:
            self.logger.error(f"Error estimating covariance matrix: {e}")
            # Fallback to sample covariance
            self.covariance_matrix = returns.cov().values * 252
            return self.covariance_matrix
    
    def estimate_expected_returns(self, returns: pd.DataFrame, 
                                regime_context: Optional[Dict] = None) -> np.ndarray:
        """Estimate expected returns with optional regime adjustment"""
        
        try:
            # Base expected returns (historical mean)
            base_returns = returns.mean().values * 252  # Annualized
            
            # Apply exponential decay weighting
            weights = np.array([self.config.alpha_decay ** i for i in range(len(returns))][::-1])
            weights = weights / weights.sum()
            
            weighted_returns = np.average(returns.values, axis=0, weights=weights) * 252
            
            # Regime adjustment if available
            if regime_context and self.config.enable_regime_awareness:
                current_regime = regime_context.get('current_regime')
                regime_confidence = regime_context.get('confidence', 0.5)
                
                # Apply regime-based adjustment
                for i, symbol in enumerate(returns.columns):
                    if symbol in self.asset_metrics:
                        regime_perf = self.asset_metrics[symbol].regime_performance.get(current_regime, 0)
                        adjustment = regime_perf * self.config.regime_weight_adjustment * regime_confidence
                        weighted_returns[i] += adjustment
            
            self.expected_returns = weighted_returns
            return weighted_returns
            
        except Exception as e:
            self.logger.error(f"Error estimating expected returns: {e}")
            # Fallback to simple mean
            self.expected_returns = returns.mean().values * 252
            return self.expected_returns
    
    def optimize_portfolio(self, price_data: pd.DataFrame, 
                         regime_context: Optional[Dict] = None) -> PortfolioWeights:
        """Optimize portfolio weights based on specified objective"""
        
        try:
            # Prepare data
            returns = price_data.pct_change().dropna()
            
            if len(returns) < self.config.lookback_period:
                lookback = len(returns)
            else:
                lookback = self.config.lookback_period
                returns = returns.tail(lookback)
            
            # Calculate inputs
            expected_returns = self.estimate_expected_returns(returns, regime_context)
            covariance_matrix = self.estimate_covariance_matrix(returns)
            
            n_assets = len(expected_returns)
            
            # Optimization based on objective
            if self.config.optimization_objective == OptimizationObjective.SHARPE_RATIO:
                weights = self._optimize_sharpe_ratio(expected_returns, covariance_matrix)
            elif self.config.optimization_objective == OptimizationObjective.MIN_VARIANCE:
                weights = self._optimize_min_variance(covariance_matrix)
            elif self.config.optimization_objective == OptimizationObjective.RISK_PARITY:
                weights = self._optimize_risk_parity(covariance_matrix)
            elif self.config.optimization_objective == OptimizationObjective.KELLY_OPTIMAL:
                weights = self._optimize_kelly_criterion(expected_returns, covariance_matrix)
            else:
                # Default to equal weight
                weights = np.ones(n_assets) / n_assets
            
            # Create weights dictionary
            weight_dict = dict(zip(returns.columns, weights))
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calculate diversification ratio
            asset_volatilities = np.sqrt(np.diag(covariance_matrix))
            weighted_avg_vol = np.dot(weights, asset_volatilities)
            diversification_ratio = weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calculate turnover if previous weights exist
            turnover = 0.0
            if self.current_weights:
                current_weight_vector = np.array([self.current_weights.get(symbol, 0) 
                                                for symbol in returns.columns])
                turnover = np.sum(np.abs(weights - current_weight_vector))
            
            # Calculate transaction costs
            transaction_costs = turnover * self.config.transaction_cost_bps / 10000 if self.config.enable_transaction_costs else 0
            
            result = PortfolioWeights(
                timestamp=datetime.now(),
                weights=weight_dict,
                expected_return=portfolio_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                diversification_ratio=diversification_ratio,
                turnover=turnover,
                transaction_costs=transaction_costs,
                current_regime=regime_context.get('current_regime') if regime_context else None,
                regime_confidence=regime_context.get('confidence', 0) if regime_context else 0
            )
            
            # Update current weights
            self.current_weights = weight_dict
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            # Return equal weight as fallback
            n_assets = len(price_data.columns)
            equal_weights = {symbol: 1/n_assets for symbol in price_data.columns}
            return PortfolioWeights(
                timestamp=datetime.now(),
                weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0
            )
    
    def _optimize_sharpe_ratio(self, expected_returns: np.ndarray, 
                              covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize for maximum Sharpe ratio using scipy"""
        
        n = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            if portfolio_volatility == 0:
                return -999
            return -portfolio_return / portfolio_volatility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n)]
        
        # Initial guess
        x0 = np.ones(n) / n
        
        try:
            result = minimize(negative_sharpe, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                weights = np.maximum(weights, self.config.min_weight)
                weights = weights / weights.sum()
                return weights
        except:
            pass
        
        # Analytical fallback
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            weights = inv_cov @ expected_returns
            weights = weights / weights.sum()
            
            # Apply constraints
            weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
            weights = weights / weights.sum()
            
            return weights
        except:
            # Final fallback: equal weights
            return np.ones(n) / n
    
    def _optimize_min_variance(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize for minimum variance"""
        
        n = covariance_matrix.shape[0]
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n)]
        x0 = np.ones(n) / n
        
        try:
            result = minimize(portfolio_variance, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
                return weights
        except:
            pass
        
        # Analytical fallback
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            ones = np.ones((n, 1))
            weights = (inv_cov @ ones) / (ones.T @ inv_cov @ ones)
            weights = weights.flatten()
            
            # Apply constraints
            weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
            weights = weights / weights.sum()
            
            return weights
        except:
            return np.ones(n) / n
    
    def _optimize_risk_parity(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize for risk parity (equal risk contribution)"""
        
        n = covariance_matrix.shape[0]
        
        def risk_parity_objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            if portfolio_vol == 0:
                return 999
            
            # Risk contributions
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # Objective: minimize sum of squared deviations from equal risk
            target_risk = portfolio_vol / n
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n)]
        
        # Initial guess: equal weights
        x0 = np.ones(n) / n
        
        try:
            result = minimize(risk_parity_objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                weights = np.maximum(weights, self.config.min_weight)
                weights = weights / weights.sum()
                return weights
        except:
            pass
        
        # Fallback to equal weights
        return np.ones(n) / n
    
    def _optimize_kelly_criterion(self, expected_returns: np.ndarray, 
                                 covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize using Kelly Criterion for growth maximization"""
        
        try:
            # Kelly optimal weights: f* = Σ^(-1) * μ
            inv_cov = np.linalg.inv(covariance_matrix)
            kelly_weights = inv_cov @ expected_returns
            
            # Apply Kelly fraction for safety
            kelly_weights = kelly_weights * self.config.kelly_fraction
            
            # Normalize to sum to 1
            if kelly_weights.sum() > 0:
                kelly_weights = kelly_weights / kelly_weights.sum()
            else:
                kelly_weights = np.ones(len(expected_returns)) / len(expected_returns)
            
            # Apply constraints
            kelly_weights = np.clip(kelly_weights, self.config.min_weight, self.config.max_weight)
            kelly_weights = kelly_weights / kelly_weights.sum()
            
            return kelly_weights
            
        except Exception as e:
            self.logger.warning(f"Kelly optimization failed: {e}")
            n = len(expected_returns)
            return np.ones(n) / n


class PortfolioPerformanceAnalyzer:
    """Portfolio performance analysis and attribution"""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        
    def calculate_performance_metrics(self, portfolio_returns: pd.Series, 
                                    benchmark_returns: Optional[pd.Series] = None) -> PortfolioPerformance:
        """Calculate comprehensive performance metrics"""
        
        try:
            # Basic return metrics
            total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            
            # Risk metrics
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR and CVaR
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            # Excess return vs benchmark
            excess_return = 0.0
            if benchmark_returns is not None and len(benchmark_returns) == len(portfolio_returns):
                benchmark_total = (1 + benchmark_returns).cumprod().iloc[-1] - 1
                excess_return = total_return - benchmark_total
            
            return PortfolioPerformance(
                period_start=portfolio_returns.index[0],
                period_end=portfolio_returns.index[-1],
                total_return=total_return,
                annualized_return=annualized_return,
                excess_return=excess_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PortfolioPerformance(
                period_start=datetime.now(),
                period_end=datetime.now(),
                total_return=0.0,
                annualized_return=0.0
            )


class DynamicRebalancer:
    """Dynamic portfolio rebalancing system"""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.last_rebalance = None
        
    def should_rebalance(self, current_weights: Dict[str, float], 
                        target_weights: Dict[str, float],
                        regime_changed: bool = False) -> Tuple[bool, str]:
        """Determine if portfolio should be rebalanced"""
        
        current_time = datetime.now()
        
        # Check regime-based rebalancing
        if regime_changed and self.config.rebalance_frequency == RebalanceFrequency.DYNAMIC:
            return True, "regime_change"
        
        # Check drift-based rebalancing
        if self.config.enable_dynamic_rebalancing:
            max_drift = 0
            for symbol in target_weights:
                current_weight = current_weights.get(symbol, 0)
                target_weight = target_weights[symbol]
                drift = abs(current_weight - target_weight)
                max_drift = max(max_drift, drift)
            
            if max_drift > self.config.drift_threshold:
                return True, "drift_threshold"
        
        # Check time-based rebalancing
        if self.last_rebalance is None:
            return True, "initial"
        
        time_since_rebalance = current_time - self.last_rebalance
        
        if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            return time_since_rebalance.days >= 1, "scheduled_daily"
        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            return time_since_rebalance.days >= 7, "scheduled_weekly"
        elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            return time_since_rebalance.days >= 30, "scheduled_monthly"
        elif self.config.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            return time_since_rebalance.days >= 90, "scheduled_quarterly"
        
        return False, "no_rebalance"
    
    def execute_rebalance(self, current_weights: Dict[str, float], 
                         target_weights: Dict[str, float]) -> Dict[str, Any]:
        """Execute portfolio rebalancing"""
        
        rebalance_info = {
            'timestamp': datetime.now(),
            'trades': {},
            'total_turnover': 0,
            'estimated_costs': 0
        }
        
        total_turnover = 0
        
        for symbol in target_weights:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights[symbol]
            trade_amount = target_weight - current_weight
            
            if abs(trade_amount) > 0.001:  # Minimum trade threshold
                rebalance_info['trades'][symbol] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'trade_amount': trade_amount,
                    'trade_type': 'buy' if trade_amount > 0 else 'sell'
                }
                
                total_turnover += abs(trade_amount)
        
        rebalance_info['total_turnover'] = total_turnover
        rebalance_info['estimated_costs'] = total_turnover * self.config.transaction_cost_bps / 10000
        
        self.last_rebalance = datetime.now()
        
        return rebalance_info


class RiskAdjustedPortfolioOptimization:
    """Main risk-adjusted portfolio optimization system"""
    
    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        
        # Initialize components
        self.optimizer = PortfolioOptimizer(self.config)
        self.performance_analyzer = PortfolioPerformanceAnalyzer(self.config)
        self.rebalancer = DynamicRebalancer(self.config)
        
        # State management
        self.optimization_history = []
        self.performance_history = []
        self.current_portfolio = None
        
        logger.info("Risk-Adjusted Portfolio Optimization system initialized")
    
    def optimize_portfolio(self, price_data: pd.DataFrame, 
                         regime_context: Optional[Dict] = None) -> PortfolioWeights:
        """Comprehensive portfolio optimization with risk adjustment"""
        
        try:
            # Calculate asset metrics
            asset_metrics = self.optimizer.calculate_asset_metrics(price_data)
            
            # Optimize portfolio
            optimal_weights = self.optimizer.optimize_portfolio(price_data, regime_context)
            
            # Store optimization result
            self.optimization_history.append(optimal_weights)
            self.current_portfolio = optimal_weights
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            # Return equal weight fallback
            n_assets = len(price_data.columns)
            equal_weights = {symbol: 1/n_assets for symbol in price_data.columns}
            return PortfolioWeights(
                timestamp=datetime.now(),
                weights=equal_weights
            )
    
    def analyze_performance(self, portfolio_returns: pd.Series, 
                          benchmark_returns: Optional[pd.Series] = None) -> PortfolioPerformance:
        """Analyze portfolio performance"""
        
        performance = self.performance_analyzer.calculate_performance_metrics(
            portfolio_returns, benchmark_returns
        )
        
        self.performance_history.append(performance)
        return performance
    
    def check_rebalancing(self, current_weights: Dict[str, float], 
                         target_weights: Dict[str, float],
                         regime_changed: bool = False) -> Tuple[bool, Dict]:
        """Check if rebalancing is needed and execute if required"""
        
        should_rebalance, reason = self.rebalancer.should_rebalance(
            current_weights, target_weights, regime_changed
        )
        
        rebalance_info = {}
        if should_rebalance:
            rebalance_info = self.rebalancer.execute_rebalance(current_weights, target_weights)
            rebalance_info['reason'] = reason
        
        return should_rebalance, rebalance_info
    
    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics"""
        
        if not self.optimization_history:
            return {}
        
        # Recent portfolio metrics
        latest_portfolio = self.optimization_history[-1]
        
        # Historical performance
        if self.performance_history:
            avg_return = np.mean([p.annualized_return for p in self.performance_history])
            avg_sharpe = np.mean([p.sharpe_ratio for p in self.performance_history])
            max_drawdown = min([p.max_drawdown for p in self.performance_history])
        else:
            avg_return = avg_sharpe = max_drawdown = 0
        
        # Optimization statistics
        avg_turnover = np.mean([opt.turnover for opt in self.optimization_history])
        avg_transaction_costs = np.mean([opt.transaction_costs for opt in self.optimization_history])
        
        return {
            'total_optimizations': len(self.optimization_history),
            'current_expected_return': latest_portfolio.expected_return,
            'current_expected_volatility': latest_portfolio.expected_volatility,
            'current_sharpe_ratio': latest_portfolio.sharpe_ratio,
            'historical_avg_return': avg_return,
            'historical_avg_sharpe': avg_sharpe,
            'historical_max_drawdown': max_drawdown,
            'average_turnover': avg_turnover,
            'average_transaction_costs': avg_transaction_costs,
            'optimization_objective': self.config.optimization_objective.value,
            'current_weights': latest_portfolio.weights
        }


def create_risk_adjusted_portfolio_optimization(custom_config: Dict = None) -> RiskAdjustedPortfolioOptimization:
    """Factory function to create portfolio optimization system"""
    
    config = PortfolioConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return RiskAdjustedPortfolioOptimization(config)


if __name__ == "__main__":
    # Example usage
    print("Risk-Adjusted Portfolio Optimization System")
    
    # Create optimization system
    system = create_risk_adjusted_portfolio_optimization({
        'optimization_objective': OptimizationObjective.SHARPE_RATIO,
        'enable_kelly_sizing': True,
        'enable_regime_awareness': True
    })
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='1D')
    
    # Simulate asset prices with different characteristics
    assets = ['GOLD', 'SILVER', 'PLATINUM', 'COPPER']
    price_data = pd.DataFrame(index=dates, columns=assets)
    
    # Different return/volatility profiles
    asset_params = {
        'GOLD': {'return': 0.08, 'vol': 0.15},
        'SILVER': {'return': 0.12, 'vol': 0.25},
        'PLATINUM': {'return': 0.06, 'vol': 0.20},
        'COPPER': {'return': 0.10, 'vol': 0.30}
    }
    
    for asset in assets:
        params = asset_params[asset]
        returns = np.random.normal(params['return']/252, params['vol']/np.sqrt(252), 252)
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        price_data[asset] = prices[1:]
    
    # Optimize portfolio
    optimal_weights = system.optimize_portfolio(price_data)
    
    print("Risk-Adjusted Portfolio Optimization Results:")
    print(f"Expected Return: {optimal_weights.expected_return:.2%}")
    print(f"Expected Volatility: {optimal_weights.expected_volatility:.2%}")
    print(f"Sharpe Ratio: {optimal_weights.sharpe_ratio:.3f}")
    print(f"Diversification Ratio: {optimal_weights.diversification_ratio:.3f}")
    
    print("\nOptimal Weights:")
    for asset, weight in optimal_weights.weights.items():
        print(f"  {asset}: {weight:.1%}")
    
    # Simulate portfolio performance
    returns = price_data.pct_change().dropna()
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    
    for date in returns.index:
        daily_return = 0
        for asset, weight in optimal_weights.weights.items():
            daily_return += weight * returns.loc[date, asset]
        portfolio_returns.loc[date] = daily_return
    
    # Analyze performance
    performance = system.analyze_performance(portfolio_returns)
    
    print(f"\nPerformance Analysis:")
    print(f"Total Return: {performance.total_return:.2%}")
    print(f"Annualized Return: {performance.annualized_return:.2%}")
    print(f"Volatility: {performance.volatility:.2%}")
    print(f"Sharpe Ratio: {performance.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {performance.max_drawdown:.2%}")
    print(f"Calmar Ratio: {performance.calmar_ratio:.3f}")
    
    # System statistics
    stats = system.get_portfolio_statistics()
    print(f"\nSystem Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            if 'ratio' in key.lower() or 'return' in key.lower():
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}") 