"""
Portfolio Optimizer Module
Advanced portfolio optimization algorithms including Mean Variance, Black-Litterman, Risk Parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import scipy.optimize as sco
from scipy import linalg
import warnings

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MAXIMUM_RETURN = "maximum_return"
    EQUAL_WEIGHT = "equal_weight"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"


class ObjectiveFunction(Enum):
    """Optimization objective functions"""
    MINIMIZE_VARIANCE = "minimize_variance"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_CVAR = "minimize_cvar"
    MAXIMIZE_UTILITY = "maximize_utility"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    sum_weights: float = 1.0
    max_turnover: Optional[float] = None
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None
    tracking_error_limit: Optional[float] = None
    max_concentration: Optional[float] = None
    min_positions: Optional[int] = None
    max_positions: Optional[int] = None


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    optimization_method: str
    objective_value: float
    success: bool
    message: str
    computation_time: float
    iterations: int


@dataclass
class BlackLittermanInputs:
    """Black-Litterman model inputs"""
    market_caps: Dict[str, float]
    risk_aversion: float = 3.0
    tau: float = 0.025
    views: Optional[Dict[str, float]] = None  # Expected returns views
    view_confidence: Optional[Dict[str, float]] = None  # Confidence in views
    reference_returns: Optional[Dict[str, float]] = None


class PortfolioOptimizer:
    """Advanced Portfolio Optimization System"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Optimization parameters
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2%
        self.confidence_level = self.config.get('confidence_level', 0.95)  # 95%
        self.lookback_period = self.config.get('lookback_period', 252)  # 1 year
        self.rebalance_frequency = self.config.get('rebalance_frequency', 'monthly')
        
        # Numerical parameters
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.tolerance = self.config.get('tolerance', 1e-8)
        self.regularization = self.config.get('regularization', 1e-5)
        
        # Data storage
        self.returns_data: Optional[pd.DataFrame] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        self.expected_returns: Optional[Dict[str, float]] = None
        self.symbols: List[str] = []
        
        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        
        logger.info("PortfolioOptimizer initialized")
    
    def set_returns_data(self, returns_data: pd.DataFrame):
        """Set historical returns data"""
        try:
            self.returns_data = returns_data.copy()
            self.symbols = list(returns_data.columns)
            
            # Calculate covariance matrix
            self.covariance_matrix = returns_data.cov().values
            
            # Calculate expected returns (mean historical returns)
            self.expected_returns = returns_data.mean().to_dict()
            
            logger.info(f"Returns data set for {len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error setting returns data: {e}")
    
    def optimize_portfolio(self, 
                          method: OptimizationMethod,
                          constraints: OptimizationConstraints = None,
                          objective: ObjectiveFunction = ObjectiveFunction.MAXIMIZE_SHARPE,
                          current_weights: Optional[Dict[str, float]] = None,
                          **kwargs) -> OptimizationResult:
        """Optimize portfolio using specified method"""
        
        start_time = datetime.now()
        
        try:
            if not self._validate_inputs():
                return OptimizationResult(
                    weights={}, expected_return=0.0, expected_volatility=0.0,
                    sharpe_ratio=0.0, max_drawdown=0.0, var_95=0.0, cvar_95=0.0,
                    optimization_method=method.value, objective_value=0.0,
                    success=False, message="Invalid inputs", computation_time=0.0, iterations=0
                )
            
            # Set default constraints
            if constraints is None:
                constraints = OptimizationConstraints()
            
            # Optimize based on method
            if method == OptimizationMethod.MEAN_VARIANCE:
                result = self._optimize_mean_variance(objective, constraints, **kwargs)
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                result = self._optimize_black_litterman(constraints, **kwargs)
            elif method == OptimizationMethod.RISK_PARITY:
                result = self._optimize_risk_parity(constraints, **kwargs)
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                result = self._optimize_minimum_variance(constraints, **kwargs)
            elif method == OptimizationMethod.MAXIMUM_SHARPE:
                result = self._optimize_maximum_sharpe(constraints, **kwargs)
            elif method == OptimizationMethod.EQUAL_WEIGHT:
                result = self._optimize_equal_weight(constraints, **kwargs)
            elif method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
                result = self._optimize_hierarchical_risk_parity(constraints, **kwargs)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # Calculate performance metrics
            result = self._calculate_portfolio_metrics(result)
            
            # Update computation time
            computation_time = (datetime.now() - start_time).total_seconds()
            result.computation_time = computation_time
            
            # Add to history
            self.optimization_history.append(result)
            
            logger.info(f"Portfolio optimization completed: {method.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return OptimizationResult(
                weights={}, expected_return=0.0, expected_volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, var_95=0.0, cvar_95=0.0,
                optimization_method=method.value, objective_value=0.0,
                success=False, message=str(e), computation_time=0.0, iterations=0
            )
    
    def _optimize_mean_variance(self, objective: ObjectiveFunction, 
                               constraints: OptimizationConstraints, **kwargs) -> OptimizationResult:
        """Mean-Variance optimization (Markowitz)"""
        
        n_assets = len(self.symbols)
        
        # Objective function
        def objective_function(weights):
            portfolio_return = np.dot(weights, list(self.expected_returns.values()))
            portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            
            if objective == ObjectiveFunction.MINIMIZE_VARIANCE:
                return portfolio_variance
            elif objective == ObjectiveFunction.MAXIMIZE_SHARPE:
                if portfolio_variance <= 0:
                    return -np.inf
                return -(portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
            elif objective == ObjectiveFunction.MAXIMIZE_RETURN:
                return -portfolio_return
            else:
                # Utility maximization (return - risk_aversion * variance)
                risk_aversion = kwargs.get('risk_aversion', 3.0)
                return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - constraints.sum_weights}]
        
        # Bounds
        bounds = tuple((constraints.min_weight, constraints.max_weight) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = sco.minimize(
            objective_function, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        # Create result
        weights_dict = {symbol: weight for symbol, weight in zip(self.symbols, result.x)}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0.0,  # Will be calculated later
            expected_volatility=0.0,  # Will be calculated later
            sharpe_ratio=0.0,  # Will be calculated later
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            optimization_method=OptimizationMethod.MEAN_VARIANCE.value,
            objective_value=result.fun,
            success=result.success,
            message=result.message,
            computation_time=0.0,
            iterations=result.nit
        )
    
    def _optimize_black_litterman(self, constraints: OptimizationConstraints, 
                                 **kwargs) -> OptimizationResult:
        """Black-Litterman optimization"""
        
        bl_inputs = kwargs.get('bl_inputs')
        if not bl_inputs:
            # Create default Black-Litterman inputs
            bl_inputs = BlackLittermanInputs(
                market_caps={symbol: 1.0 for symbol in self.symbols}
            )
        
        # Market capitalization weights
        total_market_cap = sum(bl_inputs.market_caps.values())
        market_weights = np.array([
            bl_inputs.market_caps.get(symbol, 1.0) / total_market_cap 
            for symbol in self.symbols
        ])
        
        # Implied equilibrium returns
        implied_returns = bl_inputs.risk_aversion * np.dot(self.covariance_matrix, market_weights)
        
        # Black-Litterman formula
        if bl_inputs.views and bl_inputs.view_confidence:
            # Views matrix P (which assets the views relate to)
            P = np.eye(len(self.symbols))  # Simplified: views on all assets
            
            # Views vector Q (expected returns)
            Q = np.array([bl_inputs.views.get(symbol, 0.0) for symbol in self.symbols])
            
            # Confidence matrix Omega
            omega_diag = [1.0 / bl_inputs.view_confidence.get(symbol, 1.0) for symbol in self.symbols]
            Omega = np.diag(omega_diag)
            
            # Black-Litterman expected returns
            tau_cov = bl_inputs.tau * self.covariance_matrix
            
            # M1 = inv(tau * Sigma)
            M1 = linalg.inv(tau_cov)
            
            # M2 = P' * inv(Omega) * P
            M2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
            
            # M3 = inv(tau * Sigma) * Pi + P' * inv(Omega) * Q
            M3 = np.dot(M1, implied_returns) + np.dot(P.T, np.dot(linalg.inv(Omega), Q))
            
            # New expected returns
            bl_returns = np.dot(linalg.inv(M1 + M2), M3)
            
            # New covariance matrix
            bl_cov = linalg.inv(M1 + M2)
        else:
            # No views - use implied returns
            bl_returns = implied_returns
            bl_cov = self.covariance_matrix
        
        # Optimize using Black-Litterman inputs
        n_assets = len(self.symbols)
        
        def objective_function(weights):
            portfolio_return = np.dot(weights, bl_returns)
            portfolio_variance = np.dot(weights.T, np.dot(bl_cov, weights))
            return -(portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
        
        # Constraints and bounds
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - constraints.sum_weights}]
        bounds = tuple((constraints.min_weight, constraints.max_weight) for _ in range(n_assets))
        x0 = market_weights
        
        # Optimize
        result = sco.minimize(
            objective_function, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        weights_dict = {symbol: weight for symbol, weight in zip(self.symbols, result.x)}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            optimization_method=OptimizationMethod.BLACK_LITTERMAN.value,
            objective_value=result.fun,
            success=result.success,
            message=result.message,
            computation_time=0.0,
            iterations=result.nit
        )
    
    def _optimize_risk_parity(self, constraints: OptimizationConstraints, 
                             **kwargs) -> OptimizationResult:
        """Risk Parity optimization"""
        
        n_assets = len(self.symbols)
        
        def risk_budget_objective(weights):
            """Risk parity objective function"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            
            # Marginal risk contributions
            marginal_contrib = np.dot(self.covariance_matrix, weights) / portfolio_vol
            
            # Risk contributions
            risk_contrib = weights * marginal_contrib
            
            # Target risk contribution (equal for all assets)
            target_risk = portfolio_vol / n_assets
            
            # Sum of squared deviations from target
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints and bounds
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - constraints.sum_weights}]
        bounds = tuple((constraints.min_weight, constraints.max_weight) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = sco.minimize(
            risk_budget_objective, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        weights_dict = {symbol: weight for symbol, weight in zip(self.symbols, result.x)}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            optimization_method=OptimizationMethod.RISK_PARITY.value,
            objective_value=result.fun,
            success=result.success,
            message=result.message,
            computation_time=0.0,
            iterations=result.nit
        )
    
    def _optimize_minimum_variance(self, constraints: OptimizationConstraints, 
                                  **kwargs) -> OptimizationResult:
        """Minimum Variance optimization"""
        
        n_assets = len(self.symbols)
        
        def objective_function(weights):
            return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        
        # Constraints and bounds
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - constraints.sum_weights}]
        bounds = tuple((constraints.min_weight, constraints.max_weight) for _ in range(n_assets))
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = sco.minimize(
            objective_function, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        weights_dict = {symbol: weight for symbol, weight in zip(self.symbols, result.x)}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            optimization_method=OptimizationMethod.MINIMUM_VARIANCE.value,
            objective_value=result.fun,
            success=result.success,
            message=result.message,
            computation_time=0.0,
            iterations=result.nit
        )
    
    def _optimize_maximum_sharpe(self, constraints: OptimizationConstraints, 
                                **kwargs) -> OptimizationResult:
        """Maximum Sharpe Ratio optimization"""
        
        n_assets = len(self.symbols)
        
        def objective_function(weights):
            portfolio_return = np.dot(weights, list(self.expected_returns.values()))
            portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            
            if portfolio_variance <= 0:
                return -np.inf
            
            return -(portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
        
        # Constraints and bounds
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - constraints.sum_weights}]
        bounds = tuple((constraints.min_weight, constraints.max_weight) for _ in range(n_assets))
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = sco.minimize(
            objective_function, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        weights_dict = {symbol: weight for symbol, weight in zip(self.symbols, result.x)}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            optimization_method=OptimizationMethod.MAXIMUM_SHARPE.value,
            objective_value=result.fun,
            success=result.success,
            message=result.message,
            computation_time=0.0,
            iterations=result.nit
        )
    
    def _optimize_equal_weight(self, constraints: OptimizationConstraints, 
                              **kwargs) -> OptimizationResult:
        """Equal Weight optimization"""
        
        n_assets = len(self.symbols)
        equal_weight = constraints.sum_weights / n_assets
        
        weights_dict = {symbol: equal_weight for symbol in self.symbols}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            optimization_method=OptimizationMethod.EQUAL_WEIGHT.value,
            objective_value=0.0,
            success=True,
            message="Equal weight allocation",
            computation_time=0.0,
            iterations=0
        )
    
    def _optimize_hierarchical_risk_parity(self, constraints: OptimizationConstraints, 
                                          **kwargs) -> OptimizationResult:
        """Hierarchical Risk Parity optimization"""
        
        # Simplified HRP - use correlation-based clustering
        correlation_matrix = np.corrcoef(self.returns_data.T)
        
        # Distance matrix
        distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
        
        # Hierarchical clustering (simplified)
        n_assets = len(self.symbols)
        weights = np.array([1.0 / n_assets] * n_assets)
        
        # Apply inverse volatility weighting within clusters
        volatilities = np.sqrt(np.diag(self.covariance_matrix))
        inv_vol_weights = (1.0 / volatilities) / np.sum(1.0 / volatilities)
        
        weights_dict = {symbol: weight for symbol, weight in zip(self.symbols, inv_vol_weights)}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            optimization_method=OptimizationMethod.HIERARCHICAL_RISK_PARITY.value,
            objective_value=0.0,
            success=True,
            message="Hierarchical Risk Parity allocation",
            computation_time=0.0,
            iterations=0
        )
    
    def _calculate_portfolio_metrics(self, result: OptimizationResult) -> OptimizationResult:
        """Calculate portfolio performance metrics"""
        
        try:
            weights = np.array(list(result.weights.values()))
            
            # Expected return
            expected_return = np.dot(weights, list(self.expected_returns.values()))
            result.expected_return = expected_return * 252  # Annualized
            
            # Expected volatility
            portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            result.expected_volatility = np.sqrt(portfolio_variance * 252)  # Annualized
            
            # Sharpe ratio
            if result.expected_volatility > 0:
                result.sharpe_ratio = (result.expected_return - self.risk_free_rate) / result.expected_volatility
            
            # VaR and CVaR (if returns data available)
            if self.returns_data is not None:
                portfolio_returns = np.dot(self.returns_data.values, weights)
                result.var_95 = np.percentile(portfolio_returns, 5)
                
                # CVaR (Expected Shortfall)
                var_threshold = result.var_95
                tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
                if len(tail_returns) > 0:
                    result.cvar_95 = np.mean(tail_returns)
                
                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + portfolio_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                result.max_drawdown = np.min(drawdown)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return result
    
    def _validate_inputs(self) -> bool:
        """Validate optimization inputs"""
        
        if self.returns_data is None:
            logger.error("Returns data not set")
            return False
        
        if self.covariance_matrix is None:
            logger.error("Covariance matrix not calculated")
            return False
        
        if not self.expected_returns:
            logger.error("Expected returns not calculated")
            return False
        
        if len(self.symbols) == 0:
            logger.error("No symbols available")
            return False
        
        # Check for NaN values
        if np.any(np.isnan(self.covariance_matrix)):
            logger.error("Covariance matrix contains NaN values")
            return False
        
        return True
    
    def generate_efficient_frontier(self, n_points: int = 50, 
                                   constraints: OptimizationConstraints = None) -> pd.DataFrame:
        """Generate efficient frontier"""
        
        try:
            if not self._validate_inputs():
                return pd.DataFrame()
            
            if constraints is None:
                constraints = OptimizationConstraints()
            
            # Calculate minimum and maximum returns
            min_return = min(self.expected_returns.values())
            max_return = max(self.expected_returns.values())
            
            # Generate target returns
            target_returns = np.linspace(min_return, max_return, n_points)
            
            efficient_portfolios = []
            
            for target_return in target_returns:
                try:
                    # Optimize for minimum variance given target return
                    n_assets = len(self.symbols)
                    
                    def objective_function(weights):
                        return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
                    
                    # Constraints
                    cons = [
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - constraints.sum_weights},
                        {'type': 'eq', 'fun': lambda x: np.dot(x, list(self.expected_returns.values())) - target_return}
                    ]
                    
                    bounds = tuple((constraints.min_weight, constraints.max_weight) for _ in range(n_assets))
                    x0 = np.array([1.0 / n_assets] * n_assets)
                    
                    result = sco.minimize(
                        objective_function, x0, method='SLSQP',
                        bounds=bounds, constraints=cons,
                        options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
                    )
                    
                    if result.success:
                        weights = result.x
                        portfolio_return = np.dot(weights, list(self.expected_returns.values()))
                        portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
                        portfolio_volatility = np.sqrt(portfolio_variance)
                        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                        
                        efficient_portfolios.append({
                            'return': portfolio_return * 252,  # Annualized
                            'volatility': portfolio_volatility * np.sqrt(252),  # Annualized
                            'sharpe_ratio': sharpe_ratio,
                            'weights': {symbol: weight for symbol, weight in zip(self.symbols, weights)}
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to optimize for target return {target_return}: {e}")
                    continue
            
            return pd.DataFrame(efficient_portfolios)
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            return pd.DataFrame()
    
    def backtest_strategy(self, weights: Dict[str, float], 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         rebalance_freq: str = 'monthly') -> Dict[str, Any]:
        """Backtest portfolio strategy"""
        
        try:
            if self.returns_data is None:
                logger.error("Returns data not available for backtesting")
                return {}
            
            # Filter data by date range
            returns_data = self.returns_data.copy()
            if start_date:
                returns_data = returns_data[returns_data.index >= start_date]
            if end_date:
                returns_data = returns_data[returns_data.index <= end_date]
            
            if returns_data.empty:
                logger.error("No data available for backtesting period")
                return {}
            
            # Calculate portfolio returns
            weights_array = np.array([weights.get(symbol, 0.0) for symbol in self.symbols])
            portfolio_returns = np.dot(returns_data.values, weights_array)
            
            # Calculate performance metrics
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            total_return = cumulative_returns[-1] - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # VaR and CVaR
            var_95 = np.percentile(portfolio_returns, 5)
            tail_returns = portfolio_returns[portfolio_returns <= var_95]
            cvar_95 = np.mean(tail_returns) if len(tail_returns) > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sortino ratio
            negative_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'portfolio_returns': portfolio_returns.tolist(),
                'cumulative_returns': cumulative_returns.tolist(),
                'drawdown': drawdown.tolist(),
                'start_date': returns_data.index[0].isoformat() if hasattr(returns_data.index[0], 'isoformat') else str(returns_data.index[0]),
                'end_date': returns_data.index[-1].isoformat() if hasattr(returns_data.index[-1], 'isoformat') else str(returns_data.index[-1]),
                'n_periods': len(portfolio_returns)
            }
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {}
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history"""
        return self.optimization_history.copy()
    
    def compare_strategies(self, strategies: Dict[str, Dict[str, float]], 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Compare multiple portfolio strategies"""
        
        try:
            comparison_results = []
            
            for strategy_name, weights in strategies.items():
                backtest_result = self.backtest_strategy(weights, start_date, end_date)
                
                if backtest_result:
                    comparison_results.append({
                        'strategy': strategy_name,
                        'total_return': backtest_result['total_return'],
                        'annualized_return': backtest_result['annualized_return'],
                        'volatility': backtest_result['volatility'],
                        'sharpe_ratio': backtest_result['sharpe_ratio'],
                        'max_drawdown': backtest_result['max_drawdown'],
                        'calmar_ratio': backtest_result['calmar_ratio'],
                        'sortino_ratio': backtest_result['sortino_ratio'],
                        'var_95': backtest_result['var_95'],
                        'cvar_95': backtest_result['cvar_95']
                    })
            
            return pd.DataFrame(comparison_results)
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        return {
            'n_symbols': len(self.symbols),
            'n_optimizations': len(self.optimization_history),
            'successful_optimizations': sum(1 for result in self.optimization_history if result.success),
            'average_computation_time': np.mean([result.computation_time for result in self.optimization_history]) if self.optimization_history else 0,
            'risk_free_rate': self.risk_free_rate,
            'confidence_level': self.confidence_level,
            'lookback_period': self.lookback_period,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance
        }