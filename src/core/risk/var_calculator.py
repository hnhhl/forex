"""
Value at Risk (VaR) Calculator System
Comprehensive VaR calculation with multiple methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import logging

from ..base_system import BaseSystem

logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC_NORMAL = "parametric_normal"
    PARAMETRIC_T = "parametric_t"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"


class DistributionType(Enum):
    """Distribution types for parametric VaR"""
    NORMAL = "normal"
    T_STUDENT = "t"


@dataclass
class VaRResult:
    """VaR calculation result"""
    method: VaRMethod
    confidence_level: float
    var_value: float
    cvar_value: float  # Conditional VaR (Expected Shortfall)
    portfolio_value: float
    var_absolute: float
    var_percentage: float
    calculation_date: datetime
    lookback_period: int
    additional_metrics: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'method': self.method.value,
            'confidence_level': self.confidence_level,
            'var_value': self.var_value,
            'cvar_value': self.cvar_value,
            'portfolio_value': self.portfolio_value,
            'var_absolute': self.var_absolute,
            'var_percentage': self.var_percentage,
            'calculation_date': self.calculation_date.isoformat(),
            'lookback_period': self.lookback_period,
            'additional_metrics': self.additional_metrics or {}
        }


class VaRCalculator(BaseSystem):
    """
    Comprehensive Value at Risk Calculator
    Supports multiple VaR calculation methods
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("VaRCalculator", config)
        self.config = config or {}
        
        # Configuration
        self.lookback_period = self.config.get('lookback_period', 252)  # 1 year
        self.monte_carlo_simulations = self.config.get('monte_carlo_simulations', 10000)
        self.default_confidence_levels = self.config.get('confidence_levels', [0.95, 0.99, 0.999])
        
        # Data storage
        self.returns_data: Optional[pd.DataFrame] = None
        self.portfolio_values: Optional[pd.Series] = None
        self.current_portfolio_value: float = 0.0
        
        # Results cache
        self.var_results: Dict[str, VaRResult] = {}
        self.statistics: Dict = {}
    
    def set_data(self, returns_data: pd.DataFrame, portfolio_values: pd.Series):
        """Set returns data and portfolio values"""
        try:
            self.returns_data = returns_data.copy()
            self.portfolio_values = portfolio_values.copy()
            self.current_portfolio_value = portfolio_values.iloc[-1] if len(portfolio_values) > 0 else 0.0
            
            # Validate data
            if len(returns_data) != len(portfolio_values):
                raise ValueError("Returns data and portfolio values must have same length")
            
            if len(returns_data) < self.lookback_period:
                logger.warning(f"Data length ({len(returns_data)}) less than lookback period ({self.lookback_period})")
            
            logger.info(f"Data set: {len(returns_data)} observations, portfolio value: {self.current_portfolio_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error setting data: {e}")
            raise
    
    def calculate_historical_var(self, confidence_level: float = 0.95) -> VaRResult:
        """Calculate Historical VaR"""
        try:
            if self.returns_data is None:
                raise ValueError("No data set. Call set_data() first.")
            
            # Use portfolio returns
            if len(self.returns_data.columns) > 1:
                portfolio_returns = self.returns_data.sum(axis=1)
            else:
                portfolio_returns = self.returns_data.iloc[:, 0]
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(portfolio_returns, var_percentile)
            
            # Calculate CVaR (Expected Shortfall)
            tail_losses = portfolio_returns[portfolio_returns <= var_value]
            cvar_value = tail_losses.mean() if len(tail_losses) > 0 else var_value
            
            # Convert to absolute values
            var_absolute = abs(var_value * self.current_portfolio_value)
            var_percentage = abs(var_value) * 100
            
            result = VaRResult(
                method=VaRMethod.HISTORICAL,
                confidence_level=confidence_level,
                var_value=var_value,
                cvar_value=cvar_value,
                portfolio_value=self.current_portfolio_value,
                var_absolute=var_absolute,
                var_percentage=var_percentage,
                calculation_date=datetime.now(),
                lookback_period=len(portfolio_returns),
                additional_metrics={
                    'observations': len(portfolio_returns),
                    'mean_return': portfolio_returns.mean(),
                    'std_return': portfolio_returns.std(),
                    'skewness': stats.skew(portfolio_returns),
                    'kurtosis': stats.kurtosis(portfolio_returns)
                }
            )
            
            self.var_results[f"historical_{confidence_level}"] = result
            logger.info(f"Historical VaR calculated: {var_absolute:,.2f} ({var_percentage:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating historical VaR: {e}")
            raise
    
    def calculate_parametric_var(self, confidence_level: float = 0.95, 
                               distribution: str = 'normal') -> VaRResult:
        """Calculate Parametric VaR"""
        try:
            if self.returns_data is None:
                raise ValueError("No data set. Call set_data() first.")
            
            # Use portfolio returns
            if len(self.returns_data.columns) > 1:
                portfolio_returns = self.returns_data.sum(axis=1)
            else:
                portfolio_returns = self.returns_data.iloc[:, 0]
            
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            if distribution == 'normal':
                # Normal distribution
                z_score = stats.norm.ppf(1 - confidence_level)
                var_value = mean_return + z_score * std_return
                method = VaRMethod.PARAMETRIC_NORMAL
                
            elif distribution == 't':
                # Student's t-distribution
                df = len(portfolio_returns) - 1
                t_score = stats.t.ppf(1 - confidence_level, df)
                var_value = mean_return + t_score * std_return
                method = VaRMethod.PARAMETRIC_T
                
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
            
            # Calculate CVaR for parametric methods
            if distribution == 'normal':
                cvar_multiplier = stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level)
                cvar_value = mean_return - std_return * cvar_multiplier
            else:
                # Approximate CVaR for t-distribution
                cvar_value = var_value * 1.1  # Conservative approximation
            
            # Convert to absolute values
            var_absolute = abs(var_value * self.current_portfolio_value)
            var_percentage = abs(var_value) * 100
            
            result = VaRResult(
                method=method,
                confidence_level=confidence_level,
                var_value=var_value,
                cvar_value=cvar_value,
                portfolio_value=self.current_portfolio_value,
                var_absolute=var_absolute,
                var_percentage=var_percentage,
                calculation_date=datetime.now(),
                lookback_period=len(portfolio_returns),
                additional_metrics={
                    'distribution': distribution,
                    'mean_return': mean_return,
                    'std_return': std_return,
                    'z_score': z_score if distribution == 'normal' else None,
                    't_score': t_score if distribution == 't' else None,
                    'degrees_freedom': df if distribution == 't' else None
                }
            )
            
            self.var_results[f"parametric_{distribution}_{confidence_level}"] = result
            logger.info(f"Parametric VaR ({distribution}) calculated: {var_absolute:,.2f} ({var_percentage:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}")
            raise
    
    def calculate_monte_carlo_var(self, confidence_level: float = 0.95,
                                simulations: int = None) -> VaRResult:
        """Calculate Monte Carlo VaR"""
        try:
            if self.returns_data is None:
                raise ValueError("No data set. Call set_data() first.")
            
            simulations = simulations or self.monte_carlo_simulations
            
            # Use portfolio returns
            if len(self.returns_data.columns) > 1:
                portfolio_returns = self.returns_data.sum(axis=1)
            else:
                portfolio_returns = self.returns_data.iloc[:, 0]
            
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, simulations)
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(simulated_returns, var_percentile)
            
            # Calculate CVaR
            tail_losses = simulated_returns[simulated_returns <= var_value]
            cvar_value = tail_losses.mean() if len(tail_losses) > 0 else var_value
            
            # Convert to absolute values
            var_absolute = abs(var_value * self.current_portfolio_value)
            var_percentage = abs(var_value) * 100
            
            result = VaRResult(
                method=VaRMethod.MONTE_CARLO,
                confidence_level=confidence_level,
                var_value=var_value,
                cvar_value=cvar_value,
                portfolio_value=self.current_portfolio_value,
                var_absolute=var_absolute,
                var_percentage=var_percentage,
                calculation_date=datetime.now(),
                lookback_period=len(portfolio_returns),
                additional_metrics={
                    'simulations': simulations,
                    'mean_return': mean_return,
                    'std_return': std_return,
                    'simulated_mean': simulated_returns.mean(),
                    'simulated_std': simulated_returns.std()
                }
            )
            
            self.var_results[f"monte_carlo_{confidence_level}"] = result
            logger.info(f"Monte Carlo VaR calculated: {var_absolute:,.2f} ({var_percentage:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            raise
    
    def calculate_cornish_fisher_var(self, confidence_level: float = 0.95) -> VaRResult:
        """Calculate Cornish-Fisher VaR (accounts for skewness and kurtosis)"""
        try:
            if self.returns_data is None:
                raise ValueError("No data set. Call set_data() first.")
            
            # Use portfolio returns
            if len(self.returns_data.columns) > 1:
                portfolio_returns = self.returns_data.sum(axis=1)
            else:
                portfolio_returns = self.returns_data.iloc[:, 0]
            
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            skewness = stats.skew(portfolio_returns)
            kurtosis = stats.kurtosis(portfolio_returns)
            
            # Cornish-Fisher expansion
            z = stats.norm.ppf(1 - confidence_level)
            cf_adjustment = (z**2 - 1) * skewness / 6 + (z**3 - 3*z) * kurtosis / 24 - (2*z**3 - 5*z) * skewness**2 / 36
            cf_z = z + cf_adjustment
            
            var_value = mean_return + cf_z * std_return
            
            # Approximate CVaR
            cvar_value = var_value * 1.15  # Conservative approximation
            
            # Convert to absolute values
            var_absolute = abs(var_value * self.current_portfolio_value)
            var_percentage = abs(var_value) * 100
            
            result = VaRResult(
                method=VaRMethod.CORNISH_FISHER,
                confidence_level=confidence_level,
                var_value=var_value,
                cvar_value=cvar_value,
                portfolio_value=self.current_portfolio_value,
                var_absolute=var_absolute,
                var_percentage=var_percentage,
                calculation_date=datetime.now(),
                lookback_period=len(portfolio_returns),
                additional_metrics={
                    'mean_return': mean_return,
                    'std_return': std_return,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'z_score': z,
                    'cf_adjustment': cf_adjustment,
                    'cf_z_score': cf_z
                }
            )
            
            self.var_results[f"cornish_fisher_{confidence_level}"] = result
            logger.info(f"Cornish-Fisher VaR calculated: {var_absolute:,.2f} ({var_percentage:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Cornish-Fisher VaR: {e}")
            raise
    
    def calculate_all_var_methods(self, confidence_level: float = 0.95) -> Dict[str, VaRResult]:
        """Calculate VaR using all available methods"""
        try:
            results = {}
            
            # Historical VaR
            results['historical'] = self.calculate_historical_var(confidence_level)
            
            # Parametric VaR (Normal)
            results['parametric_normal'] = self.calculate_parametric_var(confidence_level, 'normal')
            
            # Parametric VaR (t-distribution)
            results['parametric_t'] = self.calculate_parametric_var(confidence_level, 't')
            
            # Monte Carlo VaR
            results['monte_carlo'] = self.calculate_monte_carlo_var(confidence_level)
            
            # Cornish-Fisher VaR
            results['cornish_fisher'] = self.calculate_cornish_fisher_var(confidence_level)
            
            logger.info(f"All VaR methods calculated for confidence level {confidence_level}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating all VaR methods: {e}")
            raise
    
    def get_var_summary(self, confidence_levels: List[float] = None) -> Dict:
        """Get VaR summary for multiple confidence levels"""
        try:
            confidence_levels = confidence_levels or self.default_confidence_levels
            summary = {}
            
            for cl in confidence_levels:
                cl_results = self.calculate_all_var_methods(cl)
                summary[f"{cl:.1%}"] = {
                    method: {
                        'var_absolute': result.var_absolute,
                        'var_percentage': result.var_percentage,
                        'cvar_absolute': abs(result.cvar_value * self.current_portfolio_value),
                        'cvar_percentage': abs(result.cvar_value) * 100
                    }
                    for method, result in cl_results.items()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting VaR summary: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        try:
            if self.returns_data is None:
                return {}
            
            # Use portfolio returns
            if len(self.returns_data.columns) > 1:
                portfolio_returns = self.returns_data.sum(axis=1)
            else:
                portfolio_returns = self.returns_data.iloc[:, 0]
            
            stats_dict = {
                'data_points': len(portfolio_returns),
                'mean_return': portfolio_returns.mean(),
                'std_return': portfolio_returns.std(),
                'min_return': portfolio_returns.min(),
                'max_return': portfolio_returns.max(),
                'skewness': stats.skew(portfolio_returns),
                'kurtosis': stats.kurtosis(portfolio_returns),
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0,
                'var_calculations': len(self.var_results),
                'total_var_calculations': len(self.var_results),
                'current_portfolio_value': self.current_portfolio_value,
                'last_updated': datetime.now().isoformat()
            }
            
            self.statistics = stats_dict
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def export_var_data(self, filepath: str) -> bool:
        """Export VaR results to JSON file"""
        try:
            export_data = {
                'calculation_timestamp': datetime.now().isoformat(),
                'portfolio_value': self.current_portfolio_value,
                'statistics': self.get_statistics(),
                'var_results': {
                    key: result.to_dict() for key, result in self.var_results.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"VaR data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting VaR data: {e}")
            return False
    
    def validate_data(self) -> Dict[str, bool]:
        """Validate input data quality"""
        validation = {
            'has_data': self.returns_data is not None,
            'sufficient_data': False,
            'no_missing_values': False,
            'reasonable_values': False
        }
        
        if validation['has_data']:
            validation['sufficient_data'] = len(self.returns_data) >= 30  # Minimum 30 observations
            validation['no_missing_values'] = not self.returns_data.isnull().any().any()
            
            # Check for reasonable return values (not extreme outliers)
            if len(self.returns_data.columns) > 1:
                portfolio_returns = self.returns_data.sum(axis=1)
            else:
                portfolio_returns = self.returns_data.iloc[:, 0]
            
            validation['reasonable_values'] = (
                portfolio_returns.abs().max() < 1.0 and  # No single day >100% return
                portfolio_returns.std() < 0.5  # Standard deviation <50%
            )
        
        return validation


# Monte Carlo Simulator for advanced VaR calculations
class MonteCarloSimulator:
    """Advanced Monte Carlo simulator for VaR calculations"""
    
    def __init__(self, simulations: int = 10000):
        self.simulations = simulations
    
    def simulate_geometric_brownian_motion(self, S0: float, mu: float, sigma: float, 
                                         T: float, steps: int) -> np.ndarray:
        """Simulate Geometric Brownian Motion"""
        dt = T / steps
        prices = np.zeros((self.simulations, steps + 1))
        prices[:, 0] = S0
        
        for t in range(1, steps + 1):
            Z = np.random.standard_normal(self.simulations)
            prices[:, t] = prices[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        return prices
    
    def simulate_jump_diffusion(self, S0: float, mu: float, sigma: float, 
                              lambda_jump: float, mu_jump: float, sigma_jump: float,
                              T: float, steps: int) -> np.ndarray:
        """Simulate Jump Diffusion Process (Merton Model)"""
        dt = T / steps
        prices = np.zeros((self.simulations, steps + 1))
        prices[:, 0] = S0
        
        for t in range(1, steps + 1):
            # Brownian motion component
            Z = np.random.standard_normal(self.simulations)
            
            # Jump component
            N = np.random.poisson(lambda_jump * dt, self.simulations)
            J = np.sum(np.random.normal(mu_jump, sigma_jump, (self.simulations, np.max(N) if np.max(N) > 0 else 1))[:, :np.max(N)] if np.max(N) > 0 else np.zeros((self.simulations, 1)), axis=1)
            
            prices[:, t] = prices[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + J * (N > 0)
            )
        
        return prices 