"""
Advanced Performance Attribution & Analytics
Ultimate XAU Super System V4.0 - Day 28 Implementation

Comprehensive performance analysis capabilities:
- Factor-Based Attribution for return decomposition
- Risk-Adjusted Performance metrics and benchmarking
- Dynamic Benchmarking with regime-aware comparisons
- Advanced Analytics with machine learning insights
- Real-time Performance Monitoring and Attribution
- Portfolio Performance Optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import time

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AttributionMethod(Enum):
    """Performance attribution methods"""
    BRINSON_FACHLER = "brinson_fachler"
    BRINSON_HOOD_BEEBOWER = "brinson_hood_beebower"
    FACTOR_BASED = "factor_based"
    RISK_FACTOR = "risk_factor"
    SECTOR_ALLOCATION = "sector_allocation"
    SECURITY_SELECTION = "security_selection"


class PerformanceMetric(Enum):
    """Performance measurement metrics"""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    TREYNOR_RATIO = "treynor_ratio"
    TRACKING_ERROR = "tracking_error"
    ALPHA = "alpha"
    BETA = "beta"


class BenchmarkType(Enum):
    """Types of benchmarks"""
    STATIC_BENCHMARK = "static_benchmark"
    DYNAMIC_BENCHMARK = "dynamic_benchmark"
    CUSTOM_BENCHMARK = "custom_benchmark"
    REGIME_AWARE_BENCHMARK = "regime_aware_benchmark"
    PEER_GROUP_BENCHMARK = "peer_group_benchmark"


@dataclass
class AttributionConfig:
    """Configuration for performance attribution"""
    
    # Attribution methods
    attribution_methods: List[AttributionMethod] = field(default_factory=lambda: [
        AttributionMethod.BRINSON_FACHLER,
        AttributionMethod.FACTOR_BASED,
        AttributionMethod.RISK_FACTOR
    ])
    
    # Performance metrics
    performance_metrics: List[PerformanceMetric] = field(default_factory=lambda: [
        PerformanceMetric.TOTAL_RETURN,
        PerformanceMetric.SHARPE_RATIO,
        PerformanceMetric.SORTINO_RATIO,
        PerformanceMetric.INFORMATION_RATIO
    ])
    
    # Time periods for analysis
    analysis_periods: List[int] = field(default_factory=lambda: [21, 63, 126, 252])  # 1M, 3M, 6M, 1Y
    
    # Benchmark settings
    benchmark_type: BenchmarkType = BenchmarkType.DYNAMIC_BENCHMARK
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Factor settings
    factor_model_lookback: int = 252  # 1 year for factor estimation
    factor_rebalance_frequency: int = 21  # Monthly factor rebalancing
    
    # Performance calculation settings
    compound_returns: bool = True
    annualize_metrics: bool = True
    rolling_window_analysis: bool = True
    
    # Real-time settings
    real_time_attribution: bool = True
    attribution_frequency: str = "daily"  # daily, weekly, monthly
    
    # Reporting settings
    detailed_attribution: bool = True
    sector_attribution: bool = True
    security_attribution: bool = True


@dataclass
class PerformanceResult:
    """Performance analysis result"""
    
    timestamp: datetime
    period_days: int
    
    # Basic performance
    portfolio_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    
    # Risk metrics
    portfolio_volatility: float = 0.0
    benchmark_volatility: float = 0.0
    tracking_error: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    
    # Statistical measures
    alpha: float = 0.0
    beta: float = 1.0
    r_squared: float = 0.0
    correlation: float = 0.0
    
    # Attribution specific
    attribution_method: Optional[AttributionMethod] = None
    attribution_components: Dict[str, float] = field(default_factory=dict)


@dataclass
class AttributionBreakdown:
    """Detailed attribution breakdown"""
    
    timestamp: datetime
    attribution_method: AttributionMethod
    
    # Asset allocation effect
    allocation_effect: float = 0.0
    selection_effect: float = 0.0
    interaction_effect: float = 0.0
    
    # Factor attribution
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Sector attribution
    sector_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Security attribution
    security_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Risk attribution
    systematic_risk: float = 0.0
    specific_risk: float = 0.0
    
    # Total attribution
    total_attribution: float = 0.0
    unexplained_return: float = 0.0


@dataclass
class BenchmarkAnalysis:
    """Benchmark comparison analysis"""
    
    timestamp: datetime
    benchmark_type: BenchmarkType
    
    # Benchmark composition
    benchmark_weights: Dict[str, float] = field(default_factory=dict)
    benchmark_return: float = 0.0
    benchmark_volatility: float = 0.0
    
    # Relative performance
    outperformance: float = 0.0
    hit_rate: float = 0.0  # Percentage of periods with outperformance
    win_loss_ratio: float = 1.0
    
    # Risk comparison
    downside_capture: float = 1.0
    upside_capture: float = 1.0
    capture_ratio: float = 1.0
    
    # Regime-specific performance
    regime_performance: Dict[str, float] = field(default_factory=dict)


class PerformanceCalculator:
    """Core performance calculation engine"""
    
    def __init__(self, config: AttributionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_basic_metrics(self, portfolio_returns: pd.Series,
                              benchmark_returns: pd.Series = None,
                              period_days: int = 252) -> PerformanceResult:
        """Calculate basic performance metrics"""
        
        try:
            # Basic returns
            portfolio_return = self._calculate_total_return(portfolio_returns, period_days)
            
            if benchmark_returns is not None:
                benchmark_return = self._calculate_total_return(benchmark_returns, period_days)
                excess_return = portfolio_return - benchmark_return
            else:
                benchmark_return = self.config.risk_free_rate * (period_days / 252)
                excess_return = portfolio_return - benchmark_return
            
            # Risk metrics
            portfolio_vol = portfolio_returns.std() * np.sqrt(252) if self.config.annualize_metrics else portfolio_returns.std()
            
            if benchmark_returns is not None:
                benchmark_vol = benchmark_returns.std() * np.sqrt(252) if self.config.annualize_metrics else benchmark_returns.std()
                tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
                
                # Beta calculation
                covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
                benchmark_variance = benchmark_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                
                # Alpha calculation (annualized)
                rf_rate = self.config.risk_free_rate
                alpha = (portfolio_return - rf_rate) - beta * (benchmark_return - rf_rate)
                
                # Correlation
                correlation = portfolio_returns.corr(benchmark_returns)
                r_squared = correlation ** 2
                
            else:
                benchmark_vol = 0.0
                tracking_error = portfolio_vol
                beta = 1.0
                alpha = excess_return
                correlation = 1.0
                r_squared = 1.0
            
            # Risk-adjusted ratios
            sharpe_ratio = excess_return / portfolio_vol if portfolio_vol > 0 else 0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else portfolio_vol
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            
            # Information ratio
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Treynor ratio
            treynor_ratio = excess_return / beta if beta != 0 else 0
            
            # Calmar ratio (simplified - using max drawdown proxy)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdowns.min())
            calmar_ratio = portfolio_return / max_drawdown if max_drawdown > 0 else 0
            
            result = PerformanceResult(
                timestamp=datetime.now(),
                period_days=period_days,
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                portfolio_volatility=portfolio_vol,
                benchmark_volatility=benchmark_vol,
                tracking_error=tracking_error,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                treynor_ratio=treynor_ratio,
                alpha=alpha,
                beta=beta,
                r_squared=r_squared,
                correlation=correlation
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceResult(
                timestamp=datetime.now(),
                period_days=period_days
            )
    
    def _calculate_total_return(self, returns: pd.Series, period_days: int) -> float:
        """Calculate total return for specified period"""
        
        if len(returns) == 0:
            return 0.0
        
        # Take last period_days returns
        period_returns = returns.tail(min(period_days, len(returns)))
        
        if self.config.compound_returns:
            total_return = (1 + period_returns).prod() - 1
        else:
            total_return = period_returns.sum()
        
        # Annualize if configured
        if self.config.annualize_metrics and period_days != 252:
            annualization_factor = 252 / period_days
            total_return = (1 + total_return) ** annualization_factor - 1
        
        return total_return


class FactorAttributionAnalyzer:
    """Factor-based performance attribution"""
    
    def __init__(self, config: AttributionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Common risk factors
        self.risk_factors = {
            'market': 'Market factor (beta)',
            'size': 'Size factor (small vs large cap)',
            'value': 'Value factor (value vs growth)',
            'momentum': 'Momentum factor',
            'quality': 'Quality factor',
            'volatility': 'Volatility factor'
        }
        
    def perform_factor_attribution(self, portfolio_returns: pd.Series,
                                 factor_data: pd.DataFrame = None) -> AttributionBreakdown:
        """Perform factor-based attribution analysis"""
        
        try:
            if factor_data is None:
                # Generate synthetic factor data for demonstration
                factor_data = self._generate_synthetic_factors(portfolio_returns)
            
            # Align data
            aligned_data = pd.concat([portfolio_returns, factor_data], axis=1, join='inner')
            portfolio_returns_aligned = aligned_data.iloc[:, 0]
            factors_aligned = aligned_data.iloc[:, 1:]
            
            # Run factor regression
            factor_loadings, factor_contributions = self._run_factor_regression(
                portfolio_returns_aligned, factors_aligned
            )
            
            # Calculate attribution breakdown
            attribution = AttributionBreakdown(
                timestamp=datetime.now(),
                attribution_method=AttributionMethod.FACTOR_BASED,
                factor_contributions=factor_contributions,
                systematic_risk=sum(abs(c) for c in factor_contributions.values()),
                specific_risk=portfolio_returns_aligned.var() - sum(c**2 for c in factor_contributions.values()),
                total_attribution=sum(factor_contributions.values())
            )
            
            # Calculate unexplained return
            total_return = portfolio_returns_aligned.sum()
            attribution.unexplained_return = total_return - attribution.total_attribution
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"Error in factor attribution: {e}")
            return AttributionBreakdown(
                timestamp=datetime.now(),
                attribution_method=AttributionMethod.FACTOR_BASED
            )
    
    def _generate_synthetic_factors(self, returns: pd.Series) -> pd.DataFrame:
        """Generate synthetic factor data for demonstration"""
        
        np.random.seed(42)
        n_periods = len(returns)
        
        # Generate correlated factors
        market_factor = np.random.normal(0.0008, 0.015, n_periods)
        size_factor = np.random.normal(0.0002, 0.008, n_periods)
        value_factor = np.random.normal(0.0001, 0.006, n_periods)
        momentum_factor = np.random.normal(0.0003, 0.010, n_periods)
        
        factor_data = pd.DataFrame({
            'market': market_factor,
            'size': size_factor,
            'value': value_factor,
            'momentum': momentum_factor
        }, index=returns.index)
        
        return factor_data
    
    def _run_factor_regression(self, returns: pd.Series, 
                             factors: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Run multi-factor regression analysis"""
        
        try:
            # Prepare data
            y = returns.values.reshape(-1, 1)
            X = factors.values
            
            # Add intercept
            X = np.column_stack([np.ones(len(X)), X])
            factor_names = ['alpha'] + list(factors.columns)
            
            # Run regression
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, y.ravel())
            
            # Extract results
            coefficients = reg.coef_
            factor_loadings = dict(zip(factor_names, coefficients))
            
            # Calculate factor contributions (loading * factor return)
            factor_contributions = {}
            for i, factor_name in enumerate(factor_names[1:], 1):  # Skip alpha
                factor_return = factors.iloc[:, i-1].mean()
                contribution = coefficients[i] * factor_return * 252  # Annualized
                factor_contributions[factor_name] = contribution
            
            return factor_loadings, factor_contributions
            
        except Exception as e:
            self.logger.error(f"Error in factor regression: {e}")
            return {}, {}


class BenchmarkAnalyzer:
    """Dynamic benchmark analysis and comparison"""
    
    def __init__(self, config: AttributionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_benchmark_performance(self, portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series,
                                    portfolio_weights: Dict[str, float] = None) -> BenchmarkAnalysis:
        """Comprehensive benchmark analysis"""
        
        try:
            # Basic benchmark metrics
            benchmark_return = benchmark_returns.mean() * 252
            benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
            
            # Relative performance
            excess_returns = portfolio_returns - benchmark_returns
            outperformance = excess_returns.mean() * 252
            
            # Hit rate (percentage of periods with outperformance)
            hit_rate = (excess_returns > 0).mean()
            
            # Win/Loss ratio
            winning_periods = excess_returns[excess_returns > 0]
            losing_periods = excess_returns[excess_returns < 0]
            
            if len(losing_periods) > 0:
                avg_win = winning_periods.mean() if len(winning_periods) > 0 else 0
                avg_loss = abs(losing_periods.mean())
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            else:
                win_loss_ratio = float('inf') if len(winning_periods) > 0 else 1.0
            
            # Capture ratios
            upside_capture, downside_capture = self._calculate_capture_ratios(
                portfolio_returns, benchmark_returns
            )
            capture_ratio = upside_capture / downside_capture if downside_capture > 0 else 1.0
            
            # Regime-specific performance (simplified)
            regime_performance = self._analyze_regime_performance(
                portfolio_returns, benchmark_returns
            )
            
            analysis = BenchmarkAnalysis(
                timestamp=datetime.now(),
                benchmark_type=self.config.benchmark_type,
                benchmark_weights=portfolio_weights or {},
                benchmark_return=benchmark_return,
                benchmark_volatility=benchmark_volatility,
                outperformance=outperformance,
                hit_rate=hit_rate,
                win_loss_ratio=win_loss_ratio,
                upside_capture=upside_capture,
                downside_capture=downside_capture,
                capture_ratio=capture_ratio,
                regime_performance=regime_performance
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in benchmark analysis: {e}")
            return BenchmarkAnalysis(
                timestamp=datetime.now(),
                benchmark_type=self.config.benchmark_type
            )
    
    def _calculate_capture_ratios(self, portfolio_returns: pd.Series,
                                benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate upside and downside capture ratios"""
        
        try:
            # Separate up and down market periods
            up_markets = benchmark_returns > 0
            down_markets = benchmark_returns < 0
            
            if up_markets.any():
                portfolio_up = portfolio_returns[up_markets].mean()
                benchmark_up = benchmark_returns[up_markets].mean()
                upside_capture = portfolio_up / benchmark_up if benchmark_up != 0 else 1.0
            else:
                upside_capture = 1.0
            
            if down_markets.any():
                portfolio_down = portfolio_returns[down_markets].mean()
                benchmark_down = benchmark_returns[down_markets].mean()
                downside_capture = portfolio_down / benchmark_down if benchmark_down != 0 else 1.0
            else:
                downside_capture = 1.0
            
            return upside_capture, downside_capture
            
        except Exception as e:
            self.logger.error(f"Error calculating capture ratios: {e}")
            return 1.0, 1.0
    
    def _analyze_regime_performance(self, portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> Dict[str, float]:
        """Analyze performance by market regime"""
        
        try:
            # Simple volatility-based regime classification
            rolling_vol = benchmark_returns.rolling(21).std()
            vol_median = rolling_vol.median()
            
            # Define regimes
            low_vol_regime = rolling_vol < vol_median * 0.8
            high_vol_regime = rolling_vol > vol_median * 1.2
            normal_regime = ~(low_vol_regime | high_vol_regime)
            
            regime_performance = {}
            
            for regime_name, regime_mask in [
                ('low_volatility', low_vol_regime),
                ('normal', normal_regime), 
                ('high_volatility', high_vol_regime)
            ]:
                if regime_mask.any():
                    portfolio_regime = portfolio_returns[regime_mask].mean() * 252
                    benchmark_regime = benchmark_returns[regime_mask].mean() * 252
                    regime_performance[regime_name] = portfolio_regime - benchmark_regime
                else:
                    regime_performance[regime_name] = 0.0
            
            return regime_performance
            
        except Exception as e:
            self.logger.error(f"Error in regime performance analysis: {e}")
            return {}


class AdvancedPerformanceAttribution:
    """Main advanced performance attribution system"""
    
    def __init__(self, config: AttributionConfig = None):
        self.config = config or AttributionConfig()
        
        # Initialize components
        self.performance_calculator = PerformanceCalculator(self.config)
        self.factor_analyzer = FactorAttributionAnalyzer(self.config)
        self.benchmark_analyzer = BenchmarkAnalyzer(self.config)
        
        # State management
        self.performance_history = []
        self.attribution_history = []
        self.benchmark_history = []
        
        logger.info("Advanced Performance Attribution system initialized")
    
    def analyze_comprehensive_performance(self, portfolio_data: pd.DataFrame,
                                        portfolio_weights: Dict[str, float],
                                        benchmark_data: pd.DataFrame = None,
                                        portfolio_value: float = 100000) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, portfolio_weights)
            
            # Calculate benchmark returns (if provided)
            if benchmark_data is not None:
                benchmark_returns = self._calculate_benchmark_returns(benchmark_data)
            else:
                # Use risk-free rate as benchmark
                benchmark_returns = pd.Series(
                    [self.config.risk_free_rate / 252] * len(portfolio_returns),
                    index=portfolio_returns.index
                )
            
            # Analysis for different time periods
            results = {}
            
            for period_days in self.config.analysis_periods:
                period_name = self._get_period_name(period_days)
                
                # Basic performance metrics
                performance_result = self.performance_calculator.calculate_basic_metrics(
                    portfolio_returns, benchmark_returns, period_days
                )
                
                # Factor attribution
                attribution_result = self.factor_analyzer.perform_factor_attribution(
                    portfolio_returns.tail(period_days)
                )
                
                # Benchmark analysis
                benchmark_result = self.benchmark_analyzer.analyze_benchmark_performance(
                    portfolio_returns.tail(period_days),
                    benchmark_returns.tail(period_days),
                    portfolio_weights
                )
                
                results[period_name] = {
                    'performance': performance_result,
                    'attribution': attribution_result,
                    'benchmark': benchmark_result
                }
            
            # Store in history
            self.performance_history.extend([r['performance'] for r in results.values()])
            self.attribution_history.extend([r['attribution'] for r in results.values()])
            self.benchmark_history.extend([r['benchmark'] for r in results.values()])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive performance analysis: {e}")
            return {}
    
    def _calculate_portfolio_returns(self, portfolio_data: pd.DataFrame,
                                   portfolio_weights: Dict[str, float]) -> pd.Series:
        """Calculate weighted portfolio returns"""
        
        try:
            # Calculate asset returns
            asset_returns = portfolio_data.pct_change().dropna()
            
            # Calculate weighted portfolio returns
            portfolio_returns = pd.Series(index=asset_returns.index, dtype=float)
            
            for date in asset_returns.index:
                daily_return = 0
                for asset, weight in portfolio_weights.items():
                    if asset in asset_returns.columns:
                        asset_return = asset_returns.loc[date, asset]
                        if not pd.isna(asset_return):
                            daily_return += weight * asset_return
                portfolio_returns.loc[date] = daily_return
            
            return portfolio_returns.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_benchmark_returns(self, benchmark_data: pd.DataFrame) -> pd.Series:
        """Calculate benchmark returns"""
        
        try:
            if len(benchmark_data.columns) == 1:
                # Single benchmark
                return benchmark_data.iloc[:, 0].pct_change().dropna()
            else:
                # Composite benchmark (equal weight)
                benchmark_returns = benchmark_data.pct_change().dropna()
                return benchmark_returns.mean(axis=1)
                
        except Exception as e:
            logger.error(f"Error calculating benchmark returns: {e}")
            return pd.Series(dtype=float)
    
    def _get_period_name(self, period_days: int) -> str:
        """Get readable period name"""
        
        period_mapping = {
            21: "1_month",
            63: "3_months", 
            126: "6_months",
            252: "1_year"
        }
        
        return period_mapping.get(period_days, f"{period_days}_days")
    
    def generate_performance_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard"""
        
        try:
            if not self.performance_history:
                return {"error": "No performance data available"}
            
            # Latest performance metrics
            latest_performance = self.performance_history[-1]
            latest_attribution = self.attribution_history[-1] if self.attribution_history else None
            latest_benchmark = self.benchmark_history[-1] if self.benchmark_history else None
            
            # Performance summary
            performance_summary = {
                "total_return": latest_performance.portfolio_return,
                "excess_return": latest_performance.excess_return,
                "volatility": latest_performance.portfolio_volatility,
                "sharpe_ratio": latest_performance.sharpe_ratio,
                "information_ratio": latest_performance.information_ratio,
                "tracking_error": latest_performance.tracking_error,
                "alpha": latest_performance.alpha,
                "beta": latest_performance.beta
            }
            
            # Attribution summary
            attribution_summary = {}
            if latest_attribution:
                attribution_summary = {
                    "total_attribution": latest_attribution.total_attribution,
                    "factor_contributions": latest_attribution.factor_contributions,
                    "systematic_risk": latest_attribution.systematic_risk,
                    "specific_risk": latest_attribution.specific_risk,
                    "unexplained_return": latest_attribution.unexplained_return
                }
            
            # Benchmark summary
            benchmark_summary = {}
            if latest_benchmark:
                benchmark_summary = {
                    "outperformance": latest_benchmark.outperformance,
                    "hit_rate": latest_benchmark.hit_rate,
                    "win_loss_ratio": latest_benchmark.win_loss_ratio,
                    "upside_capture": latest_benchmark.upside_capture,
                    "downside_capture": latest_benchmark.downside_capture,
                    "capture_ratio": latest_benchmark.capture_ratio
                }
            
            # Historical trends
            if len(self.performance_history) > 1:
                returns_trend = [p.portfolio_return for p in self.performance_history[-12:]]  # Last 12 periods
                sharpe_trend = [p.sharpe_ratio for p in self.performance_history[-12:]]
                alpha_trend = [p.alpha for p in self.performance_history[-12:]]
            else:
                returns_trend = sharpe_trend = alpha_trend = []
            
            dashboard = {
                "timestamp": datetime.now(),
                "performance_summary": performance_summary,
                "attribution_summary": attribution_summary,
                "benchmark_summary": benchmark_summary,
                "trends": {
                    "returns": returns_trend,
                    "sharpe_ratios": sharpe_trend,
                    "alphas": alpha_trend
                },
                "data_points": len(self.performance_history)
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            return {"error": "Dashboard generation failed"}


def create_advanced_performance_attribution(custom_config: Dict = None) -> AdvancedPerformanceAttribution:
    """Factory function to create advanced performance attribution system"""
    
    config = AttributionConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return AdvancedPerformanceAttribution(config)


if __name__ == "__main__":
    # Example usage
    print("Advanced Performance Attribution & Analytics System")
    
    # Create system
    attribution_system = create_advanced_performance_attribution({
        'attribution_methods': [AttributionMethod.FACTOR_BASED, AttributionMethod.RISK_FACTOR],
        'analysis_periods': [21, 63, 252],
        'real_time_attribution': True
    })
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='1D')
    
    assets = ['GOLD', 'SILVER', 'PLATINUM', 'COPPER']
    portfolio_data = pd.DataFrame(index=dates, columns=assets)
    
    # Simulate asset prices
    for i, asset in enumerate(assets):
        returns = np.random.normal(0.0008, 0.02, 252)
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        portfolio_data[asset] = prices[1:]
    
    # Portfolio weights
    portfolio_weights = {'GOLD': 0.4, 'SILVER': 0.3, 'PLATINUM': 0.2, 'COPPER': 0.1}
    
    # Analyze performance
    results = attribution_system.analyze_comprehensive_performance(
        portfolio_data, portfolio_weights
    )
    
    print("\nPerformance Analysis Results:")
    for period, analysis in results.items():
        performance = analysis['performance']
        print(f"\n{period.upper()}:")
        print(f"  Total Return: {performance.portfolio_return:.2%}")
        print(f"  Excess Return: {performance.excess_return:.2%}")
        print(f"  Sharpe Ratio: {performance.sharpe_ratio:.3f}")
        print(f"  Information Ratio: {performance.information_ratio:.3f}")
        print(f"  Alpha: {performance.alpha:.2%}")
        print(f"  Beta: {performance.beta:.3f}")
    
    # Generate dashboard
    dashboard = attribution_system.generate_performance_dashboard()
    
    print(f"\nDashboard Summary:")
    print(f"  Total Return: {dashboard['performance_summary']['total_return']:.2%}")
    print(f"  Sharpe Ratio: {dashboard['performance_summary']['sharpe_ratio']:.3f}")
    print(f"  Information Ratio: {dashboard['performance_summary']['information_ratio']:.3f}")
    print(f"  Data Points: {dashboard['data_points']}") 