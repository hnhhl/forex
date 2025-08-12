"""
VaR Backtester System
Comprehensive backtesting and validation framework for VaR models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..base_system import BaseSystem
from .var_calculator import VaRResult, VaRMethod
from .risk_types import RiskLevel, RiskMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestType(Enum):
    """Types of VaR backtests"""
    KUPIEC = "kupiec"
    CHRISTOFFERSEN = "christoffersen"
    MIXED_KUPIEC = "mixed_kupiec"
    DYNAMIC_QUANTILE = "dynamic_quantile"
    DURATION_BASED = "duration_based"
    TRAFFIC_LIGHT = "traffic_light"


class BacktestResult(Enum):
    """Backtest result classification"""
    ACCEPT = "accept"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


@dataclass
class BacktestStatistic:
    """Individual backtest statistic"""
    test_name: str
    test_statistic: float
    p_value: float
    critical_value: float
    result: BacktestResult
    confidence_level: float
    description: str


@dataclass
class VaRBacktestReport:
    """Comprehensive VaR backtest report"""
    model_name: str
    var_method: VaRMethod
    confidence_level: float
    test_period_start: datetime
    test_period_end: datetime
    total_observations: int
    total_violations: int
    expected_violations: int
    violation_rate: float
    expected_violation_rate: float
    
    # Test statistics
    kupiec_test: BacktestStatistic
    christoffersen_test: BacktestStatistic
    mixed_kupiec_test: Optional[BacktestStatistic]
    duration_test: Optional[BacktestStatistic]
    
    # Additional metrics
    violation_dates: List[datetime]
    violation_magnitudes: List[float]
    max_violation: float
    avg_violation: float
    
    # Traffic light classification
    traffic_light_zone: str  # Green, Yellow, Red
    
    # Overall assessment
    overall_result: BacktestResult
    recommendations: List[str]
    
    timestamp: datetime


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    confidence_level: float = 0.95
    test_window: int = 252  # 1 year
    min_observations: int = 100
    significance_level: float = 0.05
    tests_to_run: List[BacktestType] = None
    
    def __post_init__(self):
        if self.tests_to_run is None:
            self.tests_to_run = [BacktestType.KUPIEC, BacktestType.CHRISTOFFERSEN, BacktestType.TRAFFIC_LIGHT]


class VaRBacktester(BaseSystem):
    """
    Advanced VaR Backtesting System
    
    Implements comprehensive backtesting framework:
    - Kupiec Test (Unconditional Coverage)
    - Christoffersen Test (Independence)
    - Mixed Kupiec Test (Conditional Coverage)
    - Duration-Based Tests
    - Traffic Light System
    - Dynamic Quantile Tests
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize VaR Backtester"""
        super().__init__("VaRBacktester", config)
        
        # Configuration
        self.default_config = BacktestConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.default_config, key):
                    setattr(self.default_config, key, value)
        
        # Data storage
        self.backtest_history: List[VaRBacktestReport] = []
        self.violation_database: Dict[str, List[Tuple[datetime, float, float]]] = {}  # model -> [(date, actual, var)]
        
        logger.info(f"VaRBacktester initialized with {len(self.default_config.tests_to_run)} default tests")
    
    def backtest_var_model(self, 
                          var_forecasts: List[VaRResult],
                          actual_returns: pd.Series,
                          model_name: str = "VaR_Model",
                          config: Optional[BacktestConfig] = None) -> VaRBacktestReport:
        """Comprehensive VaR model backtesting"""
        try:
            test_config = config or self.default_config
            
            if len(var_forecasts) == 0 or len(actual_returns) == 0:
                raise ValueError("Insufficient data for backtesting")
            
            # Align data
            min_length = min(len(var_forecasts), len(actual_returns))
            if min_length < test_config.min_observations:
                raise ValueError(f"Insufficient observations: {min_length} < {test_config.min_observations}")
            
            var_values = [result.var_value for result in var_forecasts[-min_length:]]
            returns = actual_returns.iloc[-min_length:].values
            dates = actual_returns.index[-min_length:]
            
            # Identify violations
            violations = []
            violation_dates = []
            violation_magnitudes = []
            
            for i, (ret, var_val) in enumerate(zip(returns, var_values)):
                if ret < var_val:  # Actual loss exceeds VaR
                    violations.append(1)
                    violation_dates.append(dates[i])
                    violation_magnitudes.append(abs(ret - var_val))
                else:
                    violations.append(0)
            
            violations = np.array(violations)
            total_violations = np.sum(violations)
            confidence_level = var_forecasts[0].confidence_level
            expected_violations = int((1 - confidence_level) * len(returns))
            violation_rate = total_violations / len(returns)
            expected_violation_rate = 1 - confidence_level
            
            # Run statistical tests
            kupiec_test = self._kupiec_test(violations, confidence_level, test_config.significance_level)
            christoffersen_test = self._christoffersen_test(violations, confidence_level, test_config.significance_level)
            
            # Optional tests
            mixed_kupiec_test = None
            duration_test = None
            
            if BacktestType.MIXED_KUPIEC in test_config.tests_to_run:
                mixed_kupiec_test = self._mixed_kupiec_test(violations, confidence_level, test_config.significance_level)
            
            if BacktestType.DURATION_BASED in test_config.tests_to_run:
                duration_test = self._duration_based_test(violations, confidence_level, test_config.significance_level)
            
            # Traffic light system
            traffic_light_zone = self._traffic_light_classification(total_violations, expected_violations)
            
            # Overall assessment
            overall_result = self._assess_overall_result([kupiec_test, christoffersen_test])
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                kupiec_test, christoffersen_test, traffic_light_zone, violation_rate, expected_violation_rate
            )
            
            # Create report
            report = VaRBacktestReport(
                model_name=model_name,
                var_method=var_forecasts[0].method,
                confidence_level=confidence_level,
                test_period_start=dates[0],
                test_period_end=dates[-1],
                total_observations=len(returns),
                total_violations=total_violations,
                expected_violations=expected_violations,
                violation_rate=violation_rate,
                expected_violation_rate=expected_violation_rate,
                kupiec_test=kupiec_test,
                christoffersen_test=christoffersen_test,
                mixed_kupiec_test=mixed_kupiec_test,
                duration_test=duration_test,
                violation_dates=violation_dates,
                violation_magnitudes=violation_magnitudes,
                max_violation=max(violation_magnitudes) if violation_magnitudes else 0.0,
                avg_violation=np.mean(violation_magnitudes) if violation_magnitudes else 0.0,
                traffic_light_zone=traffic_light_zone,
                overall_result=overall_result,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Store results
            self.backtest_history.append(report)
            
            # Update violation database
            if model_name not in self.violation_database:
                self.violation_database[model_name] = []
            
            for i, (ret, var_val) in enumerate(zip(returns, var_values)):
                self.violation_database[model_name].append((dates[i], ret, var_val))
            
            logger.info(f"Backtesting completed for {model_name}: {total_violations}/{len(returns)} violations")
            
            return report
            
        except Exception as e:
            logger.error(f"Error backtesting VaR model: {e}")
            raise
    
    def _kupiec_test(self, violations: np.ndarray, confidence_level: float, significance_level: float) -> BacktestStatistic:
        """Kupiec Test for Unconditional Coverage"""
        try:
            n = len(violations)
            x = np.sum(violations)  # number of violations
            p = 1 - confidence_level  # expected violation rate
            
            if x == 0:
                # No violations - test statistic is 0
                test_statistic = 0.0
                p_value = 1.0
            elif x == n:
                # All violations - test statistic is infinity (reject)
                test_statistic = np.inf
                p_value = 0.0
            else:
                # Likelihood ratio test statistic
                p_hat = x / n  # observed violation rate
                
                if p_hat > 0 and p_hat < 1:
                    lr = -2 * (x * np.log(p) + (n - x) * np.log(1 - p) - 
                              x * np.log(p_hat) - (n - x) * np.log(1 - p_hat))
                    test_statistic = lr
                    p_value = 1 - stats.chi2.cdf(lr, 1)
                else:
                    test_statistic = np.inf
                    p_value = 0.0
            
            critical_value = stats.chi2.ppf(1 - significance_level, 1)
            result = BacktestResult.ACCEPT if p_value > significance_level else BacktestResult.REJECT
            
            return BacktestStatistic(
                test_name="Kupiec Test (Unconditional Coverage)",
                test_statistic=test_statistic,
                p_value=p_value,
                critical_value=critical_value,
                result=result,
                confidence_level=confidence_level,
                description=f"Tests if violation rate equals expected rate ({1-confidence_level:.1%})"
            )
            
        except Exception as e:
            logger.error(f"Error in Kupiec test: {e}")
            return BacktestStatistic("Kupiec Test", 0, 1, 0, BacktestResult.INCONCLUSIVE, confidence_level, "Error in calculation")
    
    def _christoffersen_test(self, violations: np.ndarray, confidence_level: float, significance_level: float) -> BacktestStatistic:
        """Christoffersen Test for Independence"""
        try:
            # Count transitions
            n00 = n01 = n10 = n11 = 0
            
            for i in range(1, len(violations)):
                if violations[i-1] == 0 and violations[i] == 0:
                    n00 += 1
                elif violations[i-1] == 0 and violations[i] == 1:
                    n01 += 1
                elif violations[i-1] == 1 and violations[i] == 0:
                    n10 += 1
                elif violations[i-1] == 1 and violations[i] == 1:
                    n11 += 1
            
            # Calculate test statistic
            if n01 + n11 == 0 or n00 + n10 == 0:
                # No violations or all violations
                test_statistic = 0.0
                p_value = 1.0
            else:
                # Transition probabilities
                pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
                pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
                pi = (n01 + n11) / (n00 + n01 + n10 + n11)
                
                if pi_01 > 0 and pi_11 > 0 and pi > 0 and pi < 1:
                    # Likelihood ratio for independence
                    lr_ind = -2 * (n00 * np.log(1 - pi) + n01 * np.log(pi) + 
                                  n10 * np.log(1 - pi) + n11 * np.log(pi) -
                                  n00 * np.log(1 - pi_01) - n01 * np.log(pi_01) -
                                  n10 * np.log(1 - pi_11) - n11 * np.log(pi_11))
                    
                    test_statistic = lr_ind
                    p_value = 1 - stats.chi2.cdf(lr_ind, 1)
                else:
                    test_statistic = 0.0
                    p_value = 1.0
            
            critical_value = stats.chi2.ppf(1 - significance_level, 1)
            result = BacktestResult.ACCEPT if p_value > significance_level else BacktestResult.REJECT
            
            return BacktestStatistic(
                test_name="Christoffersen Test (Independence)",
                test_statistic=test_statistic,
                p_value=p_value,
                critical_value=critical_value,
                result=result,
                confidence_level=confidence_level,
                description="Tests if violations are independent over time"
            )
            
        except Exception as e:
            logger.error(f"Error in Christoffersen test: {e}")
            return BacktestStatistic("Christoffersen Test", 0, 1, 0, BacktestResult.INCONCLUSIVE, confidence_level, "Error in calculation")
    
    def _mixed_kupiec_test(self, violations: np.ndarray, confidence_level: float, significance_level: float) -> BacktestStatistic:
        """Mixed Kupiec Test for Conditional Coverage"""
        try:
            # This combines unconditional coverage and independence tests
            kupiec_result = self._kupiec_test(violations, confidence_level, significance_level)
            christoffersen_result = self._christoffersen_test(violations, confidence_level, significance_level)
            
            # Combined test statistic (sum of individual test statistics)
            test_statistic = kupiec_result.test_statistic + christoffersen_result.test_statistic
            p_value = 1 - stats.chi2.cdf(test_statistic, 2)  # 2 degrees of freedom
            
            critical_value = stats.chi2.ppf(1 - significance_level, 2)
            result = BacktestResult.ACCEPT if p_value > significance_level else BacktestResult.REJECT
            
            return BacktestStatistic(
                test_name="Mixed Kupiec Test (Conditional Coverage)",
                test_statistic=test_statistic,
                p_value=p_value,
                critical_value=critical_value,
                result=result,
                confidence_level=confidence_level,
                description="Combined test for correct coverage and independence"
            )
            
        except Exception as e:
            logger.error(f"Error in Mixed Kupiec test: {e}")
            return BacktestStatistic("Mixed Kupiec Test", 0, 1, 0, BacktestResult.INCONCLUSIVE, confidence_level, "Error in calculation")
    
    def _duration_based_test(self, violations: np.ndarray, confidence_level: float, significance_level: float) -> BacktestStatistic:
        """Duration-Based Test"""
        try:
            # Calculate durations between violations
            violation_indices = np.where(violations == 1)[0]
            
            if len(violation_indices) < 2:
                # Not enough violations for duration test
                return BacktestStatistic(
                    "Duration-Based Test", 0, 1, 0, BacktestResult.INCONCLUSIVE, 
                    confidence_level, "Insufficient violations for duration test"
                )
            
            durations = np.diff(violation_indices)
            expected_duration = 1 / (1 - confidence_level)
            
            # Test if durations follow geometric distribution
            # Using Kolmogorov-Smirnov test
            from scipy.stats import kstest, geom
            
            # Geometric distribution parameter
            p = 1 - confidence_level
            
            # KS test
            ks_statistic, p_value = kstest(durations, lambda x: geom.cdf(x, p))
            
            critical_value = 0.05  # Standard KS critical value approximation
            result = BacktestResult.ACCEPT if p_value > significance_level else BacktestResult.REJECT
            
            return BacktestStatistic(
                test_name="Duration-Based Test",
                test_statistic=ks_statistic,
                p_value=p_value,
                critical_value=critical_value,
                result=result,
                confidence_level=confidence_level,
                description="Tests if time between violations follows geometric distribution"
            )
            
        except Exception as e:
            logger.error(f"Error in Duration-Based test: {e}")
            return BacktestStatistic("Duration-Based Test", 0, 1, 0, BacktestResult.INCONCLUSIVE, confidence_level, "Error in calculation")
    
    def _traffic_light_classification(self, actual_violations: int, expected_violations: int) -> str:
        """Basel Traffic Light System Classification"""
        try:
            if expected_violations == 0:
                return "Green"
            
            # Basel Committee thresholds (simplified)
            if actual_violations <= expected_violations + 1:
                return "Green"
            elif actual_violations <= expected_violations + 5:  # More lenient for Yellow zone
                return "Yellow"
            else:
                return "Red"
                
        except Exception as e:
            logger.error(f"Error in traffic light classification: {e}")
            return "Unknown"
    
    def _assess_overall_result(self, test_results: List[BacktestStatistic]) -> BacktestResult:
        """Assess overall backtest result"""
        try:
            reject_count = sum(1 for test in test_results if test.result == BacktestResult.REJECT)
            inconclusive_count = sum(1 for test in test_results if test.result == BacktestResult.INCONCLUSIVE)
            
            if reject_count > 0:
                return BacktestResult.REJECT
            elif inconclusive_count > 0:
                return BacktestResult.INCONCLUSIVE
            else:
                return BacktestResult.ACCEPT
                
        except Exception as e:
            logger.error(f"Error assessing overall result: {e}")
            return BacktestResult.INCONCLUSIVE
    
    def _generate_recommendations(self, 
                                kupiec_test: BacktestStatistic,
                                christoffersen_test: BacktestStatistic,
                                traffic_light: str,
                                violation_rate: float,
                                expected_rate: float) -> List[str]:
        """Generate recommendations based on backtest results"""
        recommendations = []
        
        try:
            # Kupiec test recommendations
            if kupiec_test.result == BacktestResult.REJECT:
                if violation_rate > expected_rate:
                    recommendations.append("Model underestimates risk - consider increasing VaR estimates")
                else:
                    recommendations.append("Model overestimates risk - consider decreasing VaR estimates")
            
            # Christoffersen test recommendations
            if christoffersen_test.result == BacktestResult.REJECT:
                recommendations.append("Violations show clustering - consider using time-varying volatility models")
            
            # Traffic light recommendations
            if traffic_light == "Red":
                recommendations.append("Model fails regulatory standards - immediate model revision required")
            elif traffic_light == "Yellow":
                recommendations.append("Model performance is concerning - monitor closely and consider improvements")
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("Model performance is acceptable - continue monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple VaR models"""
        try:
            if not model_names:
                return {}
            
            comparison = {
                'models': model_names,
                'comparison_metrics': {},
                'rankings': {},
                'summary': {}
            }
            
            # Get latest reports for each model
            model_reports = {}
            for model_name in model_names:
                reports = [r for r in self.backtest_history if r.model_name == model_name]
                if reports:
                    model_reports[model_name] = reports[-1]  # Latest report
            
            if not model_reports:
                return comparison
            
            # Compare metrics
            metrics = ['violation_rate', 'kupiec_test', 'christoffersen_test', 'traffic_light_zone']
            
            for metric in metrics:
                comparison['comparison_metrics'][metric] = {}
                for model_name, report in model_reports.items():
                    if metric == 'kupiec_test':
                        comparison['comparison_metrics'][metric][model_name] = report.kupiec_test.p_value
                    elif metric == 'christoffersen_test':
                        comparison['comparison_metrics'][metric][model_name] = report.christoffersen_test.p_value
                    else:
                        comparison['comparison_metrics'][metric][model_name] = getattr(report, metric)
            
            # Rank models
            # Higher p-values are better for statistical tests
            kupiec_ranking = sorted(model_reports.items(), 
                                  key=lambda x: x[1].kupiec_test.p_value, reverse=True)
            comparison['rankings']['kupiec_test'] = [model for model, _ in kupiec_ranking]
            
            christoffersen_ranking = sorted(model_reports.items(), 
                                          key=lambda x: x[1].christoffersen_test.p_value, reverse=True)
            comparison['rankings']['christoffersen_test'] = [model for model, _ in christoffersen_ranking]
            
            # Overall ranking (simplified)
            overall_scores = {}
            for model_name, report in model_reports.items():
                score = 0
                if report.kupiec_test.result == BacktestResult.ACCEPT:
                    score += 1
                if report.christoffersen_test.result == BacktestResult.ACCEPT:
                    score += 1
                if report.traffic_light_zone == "Green":
                    score += 1
                overall_scores[model_name] = score
            
            overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            comparison['rankings']['overall'] = [model for model, _ in overall_ranking]
            
            # Summary
            best_model = overall_ranking[0][0] if overall_ranking else None
            comparison['summary'] = {
                'best_model': best_model,
                'total_models_compared': len(model_reports),
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get comprehensive backtesting summary"""
        try:
            if not self.backtest_history:
                return {'message': 'No backtests performed yet'}
            
            recent_reports = self.backtest_history[-10:]  # Last 10 reports
            
            summary = {
                'total_backtests': len(self.backtest_history),
                'unique_models': len(set(r.model_name for r in self.backtest_history)),
                'recent_results': [],
                'performance_summary': {},
                'recommendations_summary': {}
            }
            
            # Recent results
            for report in recent_reports:
                summary['recent_results'].append({
                    'model_name': report.model_name,
                    'timestamp': report.timestamp.isoformat(),
                    'confidence_level': report.confidence_level,
                    'violation_rate': report.violation_rate,
                    'expected_rate': report.expected_violation_rate,
                    'kupiec_result': report.kupiec_test.result.value,
                    'christoffersen_result': report.christoffersen_test.result.value,
                    'traffic_light': report.traffic_light_zone,
                    'overall_result': report.overall_result.value
                })
            
            # Performance summary
            accept_count = sum(1 for r in recent_reports if r.overall_result == BacktestResult.ACCEPT)
            reject_count = sum(1 for r in recent_reports if r.overall_result == BacktestResult.REJECT)
            
            summary['performance_summary'] = {
                'accept_rate': accept_count / len(recent_reports) if recent_reports else 0,
                'reject_rate': reject_count / len(recent_reports) if recent_reports else 0,
                'avg_violation_rate': np.mean([r.violation_rate for r in recent_reports]),
                'green_light_rate': sum(1 for r in recent_reports if r.traffic_light_zone == "Green") / len(recent_reports) if recent_reports else 0
            }
            
            # Common recommendations
            all_recommendations = []
            for report in recent_reports:
                all_recommendations.extend(report.recommendations)
            
            from collections import Counter
            recommendation_counts = Counter(all_recommendations)
            summary['recommendations_summary'] = dict(recommendation_counts.most_common(5))
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating backtest summary: {e}")
            return {}
    
    def export_backtest_data(self, filepath: str) -> bool:
        """Export backtest data to file"""
        try:
            export_data = {
                'backtest_history': [
                    {
                        'model_name': report.model_name,
                        'var_method': report.var_method.value,
                        'confidence_level': report.confidence_level,
                        'test_period_start': report.test_period_start.isoformat(),
                        'test_period_end': report.test_period_end.isoformat(),
                        'total_observations': int(report.total_observations),
                        'total_violations': int(report.total_violations),
                        'expected_violations': float(report.expected_violations),
                        'violation_rate': report.violation_rate,
                        'kupiec_test': {
                            'test_statistic': report.kupiec_test.test_statistic,
                            'p_value': report.kupiec_test.p_value,
                            'result': report.kupiec_test.result.value
                        },
                        'christoffersen_test': {
                            'test_statistic': report.christoffersen_test.test_statistic,
                            'p_value': report.christoffersen_test.p_value,
                            'result': report.christoffersen_test.result.value
                        },
                        'traffic_light_zone': report.traffic_light_zone,
                        'overall_result': report.overall_result.value,
                        'recommendations': report.recommendations,
                        'timestamp': report.timestamp.isoformat()
                    }
                    for report in self.backtest_history
                ],
                'violation_database': {
                    model: [(date.isoformat(), actual, var_val) for date, actual, var_val in violations]
                    for model, violations in self.violation_database.items()
                },
                'configuration': {
                    'default_confidence_level': self.default_config.confidence_level,
                    'default_test_window': self.default_config.test_window,
                    'default_significance_level': self.default_config.significance_level,
                    'default_tests': [test.value for test in self.default_config.tests_to_run]
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Backtest data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting backtest data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get VaR backtester statistics"""
        return {
            'total_backtests_performed': len(self.backtest_history),
            'unique_models_tested': len(set(r.model_name for r in self.backtest_history)),
            'models_in_violation_database': len(self.violation_database),
            'default_confidence_level': self.default_config.confidence_level,
            'default_test_window': self.default_config.test_window,
            'default_significance_level': self.default_config.significance_level,
            'available_tests': [test.value for test in BacktestType],
            'default_tests': [test.value for test in self.default_config.tests_to_run],
            'last_backtest': self.backtest_history[-1].timestamp.isoformat() if self.backtest_history else None
        } 