"""
Kelly Criterion Calculator
Specialized calculator for Kelly Criterion position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from scipy import stats

from ..base_system import BaseSystem

logger = logging.getLogger(__name__)


class KellyMethod(Enum):
    """Kelly calculation methods"""
    CLASSIC = "classic"
    FRACTIONAL = "fractional"
    CONTINUOUS = "continuous"
    MODIFIED = "modified"
    MULTI_ASSET = "multi_asset"


@dataclass
class KellyResult:
    """Kelly calculation result"""
    method: KellyMethod
    kelly_fraction: float
    safe_kelly: float
    recommended_size: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_ratio: float
    safety_factor: float
    confidence_level: float
    risk_metrics: Dict
    calculation_time: datetime
    additional_info: Dict


class KellyCalculator(BaseSystem):
    """
    Advanced Kelly Criterion Calculator
    
    Implements various Kelly Criterion methods:
    - Classic Kelly
    - Fractional Kelly
    - Continuous Kelly
    - Modified Kelly for multiple outcomes
    - Multi-asset Kelly
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("KellyCalculator", config)
        
        # Configuration
        self.config = config or {}
        self.default_safety_factor = self.config.get('safety_factor', 0.25)
        self.max_kelly = self.config.get('max_kelly', 0.20)  # 20% max
        self.min_kelly = self.config.get('min_kelly', 0.001)  # 0.1% min
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Data storage
        self.trade_results: List[float] = []
        self.win_rate: float = 0.0
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0
        self.profit_ratio: float = 0.0
        
        # Results storage
        self.kelly_results: Dict = {}
        self.calculation_history: List[KellyResult] = []
        
        logger.info("KellyCalculator initialized")
    
    def set_trade_data(self, trade_results: List[float]):
        """Set historical trade results"""
        try:
            self.trade_results = trade_results
            self._calculate_statistics()
            
            logger.info(f"Trade data set: {len(trade_results)} trades")
            
        except Exception as e:
            logger.error(f"Error setting trade data: {e}")
            raise
    
    def _calculate_statistics(self):
        """Calculate win rate, average win/loss from trade results"""
        try:
            if not self.trade_results:
                return
            
            wins = [r for r in self.trade_results if r > 0]
            losses = [r for r in self.trade_results if r < 0]
            
            self.win_rate = len(wins) / len(self.trade_results)
            self.avg_win = np.mean(wins) if wins else 0.0
            self.avg_loss = abs(np.mean(losses)) if losses else 0.0
            self.profit_ratio = self.avg_win / self.avg_loss if self.avg_loss > 0 else 0.0
            
            logger.info(f"Statistics calculated: Win rate {self.win_rate:.2%}, Profit ratio {self.profit_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
    
    def calculate_classic_kelly(self, win_rate: float = None, avg_win: float = None, 
                              avg_loss: float = None, safety_factor: float = None) -> KellyResult:
        """Calculate classic Kelly Criterion"""
        try:
            # Use provided values or calculated statistics
            win_rate = win_rate or self.win_rate
            avg_win = avg_win or self.avg_win
            avg_loss = avg_loss or self.avg_loss
            safety_factor = safety_factor or self.default_safety_factor
            
            if win_rate == 0 or avg_win == 0 or avg_loss == 0:
                raise ValueError("Insufficient data for Kelly calculation")
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety factor
            safe_kelly = kelly_fraction * safety_factor
            
            # Ensure within limits
            recommended_size = max(self.min_kelly, min(safe_kelly, self.max_kelly))
            
            # Calculate confidence based on sample size and win rate
            sample_size = len(self.trade_results) if self.trade_results else 100
            confidence = min(win_rate * np.sqrt(sample_size / 100), 1.0)
            
            result = KellyResult(
                method=KellyMethod.CLASSIC,
                kelly_fraction=kelly_fraction,
                safe_kelly=safe_kelly,
                recommended_size=recommended_size,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_ratio=b,
                safety_factor=safety_factor,
                confidence_level=confidence,
                risk_metrics={
                    'expected_value': p * avg_win - q * avg_loss,
                    'variance': p * (avg_win ** 2) + q * (avg_loss ** 2),
                    'sharpe_ratio': (p * avg_win - q * avg_loss) / np.sqrt(p * (avg_win ** 2) + q * (avg_loss ** 2)),
                    'sample_size': sample_size
                },
                calculation_time=datetime.now(),
                additional_info={
                    'method_description': 'Classic Kelly Criterion',
                    'formula': 'f = (bp - q) / b',
                    'full_kelly': kelly_fraction,
                    'applied_safety': safety_factor
                }
            )
            
            self.kelly_results['classic'] = result
            self.calculation_history.append(result)
            
            logger.info(f"Classic Kelly calculated: {recommended_size:.4f} (Full: {kelly_fraction:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating classic Kelly: {e}")
            raise
    
    def calculate_fractional_kelly(self, fraction: float = 0.25) -> KellyResult:
        """Calculate fractional Kelly (fixed fraction of full Kelly)"""
        try:
            # First calculate full Kelly
            full_kelly_result = self.calculate_classic_kelly(safety_factor=1.0)
            
            # Apply fractional scaling
            fractional_kelly = full_kelly_result.kelly_fraction * fraction
            recommended_size = max(self.min_kelly, min(fractional_kelly, self.max_kelly))
            
            result = KellyResult(
                method=KellyMethod.FRACTIONAL,
                kelly_fraction=full_kelly_result.kelly_fraction,
                safe_kelly=fractional_kelly,
                recommended_size=recommended_size,
                win_rate=full_kelly_result.win_rate,
                avg_win=full_kelly_result.avg_win,
                avg_loss=full_kelly_result.avg_loss,
                profit_ratio=full_kelly_result.profit_ratio,
                safety_factor=fraction,
                confidence_level=full_kelly_result.confidence_level,
                risk_metrics=full_kelly_result.risk_metrics.copy(),
                calculation_time=datetime.now(),
                additional_info={
                    'method_description': f'Fractional Kelly ({fraction:.0%})',
                    'fraction_applied': fraction,
                    'full_kelly': full_kelly_result.kelly_fraction
                }
            )
            
            self.kelly_results['fractional'] = result
            self.calculation_history.append(result)
            
            logger.info(f"Fractional Kelly calculated: {recommended_size:.4f} ({fraction:.0%} of full Kelly)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating fractional Kelly: {e}")
            raise
    
    def calculate_continuous_kelly(self, returns_series: pd.Series) -> KellyResult:
        """Calculate continuous Kelly for continuous returns"""
        try:
            if len(returns_series) < 10:
                raise ValueError("Insufficient data for continuous Kelly")
            
            # Calculate mean and variance of returns
            mean_return = returns_series.mean()
            variance_return = returns_series.var()
            
            if variance_return == 0:
                raise ValueError("Zero variance in returns")
            
            # Continuous Kelly formula: f = μ / σ²
            kelly_fraction = mean_return / variance_return
            
            # Apply safety factor
            safety_factor = self.default_safety_factor
            safe_kelly = kelly_fraction * safety_factor
            
            # Ensure within limits
            recommended_size = max(self.min_kelly, min(safe_kelly, self.max_kelly))
            
            # Calculate statistics for compatibility
            positive_returns = returns_series[returns_series > 0]
            negative_returns = returns_series[returns_series < 0]
            
            win_rate = len(positive_returns) / len(returns_series)
            avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
            avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.0
            
            # Confidence based on Sharpe ratio and sample size
            sharpe_ratio = mean_return / np.sqrt(variance_return)
            confidence = min(abs(sharpe_ratio) * np.sqrt(len(returns_series) / 252), 1.0)
            
            result = KellyResult(
                method=KellyMethod.CONTINUOUS,
                kelly_fraction=kelly_fraction,
                safe_kelly=safe_kelly,
                recommended_size=recommended_size,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_ratio=avg_win / avg_loss if avg_loss > 0 else 0.0,
                safety_factor=safety_factor,
                confidence_level=confidence,
                risk_metrics={
                    'mean_return': mean_return,
                    'variance_return': variance_return,
                    'sharpe_ratio': sharpe_ratio,
                    'skewness': returns_series.skew(),
                    'kurtosis': returns_series.kurtosis(),
                    'sample_size': len(returns_series)
                },
                calculation_time=datetime.now(),
                additional_info={
                    'method_description': 'Continuous Kelly for continuous returns',
                    'formula': 'f = μ / σ²',
                    'return_distribution': 'continuous'
                }
            )
            
            self.kelly_results['continuous'] = result
            self.calculation_history.append(result)
            
            logger.info(f"Continuous Kelly calculated: {recommended_size:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating continuous Kelly: {e}")
            raise
    
    def calculate_modified_kelly(self, outcome_probabilities: Dict[float, float]) -> KellyResult:
        """Calculate modified Kelly for multiple outcomes"""
        try:
            if not outcome_probabilities or abs(sum(outcome_probabilities.values()) - 1.0) > 0.01:
                raise ValueError("Invalid probability distribution")
            
            # Modified Kelly for multiple outcomes
            # f = Σ(pi * ri) / Σ(pi * ri²) where pi = probability, ri = return
            
            numerator = 0.0
            denominator = 0.0
            expected_return = 0.0
            
            for outcome, probability in outcome_probabilities.items():
                numerator += probability * outcome
                denominator += probability * (outcome ** 2)
                expected_return += probability * outcome
            
            if denominator == 0:
                raise ValueError("Zero denominator in modified Kelly calculation")
            
            kelly_fraction = numerator / denominator
            
            # Apply safety factor
            safety_factor = self.default_safety_factor
            safe_kelly = kelly_fraction * safety_factor
            
            # Ensure within limits
            recommended_size = max(self.min_kelly, min(safe_kelly, self.max_kelly))
            
            # Calculate equivalent win rate and profit metrics
            positive_outcomes = {k: v for k, v in outcome_probabilities.items() if k > 0}
            negative_outcomes = {k: v for k, v in outcome_probabilities.items() if k < 0}
            
            win_rate = sum(positive_outcomes.values())
            avg_win = sum(k * v for k, v in positive_outcomes.items()) / win_rate if win_rate > 0 else 0.0
            avg_loss = abs(sum(k * v for k, v in negative_outcomes.items()) / sum(negative_outcomes.values())) if negative_outcomes else 0.0
            
            # Confidence based on expected return and variance
            variance = sum(p * (r - expected_return) ** 2 for r, p in outcome_probabilities.items())
            confidence = min(abs(expected_return) / np.sqrt(variance) if variance > 0 else 0.0, 1.0)
            
            result = KellyResult(
                method=KellyMethod.MODIFIED,
                kelly_fraction=kelly_fraction,
                safe_kelly=safe_kelly,
                recommended_size=recommended_size,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_ratio=avg_win / avg_loss if avg_loss > 0 else 0.0,
                safety_factor=safety_factor,
                confidence_level=confidence,
                risk_metrics={
                    'expected_return': expected_return,
                    'variance': variance,
                    'num_outcomes': len(outcome_probabilities),
                    'outcome_distribution': outcome_probabilities
                },
                calculation_time=datetime.now(),
                additional_info={
                    'method_description': 'Modified Kelly for multiple outcomes',
                    'formula': 'f = Σ(pi * ri) / Σ(pi * ri²)',
                    'outcomes_count': len(outcome_probabilities)
                }
            )
            
            self.kelly_results['modified'] = result
            self.calculation_history.append(result)
            
            logger.info(f"Modified Kelly calculated: {recommended_size:.4f} ({len(outcome_probabilities)} outcomes)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating modified Kelly: {e}")
            raise
    
    def calculate_multi_asset_kelly(self, asset_returns: Dict[str, pd.Series], 
                                  correlation_matrix: pd.DataFrame = None) -> Dict[str, KellyResult]:
        """Calculate Kelly for multiple correlated assets"""
        try:
            if not asset_returns:
                raise ValueError("No asset returns provided")
            
            # Calculate correlation matrix if not provided
            if correlation_matrix is None:
                returns_df = pd.DataFrame(asset_returns)
                correlation_matrix = returns_df.corr()
            
            # Calculate mean returns and covariance matrix
            mean_returns = {}
            for asset, returns in asset_returns.items():
                mean_returns[asset] = returns.mean()
            
            returns_df = pd.DataFrame(asset_returns)
            cov_matrix = returns_df.cov()
            
            # Multi-asset Kelly: f = Σ⁻¹ * μ where Σ is covariance matrix, μ is mean returns vector
            try:
                inv_cov_matrix = np.linalg.inv(cov_matrix.values)
                mean_returns_vector = np.array(list(mean_returns.values()))
                
                kelly_weights = inv_cov_matrix @ mean_returns_vector
                
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                inv_cov_matrix = np.linalg.pinv(cov_matrix.values)
                mean_returns_vector = np.array(list(mean_returns.values()))
                kelly_weights = inv_cov_matrix @ mean_returns_vector
            
            # Apply safety factor and normalize
            safety_factor = self.default_safety_factor
            safe_weights = kelly_weights * safety_factor
            
            # Ensure positive weights and within limits
            results = {}
            asset_names = list(asset_returns.keys())
            
            for i, asset in enumerate(asset_names):
                weight = max(0, min(safe_weights[i], self.max_kelly))
                
                # Calculate individual asset statistics
                returns_series = asset_returns[asset]
                positive_returns = returns_series[returns_series > 0]
                negative_returns = returns_series[returns_series < 0]
                
                win_rate = len(positive_returns) / len(returns_series)
                avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
                avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.0
                
                # Confidence based on Sharpe ratio
                sharpe_ratio = mean_returns[asset] / returns_series.std()
                confidence = min(abs(sharpe_ratio) * 0.5, 1.0)
                
                result = KellyResult(
                    method=KellyMethod.MULTI_ASSET,
                    kelly_fraction=kelly_weights[i],
                    safe_kelly=safe_weights[i],
                    recommended_size=weight,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    profit_ratio=avg_win / avg_loss if avg_loss > 0 else 0.0,
                    safety_factor=safety_factor,
                    confidence_level=confidence,
                    risk_metrics={
                        'mean_return': mean_returns[asset],
                        'volatility': returns_series.std(),
                        'sharpe_ratio': sharpe_ratio,
                        'correlation_with_others': correlation_matrix[asset].drop(asset).mean(),
                        'portfolio_weight': weight
                    },
                    calculation_time=datetime.now(),
                    additional_info={
                        'method_description': 'Multi-asset Kelly with correlation',
                        'asset_name': asset,
                        'total_assets': len(asset_names),
                        'raw_kelly_weight': kelly_weights[i]
                    }
                )
                
                results[asset] = result
                self.kelly_results[f'multi_asset_{asset}'] = result
                self.calculation_history.append(result)
            
            logger.info(f"Multi-asset Kelly calculated for {len(results)} assets")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating multi-asset Kelly: {e}")
            raise
    
    def get_optimal_kelly(self, methods: List[KellyMethod] = None) -> KellyResult:
        """Get optimal Kelly using ensemble of methods"""
        try:
            methods = methods or [KellyMethod.CLASSIC, KellyMethod.FRACTIONAL]
            
            results = []
            weights = []
            
            for method in methods:
                if method == KellyMethod.CLASSIC:
                    result = self.calculate_classic_kelly()
                    weight = 0.4
                elif method == KellyMethod.FRACTIONAL:
                    result = self.calculate_fractional_kelly()
                    weight = 0.6  # Higher weight for safer fractional
                else:
                    continue
                
                results.append(result)
                weights.append(weight * result.confidence_level)
            
            if not results:
                raise ValueError("No valid methods provided")
            
            # Weighted average
            total_weight = sum(weights)
            optimal_size = sum(r.recommended_size * w for r, w in zip(results, weights)) / total_weight
            optimal_kelly = sum(r.kelly_fraction * w for r, w in zip(results, weights)) / total_weight
            avg_confidence = sum(r.confidence_level * w for r, w in zip(results, weights)) / total_weight
            
            # Use statistics from best confidence result
            best_result = max(results, key=lambda x: x.confidence_level)
            
            result = KellyResult(
                method=KellyMethod.CLASSIC,  # Ensemble method
                kelly_fraction=optimal_kelly,
                safe_kelly=optimal_size,
                recommended_size=optimal_size,
                win_rate=best_result.win_rate,
                avg_win=best_result.avg_win,
                avg_loss=best_result.avg_loss,
                profit_ratio=best_result.profit_ratio,
                safety_factor=self.default_safety_factor,
                confidence_level=avg_confidence,
                risk_metrics={
                    'methods_used': [m.value for m in methods],
                    'individual_sizes': [r.recommended_size for r in results],
                    'weights': weights,
                    'ensemble_type': 'weighted_average'
                },
                calculation_time=datetime.now(),
                additional_info={
                    'method_description': 'Optimal Kelly ensemble',
                    'methods_count': len(results),
                    'best_confidence_method': best_result.method.value
                }
            )
            
            self.kelly_results['optimal'] = result
            self.calculation_history.append(result)
            
            logger.info(f"Optimal Kelly calculated: {optimal_size:.4f} (Ensemble of {len(methods)} methods)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating optimal Kelly: {e}")
            raise
    
    def get_kelly_summary(self) -> Dict:
        """Get summary of all Kelly calculations"""
        try:
            if not self.kelly_results:
                return {'message': 'No Kelly calculations performed yet'}
            
            summary = {
                'trade_statistics': {
                    'total_trades': len(self.trade_results),
                    'win_rate': self.win_rate,
                    'avg_win': self.avg_win,
                    'avg_loss': self.avg_loss,
                    'profit_ratio': self.profit_ratio
                },
                'kelly_results': {},
                'recommendations': {}
            }
            
            for method_name, result in self.kelly_results.items():
                summary['kelly_results'][method_name] = {
                    'method': result.method.value,
                    'kelly_fraction': result.kelly_fraction,
                    'safe_kelly': result.safe_kelly,
                    'recommended_size': result.recommended_size,
                    'confidence_level': result.confidence_level,
                    'safety_factor': result.safety_factor
                }
            
            # Add recommendations
            if self.kelly_results:
                sizes = [r.recommended_size for r in self.kelly_results.values()]
                confidences = [r.confidence_level for r in self.kelly_results.values()]
                
                summary['recommendations'] = {
                    'conservative_size': min(sizes),
                    'aggressive_size': max(sizes),
                    'average_size': np.mean(sizes),
                    'high_confidence_size': [r.recommended_size for r in self.kelly_results.values() 
                                           if r.confidence_level == max(confidences)][0],
                    'recommended_range': f"{min(sizes):.3f} - {max(sizes):.3f}"
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting Kelly summary: {e}")
            return {'error': str(e)}
    
    def export_kelly_data(self, filepath: str) -> bool:
        """Export Kelly calculation data to JSON file"""
        try:
            import json
            
            export_data = {
                'system_info': self.get_system_info(),
                'configuration': {
                    'default_safety_factor': self.default_safety_factor,
                    'max_kelly': self.max_kelly,
                    'min_kelly': self.min_kelly,
                    'confidence_threshold': self.confidence_threshold
                },
                'trade_statistics': {
                    'total_trades': len(self.trade_results),
                    'win_rate': self.win_rate,
                    'avg_win': self.avg_win,
                    'avg_loss': self.avg_loss,
                    'profit_ratio': self.profit_ratio
                },
                'kelly_results': {},
                'calculation_history': []
            }
            
            # Add current results
            for method_name, result in self.kelly_results.items():
                export_data['kelly_results'][method_name] = {
                    'method': result.method.value,
                    'kelly_fraction': result.kelly_fraction,
                    'safe_kelly': result.safe_kelly,
                    'recommended_size': result.recommended_size,
                    'confidence_level': result.confidence_level,
                    'risk_metrics': result.risk_metrics,
                    'calculation_time': result.calculation_time.isoformat(),
                    'additional_info': result.additional_info
                }
            
            # Add history
            for result in self.calculation_history[-50:]:  # Last 50 calculations
                export_data['calculation_history'].append({
                    'method': result.method.value,
                    'kelly_fraction': result.kelly_fraction,
                    'recommended_size': result.recommended_size,
                    'confidence_level': result.confidence_level,
                    'calculation_time': result.calculation_time.isoformat()
                })
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Kelly calculation data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Kelly data: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get Kelly calculator statistics - required by BaseSystem"""
        try:
            return {
                'total_trades': len(self.trade_results),
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'profit_ratio': self.profit_ratio,
                'calculations_performed': len(self.calculation_history),
                'methods_available': len(KellyMethod),
                'current_results': len(self.kelly_results),
                'default_safety_factor': self.default_safety_factor,
                'max_kelly': self.max_kelly,
                'min_kelly': self.min_kelly,
                'system_name': self.system_name,
                'is_active': self.is_active,
                'operation_count': self.operation_count,
                'error_count': self.error_count,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'error': str(e),
                'system_name': self.system_name,
                'is_active': self.is_active
            } 