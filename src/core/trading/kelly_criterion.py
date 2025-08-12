#!/usr/bin/env python3
"""
Kelly Criterion Calculator for Optimal Position Sizing
Part of Ultimate XAU Super System V4.0

The Kelly Criterion determines the optimal position size to maximize long-term growth
while minimizing the risk of ruin.

Formula: f* = (bp - q) / b
Where:
- f* = fraction of capital to wager
- b = odds received on the wager (profit/loss ratio)
- p = probability of winning
- q = probability of losing (1-p)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)

class KellyMethod(Enum):
    """Kelly Criterion calculation methods"""
    CLASSIC = "classic"
    FRACTIONAL = "fractional"
    DYNAMIC = "dynamic"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"

@dataclass
class TradeResult:
    """Individual trade result for Kelly calculation"""
    profit_loss: float
    win: bool
    trade_date: datetime
    symbol: str
    entry_price: float
    exit_price: float
    volume: float
    duration_minutes: int
    
    def __post_init__(self):
        if self.profit_loss > 0 and not self.win:
            self.win = True
        elif self.profit_loss <= 0 and self.win:
            self.win = False

@dataclass
class KellyParameters:
    """Parameters for Kelly Criterion calculation"""
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    total_trades: int
    lookback_period: int = 100
    confidence_level: float = 0.95
    max_kelly_fraction: float = 0.25
    min_kelly_fraction: float = 0.01
    
    def __post_init__(self):
        # Validate parameters
        if not 0 <= self.win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")
        if self.average_loss >= 0:
            raise ValueError("Average loss must be negative")
        if self.average_win <= 0:
            raise ValueError("Average win must be positive")

@dataclass
class KellyResult:
    """Result of Kelly Criterion calculation"""
    kelly_fraction: float
    recommended_position_size: float
    confidence_score: float
    method_used: KellyMethod
    parameters: KellyParameters
    risk_metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'kelly_fraction': self.kelly_fraction,
            'recommended_position_size': self.recommended_position_size,
            'confidence_score': self.confidence_score,
            'method_used': self.method_used.value,
            'risk_metrics': self.risk_metrics,
            'warnings': self.warnings,
            'calculation_timestamp': self.calculation_timestamp.isoformat(),
            'parameters': {
                'win_rate': self.parameters.win_rate,
                'average_win': self.parameters.average_win,
                'average_loss': self.parameters.average_loss,
                'profit_factor': self.parameters.profit_factor,
                'total_trades': self.parameters.total_trades
            }
        }

class KellyCriterionCalculator:
    """
    Advanced Kelly Criterion Calculator with multiple methods and risk controls
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Kelly Criterion Calculator"""
        self.config = config or {}
        
        # Configuration parameters
        self.max_lookback_trades = self.config.get('max_lookback_trades', 500)
        self.min_trades_required = self.config.get('min_trades_required', 30)
        self.default_max_kelly = self.config.get('default_max_kelly', 0.25)
        self.default_min_kelly = self.config.get('default_min_kelly', 0.01)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Trade history storage
        self.trade_history: deque = deque(maxlen=self.max_lookback_trades)
        self.performance_metrics = {}
        self.calculation_history = []
        
        # Risk controls
        self.risk_controls = {
            'max_consecutive_losses': 5,
            'max_drawdown_threshold': 0.15,
            'min_profit_factor': 1.1,
            'volatility_adjustment': True,
            'trend_adjustment': True
        }
        
        logger.info("Kelly Criterion Calculator initialized")
    
    def add_trade_result(self, trade_result: TradeResult):
        """Add a trade result to the history"""
        try:
            self.trade_history.append(trade_result)
            self._update_performance_metrics()
            logger.debug(f"Added trade result: P&L={trade_result.profit_loss}, Win={trade_result.win}")
        except Exception as e:
            logger.error(f"Error adding trade result: {e}")
    
    def calculate_kelly_fraction(self, 
                                method: KellyMethod = KellyMethod.CLASSIC,
                                custom_parameters: Optional[KellyParameters] = None) -> KellyResult:
        """
        Calculate Kelly fraction using specified method
        
        Args:
            method: Kelly calculation method
            custom_parameters: Override calculated parameters
            
        Returns:
            KellyResult with recommended position size
        """
        try:
            # Use custom parameters or calculate from trade history
            if custom_parameters:
                parameters = custom_parameters
            else:
                parameters = self._calculate_parameters_from_history()
            
            # Calculate Kelly fraction based on method
            if method == KellyMethod.CLASSIC:
                kelly_fraction = self._calculate_classic_kelly(parameters)
            elif method == KellyMethod.FRACTIONAL:
                kelly_fraction = self._calculate_fractional_kelly(parameters)
            elif method == KellyMethod.DYNAMIC:
                kelly_fraction = self._calculate_dynamic_kelly(parameters)
            elif method == KellyMethod.CONSERVATIVE:
                kelly_fraction = self._calculate_conservative_kelly(parameters)
            elif method == KellyMethod.ADAPTIVE:
                kelly_fraction = self._calculate_adaptive_kelly(parameters)
            else:
                raise ValueError(f"Unknown Kelly method: {method}")
            
            # Apply risk controls
            kelly_fraction = self._apply_risk_controls(kelly_fraction, parameters)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(parameters)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(kelly_fraction, parameters)
            
            # Generate warnings
            warnings = self._generate_warnings(kelly_fraction, parameters, confidence_score)
            
            # Create result
            result = KellyResult(
                kelly_fraction=kelly_fraction,
                recommended_position_size=kelly_fraction,  # Can be adjusted based on account size
                confidence_score=confidence_score,
                method_used=method,
                parameters=parameters,
                risk_metrics=risk_metrics,
                warnings=warnings
            )
            
            # Store calculation history
            self.calculation_history.append(result)
            
            logger.info(f"Kelly fraction calculated: {kelly_fraction:.4f} using {method.value} method")
            return result
            
        except ValueError as e:
            # Re-raise ValueError for insufficient data
            raise e
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            # Return conservative default
            return self._create_default_result(method)
    
    def _calculate_classic_kelly(self, parameters: KellyParameters) -> float:
        """Calculate classic Kelly fraction: f* = (bp - q) / b"""
        try:
            p = parameters.win_rate
            q = 1 - p
            b = abs(parameters.average_win / parameters.average_loss)  # Odds ratio
            
            kelly_fraction = (b * p - q) / b
            
            return max(0, kelly_fraction)  # Kelly can't be negative
            
        except Exception as e:
            logger.error(f"Error in classic Kelly calculation: {e}")
            return 0.0
    
    def _calculate_fractional_kelly(self, parameters: KellyParameters) -> float:
        """Calculate fractional Kelly (typically 25-50% of full Kelly)"""
        try:
            full_kelly = self._calculate_classic_kelly(parameters)
            
            # Use fraction based on confidence
            if parameters.total_trades >= 100:
                fraction = 0.5  # 50% of full Kelly for high confidence
            elif parameters.total_trades >= 50:
                fraction = 0.35  # 35% for medium confidence
            else:
                fraction = 0.25  # 25% for low confidence
            
            return full_kelly * fraction
            
        except Exception as e:
            logger.error(f"Error in fractional Kelly calculation: {e}")
            return 0.0
    
    def _calculate_dynamic_kelly(self, parameters: KellyParameters) -> float:
        """Calculate dynamic Kelly that adjusts based on recent performance"""
        try:
            base_kelly = self._calculate_classic_kelly(parameters)
            
            # Adjust based on recent performance
            recent_trades = list(self.trade_history)[-20:]  # Last 20 trades
            if len(recent_trades) >= 10:
                recent_win_rate = sum(1 for t in recent_trades if t.win) / len(recent_trades)
                recent_profit_factor = self._calculate_profit_factor(recent_trades)
                
                # Adjustment factor based on recent vs overall performance
                win_rate_ratio = recent_win_rate / max(parameters.win_rate, 0.01)
                pf_ratio = recent_profit_factor / max(parameters.profit_factor, 0.01)
                
                adjustment = (win_rate_ratio + pf_ratio) / 2
                adjustment = np.clip(adjustment, 0.5, 1.5)  # Limit adjustment
                
                return base_kelly * adjustment
            
            return base_kelly
            
        except Exception as e:
            logger.error(f"Error in dynamic Kelly calculation: {e}")
            return 0.0
    
    def _calculate_conservative_kelly(self, parameters: KellyParameters) -> float:
        """Calculate conservative Kelly with additional safety margins"""
        try:
            base_kelly = self._calculate_classic_kelly(parameters)
            
            # Apply conservative adjustments
            safety_factors = []
            
            # Factor 1: Trade count confidence
            if parameters.total_trades < 50:
                safety_factors.append(0.5)
            elif parameters.total_trades < 100:
                safety_factors.append(0.7)
            else:
                safety_factors.append(0.9)
            
            # Factor 2: Profit factor confidence
            if parameters.profit_factor < 1.2:
                safety_factors.append(0.6)
            elif parameters.profit_factor < 1.5:
                safety_factors.append(0.8)
            else:
                safety_factors.append(1.0)
            
            # Factor 3: Win rate confidence
            if parameters.win_rate < 0.4:
                safety_factors.append(0.5)
            elif parameters.win_rate < 0.6:
                safety_factors.append(0.8)
            else:
                safety_factors.append(1.0)
            
            # Apply most conservative factor
            safety_factor = min(safety_factors)
            
            return base_kelly * safety_factor
            
        except Exception as e:
            logger.error(f"Error in conservative Kelly calculation: {e}")
            return 0.0
    
    def _calculate_adaptive_kelly(self, parameters: KellyParameters) -> float:
        """Calculate adaptive Kelly that adjusts to market conditions"""
        try:
            base_kelly = self._calculate_classic_kelly(parameters)
            
            # Market condition adjustments
            adjustments = []
            
            # Volatility adjustment
            if self.risk_controls['volatility_adjustment']:
                volatility_adj = self._calculate_volatility_adjustment()
                adjustments.append(volatility_adj)
            
            # Trend adjustment
            if self.risk_controls['trend_adjustment']:
                trend_adj = self._calculate_trend_adjustment()
                adjustments.append(trend_adj)
            
            # Drawdown adjustment
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown > 0.05:  # 5% drawdown
                dd_adjustment = 1 - (current_drawdown * 2)  # Reduce Kelly based on drawdown
                adjustments.append(max(0.3, dd_adjustment))
            
            # Apply adjustments
            if adjustments:
                final_adjustment = np.mean(adjustments)
                return base_kelly * final_adjustment
            
            return base_kelly
            
        except Exception as e:
            logger.error(f"Error in adaptive Kelly calculation: {e}")
            return 0.0
    
    def _calculate_parameters_from_history(self) -> KellyParameters:
        """Calculate Kelly parameters from trade history"""
        try:
            if len(self.trade_history) < self.min_trades_required:
                raise ValueError(f"Insufficient trade history. Need at least {self.min_trades_required} trades")
            
            trades = list(self.trade_history)
            
            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.win]
            losing_trades = [t for t in trades if not t.win]
            
            win_rate = len(winning_trades) / total_trades
            
            average_win = np.mean([t.profit_loss for t in winning_trades]) if winning_trades else 0
            average_loss = np.mean([t.profit_loss for t in losing_trades]) if losing_trades else -1
            
            # Calculate profit factor
            total_profit = sum(t.profit_loss for t in winning_trades)
            total_loss = abs(sum(t.profit_loss for t in losing_trades))
            profit_factor = total_profit / max(total_loss, 0.01)
            
            return KellyParameters(
                win_rate=win_rate,
                average_win=average_win,
                average_loss=average_loss,
                profit_factor=profit_factor,
                total_trades=total_trades,
                lookback_period=min(total_trades, self.max_lookback_trades)
            )
            
        except Exception as e:
            logger.error(f"Error calculating parameters from history: {e}")
            raise
    
    def _apply_risk_controls(self, kelly_fraction: float, parameters: KellyParameters) -> float:
        """Apply risk control limits to Kelly fraction"""
        try:
            # Apply maximum Kelly limit
            kelly_fraction = min(kelly_fraction, parameters.max_kelly_fraction)
            
            # Apply minimum Kelly limit
            kelly_fraction = max(kelly_fraction, parameters.min_kelly_fraction)
            
            # Check for consecutive losses
            recent_trades = list(self.trade_history)[-self.risk_controls['max_consecutive_losses']:]
            if len(recent_trades) >= self.risk_controls['max_consecutive_losses']:
                if all(not t.win for t in recent_trades):
                    kelly_fraction *= 0.5  # Reduce by 50% after consecutive losses
            
            # Check profit factor minimum
            if parameters.profit_factor < self.risk_controls['min_profit_factor']:
                kelly_fraction *= 0.7  # Reduce for low profit factor
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error applying risk controls: {e}")
            return min(kelly_fraction, 0.01)  # Conservative fallback
    
    def _calculate_confidence_score(self, parameters: KellyParameters) -> float:
        """Calculate confidence score for the Kelly calculation"""
        try:
            confidence_factors = []
            
            # Trade count confidence
            if parameters.total_trades >= 200:
                confidence_factors.append(1.0)
            elif parameters.total_trades >= 100:
                confidence_factors.append(0.9)
            elif parameters.total_trades >= 50:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Statistical significance
            if parameters.win_rate > 0.6 and parameters.profit_factor > 1.5:
                confidence_factors.append(1.0)
            elif parameters.win_rate > 0.5 and parameters.profit_factor > 1.2:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            # Consistency check
            if len(self.trade_history) >= 50:
                recent_performance = self._calculate_recent_performance_consistency()
                confidence_factors.append(recent_performance)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _calculate_risk_metrics(self, kelly_fraction: float, parameters: KellyParameters) -> Dict[str, float]:
        """Calculate risk metrics for the Kelly result"""
        try:
            risk_metrics = {}
            
            # Risk of ruin approximation
            if parameters.win_rate > 0 and parameters.profit_factor > 1:
                risk_of_ruin = self._calculate_risk_of_ruin(kelly_fraction, parameters)
                risk_metrics['risk_of_ruin'] = risk_of_ruin
            
            # Expected growth rate
            expected_return = (parameters.win_rate * parameters.average_win + 
                             (1 - parameters.win_rate) * parameters.average_loss)
            risk_metrics['expected_return_per_trade'] = expected_return
            
            # Volatility of returns
            if len(self.trade_history) > 1:
                returns = [t.profit_loss for t in self.trade_history]
                risk_metrics['return_volatility'] = np.std(returns)
            
            # Maximum theoretical loss
            risk_metrics['max_theoretical_loss'] = kelly_fraction * abs(parameters.average_loss)
            
            # Sharpe ratio approximation
            if 'return_volatility' in risk_metrics and risk_metrics['return_volatility'] > 0:
                risk_metrics['sharpe_ratio'] = expected_return / risk_metrics['return_volatility']
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _generate_warnings(self, kelly_fraction: float, parameters: KellyParameters, 
                          confidence_score: float) -> List[str]:
        """Generate warnings based on Kelly calculation"""
        warnings = []
        
        try:
            # Low confidence warning
            if confidence_score < self.confidence_threshold:
                warnings.append(f"Low confidence score ({confidence_score:.2f}). Consider more conservative sizing.")
            
            # Insufficient data warning
            if parameters.total_trades < 50:
                warnings.append(f"Limited trade history ({parameters.total_trades} trades). Results may be unreliable.")
            
            # High Kelly warning
            if kelly_fraction > 0.2:
                warnings.append("High Kelly fraction detected. Consider fractional Kelly for safety.")
            
            # Low profit factor warning
            if parameters.profit_factor < 1.2:
                warnings.append(f"Low profit factor ({parameters.profit_factor:.2f}). Strategy may not be profitable long-term.")
            
            # Low win rate warning
            if parameters.win_rate < 0.4:
                warnings.append(f"Low win rate ({parameters.win_rate:.1%}). Ensure average wins significantly exceed average losses.")
            
            # Recent performance warning
            if len(self.trade_history) >= 20:
                recent_trades = list(self.trade_history)[-20:]
                recent_losses = sum(1 for t in recent_trades if not t.win)
                if recent_losses >= 15:  # 75% recent losses
                    warnings.append("Recent performance is poor. Consider reducing position size or reviewing strategy.")
            
            # Check for consecutive losses at the end (regardless of total trades)
            if len(self.trade_history) >= 5:
                recent_trades = list(self.trade_history)[-5:]
                if all(not t.win for t in recent_trades):
                    warnings.append("Recent performance shows consecutive losses. Consider reducing position size.")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error generating warnings: {e}")
            return ["Error generating warnings"]
    
    def _calculate_volatility_adjustment(self) -> float:
        """Calculate volatility-based adjustment factor"""
        try:
            if len(self.trade_history) < 20:
                return 1.0
            
            recent_returns = [t.profit_loss for t in list(self.trade_history)[-20:]]
            volatility = np.std(recent_returns)
            
            # Normalize volatility (assuming typical range 0-0.1)
            normalized_vol = min(volatility / 0.05, 2.0)
            
            # Higher volatility = lower Kelly
            adjustment = 1 / (1 + normalized_vol * 0.5)
            return max(0.5, adjustment)
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
    def _calculate_trend_adjustment(self) -> float:
        """Calculate trend-based adjustment factor"""
        try:
            if len(self.trade_history) < 10:
                return 1.0
            
            recent_trades = list(self.trade_history)[-10:]
            cumulative_pnl = np.cumsum([t.profit_loss for t in recent_trades])
            
            # Check if trend is positive
            if cumulative_pnl[-1] > cumulative_pnl[0]:
                return 1.1  # Slight increase for positive trend
            else:
                return 0.9  # Slight decrease for negative trend
                
        except Exception as e:
            logger.error(f"Error calculating trend adjustment: {e}")
            return 1.0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak equity"""
        try:
            if len(self.trade_history) < 5:
                return 0.0
            
            # Calculate cumulative P&L
            cumulative_pnl = np.cumsum([t.profit_loss for t in self.trade_history])
            
            # Find peak and current drawdown
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = (peak - cumulative_pnl) / np.maximum(peak, 1)
            
            return float(drawdown[-1])
            
        except Exception as e:
            logger.error(f"Error calculating current drawdown: {e}")
            return 0.0
    
    def _calculate_profit_factor(self, trades: List[TradeResult]) -> float:
        """Calculate profit factor for given trades"""
        try:
            winning_trades = [t for t in trades if t.win]
            losing_trades = [t for t in trades if not t.win]
            
            total_profit = sum(t.profit_loss for t in winning_trades)
            total_loss = abs(sum(t.profit_loss for t in losing_trades))
            
            return total_profit / max(total_loss, 0.01)
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 1.0
    
    def _calculate_recent_performance_consistency(self) -> float:
        """Calculate consistency of recent performance"""
        try:
            # Compare recent 25 trades with previous 25 trades
            if len(self.trade_history) < 50:
                return 0.7
            
            recent_trades = list(self.trade_history)[-25:]
            previous_trades = list(self.trade_history)[-50:-25]
            
            recent_win_rate = sum(1 for t in recent_trades if t.win) / len(recent_trades)
            previous_win_rate = sum(1 for t in previous_trades if t.win) / len(previous_trades)
            
            recent_pf = self._calculate_profit_factor(recent_trades)
            previous_pf = self._calculate_profit_factor(previous_trades)
            
            # Calculate consistency score
            win_rate_consistency = 1 - abs(recent_win_rate - previous_win_rate)
            pf_consistency = 1 - abs(recent_pf - previous_pf) / max(previous_pf, 0.1)
            
            return (win_rate_consistency + pf_consistency) / 2
            
        except Exception as e:
            logger.error(f"Error calculating performance consistency: {e}")
            return 0.7
    
    def _calculate_risk_of_ruin(self, kelly_fraction: float, parameters: KellyParameters) -> float:
        """Calculate approximate risk of ruin"""
        try:
            # Simplified risk of ruin calculation
            # This is an approximation - exact calculation requires more complex math
            
            p = parameters.win_rate
            q = 1 - p
            a = abs(parameters.average_loss)
            b = parameters.average_win
            
            if p <= 0.5:
                return 1.0  # High risk if win rate <= 50%
            
            # Approximate formula
            risk_ratio = (q * a) / (p * b)
            if risk_ratio >= 1:
                return 1.0
            
            # Risk of ruin approximation
            ror = (risk_ratio ** (1 / kelly_fraction)) if kelly_fraction > 0 else 1.0
            return min(ror, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating risk of ruin: {e}")
            return 0.5
    
    def _create_default_result(self, method: KellyMethod) -> KellyResult:
        """Create default conservative result when calculation fails"""
        default_params = KellyParameters(
            win_rate=0.5,
            average_win=1.0,
            average_loss=-1.0,
            profit_factor=1.0,
            total_trades=0
        )
        
        return KellyResult(
            kelly_fraction=0.01,  # Very conservative
            recommended_position_size=0.01,
            confidence_score=0.0,
            method_used=method,
            parameters=default_params,
            risk_metrics={},
            warnings=["Calculation failed - using conservative default"]
        )
    
    def _update_performance_metrics(self):
        """Update internal performance metrics"""
        try:
            if len(self.trade_history) == 0:
                return
            
            trades = list(self.trade_history)
            
            self.performance_metrics = {
                'total_trades': len(trades),
                'winning_trades': sum(1 for t in trades if t.win),
                'losing_trades': sum(1 for t in trades if not t.win),
                'win_rate': sum(1 for t in trades if t.win) / len(trades),
                'total_pnl': sum(t.profit_loss for t in trades),
                'average_win': np.mean([t.profit_loss for t in trades if t.win]) if any(t.win for t in trades) else 0,
                'average_loss': np.mean([t.profit_loss for t in trades if not t.win]) if any(not t.win for t in trades) else 0,
                'profit_factor': self._calculate_profit_factor(trades),
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {
                'performance_metrics': self.performance_metrics.copy(),
                'trade_history_length': len(self.trade_history),
                'calculation_history_length': len(self.calculation_history),
                'risk_controls': self.risk_controls.copy()
            }
            
            if self.calculation_history:
                latest_calculation = self.calculation_history[-1]
                summary['latest_kelly_result'] = latest_calculation.to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def export_data(self, filepath: str) -> bool:
        """Export Kelly calculator data to file"""
        try:
            export_data = {
                'performance_metrics': {
                    k: v.isoformat() if isinstance(v, datetime) else v 
                    for k, v in self.performance_metrics.items()
                },
                'calculation_history': [calc.to_dict() for calc in self.calculation_history],
                'trade_history': [
                    {
                        'profit_loss': t.profit_loss,
                        'win': t.win,
                        'trade_date': t.trade_date.isoformat(),
                        'symbol': t.symbol,
                        'entry_price': t.entry_price,
                        'exit_price': t.exit_price,
                        'volume': t.volume,
                        'duration_minutes': t.duration_minutes
                    }
                    for t in self.trade_history
                ],
                'config': self.config,
                'risk_controls': self.risk_controls,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Kelly calculator data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Kelly calculator statistics"""
        return {
            'total_trades_processed': len(self.trade_history),
            'total_calculations_performed': len(self.calculation_history),
            'performance_metrics': self.performance_metrics,
            'risk_controls': self.risk_controls,
            'config': self.config,
            'last_calculation': self.calculation_history[-1].to_dict() if self.calculation_history else None
        }

# Create alias for backward compatibility
KellyCalculator = KellyCriterionCalculator

# Export main classes and functions
__all__ = [
    'KellyMethod',
    'TradeResult', 
    'KellyParameters',
    'KellyResult',
    'KellyCriterionCalculator',
    'KellyCalculator'  # Alias
] 
# Create alias for backward compatibility
KellyCalculator = KellyCriterionCalculator
