"""
Position Sizing System
Advanced position sizing algorithms including Kelly Criterion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
import json

from ..base_system import BaseSystem

# Import Kelly Criterion Calculator
try:
    from ..trading.kelly_criterion import (
        KellyCriterionCalculator, 
        TradeResult, 
        KellyMethod,
        KellyParameters as KellyCalcParameters
    )
    KELLY_CALCULATOR_AVAILABLE = True
except ImportError:
    KELLY_CALCULATOR_AVAILABLE = False
    print("⚠️ Kelly Criterion Calculator not available")

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    RISK_BASED = "risk_based"
    KELLY_CRITERION = "kelly_criterion"
    KELLY_CLASSIC = "kelly_classic"
    KELLY_FRACTIONAL = "kelly_fractional"
    KELLY_DYNAMIC = "kelly_dynamic"
    KELLY_CONSERVATIVE = "kelly_conservative"
    KELLY_ADAPTIVE = "kelly_adaptive"
    VOLATILITY_BASED = "volatility_based"
    ATR_BASED = "atr_based"
    OPTIMAL_F = "optimal_f"
    RISK_PARITY = "risk_parity"


class RiskLevel(Enum):
    """Risk level categories"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


@dataclass
class SizingParameters:
    """Position sizing parameters"""
    method: SizingMethod = SizingMethod.OPTIMAL_F
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.1  # 10% max position
    min_position_size: float = 0.001  # 0.1% min position
    kelly_fraction: float = 0.25  # Kelly fraction (conservative)
    volatility_lookback: int = 20  # Days for volatility calculation
    atr_period: int = 14  # ATR period
    confidence_level: float = 0.95  # Confidence level for calculations
    
    # Advanced Kelly parameters
    kelly_max_fraction: float = 0.25  # Maximum Kelly fraction
    kelly_min_fraction: float = 0.01  # Minimum Kelly fraction
    kelly_confidence_threshold: float = 0.7  # Confidence threshold
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'method': self.method.value,
            'risk_per_trade': self.risk_per_trade,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'kelly_fraction': self.kelly_fraction,
            'volatility_lookback': self.volatility_lookback,
            'atr_period': self.atr_period,
            'confidence_level': self.confidence_level,
            'kelly_max_fraction': self.kelly_max_fraction,
            'kelly_min_fraction': self.kelly_min_fraction,
            'kelly_confidence_threshold': self.kelly_confidence_threshold
        }


@dataclass
class SizingResult:
    """Position sizing result"""
    method: SizingMethod
    position_size: float
    risk_amount: float
    confidence_score: float
    calculation_date: datetime
    parameters_used: SizingParameters
    additional_metrics: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'method': self.method.value,
            'position_size': self.position_size,
            'risk_amount': self.risk_amount,
            'confidence_score': self.confidence_score,
            'calculation_date': self.calculation_date.isoformat(),
            'parameters_used': self.parameters_used.to_dict(),
            'additional_metrics': self.additional_metrics or {}
        }


class PositionSizer(BaseSystem):
    """
    Advanced Position Sizing System
    Implements multiple position sizing algorithms with professional Kelly Criterion
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("PositionSizer", config)
        self.config = config or {}
        
        # Default parameters
        self.default_parameters = SizingParameters()
        
        # Data storage
        self.price_data: Optional[pd.DataFrame] = None
        self.returns_data: Optional[pd.DataFrame] = None
        self.portfolio_value: float = 100000.0
        
        # Results cache
        self.sizing_results: Dict[str, SizingResult] = {}
        self.statistics: Dict = {}
        
        # Performance tracking
        self.win_rate: float = 0.6  # Default win rate
        self.avg_win: float = 0.02  # Average win
        self.avg_loss: float = -0.01  # Average loss
        
        # Initialize Kelly Criterion Calculator
        self.kelly_calculator = None
        self.trade_history: List[TradeResult] = []
        
        if KELLY_CALCULATOR_AVAILABLE:
            self.kelly_calculator = KellyCriterionCalculator()
            logger.info("✅ Professional Kelly Criterion Calculator integrated")
        else:
            logger.warning("⚠️ Using basic Kelly implementation")
    
    def set_data(self, price_data: pd.DataFrame, portfolio_value: float = None):
        """Set price data and portfolio value"""
        try:
            self.price_data = price_data.copy()
            
            # Calculate returns
            self.returns_data = price_data.pct_change().dropna()
            
            if portfolio_value is not None:
                self.portfolio_value = portfolio_value
            
            logger.info(f"Data set: {len(price_data)} price observations, portfolio: ${self.portfolio_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error setting data: {e}")
            raise
    
    def set_performance_metrics(self, win_rate: float, avg_win: float, avg_loss: float):
        """Set performance metrics for Kelly calculation"""
        self.win_rate = max(0.01, min(0.99, win_rate))  # Clamp between 1% and 99%
        self.avg_win = max(0.001, avg_win)  # Minimum 0.1% win
        self.avg_loss = min(-0.001, avg_loss)  # Maximum -0.1% loss
        
        logger.info(f"Performance metrics set: WR={self.win_rate:.1%}, AvgWin={self.avg_win:.2%}, AvgLoss={self.avg_loss:.2%}")
    
    def add_trade_result(self, profit_loss: float, win: bool, trade_date: datetime = None, 
                        symbol: str = "XAUUSD", entry_price: float = 2000.0, 
                        exit_price: float = 2000.0, volume: float = 0.1, 
                        duration_minutes: int = 60):
        """Add trade result to Kelly Calculator"""
        if not KELLY_CALCULATOR_AVAILABLE or not self.kelly_calculator:
            return
        
        try:
            trade_result = TradeResult(
                profit_loss=profit_loss,
                win=win,
                trade_date=trade_date or datetime.now(),
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                volume=volume,
                duration_minutes=duration_minutes
            )
            
            self.kelly_calculator.add_trade_result(trade_result)
            self.trade_history.append(trade_result)
            
            logger.debug(f"Added trade result: P&L={profit_loss:.4f}, Win={win}")
            
        except Exception as e:
            logger.error(f"Error adding trade result: {e}")
    
    def calculate_fixed_amount_size(self, amount: float, current_price: float, 
                                  parameters: SizingParameters = None) -> SizingResult:
        """Calculate fixed amount position size"""
        try:
            parameters = parameters or self.default_parameters
            
            # Calculate position size
            position_size = amount / current_price
            
            # Apply limits
            max_size = self.portfolio_value * parameters.max_position_size / current_price
            min_size = self.portfolio_value * parameters.min_position_size / current_price
            
            position_size = max(min_size, min(position_size, max_size))
            
            # Calculate risk amount
            risk_amount = position_size * current_price * parameters.risk_per_trade
            
            result = SizingResult(
                method=SizingMethod.FIXED_AMOUNT,
                position_size=position_size,
                risk_amount=risk_amount,
                confidence_score=0.8,  # Fixed confidence for fixed amount
                calculation_date=datetime.now(),
                parameters_used=parameters,
                additional_metrics={
                    'target_amount': amount,
                    'current_price': current_price,
                    'position_value': position_size * current_price,
                    'portfolio_percentage': (position_size * current_price) / self.portfolio_value
                }
            )
            
            self.sizing_results[f"fixed_amount_{datetime.now().timestamp()}"] = result
            logger.info(f"Fixed amount sizing: {position_size:.4f} units (${position_size * current_price:,.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating fixed amount size: {e}")
            raise
    
    def calculate_fixed_percentage_size(self, percentage: float, current_price: float,
                                      parameters: SizingParameters = None) -> SizingResult:
        """Calculate fixed percentage position size"""
        try:
            parameters = parameters or self.default_parameters
            
            # Calculate position value
            position_value = self.portfolio_value * percentage
            position_size = position_value / current_price
            
            # Apply limits
            max_size = self.portfolio_value * parameters.max_position_size / current_price
            min_size = self.portfolio_value * parameters.min_position_size / current_price
            
            position_size = max(min_size, min(position_size, max_size))
            
            # Calculate risk amount
            risk_amount = position_size * current_price * parameters.risk_per_trade
            
            result = SizingResult(
                method=SizingMethod.FIXED_PERCENTAGE,
                position_size=position_size,
                risk_amount=risk_amount,
                confidence_score=0.8,
                calculation_date=datetime.now(),
                parameters_used=parameters,
                additional_metrics={
                    'target_percentage': percentage,
                    'current_price': current_price,
                    'position_value': position_size * current_price,
                    'actual_percentage': (position_size * current_price) / self.portfolio_value
                }
            )
            
            self.sizing_results[f"fixed_percentage_{datetime.now().timestamp()}"] = result
            logger.info(f"Fixed percentage sizing: {position_size:.4f} units ({percentage:.1%} of portfolio)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating fixed percentage size: {e}")
            raise
    
    def calculate_risk_based_size(self, current_price: float, stop_loss_price: float,
                                parameters: SizingParameters = None) -> SizingResult:
        """Calculate risk-based position size"""
        try:
            parameters = parameters or self.default_parameters
            
            # Calculate risk per unit
            risk_per_unit = abs(current_price - stop_loss_price)
            
            if risk_per_unit <= 0:
                raise ValueError("Stop loss price must be different from current price")
            
            # Calculate maximum risk amount
            max_risk_amount = self.portfolio_value * parameters.risk_per_trade
            
            # Calculate position size
            position_size = max_risk_amount / risk_per_unit
            
            # Apply limits
            max_size = self.portfolio_value * parameters.max_position_size / current_price
            min_size = self.portfolio_value * parameters.min_position_size / current_price
            
            position_size = max(min_size, min(position_size, max_size))
            
            # Calculate actual risk amount
            risk_amount = position_size * risk_per_unit
            
            result = SizingResult(
                method=SizingMethod.RISK_BASED,
                position_size=position_size,
                risk_amount=risk_amount,
                confidence_score=0.9,  # High confidence for risk-based
                calculation_date=datetime.now(),
                parameters_used=parameters,
                additional_metrics={
                    'current_price': current_price,
                    'stop_loss_price': stop_loss_price,
                    'risk_per_unit': risk_per_unit,
                    'risk_percentage': risk_amount / self.portfolio_value,
                    'position_value': position_size * current_price
                }
            )
            
            self.sizing_results[f"risk_based_{datetime.now().timestamp()}"] = result
            logger.info(f"Risk-based sizing: {position_size:.4f} units (Risk: ${risk_amount:,.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating risk-based size: {e}")
            raise
    
    def calculate_kelly_criterion_size(self, current_price: float,
                                     parameters: SizingParameters = None,
                                     kelly_method: KellyMethod = None) -> SizingResult:
        """Calculate Kelly Criterion position size using professional calculator"""
        try:
            parameters = parameters or self.default_parameters
            kelly_method = kelly_method or KellyMethod.ADAPTIVE
            
            # Use professional Kelly Calculator if available
            if KELLY_CALCULATOR_AVAILABLE and self.kelly_calculator:
                return self._calculate_professional_kelly(current_price, parameters, kelly_method)
            else:
                return self._calculate_basic_kelly(current_price, parameters)
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion size: {e}")
            raise
    
    def _calculate_professional_kelly(self, current_price: float, 
                                    parameters: SizingParameters,
                                    kelly_method: KellyMethod) -> SizingResult:
        """Calculate Kelly using professional Kelly Calculator"""
        try:
            # Calculate Kelly fraction using professional calculator
            kelly_result = self.kelly_calculator.calculate_kelly_fraction(kelly_method)
            
            # Extract Kelly fraction and apply limits
            kelly_f = kelly_result.kelly_fraction
            kelly_f = max(parameters.kelly_min_fraction, 
                         min(parameters.kelly_max_fraction, kelly_f))
            
            # Calculate position size
            position_value = self.portfolio_value * kelly_f
            position_size = position_value / current_price
            
            # Apply position limits
            max_size = self.portfolio_value * parameters.max_position_size / current_price
            min_size = self.portfolio_value * parameters.min_position_size / current_price
            position_size = max(min_size, min(position_size, max_size))
            
            # Calculate risk amount
            risk_amount = position_size * current_price * kelly_f
            
            # Use professional confidence score
            confidence_score = kelly_result.confidence_score
            
            # Map Kelly method to sizing method
            method_mapping = {
                KellyMethod.CLASSIC: SizingMethod.KELLY_CLASSIC,
                KellyMethod.FRACTIONAL: SizingMethod.KELLY_FRACTIONAL,
                KellyMethod.DYNAMIC: SizingMethod.KELLY_DYNAMIC,
                KellyMethod.CONSERVATIVE: SizingMethod.KELLY_CONSERVATIVE,
                KellyMethod.ADAPTIVE: SizingMethod.KELLY_ADAPTIVE
            }
            sizing_method = method_mapping.get(kelly_method, SizingMethod.KELLY_CRITERION)
            
            result = SizingResult(
                method=sizing_method,
                position_size=position_size,
                risk_amount=risk_amount,
                confidence_score=confidence_score,
                calculation_date=datetime.now(),
                parameters_used=parameters,
                additional_metrics={
                    'kelly_fraction': kelly_f,
                    'original_kelly_fraction': kelly_result.kelly_fraction,
                    'kelly_method': kelly_method.value,
                    'confidence_score': confidence_score,
                    'risk_metrics': kelly_result.risk_metrics,
                    'warnings': kelly_result.warnings,
                    'current_price': current_price,
                    'position_value': position_size * current_price,
                    'professional_calculator': True
                }
            )
            
            self.sizing_results[f"kelly_pro_{datetime.now().timestamp()}"] = result
            logger.info(f"Professional Kelly sizing ({kelly_method.value}): {position_size:.4f} units (Kelly f={kelly_f:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in professional Kelly calculation: {e}")
            # Fallback to basic Kelly
            return self._calculate_basic_kelly(current_price, parameters)
    
    def _calculate_basic_kelly(self, current_price: float, 
                             parameters: SizingParameters) -> SizingResult:
        """Calculate Kelly using basic implementation (fallback)"""
        try:
            # Kelly formula: f = (bp - q) / b
            b = abs(self.avg_win / self.avg_loss)  # Odds ratio
            p = self.win_rate  # Win probability
            q = 1 - p  # Loss probability
            
            # Calculate Kelly fraction
            kelly_f = (b * p - q) / b
            
            # Apply Kelly fraction (conservative approach)
            kelly_f = max(0, kelly_f * parameters.kelly_fraction)
            
            # Calculate position size
            position_value = self.portfolio_value * kelly_f
            position_size = position_value / current_price
            
            # Apply limits
            max_size = self.portfolio_value * parameters.max_position_size / current_price
            min_size = self.portfolio_value * parameters.min_position_size / current_price
            position_size = max(min_size, min(position_size, max_size))
            
            # Calculate risk amount
            risk_amount = position_size * current_price * abs(self.avg_loss)
            
            # Calculate confidence score based on Kelly edge
            edge = p * self.avg_win + q * self.avg_loss
            confidence_score = min(0.95, max(0.1, edge * 10))  # Scale edge to confidence
            
            result = SizingResult(
                method=SizingMethod.KELLY_CRITERION,
                position_size=position_size,
                risk_amount=risk_amount,
                confidence_score=confidence_score,
                calculation_date=datetime.now(),
                parameters_used=parameters,
                additional_metrics={
                    'kelly_fraction': kelly_f,
                    'full_kelly': (b * p - q) / b,
                    'win_rate': p,
                    'avg_win': self.avg_win,
                    'avg_loss': self.avg_loss,
                    'odds_ratio': b,
                    'edge': edge,
                    'current_price': current_price,
                    'position_value': position_size * current_price,
                    'professional_calculator': False
                }
            )
            
            self.sizing_results[f"kelly_basic_{datetime.now().timestamp()}"] = result
            logger.info(f"Basic Kelly sizing: {position_size:.4f} units (Kelly f={kelly_f:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in basic Kelly calculation: {e}")
            raise
    
    def calculate_kelly_classic_size(self, current_price: float,
                                   parameters: SizingParameters = None) -> SizingResult:
        """Calculate Classic Kelly Criterion position size"""
        return self.calculate_kelly_criterion_size(current_price, parameters, KellyMethod.CLASSIC)
    
    def calculate_kelly_fractional_size(self, current_price: float,
                                      parameters: SizingParameters = None) -> SizingResult:
        """Calculate Fractional Kelly Criterion position size"""
        return self.calculate_kelly_criterion_size(current_price, parameters, KellyMethod.FRACTIONAL)
    
    def calculate_kelly_dynamic_size(self, current_price: float,
                                   parameters: SizingParameters = None) -> SizingResult:
        """Calculate Dynamic Kelly Criterion position size"""
        return self.calculate_kelly_criterion_size(current_price, parameters, KellyMethod.DYNAMIC)
    
    def calculate_kelly_conservative_size(self, current_price: float,
                                        parameters: SizingParameters = None) -> SizingResult:
        """Calculate Conservative Kelly Criterion position size"""
        return self.calculate_kelly_criterion_size(current_price, parameters, KellyMethod.CONSERVATIVE)
    
    def calculate_kelly_adaptive_size(self, current_price: float,
                                    parameters: SizingParameters = None) -> SizingResult:
        """Calculate Adaptive Kelly Criterion position size"""
        return self.calculate_kelly_criterion_size(current_price, parameters, KellyMethod.ADAPTIVE)
    
    def get_kelly_analysis(self, current_price: float) -> Dict:
        """Get comprehensive Kelly analysis with all methods"""
        try:
            if not KELLY_CALCULATOR_AVAILABLE or not self.kelly_calculator:
                return {'error': 'Professional Kelly Calculator not available'}
            
            analysis = {}
            kelly_methods = [
                KellyMethod.CLASSIC,
                KellyMethod.FRACTIONAL,
                KellyMethod.DYNAMIC,
                KellyMethod.CONSERVATIVE,
                KellyMethod.ADAPTIVE
            ]
            
            for method in kelly_methods:
                try:
                    result = self.calculate_kelly_criterion_size(current_price, kelly_method=method)
                    analysis[method.value] = {
                        'position_size': result.position_size,
                        'kelly_fraction': result.additional_metrics.get('kelly_fraction', 0),
                        'confidence_score': result.confidence_score,
                        'risk_amount': result.risk_amount,
                        'position_value': result.additional_metrics.get('position_value', 0)
                    }
                except Exception as e:
                    analysis[method.value] = {'error': str(e)}
            
            # Get performance summary from Kelly Calculator
            performance_summary = self.kelly_calculator.get_performance_summary()
            
            return {
                'kelly_analysis': analysis,
                'performance_summary': performance_summary,
                'trade_count': len(self.trade_history),
                'current_price': current_price,
                'portfolio_value': self.portfolio_value
            }
            
        except Exception as e:
            logger.error(f"Error in Kelly analysis: {e}")
            return {'error': str(e)}
    
    def calculate_volatility_based_size(self, current_price: float,
                                      parameters: SizingParameters = None) -> SizingResult:
        """Calculate volatility-based position size"""
        try:
            parameters = parameters or self.default_parameters
            
            if self.returns_data is None:
                raise ValueError("No returns data available. Call set_data() first.")
            
            # Calculate recent volatility
            recent_returns = self.returns_data.iloc[-parameters.volatility_lookback:]
            volatility = recent_returns.std().iloc[0] if len(recent_returns.columns) > 0 else recent_returns.std()
            
            # Annualize volatility
            annual_volatility = volatility * np.sqrt(252)
            
            # Inverse volatility sizing (higher vol = smaller position)
            base_volatility = 0.2  # 20% base volatility
            volatility_factor = base_volatility / max(annual_volatility, 0.01)
            
            # Calculate position size
            position_value = self.portfolio_value * parameters.risk_per_trade * volatility_factor
            position_size = position_value / current_price
            
            # Apply limits
            max_size = self.portfolio_value * parameters.max_position_size / current_price
            min_size = self.portfolio_value * parameters.min_position_size / current_price
            
            position_size = max(min_size, min(position_size, max_size))
            
            # Calculate risk amount
            risk_amount = position_size * current_price * annual_volatility
            
            # Confidence score based on volatility stability
            vol_stability = 1 / (1 + recent_returns.std().iloc[0] / recent_returns.mean().iloc[0] if recent_returns.mean().iloc[0] != 0 else 1)
            confidence_score = min(0.9, max(0.3, vol_stability))
            
            result = SizingResult(
                method=SizingMethod.VOLATILITY_BASED,
                position_size=position_size,
                risk_amount=risk_amount,
                confidence_score=confidence_score,
                calculation_date=datetime.now(),
                parameters_used=parameters,
                additional_metrics={
                    'daily_volatility': volatility,
                    'annual_volatility': annual_volatility,
                    'volatility_factor': volatility_factor,
                    'lookback_period': parameters.volatility_lookback,
                    'current_price': current_price,
                    'position_value': position_size * current_price
                }
            )
            
            self.sizing_results[f"volatility_{datetime.now().timestamp()}"] = result
            logger.info(f"Volatility-based sizing: {position_size:.4f} units (Vol: {annual_volatility:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating volatility-based size: {e}")
            raise
    
    def calculate_atr_based_size(self, current_price: float,
                               parameters: SizingParameters = None) -> SizingResult:
        """Calculate ATR-based position size"""
        try:
            parameters = parameters or self.default_parameters
            
            if self.price_data is None:
                raise ValueError("No price data available. Call set_data() first.")
            
            # Calculate ATR
            atr = self._calculate_atr(parameters.atr_period)
            
            # ATR-based risk per unit
            risk_per_unit = atr * 2  # 2x ATR stop
            
            # Calculate maximum risk amount
            max_risk_amount = self.portfolio_value * parameters.risk_per_trade
            
            # Calculate position size
            position_size = max_risk_amount / risk_per_unit
            
            # Apply limits
            max_size = self.portfolio_value * parameters.max_position_size / current_price
            min_size = self.portfolio_value * parameters.min_position_size / current_price
            
            position_size = max(min_size, min(position_size, max_size))
            
            # Calculate actual risk amount
            risk_amount = position_size * risk_per_unit
            
            result = SizingResult(
                method=SizingMethod.ATR_BASED,
                position_size=position_size,
                risk_amount=risk_amount,
                confidence_score=0.85,
                calculation_date=datetime.now(),
                parameters_used=parameters,
                additional_metrics={
                    'atr': atr,
                    'atr_period': parameters.atr_period,
                    'risk_per_unit': risk_per_unit,
                    'atr_multiple': 2,
                    'current_price': current_price,
                    'position_value': position_size * current_price
                }
            )
            
            self.sizing_results[f"atr_{datetime.now().timestamp()}"] = result
            logger.info(f"ATR-based sizing: {position_size:.4f} units (ATR: {atr:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating ATR-based size: {e}")
            raise
    
    def _calculate_atr(self, period: int) -> float:
        """Calculate Average True Range"""
        try:
            if len(self.price_data) < period:
                raise ValueError(f"Insufficient data for ATR calculation (need {period}, have {len(self.price_data)})")
            
            # Assume OHLC data
            if 'High' in self.price_data.columns and 'Low' in self.price_data.columns and 'Close' in self.price_data.columns:
                high = self.price_data['High']
                low = self.price_data['Low']
                close = self.price_data['Close']
                prev_close = close.shift(1)
                
                # True Range calculation
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(window=period).mean().iloc[-1]
                
            else:
                # Fallback: use price range if OHLC not available
                price_col = self.price_data.columns[0]
                price_range = self.price_data[price_col].rolling(window=2).apply(lambda x: abs(x.iloc[1] - x.iloc[0]))
                atr = price_range.rolling(window=period).mean().iloc[-1]
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return self.price_data.iloc[-1, 0] * 0.01  # Fallback: 1% of current price
    
    def calculate_optimal_size(self, current_price: float, stop_loss_price: float = None,
                             parameters: SizingParameters = None) -> SizingResult:
        """Calculate optimal position size using multiple methods"""
        try:
            parameters = parameters or self.default_parameters
            
            # Calculate sizes using different methods
            sizes = []
            
            # Kelly Criterion
            kelly_result = self.calculate_kelly_criterion_size(current_price, parameters)
            sizes.append((kelly_result.position_size, kelly_result.confidence_score, 'kelly'))
            
            # Volatility-based
            vol_result = self.calculate_volatility_based_size(current_price, parameters)
            sizes.append((vol_result.position_size, vol_result.confidence_score, 'volatility'))
            
            # ATR-based
            atr_result = self.calculate_atr_based_size(current_price, parameters)
            sizes.append((atr_result.position_size, atr_result.confidence_score, 'atr'))
            
            # Risk-based (if stop loss provided)
            if stop_loss_price is not None:
                risk_result = self.calculate_risk_based_size(current_price, stop_loss_price, parameters)
                sizes.append((risk_result.position_size, risk_result.confidence_score, 'risk'))
            
            # Weighted average based on confidence scores
            total_weight = sum(confidence for _, confidence, _ in sizes)
            if total_weight > 0:
                weighted_size = sum(size * confidence for size, confidence, _ in sizes) / total_weight
                avg_confidence = total_weight / len(sizes)
            else:
                weighted_size = sum(size for size, _, _ in sizes) / len(sizes)
                avg_confidence = 0.5
            
            # Apply limits
            max_size = self.portfolio_value * parameters.max_position_size / current_price
            min_size = self.portfolio_value * parameters.min_position_size / current_price
            
            optimal_size = max(min_size, min(weighted_size, max_size))
            
            # Calculate risk amount
            if stop_loss_price is not None:
                risk_amount = optimal_size * abs(current_price - stop_loss_price)
            else:
                risk_amount = optimal_size * current_price * parameters.risk_per_trade
            
            result = SizingResult(
                method=SizingMethod.OPTIMAL_F,
                position_size=optimal_size,
                risk_amount=risk_amount,
                confidence_score=avg_confidence,
                calculation_date=datetime.now(),
                parameters_used=parameters,
                additional_metrics={
                    'component_sizes': {method: size for size, _, method in sizes},
                    'component_confidences': {method: conf for _, conf, method in sizes},
                    'weighted_average': weighted_size,
                    'current_price': current_price,
                    'stop_loss_price': stop_loss_price,
                    'position_value': optimal_size * current_price
                }
            )
            
            self.sizing_results[f"optimal_{datetime.now().timestamp()}"] = result
            logger.info(f"Optimal sizing: {optimal_size:.4f} units (Confidence: {avg_confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating optimal size: {e}")
            raise
    
    def get_sizing_recommendation(self, current_price: float, stop_loss_price: float = None,
                                risk_level: RiskLevel = RiskLevel.MODERATE) -> Dict:
        """Get comprehensive sizing recommendation"""
        try:
            # Adjust parameters based on risk level
            params = SizingParameters()
            
            if risk_level == RiskLevel.CONSERVATIVE:
                params.risk_per_trade = 0.01
                params.max_position_size = 0.05
                params.kelly_fraction = 0.1
            elif risk_level == RiskLevel.MODERATE:
                params.risk_per_trade = 0.02
                params.max_position_size = 0.1
                params.kelly_fraction = 0.25
            elif risk_level == RiskLevel.AGGRESSIVE:
                params.risk_per_trade = 0.03
                params.max_position_size = 0.15
                params.kelly_fraction = 0.5
            elif risk_level == RiskLevel.VERY_AGGRESSIVE:
                params.risk_per_trade = 0.05
                params.max_position_size = 0.2
                params.kelly_fraction = 0.75
            
            # Calculate optimal size
            optimal_result = self.calculate_optimal_size(current_price, stop_loss_price, params)
            
            # Calculate alternative sizes
            alternatives = {}
            
            # Conservative alternative
            conservative_params = SizingParameters(
                method=SizingMethod.RISK_BASED,
                risk_per_trade=params.risk_per_trade * 0.5,
                max_position_size=params.max_position_size * 0.5
            )
            
            if stop_loss_price is not None:
                alternatives['conservative'] = self.calculate_risk_based_size(
                    current_price, stop_loss_price, conservative_params
                )
            
            # Aggressive alternative
            aggressive_params = SizingParameters(
                method=SizingMethod.KELLY_CRITERION,
                risk_per_trade=params.risk_per_trade * 1.5,
                max_position_size=params.max_position_size * 1.2,
                kelly_fraction=params.kelly_fraction * 1.5
            )
            
            alternatives['aggressive'] = self.calculate_kelly_criterion_size(
                current_price, aggressive_params
            )
            
            recommendation = {
                'recommended': optimal_result.to_dict(),
                'alternatives': {k: v.to_dict() for k, v in alternatives.items()},
                'risk_level': risk_level.value,
                'market_conditions': self._assess_market_conditions(),
                'sizing_rationale': self._generate_sizing_rationale(optimal_result, risk_level)
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating sizing recommendation: {e}")
            raise
    
    def _assess_market_conditions(self) -> Dict:
        """Assess current market conditions"""
        try:
            if self.returns_data is None:
                return {'assessment': 'unknown', 'reason': 'no_data'}
            
            recent_returns = self.returns_data.iloc[-20:]  # Last 20 days
            
            # Calculate metrics
            volatility = recent_returns.std().iloc[0] if len(recent_returns.columns) > 0 else recent_returns.std()
            trend = recent_returns.mean().iloc[0] if len(recent_returns.columns) > 0 else recent_returns.mean()
            
            # Assess conditions
            if volatility > 0.03:  # High volatility
                condition = 'high_volatility'
                recommendation = 'reduce_position_sizes'
            elif volatility < 0.01:  # Low volatility
                condition = 'low_volatility'
                recommendation = 'normal_position_sizes'
            else:
                condition = 'normal_volatility'
                recommendation = 'normal_position_sizes'
            
            return {
                'volatility_condition': condition,
                'trend_direction': 'bullish' if trend > 0 else 'bearish',
                'volatility_value': volatility,
                'trend_value': trend,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return {'assessment': 'error', 'reason': str(e)}
    
    def _generate_sizing_rationale(self, result: SizingResult, risk_level: RiskLevel) -> str:
        """Generate rationale for sizing decision"""
        try:
            rationale_parts = []
            
            # Method rationale
            if result.method == SizingMethod.OPTIMAL_F:
                rationale_parts.append("Using optimal sizing based on multiple methods")
            elif result.method == SizingMethod.KELLY_CRITERION:
                rationale_parts.append("Using Kelly Criterion based on historical performance")
            elif result.method == SizingMethod.RISK_BASED:
                rationale_parts.append("Using risk-based sizing with defined stop loss")
            
            # Risk level rationale
            rationale_parts.append(f"Adjusted for {risk_level.value} risk tolerance")
            
            # Confidence rationale
            if result.confidence_score > 0.8:
                rationale_parts.append("High confidence in sizing calculation")
            elif result.confidence_score > 0.6:
                rationale_parts.append("Moderate confidence in sizing calculation")
            else:
                rationale_parts.append("Lower confidence - consider reducing position")
            
            # Position size rationale
            position_pct = (result.position_size * result.additional_metrics.get('current_price', 0)) / self.portfolio_value
            if position_pct > 0.1:
                rationale_parts.append("Large position - monitor closely")
            elif position_pct < 0.01:
                rationale_parts.append("Small position - low impact on portfolio")
            
            return ". ".join(rationale_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating rationale: {e}")
            return "Standard position sizing applied."
    
    def export_sizing_data(self, filepath: str) -> bool:
        """Export sizing results to JSON file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'performance_metrics': {
                    'win_rate': self.win_rate,
                    'avg_win': self.avg_win,
                    'avg_loss': self.avg_loss
                },
                'sizing_results': {
                    key: result.to_dict() for key, result in self.sizing_results.items()
                },
                'statistics': self.get_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Sizing data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting sizing data: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        try:
            if not self.sizing_results:
                return {'message': 'No sizing calculations performed yet'}
            
            # Calculate statistics
            sizes = [result.position_size for result in self.sizing_results.values()]
            risks = [result.risk_amount for result in self.sizing_results.values()]
            confidences = [result.confidence_score for result in self.sizing_results.values()]
            
            stats = {
                'total_calculations': len(self.sizing_results),
                'portfolio_value': self.portfolio_value,
                'performance_metrics': {
                    'win_rate': self.win_rate,
                    'avg_win': self.avg_win,
                    'avg_loss': self.avg_loss
                },
                'position_sizes': {
                    'mean': np.mean(sizes),
                    'median': np.median(sizes),
                    'std': np.std(sizes),
                    'min': np.min(sizes),
                    'max': np.max(sizes)
                },
                'risk_amounts': {
                    'mean': np.mean(risks),
                    'median': np.median(risks),
                    'std': np.std(risks),
                    'min': np.min(risks),
                    'max': np.max(risks)
                },
                'confidence_scores': {
                    'mean': np.mean(confidences),
                    'median': np.median(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                },
                'method_usage': self._get_method_usage_stats(),
                'last_updated': datetime.now().isoformat()
            }
            
            self.statistics = stats
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    def _get_method_usage_stats(self) -> Dict:
        """Get method usage statistics"""
        method_counts = {}
        for result in self.sizing_results.values():
            method = result.method.value
            method_counts[method] = method_counts.get(method, 0) + 1
        
        total = len(self.sizing_results)
        return {
            method: {'count': count, 'percentage': count / total * 100}
            for method, count in method_counts.items()
        } 