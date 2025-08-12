"""
Position Calculator Module
Advanced P&L calculation và position sizing algorithms
"""

import math
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from enum import Enum
import logging

from .position_types import Position, PositionType

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    RISK_BASED = "risk_based"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    ATR_BASED = "atr_based"


class PnLCalculationType(Enum):
    """P&L calculation types"""
    UNREALIZED = "unrealized"
    REALIZED = "realized"
    TOTAL = "total"


class PositionCalculator:
    """Advanced position calculation system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Default configuration
        self.default_risk_percentage = self.config.get('default_risk_percentage', 2.0)  # 2%
        self.max_risk_percentage = self.config.get('max_risk_percentage', 5.0)  # 5%
        self.min_position_size = self.config.get('min_position_size', 0.01)
        self.max_position_size = self.config.get('max_position_size', 10.0)
        
        # Currency conversion rates (should be updated from MT5)
        self.conversion_rates = self.config.get('conversion_rates', {})
        
        logger.info("PositionCalculator initialized")
    
    def calculate_pnl(self, position: Position, current_price: Optional[float] = None,
                     calculation_type: PnLCalculationType = PnLCalculationType.UNREALIZED) -> float:
        """Calculate P&L for position"""
        try:
            if not current_price:
                current_price = position.current_price
            
            if calculation_type == PnLCalculationType.REALIZED:
                return position.realized_profit
            elif calculation_type == PnLCalculationType.UNREALIZED:
                return self._calculate_unrealized_pnl(position, current_price)
            else:  # TOTAL
                unrealized = self._calculate_unrealized_pnl(position, current_price)
                return position.realized_profit + unrealized
                
        except Exception as e:
            logger.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def _calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized P&L"""
        try:
            price_diff = 0.0
            
            if position.position_type == PositionType.BUY:
                price_diff = current_price - position.open_price
            else:  # SELL
                price_diff = position.open_price - current_price
            
            # Calculate P&L in base currency
            pnl = price_diff * position.remaining_volume
            
            # Apply contract size and point value
            symbol_info = self._get_symbol_info(position.symbol)
            if symbol_info:
                pnl *= symbol_info.get('contract_size', 100000)
                pnl *= symbol_info.get('point_value', 1.0)
            
            return pnl
            
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {e}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float,
                              account_balance: float, risk_percentage: Optional[float] = None,
                              method: PositionSizingMethod = PositionSizingMethod.RISK_BASED) -> float:
        """Calculate optimal position size"""
        try:
            if not risk_percentage:
                risk_percentage = self.default_risk_percentage
            
            # Ensure risk percentage is within limits
            risk_percentage = min(risk_percentage, self.max_risk_percentage)
            
            if method == PositionSizingMethod.FIXED_AMOUNT:
                return self._calculate_fixed_amount_size()
            elif method == PositionSizingMethod.FIXED_PERCENTAGE:
                return self._calculate_fixed_percentage_size(account_balance, risk_percentage)
            elif method == PositionSizingMethod.RISK_BASED:
                return self._calculate_risk_based_size(symbol, entry_price, stop_loss, 
                                                     account_balance, risk_percentage)
            elif method == PositionSizingMethod.KELLY_CRITERION:
                return self._calculate_kelly_size(symbol, account_balance)
            elif method == PositionSizingMethod.VOLATILITY_BASED:
                return self._calculate_volatility_based_size(symbol, account_balance, risk_percentage)
            elif method == PositionSizingMethod.ATR_BASED:
                return self._calculate_atr_based_size(symbol, account_balance, risk_percentage)
            else:
                return self._calculate_risk_based_size(symbol, entry_price, stop_loss, 
                                                     account_balance, risk_percentage)
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.min_position_size
    
    def _calculate_fixed_amount_size(self) -> float:
        """Fixed amount position sizing"""
        return self.config.get('fixed_amount', 0.1)
    
    def _calculate_fixed_percentage_size(self, account_balance: float, risk_percentage: float) -> float:
        """Fixed percentage position sizing"""
        risk_amount = account_balance * (risk_percentage / 100)
        # Assume average risk per lot
        average_risk_per_lot = self.config.get('average_risk_per_lot', 1000)
        size = risk_amount / average_risk_per_lot
        return max(self.min_position_size, min(size, self.max_position_size))
    
    def _calculate_risk_based_size(self, symbol: str, entry_price: float, stop_loss: float,
                                  account_balance: float, risk_percentage: float) -> float:
        """Risk-based position sizing"""
        try:
            # Calculate risk amount
            risk_amount = account_balance * (risk_percentage / 100)
            
            # Calculate price distance
            price_distance = abs(entry_price - stop_loss)
            if price_distance == 0:
                return self.min_position_size
            
            # Get symbol information
            symbol_info = self._get_symbol_info(symbol)
            contract_size = symbol_info.get('contract_size', 100000) if symbol_info else 100000
            point_value = symbol_info.get('point_value', 1.0) if symbol_info else 1.0
            
            # Calculate position size
            risk_per_unit = price_distance * contract_size * point_value
            if risk_per_unit == 0:
                return self.min_position_size
            
            position_size = risk_amount / risk_per_unit
            
            # Apply limits
            position_size = max(self.min_position_size, min(position_size, self.max_position_size))
            
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"Error in risk-based sizing: {e}")
            return self.min_position_size
    
    def _calculate_kelly_size(self, symbol: str, account_balance: float) -> float:
        """Kelly Criterion position sizing"""
        try:
            # Get historical performance data (mock for now)
            win_rate = self.config.get('kelly_win_rate', 0.55)  # 55%
            avg_win = self.config.get('kelly_avg_win', 100)
            avg_loss = self.config.get('kelly_avg_loss', 80)
            
            if avg_loss == 0:
                return self.min_position_size
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply Kelly fraction with safety factor
            safety_factor = self.config.get('kelly_safety_factor', 0.25)  # Use 25% of Kelly
            kelly_fraction *= safety_factor
            
            # Ensure positive and within limits
            kelly_fraction = max(0.01, min(kelly_fraction, 0.1))  # Max 10% of account
            
            # Convert to position size (simplified)
            position_size = kelly_fraction * 2  # Rough conversion
            
            return max(self.min_position_size, min(position_size, self.max_position_size))
            
        except Exception as e:
            logger.error(f"Error in Kelly sizing: {e}")
            return self.min_position_size
    
    def _calculate_volatility_based_size(self, symbol: str, account_balance: float, 
                                       risk_percentage: float) -> float:
        """Volatility-based position sizing"""
        try:
            # Get volatility data (mock for now)
            volatility = self.config.get('symbol_volatility', {}).get(symbol, 0.02)  # 2% daily
            
            # Adjust position size based on volatility
            # Higher volatility = smaller position
            base_size = account_balance * (risk_percentage / 100) / 1000  # Base calculation
            volatility_adjustment = 1 / (1 + volatility * 10)  # Adjustment factor
            
            position_size = base_size * volatility_adjustment
            
            return max(self.min_position_size, min(position_size, self.max_position_size))
            
        except Exception as e:
            logger.error(f"Error in volatility-based sizing: {e}")
            return self.min_position_size
    
    def _calculate_atr_based_size(self, symbol: str, account_balance: float, 
                                 risk_percentage: float) -> float:
        """ATR-based position sizing"""
        try:
            # Get ATR data (mock for now)
            atr = self.config.get('symbol_atr', {}).get(symbol, 0.001)  # Mock ATR
            
            if atr == 0:
                return self.min_position_size
            
            # Calculate risk amount
            risk_amount = account_balance * (risk_percentage / 100)
            
            # Position size based on ATR
            # Use 2 * ATR as stop distance
            stop_distance = 2 * atr
            
            symbol_info = self._get_symbol_info(symbol)
            contract_size = symbol_info.get('contract_size', 100000) if symbol_info else 100000
            point_value = symbol_info.get('point_value', 1.0) if symbol_info else 1.0
            
            risk_per_unit = stop_distance * contract_size * point_value
            if risk_per_unit == 0:
                return self.min_position_size
            
            position_size = risk_amount / risk_per_unit
            
            return max(self.min_position_size, min(position_size, self.max_position_size))
            
        except Exception as e:
            logger.error(f"Error in ATR-based sizing: {e}")
            return self.min_position_size
    
    def calculate_margin_required(self, symbol: str, volume: float, price: float) -> float:
        """Calculate margin required for position"""
        try:
            symbol_info = self._get_symbol_info(symbol)
            if not symbol_info:
                return 0.0
            
            contract_size = symbol_info.get('contract_size', 100000)
            margin_rate = symbol_info.get('margin_rate', 0.01)  # 1% margin
            
            margin_required = volume * contract_size * price * margin_rate
            
            return margin_required
            
        except Exception as e:
            logger.error(f"Error calculating margin: {e}")
            return 0.0
    
    def calculate_pip_value(self, symbol: str, volume: float) -> float:
        """Calculate pip value for position"""
        try:
            symbol_info = self._get_symbol_info(symbol)
            if not symbol_info:
                return 0.0
            
            contract_size = symbol_info.get('contract_size', 100000)
            point_value = symbol_info.get('point_value', 1.0)
            
            pip_value = volume * contract_size * point_value
            
            return pip_value
            
        except Exception as e:
            logger.error(f"Error calculating pip value: {e}")
            return 0.0
    
    def calculate_break_even_price(self, position: Position, spread: float = 0.0) -> float:
        """Calculate break-even price including spread and fees"""
        try:
            # Basic break-even is open price
            break_even = position.open_price
            
            # Adjust for spread
            if position.position_type == PositionType.BUY:
                break_even += spread
            else:
                break_even -= spread
            
            # Add fees/commission if any
            fees_per_unit = self.config.get('fees_per_unit', 0.0)
            if fees_per_unit > 0:
                if position.position_type == PositionType.BUY:
                    break_even += fees_per_unit
                else:
                    break_even -= fees_per_unit
            
            return break_even
            
        except Exception as e:
            logger.error(f"Error calculating break-even price: {e}")
            return position.open_price
    
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, 
                                   take_profit: float) -> float:
        """Calculate risk-reward ratio"""
        try:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk == 0:
                return 0.0
            
            return reward / risk
            
        except Exception as e:
            logger.error(f"Error calculating risk-reward ratio: {e}")
            return 0.0
    
    def calculate_position_metrics(self, position: Position, current_price: Optional[float] = None) -> Dict:
        """Calculate comprehensive position metrics"""
        try:
            if not current_price:
                current_price = position.current_price
            
            metrics = {
                'unrealized_pnl': self._calculate_unrealized_pnl(position, current_price),
                'realized_pnl': position.realized_profit,
                'total_pnl': 0.0,
                'pip_value': self.calculate_pip_value(position.symbol, position.remaining_volume),
                'margin_required': self.calculate_margin_required(position.symbol, 
                                                                position.remaining_volume, 
                                                                position.open_price),
                'break_even_price': self.calculate_break_even_price(position),
                'days_held': (datetime.now() - position.open_time).days,
                'pnl_percentage': 0.0
            }
            
            metrics['total_pnl'] = metrics['unrealized_pnl'] + metrics['realized_pnl']
            
            # Calculate P&L percentage
            if metrics['margin_required'] > 0:
                metrics['pnl_percentage'] = (metrics['total_pnl'] / metrics['margin_required']) * 100
            
            # Risk-reward ratio if stop loss and take profit are set
            if position.stop_loss and position.take_profit:
                metrics['risk_reward_ratio'] = self.calculate_risk_reward_ratio(
                    position.open_price, position.stop_loss, position.take_profit
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating position metrics: {e}")
            return {}
    
    def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information (mock implementation)"""
        # In real implementation, this would fetch from MT5
        symbol_info = {
            'XAUUSD': {
                'contract_size': 100,
                'point_value': 1.0,
                'margin_rate': 0.01,
                'spread': 0.5
            },
            'EURUSD': {
                'contract_size': 100000,
                'point_value': 1.0,
                'margin_rate': 0.033,
                'spread': 0.00001
            },
            'GBPUSD': {
                'contract_size': 100000,
                'point_value': 1.0,
                'margin_rate': 0.033,
                'spread': 0.00002
            }
        }
        
        return symbol_info.get(symbol)
    
    def update_symbol_info(self, symbol: str, info: Dict):
        """Update symbol information"""
        # This would update cached symbol info in real implementation
        pass
    
    def get_statistics(self) -> Dict:
        """Get calculator statistics"""
        return {
            'default_risk_percentage': self.default_risk_percentage,
            'max_risk_percentage': self.max_risk_percentage,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'supported_sizing_methods': [method.value for method in PositionSizingMethod],
            'supported_symbols': list(self._get_symbol_info('XAUUSD').keys()) if self._get_symbol_info('XAUUSD') else []
        } 