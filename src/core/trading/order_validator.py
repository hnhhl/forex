"""
Order Validator Module
Validation logic cho orders và trading rules
"""

import MetaTrader5 as mt5
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time
import logging
from .order_types import OrderRequest, OrderType, OrderTimeInForce

logger = logging.getLogger(__name__)


class OrderValidator:
    """Validator cho orders với comprehensive rules"""
    
    def __init__(self):
        self.symbol_info_cache = {}
        self.trading_hours_cache = {}
        
        # Trading rules
        self.min_volume = 0.01
        self.max_volume = 100.0
        self.max_daily_trades = 50
        self.max_open_orders = 20
        self.min_distance_points = 10  # Minimum distance for stop/limit orders
        
        # Risk limits
        self.max_risk_per_trade = 0.02  # 2%
        self.max_daily_risk = 0.05      # 5%
        self.max_correlation = 0.7      # Maximum correlation between positions
        
        # Daily counters (should be reset daily)
        self.daily_trade_count = 0
        self.daily_risk_used = 0.0
        
    def validate_order(self, order_request: OrderRequest) -> Tuple[bool, str]:
        """Comprehensive order validation"""
        try:
            # 1. Basic validation
            is_valid, message = self._validate_basic_requirements(order_request)
            if not is_valid:
                return False, message
            
            # 2. Symbol validation
            is_valid, message = self._validate_symbol(order_request.symbol)
            if not is_valid:
                return False, message
            
            # 3. Volume validation
            is_valid, message = self._validate_volume(order_request)
            if not is_valid:
                return False, message
            
            # 4. Price validation
            is_valid, message = self._validate_prices(order_request)
            if not is_valid:
                return False, message
            
            # 5. Trading hours validation
            is_valid, message = self._validate_trading_hours(order_request.symbol)
            if not is_valid:
                return False, message
            
            # 6. Risk validation
            is_valid, message = self._validate_risk_limits(order_request)
            if not is_valid:
                return False, message
            
            # 7. Daily limits validation
            is_valid, message = self._validate_daily_limits()
            if not is_valid:
                return False, message
            
            # 8. Market conditions validation
            is_valid, message = self._validate_market_conditions(order_request)
            if not is_valid:
                return False, message
            
            return True, "Order validation passed"
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_basic_requirements(self, order_request: OrderRequest) -> Tuple[bool, str]:
        """Validate basic order requirements"""
        # Symbol check
        if not order_request.symbol or len(order_request.symbol.strip()) == 0:
            return False, "Symbol is required"
        
        # Volume check
        if order_request.volume <= 0:
            return False, "Volume must be positive"
        
        # Order type specific validation
        if order_request.order_type in [OrderType.LIMIT_BUY, OrderType.LIMIT_SELL]:
            if order_request.price is None or order_request.price <= 0:
                return False, "Price is required for limit orders"
        
        if order_request.order_type in [OrderType.STOP_BUY, OrderType.STOP_SELL,
                                      OrderType.STOP_LIMIT_BUY, OrderType.STOP_LIMIT_SELL]:
            if order_request.stop_price is None or order_request.stop_price <= 0:
                return False, "Stop price is required for stop orders"
        
        if order_request.order_type in [OrderType.STOP_LIMIT_BUY, OrderType.STOP_LIMIT_SELL]:
            if order_request.price is None or order_request.price <= 0:
                return False, "Limit price is required for stop-limit orders"
        
        return True, "Basic validation passed"
    
    def _validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Validate symbol and get symbol info"""
        try:
            # Check if symbol info is cached
            if symbol not in self.symbol_info_cache:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    return False, f"Symbol {symbol} not found"
                
                if not symbol_info.visible:
                    # Try to select symbol
                    if not mt5.symbol_select(symbol, True):
                        return False, f"Cannot select symbol {symbol}"
                
                self.symbol_info_cache[symbol] = symbol_info
            
            symbol_info = self.symbol_info_cache[symbol]
            
            # Check if trading is allowed
            if not symbol_info.trade_mode:
                return False, f"Trading not allowed for {symbol}"
            
            return True, "Symbol validation passed"
            
        except Exception as e:
            return False, f"Symbol validation error: {str(e)}"
    
    def _validate_volume(self, order_request: OrderRequest) -> Tuple[bool, str]:
        """Validate volume constraints"""
        try:
            symbol_info = self.symbol_info_cache.get(order_request.symbol)
            if symbol_info is None:
                return False, "Symbol info not available"
            
            # Check minimum volume
            if order_request.volume < symbol_info.volume_min:
                return False, f"Volume {order_request.volume} below minimum {symbol_info.volume_min}"
            
            # Check maximum volume
            if order_request.volume > symbol_info.volume_max:
                return False, f"Volume {order_request.volume} above maximum {symbol_info.volume_max}"
            
            # Check volume step
            volume_step = symbol_info.volume_step
            if volume_step > 0:
                remainder = (order_request.volume - symbol_info.volume_min) % volume_step
                if abs(remainder) > 1e-8:  # Allow for floating point precision
                    return False, f"Volume {order_request.volume} not aligned with step {volume_step}"
            
            # Check our internal limits
            if order_request.volume < self.min_volume:
                return False, f"Volume below internal minimum {self.min_volume}"
            
            if order_request.volume > self.max_volume:
                return False, f"Volume above internal maximum {self.max_volume}"
            
            return True, "Volume validation passed"
            
        except Exception as e:
            return False, f"Volume validation error: {str(e)}"
    
    def _validate_prices(self, order_request: OrderRequest) -> Tuple[bool, str]:
        """Validate price levels and distances"""
        try:
            symbol_info = self.symbol_info_cache.get(order_request.symbol)
            if symbol_info is None:
                return False, "Symbol info not available"
            
            # Get current prices
            tick = mt5.symbol_info_tick(order_request.symbol)
            if tick is None:
                return False, f"Cannot get current price for {order_request.symbol}"
            
            current_bid = tick.bid
            current_ask = tick.ask
            point = symbol_info.point
            
            # Validate stop loss and take profit distances
            if order_request.stop_loss is not None:
                if order_request.order_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY, OrderType.STOP_BUY]:
                    # Buy order - stop loss should be below current price
                    if order_request.stop_loss >= current_bid:
                        return False, "Stop loss for buy order should be below current price"
                    
                    distance = (current_bid - order_request.stop_loss) / point
                    if distance < self.min_distance_points:
                        return False, f"Stop loss too close to current price (min {self.min_distance_points} points)"
                
                elif order_request.order_type in [OrderType.MARKET_SELL, OrderType.LIMIT_SELL, OrderType.STOP_SELL]:
                    # Sell order - stop loss should be above current price
                    if order_request.stop_loss <= current_ask:
                        return False, "Stop loss for sell order should be above current price"
                    
                    distance = (order_request.stop_loss - current_ask) / point
                    if distance < self.min_distance_points:
                        return False, f"Stop loss too close to current price (min {self.min_distance_points} points)"
            
            # Validate take profit distances
            if order_request.take_profit is not None:
                if order_request.order_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY, OrderType.STOP_BUY]:
                    # Buy order - take profit should be above current price
                    if order_request.take_profit <= current_ask:
                        return False, "Take profit for buy order should be above current price"
                    
                    distance = (order_request.take_profit - current_ask) / point
                    if distance < self.min_distance_points:
                        return False, f"Take profit too close to current price (min {self.min_distance_points} points)"
                
                elif order_request.order_type in [OrderType.MARKET_SELL, OrderType.LIMIT_SELL, OrderType.STOP_SELL]:
                    # Sell order - take profit should be below current price
                    if order_request.take_profit >= current_bid:
                        return False, "Take profit for sell order should be below current price"
                    
                    distance = (current_bid - order_request.take_profit) / point
                    if distance < self.min_distance_points:
                        return False, f"Take profit too close to current price (min {self.min_distance_points} points)"
            
            # Validate limit order prices
            if order_request.order_type == OrderType.LIMIT_BUY:
                if order_request.price >= current_ask:
                    return False, "Limit buy price should be below current ask price"
                
                distance = (current_ask - order_request.price) / point
                if distance < self.min_distance_points:
                    return False, f"Limit buy price too close to current price (min {self.min_distance_points} points)"
            
            elif order_request.order_type == OrderType.LIMIT_SELL:
                if order_request.price <= current_bid:
                    return False, "Limit sell price should be above current bid price"
                
                distance = (order_request.price - current_bid) / point
                if distance < self.min_distance_points:
                    return False, f"Limit sell price too close to current price (min {self.min_distance_points} points)"
            
            # Validate stop order prices
            if order_request.order_type == OrderType.STOP_BUY:
                if order_request.stop_price <= current_ask:
                    return False, "Stop buy price should be above current ask price"
                
                distance = (order_request.stop_price - current_ask) / point
                if distance < self.min_distance_points:
                    return False, f"Stop buy price too close to current price (min {self.min_distance_points} points)"
            
            elif order_request.order_type == OrderType.STOP_SELL:
                if order_request.stop_price >= current_bid:
                    return False, "Stop sell price should be below current bid price"
                
                distance = (current_bid - order_request.stop_price) / point
                if distance < self.min_distance_points:
                    return False, f"Stop sell price too close to current price (min {self.min_distance_points} points)"
            
            return True, "Price validation passed"
            
        except Exception as e:
            return False, f"Price validation error: {str(e)}"
    
    def _validate_trading_hours(self, symbol: str) -> Tuple[bool, str]:
        """Validate trading hours for symbol"""
        try:
            # For now, simple check - can be enhanced with detailed trading sessions
            symbol_info = self.symbol_info_cache.get(symbol)
            if symbol_info is None:
                return False, "Symbol info not available"
            
            # Check if market is open (simplified)
            current_time = datetime.now().time()
            
            # Basic trading hours check (can be enhanced)
            if symbol.startswith("XAU"):  # Gold trading
                # Gold trades almost 24/5, but has brief closures
                weekday = datetime.now().weekday()
                if weekday >= 5:  # Weekend
                    return False, "Market closed on weekends"
            
            return True, "Trading hours validation passed"
            
        except Exception as e:
            return False, f"Trading hours validation error: {str(e)}"
    
    def _validate_risk_limits(self, order_request: OrderRequest) -> Tuple[bool, str]:
        """Validate risk management limits"""
        try:
            # Calculate potential risk for this trade
            if order_request.stop_loss is not None:
                symbol_info = self.symbol_info_cache.get(order_request.symbol)
                if symbol_info is None:
                    return False, "Symbol info not available"
                
                tick = mt5.symbol_info_tick(order_request.symbol)
                if tick is None:
                    return False, "Cannot get current price"
                
                # Calculate risk based on stop loss
                if order_request.order_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY, OrderType.STOP_BUY]:
                    entry_price = tick.ask if order_request.price is None else order_request.price
                    risk_points = abs(entry_price - order_request.stop_loss)
                else:
                    entry_price = tick.bid if order_request.price is None else order_request.price
                    risk_points = abs(order_request.stop_loss - entry_price)
                
                # Calculate risk in account currency
                risk_amount = risk_points * order_request.volume * symbol_info.trade_contract_size
                
                # Get account info
                account_info = mt5.account_info()
                if account_info is None:
                    return False, "Cannot get account info"
                
                account_balance = account_info.balance
                risk_percentage = risk_amount / account_balance
                
                # Check per-trade risk limit
                if risk_percentage > self.max_risk_per_trade:
                    return False, f"Risk {risk_percentage:.2%} exceeds maximum per-trade risk {self.max_risk_per_trade:.2%}"
                
                # Check daily risk limit
                if self.daily_risk_used + risk_percentage > self.max_daily_risk:
                    return False, f"Daily risk limit would be exceeded: {(self.daily_risk_used + risk_percentage):.2%} > {self.max_daily_risk:.2%}"
            
            return True, "Risk validation passed"
            
        except Exception as e:
            return False, f"Risk validation error: {str(e)}"
    
    def _validate_daily_limits(self) -> Tuple[bool, str]:
        """Validate daily trading limits"""
        try:
            # Check daily trade count
            if self.daily_trade_count >= self.max_daily_trades:
                return False, f"Daily trade limit reached: {self.daily_trade_count}/{self.max_daily_trades}"
            
            # Check number of open orders
            orders = mt5.orders_get()
            if orders is not None and len(orders) >= self.max_open_orders:
                return False, f"Maximum open orders reached: {len(orders)}/{self.max_open_orders}"
            
            return True, "Daily limits validation passed"
            
        except Exception as e:
            return False, f"Daily limits validation error: {str(e)}"
    
    def _validate_market_conditions(self, order_request: OrderRequest) -> Tuple[bool, str]:
        """Validate current market conditions"""
        try:
            # Get current market data
            tick = mt5.symbol_info_tick(order_request.symbol)
            if tick is None:
                return False, "Cannot get current market data"
            
            # Check spread
            spread = tick.ask - tick.bid
            symbol_info = self.symbol_info_cache.get(order_request.symbol)
            if symbol_info is None:
                return False, "Symbol info not available"
            
            # Check if spread is too wide (basic check)
            max_spread = symbol_info.spread * symbol_info.point * 3  # 3x normal spread
            if spread > max_spread:
                return False, f"Spread too wide: {spread} > {max_spread}"
            
            return True, "Market conditions validation passed"
            
        except Exception as e:
            return False, f"Market conditions validation error: {str(e)}"
    
    def update_daily_counters(self, trade_executed: bool = False, risk_used: float = 0.0):
        """Update daily counters after trade execution"""
        if trade_executed:
            self.daily_trade_count += 1
            self.daily_risk_used += risk_used
    
    def reset_daily_counters(self):
        """Reset daily counters (should be called at start of each trading day)"""
        self.daily_trade_count = 0
        self.daily_risk_used = 0.0
    
    def get_validation_summary(self) -> Dict:
        """Get current validation status summary"""
        return {
            'daily_trades': f"{self.daily_trade_count}/{self.max_daily_trades}",
            'daily_risk_used': f"{self.daily_risk_used:.2%}/{self.max_daily_risk:.2%}",
            'cached_symbols': len(self.symbol_info_cache),
            'max_open_orders': self.max_open_orders,
            'min_distance_points': self.min_distance_points
        } 