"""
Stop Loss Manager Module
Advanced stop loss management với trailing stops và dynamic adjustments
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import MetaTrader5 as mt5

from .position_types import Position, PositionType, PositionStatus

logger = logging.getLogger(__name__)


class StopLossType(Enum):
    """Stop loss types"""
    FIXED = "fixed"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"
    PERCENTAGE_BASED = "percentage_based"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"
    BREAKEVEN = "breakeven"


class TrailingStopMethod(Enum):
    """Trailing stop methods"""
    FIXED_DISTANCE = "fixed_distance"
    PERCENTAGE = "percentage"
    ATR_MULTIPLE = "atr_multiple"
    PARABOLIC_SAR = "parabolic_sar"
    MOVING_AVERAGE = "moving_average"
    SUPPORT_RESISTANCE = "support_resistance"


class StopLossRule:
    """Stop loss rule configuration"""
    
    def __init__(self, rule_id: str, stop_type: StopLossType, **kwargs):
        self.rule_id = rule_id
        self.stop_type = stop_type
        self.is_active = True
        self.created_time = datetime.now()
        
        # Common parameters
        self.distance = kwargs.get('distance', 0.001)  # Distance in price units
        self.percentage = kwargs.get('percentage', 1.0)  # Percentage distance
        self.min_distance = kwargs.get('min_distance', 0.0005)  # Minimum distance
        self.max_distance = kwargs.get('max_distance', 0.01)  # Maximum distance
        
        # Trailing stop parameters
        self.trailing_method = kwargs.get('trailing_method', TrailingStopMethod.FIXED_DISTANCE)
        self.trail_start_profit = kwargs.get('trail_start_profit', 0.001)  # When to start trailing
        self.trail_step = kwargs.get('trail_step', 0.0005)  # Trailing step size
        
        # ATR parameters
        self.atr_period = kwargs.get('atr_period', 14)
        self.atr_multiplier = kwargs.get('atr_multiplier', 2.0)
        
        # Time-based parameters
        self.max_hold_time = kwargs.get('max_hold_time', timedelta(hours=24))
        
        # Breakeven parameters
        self.breakeven_trigger = kwargs.get('breakeven_trigger', 0.001)  # Profit to trigger breakeven
        self.breakeven_buffer = kwargs.get('breakeven_buffer', 0.0002)  # Buffer above/below entry


class StopLossManager:
    """Advanced Stop Loss Management System"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Configuration
        self.update_interval = self.config.get('update_interval', 1)  # seconds
        self.max_adjustment_frequency = self.config.get('max_adjustment_frequency', 5)  # seconds
        
        # Stop loss rules and tracking
        self.stop_rules: Dict[str, StopLossRule] = {}
        self.position_stops: Dict[str, List[str]] = {}  # position_id -> rule_ids
        self.last_adjustments: Dict[str, datetime] = {}  # position_id -> last_adjustment_time
        
        # Market data cache
        self.price_cache: Dict[str, List[Tuple[datetime, float]]] = {}  # symbol -> price history
        self.atr_cache: Dict[str, float] = {}  # symbol -> ATR value
        
        # Event callbacks
        self.stop_callbacks: Dict[str, List[Callable]] = {
            'stop_adjusted': [],
            'stop_triggered': [],
            'trailing_activated': [],
            'breakeven_set': []
        }
        
        # Threading
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Statistics
        self.stats = {
            'total_adjustments': 0,
            'trailing_activations': 0,
            'breakeven_sets': 0,
            'stops_triggered': 0,
            'average_adjustment_size': 0.0
        }
        
        logger.info("StopLossManager initialized")
    
    def start(self):
        """Start stop loss manager"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_stops, daemon=True)
        self.monitoring_thread.start()
        logger.info("StopLossManager started")
    
    def stop(self):
        """Stop stop loss manager"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("StopLossManager stopped")
    
    def add_stop_rule(self, rule: StopLossRule) -> bool:
        """Add stop loss rule"""
        try:
            with self.lock:
                self.stop_rules[rule.rule_id] = rule
                logger.info(f"Stop rule {rule.rule_id} added: {rule.stop_type.value}")
                return True
        except Exception as e:
            logger.error(f"Error adding stop rule: {e}")
            return False
    
    def apply_stop_to_position(self, position_id: str, rule_id: str) -> bool:
        """Apply stop loss rule to position"""
        try:
            with self.lock:
                if rule_id not in self.stop_rules:
                    logger.error(f"Stop rule {rule_id} not found")
                    return False
                
                if position_id not in self.position_stops:
                    self.position_stops[position_id] = []
                
                if rule_id not in self.position_stops[position_id]:
                    self.position_stops[position_id].append(rule_id)
                    logger.info(f"Stop rule {rule_id} applied to position {position_id}")
                
                return True
        except Exception as e:
            logger.error(f"Error applying stop to position: {e}")
            return False
    
    def update_position_stop(self, position: Position) -> bool:
        """Update stop loss for position based on rules"""
        try:
            if position.status != PositionStatus.OPEN:
                return False
            
            # Check adjustment frequency
            if not self._can_adjust_stop(position.position_id):
                return False
            
            with self.lock:
                rules = self.position_stops.get(position.position_id, [])
                if not rules:
                    return False
                
                new_stop_loss = None
                adjustment_made = False
                
                for rule_id in rules:
                    rule = self.stop_rules.get(rule_id)
                    if not rule or not rule.is_active:
                        continue
                    
                    calculated_stop = self._calculate_stop_loss(position, rule)
                    if calculated_stop is None:
                        continue
                    
                    # Choose the most restrictive (closest to current price) stop
                    if new_stop_loss is None:
                        new_stop_loss = calculated_stop
                    else:
                        if position.position_type == PositionType.BUY:
                            new_stop_loss = max(new_stop_loss, calculated_stop)  # Higher stop for buy
                        else:
                            new_stop_loss = min(new_stop_loss, calculated_stop)  # Lower stop for sell
                
                # Apply new stop loss if different
                if new_stop_loss and new_stop_loss != position.stop_loss:
                    if self._should_update_stop(position, new_stop_loss):
                        success = self._execute_stop_update(position, new_stop_loss)
                        if success:
                            self.last_adjustments[position.position_id] = datetime.now()
                            self.stats['total_adjustments'] += 1
                            adjustment_made = True
                            
                            # Trigger callbacks
                            self._trigger_callbacks('stop_adjusted', position, new_stop_loss)
                
                return adjustment_made
                
        except Exception as e:
            logger.error(f"Error updating position stop: {e}")
            return False
    
    def _calculate_stop_loss(self, position: Position, rule: StopLossRule) -> Optional[float]:
        """Calculate stop loss based on rule"""
        try:
            current_price = position.current_price
            
            if rule.stop_type == StopLossType.FIXED:
                return self._calculate_fixed_stop(position, rule)
            elif rule.stop_type == StopLossType.TRAILING:
                return self._calculate_trailing_stop(position, rule)
            elif rule.stop_type == StopLossType.ATR_BASED:
                return self._calculate_atr_stop(position, rule)
            elif rule.stop_type == StopLossType.PERCENTAGE_BASED:
                return self._calculate_percentage_stop(position, rule)
            elif rule.stop_type == StopLossType.VOLATILITY_BASED:
                return self._calculate_volatility_stop(position, rule)
            elif rule.stop_type == StopLossType.TIME_BASED:
                return self._calculate_time_stop(position, rule)
            elif rule.stop_type == StopLossType.BREAKEVEN:
                return self._calculate_breakeven_stop(position, rule)
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return None
    
    def _calculate_fixed_stop(self, position: Position, rule: StopLossRule) -> Optional[float]:
        """Calculate fixed stop loss"""
        if position.position_type == PositionType.BUY:
            return position.open_price - rule.distance
        else:
            return position.open_price + rule.distance
    
    def _calculate_trailing_stop(self, position: Position, rule: StopLossRule) -> Optional[float]:
        """Calculate trailing stop loss"""
        try:
            current_price = position.current_price
            current_profit = self._calculate_current_profit(position)
            
            # Check if profit threshold reached to start trailing
            if current_profit < rule.trail_start_profit:
                return position.stop_loss  # Keep current stop
            
            # Calculate trailing stop based on method
            if rule.trailing_method == TrailingStopMethod.FIXED_DISTANCE:
                if position.position_type == PositionType.BUY:
                    new_stop = current_price - rule.distance
                    # Only move stop up for buy positions
                    return max(new_stop, position.stop_loss or 0) if position.stop_loss else new_stop
                else:
                    new_stop = current_price + rule.distance
                    # Only move stop down for sell positions
                    return min(new_stop, position.stop_loss or float('inf')) if position.stop_loss else new_stop
            
            elif rule.trailing_method == TrailingStopMethod.PERCENTAGE:
                if position.position_type == PositionType.BUY:
                    new_stop = current_price * (1 - rule.percentage / 100)
                    return max(new_stop, position.stop_loss or 0) if position.stop_loss else new_stop
                else:
                    new_stop = current_price * (1 + rule.percentage / 100)
                    return min(new_stop, position.stop_loss or float('inf')) if position.stop_loss else new_stop
            
            elif rule.trailing_method == TrailingStopMethod.ATR_MULTIPLE:
                atr = self._get_atr(position.symbol, rule.atr_period)
                if atr:
                    if position.position_type == PositionType.BUY:
                        new_stop = current_price - (atr * rule.atr_multiplier)
                        return max(new_stop, position.stop_loss or 0) if position.stop_loss else new_stop
                    else:
                        new_stop = current_price + (atr * rule.atr_multiplier)
                        return min(new_stop, position.stop_loss or float('inf')) if position.stop_loss else new_stop
            
            return position.stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return position.stop_loss
    
    def _calculate_atr_stop(self, position: Position, rule: StopLossRule) -> Optional[float]:
        """Calculate ATR-based stop loss"""
        try:
            atr = self._get_atr(position.symbol, rule.atr_period)
            if not atr:
                return None
            
            if position.position_type == PositionType.BUY:
                return position.open_price - (atr * rule.atr_multiplier)
            else:
                return position.open_price + (atr * rule.atr_multiplier)
                
        except Exception as e:
            logger.error(f"Error calculating ATR stop: {e}")
            return None
    
    def _calculate_percentage_stop(self, position: Position, rule: StopLossRule) -> Optional[float]:
        """Calculate percentage-based stop loss"""
        if position.position_type == PositionType.BUY:
            return position.open_price * (1 - rule.percentage / 100)
        else:
            return position.open_price * (1 + rule.percentage / 100)
    
    def _calculate_volatility_stop(self, position: Position, rule: StopLossRule) -> Optional[float]:
        """Calculate volatility-based stop loss"""
        try:
            # Get recent price volatility
            volatility = self._get_volatility(position.symbol)
            if not volatility:
                return None
            
            # Adjust distance based on volatility
            adjusted_distance = rule.distance * (1 + volatility)
            adjusted_distance = max(rule.min_distance, min(adjusted_distance, rule.max_distance))
            
            if position.position_type == PositionType.BUY:
                return position.current_price - adjusted_distance
            else:
                return position.current_price + adjusted_distance
                
        except Exception as e:
            logger.error(f"Error calculating volatility stop: {e}")
            return None
    
    def _calculate_time_stop(self, position: Position, rule: StopLossRule) -> Optional[float]:
        """Calculate time-based stop loss"""
        try:
            hold_time = datetime.now() - position.open_time
            if hold_time >= rule.max_hold_time:
                # Time limit reached, set stop at current price (market close)
                return position.current_price
            
            return position.stop_loss  # Keep current stop
            
        except Exception as e:
            logger.error(f"Error calculating time stop: {e}")
            return None
    
    def _calculate_breakeven_stop(self, position: Position, rule: StopLossRule) -> Optional[float]:
        """Calculate breakeven stop loss"""
        try:
            current_profit = self._calculate_current_profit(position)
            
            if current_profit >= rule.breakeven_trigger:
                # Set stop to breakeven with buffer
                if position.position_type == PositionType.BUY:
                    breakeven_stop = position.open_price + rule.breakeven_buffer
                else:
                    breakeven_stop = position.open_price - rule.breakeven_buffer
                
                # Only move to breakeven if it's better than current stop
                if position.stop_loss:
                    if position.position_type == PositionType.BUY:
                        return max(breakeven_stop, position.stop_loss)
                    else:
                        return min(breakeven_stop, position.stop_loss)
                else:
                    self._trigger_callbacks('breakeven_set', position, breakeven_stop)
                    self.stats['breakeven_sets'] += 1
                    return breakeven_stop
            
            return position.stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating breakeven stop: {e}")
            return None
    
    def _calculate_current_profit(self, position: Position) -> float:
        """Calculate current profit in price units"""
        if position.position_type == PositionType.BUY:
            return position.current_price - position.open_price
        else:
            return position.open_price - position.current_price
    
    def _should_update_stop(self, position: Position, new_stop: float) -> bool:
        """Check if stop should be updated"""
        if not position.stop_loss:
            return True
        
        # For buy positions, only move stop up
        if position.position_type == PositionType.BUY:
            return new_stop > position.stop_loss
        # For sell positions, only move stop down
        else:
            return new_stop < position.stop_loss
    
    def _execute_stop_update(self, position: Position, new_stop: float) -> bool:
        """Execute stop loss update with MT5"""
        try:
            if not position.ticket:
                # Mock update for testing
                position.stop_loss = new_stop
                return True
            
            # MT5 position modification
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_stop,
                "tp": position.take_profit
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                position.stop_loss = new_stop
                logger.info(f"Stop loss updated for position {position.position_id}: {new_stop}")
                return True
            else:
                logger.error(f"Failed to update stop loss: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing stop update: {e}")
            return False
    
    def _can_adjust_stop(self, position_id: str) -> bool:
        """Check if stop can be adjusted (frequency limit)"""
        last_adjustment = self.last_adjustments.get(position_id)
        if not last_adjustment:
            return True
        
        time_since_last = datetime.now() - last_adjustment
        return time_since_last.total_seconds() >= self.max_adjustment_frequency
    
    def _get_atr(self, symbol: str, period: int) -> Optional[float]:
        """Get ATR value for symbol (mock implementation)"""
        # In real implementation, this would calculate ATR from price data
        return self.atr_cache.get(symbol, 0.001)
    
    def _get_volatility(self, symbol: str) -> Optional[float]:
        """Get volatility for symbol (mock implementation)"""
        # In real implementation, this would calculate volatility from price data
        return 0.02  # 2% volatility
    
    def _monitor_stops(self):
        """Monitor and update stops continuously"""
        while self.is_monitoring:
            try:
                # This would be called by position manager for each position
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in stop monitoring: {e}")
                time.sleep(self.update_interval)
    
    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event in self.stop_callbacks:
            self.stop_callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, position: Position, new_stop: Optional[float] = None):
        """Trigger event callbacks"""
        try:
            for callback in self.stop_callbacks.get(event, []):
                if new_stop is not None:
                    callback(position, new_stop)
                else:
                    callback(position)
        except Exception as e:
            logger.error(f"Error triggering callbacks: {e}")
    
    def remove_position_stops(self, position_id: str):
        """Remove all stops for position"""
        with self.lock:
            if position_id in self.position_stops:
                del self.position_stops[position_id]
            if position_id in self.last_adjustments:
                del self.last_adjustments[position_id]
    
    def get_position_stops(self, position_id: str) -> List[StopLossRule]:
        """Get all stop rules for position"""
        rule_ids = self.position_stops.get(position_id, [])
        return [self.stop_rules[rule_id] for rule_id in rule_ids if rule_id in self.stop_rules]
    
    def get_statistics(self) -> Dict:
        """Get stop loss manager statistics"""
        with self.lock:
            return {
                **self.stats,
                'active_rules': len([r for r in self.stop_rules.values() if r.is_active]),
                'total_rules': len(self.stop_rules),
                'positions_with_stops': len(self.position_stops),
                'update_interval': self.update_interval,
                'max_adjustment_frequency': self.max_adjustment_frequency
            }
    
    def export_rules(self, filename: str = None) -> str:
        """Export stop rules to JSON"""
        try:
            import json
            
            if not filename:
                filename = f"stop_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            export_data = {
                'rules': {
                    rule_id: {
                        'stop_type': rule.stop_type.value,
                        'distance': rule.distance,
                        'percentage': rule.percentage,
                        'is_active': rule.is_active,
                        'created_time': rule.created_time.isoformat()
                    }
                    for rule_id, rule in self.stop_rules.items()
                },
                'position_assignments': self.position_stops,
                'statistics': self.get_statistics(),
                'export_time': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Stop rules exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting rules: {e}")
            return "" 